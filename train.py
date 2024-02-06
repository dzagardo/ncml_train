################################################################################
# Define Imports
################################################################################

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    AdamW,
)
from datasets import load_dataset, DatasetDict, load_from_disk
from torch.utils.data import Dataset
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from torch.utils.data import DataLoader
from typing import Dict
from torch.optim import SGD
from torch.nn.utils import clip_grad_norm_
from scipy import optimize
from scipy import stats
import numpy as np

print(torch.__version__)
print("GPU available:", torch.cuda.is_available())

################################################################################
# DP Helper Functions
################################################################################

def compute_mu_uniform(epoch, noise_multiplier, n, batch_size):
    """Compute mu from uniform subsampling with safeguards for low noise multipliers."""
    t = epoch * n / batch_size
    c = batch_size * np.sqrt(t) / n

    # Safeguard against overflow in exp calculation
    exp_arg = noise_multiplier**(-2)
    if exp_arg > 700:  # np.exp(700) is close to the upper limit for float64
        exp_term = np.float64(np.inf)
    else:
        exp_term = np.exp(exp_arg)

    sqrt_term = np.sqrt(exp_term * stats.norm.cdf(1.5 / noise_multiplier) +
                        3 * stats.norm.cdf(-0.5 / noise_multiplier) - 2)

    return np.sqrt(2) * c * sqrt_term

def delta_eps_mu(eps, mu):
    """Compute dual between mu-GDP and (epsilon, delta)-DP with numerical stability checks."""
    # Check if mu is extremely large to prevent numerical errors
    if mu > 1e10:
        return 0.0
    return stats.norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * stats.norm.cdf(-eps / mu - mu / 2)

def compute_eps_uniform(epoch, noise_multiplier, n, batch_size, delta):
    """Compute epsilon given delta from inverse dual of uniform subsampling with convergence checks."""
    mu = compute_mu_uniform(epoch, noise_multiplier, n, batch_size)
    def f(x):
        return delta_eps_mu(x, mu) - delta

    # Safely find the root with a check for convergence
    try:
        epsilon = optimize.root_scalar(f, bracket=[0, 500], method='brentq').root
    except ValueError:
        # Handle non-convergence
        print("Root finding did not converge. Adjust the parameters.")
        epsilon = None

    return epsilon

################################################################################
# Stream Dataset Class
################################################################################

class StreamDataset(Dataset):
    def __init__(self, tokenized_dataset, tokenizer):
        self.tokenized_dataset = tokenized_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        example = self.tokenized_dataset[idx]
        input_ids = torch.tensor(example['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(example['attention_mask'], dtype=torch.long)

        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

################################################################################
# Overridden Trainer Class
################################################################################

class PrivacyAwareTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize variables for privacy accounting
        self.delta = 1e-7
        self.global_privacy_budget = float('inf')
        self.total_noise_multiplier = 0.0
        self.total_clipping_norm = 0.0
        self.num_training_steps = 0

        # Extract total dataset size and training sample size from training arguments
        self.total_dataset_size = self.args.total_dataset_size
        self.training_sample_size = self.args.training_sample_size
        self.optimizer = self.create_optimizer()

    ### NEED TO FIX. AM I USING ADAMW OR SGD?????... ###
    ### NEED TO FIX. AM I USING ADAMW OR SGD?????... ###
    ### NEED TO FIX. AM I USING ADAMW OR SGD?????... ###
    def create_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        return optimizer

    def log(self, logs: Dict[str, float]) -> None:
        # Call the original log method to maintain its functionality
        super().log(logs)

        # Log the current global privacy budget (epsilon)
        logs["privacy_epsilon"] = self.global_privacy_budget
        print(f"Current Global Privacy Budget (Epsilon): {self.global_privacy_budget}, Delta: {self.delta}")

    def compute_privacy_budget(self):
        if self.num_training_steps == 0:
            return self.global_privacy_budget, self.delta

        # Calculate average noise multiplier
        avg_noise_multiplier = self.total_noise_multiplier / self.num_training_steps

        # Parameters for GDP calculation
        epoch = self.args.num_train_epochs
        n = self.total_dataset_size
        batch_size = self.args.per_device_train_batch_size

        # Compute epsilon using GDP accountant
        new_epsilon = compute_eps_uniform(epoch, avg_noise_multiplier, n, batch_size, self.delta)

        # Update the global privacy budget by summing the new epsilon
        if self.global_privacy_budget == float('inf'):
            self.global_privacy_budget = new_epsilon
        else:
            self.global_privacy_budget += new_epsilon

        return self.global_privacy_budget, self.delta

    # def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
    #     self.train_dataloader = self.get_train_dataloader()
    #     self.model.train()

    #     total_loss = 0.0
    #     total_steps = 0

    #     for epoch in range(self.args.num_train_epochs):
    #         epoch_loss = 0.0
    #         epoch_steps = 0

    #         for step, batch in enumerate(self.train_dataloader):
    #             batch = {k: v.to(self.device) for k, v in batch.items()}
    #             loss = self.training_step(self.model, batch)

    #             epoch_loss += loss
    #             epoch_steps += 1
    #             total_loss += loss
    #             total_steps += 1

    #             if step % self.args.logging_steps == 0:
    #                 print(f"Epoch {epoch}, Step {step}, Loss: {loss}")

    #         avg_epoch_loss = epoch_loss / epoch_steps
    #         print(f"Average loss for Epoch {epoch}: {avg_epoch_loss}")

    #     avg_total_loss = total_loss / total_steps
    #     print(f"Average training loss: {avg_total_loss}")

    # def training_step(self, model, inputs):
    #     model.zero_grad()  # Ensure gradients are zeroed before forward pass

    #     outputs = model(**inputs)
    #     loss = outputs.loss
    #     loss.backward()

    #     self.optimizer.step()

    #     return loss.item()

########### UNCOMMENT TO TRAIN NO DP ###########
########### UNCOMMENT TO TRAIN NO DP ###########
########### UNCOMMENT TO TRAIN NO DP ###########
###### ^^ COMMENT OUT OTHER TRAIN STEP ^^ ######
###### ^^ COMMENT OUT OTHER TRAIN STEP ^^ ######
###### ^^ COMMENT OUT OTHER TRAIN STEP ^^ ######
###### vv COMMENT OUT OTHER TRAIN STEP vv ######
###### vv COMMENT OUT OTHER TRAIN STEP vv ######
###### vv COMMENT OUT OTHER TRAIN STEP vv ######
########## UNCOMMENT TO TRAIN WITH DP ##########
########## UNCOMMENT TO TRAIN WITH DP ##########
########## UNCOMMENT TO TRAIN WITH DP ##########

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        self.train_dataloader = self.get_train_dataloader()
        self.model.train()

        total_loss = 0.0
        total_steps = 0

        for epoch in range(self.args.num_train_epochs):
            epoch_loss = 0.0
            epoch_steps = 0

            for step, batch in enumerate(self.train_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = self.training_step(self.model, batch)

                epoch_loss += loss
                epoch_steps += 1
                total_loss += loss
                total_steps += 1

            avg_epoch_loss = epoch_loss / epoch_steps
            print(f"Average loss for Epoch {epoch}: {avg_epoch_loss}")

        avg_total_loss = total_loss / total_steps
        print(f"Average training loss: {avg_total_loss}")

        epsilon, delta = self.compute_privacy_budget()
        print(f"Final Global Privacy Budget (Epsilon): {epsilon}, with Delta: {delta}")

    def training_step(self, model, inputs):
        model.zero_grad()  # Ensure gradients are zeroed before forward pass

        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()

        # Check if any gradients are None
        grads_none = any(p.grad is None for p in model.parameters() if p.requires_grad)
        if grads_none:
            print("Warning: One or more gradients are None")

        # Compute maximum and average gradient magnitudes
        grad_magnitudes = [p.grad.data.abs() for p in model.parameters() if p.grad is not None]
        max_grad = max(g.max() for g in grad_magnitudes)
        avg_grad = sum(g.mean() for g in grad_magnitudes) / len(grad_magnitudes)

        # Apply gradient clipping
        max_grad_norm = 0.3
        clip_grad_norm_(model.parameters(), max_grad_norm)

        # Apply noise addition
        noise_multiplier = 2.0
        for p in model.parameters():
            if p.grad is not None:
                noise = torch.normal(0, max_grad_norm, p.grad.shape, device=p.grad.device)
                p.grad += noise_multiplier * noise / max_grad_norm

        self.total_noise_multiplier += noise_multiplier
        self.total_clipping_norm += max_grad_norm
        self.num_training_steps += 1

        self.optimizer.step()

        return loss.item()

class ExtendedTrainingArguments(TrainingArguments):
    def __init__(self, total_dataset_size: int = 0, training_sample_size: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.total_dataset_size = total_dataset_size
        self.training_sample_size = training_sample_size

################################################################################
# Model and Dataset parameters
################################################################################

# The model that you want to train from the Hugging Face hub
model_name = "NousResearch/Llama-2-7b-chat-hf"

# The instruction dataset to use
dataset_name = "mlabonne/guanaco-llama2-1k"

# Fine-tuned model name
new_model = "llama-2-7b-miniguanaco"

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 10

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Micro batch size
micro_batch_size = 1

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 4

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}

################################################################################
# Model Pipeline
################################################################################

# Load dataset (you can process it here)
dataset = load_dataset(dataset_name, split="train")

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load Tiny Llama Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pdding_side = "right"

# Load LoRA Config
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Instantiate the SGD optimizer
optimizer = SGD(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)

# Total number of iterations (each with 100 samples)
total_iterations = 10
iteration_size = 100

# Total size of the dataset
total_dataset_size = total_iterations * iteration_size
training_sample_size = iteration_size

# Set training parameters
training_arguments = ExtendedTrainingArguments(
    output_dir='/content/drive/My Drive/Software/LLM/Models',
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
    total_dataset_size=total_dataset_size,
    training_sample_size=training_sample_size,
)

################################################################################
# Training Loop
################################################################################

# Initialize the global privacy budget
current_global_privacy_budget = float('inf')

for i in range(total_iterations):
    print(f"Iteration: {i} of {total_iterations}")

    # Calculate start and end index for each chunk
    start_idx = i * iteration_size
    end_idx = start_idx + iteration_size

    # Select the next 100 rows from the dataset
    small_dataset = dataset.select(range(start_idx, end_idx))

    # Initialize the StreamDataset with the selected subset
    stream_dataset = StreamDataset(tokenized_dataset=small_dataset, tokenizer=tokenizer)

    # Initialize the Trainer with the optimizer
    trainer = PrivacyAwareTrainer(
        model=model,
        args=training_arguments,
        train_dataset=stream_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=packing,
        optimizers=(optimizer, None)  # LR scheduler will be handled automatically
    )
    # Update the trainer with the current global privacy state
    trainer.global_privacy_budget = current_global_privacy_budget

    # Train the model on this subset
    trainer.train()

    # Update the global privacy budget for the next iteration
    current_global_privacy_budget = trainer.global_privacy_budget

################################################################################
# Save Model
################################################################################

# Fine-tuned model name
new_model = "ncmltest"

# Save the model using the updated save_model method in PrivacyAwareTrainer
trainer.model.save_pretrained(new_model)

################################################################################
# Sanity Check Model
################################################################################

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# Run text generation pipeline with our next model
prompt = "What is a large language model?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

################################################################################
# Reload, Merge Model with LoRA Weights
################################################################################

# Empty VRAM
del model
del pipe
import gc
gc.collect()
gc.collect()

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
# Now, push the model and tokenizer to the hub
model.push_to_hub("tiny-llama-orca-amp-gclip-dp-pa-sgd-dz-v1.2.1000")
tokenizer.push_to_hub("tiny-llama-orca-amp-gclip-dp-pa-sgd-dz-v1.2.1000")