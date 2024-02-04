# Use an official NVIDIA runtime as a parent image
FROM nvidia/cuda:12.2.2-runtime-ubuntu20.04

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip python3-dev && \
    pip3 install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run train.py when the container launches
CMD ["python3", "train.py"]
