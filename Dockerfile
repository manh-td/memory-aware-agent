# Use an official Python base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install system tools
RUN apt-get update && \
    apt-get install -y git tmux && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY requirements.txt .

# Create a virtual environment and install dependencies
RUN pip install -r requirements.txt