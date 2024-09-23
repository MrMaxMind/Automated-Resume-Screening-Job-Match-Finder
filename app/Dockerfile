# Use Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (if required)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files into the container
COPY . /app

# Create a virtual environment in the container
RUN python -m venv venv

# Activate the virtual environment and install dependencies
RUN . /app/venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

# Ensure the virtual environment is used when running commands
ENV PATH="/app/venv/bin:$PATH"

# Ensure necessary model files are copied
COPY app/bert_model.pth /app/bert_model.pth
COPY app/bert_tokenizer.pkl /app/bert_tokenizer.pkl
COPY app/candidate_rankings.pkl /app/candidate_rankings.pkl
COPY candidate_categories.pkl /app/candidate_categories.pkl
COPY candidate_embeddings.pkl /app/candidate_embeddings.pkl

# Expose port
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app/main.py"]
