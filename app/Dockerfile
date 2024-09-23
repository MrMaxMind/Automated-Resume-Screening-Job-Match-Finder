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

# Download the BERT model from Hugging Face
RUN . /app/venv/bin/activate && python -c "from transformers import BertTokenizer, BertModel; BertModel.from_pretrained('bert-base-uncased'); BertTokenizer.from_pretrained('bert-base-uncased')"

# Ensure necessary model files are copied (this step assumes the model/tokenizer files are pre-downloaded)
COPY bert_model.pth /app/bert_model.pth
COPY bert_tokenizer.pkl /app/bert_tokenizer.pkl
COPY candidate_rankings.pkl /app/candidate_rankings.pkl

# Expose port
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app.py"]
