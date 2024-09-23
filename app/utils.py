import torch
import pdfplumber
import numpy as np
import pickle
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return ' '.join([page.extract_text() for page in pdf.pages])

def process_resume(file):
    if file.filename.endswith('.pdf'):
        return extract_text_from_pdf(file)
    else:
        return file.read().decode('utf-8')

def process_job_description(file):
    if file.filename.endswith('.pdf'):
        return extract_text_from_pdf(file)
    else:
        return file.read().decode('utf-8')

def rank_candidates(resume_text, job_desc_text, model, tokenizer):
    # Tokenize the input texts
    resume_inputs = tokenizer(resume_text, return_tensors='pt', truncation=True, padding=True)
    job_desc_inputs = tokenizer(job_desc_text, return_tensors='pt', truncation=True, padding=True)
    
    # Move the tensors to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resume_inputs = {key: val.to(device) for key, val in resume_inputs.items()}
    job_desc_inputs = {key: val.to(device) for key, val in job_desc_inputs.items()}
    
    # Move the model to the device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        # Get the embeddings for both resume and job description
        resume_outputs = model(**resume_inputs)
        job_desc_outputs = model(**job_desc_inputs)
    
    # Extract the embeddings from the last hidden state
    resume_embedding = resume_outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    job_desc_embedding = job_desc_outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    
    # Compute Euclidean distance or Cosine_Similarity between the two embeddings
    similarity_score = cosine_similarity(resume_embedding, job_desc_embedding).flatten()
    
    # Return the similarity score as the fit score (higher is better for Cosine Similarity)
    return similarity_score[0]

def suggest_categories(resume_text, model, tokenizer):
    # Tokenize the input resume text
    inputs = tokenizer(resume_text, return_tensors='pt', truncation=True, padding=True)
    
    # Check for GPU availability and move inputs to the correct device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Move the model to the device and set to evaluation mode
    model.to(device)  # Ensure model is on the correct device
    model.eval()
    
    with torch.no_grad():
        # Get the embedding for the resume text
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    
    # Load the candidate embeddings and categories
    with open('candidate_embeddings.pkl', 'rb') as f:
        candidate_embeddings = pickle.load(f)
    
    with open('candidate_categories.pkl', 'rb') as f:
        candidate_categories = pickle.load(f)
    
    # Compute the distances to all candidate embeddings
    distances = euclidean_distances(embedding, candidate_embeddings).flatten()
    
    # Find the indices of the closest candidates
    closest_indices = distances.argsort()[:5]  # Get top 5 closest candidates
    
    # Get associated categories for the closest candidates
    suggested_categories = [candidate_categories[i] for i in closest_indices]

    return suggested_categories
