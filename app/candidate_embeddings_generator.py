import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import pickle

# Load your dataset
resume_df = pd.read_csv('/home/hp/Downloads/dataset/Resume.csv')  # Update with your actual path

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()  # Set to evaluation mode

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Generate embeddings and store categories
candidate_embeddings = []
categories = []  # List to store categories

for index, row in resume_df.iterrows():
    resume = row['Resume_str']
    category = row['Category']  # Get the associated category
    
    inputs = tokenizer(resume, return_tensors='pt', truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pooling of the last hidden state
    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    candidate_embeddings.append(embedding)
    categories.append(category)  # Store the category

# Convert to NumPy array
candidate_embeddings = np.vstack(candidate_embeddings)

# Save the embeddings and categories
with open('candidate_embeddings.pkl', 'wb') as f:
    pickle.dump(candidate_embeddings, f)

# Save the categories as well
with open('candidate_categories.pkl', 'wb') as f:
    pickle.dump(categories, f)

print("Candidate embeddings and categories saved successfully.")
