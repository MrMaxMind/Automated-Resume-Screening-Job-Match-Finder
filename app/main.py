import os
from flask import Flask, request, render_template
import torch
import pickle
from transformers import BertTokenizer, BertModel
from .utils import process_resume, process_job_description, rank_candidates, suggest_categories

app = Flask(__name__)

# Load pre-trained model architecture and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
model.load_state_dict(torch.load('app/bert_model.pth', map_location=torch.device('cpu')))

# Load the tokenizer dynamically from Hugging Face
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load rankings if needed
with open('app/candidate_rankings.pkl', 'rb') as f:
    rankings = pickle.load(f)

# Create the uploads directory if it doesn't exist
upload_folder = 'uploads'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    case_type = request.form.get('case_type')
    resume_file = request.files['resume']
    job_description_file = request.files.get('job_description')

    # Process Resume
    resume_text = process_resume(resume_file)

    if case_type == 'both':
        # Case 1: Resume + Job Description
        if not job_description_file:
            return render_template('index.html', error="Job description file is required for this case.")

        job_desc_text = process_job_description(job_description_file)
        fit_score = rank_candidates(resume_text, job_desc_text, model, tokenizer)
        
        # Render fit score on the main page
        return render_template('result.html', fit_score=fit_score)

    elif case_type == 'resume_only':
        # Case 2: Resume only
        suggested_categories = suggest_categories(resume_text, model, tokenizer)
        
        # Render suggested roles on the main page
        return render_template('result.html', suggested_categories=suggested_categories)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
