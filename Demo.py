import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, RobertaTokenizer
import gradio as gr
import re
from model import HybridBERTRoBERTaModel


# Load the tokenizers
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Load your trained model
model = HybridBERTRoBERTaModel()
model.load_state_dict(torch.load('best_hybrid_model.pt', map_location=torch.device('cpu')))
model.eval()


# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags symbol while keeping the texts 
        text = re.sub(r'#', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return "empty"

# Prediction function
def predict_disaster(text, keyword="", location=""):
    # Preprocess the text
    clean_text = preprocess_text(text)
    
    # Combine features
    combined_text = f"{clean_text} keyword: {keyword} location: {location}"
    
    # Tokenize for BERT
    bert_encoding = bert_tokenizer.encode_plus(
        combined_text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Tokenize for RoBERTa
    roberta_encoding = roberta_tokenizer.encode_plus(
        combined_text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Get model inputs
    bert_input_ids = bert_encoding['input_ids']
    bert_attention_mask = bert_encoding['attention_mask']
    bert_token_type_ids = bert_encoding['token_type_ids']
    roberta_input_ids = roberta_encoding['input_ids']
    roberta_attention_mask = roberta_encoding['attention_mask']
    
    # Make prediction
    with torch.no_grad():
        outputs = model(
            bert_input_ids=bert_input_ids,
            bert_attention_mask=bert_attention_mask,
            bert_token_type_ids=bert_token_type_ids,
            roberta_input_ids=roberta_input_ids,
            roberta_attention_mask=roberta_attention_mask
        )
        
        probs = torch.nn.functional.softmax(outputs, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()
    
    result = "Disaster" if prediction == 1 else "Not a Disaster"
    return result, confidence

# Create the Gradio interface
def gradio_interface(text, keyword, location):
    result, confidence = predict_disaster(text, keyword, location)
    return f"{result} (confidence: {confidence:.2f})"

# Launch the interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(lines=3, placeholder="Enter tweet text here...", label="Tweet Text"),
        gr.Textbox(placeholder="Enter keyword (optional)", label="Keyword"),
        gr.Textbox(placeholder="Enter location (optional)", label="Location")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Disaster Tweet Classifier",
    description="This model classifies whether a tweet is about a real disaster or not."
)

# Launch the interface
iface.launch()