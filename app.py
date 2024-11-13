import streamlit as st
import openai
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz  # PyMuPDF for extracting text from PDFs
import pytesseract
from pdf2image import convert_from_path

# Option 1: Set API key using environment variable (recommended for production)
openai.api_key = os.getenv("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")

# Option 2: Or directly set the API key (for testing purposes only, not recommended for production)
# openai.api_key = "your-openai-api-key-here"  # Direct API key assignment (not recommended for production)

# Initialize SentenceTransformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to load and preprocess the PDF
def load_pdf(file):
    try:
        doc = fitz.open(file)  # Open the uploaded PDF file
        text = ""
        
        # Extract text from each page
     
