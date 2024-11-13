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
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_text = page.get_text("text")  # Extract text from the page
            
            if page_text:
                text += page_text
            else:
                # If no text is extracted, log and try OCR
                st.warning(f"Page {page_num + 1} does not contain extractable text. Attempting OCR...")
                images = convert_from_path(file, first_page=page_num+1, last_page=page_num+1)
                page_image = images[0]
                ocr_text = pytesseract.image_to_string(page_image)  # Perform OCR on the image
                text += ocr_text

        # If no text was extracted, log an error
        if not text:
            st.error("No text could be extracted from the PDF. It might be an image-based or scanned PDF.")
        
        return text
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return None

# Function to split text into smaller chunks (for better retrieval)
def split_text(text, chunk_size=500):
    sentences = text.split('\n')
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk + sentence) <= chunk_size:
            chunk += sentence + "\n"
        else:
            chunks.append(chunk.strip())
            chunk = sentence + "\n"
    if chunk:
        chunks.append(chunk.strip())  # Add the final chunk
    return chunks

# Function to create FAISS index from document embeddings
def create_faiss_index(documents):
    try:
        embeddings = embedding_model.encode(documents)
        # Initialize FAISS index (L2 distance)
        index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance metric
        index.add(np.array(embeddings, dtype=np.float32))
        return index, embeddings
    except Exception as e:
        st.error(f"Error creating FAISS index: {e}")
        return None, None

# Function to retrieve the most relevant document based on a query
def retrieve_document(query, index, documents):
    try:
        query_embedding = embedding_model.encode([query])
        distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k=1)
        return documents[indices[0][0]]
    except Exception as e:
        st.error(f"Error retrieving document: {e}")
        return None

# Function to interact with GPT-3 for generation
def generate_response(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].text.strip()
    except openai.error.AuthenticationError:
        st.error("OpenAI API key is missing or incorrect. Please check your API key.")
        return None
    except openai.error.OpenAIError as e:
        st.error(f"Error generating response from OpenAI: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error while generating response: {e}")
        return None

# Streamlit app layout
st.title('Retrieval-Augmented Generation (RAG) with PDF Upload')

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.write("File uploaded successfully!")

    # Load and process the PDF
    pdf_text = load_pdf(uploaded_file)

    if pdf_text:
        st.subheader('Extracted Text from PDF:')
        st.text_area("PDF Content", pdf_text, height=200)

        # Split the extracted text into smaller chunks for embedding
        documents = split_text(pdf_text)

        # Create FAISS index for document retrieval
        index, embeddings = create_faiss_index(documents)

        # Check if FAISS index was created successfully
        if index is None:
            st.error("Failed to create the FAISS index. The application cannot function without it.")
            st.stop()

        # Query input from user
        query = st.text_input('Ask a question about the PDF:', '')

        if query:
            # Retrieve the most relevant document based on the query
            relevant_document = retrieve_document(query, index, documents)

            # Check if a relevant document was retrieved
            if relevant_document is None:
                st.error("No relevant document could be retrieved for the query.")
            else:
                # Create the prompt by combining the query and the relevant document
                prompt = f"Context: {relevant_document}\n\nQuestion: {query}\nAnswer:"

                # Generate the answer using GPT-3
                response = generate_response(prompt)

                # Display the answer if the response is valid
                if response:
                    st.subheader('Answer:')
                    st.write(response)
                else:
                    st.error("Failed to generate an answer.")
    else:
        st.error("Failed to extract text from the uploaded PDF.")
else:
    st.info("Please upload a PDF file to get started.")
