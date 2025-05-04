import ollama
import pypdf
from pathlib import Path
import pickle
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

MESSAGES = [
    {"role": "system", "content": """You are an chatbot for Chist University.
        sometimes the context may be incomplete or not fully relevant.
        Please answer the question to the best of your ability based on the context provided. """
    },
]


load_dotenv()
# 1. Extract text from PDF
def extract_text_from_pdf(pdf_path):
    pdf = pypdf.PdfReader(pdf_path)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# 2. Split text into chunks
def split_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# 3. Create embeddings with the Ollama model
def embed_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        response = ollama.embed(model='nomic-embed-text', input=chunk)
        embeddings.append(np.array(response['embeddings']))
    return embeddings

# 4. Save embeddings to disk
def save_data(chunks, embeddings, file_path="pdf_embeddings.pkl"):
    data = {
        'chunks': chunks,
        'embeddings': embeddings
    }
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved embeddings to {file_path}")

# 5. Load previously saved embeddings
def load_data(file_path="pdf_embeddings.pkl"):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['chunks'], data['embeddings']

# 6. Find most similar chunks to a query
def find_similar_chunks(query, chunks, embeddings, top_k=3):
    # Get embedding for the query
    query_embedding = np.array(ollama.embed(model='nomic-embed-text', input=query)['embeddings'])
    
    # Calculate similarity using dot product
    similarities = []
    for emb in embeddings:
        # Make sure both are flattened 1D arrays
        similarity = np.dot(query_embedding.flatten(), emb.flatten())
        similarities.append(similarity)
    # Get top-k similar chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# 7. Generate response using Ollama
def generate_response(query, context_chunks):
    # Combine context chunks
    context = "\n\n".join(context_chunks)
    
    # Create prompt with context
    prompt = f"""
Context information:
{context}

*sometimes the context may be incomplete or not fully relevant.*

YOU WILL JUST ANSWER THE QUESTIONS ABOUT CHRIST UNIVERSITY.

QUESTION:
{query}
    """
    
# Generate response
    # response = ollama.generate(model='gemma3:4b', prompt=prompt)
    endpoint = "https://cto-m6zbzctf-swedencentral.openai.azure.com/"
    model_name = "gpt-4.1"
    deployment = "gpt-4.1"

    subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = "2024-12-01-preview"
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

    response = client.chat.completions.create(
        stream=True,
        messages=MESSAGES + [{"role": "user", "content": prompt}],
        max_completion_tokens=800,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        model=deployment,
    )

    full_response = ""
    for update in response:
        if update.choices:
            content = update.choices[0].delta.content or ""
            print(content, end="", flush=True)
            full_response += content

    client.close()
    
    return full_response

# Main process
def process_pdf(pdf_path, embedding_path=None):
    # Extract and process text
    print(f"Processing {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)
    with open(pdf_path, "r", encoding="utf-8") as file:
        text = file.read()
    chunks = split_text(text)
    print(f"Split into {len(chunks)} chunks")
    
    # Create and save embeddings
    print("Creating embeddings...")
    embeddings = embed_chunks(chunks)
    save_path = embedding_path or f"{Path(pdf_path).stem}_embeddings.pkl"
    save_data(chunks, embeddings, save_path)
    return chunks, embeddings

# Interactive query function
def query_pdf(chunks, embeddings):
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        # Find relevant context
        relevant_chunks = find_similar_chunks(query, chunks, embeddings)
        
        # Generate answer
        answer = generate_response(query, relevant_chunks)
        # print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    # Option 1: Process a new PDF
    pdf_path = "prompt_engg.pdf" 
    # chunks, embeddings = process_pdf(pdf_path)    
    
    # Option 2: Load existing embeddings
    chunks, embeddings = load_data("embeddings/rag_embedding.pkl")
    
    # Start querying
    query_pdf(chunks, embeddings)



