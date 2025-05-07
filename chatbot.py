import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
import ollama
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings

# Load environment variables
load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    api_version="2023-07-01-preview",
    azure_endpoint=os.getenv("ENDPOINT_URL"),
)

chat_deployment = os.getenv("DEPLOYMENT_NAME")

class OllamaEmbeddings(Embeddings):
    def __init__(self, model="nomic-embed-text"):
        self.model = model

    def embed_documents(self, texts):
        cleaned_texts = [str(text) if isinstance(text, str) else json.dumps(text) for text in texts]
        return ollama.embed(model=self.model, input=cleaned_texts)["embeddings"]

    def embed_query(self, query):
        return ollama.embed(model=self.model, input=[query])["embeddings"][0]

# Load JSON course files from folder
def load_json_documents(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract only relevant content from the JSON (for example, the 'course_title' or description)
                content = data.get('course_title', '')  # Modify this to get any relevant field
                docs.append(Document(page_content=content, metadata={"source": file}))
    return docs

# Chunk the documents
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# Embed & store in vector DB using Ollama
def create_vectorstore(docs):
    # Initialize Ollama embeddings class
    embeddings = OllamaEmbeddings()
    
    # Create the Chroma vectorstore with Ollama embeddings
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_store"
    )

# Retrieve relevant documents
def retrieve_relevant_docs(vectorstore, query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever.get_relevant_documents(query)

# Ask question with context
def ask_with_context(query, vectorstore):
    relevant_docs = retrieve_relevant_docs(vectorstore, query)
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

    chat_prompt = [
        {   
            "role": "system",
            "content": "You are an AI assistant that helps people find information about university courses using internal data."
        },
        {
            "role": "user",
            "content": f"Using the following data:\n\n{context_text}\n\nAnswer the question: {query}"
        }
    ]

    response = client.chat.completions.create(
        model=chat_deployment,
        messages=chat_prompt,
        max_tokens=800,
        temperature=0.7
    )

    print(response.choices[0].message.content)

# === Run RAG Pipeline ===
if __name__ == "__main__":
    data_folder = r"C:\Users\anson\OneDrive\Desktop\sporthunt\proj\data"
    print("[+] Loading JSON course documents...")
    docs = load_json_documents(data_folder)

    print("[+] Splitting documents into chunks...")
    chunks = chunk_documents(docs)

    print("[+] Creating or loading vector database...")
    vectorstore = create_vectorstore(chunks)

    print("[+] Ready for Q&A. Example question:")
    while True:
        query = input("\nAsk a question about the course data (or type 'exit' to quit): ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        ask_with_context(query, vectorstore)
