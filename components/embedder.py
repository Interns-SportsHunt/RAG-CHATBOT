from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
import pypdf
import json
from pathlib import Path
import pickle
import numpy as np

# Optional: ChromaDB dependencies
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None

# Optional: PGVector dependencies
try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    psycopg2 = None

def extract_text_from_pdf(pdf_path):
    pdf = pypdf.PdfReader(pdf_path)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_text_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # print(len(data))  # Debugging line to check JSON size
    # Simply stringify the entire JSON, preserving keys and structure
    
    for _, data_ in enumerate(data):
        print(data_["course_title"])
        
    print("==="*20)
    print(_)
    return json.dumps(data, ensure_ascii=False, indent=2)

def split_text(text, chunk_size=1000, chunk_overlap=100):
    # Use RecursiveCharacterTextSplitter from langchain_text_splitters for better chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def embed_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        response = ollama.embed(model='nomic-embed-text', input=chunk)
        embeddings.append(np.array(response['embeddings']))
    return embeddings

def save_data(all_chunks, all_embeddings, all_files, file_path="multi_embeddings.pkl"):
    data = {
        'chunks': all_chunks,
        'embeddings': all_embeddings,
        'files': all_files
    }
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved embeddings to {file_path}")

def save_to_chromadb(chunks, embeddings, files, db_path=None, collection_name="default"):
    """Save chunks and embeddings to a local ChromaDB collection."""
    if chromadb is None:
        raise ImportError('chromadb is not installed. Install it to use ChromaDB.')
    client = chromadb.PersistentClient(path=db_path) if db_path else chromadb.Client()
    collection = client.get_or_create_collection(collection_name)
    # Chroma expects embeddings as lists, not numpy arrays
    embeddings_list = [e.tolist() if hasattr(e, 'tolist') else e for e in embeddings]
    # Use file+index as unique ids
    ids = [f"{Path(f).stem}_{i}" for i, f in enumerate(files)]
    collection.add(
        embeddings=embeddings_list,
        documents=chunks,
        metadatas=[{"file": f} for f in files],
        ids=ids
    )
    print(f"Saved {len(chunks)} chunks to ChromaDB collection '{collection_name}' at '{db_path or 'memory'}'")

def save_to_pgvector(chunks, embeddings, files, db_params):
    """Save chunks and embeddings to a local PostgreSQL (PGVector) table."""
    if psycopg2 is None:
        raise ImportError('psycopg2 is not installed. Install it to use PGVector.')
    conn = psycopg2.connect(
        dbname=db_params['dbname'], user=db_params['user'], password=db_params['password'],
        host=db_params['host'], port=db_params.get('port', 5432)
    )
    cur = conn.cursor()
    # Create table if not exists
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {db_params['table']} (
            id SERIAL PRIMARY KEY,
            chunk TEXT,
            embedding VECTOR({len(embeddings[0])}),
            file TEXT
        );
    """)
    # Insert data
    records = [(c, list(e), f) for c, e, f in zip(chunks, embeddings, files)]
    execute_values(
        cur,
        f"INSERT INTO {db_params['table']} (chunk, embedding, file) VALUES %s",
        records
    )
    conn.commit()
    cur.close()
    conn.close()
    print(f"Saved {len(chunks)} chunks to PGVector table '{db_params['table']}' in DB '{db_params['dbname']}'")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Embed multiple files (txt, pdf, json) using Ollama.")
    parser.add_argument('files', nargs='+', help='List of files to embed')
    parser.add_argument('--output', default='multi_embeddings.pkl', help='Output embedding file (if not using chroma/pgvector)')
    parser.add_argument('--chroma', action='store_true', help='Store embeddings in a local ChromaDB')
    parser.add_argument('--chroma_path', help='ChromaDB: path to local persistent DB directory')
    parser.add_argument('--chroma_collection', default='default', help='ChromaDB: collection name')
    parser.add_argument('--pgvector', action='store_true', help='Store embeddings in a local PostgreSQL (PGVector) DB')
    parser.add_argument('--pgvector_dbname', help='PGVector: database name')
    parser.add_argument('--pgvector_user', help='PGVector: user')
    parser.add_argument('--pgvector_password', help='PGVector: password')
    parser.add_argument('--pgvector_host', help='PGVector: host')
    parser.add_argument('--pgvector_port', type=int, default=5432, help='PGVector: port')
    parser.add_argument('--pgvector_table', default='embeddings', help='PGVector: table name')
    args = parser.parse_args()

    all_chunks = []
    all_embeddings = []
    all_files = []

    for file_path in args.files:
        ext = Path(file_path).suffix.lower()
        if ext == '.pdf':
            text = extract_text_from_pdf(file_path)
            chunks = split_text(text)
        elif ext == '.txt':
            text = extract_text_from_txt(file_path)
            chunks = split_text(text)
        elif ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Each chunk is a single JSON object from the array
            chunks = [json.dumps(obj, ensure_ascii=False, indent=2) for obj in data]
        else:
            print(f"Unsupported file extension: {ext} for {file_path}")
            continue
        print(f"{file_path}: Split into {len(chunks)} chunks")
        embeddings = embed_chunks(chunks)
        all_chunks.extend(chunks)
        all_embeddings.extend(embeddings)
        all_files.extend([file_path]*len(chunks))

    if args.chroma:
        save_to_chromadb(all_chunks, all_embeddings, all_files, db_path=args.chroma_path, collection_name=args.chroma_collection)
    elif args.pgvector:
        db_params = {
            'dbname': args.pgvector_dbname,
            'user': args.pgvector_user,
            'password': args.pgvector_password,
            'host': args.pgvector_host,
            'port': args.pgvector_port,
            'table': args.pgvector_table,
        }
        save_to_pgvector(all_chunks, all_embeddings, all_files, db_params)
    else:
        save_data(all_chunks, all_embeddings, all_files, args.output)

if __name__ == "__main__":
    main()
