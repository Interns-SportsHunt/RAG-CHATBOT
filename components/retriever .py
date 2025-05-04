"""
Retriever Module
----------------
Supports semantic search over text chunks using different vector stores:
- In-memory (numpy)
- FAISS (if installed)
- PGVector (PostgreSQL, requires pgvector & psycopg2)
- ChromaDB (local vector DB)
- LangChain InMemoryVectorStore

Usage:
    python retriver.py <embedding_file> [--top_k N] [--method dot|cosine|faiss|pgvector|chromadb|langchain_mem]

"""
import pickle
import numpy as np
import ollama
import importlib

# Optional: PGVector dependencies
try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    psycopg2 = None

# Optional: ChromaDB dependencies
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None

# Optional: LangChain InMemoryVectorStore dependencies
try:
    from langchain_community.vectorstores import InMemoryVectorStore
    from langchain_community.embeddings import OllamaEmbeddings
except ImportError:
    InMemoryVectorStore = None
    OllamaEmbeddings = None

# ----------------------
# Data Loading Utilities
# ----------------------
def load_data(file_path):
    """Load chunks and embeddings from a pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['chunks'], data['embeddings']

# ----------------------
# In-Memory Vector Search
# ----------------------
def in_memory_search(query_vec, emb_matrix, top_k=3, method='dot'):
    """Search for similar vectors in-memory using dot or cosine similarity."""
    if method == 'dot':
        similarities = emb_matrix @ query_vec
    elif method == 'cosine':
        emb_norms = np.linalg.norm(emb_matrix, axis=1)
        query_norm = np.linalg.norm(query_vec)
        similarities = (emb_matrix @ query_vec) / (emb_norms * query_norm + 1e-8)
    else:
        raise ValueError(f'Unknown in-memory method: {method}')
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return top_indices

# ----------------------
# FAISS Vector Search
# ----------------------
def faiss_search(query_vec, emb_matrix, top_k=3):
    """Search for similar vectors using FAISS (if installed)."""
    faiss_spec = importlib.util.find_spec('faiss')
    if faiss_spec is None:
        print('faiss not installed, falling back to dot product')
        return in_memory_search(query_vec, emb_matrix, top_k, method='dot')
    import faiss
    index = faiss.IndexFlatL2(emb_matrix.shape[1])
    index.add(emb_matrix.astype('float32'))
    D, I = index.search(query_vec.reshape(1, -1).astype('float32'), top_k)
    return I[0]

# ----------------------
# PGVector Vector Search
# ----------------------
def pgvector_search(query_vec, top_k=3, db_params=None):
    """
    Search for similar vectors using PGVector (PostgreSQL).
    db_params: dict with keys: dbname, user, password, host, port, table, embedding_col, chunk_col
    """
    if psycopg2 is None:
        raise ImportError('psycopg2 is not installed. Install it to use PGVector.')
    conn = psycopg2.connect(
        dbname=db_params['dbname'], user=db_params['user'], password=db_params['password'],
        host=db_params['host'], port=db_params.get('port', 5432)
    )
    cur = conn.cursor()
    # PGVector: '<embedding_col> <-> %s' computes L2 distance
    sql = f"""
        SELECT {db_params['chunk_col']}, {db_params['embedding_col']} <-> %s AS distance
        FROM {db_params['table']}
        ORDER BY distance ASC
        LIMIT %s;
    """
    cur.execute(sql, (list(query_vec), top_k))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return [row[0] for row in results]

# ----------------------
# ChromaDB Vector Search
# ----------------------
def chromadb_search(query, top_k=3, db_path=None, collection_name="default"):
    """
    Search for similar chunks using ChromaDB (local vector DB).
    db_path: path to ChromaDB persistent directory
    collection_name: name of the ChromaDB collection
    """
    if chromadb is None:
        raise ImportError('chromadb is not installed. Install it to use ChromaDB.')
    client = chromadb.PersistentClient(path=db_path) if db_path else chromadb.Client()
    collection = client.get_or_create_collection(collection_name)
    # Embed the query
    query_embedding = ollama.embed(model='nomic-embed-text', input=query)['embeddings']
    # Query ChromaDB
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results['documents'][0] if results['documents'] else []

# ----------------------
# LangChain InMemoryVectorStore Search
# ----------------------
def langchain_inmemory_search(query, chunks, embeddings, top_k=3):
    """Search using LangChain's InMemoryVectorStore."""
    if InMemoryVectorStore is None or OllamaEmbeddings is None:
        raise ImportError('langchain-community is not installed. Install it to use LangChain InMemoryVectorStore.')
    embedding_function = OllamaEmbeddings(model='nomic-embed-text')
    vector_store = InMemoryVectorStore.from_texts(chunks, embedding=embedding_function)
    results = vector_store.similarity_search(query, k=top_k)
    return [doc.page_content for doc in results]

# ----------------------
# Main Retrieval Function
# ----------------------
def find_similar_chunks(query, chunks, embeddings, top_k=3, method='dot', db_params=None, chroma_params=None):
    """
    Find similar chunks to the query using the specified method.
    method: 'dot', 'cosine', 'faiss', 'pgvector', 'chromadb', or 'langchain_mem'
    db_params: required for 'pgvector' method
    chroma_params: required for 'chromadb' method
    """
    if method == 'langchain_mem':
        return langchain_inmemory_search(query, chunks, embeddings, top_k)
    query_embedding = np.array(ollama.embed(model='nomic-embed-text', input=query)['embeddings'])
    query_vec = query_embedding.flatten()
    emb_matrix = np.vstack([np.array(e).flatten() for e in embeddings]) if embeddings is not None else None

    if method in ('dot', 'cosine'):
        top_indices = in_memory_search(query_vec, emb_matrix, top_k, method)
        return [chunks[i] for i in top_indices]
    elif method == 'faiss':
        top_indices = faiss_search(query_vec, emb_matrix, top_k)
        return [chunks[i] for i in top_indices]
    elif method == 'pgvector':
        if db_params is None:
            raise ValueError('db_params must be provided for pgvector search.')
        return pgvector_search(query_vec, top_k, db_params)
    elif method == 'chromadb':
        if chroma_params is None:
            raise ValueError('chroma_params must be provided for chromadb search.')
        return chromadb_search(query, top_k, db_path=chroma_params.get('db_path'), collection_name=chroma_params.get('collection_name', 'default'))
    else:
        raise ValueError(f'Unknown method: {method}')

# ----------------------
# CLI Entry Point
# ----------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Query an embedding file for similar chunks.')
    parser.add_argument('embedding_file', help='Path to the embedding pickle file (not used for pgvector or chromadb)')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top similar chunks to return')
    parser.add_argument('--method', default='dot', choices=['dot', 'cosine', 'faiss', 'pgvector', 'chromadb', 'langchain_mem'], help='Similarity search method')
    parser.add_argument('--pgvector_dbname', help='PGVector: database name')
    parser.add_argument('--pgvector_user', help='PGVector: user')
    parser.add_argument('--pgvector_password', help='PGVector: password')
    parser.add_argument('--pgvector_host', help='PGVector: host')
    parser.add_argument('--pgvector_port', type=int, default=5432, help='PGVector: port')
    parser.add_argument('--pgvector_table', help='PGVector: table name')
    parser.add_argument('--pgvector_embedding_col', help='PGVector: embedding column name')
    parser.add_argument('--pgvector_chunk_col', help='PGVector: chunk column name')
    parser.add_argument('--chromadb_path', help='ChromaDB: path to local persistent DB directory')
    parser.add_argument('--chromadb_collection', default='default', help='ChromaDB: collection name')
    args = parser.parse_args()

    if args.method == 'chromadb':
        chroma_params = {
            'db_path': args.chromadb_path,
            'collection_name': args.chromadb_collection,
        }
        chunks = None
        embeddings = None
        db_params = None
    elif args.method == 'pgvector':
        db_params = {
            'dbname': args.pgvector_dbname,
            'user': args.pgvector_user,
            'password': args.pgvector_password,
            'host': args.pgvector_host,
            'port': args.pgvector_port,
            'table': args.pgvector_table,
            'embedding_col': args.pgvector_embedding_col,
            'chunk_col': args.pgvector_chunk_col,
        }
        chunks = None
        embeddings = None
        chroma_params = None
    else:
        chroma_params = None
        db_params = None
        chunks, embeddings = load_data(args.embedding_file)

    while True:
        query = input("Enter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        results = find_similar_chunks(
            query, chunks, embeddings, top_k=args.top_k, method=args.method, db_params=db_params, chroma_params=chroma_params
        )
        print("\nMost similar chunks:")
        for i, chunk in enumerate(results, 1):
            print(f"[{i}] {chunk}\n")

if __name__ == "__main__":
    main()
