# Christ University RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for Christ University. This project scrapes, embeds, and semantically retrieves course data to answer user questions about university courses using modern LLMs and vector databases.

---

## Features
- **Web Scraping**: Collects all course listings and detailed course data from the Christ University website.
- **Data Embedding**: Converts scraped data into vector embeddings using Ollama's embedding model.
- **Semantic Retrieval**: Finds the most relevant course information for a user's query.
- **RAG Chatbot**: Answers questions using retrieved context and an LLM (Ollama or Azure OpenAI).
- **Modular Design**: Easily extend or swap out components for scraping, embedding, or retrieval.

---

## Directory Structure

```
main.py
pyproject.toml
README.md
components/
    embedder.py
    retriever .py
    tests/
embeddings/
    rag_embedding.pkl
scrape_data/
    data/
        all_courses.json
        courses_json/
    web_scraping/
        course_data.py
        get_courses.py
simple_rag/
    rag.py
test_files/
```

---

## Data Preparation Workflow

### 1. Scrape All Course URLs
Run the following to collect all course names, URLs, and locations:
```bash
python scrape_data/web_scraping/get_courses.py
```
- Output: `scrape_data/data/all_courses.json`

### 2. Scrape Detailed Data for Each Course
Run the following to fetch detailed information for each course:
```bash
python scrape_data/web_scraping/course_data.py --input scrape_data/data/all_courses.json
```
- Output: Individual JSON files for each course in `scrape_data/data/courses_json/`

---

## Embedding and Retrieval

### 3. Embed the Scraped Data
Use the embedder script to convert your data into vector embeddings for semantic search.

#### Basic Usage (Save to Pickle File)
```bash
python components/embedder.py scrape_data/data/all_courses.json --output embeddings/rag_embedding.pkl
```

#### Advanced Usage
- **Store in ChromaDB:**
  ```bash
  python components/embedder.py scrape_data/data/all_courses.json --chroma --chroma_path ./chromadb_dir --chroma_collection christ_courses
  ```
- **Store in PGVector (PostgreSQL):**
  ```bash
  python components/embedder.py scrape_data/data/all_courses.json --pgvector --pgvector_dbname mydb --pgvector_user myuser --pgvector_password mypass --pgvector_host localhost --pgvector_table christ_courses
  ```

#### Supported File Types
- PDF (`.pdf`)
- Text (`.txt`)
- JSON (array of course objects)

#### What Happens Internally
- The script detects the file type and extracts text accordingly.
- Text is split into manageable chunks using LangChain’s text splitter.
- Each chunk is embedded using Ollama’s `nomic-embed-text` model.
- Embeddings are saved to your chosen backend (pickle, ChromaDB, or PGVector).

---

## RAG Chatbot Pipeline

### 4. Run the Chatbot
Use the chatbot script to answer questions about Christ University courses:
```bash
python simple_rag/rag.py
```
- Loads embeddings and retrieves relevant context for a user query.
- Generates answers using an LLM (Ollama or Azure OpenAI).

---

## Components Details

### `components/embedder.py`
- Embeds PDF, TXT, or JSON files using Ollama.
- Supports saving embeddings to a pickle file, ChromaDB, or PGVector.
- CLI arguments allow flexible backend selection and file input.

### `components/retriever .py`
- Provides semantic search over the generated embeddings.
- Supports in-memory, ChromaDB, PGVector, and other retrieval backends.
- Use this to query for the most relevant course chunks for a given question.

---

## Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set up environment variables:**
   - Create a `.env` file with your Azure OpenAI API key if using Azure.
3. **Run the workflow:**
   - Scrape data → Embed data → Run chatbot (see steps above).

---

## Example Usage

- Ask questions like:
  - "What is the eligibility for MSc Data Science?"
  - "Show me the fee structure for BBA courses."
  - "What are the career prospects after BCom?"

---

## Extending the Project
- Add new data sources by updating the scraping scripts.
- Swap out embedding or retrieval backends in the components folder.
- Integrate new LLMs or vector stores as needed.

---

## Troubleshooting
- If scraping fails, check your internet connection and the structure of the Christ University website.
- For embedding issues, ensure Ollama and all dependencies are installed.
- For database issues, verify your ChromaDB or PostgreSQL setup.

---

## License
MIT License

---

## Credits
- Christ University (data source)
- Ollama, LangChain, ChromaDB, PGVector, Azure OpenAI (technologies)
