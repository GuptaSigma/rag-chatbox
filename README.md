# Agentic AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with **Python, FastAPI, LangGraph, FAISS, BM25, and Sentence Transformers**.

The system processes the provided **Agentic AI eBook PDF**, retrieves relevant document chunks using hybrid semantic and keyword search, and returns grounded answers with source document and page information.

---

## 🚀 Features

- PDF document ingestion
- Automatic text extraction and chunking
- Sentence Transformer embeddings
- FAISS vector search
- BM25 keyword search
- Hybrid retrieval using Reciprocal Rank Fusion (RRF)
- LangGraph-based RAG workflow
- Source and page metadata
- FastAPI REST API
- Swagger/OpenAPI documentation
- Knowledge-base grounded responses
- No external LLM/API key required for the current implementation

---

## 🏗️ Architecture

```text
                 PDF Document
                      │
                      ▼
               PDF Text Extraction
                      │
                      ▼
                  Chunking
                      │
                      ▼
          Sentence Transformer Embeddings
                      │
              ┌───────┴────────┐
              ▼                ▼
          FAISS Search      BM25 Search
              │                │
              └───────┬────────┘
                      ▼
             Reciprocal Rank Fusion
                      │
                      ▼
                  LangGraph
                      │
              ┌───────┴────────┐
              ▼                ▼
        Retrieve Documents   Generate Answer
              │                │
              └───────┬────────┘
                      ▼
                  FastAPI
                      │
                      ▼
                   Response
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| FastAPI | REST API |
| LangGraph | RAG workflow orchestration |
| Sentence Transformers | Text embeddings |
| FAISS | Vector similarity search |
| BM25 | Keyword-based retrieval |
| LangChain Text Splitters | Document chunking |
| PyPDF | PDF text extraction |
| Pydantic | API request validation |
| Uvicorn | ASGI server |

---

## 📁 Project Structure

```text
rag-chatbox/
│
├── main.py
├── rag_engine.py
├── langgraph_rag.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── Agentic AI PDF
    └── Ebook-Agentic-AI.pdf
```

Generated files such as the FAISS index and document metadata are created locally and are excluded from Git.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd rag-chatbox
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 📄 Upload the PDF

Use the `/upload-pdf` endpoint.

Upload:

```text
Ebook-Agentic-AI.pdf
```

The system will:

1. Extract text from the PDF.
2. Split the text into chunks.
3. Generate embeddings.
4. Store embeddings in FAISS.
5. Build the BM25 keyword index.
6. Store document metadata including page numbers.

---

## 💬 Ask Questions

Use:

```text
POST /ask-ai
```

Example request:

```json
{
  "query": "What is Agentic AI?",
  "k": 3
}
```

Example response:

```json
{
  "query": "What is Agentic AI?",
  "answer": "[Ebook-Agentic-AI.pdf, Page 7] ...",
  "confidence": 1.0,
  "retrieved_docs": [
    {
      "text": "...",
      "page": 7,
      "doc_name": "Ebook-Agentic-AI.pdf"
    }
  ]
}
```

The response includes the retrieved document chunks and their source page information.

---

## 🔎 Hybrid Retrieval

The project combines two retrieval techniques.

### Semantic Search

Sentence Transformers convert the query and document chunks into embeddings.

FAISS is then used to find semantically similar chunks.

### Keyword Search

BM25 searches for keyword-level relevance between the query and document chunks.

### Reciprocal Rank Fusion

The results from FAISS and BM25 are combined using Reciprocal Rank Fusion (RRF).

```text
FAISS Results
      +
BM25 Results
      ↓
    RRF
      ↓
Top Relevant Chunks
```

This helps combine semantic similarity with exact keyword matching.

---

## 🧠 LangGraph Workflow

The RAG pipeline is implemented using LangGraph.

```text
User Query
    ↓
retrieve_docs
    ↓
Hybrid Search
    ↓
generate_answer
    ↓
Final Response
```

### `retrieve_docs`

Retrieves the most relevant document chunks using the hybrid FAISS + BM25 search system.

### `generate_answer`

Builds the final response from the retrieved knowledge-base content and attaches source/page information.

---

## 🧪 Sample Queries

The following questions can be used to test the chatbot:

```text
1. What is Agentic AI?

2. How does Agentic AI differ from traditional AI?

3. What can Agentic AI do?

4. What value does Agentic AI provide?

5. What are the characteristics of Agentic AI?

6. What are the key concepts discussed in the Agentic AI book?
```

The chatbot should answer questions based on the uploaded Agentic AI eBook.

For questions outside the available knowledge base, the system should return:

```text
Information not found in the knowledge base.
```

---

## 🔌 API Endpoints

### `GET /`

Health/welcome endpoint.

### `POST /upload-pdf`

Uploads and indexes a PDF document.

### `POST /search-hybrid`

Performs hybrid FAISS + BM25 retrieval.

Example:

```json
{
  "query": "What is Agentic AI?",
  "k": 3
}
```

### `POST /ask-ai`

Runs the complete LangGraph RAG workflow.

### `DELETE /clear-db`

Clears the local FAISS index and document metadata.

---

## 🔐 Environment Variables

The current implementation does not require an external LLM API key.

No Gemini API key is required.

---

## 📌 Design Decisions

### Why FAISS?

FAISS provides efficient local vector similarity search and avoids requiring an external vector database.

### Why BM25?

BM25 improves retrieval for exact keywords and terminology that may not always be captured by semantic similarity.

### Why Hybrid Search?

Combining semantic and keyword retrieval can provide more relevant document chunks than relying on only one retrieval method.

### Why LangGraph?

LangGraph provides a structured graph-based workflow for orchestrating the retrieval and answer-generation stages of the RAG pipeline.

### Why FastAPI?

FastAPI provides a lightweight REST API and automatically generates interactive Swagger documentation.

---

## 📦 Requirements

Main dependencies include:

```text
fastapi
uvicorn
sentence-transformers
faiss-cpu
numpy
pydantic
pypdf
langchain-text-splitters
python-multipart
python-dotenv
rank-bm25
langgraph
```

---

## 👨‍💻 Author

**Sagar Gupta**

GitHub: `GuptaSigma`

Project: **Agentic AI RAG Chatbot**

---

## 📜 License

This project was created as a technical assignment and demonstration of a RAG-based AI system.