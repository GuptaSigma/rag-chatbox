from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from rag_engine import rag

app = FastAPI(title="Enterprise RAG Engine API", version="4.0")

class AdvancedQueryRequest(BaseModel):
    query: str
    k: int = 3
    doc_filter: Optional[str] = None  # Specific PDF name to filter

@app.get("/")
def home():
    return {"message": "Welcome to Enterprise Hybrid RAG Engine API! Use /docs for OpenAPI specs."}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed.")
    
    content = await file.read()
    total_chunks = rag.add_pdf(content, file.filename)
    
    return {
        "filename": file.filename,
        "total_chunks_in_db": total_chunks,
        "status": "Processed with Page Metadata & Vector+BM25 indexing."
    }

@app.post("/search-hybrid")
def search_hybrid(payload: AdvancedQueryRequest):
    results = rag.hybrid_search(payload.query, payload.k, payload.doc_filter)
    if not results:
        raise HTTPException(status_code=400, detail="No matching document chunks found.")
    return {"query": payload.query, "results": results}

@app.post("/ask-ai")
def ask_ai(payload: AdvancedQueryRequest):
    retrieved_docs = rag.hybrid_search(payload.query, payload.k, payload.doc_filter)
    if not retrieved_docs:
        raise HTTPException(status_code=400, detail="No documents available in database.")
    
    ai_answer = rag.generate_answer(payload.query, retrieved_docs)
    
    return {
        "query": payload.query,
        "answer": ai_answer,
        "sources": [
            {"doc": d["doc_name"], "page": d["page"], "snippet": d["text"][:100] + "..."} 
            for d in retrieved_docs
        ]
    }

@app.delete("/clear-db")
def clear_db():
    rag.clear_storage()
    return {"message": "Vector DB & Keyword Index cleared successfully."}