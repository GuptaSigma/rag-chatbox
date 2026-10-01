import os
import io
import pickle
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


INDEX_FILE = "faiss_index.bin"
DOCS_FILE = "documents.pkl"


class AdvancedRAGEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print("Loading Embedding Model & Engine...")

        self.model = SentenceTransformer(model_name)
        self.dimension = 384

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )

        self.documents: List[Dict[str, Any]] = []
        self.bm25 = None

        self._load_storage()

    def _rebuild_bm25(self):
        """Build BM25 index for keyword search."""

        if self.documents:
            corpus = [
                doc["text"].lower().split()
                for doc in self.documents
            ]

            self.bm25 = BM25Okapi(corpus)

        else:
            self.bm25 = None

    def _load_storage(self):
        if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):

            print("Loading existing index and metadata from disk...")

            self.index = faiss.read_index(INDEX_FILE)

            with open(DOCS_FILE, "rb") as f:
                self.documents = pickle.load(f)

            self._rebuild_bm25()

        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = []

    def _save_storage(self):
        faiss.write_index(
            self.index,
            INDEX_FILE
        )

        with open(DOCS_FILE, "wb") as f:
            pickle.dump(
                self.documents,
                f
            )

    def process_pdf(
        self,
        file_bytes: bytes,
        filename: str
    ) -> List[Dict[str, Any]]:

        pdf = PdfReader(
            io.BytesIO(file_bytes)
        )

        chunks_with_metadata = []

        for page_idx, page in enumerate(pdf.pages):

            text = page.extract_text()

            if text:
                page_chunks = self.text_splitter.split_text(text)

                for chunk in page_chunks:

                    chunks_with_metadata.append({
                        "text": chunk,
                        "page": page_idx + 1,
                        "doc_name": filename
                    })

        return chunks_with_metadata

    def add_pdf(
        self,
        file_bytes: bytes,
        filename: str
    ) -> int:

        chunks = self.process_pdf(
            file_bytes,
            filename
        )

        if not chunks:
            return len(self.documents)

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        vectors = self.model.encode(
            texts
        ).astype("float32")

        self.index.add(vectors)

        self.documents.extend(chunks)

        self._rebuild_bm25()
        self._save_storage()

        return len(self.documents)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 3,
        doc_filter: str = None
    ) -> List[Dict[str, Any]]:

        """Hybrid search using FAISS + BM25."""

        if not self.documents:
            return []

        # -------------------------
        # 1. Vector Search
        # -------------------------

        query_vector = self.model.encode(
            [query]
        ).astype("float32")

        faiss_k = min(
            top_k * 2,
            len(self.documents)
        )

        distances, indices = self.index.search(
            query_vector,
            faiss_k
        )

        vector_results = {}

        for rank, idx in enumerate(indices[0]):

            if idx < len(self.documents):
                vector_results[idx] = rank + 1

        # -------------------------
        # 2. BM25 Search
        # -------------------------

        bm25_results = {}

        if self.bm25:

            tokenized_query = query.lower().split()

            scores = self.bm25.get_scores(
                tokenized_query
            )

            top_bm25_indices = np.argsort(
                scores
            )[::-1][:faiss_k]

            for rank, idx in enumerate(
                top_bm25_indices
            ):
                bm25_results[idx] = rank + 1

        # -------------------------
        # 3. Reciprocal Rank Fusion
        # -------------------------

        combined_scores = {}

        all_indices = (
            set(vector_results.keys())
            .union(set(bm25_results.keys()))
        )

        for idx in all_indices:

            v_rank = vector_results.get(
                idx,
                100
            )

            b_rank = bm25_results.get(
                idx,
                100
            )

            combined_scores[idx] = (
                1.0 / (60 + v_rank)
                +
                1.0 / (60 + b_rank)
            )

        sorted_indices = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # -------------------------
        # 4. Build Results
        # -------------------------

        results = []

        for idx, score in sorted_indices:

            doc = self.documents[idx]

            if (
                doc_filter
                and doc["doc_name"] != doc_filter
            ):
                continue

            results.append(doc)

            if len(results) == top_k:
                break

        return results

    def clear_storage(self):

        self.index = faiss.IndexFlatL2(
            self.dimension
        )

        self.documents = []

        self.bm25 = None

        if os.path.exists(INDEX_FILE):
            os.remove(INDEX_FILE)

        if os.path.exists(DOCS_FILE):
            os.remove(DOCS_FILE)


rag = AdvancedRAGEngine()