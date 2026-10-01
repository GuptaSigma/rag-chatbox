from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from rag_engine import rag


class RAGState(TypedDict):
    query: str
    retrieved_docs: List[Dict[str, Any]]
    answer: str
    confidence: float


def retrieve_docs(state: RAGState) -> RAGState:
    query = state["query"]

    retrieved_docs = rag.hybrid_search(
        query=query,
        top_k=3
    )

    state["retrieved_docs"] = retrieved_docs
    return state


def generate_answer(state: RAGState) -> RAGState:
    retrieved_docs = state["retrieved_docs"]

    if not retrieved_docs:
        state["answer"] = "Information not found in the knowledge base."
        state["confidence"] = 0.0
        return state

    # Build answer directly from retrieved PDF chunks
    answer_parts = []

    for doc in retrieved_docs:
        answer_parts.append(
            f"[{doc['doc_name']}, Page {doc['page']}]\n"
            f"{doc['text']}"
        )

    state["answer"] = "\n\n---\n\n".join(answer_parts)

    # Temporary retrieval confidence
    state["confidence"] = min(
        1.0,
        len(retrieved_docs) / 3
    )

    return state


# Create LangGraph workflow
workflow = StateGraph(RAGState)

workflow.add_node("retrieve_docs", retrieve_docs)
workflow.add_node("generate_answer", generate_answer)

workflow.set_entry_point("retrieve_docs")

workflow.add_edge(
    "retrieve_docs",
    "generate_answer"
)

workflow.add_edge(
    "generate_answer",
    END
)

rag_graph = workflow.compile()