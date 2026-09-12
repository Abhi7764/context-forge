"""Prompt templates for RAG.

Separates system instructions, retrieved context, and user question
into a structured prompt. Keeps prompt logic out of API routes and
the RAG pipeline.
"""

from src.ingestion.loader import Document

RAG_SYSTEM_PROMPT = """You are a helpful, accurate, and professional assistant.

Your task is to answer the user's question using ONLY the provided context documents.

Rules:
1. Base your answer strictly on the provided context.
2. Do NOT invent, fabricate, or assume information that is not in the context.
3. If the context does not contain enough information to answer the question, clearly state.
4. Be concise, factual, and well-structured.
5. If the context contains conflicting information, acknowledge the conflict and present both perspectives.
"""


def build_context_string(documents: list[Document]) -> str:
    """Build a formatted context string from retrieved documents.

    Args:
        documents: List of retrieved Document objects.

    Returns:
        A formatted string with numbered context blocks and source info.
    """
    if not documents:
        return "No relevant documents found."

    context_parts: list[str] = []

    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("filename", "Unknown")
        page = doc.metadata.get("page")
        score = doc.metadata.get("similarity_score")

        header = f"[Document {i}] Source: {source}"
        if page:
            header += f", Page: {page}"
        if score:
            header += f", Relevance: {score}"

        context_parts.append(f"{header}\n{doc.text}")

    return "\n\n---\n\n".join(context_parts)


def build_rag_prompt(context_docs: list[Document], question: str) -> str:
    """Build the full RAG prompt combining context and question.

    Args:
        context_docs: Retrieved document chunks.
        question: The user's question.

    Returns:
        The formatted prompt string for the LLM.
    """
    context = build_context_string(context_docs)

    prompt = f"""CONTEXT:
{context}

USER QUESTION:
{question}

Based on the context provided above, please answer the user's question. 
If the answer cannot be found in the context, say so explicitly."""

    return prompt
