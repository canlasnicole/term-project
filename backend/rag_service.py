"""Module 2.5 - RAG (Retrieval-Augmented Generation) document Q&A with LangChain.

This re-implements a typical n8n RAG workflow, node by node:

  n8n node                         ->  LangChain code here
  -------------------------------      ------------------------------------------
  Default Data Loader (PDF/TXT)    ->  PyPDFLoader / TextLoader
  Recursive Character Text Splitter->  RecursiveCharacterTextSplitter
  Embeddings Google Gemini node    ->  GeminiEmbeddings (gemini-embedding-001)
  Vector Store (insert)            ->  FAISS.from_documents
  Vector Store Retriever (top-k)   ->  vectorstore.similarity_search_with_score
  AI Agent / Chat model (Gemini)   ->  ChatPromptTemplate | ChatGoogleGenerativeAI
"""
import os
import tempfile
import threading
from typing import List

import numpy as np

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import EMBEDDING_MODEL, GEMINI_API_KEY, GEMINI_MODEL

_lock = threading.Lock()
_state = {"store": None, "filename": None, "chunks": 0}
_embedder = None


class GeminiEmbeddings(Embeddings):
    """Turns text into vectors (lists of numbers) with the Gemini embedding model.
    The model runs on Google's servers, so it uses almost no memory on our free server."""

    BATCH = 100          # the API accepts up to 100 texts per request
    DIM = 768            # shorter vectors: smaller and faster, still accurate (MRL)

    def __init__(self, model_name: str):
        from google.genai import types
        from .gemini_service import get_client
        self.client, self.types, self.model = get_client(), types, model_name

    def _embed(self, texts: List[str], task: str) -> List[List[float]]:
        vectors = []
        for i in range(0, len(texts), self.BATCH):
            resp = self.client.models.embed_content(
                model=self.model, contents=texts[i:i + self.BATCH],
                config=self.types.EmbedContentConfig(task_type=task,
                                                     output_dimensionality=self.DIM))
            for e in resp.embeddings:
                v = np.array(e.values, dtype="float32")
                vectors.append((v / (np.linalg.norm(v) or 1.0)).tolist())   # unit length
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def get_embedder() -> Embeddings:
    global _embedder
    if _embedder is None:
        _embedder = GeminiEmbeddings(EMBEDDING_MODEL)
    return _embedder


PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions using ONLY the context from the uploaded document. "
     "If the answer is not in the context, say: \"I could not find this in the document.\" "
     "Answer in the same language as the question. Keep it short and clear."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def ingest(filename: str, data: bytes, chunk_size: int = 800, chunk_overlap: int = 150) -> dict:
    """Load -> split -> embed -> store in FAISS (replaces the previous document)."""
    suffix = os.path.splitext(filename)[1].lower()
    if suffix not in (".pdf", ".txt", ".md"):
        raise ValueError("Upload a .pdf, .txt or .md file.")

    # Loaders need a file path, so write the upload to a temporary file first
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        loader = PyPDFLoader(path) if suffix == ".pdf" else TextLoader(path, encoding="utf-8",
                                                                      autodetect_encoding=True)
        docs = loader.load()
    finally:
        os.remove(path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    if not chunks:
        raise ValueError("No text found. If this is a scanned PDF, it has no text layer.")

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set (needed for embeddings).")
    if len(chunks) > 400:
        raise ValueError(f"Document is too long ({len(chunks)} chunks). "
                         "Use a shorter file (about 100 pages or less).")
    try:
        store = FAISS.from_documents(chunks, get_embedder())
    except Exception as e:
        raise ValueError(f"Embedding failed (Gemini): {e}") from e
    with _lock:
        _state.update(store=store, filename=filename, chunks=len(chunks))
    return status()


def status() -> dict:
    return {"ready": _state["store"] is not None,
            "filename": _state["filename"], "chunks": _state["chunks"]}


def ask(question: str, k: int = 4) -> dict:
    if _state["store"] is None:
        raise ValueError("Upload a document first.")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set.")

    # 1) Retrieve: find the k chunks whose vectors are closest to the question vector
    hits = _state["store"].similarity_search_with_score(question, k=k)
    context = "\n\n---\n\n".join(doc.page_content for doc, _ in hits)

    # 2) Generate: Gemini writes the answer from that context (LCEL chain: prompt | llm | parser)
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GEMINI_API_KEY,
                                 temperature=0.2)
    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    sources = []
    for rank, (doc, score) in enumerate(hits, start=1):
        page = doc.metadata.get("page")
        sources.append({"rank": rank, "text": doc.page_content,
                        "page": page + 1 if isinstance(page, int) else None,   # 1-based page
                        "score": round(float(score), 4)})
    return {"answer": answer, "sources": sources}
