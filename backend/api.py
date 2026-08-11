"""
FastAPI wrapper around the existing RAG pipeline (main.py logic).

WHY THIS FILE EXISTS:
main.py is a CLI app - it calls input()/print() in a while loop. A browser
can't call Python functions directly, so this file exposes the same
pipeline as an HTTP API that a React frontend (or anything else - curl,
Postman, a mobile app) can call over the network.

Architecture:
  React (browser) --HTTP POST /ask--> FastAPI (this file) --calls--> main.py functions

Run with:
  uvicorn api:main_app --reload --port 8000
"""

import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Reuse every function you already built and tested in main.py - the API
# layer does NOT reimplement retrieval/generation logic, it just exposes it.
from main import (
    load_environment,
    get_device,
    load_documents,
    build_chunks,
    build_vector_store,
    build_hybrid_retriever,
    build_reranking_retriever,
    load_llm,
    ask_llm,
    retrieve_context,
    build_prompt,
    format_response,
    get_persist_dir,
    DEFAULT_EMBEDDING_MODEL,
)
from langchain_huggingface import HuggingFaceEmbeddings

main_app = FastAPI(title="Medical RAG API")

# CORS: the browser blocks JS on one origin (e.g. localhost:5173, the React
# dev server) from calling a different origin (localhost:8000, this API)
# unless the API explicitly allows it. This is a browser security rule, not
# a Python one - curl/Postman never hit this. In production, replace "*"
# with your actual deployed frontend URL.
main_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pipeline objects are built ONCE at startup, not per-request ----------
# Loading the embedding model + LLM + vector DB takes real time (seconds).
# Doing that on every HTTP request would make each API call painfully slow.
# Instead we build everything once when the server starts, and reuse it for
# every request that comes in afterward.
_state = {}


@main_app.on_event("startup")
def startup():
    load_environment()
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY is not set. Create a .env file.")

    embeddings = HuggingFaceEmbeddings(
        model_name=DEFAULT_EMBEDDING_MODEL,
        model_kwargs={"device": get_device()},
    )
    persist_dir = get_persist_dir(DEFAULT_EMBEDDING_MODEL)

    documents = load_documents()
    chunks = build_chunks(documents, embeddings, use_semantic_chunking=False)
    vector_db = build_vector_store(chunks, embeddings, persist_dir)

    hybrid = build_hybrid_retriever(chunks, vector_db, fetch_k=15)
    retriever = build_reranking_retriever(hybrid, top_n=4)

    llm = load_llm(provider="gemini", model_name="gemini-flash-latest", temperature=0.2)

    _state["retriever"] = retriever
    _state["llm"] = llm
    print("RAG pipeline ready.")


class AskRequest(BaseModel):
    question: str
    k: int = 3


class AskResponse(BaseModel):
    answer: str
    sources: List[str]


@main_app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    context, sources = retrieve_context(_state["retriever"], req.question, k=req.k)
    prompt = build_prompt(req.question, context)
    raw_answer = ask_llm(_state["llm"], prompt)
    full_text = format_response(raw_answer, sources)
    return AskResponse(answer=full_text, sources=sources)


@main_app.get("/health")
def health():
    return {"status": "ok"}
