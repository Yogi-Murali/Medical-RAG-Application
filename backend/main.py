import os
import hashlib
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    JSONLoader,
    CSVLoader,
    DirectoryLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

# --- NEW: hybrid retrieval + reranking imports -----------------------------
# BM25Retriever = classic keyword/frequency retriever, complements dense
# vector search (which is weak on exact terms like drug names / dosages).
from langchain_classic.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
)

from langchain_classic.retrievers.document_compressors import (
    CrossEncoderReranker,
)

from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
# SemanticChunker splits text where meaning actually shifts (via embedding
# similarity between sentences) instead of blind fixed-size cuts.
from langchain_experimental.text_splitter import SemanticChunker
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.resolve()
DATASET_DIR = BASE_DIR / "dataset"

# --- NEW: default embedding model swapped to a medical/biomedical model ----
# pritamdeka/S-PubMedBert-MS-MARCO = PubMedBERT further fine-tuned for
# retrieval (MS-MARCO task), so it understands both medical terminology AND
# "does this passage answer this question" relevance - better fit for RAG
# than a generic sentence-transformer.
#
# IMPORTANT CONCEPT: this embedding model is completely independent from the
# GOOGLE_API_KEY / Gemini setup below. Embeddings run 100% locally on your
# machine/GPU and are only used for retrieval (turning text into vectors to
# search over). The Gemini API key is only used later, for the LLM that
# generates the final written answer from the retrieved context. Swapping
# the embedding model does not touch your API key or quota at all - the two
# stages (retrieve -> generate) are fully decoupled. The ONLY case they'd
# interlink is if you used Gemini's own embedding endpoint
# (GoogleGenerativeAIEmbeddings) instead of a local HuggingFace model - not
# what we're doing here.
DEFAULT_EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"


def load_environment() -> None:
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        print("Warning: .env file not found. Copy .env.example to .env and set GOOGLE_API_KEY.")


def get_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# --- NEW: helper to derive a per-embedding-model persist directory ---------
# Vectors from different embedding models are NOT compatible with each other
# (different dimensions / semantic space). If you change embedding models
# but keep writing to the same Chroma folder, you'll get silent garbage
# results or errors. We namespace the persist dir by a short hash of the
# model name so switching models automatically builds a fresh index instead
# of corrupting the old one, and switching back reuses the old one.
def get_persist_dir(embedding_model_name: str) -> Path:
    model_hash = hashlib.sha1(embedding_model_name.encode()).hexdigest()[:8]
    return BASE_DIR / "medical_db" / f"{model_hash}"
# -----------------------------------------------------------------------------


def load_documents() -> List:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

    documents = []

    pdf_loader = DirectoryLoader(DATASET_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents.extend(pdf_loader.load())

    txt_loader = DirectoryLoader(DATASET_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents.extend(txt_loader.load())

    csv_loader = DirectoryLoader(DATASET_DIR, glob="**/*.csv", loader_cls=CSVLoader)
    documents.extend(csv_loader.load())

    json_loader = DirectoryLoader(
        DATASET_DIR,
        glob="**/*.json",
        loader_cls=JSONLoader,
        loader_kwargs={"jq_schema": ".", "text_content": False},
    )
    try:
        documents.extend(json_loader.load())
    except Exception as e:
        print(f"Note: skipped JSON files ({e})")

    print(f"Loaded {len(documents)} document pages/files.")
    return documents


# --- NEW: chunking is now its own function, with a semantic option ---------
def build_chunks(documents, embeddings, use_semantic_chunking: bool = False) -> List:
    """
    Split loaded documents into chunks.

    use_semantic_chunking=False (default): fast, fixed-size splitting with
    sentence-aware separators. Chunk size raised from 600->1000 chars and
    overlap 120->200 so dosage/contraindication info is less likely to be
    cut apart from its surrounding context.

    use_semantic_chunking=True: uses embedding similarity between sentences
    to split where the TOPIC actually changes, producing more coherent
    chunks. Costs more compute at ingestion time (one embedding call per
    sentence group) - worth it for higher-value/smaller document sets,
    probably overkill for very large corpora.
    """
    if use_semantic_chunking:
        print("Using semantic chunking (splits on meaning shift, not fixed size)...")
        splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    else:
        print("Using recursive character chunking (fixed size, sentence-aware)...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks
# -----------------------------------------------------------------------------


def build_vector_store(chunks, embeddings, persist_dir: Path):
    if persist_dir.exists() and any(persist_dir.iterdir()):
        print(f"Loading existing Chroma vector database from {persist_dir}...")
        vector_db = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )
    else:
        print(f"Creating new Chroma vector database at {persist_dir}...")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(persist_dir),
        )
        print("Database persisted.")

    return vector_db


# --- NEW: hybrid retriever (BM25 keyword search + dense vector search) -----
def build_hybrid_retriever(chunks, vector_db, fetch_k: int = 15, bm25_weight: float = 0.4, vector_weight: float = 0.6):
    """
    Combines two retrieval strategies:
      - BM25Retriever: keyword/frequency based, strong on exact terms
        (drug names, ICD codes, precise numbers) that embeddings often miss.
      - vector retriever: dense semantic search, strong on paraphrased /
        conceptually related queries.

    fetch_k is intentionally larger than the final answer count (args.k)
    because this retriever's output gets passed through the reranker below,
    which is much better at picking the truly best passages out of a larger
    candidate pool. Over-fetch here, then narrow with the reranker.
    """
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = fetch_k

    vector_retriever = vector_db.as_retriever(search_kwargs={"k": fetch_k})

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight],
    )
    return hybrid_retriever
# -----------------------------------------------------------------------------


# --- NEW: cross-encoder reranking stage -------------------------------------
def build_reranking_retriever(base_retriever, top_n: int = 4, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """
    Wraps the hybrid retriever with a cross-encoder reranker.

    Why this helps: BM25/vector retrievers score query and passage
    SEPARATELY (fast but approximate). A cross-encoder scores the query and
    passage TOGETHER in one pass, which is much more accurate at judging
    "does this passage actually answer this question" - but too slow to run
    over an entire corpus. So the pattern is: retrieve a wider candidate
    pool cheaply (hybrid retriever, fetch_k above), then rerank down to the
    best top_n with the more expensive/accurate cross-encoder.

    model_name default is a strong general-purpose reranker. For a more
    medical-tuned option, try "ncbi/MedCPT-Cross-Encoder" instead.
    """
    cross_encoder = HuggingFaceCrossEncoder(model_name=model_name)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)

    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )
# -----------------------------------------------------------------------------


def load_llm(provider: str = "gemini", model_name: str = "gemini-flash-latest", temperature: float = 0.2):
    provider = provider.lower().strip()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    elif provider == "openai":
        return ChatOpenAI(model=model_name, temperature=temperature)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def ask_llm(llm, prompt: str) -> str:
    if hasattr(llm, "generate"):
        result = llm.generate([[HumanMessage(content=prompt)]])
        if hasattr(result, "generations") and result.generations:
            return result.generations[0][0].text
        return str(result)
    if hasattr(llm, "invoke"):
        response = llm.invoke(prompt)
        if hasattr(response, "content"):
            return response.content
        return str(response)
    return str(llm(prompt))


def retrieve_context(retriever, question: str, k: int = 3) -> Tuple[str, List[str]]:
    # NOTE: when `retriever` is the reranking retriever, it already returns
    # a narrowed, reranked set (top_n set in build_reranking_retriever), so
    # slicing with [:k] below is just a safety cap.
    if hasattr(retriever, "get_relevant_documents"):
        docs = retriever.get_relevant_documents(question)
    elif hasattr(retriever, "invoke"):
        docs = retriever.invoke(question)
    else:
        raise RuntimeError("Retriever does not support document retrieval")

    context_parts = []
    sources = []
    for doc in docs[:k]:
        context_parts.append(doc.page_content)
        page = doc.metadata.get("page", "N/A")
        source = Path(doc.metadata.get("source", "Unknown")).name
        sources.append(f"{source} (Page {page})")

    return "\n\n".join(context_parts), sources


def build_prompt(question: str, context: str) -> str:
    return f"""
You are a medical information assistant operating in a retrieval-augmented
question-answering system.

Your answer must be grounded ONLY in the supplied medical reference passages.

Rules:
1. Do not invent facts that are not supported by the context.
2. Do not use outside medical knowledge when the context is insufficient.
3. If the context does not contain enough information to answer the question,
   clearly say that the information was not found in the provided reference.
4. Distinguish between symptoms, causes, diagnosis, treatment, complications,
   and prevention when relevant.
5. Do not make a personal diagnosis.
6. Do not prescribe or recommend a personalized treatment plan.
7. Preserve important medical qualifiers such as:
   - may
   - can
   - commonly
   - rarely
   - contraindicated
8. Give a concise answer first, followed by supporting details.
9. Mention the relevant source/page when possible.

Medical reference passages:
---------------------------
{context}
---------------------------

Question:
{question}

Answer:
""".strip()


def format_response(response_text: str, sources: List[str]) -> str:
    def sanitize_response_text(text: str) -> str:
        # Convert Markdown bullet markers at line starts from '*' to '-'
        text = re.sub(r'(?m)^[ \t]*\*\s+', '- ', text)
        # Remove bold and italic markers
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"\*(.*?)\*", r"\1", text)
        # Remove any stray asterisks
        text = text.replace('*', '')
        return text

    cleaned = sanitize_response_text(response_text or "")
    ref_list = "\n".join(f"- {source}" for source in sorted(set(sources))) or "None"
    return f"{cleaned.strip()}\n\n---\n\nDISCLAIMER: Educational purposes only. Consult a qualified healthcare professional."


def run_console(llm, retriever, k: int):
    print("Medical RAG Assistant (type 'exit' to quit)")
    while True:
        question = input("\nAsk a question: ")
        if question.strip().lower() == "exit":
            break
        context, sources = retrieve_context(retriever, question, k=k)
        prompt = build_prompt(question, context)
        answer_text = ask_llm(llm, prompt)
        print(format_response(answer_text, sources))


def main():
    import argparse

    load_environment()
    parser = argparse.ArgumentParser(description="Medical RAG assistant")
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "gemini"), help="LLM provider")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "gemini-flash-latest"), help="Model name")
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.2")), help="LLM temperature")
    parser.add_argument("--k", type=int, default=4, help="Number of final chunks fed into the prompt")

    # --- NEW CLI options for the added techniques ---------------------------
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help="HuggingFace embedding model name (default: medical PubMedBERT retrieval model)",
    )
    parser.add_argument(
        "--semantic-chunking",
        action="store_true",
        default=os.getenv("SEMANTIC_CHUNKING", "false").lower() == "true",
        help="Use embedding-based semantic chunking instead of fixed-size chunking",
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Disable BM25+vector hybrid retrieval and use plain vector search only",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder reranking stage",
    )
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=20,
        help="Number of candidates fetched by the hybrid retriever before reranking",
    )
    # -------------------------------------------------------------------------

    args = parser.parse_args()

    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY is not set. Create a .env file from .env.example.")

    # Embedding model instantiated once, shared by chunking (if semantic)
    # and the vector store. BM25 is unaffected by this - it's pure keyword
    # statistics, not embedding-based at all.
    embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={"device": get_device()},
    )

    persist_dir = get_persist_dir(args.embedding_model)

    documents = load_documents()
    # NOTE: we now always chunk documents in memory (even if a persisted
    # Chroma DB already exists) because the BM25 retriever needs the raw
    # chunk text and is rebuilt fresh each run - it's not persisted to disk
    # like the vector store, and rebuilding it is cheap/fast.
    chunks = build_chunks(documents, embeddings, use_semantic_chunking=args.semantic_chunking)

    vector_db = build_vector_store(chunks, embeddings, persist_dir)

    if args.no_hybrid:
        print("Hybrid retrieval disabled - using plain vector search.")
        retriever = vector_db.as_retriever(search_kwargs={"k": args.fetch_k if not args.no_rerank else args.k})
    else:
        retriever = build_hybrid_retriever(chunks, vector_db, fetch_k=args.fetch_k)

    if not args.no_rerank:
        retriever = build_reranking_retriever(retriever, top_n=args.k)

    llm = load_llm(provider=args.provider, model_name=args.model, temperature=args.temperature)

    run_console(llm, retriever, k=args.k)


if __name__ == "__main__":
    main()
