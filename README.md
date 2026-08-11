# Medical RAG Assistant

A Retrieval-Augmented Generation (RAG) system that answers medical questions grounded in a reference document, using hybrid retrieval (BM25 + dense vector search), cross-encoder reranking, and a domain-specific biomedical embedding model. Full-stack (FastAPI + React), containerized with Docker, and deployed live on AWS with a CI/CD pipeline.

**🔗 Live demo:** `http://13.232.59.169:5173/`

> **Disclaimer:** Educational project. Not a substitute for professional medical advice.

---

## Architecture

```
┌─────────────┐      HTTP POST /ask      ┌──────────────────┐
│   React UI   │  ──────────────────────>  │   FastAPI backend │
│ (Vite, port  │                            │   (port 8000)     │
│    5173)     │  <──────────────────────  │                    │
└─────────────┘      JSON { answer,        └─────────┬─────────┘
                       sources }                       │
                                                        ▼
                                         ┌──────────────────────────┐
                                         │  Hybrid Retriever         │
                                         │  BM25 + Chroma vector DB  │
                                         │  (medical embeddings)     │
                                         └────────────┬─────────────┘
                                                       ▼
                                         ┌──────────────────────────┐
                                         │  Cross-Encoder Reranker   │
                                         └────────────┬─────────────┘
                                                       ▼
                                         ┌──────────────────────────┐
                                         │  Gemini LLM (generation)  │
                                         └──────────────────────────┘
```

**Deployment:** Docker Compose (backend + frontend containers) → AWS EC2 (Ubuntu, Elastic IP) → GitHub Actions CI/CD (build → push to Amazon ECR → SSH redeploy) on every push to `main`.

---

## Highlights

- **Retrieval:** domain-tuned embeddings (`S-PubMedBert-MS-MARCO`) + hybrid BM25/vector search + cross-encoder reranking — not just a naive top-k vector lookup.
- **Full-stack, containerized:** FastAPI backend and React frontend, each with their own Dockerfile, orchestrated via `docker-compose`.
- **Live on AWS**, real infrastructure: EC2, Elastic IP, IAM (least-privilege user, no root usage), security groups, persistent Docker volumes for the model cache and vector store.
- **CI/CD:** GitHub Actions pipeline builds the image, pushes to Amazon ECR, and redeploys to EC2 over SSH automatically on push — no manual server access needed for routine updates.
- **In-progress:** custom embedding fine-tuning pipeline (synthetic query generation + contrastive training via `sentence-transformers`) to adapt the base model to this specific document set.

---

## Tech stack

| Layer | Tools |
|---|---|
| LLM | Google Gemini (`gemini-flash-latest`) via `langchain-google-genai` |
| Embeddings | `pritamdeka/S-PubMedBert-MS-MARCO` (HuggingFace), with a custom fine-tuning pipeline in progress |
| Retrieval | BM25 + Chroma vector store, combined via `EnsembleRetriever` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Orchestration | LangChain |
| Backend / Frontend | FastAPI + Uvicorn / React 18 + Vite |
| Infra | Docker, Docker Compose, AWS EC2, Elastic IP, IAM, Amazon ECR |
| CI/CD | GitHub Actions (build → ECR → SSH redeploy) |

---

## Project structure

```
project/
├── backend/
│   ├── api.py                # FastAPI app: /ask, /health
│   ├── main.py                 # RAG pipeline: chunking, retrieval, reranking, generation
│   ├── training/
│   │   └── finetune_embeddings.py   # synthetic-data embedding fine-tuning pipeline
│   ├── requirements.txt
│   ├── Dockerfile / .dockerignore
│   └── dataset/ · medical_db/       # source docs and vector store (gitignored)
├── frontend/
│   ├── src/ (App.jsx, main.jsx, index.css)
│   ├── package.json · vite.config.js
│   └── Dockerfile / .dockerignore
├── .github/workflows/deploy.yml   # CI/CD: build → ECR → EC2 redeploy
├── docker-compose.yml
└── .gitignore
```

---

## Running it

**Docker (recommended, matches production):**
```bash
docker compose build
docker compose up
```
Backend: `http://localhost:8000` (`/health` check) · Frontend: `http://localhost:5173`
Requires `backend/.env` with `GOOGLE_API_KEY` (see `backend/.env.example`).

**Without Docker:**
```bash
cd backend && python -m venv .venv && source .venv/Scripts/activate   # or Activate.ps1 on Windows
pip install -r requirements.txt
uvicorn api:main_app --reload --port 8000
```
```bash
cd frontend && npm install && npm run dev
```

---

## Deployment notes

Deployed on an AWS EC2 (Ubuntu) instance behind an Elastic IP, running the same Docker Compose stack as local. Key production considerations handled along the way:
- **Persistent volumes** for the Hugging Face model cache and Chroma vector store, so containers don't re-download/re-embed on every restart.
- **Disk sizing** — increased EBS volume and cleaned Docker's build cache/image layers to fit the ML dependency footprint (PyTorch + two transformer models) comfortably.
- **Build-time vs. runtime config** — the frontend's API URL is injected as a Docker build argument (`VITE_API_URL`), since Vite bakes env vars in at build time, not container start.
- **IAM-based access** — an IAM user with scoped credentials handles both local AWS CLI use and CI/CD, rather than using root account keys.

---

## Roadmap

- [x] Full-stack app (FastAPI + React)
- [x] Dockerized, deployed live on AWS EC2 with a stable public URL
- [x] CI/CD via GitHub Actions (build → ECR → automated redeploy)
- [ ] Embedding fine-tuning on domain-specific synthetic query data (pipeline built, evaluation pending)
<!-- - [ ] Groundedness/hallucination check on generated answers -->
<!-- g -->
- [ ] Multi-query retrieval / HyDE
<!-- - [ ] Parent-document retrieval -->
<!-- - [ ] Nginx reverse proxy + HTTPS -->
- [ ] Frontend polish (streaming responses, mobile styling)