# --- Backend Dockerfile ------------------------------------------------
# This packages the Python backend + ALL its dependencies (PyTorch,
# LangChain, embedding models' runtime libs, etc.) into one self-contained
# image. Whoever runs this image - your laptop, a teammate's laptop, or
# AWS - gets the exact same environment. No more "works in PowerShell but
# not Git Bash" style mismatches, because there's no dependency on
# whatever happens to already be installed on the host machine.

FROM python:3.11-slim

# System-level libs some ML packages (torch, chromadb) need under the hood
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first and install BEFORE copying the rest of the code.
# Docker caches layers - if only your .py files change (not requirements),
# this pip install layer is reused instead of re-run, making rebuilds fast.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# NOTE: the "dataset" folder and "medical_db" (persisted vector store)
# should be mounted as volumes at runtime (see docker-compose.yml) rather
# than baked into the image, so you can update documents without
# rebuilding the whole image.
CMD ["uvicorn", "api:main_app", "--host", "0.0.0.0", "--port", "8000"]
