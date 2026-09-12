# ContextForge — Production Dockerfile
# Multi-stage build for smaller image size

FROM python:3.11-slim AS base

# Prevent Python from buffering stdout/stderr (important for Docker logs)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ENV=production

WORKDIR /app

# Install system dependencies (for grpcio / h2)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy application code
COPY . .

# Create data and logs directories
RUN mkdir -p data/documents data/chroma logs

# Expose API port
EXPOSE 8000

# Run with uvicorn (production settings)
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--no-access-log"]
