<p align="center">

  <h1 align="center">🔥 ContextForge</h1>

  <p align="center">
    <strong>RAG system with Hybrid Search, Reranking, and Source Citations</strong>
  </p>

  <p align="center">
    Search your documents using natural language and get answers based on the information in your files.
  </p>

  <p align="center">
    <a href="#-the-problem">Problem</a> •
    <a href="#-features">Features</a> •
    <a href="#-architecture">Architecture</a> •
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-api-usage">API</a> •
    <a href="#-retrieval-pipeline">Pipeline</a> •
    <a href="#-deployment">Deployment</a>
    <a href="#-engineering-decisions--design-patterns">Design Patterns</a> •

  </p>

</p>

---

## 🧩 The Problem

When a company has hundreds of PDFs, contracts, policy documents, CSV files, and internal guides, finding one specific piece of information can take a lot of time.

Traditional keyword search has its limits. A document might say:

> "The cancellation period is 14 days."

But if someone searches:

> "How can I get a refund?"

a keyword-based search may not find the relevant section because the word **refund** isn't present.

There is another problem with using an LLM directly. A general-purpose LLM doesn't automatically have access to a company's private documents. Without the right context, it may generate an answer that doesn't match the actual company policy.

### 💡 What ContextForge Does

ContextForge adds a retrieval layer between the user's question and the LLM.

Instead of sending the question directly to the model, the system:

1. Searches the document collection for relevant information.
2. Combines semantic and keyword-based results.
3. Reranks the retrieved passages.
4. Sends the most relevant context to the LLM.
5. Returns an answer along with the source document information.

This makes the answer easier to verify and keeps the LLM focused on the information retrieved from the user's documents.

---

## 🏢 Real-World Use Cases

ContextForge is useful when teams have information spread across many documents and need to find specific information quickly.

| Use Case | What the Team Has | How ContextForge Can Help | Example Query |
| :--- | :--- | :--- | :--- |
| **Customer Support** | Product manuals, refund policies, FAQs, and support documents. | Find answers without manually searching through several documents. | *"Can a customer request a refund after 14 days?"* |
| **Legal & Contracts** | Vendor agreements, NDAs, contracts, and legal documents. | Find specific clauses, dates, obligations, and contract terms. | *"Which contracts have a 60-day termination notice period?"* |
| **Engineering & IT** | API documentation, technical guides, configuration files, and logs. | Find technical information or troubleshoot errors using natural language. | *"What causes error code 504 during the webhook request?"* |
| **HR & Internal Policies** | Employee handbooks, leave policies, benefits documents, and company guidelines. | Quickly find answers to common internal policy questions. | *"How many days of parental leave are available?"* |
| **Finance & Operations** | Financial policies, invoices, audit documents, and process guides. | Search for specific requirements, rules, and processes. | *"What documents are required for the vendor payment process?"* |
| **Compliance** | Regulatory documents, company policies, and compliance guidelines. | Find relevant requirements and verify them against the source document. | *"What is the reporting requirement for this type of transaction?"* |

---

## ✨ Features

Current capabilities of ContextForge:

- 🔍 **Hybrid Search** — Combines dense vector search with BM25 keyword search.
- 🎯 **Cross-Encoder Reranking** — Re-ranks retrieved chunks before sending them to the LLM.
- 🤖 **Multiple LLM Providers** — Supports OpenAI, Gemini, Anthropic, and Ollama.
- 🗄️ **Pluggable Vector Stores** — Supports Pinecone and ChromaDB through a common interface.
- 📄 **Document Support** — Supports PDF, TXT, and CSV.
- 🔐 **API Key Authentication** — Optional API-key based authentication.
- 🛡️ **File Validation** — File size limits, filename sanitization, and file-type validation.
- 🌐 **CORS Configuration** — Configure allowed frontend origins through environment variables.
- 🐳 **Docker Support** — Run the application inside a container.
- ⚡ **Resource Reuse** — Expensive models and external connections are initialized once per application process and reused across requests.
- 💾 **Caching** — TTL caching is used for operations where repeated network calls are unnecessary.

--

## 🏗️ Architecture

At a high level, ContextForge follows a standard RAG architecture with separate document ingestion, retrieval, generation, and API concerns.

```text
                         ┌──────────────────┐
                         │      Client      │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     FastAPI      │
                         │       API        │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
             Document Upload               User Query
                    │                           │
                    ▼                           ▼
             Load Document              Hybrid Retrieval
                    │                           │
                    ▼                    ┌──────┴──────┐
                 Chunking                │             │
                    │                    ▼             ▼
                    ▼                Vector         BM25
                Embedding             Search        Search
                    │                    │             │
                    ▼                    └──────┬──────┘
              Vector Store                     │
                                               ▼
                                         RRF Fusion
                                               │
                                               ▼
                                      Cross-Encoder
                                       Reranking
                                               │
                                               ▼
                                         Top Results
                                               │
                                               ▼
                                              LLM
                                               │
                                               ▼
                                     Answer + Sources
```

---

## 🔬 Retrieval Pipeline

ContextForge uses multiple retrieval stages instead of relying on a single vector similarity search.

For example, for a question such as:

```text
"What is the refund policy for enterprise customers?"
```

the query goes through the following pipeline:

```text
User Query
    │
    ▼
┌───────────────────────┐
│   Semantic Search     │
│                       │
│ Finds similar meaning │
│       10 results      │
└───────────┬───────────┘
            │
            │
┌───────────▼───────────┐
│     BM25 Search       │
│                       │
│ Finds exact keywords  │
│       10 results      │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│      RRF Fusion       │
│                       │
│ Combines both lists   │
│ and removes duplicates│
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Cross-Encoder         │
│ Reranking             │
│                       │
│ Scores query + chunk  │
│       together        │
└───────────┬───────────┘
            │
            ▼
      Top 3 Chunks
            │
            ▼
┌───────────────────────┐
│          LLM          │
│                       │
│ Question + Context    │
└───────────┬───────────┘
            │
            ▼
      Final Answer
      + Source Info
```

### Why Hybrid Search?

Semantic search is good at understanding the meaning of a query, but it can struggle with exact terms such as:

- Error codes
- Product IDs
- Model numbers
- Version numbers
- Contract numbers
- Technical identifiers

BM25 is better suited for exact keyword matching.

For example:

```text
Question:
"What is the refund SLA for Enterprise customers?"
```

Semantic search may find:

```text
"Enterprise clients may request a return..."
```

while BM25 may find:

```text
"Refund SLA: 30 days"
```

Using both approaches gives the retriever a better chance of finding the relevant information.

ContextForge combines the two result sets using **Reciprocal Rank Fusion (RRF)** and then uses a cross-encoder to rerank the candidates.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **API** | FastAPI + Uvicorn |
| **Language** | Python 3.11+ |
| **Embeddings** | SentenceTransformers |
| **Vector Search** | Pinecone / ChromaDB |
| **Keyword Search** | BM25 |
| **Reranking** | CrossEncoder |
| **LLM** | OpenAI / Gemini / Anthropic / Ollama |
| **HTTP Client** | httpx |
| **Containerization** | Docker |
| **Configuration** | YAML + Environment Variables |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- API key for at least one supported LLM provider

You can also use **Ollama + ChromaDB** for a fully local setup.

### Clone the Repository

```bash
git clone https://github.com/Abhi7764/context-forge.git

cd context-forge
```

### Install Dependencies

Using `uv`:

```bash
uv sync
```

Or using pip:

```bash
pip install -e .
```

### Environment Configuration

```bash
cp .env.example .env
```

Add the API keys for the providers you want to use.

### Provider Configuration

Provider selection is handled through `config.yaml`.

```yaml
llm:
  provider: gemini
  model: gemini-2.0-flash

vector_db:
  provider: chroma
```

Supported LLM providers:

```text
openai
gemini
anthropic
ollama
```

Supported vector stores:

```text
pinecone
chroma
```

For local development:

```yaml
llm:
  provider: ollama

vector_db:
  provider: chroma
```

This allows the application to run without external LLM or vector database API keys.

### Run the Application

```bash
python main.py
```

The API will be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

## 📡 API Usage

### Upload a Document

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@company_handbook.pdf"
```

The ingestion pipeline loads the document, splits it into chunks, generates embeddings, and stores the resulting data.

### Ask a Question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the leave policy for new employees?"}'
```

### Example Response

```json
{
  "answer": "New employees get 18 days of paid leave per year.",
  "sources": [
    {
      "filename": "company_handbook.pdf",
      "page": 12,
      "relevance_score": 0.97
    }
  ],
  "context_documents": []
}
```

The exact response fields may vary depending on the configured retrieval and response settings.

### Health Check

```bash
curl http://localhost:8000/health
```

### API Key Authentication

Authentication can be enabled through the `.env` file:

```env
API_KEY=your-secret-key
```

Then include the key in requests:

```bash
curl \
  -H "X-API-Key: your-secret-key" \
  http://localhost:8000/health
```

---

## ⚙️ Configuration

Application settings are managed through `config.yaml` and environment variables.

### `config.yaml`

| Setting | Example | Description |
| :--- | :--- | :--- |
| `retrieval.strategy` | `hybrid` | Selects semantic or hybrid retrieval strategy |
| `retrieval.top_k` | `3` | Number of final chunks sent to the LLM |
| `reranking.enabled` | `true` | Enable or disable reranking |
| `llm.provider` | `openai` | LLM provider |
| `vector_db.provider` | `pinecone` | Vector database |
| `chunking.chunk_size` | `500` | Controls the size of each chunk |
| `chunking.chunk_overlap` | `50` | Controls the overlap between chunks |

### Environment Variables

| Variable | Required | Description |
| :--- | :--- | :--- |
| `GOOGLE_API_KEY` | Gemini | Gemini API key |
| `OPENAI_API_KEY` | OpenAI | OpenAI API key |
| `PINECONE_API_KEY` | Pinecone | Pinecone API key |
| `ENV` | No | Application environment |
| `API_KEY` | No | Enables API-key authentication |
| `CORS_ORIGINS` | No | Allowed frontend origins |

---

## 📁 Project Structure

```text
context-forge/
│
├── main.py
├── config.yaml
├── .env.example
├── Dockerfile
├── pyproject.toml
│
├── src/
│   ├── api/
│   │   └── routes.py
│   │
│   ├── ingestion/
│   │   └── document_loader.py
│   │
│   ├── chunking/
│   │   └── chunker.py
│   │
│   ├── embeddings/
│   │   ├── base.py
│   │   ├── embedder.py
│   │   └── embedder_factory.py
│   │
│   ├── vectordb/
│   │   ├── base.py
│   │   ├── pinecone_store.py
│   │   ├── chroma_store.py
│   │   └── vector_store_factory.py
│   │
│   ├── retrieval/
│   │   ├── retriever.py
│   │   ├── bm25_search.py
│   │   ├── reranker.py
│   │   └── hybrid_retriever.py
│   │
│   ├── llm/
│   │   ├── base.py
│   │   ├── gemini.py
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   ├── ollama.py
│   │   └── llm_factory.py
│   │
│   ├── prompts/
│   │   └── ...
│   │
│   └── rag/
│       └── pipeline.py
│
└── tests/
```

The structure separates ingestion, retrieval, model providers, vector stores, and API logic so that individual components can be changed without rewriting the complete pipeline.

---

## 🐳 Deployment

### Docker

Build the image:

```bash
docker build -t context-forge .
```

Run:

```bash
docker run \
  -p 8000:8000 \
  --env-file .env \
  context-forge
```

### Production

For a multi-worker deployment:

```bash
gunicorn main:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

---

## 🧠 Engineering Decisions & Design Patterns

This project follows real software engineering principles. Here's every pattern used and exactly where it's applied:

---

### 1. Factory Pattern

> *"Don't create objects directly — let a factory decide which class to use based on configuration."*

You just change one line in `config.yaml`, and the factory creates the right object automatically. No `if-else` chains scattered across the codebase.

```
config.yaml says:              Factory creates:
─────────────────              ─────────────────
llm.provider: gemini      →   GeminiClient()
llm.provider: openai      →   OpenAIClient()
vector_db.provider: pinecone → PineconeStore()
vector_db.provider: chroma →   ChromaStore()
```

| Factory | File | Creates |
|---------|------|---------|
| `get_llm_client()` | `src/llm/llm_factory.py` | Gemini / OpenAI / Anthropic / Ollama client |
| `get_vector_store()` | `src/vectordb/vector_store_factory.py` | Pinecone / ChromaDB store |
| `get_embedder()` | `src/embeddings/embedder_factory.py` | SentenceTransformer embedder |
| `load_document()` | `src/ingestion/document_loader.py` | PDF / TXT / CSV loader |

**Why it matters:** Adding a new provider (say, Mistral) = one new file + 3 lines in the factory. Zero changes to pipeline, routes, or any other code.

---

### 2. Singleton Pattern

> *"Load expensive resources only once. Reuse them across all requests."*

AI models are 80-90 MB. Loading on every request would kill performance. Singletons load the model once into memory and reuse the same instance forever.

```python
# Without Singleton: loads 90MB model on EVERY request (❌ slow)
embedder = SentenceTransformerEmbedder()  # 2-3 seconds each time

# With Singleton: loads once, reuses forever (✅ fast)
embedder = SentenceTransformerEmbedder()  # First call: 2-3 seconds
embedder = SentenceTransformerEmbedder()  # Second call: instant (same object)
```

| Singleton | What it saves |
|-----------|--------------|
| `SentenceTransformerEmbedder` | ~90MB embedding model loaded once |
| `Reranker` | ~80MB cross-encoder loaded once |
| `GeminiClient` | HTTP/2 connection pool created once |

---

### 3. Strategy Pattern

> *"Switch between different algorithms at runtime without changing the code that uses them."*

```yaml
# Just change this one line — no code changes needed
retrieval:
  strategy: semantic    # Only vector similarity search
  strategy: hybrid      # Vector + keyword + reranking (production-grade)
```

Works the same way for LLM providers and vector stores — swap via config, not code.

---

### 4. Dependency Injection

> *"Don't create your own dependencies — receive them from outside."*

`RAGPipeline` doesn't hard-create its clients. They're injected, which makes testing dead simple:

```python
# Production: factories create real clients
pipeline = RAGPipeline()

# Testing: inject mocks — no API calls, no database
pipeline = RAGPipeline(
    llm_client=MockLLMClient(),
    vector_store=MockVectorStore(),
    embedder=MockEmbedder(),
)
```

---

### 5. Abstract Base Class (Interface Pattern)

> *"Define a contract that every implementation must follow."*

Every external service has an abstract interface. New implementations just follow the same contract:

```
BaseLLMClient (abstract)          BaseVectorStore (abstract)
├── generate_response()           ├── add_documents()
├── get_model_name()              ├── similarity_search()
│                                 ├── delete_collection()
Implementations:                  ├── count()
├── GeminiClient                  │
├── OpenAIClient                  Implementations:
├── AnthropicClient               ├── PineconeStore
├── OllamaClient                  ├── ChromaStore
└── (add your own...)             └── (add your own...)
```

---

### 6. Pipeline Pattern

> *"Break complex work into stages. Each stage does one thing and passes output to the next."*

```
Ingestion:  File → Load → Chunk → Embed → Store → Update BM25 Index

Query:      Question → Semantic Search → BM25 → RRF Merge → Rerank → LLM → Answer
```

---

### 7. Connection Pooling & Caching

> *"Don't open a new connection for every request. Reuse connections and cache expensive results."*

| Optimization | Where | What it saves |
|-------------|-------|---------------|
| **gRPC transport** | `PineconeStore` | Persistent TCP connection, no TLS handshake per query |
| **HTTP/2 pool** | `GeminiClient` via `httpx` | Reuses TCP connections across requests |
| **TTL cache (60s)** | `PineconeStore.count()` | Avoids network call on every health check |

---

### 8. Lifespan Pattern

> *"Set up resources when server starts, clean them up when it stops."*

```python
@asynccontextmanager
async def lifespan(app):
    # Startup: init pipeline, build BM25 index
    pipeline = RAGPipeline()
    pipeline.initialize_bm25()
    yield
    # Shutdown: cleanup connections
```


---

## ⚠️ Current Limitations

Current limitations include:

- **Scanned PDFs** — Image-based PDFs cannot be processed directly and require separate OCR pre-processing.
- **Access Control** — Document-level permissions are not implemented, so all uploaded documents are accessible to any query.
- **In-Memory BM25 Index** — Keyword search indexes are stored in application memory and rebuilt when the server starts.
- **Scaling Constraints** — Very large document collections with millions of chunks may require dedicated external indexing infrastructure.
- **Model Dependency** — Retrieval quality depends on the capabilities of the selected embedding and reranking models.
- **RAG Evaluation** — Automated testing tools for evaluating end-to-end response accuracy are still being expanded.

---

## 🚧 Future Improvements

Planned improvements include:

- [ ] **API rate limiting** — Implement request rate limits per IP to protect endpoints against abuse and manage LLM API costs.
- [ ] **Update existing documents** — Replace an existing document and automatically update its chunks, vector embeddings, and BM25 index.
- [ ] **OCR support for scanned PDFs** — Extract text from scanned or image-based PDFs using optical character recognition.
- [ ] **Document-level access control** — Restrict search results based on user roles and file access permissions.
- [ ] **Multi-tenant isolation** — Keep document collections and vector indexes completely separated for different teams or clients.
- [ ] **Automated RAG evaluation** — Measure answer accuracy, context relevance, and retrieval precision automatically.
- [ ] **Retrieval & LLM observability** — Track query latency, token counts, and pipeline performance metrics.
- [ ] **Distributed BM25 indexing** — Integrate external search engines to handle massive document collections across servers.
- [ ] **Streaming responses** — Stream LLM answers token-by-token to the user interface for lower latency.
- [ ] **Additional document formats** — Add support for parsing DOCX, Markdown, and HTML files.
- [ ] **Advanced document parsing** — Improve text extraction from complex PDFs containing tables and multi-column layouts.
- [ ] **Conversation history** — Support multi-turn chat memory so users can ask follow-up questions with context.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute it in your own projects.