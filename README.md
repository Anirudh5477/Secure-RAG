<div align="center">

#  Secure-RAG
### Enterprise-Grade, Role-Isolated AI Knowledge Assistant

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Azure%20App%20Service-0078d4?style=for-the-badge&logo=microsoftazure&logoColor=white)](https://secure-rag-chatbot-f6cneef5hrbmgpa0.centralindia-01.azurewebsites.net)
[![Python](https://img.shields.io/badge/Python-3.10-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.3-1c3c3c?style=for-the-badge)](https://langchain.com)
[![Groq](https://img.shields.io/badge/LLM-Llama%203.3%2070B-f55036?style=for-the-badge)](https://groq.com)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant%20Cloud-dc244c?style=for-the-badge)](https://qdrant.tech)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ed?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

> **Secure-RAG** is not a chatbot that answers questions from documents.
> It is an enterprise-grade AI system that enforces **who is allowed to even ask those questions.**

</div>

---

##  Live Demo

**URL:** https://secure-rag-chatbot-f6cneef5hrbmgpa0.centralindia-01.azurewebsites.net

Use the following credentials to explore role isolation in action:

| Role | Employee ID | Password |
|------|-------------|----------|
| 🔵 Engineering | `EMP-677091` | `GZvNP5GOurp` |
| 🟢 HR | `EMP-133065` | `9a6Q7GO4od3d` |
| 🟡 Marketing | `EMP-218289` | `7K1ty2h@` |
| 🔴 Finance | `EMP-298449` | `0Wyd1ZMt1` |

> **Try this:** Log in as Engineering and ask a question. Log out, log in as HR and ask the same question. You will get completely different results — even though they hit the exact same vector database.

---

##  The Problem This Solves

Most RAG chatbot tutorials build a system that is dangerously naive:

```
User asks question → Search ALL documents → LLM answers → Done
```

In a real enterprise, this is a critical security failure. An intern in Marketing should never retrieve salary data from the HR knowledge base. A contractor should not surface confidential engineering roadmaps.

**Secure-RAG solves this with a multi-layered defense architecture — not a single gate.**

---

##  System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      SECURE-RAG PIPELINE                       │
│                                                                │
│  User ──► [1. Auth Gate] ──► [2. Intent Guardrail] ─► BLOCK  │
│                                        │                       │
│                                      SAFE                      │
│                                        │                       │
│                               [3. PII Scrubber]                │
│                                        │                       │
│                          [4. Role-Scoped Cache Check]          │
│                              │               │                 │
│                           HIT ──► Response  MISS              │
│                                              │                 │
│                          [5. RBAC Vector Search]               │
│                           (Qdrant + dept filter)               │
│                                        │                       │
│                          [6. LLM Synthesis]                    │
│                           (Llama 3.3 70B via Groq)             │
│                                        │                       │
│                          [7. Source Citation & Cache Write]    │
│                                        │                       │
│                          [8. LangSmith Telemetry] ──► Response │
└────────────────────────────────────────────────────────────────┘
```

Every single request passes through **8 sequential stages** before an answer is returned.

---

##  Core Features & Engineering Challenges

### 🔒 1. Role-Based Access Control (RBAC) — The Central Challenge

This is the most complex and critical component. Standard RAG has no concept of "who is asking." Secure-RAG enforces role isolation at the **database query level**, not the application level.

**How it works:**

1. **Ingestion Phase:** Documents loaded from the `data/` folder are tagged with a `department` metadata field matching their subdirectory (`data/hr/`, `data/engineering/`, etc.) before being upserted into Qdrant.
2. **Query Phase:** The authenticated user's role is used to construct a Qdrant `FieldCondition` filter before any search executes.
3. **Index-Level Enforcement:** The filter is passed directly to the Qdrant HNSW index — the vector engine **physically cannot return documents from other departments**. This is not a post-retrieval filter that can be bypassed by clever prompting.

```python
rbac_filter = qmodels.Filter(
    must=[
        qmodels.FieldCondition(
            key="metadata.department",
            match=qmodels.MatchValue(value=user_role)  # from session, not user input
        )
    ]
)
results = vector_store.similarity_search(query, k=6, filter=rbac_filter)
```

**Why this is hard:** LangChain's `MarkdownHeaderTextSplitter` and `RecursiveCharacterTextSplitter` silently drop custom metadata when rechunking documents. The ingestion pipeline explicitly re-attaches `department` and `file_name` tags to every single resulting chunk before upload — without this, RBAC silently breaks.

---

###  2. Intent Guardrail — LLM-as-a-Judge Security Router

Before a query ever touches the knowledge base, a separate LLM call evaluates it as a security router. This prevents prompt injection, off-topic abuse, and compliance violations.

The guardrail handles tricky edge cases explicitly:
- Simple greetings → `SAFE`, routed to a friendly response (not a document search)
- Marketing/advertising questions → explicitly `SAFE` (often misclassified as political)
- Code generation, personal advice, politics → `BLOCK`

The decision is parsed by checking for the **substring** `"SAFE"` rather than an exact match, making it robust to LLM formatting variability.

---

###  3. PII Scrubbing — Pre-Query Anonymization

After guardrail approval, the query passes through Microsoft **Presidio** (`presidio_analyzer` + `presidio_anonymizer`) powered by the `en_core_web_lg` spaCy NER model. Names, emails, phone numbers, and IDs are redacted before the query reaches the LLM.

This ensures the system cannot be weaponized as a PII exfiltration tool even by authorized employees.

---

###  4. Role-Scoped Semantic LRU Cache

A naive cache would let Employee A receive Employee B's answer if they asked a similar question. Secure-RAG implements a custom **LRU semantic cache** scoped by department.

The cache key is a combination of:
- A **semantic embedding** of the query (typos and paraphrases still hit the cache)
- The user's **department role**

The same question from HR and Engineering generates separate cache entries with role-appropriate answers. The LRU eviction policy prevents unbounded memory growth in production.

---

###  5. Markdown-Aware Two-Stage Ingestion

The ingestion pipeline (`ingest.py`) uses a two-stage splitting strategy:

1. **`MarkdownHeaderTextSplitter`** — Splits on `#`, `##`, `###` headers first, keeping related sections logically grouped
2. **`RecursiveCharacterTextSplitter`** — Secondary split with `chunk_size=1500`, `chunk_overlap=200`

The Qdrant collection is **dropped and recreated** on each ingestion run, preventing stale data and embedding drift from contaminating the knowledge base.

---

###  6. LangSmith Observability & Synthetic Cost Tracking

Every LLM call is traced to the **LangSmith cloud dashboard**. Per-request token usage is extracted from Groq response metadata and used to calculate synthetic cost:

- Input: ~$0.59 / 1M tokens
- Output: ~$0.79 / 1M tokens

This enables per-query cost tracking, latency monitoring, and conversation-level debugging in production.

---

##  Project Structure

```
Secure-RAG/
├── .github/workflows/   # Azure App Service build & deployment CI
├── Dockerfile           # Production container definition
├── Guardrails.py        # Intent guardrail & PII scrubbing middleware
├── LICENSE
├── README.md
├── app.py               # Streamlit frontend — auth, chat UI, session management
├── cache_layer.py       # Role-scoped semantic LRU cache
├── employee_data (1).csv # Employee credentials / role mapping
├── ingest.py            # Markdown-aware two-stage ingestion pipeline
├── requirements.txt     # Cross-platform Python dependencies
├── retrieval.py         # Core RAG engine — RBAC search, LLM synthesis, telemetry
└── data/                # Department knowledge bases (gitignored)
    ├── engineering/
    ├── hr/
    ├── finance/
    ├── marketing/
    └── general/
```

---

##  Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM** | Llama 3.3 70B (Groq) | Answer generation & guardrail classification |
| **Embeddings** | FastEmbed | Document and query vectorization |
| **Vector DB** | Qdrant Cloud | RBAC-enforced similarity search |
| **Orchestration** | LangChain | Chain composition & document handling |
| **PII Protection** | Microsoft Presidio + spaCy | Pre-query anonymization |
| **Observability** | LangSmith | Tracing, latency, cost tracking |
| **Frontend** | Streamlit | Chat UI with glassmorphism dark theme |
| **Deployment** | Azure App Service + Docker | Production containerized hosting |

---

##  Local Setup

### Prerequisites
- Python 3.10+
- [Groq API key](https://console.groq.com) (free tier)
- [Qdrant Cloud](https://qdrant.tech) cluster (free tier)
- [LangSmith account](https://smith.langchain.com) (optional, for tracing)

### 1. Clone and Install

```bash
git clone https://github.com/your-username/Secure-RAG.git
cd Secure-RAG
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
QDRANT_URL="your-qdrant-cluster-url"
QDRANT_API_KEY="your-qdrant-api-key"
GROQ_API_KEY="your-groq-api-key"
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY="your-langsmith-api-key"
LANGCHAIN_PROJECT="Secure-RAG"
```

### 3. Ingest Documents

Place department documents (`.md`, `.txt`, `.pdf`) into `data/<department>/`:

```bash
python ingest.py
```

### 4. Run

```bash
streamlit run app.py
```

---

##  Docker Deployment

```bash
# Build the image
docker build -t secure-rag .

# Run with injected secrets (never bake keys into the image)
docker run -p 8501:8501 \
  -e GROQ_API_KEY="..." \
  -e QDRANT_URL="..." \
  -e QDRANT_API_KEY="..." \
  secure-rag
```

---

##  Key Design Decisions

| Decision | Why |
|----------|-----|
| RBAC filter at **Qdrant index level** | Cannot be bypassed by prompt injection; the LLM never sees forbidden data |
| **Separate guardrail LLM call** before retrieval | Prevents wasted vector search compute on blocked queries |
| **Semantic cache scoped by role** | Prevents cross-department cache pollution while reducing API costs |
| **Metadata re-attached after every split** | LangChain splitters silently drop metadata; explicit re-attachment is critical for RBAC |
| **Qdrant collection drop-and-recreate** on ingest | Eliminates stale data and embedding drift from partial re-ingests |
| **`python:3.10-slim` Docker base** | Minimal attack surface and fast build times |

##  Future Roadmap

- [ ] **Multi-modal support** — Ingest PDFs with embedded charts and tables
- [ ] **Admin dashboard** — Real-time usage analytics per department and employee
- [ ] **Immutable audit log** — Compliance-grade record of every query and document retrieved
- [ ] **Dynamic role hierarchy** — Nested roles (e.g., `senior-engineering` inherits `engineering`)
- [ ] **Document-level ACL** — Per-file access control beyond department isolation

---

<div align="center">

**Not just a chatbot. An enterprise AI access control system.**

*Built to demonstrate what production-grade, security-first RAG architecture looks like.*

</div>
