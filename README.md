# 🤖 Agentic RAG API

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-latest-4B8BBE?style=flat)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3-F55036?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

A production-ready **Agentic Retrieval-Augmented Generation (RAG)** system built with LangGraph, FastAPI, and Groq. The agent intelligently routes queries, retrieves relevant documents, grades their relevance, and rewrites queries when needed — all in a single, streamlined graph.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Intent Router ──── (no tool needed) ──────────────────► Direct Answer (END)
    │
    │ (tool call)
    ▼
Vector Retriever (FAISS)
    │
    ▼
Relevance Evaluator
    │                    │
    ▼ (relevant)         ▼ (not relevant)
RAG Generator       Query Optimizer
    │                    │
    ▼                    └──► Intent Router (retry)
  END
```

**Key nodes:**
- **Intent Router** — decides whether to use RAG or answer directly
- **Vector Retriever** — performs semantic search over the FAISS index
- **Relevance Evaluator** — grades whether retrieved docs match the query
- **Query Optimizer** — rewrites unclear queries and retries retrieval
- **RAG Generator** — synthesizes a grounded answer from context

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com/)

### Installation

```bash
git clone <your-repo-url>
cd agentic-rag

pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Run the API

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

---

## 📡 API Endpoints

### `GET /`
Health check.

**Response:**
```json
{ "message": "Agentic RAG API running 🚀" }
```

---

### `POST /chat`
Send a question to the RAG agent.

**Request body:**
```json
{ "message": "What is an AI agent?" }
```

**Response:**
```json
{ "response": "An AI agent is a system that perceives its environment and takes actions..." }
```

---

## 🧱 Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI |
| Agent Orchestration | LangGraph |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS |
| Document Loader | LangChain WebBaseLoader |

---

## 📁 Project Structure

```
.
├── main.py               # Application entry point
├── cached_blog.txt       # Cached blog content (auto-generated)
├── faiss_index/          # Persisted FAISS vector index (auto-generated)
├── .env                  # Environment variables
└── requirements.txt      # Python dependencies
```

---

## ⚙️ Configuration

| Variable | Location | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | `.env` | Groq API authentication key |
| `BLOG_URL` | `main.py` | Source URL for the knowledge base |
| `chunk_size` | `main.py` | Text splitter chunk size (default: 300) |
| `chunk_overlap` | `main.py` | Chunk overlap (default: 50) |

### Swapping the Knowledge Base

To use a different data source, update `BLOG_URL` in `main.py` and delete `cached_blog.txt` and `faiss_index/` to trigger a fresh index build.

---

## 📦 Requirements

```
fastapi
uvicorn
python-dotenv
langgraph
langchain-core
langchain-community
langchain-text-splitters
langchain-huggingface
langchain-groq
faiss-cpu
pydantic
```

---

## 📝 Notes

- The blog content and FAISS index are cached locally on first run to avoid redundant downloads and embedding computation.
- The relevance grader uses structured output with Pydantic to enforce binary `yes/no` scoring.
- Follow-up questions (multi-turn) bypass the retriever and are answered directly by the LLM.

---

## 🙈 .gitignore

Make sure to add the following to your `.gitignore` before pushing:

```
.env
cached_blog.txt
faiss_index/
__pycache__/
*.pyc
.DS_Store
```

---

## 🤝 Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
