# 🤖 Agentic RAG with LangGraph + FastAPI

An intelligent **Retrieval-Augmented Generation (RAG)** system built with LangGraph, LangChain, FastAPI, and Groq LLMs. This agent dynamically decides whether to retrieve external knowledge, rewrite ambiguous questions, or answer directly — all through a structured graph-based workflow.

---

## 🧠 How It Works

The system uses a **LangGraph state machine** to route each query through the most appropriate pipeline:

```
START → Agent → [Retrieve / End]
                   ↓
              Grade Documents
               ↙         ↘
          Generate       Rewrite → Agent
              ↓
             END
```

## 🗺️ Graph Visualization

![Agentic RAG Graph](visualization.png)


1. **Agent Node** — Decides whether to use the retriever tool or answer directly.
2. **Retrieve Node** — Fetches relevant chunks from the FAISS vector store.
3. **Grade Documents Node** — Assesses whether retrieved docs are relevant to the question.
4. **Generate Node** — Produces a final answer using the retrieved context.
5. **Rewrite Node** — Reformulates unclear questions for better retrieval.

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| LLM | Groq (`llama-3.3-70b-versatile`, `Gemma2-9b-It`) |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS |
| Orchestration | LangGraph |
| API Framework | FastAPI |
| Document Loading | LangChain WebBaseLoader |

---

## 📁 Project Structure

```
├── main.py                  # FastAPI app + LangGraph pipeline
├── cached_blog.txt          # Cached document content (auto-generated)
├── faiss_index/             # Persisted FAISS vector store (auto-generated)
├── visualization.png        # LangGraph DAG visualization (auto-generated)
├── .env                     # Environment variables
└── requirements.txt         # Python dependencies
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at [console.groq.com](https://console.groq.com).

### 4. Run the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

---

## 📡 API Endpoints

### `GET /`
Health check — confirms the API is running.

**Response:**
```json
{ "message": "Agentic RAG API is running 🚀" }
```

### `POST /chat`
Send a message to the RAG agent.

**Request Body:**
```json
{ "message": "What is an AI agent?" }
```

**Response:**
```json
{ "response": "An AI agent is a system that perceives its environment and takes actions..." }
```

---

## 📦 Requirements

Create a `requirements.txt` with:

```
fastapi
uvicorn
python-dotenv
langchain
langchain-community
langchain-groq
langchain-huggingface
langchain-text-splitters
langgraph
faiss-cpu
sentence-transformers
pillow
pydantic
```

---

## 🔍 Knowledge Base

By default, the agent is grounded in this article:

> [What is an AI Agent? Complete Beginner Guide with Python (2026)](https://medium.com/@metafluxtech/what-is-an-ai-agent-complete-beginner-guide-with-python-2026-60ebd5085375)

To use your own document, replace the URL in `WebBaseLoader(...)` inside `main.py` and delete the `cached_blog.txt` and `faiss_index/` directory to force a rebuild.

---

## 🗺️ Graph Visualization

On first run, the app automatically generates a `visualization.png` of the LangGraph DAG so you can see the full agent flow.

---

## 📄 License

MIT License. Feel free to use, modify, and distribute.

