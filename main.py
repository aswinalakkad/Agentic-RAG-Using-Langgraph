# ==========================================================
#                AGENTIC RAG - CLEAN VERSION
# ==========================================================

# =======================
# 1️⃣ Imports
# =======================
import os
import logging
import warnings
from pathlib import Path
from typing import Annotated, Sequence, TypedDict, Literal, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import create_retriever_tool
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


# =======================
# 2️⃣ Setup
# =======================
warnings.filterwarnings("ignore")
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =======================
# 3️⃣ FastAPI App
# =======================
app = FastAPI(title="Agentic RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =======================
# 4️⃣ Load Models Once
# =======================
LLM_MAIN = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

LLM_GRADER = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# =======================
# 5️⃣ RAG Setup
# =======================

BLOG_URL = "https://medium.com/@metafluxtech/what-is-an-ai-agent-complete-beginner-guide-with-python-2026-60ebd5085375"
CACHE_FILE = "cached_blog.txt"
VECTOR_DIR = "faiss_index"


def load_blog_content():
    if Path(CACHE_FILE).exists():
        logger.info("Loading blog from cache...")
        return Path(CACHE_FILE).read_text(encoding="utf-8")

    logger.info("Downloading blog...")
    docs = WebBaseLoader(BLOG_URL).load()
    content = docs[0].page_content

    Path(CACHE_FILE).write_text(content, encoding="utf-8")
    return content


def create_vectorstore():
    raw_text = load_blog_content()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    documents = splitter.split_documents(
        [Document(page_content=raw_text)]
    )

    if os.path.exists(f"{VECTOR_DIR}/index.faiss"):
        logger.info("Loading existing FAISS index...")
        return FAISS.load_local(
            VECTOR_DIR,
            EMBEDDINGS,
            allow_dangerous_deserialization=True
        )

    logger.info("Creating FAISS index...")
    vectorstore = FAISS.from_documents(documents, EMBEDDINGS)
    vectorstore.save_local(VECTOR_DIR)
    return vectorstore


VECTORSTORE = create_vectorstore()
RETRIEVER = VECTORSTORE.as_retriever()

RETRIEVER_TOOL = create_retriever_tool(
    RETRIEVER,
    "ai_agent_knowledge_base_search",
    "Search knowledge base about AI agents"
)

TOOLS = [RETRIEVER_TOOL]


# =======================
# 6️⃣ State Definition
# =======================
class ConversationState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    retrieved_context: Optional[str]


# =======================
# 7️⃣ Relevance Grader
# =======================
class RelevanceScore(BaseModel):
    binary_score: str = Field(description="'yes' or 'no'")


def relevance_evaluator(state: ConversationState) -> Literal["rag_generator", "query_optimizer"]:
    logger.info("Entering relevance evaluator node")

    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = PromptTemplate(
        template="""
You are a grader assessing document relevance.

Document:
{context}

Question:
{question}

Is the document relevant? Answer only 'yes' or 'no'.
""",
        input_variables=["context", "question"],
    )

    model = LLM_GRADER.with_structured_output(RelevanceScore)
    result = (prompt | model).invoke(
        {"context": context, "question": question}
    )

    if result.binary_score.lower() == "yes":
        return "rag_generator"

    return "query_optimizer"


# =======================
# 8️⃣ Intent Router Node
# =======================
def intent_router(state: ConversationState):
    logger.info("Entering intent router node")

    messages = state["messages"]

    # First message → allow tool usage
    if len(messages) == 1:
        llm_with_tools = LLM_MAIN.bind_tools(TOOLS)
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Follow-up question → direct answer
    last_question = messages[-1].content

    prompt = PromptTemplate(
        template="""
Answer the question concisely but clearly.

Question: {question}
""",
        input_variables=["question"],
    )

    response = (prompt | LLM_MAIN).invoke(
        {"question": last_question}
    )

    return {"messages": [AIMessage(content=response.content)]}


# =======================
# 9️⃣ Query Rewrite Node
# =======================
def query_optimizer(state: ConversationState):
    logger.info("Entering query optimizer node")

    question = state["messages"][0].content

    prompt = PromptTemplate(
        template="""
Improve the following question to make it clearer and more detailed:

{question}
""",
        input_variables=["question"],
    )

    improved_question = (prompt | LLM_MAIN).invoke(
        {"question": question}
    )

    return {"messages": [HumanMessage(content=improved_question.content)]}


# =======================
# 🔟 RAG Generator Node
# =======================
def rag_generator(state: ConversationState):
    logger.info("Entering RAG generation node")

    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"],
    )

    response = (prompt | LLM_MAIN | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )

    return {"messages": [AIMessage(content=response)]}


# =======================
# 1️⃣1️⃣ Build Graph
# =======================
workflow = StateGraph(ConversationState)

workflow.add_node("intent_router", intent_router)
workflow.add_node("vector_retriever", ToolNode([RETRIEVER_TOOL]))
workflow.add_node("relevance_evaluator", relevance_evaluator)
workflow.add_node("query_optimizer", query_optimizer)
workflow.add_node("rag_generator", rag_generator)

workflow.add_edge(START, "intent_router")

workflow.add_conditional_edges(
    "intent_router",
    tools_condition,
    {"tools": "vector_retriever", END: END},
)

workflow.add_conditional_edges(
    "vector_retriever",
    relevance_evaluator,
)

workflow.add_edge("rag_generator", END)
workflow.add_edge("query_optimizer", "intent_router")

GRAPH = workflow.compile()


# =======================
# 1️⃣2️⃣ API Schema
# =======================
class Query(BaseModel):
    message: str


# =======================
# 1️⃣3️⃣ API Endpoints
# =======================
@app.get("/")
async def root():
    return {"message": "Agentic RAG API running 🚀"}


@app.post("/chat")
async def chat_endpoint(query: Query):
    try:
        logger.info("Starting graph execution")

        events = GRAPH.stream(
            {"messages": [HumanMessage(content=query.message)]},
            stream_mode="values",
        )

        final_response = ""
        for event in events:
            final_response = event["messages"][-1].content

        return JSONResponse(content={"response": final_response})

    except Exception as e:
        logger.error(str(e))
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )