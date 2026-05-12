# app/rag_pipeline.py
import os
from dotenv import load_dotenv
load_dotenv()
from typing import List, Tuple, AsyncGenerator

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
from opentelemetry import trace, context as otel_context
from opentelemetry.util.genai.types import (
    Workflow, AgentInvocation, RetrievalInvocation,
    InputMessage, OutputMessage, Text, Step,
)
from opentelemetry.util.genai.handler import get_telemetry_handler

_genai_handler = get_telemetry_handler()

from langchain_community.chat_message_histories import RedisChatMessageHistory, ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

# ────────────────────────────────────────────────────────────────────────────────
# Models / vector DB config
# ────────────────────────────────────────────────────────────────────────────────
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLL = os.getenv("MILVUS_COLLECTION", "rag_chunks")
retrieval_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "100"))
rerank_TOP_K = int(os.getenv("RERANK_TOP_K", "70"))

_milvus_client = None

def _get_milvus_client():
    global _milvus_client
    if _milvus_client is None:
        from pymilvus import MilvusClient
        _milvus_client = MilvusClient(uri=MILVUS_URI)
    return _milvus_client

def _milvus_search(question: str, k: int = retrieval_TOP_K) -> List[Document]:
    vec = emb.embed_query(question)
    client = _get_milvus_client()
    results = client.search(
        collection_name=COLL,
        data=[vec],
        limit=k,
        output_fields=["text", "source"],
        search_params={"metric_type": "IP", "params": {"nprobe": 100}},
    )[0]
    return [
        Document(
            page_content=hit["entity"].get("text", ""),
            metadata={"source": hit["entity"].get("source", "")},
        )
        for hit in results
    ]

try:
    _cross = CrossEncoder("BAAI/bge-reranker-v2-m3")
except Exception:
    _cross = None

def _rerank_impl(question: str, docs: List[Document]) -> List[Document]:
    if not docs:
        return []
    if not _cross:
        return docs[:rerank_TOP_K]
    pairs = [[question, d.page_content] for d in docs]
    scores = _cross.predict(pairs)
    ranked: List[Tuple[Document, float]] = sorted(
        zip(docs, scores), key=lambda x: x[1], reverse=True
    )[:rerank_TOP_K]
    return [d for d, _ in ranked]

Rerank = RunnableLambda(
    lambda x: {"question": x["question"], "docs": _rerank_impl(x["question"], x["docs"])}
).with_config({"run_name": "bge-reranker"})

def _format_ctx(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)

FormatContext = RunnableLambda(
    lambda x: {"question": x["question"], "docs": x["docs"], "context": _format_ctx(x["docs"])}
).with_config({"run_name": "FormatContext"})

# ────────────────────────────────────────────────────────────────────────────────
# Prompt & LLM
# ────────────────────────────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a document Q&A assistant. "
     "Answer questions using the context below or the conversation history. "
     "If a question is a follow-up to something already discussed, use the conversation history to answer. "
     "Only block questions that are completely unrelated to the documents and have no connection to the conversation history — "
     "for those, respond with exactly: 'I can only answer questions related to the documents in my knowledge base.' "
     "Always follow the user's instructions carefully.\n\n"
     "Context:\n{context}"),
    ("placeholder", "{history}"),
    ("human", "{question}"),
]).with_config({"run_name": "ChatPromptTemplate"})

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

llm = ChatOpenAI(model=CHAT_MODEL, temperature=TEMPERATURE, stream_usage=True, streaming=True)\
    .with_config({"run_name": "ChatOpenAIChat"})

retrieve_stage = RunnableLambda(lambda x: {
    "docs": _milvus_search(x),
    "question": x,
}).with_config({"run_name": "RetrieveDocs"})

llm_chain_stream = prompt | llm

# ────────────────────────────────────────────────────────────────────────────────
# Chat history (Redis, 7-day TTL)
# ────────────────────────────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_KEY_PREFIX = os.getenv("REDIS_KEY_PREFIX", "rag:msgs")
_MEMORY_TTL_SEC = int(os.getenv("MEMORY_TTL_SEC", "604800"))

_inmem_histories: dict[str, ChatMessageHistory] = {}

def _get_history(session_id: str) -> ChatMessageHistory:
    if REDIS_URL:
        return RedisChatMessageHistory(
            session_id=session_id,
            url=REDIS_URL,
            ttl=_MEMORY_TTL_SEC,
            key_prefix=REDIS_KEY_PREFIX,
        )
    hist = _inmem_histories.get(session_id)
    if not hist:
        hist = ChatMessageHistory()
        _inmem_histories[session_id] = hist
    return hist

llm_chain_stream_with_mem = RunnableWithMessageHistory(
    llm_chain_stream,
    _get_history,
    input_messages_key="question",
    history_messages_key="history"
)

# ────────────────────────────────────────────────────────────────────────────────
# Streaming entry point
# ────────────────────────────────────────────────────────────────────────────────
async def stream_generate(question: str, session_id: str = "default") -> AsyncGenerator[str, None]:
    workflow = Workflow(
        name="rag-pipeline",
        workflow_type="rag",
        input_messages=[InputMessage(role="user", parts=[Text(content=question)])],
        conversation_id=session_id,
        agent_name="rag-pipeline",
    )
    _genai_handler.start_workflow(workflow)

    # SDK skips context.attach() in async contexts — attach manually so auto-instrumented
    # spans (openai.chat, milvus) nest inside the workflow span.
    _wf_token = None
    try:
        if getattr(workflow, "span", None) is not None:
            _wf_token = otel_context.attach(trace.set_span_in_context(workflow.span))

        # 1. Milvus retrieval
        retrieval = RetrievalInvocation(
            retriever_type="milvus",
            query=question,
            top_k=retrieval_TOP_K,
        )
        _genai_handler.start_retrieval(retrieval)
        _ret_token = None
        if getattr(retrieval, "span", None) is not None:
            _ret_token = otel_context.attach(trace.set_span_in_context(retrieval.span))
        try:
            raw = await retrieve_stage.ainvoke(question)
            retrieval.documents_retrieved = len(raw["docs"])
        finally:
            if _ret_token is not None:
                otel_context.detach(_ret_token)
            _genai_handler.stop_retrieval(retrieval)

        # 2. Reranking
        rerank_step = Step(name="reranking", step_type="execution",
                           objective=f"Cross-encoder rerank top {rerank_TOP_K}")
        _genai_handler.start_step(rerank_step)
        try:
            reranked = await Rerank.ainvoke(raw)
        finally:
            _genai_handler.stop_step(rerank_step)

        # 3. Prompt augmentation
        aug_step = Step(name="prompt-augmentation", step_type="execution",
                        objective="Format retrieved context into prompt")
        _genai_handler.start_step(aug_step)
        try:
            vals = await FormatContext.ainvoke(reranked)
        finally:
            _genai_handler.stop_step(aug_step)

        # 4. LLM call wrapped in AgentInvocation
        agent = AgentInvocation(
            name="rag-pipeline",
            agent_type="rag",
            provider="openai",
            conversation_id=session_id,
            input_messages=[InputMessage(role="user", parts=[Text(content=question)])],
        )
        _genai_handler.start_agent(agent)
        _agent_token = None
        if getattr(agent, "span", None) is not None:
            _agent_token = otel_context.attach(trace.set_span_in_context(agent.span))
        buf: list[str] = []
        try:
            async for chunk in llm_chain_stream_with_mem.astream(
                {"question": vals["question"], "context": vals["context"]},
                config={"configurable": {"session_id": session_id}}
            ):
                text = chunk.content if isinstance(chunk, AIMessageChunk) else str(chunk)
                if text:
                    buf.append(text)
                    yield text
            agent.output_messages.append(
                OutputMessage(role="assistant", parts=[Text(content="".join(buf))])
            )
        finally:
            if _agent_token is not None:
                otel_context.detach(_agent_token)
            _genai_handler.stop_agent(agent)

    finally:
        if _wf_token is not None:
            otel_context.detach(_wf_token)
        _genai_handler.stop_workflow(workflow)
