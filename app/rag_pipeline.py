# app/rag_pipeline.py
import os
import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple, AsyncGenerator

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain.evaluation import load_evaluator
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
from opentelemetry import trace, metrics

# ────────────────────────────────────────────────────────────────────────────────
# Chat History
# ────────────────────────────────────────────────────────────────────────────────
from langchain_community.chat_message_histories import RedisChatMessageHistory, ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory


# ────────────────────────────────────────────────────────────────────────────────
# Telemetry / Traceloop
# ────────────────────────────────────────────────────────────────────────────────
# from traceloop.sdk import Traceloop
# from traceloop.sdk.instruments import Instruments
# from opentelemetry.instrumentation.milvus import MilvusInstrumentor
# from opentelemetry.instrumentation.requests import RequestsInstrumentor
# from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient
# from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
# from opentelemetry.instrumentation.threading import ThreadingInstrumentor
# from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
# from opentelemetry.instrumentation.threading import ThreadingInstrumentor
# from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

# RequestsInstrumentor().instrument()          # only covers code using 'requests'
# HTTPXClientInstrumentor().instrument()       # openai SDK / langchain_openai
# GrpcInstrumentorClient().instrument()        # pymilvus grpc client
# MilvusInstrumentor().instrument()            # Milvus-specific spans (search/insert/etc.)
# ThreadingInstrumentor().instrument()
# AsyncioInstrumentor().instrument()

# from openinference.instrumentation.langchain import LangChainInstrumentor
# # from opentelemetry.instrumentation.langchain import LangChainInstrumentor
# from opentelemetry.instrumentation.openai import OpenAIInstrumentor
# from opentelemetry.instrumentation.redis import RedisInstrumentor

# LangChainInstrumentor().instrument()
# # OpenAIInstrumentor().instrument()
# # RedisInstrumentor().instrument()

# Traceloop.init(
#     app_name=os.getenv("OTEL_SERVICE_NAME", "chat-rag"),
#     resource_attributes={"deployment.environment": os.getenv("DEPLOY_ENV", "Nick-LLM")},
#     # instruments={Instruments.LANGCHAIN, Instruments.OPENAI, Instruments.MILVUS},
#     disable_batch=True,
# )

# Meter for custom metrics
_meter = metrics.get_meter("app.scoring")
_score_hist = _meter.create_histogram(
    name="rag.score.answer_context_similarity",
    unit="1",
    description="Cosine similarity between model answer and retrieved context",
)



# ────────────────────────────────────────────────────────────────────────────────
# Models / vector DB config
# ────────────────────────────────────────────────────────────────────────────────
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
COLL = os.getenv("MILVUS_COLLECTION", "rag_chunks")
retrieval_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "100"))
rerank_TOP_K = int(os.getenv("RERANK_TOP_K", "70"))

# Lazy retriever (so Milvus init spans belong to first request trace)
_retriever = None
def get_retriever():
    global _retriever
    if _retriever is None:
        vs = Milvus(
            collection_name=COLL,
            embedding_function=emb,
            connection_args={"uri": MILVUS_URI},
            text_field="text",
            vector_field="vector",
            search_params={"metric_type":"IP","params":{"nprobe":100}},
        )
        #1)
        _retriever = vs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retrieval_TOP_K},
        ).with_config({"run_name": "MilvusSearch"})
        # ------------ side note --------------
        # other options available for search types
        #2)
        # _retriever = vs.as_retriever(
        #     search_type="mmr",
        #     search_kwargs={"k": retrieval_TOP_K,
        #                    "fetch_k": max(60, retrieval_TOP_K * 3),
        #                 #    "param": {"metric_type": "IP", "params": {"nprobe": 100}},
        #                    "lambda_mult": 1
        #                    },
        # ).with_config({"run_name": "MilvusSearch"})
        #3)
        # _retriever = vs.as_retriever(
        #     search_type="similarity_score_threshold",
        #     search_kwargs={"k": retrieval_TOP_K,
        #                    "score_threshold": 0.3
        #                    },
        # ).with_config({"run_name": "MilvusSearch"})

    return _retriever

# Make retrieval truly async for better parenting & no blocking
# async def _retrieve_async(q: str):
#     return await get_retriever().ainvoke(q)

# Re-ranker to improve answer relevance
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
    # print("pairs:", pairs)
    # print("scores:", scores)
    ranked: List[Tuple[Document, float]] = sorted(
        zip(docs, scores), key=lambda x: x[1], reverse=True
    )[:rerank_TOP_K]
    # print("ranked:", ranked)
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
# Prompt & LLMs
# ────────────────────────────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context to answer. Include references when helpful.\n\nContext:\n{context}"),
    ("placeholder", "{history}"), #injecting prior chat history
    ("human", "{question}"),
]).with_config({"run_name": "ChatPromptTemplate"})

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# Stream for UI
llm = ChatOpenAI(model=CHAT_MODEL, temperature=TEMPERATURE, stream_usage=True, streaming=True)\
    .with_config({"run_name": "ChatOpenAIChat"})

# Non-stream for /generate
_llm_block = ChatOpenAI(model=CHAT_MODEL, temperature=TEMPERATURE, streaming=False)\
    .with_config({"run_name": "ChatOpenAIChat"})

# THis line causing "initialize AsyncMilvusClient during Milvus initialization: There is no current event loop in thread 'ThreadPoolExecutor-3_0'"
# due to the runnableParallel is offload to threadpool than event loop
# retrieve_stage = RunnableParallel(
#     docs=RunnableLambda(lambda x: get_retriever().invoke(x)),  # async retrieval
#     question=RunnablePassthrough(),
# ).with_config({"run_name": "RetrieveDocs"})

retrieve_stage = (RunnableLambda(lambda x: {
        "docs": get_retriever().invoke(x),
        "question": x,
    })
).with_config({"run_name": "RetrieveDocs"})


prep_context = retrieve_stage | Rerank | FormatContext
llm_chain_stream = prompt | llm
# llm_chain_block = prompt | _llm_block

# Chat History
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_KEY_PREFIX = os.getenv("REDIS_KEY_PREFIX", "rag:msgs")    # easy to SCAN later
_MEMORY_TTL_SEC = int(os.getenv("MEMORY_TTL_SEC", "604800"))  # 7 days

_inmem_histories: dict[str, ChatMessageHistory] = {}

def _get_history(session_id: str) -> ChatMessageHistory:
    if REDIS_URL:
        return RedisChatMessageHistory(
            session_id=session_id,
            url=REDIS_URL,
            ttl=_MEMORY_TTL_SEC,
            key_prefix=REDIS_KEY_PREFIX,
        )
    # simple per-process fallback (lost on restart)
    hist = _inmem_histories.get(session_id)
    if not hist:
        hist = ChatMessageHistory()
        _inmem_histories[session_id] = hist
    return hist

# Wrapped with message history:
llm_chain_stream_with_mem = RunnableWithMessageHistory(
    llm_chain_stream,
    _get_history,
    input_messages_key="question",     # which field is user input
    history_messages_key="history"     # which prompt key receives history
)

# ────────────────────────────────────────────────────────────────────────────────
# Eval setup (LangChain native)
# ────────────────────────────────────────────────────────────────────────────────
_EVAL_MODEL = os.getenv("EVAL_MODEL", "gpt-4o-mini")
_evaluator_llm = ChatOpenAI(model=_EVAL_MODEL, temperature=0)

# 1) Answer relevance (criteria)
_answer_rel_eval = load_evaluator(
    "criteria",
    llm=_evaluator_llm,
    criteria={
        "answer_relevance": (
            "Does the answer directly and completely address the user's question "
            "without going off-topic? Give a score 0–1 and a short explanation."
        )
    },
)
# 2) Grounding in provided context
_context_rel_eval = load_evaluator("context_qa", llm=_evaluator_llm)

async def _aevaluate_strings(evaluator, **kwargs) -> dict:
    if hasattr(evaluator, "aevaluate_strings"):
        return await evaluator.aevaluate_strings(**kwargs)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: evaluator.evaluate_strings(**kwargs))

async def _run_evals(question: str, answer: str, docs: List[Document]) -> dict:
    contexts_list = [d.page_content[:4000] for d in docs]  # optional truncation
    reference_text = "\n\n".join(contexts_list)
    print("Reference text for eval:", reference_text)
    print("contexts_list:", contexts_list)
    print("question:", question)
    # Answer relevance
    try:
        ans_rel = await _aevaluate_strings(
            _answer_rel_eval,
            prediction=answer,
            input=question,
        )
    except Exception as e:
        ans_rel = {"score": None, "value": "error", "reasoning": f"{e}"}

    # Context QA – try new signature first, then fallback to old
    try:
        ctx_rel = await _aevaluate_strings(
            _context_rel_eval,
            prediction=answer,
            input=question,
            contexts=contexts_list,
        )
    except Exception:
        try:
            ctx_rel = await _aevaluate_strings(
                _context_rel_eval,
                prediction=answer,
                input=question,
                reference=reference_text,
            )
        except Exception as e2:
            ctx_rel = {"score": None, "value": "error", "reasoning": f"{e2}"}

    # def _norm(x: dict) -> dict:
    #     return {
    #         "score": x.get("score"),
    #         "label": x.get("value") or x.get("label"),
    #         "explanation": x.get("reasoning") or x.get("explanation"),
    #         "reasoning": print(x.get("reasoning"))[:1000] if x.get("reasoning") else None,
    #     }
    # return {"answer_relevance": _norm(ans_rel), "context_relevance": _norm(ctx_rel)}

    def _norm(x: dict) -> dict:
        if not isinstance(x, dict):
            return {"score": None, "label": "error", "explanation": None, "reasoning": None}
        # prefer 'reasoning', fallback to 'explanation'
        reasoning = x.get("reasoning") or x.get("explanation")
        label = x.get("value") or x.get("label")
        return {
            "score": x.get("score"),
            "label": label,
            "explanation": (x.get("explanation") or reasoning or "")[:1000] or None,
            "reasoning": (reasoning or "")[:1000] or None,
        }

def _tag_eval_span(question: str, answer: str, docs: List[Document], sim: float, ev: dict):
    span = trace.get_current_span()
    span.set_attribute("rag.eval.cosine_similarity", float(sim))
    span.set_attribute("rag.eval.num_docs", len(docs))
    span.set_attribute("rag.eval.top_sources", [str(d.metadata.get("source", "")) for d in docs[:5]])

    ar = (ev or {}).get("answer_relevance", {}) or {}
    cr = (ev or {}).get("context_relevance", {}) or {}

    span.set_attribute("rag.eval.answer_relevance.score", float(ar.get("score") if ar.get("score") is not None else -1.0))
    span.set_attribute("rag.eval.answer_relevance.label", str(ar.get("label") or "unknown"))
    if ar.get("explanation"):
        span.set_attribute("rag.eval.answer_relevance.reason", str(ar["explanation"])[:800])

    span.set_attribute("rag.eval.context_relevance.score", float(cr.get("score") if cr.get("score") is not None else -1.0))
    span.set_attribute("rag.eval.context_relevance.label", str(cr.get("label") or "unknown"))
    if cr.get("explanation"):
        span.set_attribute("rag.eval.context_relevance.reason", str(cr["explanation"])[:800])

# ────────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ────────────────────────────────────────────────────────────────────────────────
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b) / denom)

def _score_answer_vs_context(answer: str, docs: List[Document]) -> float:
    try:
        av = np.array(emb.embed_query(answer), dtype=np.float32)
        cv = np.mean([emb.embed_query(d.page_content) for d in docs], axis=0)
        return _cosine(av, np.array(cv, dtype=np.float32))
    except Exception:
        return 0.0

# ---------------- Non-streaming path ----------------
# async def generate(question: str) -> Dict[str, Any]:
#     # Prep context/docs via Runnable pipeline
#     vals = await prep_context.ainvoke(question)
#     # vals = {"question", "docs", "context"}

#     ai_msg: AIMessage = await llm_chain.ainvoke(
#         {"question": vals["question"], "context": vals["context"]}
#     )
#     answer_text = ai_msg.content or ""

async def generate(question: str) -> dict:
    vals = await prep_context.ainvoke(question)
    ai_msg = await (_llm_block | prompt).ainvoke({"question": vals["question"], "context": vals["context"]})
    answer_text = ai_msg.content or ""

    # record score metric
    sim = _score_answer_vs_context(answer_text, vals["docs"])
    try:
        _score_hist.record(sim, attributes={"k": rerank_TOP_K})
    except Exception:
        pass

    return {
        "answer": answer_text,
        "contexts": [d.page_content[:400] for d in vals["docs"]],
        "score": sim,
        # token usage & LLM attrs are captured by OpenLLMetry/OTel auto-instr
    }


# ────────────────────────────────────────────────────────────────────────────────
# Streaming path (text/plain)
# ────────────────────────────────────────────────────────────────────────────────
async def stream_generate(question: str, session_id: str = "default") -> AsyncGenerator[str, None]:
    vals = await prep_context.ainvoke(question)

    buf: List[str] = []
    # ⬇️ Below part decides to run with history or without chat history
    async for chunk in llm_chain_stream_with_mem.astream(
        {"question": vals["question"], "context": vals["context"]},
        config={"configurable": {"session_id": session_id}}
    ):
        text = chunk.content if isinstance(chunk, AIMessageChunk) else str(chunk)
        if text:
            buf.append(text)
            yield text

    full_answer = "".join(buf).strip()

    # similarity metric
    sim = _score_answer_vs_context(full_answer, vals["docs"])
    try:
        _score_hist.record(sim, attributes={"k": rerank_TOP_K})
    except Exception:
        pass

    # evals (robust to LC version)
    evals = await _run_evals(vals["question"], full_answer, vals["docs"])

    # A tracer for our custom eval span
    _otel_tracer = trace.get_tracer("app.rag.eval")
    # attach metrics to a dedicated child span so they always appear
    with _otel_tracer.start_as_current_span("rag.eval"):
        _tag_eval_span(vals["question"], full_answer, vals["docs"], sim, evals)

    # final informative line for the stream
    yield (
        f"\n\n[EVAL] cos={sim:.2f}  "
        f"ans_rel={evals['answer_relevance']['score']}  "
        f"ctx_rel={evals['context_relevance']['score']}"
    )
