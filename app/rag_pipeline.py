# app/rag_pipeline.py
import os
from dotenv import load_dotenv
load_dotenv()
from typing import List, Tuple, AsyncGenerator

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
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

from galileo import GalileoLogger
from agent_control import control, ControlViolationError, ControlSteerError

_galileo: GalileoLogger | None = None
_galileo_sessions: set[str] = set()  # tracks which session_ids have start_session() called

def _get_galileo() -> GalileoLogger | None:
    """Initialize GalileoLogger using project/log-stream IDs (like golden demo)."""
    global _galileo
    if _galileo is not None:
        return _galileo
    import logging
    _log = logging.getLogger(__name__)
    try:
        project_name    = os.getenv("GALILEO_PROJECT", "nkim-chatbot")
        log_stream_name = os.getenv("GALILEO_LOG_STREAM", "production")

        # Resolve IDs first (golden demo pattern — avoids accidental project creation)
        from galileo.projects import Projects
        from galileo.log_streams import LogStreams
        project    = Projects().get_with_env_fallbacks(name=project_name)
        project_id = str(project.id) if project else None

        log_stream    = LogStreams().get(name=log_stream_name, project_name=project_name) if project_id else None
        log_stream_id = str(log_stream.id) if log_stream else None

        if project_id and log_stream_id:
            _galileo = GalileoLogger(project_id=project_id, log_stream_id=log_stream_id)
        elif project_id:
            _galileo = GalileoLogger(project_id=project_id, log_stream=log_stream_name)
        else:
            _galileo = GalileoLogger(project=project_name, log_stream=log_stream_name)

        # Wire Agent Control bridge (golden demo calls this immediately after logger creation)
        try:
            _galileo.enable_agent_control()
        except Exception:
            pass  # requires agent-control-sdk; skip if not installed

    except Exception as e:
        _log.warning("Galileo init failed: %s", e)
    return _galileo


def _ensure_session(g: GalileoLogger, session_id: str) -> None:
    """Start a Galileo session for this session_id if not already started (golden demo pattern)."""
    if session_id in _galileo_sessions:
        return
    try:
        g.start_session(name=f"session-{session_id}", external_id=session_id)
        _galileo_sessions.add(session_id)
    except Exception:
        pass

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
# Single source of truth: the chain renders this, and the Galileo LLM span logs the
# same rendered text — so what Galileo scores is what OpenAI actually received.
SYSTEM_PROMPT = (
    "You are a document Q&A assistant. "
    "Answer questions using the context below or the conversation history. "
    "If a question is a follow-up to something already discussed, use the conversation history to answer. "
    "Only block questions that are completely unrelated to the documents and have no connection to the conversation history — "
    "for those, respond with exactly: 'I can only answer questions related to the documents in my knowledge base.' "
    "Always follow the user's instructions carefully.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
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

@control(step_name="retrieval_step")
async def _controlled_retrieve(question: str) -> dict:
    """Milvus retrieval wrapped with Agent Control PRE/POST guardrails (golden demo pattern)."""
    return await retrieve_stage.ainvoke(question)


def _capture_usage(result, sink: dict | None) -> None:
    """Copy LangChain's usage_metadata into `sink` so the Galileo LLM span can report tokens."""
    if sink is None:
        return
    um = getattr(result, "usage_metadata", None) or {}
    if um:
        sink["input_tokens"] = um.get("input_tokens")
        sink["output_tokens"] = um.get("output_tokens")
        sink["total_tokens"] = um.get("total_tokens")


@control(step_name="rag-llm-step")
async def _controlled_llm_invoke(full_prompt: str, session_id: str, _question: str, _context: str,
                                 _usage_sink: dict | None = None) -> str:
    """LLM call with Agent Control PRE (input) + POST (output) guardrails.

    `full_prompt` is the complete assembled prompt (system context + user question) so
    Agent Control's PRE evaluation sees the full text including any injected content
    in the retrieved documents — enabling indirect prompt injection detection.

    `_question` and `_context` are passed separately so the LLM chain can use them
    as distinct template variables; they are prefixed with _ to signal they are not
    the primary evaluation target. `_usage_sink` is an out-param mutated with token counts.
    """
    from langchain_core.messages import AIMessage
    result = await llm_chain_stream_with_mem.ainvoke(
        {"question": _question, "context": _context},
        config={"configurable": {"session_id": session_id}},
    )
    _capture_usage(result, _usage_sink)
    return result.content if isinstance(result, AIMessage) else str(result)


def _ac_infra_error(e: Exception) -> bool:
    """True when Agent Control server is unreachable/misconfigured (not an actual control violation)."""
    msg = str(e).lower()
    return "failed unexpectedly" in msg or "404" in msg or "503" in msg or "connection" in msg


async def _run_retrieve(question: str) -> dict:
    """Retrieval with Agent Control, falling back to direct call if service unavailable."""
    try:
        return await _controlled_retrieve(question)
    except (ControlViolationError, ControlSteerError) as e:
        if _ac_infra_error(e):
            import logging
            logging.getLogger(__name__).warning("Agent Control unavailable for retrieval, falling back: %s", e)
            return await retrieve_stage.ainvoke(question)
        raise
    except Exception as e:
        if _ac_infra_error(e):
            import logging
            logging.getLogger(__name__).warning("Agent Control unavailable for retrieval, falling back: %s", e)
            return await retrieve_stage.ainvoke(question)
        raise


async def _run_llm(question: str, context: str, session_id: str, usage_sink: dict | None = None) -> str:
    """LLM with Agent Control, falling back to direct ainvoke if service unavailable.

    Passes `full_prompt` as the first arg so Agent Control PRE evaluation analyzes
    the complete input (context + question), catching indirect prompt injection from
    retrieved documents — not just the user's (potentially clean) question.

    `usage_sink` is mutated in place with the OpenAI token counts.
    """
    # Deliberately a MINIMAL framing, not SYSTEM_PROMPT: this string is what Agent
    # Control PRE-evaluates. SYSTEM_PROMPT's meta-instructions ("Always follow the
    # user's instructions carefully", "Only block questions that...") score as a
    # prompt injection themselves and steer every request. Keep the framing plain so
    # the detector judges the context + question, which is the point of the check.
    full_prompt = (
        "You are a document Q&A assistant. Answer using the context below.\n\n"
        f"Context:\n{context}\n\nUser: {question}"
    )

    async def _direct() -> str:
        from langchain_core.messages import AIMessage
        result = await llm_chain_stream_with_mem.ainvoke(
            {"question": question, "context": context},
            config={"configurable": {"session_id": session_id}},
        )
        _capture_usage(result, usage_sink)
        return result.content if isinstance(result, AIMessage) else str(result)

    try:
        return await _controlled_llm_invoke(full_prompt, session_id, question, context, usage_sink)
    except Exception as e:  # ControlViolationError/ControlSteerError both subclass Exception
        if _ac_infra_error(e):
            import logging
            logging.getLogger(__name__).warning("Agent Control unavailable for LLM, falling back: %s", e)
            return await _direct()
        raise

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
    import time as _time
    g = _get_galileo()
    if g:
        _ensure_session(g, session_id)
        g.start_trace(input=question, name="rag-pipeline", metadata={"session_id": session_id})
        # Push a workflow span so every span below nests under it instead of sitting
        # flat on the trace. Popped by conclude(conclude_all=True) in the finally block.
        g.add_workflow_span(
            input=question,
            name="rag-pipeline",
            metadata={"session_id": session_id, "retrieval_top_k": str(retrieval_TOP_K),
                      "rerank_top_k": str(rerank_TOP_K)},
        )

    workflow = Workflow(
        name="rag-pipeline",
        workflow_type="rag",
        input_messages=[InputMessage(role="user", parts=[Text(content=question)])],
        conversation_id=session_id,
        agent_name="rag-pipeline",
    )
    _genai_handler.start_workflow(workflow)
    buf: list[str] = []
    _retrieved_context: str = ""
    _llm_start_ns: int = 0
    _usage: dict = {}          # filled in by _run_llm with real OpenAI token counts
    _history_msgs: list = []   # chat history actually rendered into the prompt

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
        _t0 = _time.monotonic_ns()
        try:
            raw = await _run_retrieve(question)
            retrieval.documents_retrieved = len(raw["docs"])
        except (ControlViolationError, ControlSteerError) as e:
            blocked_msg = f"[Blocked] {e}" if isinstance(e, ControlViolationError) else f"[Steered] {e}"
            if g:
                # conclude_all pops both the workflow span and the trace.
                g.conclude(output=blocked_msg, conclude_all=True)
                await g.async_flush()
            yield blocked_msg
            return
        finally:
            if _ret_token is not None:
                otel_context.detach(_ret_token)
            _genai_handler.stop_retrieval(retrieval)

        # Galileo retriever span — the RAW Milvus candidates, before reranking.
        # Logged separately from the reranked set so the rerank stage's effect is visible.
        if g:
            g.add_retriever_span(
                input=question,
                output=[d.page_content for d in raw["docs"]],
                name="milvus-raw",
                step_number=1,
                duration_ns=_time.monotonic_ns() - _t0,
                metadata={"stage": "retrieval", "top_k": str(retrieval_TOP_K),
                          "num_retrieved": str(len(raw["docs"]))},
            )

        # 2. Reranking
        rerank_step = Step(name="reranking", step_type="execution",
                           objective=f"Cross-encoder rerank top {rerank_TOP_K}")
        _genai_handler.start_step(rerank_step)
        _t0 = _time.monotonic_ns()
        try:
            reranked = await Rerank.ainvoke(raw)
        finally:
            _genai_handler.stop_step(rerank_step)
        _rerank_ns = _time.monotonic_ns() - _t0

        if g:
            g.add_tool_span(
                input=f"rerank {len(raw['docs'])} candidates for: {question}",
                output=f"kept top {len(reranked['docs'])} of {len(raw['docs'])}",
                name="bge-reranker-v2-m3",
                step_number=2,
                duration_ns=_rerank_ns,
                metadata={"stage": "rerank", "model": "BAAI/bge-reranker-v2-m3",
                          "num_in": str(len(raw["docs"])), "num_out": str(len(reranked["docs"]))},
            )

        # Galileo retriever span — the reranked docs actually sent to the LLM.
        # Chunk-level metrics (chunk_relevance, precision@k) score THIS span.
        if g:
            g.add_retriever_span(
                input=question,
                output=[d.page_content for d in reranked["docs"]],
                name="milvus-reranked",
                step_number=3,
                metadata={"stage": "rerank-output",
                          "num_retrieved": str(len(raw["docs"])), "num_reranked": str(len(reranked["docs"]))},
            )

        # 3. Prompt augmentation
        aug_step = Step(name="prompt-augmentation", step_type="execution",
                        objective="Format retrieved context into prompt")
        _genai_handler.start_step(aug_step)
        _t0 = _time.monotonic_ns()
        try:
            vals = await FormatContext.ainvoke(reranked)
        finally:
            _genai_handler.stop_step(aug_step)
        _aug_ns = _time.monotonic_ns() - _t0

        _retrieved_context = vals["context"]

        if g:
            g.add_tool_span(
                input=f"format {len(reranked['docs'])} documents into prompt context",
                output=f"{len(_retrieved_context)} chars of context",
                name="prompt-augmentation",
                step_number=4,
                duration_ns=_aug_ns,
                metadata={"stage": "augment", "num_docs": str(len(reranked["docs"])),
                          "context_chars": str(len(_retrieved_context))},
            )

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

        # Snapshot the history the chain is about to load, BEFORE ainvoke appends this
        # turn — this is exactly what RunnableWithMessageHistory renders into {history},
        # so the logged LLM span input matches what OpenAI actually receives.
        try:
            _history_msgs = list(_get_history(session_id).messages)
        except Exception:
            _history_msgs = []

        _llm_start_ns = _time.monotonic_ns()
        try:
            response_text = await _run_llm(
                question=vals["question"],
                context=vals["context"],
                session_id=session_id,
                usage_sink=_usage,
            )
            buf.append(response_text)
            yield response_text
            agent.output_messages.append(
                OutputMessage(role="assistant", parts=[Text(content=response_text)])
            )
        except (ControlViolationError, ControlSteerError) as e:
            blocked_msg = (
                f"[Blocked] {e}" if isinstance(e, ControlViolationError)
                else f"[Steered] {e}"
            )
            buf.append(blocked_msg)
            yield blocked_msg
        finally:
            if _agent_token is not None:
                otel_context.detach(_agent_token)
            _genai_handler.stop_agent(agent)

    finally:
        if _wf_token is not None:
            otel_context.detach(_wf_token)
        _genai_handler.stop_workflow(workflow)
        # Galileo LLM span + conclude + flush
        if g:
            full_response = "".join(buf)
            duration_ns = _time.monotonic_ns() - _llm_start_ns if _llm_start_ns else None

            # Reconstruct the exact message list the chain sent: rendered system prompt
            # (with context) + prior turns from Redis + this question.
            _msgs = [{"role": "system", "content": SYSTEM_PROMPT.format(context=_retrieved_context)}]
            for _m in _history_msgs:
                _role = {"human": "user", "ai": "assistant"}.get(
                    getattr(_m, "type", ""), getattr(_m, "type", "user")
                )
                _msgs.append({"role": _role, "content": str(getattr(_m, "content", ""))})
            _msgs.append({"role": "user", "content": question})

            g.add_llm_span(
                input=_msgs,
                output={"role": "assistant", "content": full_response},
                model=CHAT_MODEL,
                temperature=TEMPERATURE,
                name="openai-chat",
                # step_number is the "topological step number" the Agent graph builds
                # edges from. Left unset, every span serializes as 0 and the graph can
                # only guess — the llm node ends up floating with no incoming arrow.
                step_number=5,
                duration_ns=duration_ns,
                num_input_tokens=_usage.get("input_tokens"),
                num_output_tokens=_usage.get("output_tokens"),
                total_tokens=_usage.get("total_tokens"),
                metadata={"num_history_messages": str(len(_history_msgs))},
            )
            # conclude_all pops the workflow span AND the trace.
            g.conclude(output=full_response, conclude_all=True)
            await g.async_flush()
