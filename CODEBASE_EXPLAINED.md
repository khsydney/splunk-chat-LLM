# Codebase Explained — Splunk Chat-LLM RAG App

A line-by-line walkthrough of every file that matters.

---

## Architecture Overview

```
Browser
  └── Streamlit UI  (streamlit_app.py, port 8501)
        └── HTTP POST /chat  (streaming text/plain)
              └── FastAPI backend  (app/main.py, port 8000)
                    └── RAG pipeline  (app/rag_pipeline.py)
                          ├── Milvus vector DB  (port 19530) — document retrieval
                          ├── Redis  (port 6379)             — conversation history
                          └── OpenAI API                     — LLM answer generation
                                ↓
                    OTel SDK  (opentelemetry-instrument CLI wrapper)
                          └── OTLP/gRPC → local Collector  (port 4317)
                                └── OTLP/HTTPS → Splunk Observability Cloud
```

---

## `.env` — Configuration

```
API_BASE=http://localhost:8000
```
The Streamlit UI reads this to know where to POST chat requests.

```
CHUNK_SIZE=1200
CHUNK_OVERLAP=120
RETRIEVAL_TOP_K=20
RERANK_TOP_K=8
```
RAG tuning knobs. Chunks are split at 1 200 characters with 120-character overlap so context isn't lost at boundaries. Retrieval fetches the top 20 closest vectors; the cross-encoder then re-scores and keeps the top 8.

```
OPENAI_API_KEY=sk-proj-...
CHAT_MODEL=gpt-3.5-turbo
TEMPERATURE=0.2
```
OpenAI credentials. Temperature 0.2 keeps answers factual and low-variance.

```
EMBEDDING_MODEL=BAAI/bge-m3
HF_HUB_OFFLINE=1
```
Uses BGE-M3 (a multilingual bi-encoder) for embedding both documents and queries. `HF_HUB_OFFLINE=1` prevents Hugging Face from making a network call on startup to check for model updates; it uses the local cache instead.

```
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=rag_chunks
```
Address and collection name for the Milvus vector store.

```
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_EXPORTER_OTLP_PROTOCOL=grpc
OTEL_TRACES_EXPORTER=otlp
OTEL_METRICS_EXPORTER=otlp
OTEL_LOGS_EXPORTER=otlp
```
Tells the OTel SDK to export all three signal types (traces, metrics, logs) via gRPC to the local collector on port 4317.

```
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event,splunk
```
Activates four emitters inside the Splunk GenAI SDK:
- `span` — creates OTel spans for every Workflow/Agent/Retrieval/Step
- `metric` — records duration histograms (e.g. `gen_ai.agent.duration`)
- `event` — emits structured log events (conversation content)
- `splunk` — Splunk-specific conversation and evaluation log records

```
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE=SPAN
```
Tells the OpenAI auto-instrumentor to write the full prompt and completion text as span attributes (`gen_ai.prompt.*`, `gen_ai.completion.*`). Splunk's platform-side evaluator reads these attributes to produce quality scores (Relevance, Sentiment, Hallucination, Toxicity, Bias) without any extra code.

```
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta
```
Sends metric deltas (counts since last export) rather than cumulative totals. Splunk's ingest expects delta temporality for histograms.

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```
Opts into the latest draft GenAI semantic conventions so attribute names match what Splunk's AI dashboards query.

```
OTEL_RESOURCE_ATTRIBUTES=deployment.environment=Nick-Chat-LLM,service.namespace=Nick-LLM
```
Tags every span/metric/log with the environment and namespace. Splunk uses `deployment.environment` as the primary filter in the AI Agents dashboard.

```
OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS=none
DEEPEVAL_EVALUATION_MODEL=gpt-4o-mini
```
Disables the Splunk SDK's background DeepEval evaluator (it fires in a thread after `stop_agent()` but by then the span is already exported so results never land on it). Evaluation is handled by Splunk's platform-side scanner instead.

---

## `app/main.py` — FastAPI Backend

### Lines 1–11: Imports
```python
import os
from dotenv import load_dotenv
load_dotenv()
```
`load_dotenv()` reads `.env` into `os.environ` so every subsequent `os.getenv()` call works. This runs before any other module-level code.

```python
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import SpanProcessor
```
`trace` is used to unwrap the `ProxyTracerProvider` that `opentelemetry-instrument` installs. `metrics` is used in the startup event to get the live `MeterProvider`. `SpanProcessor` is the base class for our custom sanitizer.

### Lines 33–52: `_SanitizingProcessor`
```python
def _sanitize(v):
    if isinstance(v, str):
        return v.encode("utf-8", errors="replace").decode("utf-8")
```
Encodes the string to UTF-8 with `errors="replace"` (lone Unicode surrogates become `�`) then decodes back. Python strings can contain lone surrogates (e.g. from malformed document text) but Splunk's ingest rejects them.

```python
class _SanitizingProcessor(SpanProcessor):
    def on_end(self, span):
        attrs = span._attributes
        if attrs:
            for k, v in list(attrs.items()):
                attrs[k] = _sanitize(v)
```
A custom `SpanProcessor` that runs on every span just before it is handed to the batch exporter. It walks every attribute value and sanitizes strings in place. `on_start` is a no-op because attributes may not be final yet.

### Lines 54–63: FastAPI app and CORS
```python
app = FastAPI(title="RAG Server")
app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"], ...)
```
Creates the app and allows cross-origin requests from Streamlit (port 8501) so the browser can call the API directly if needed.

### Lines 65–74: `Ask` request model
```python
class Ask(BaseModel):
    question: Optional[str] = None
    q: Optional[str] = None
    session_id: Optional[str] = None

    def text(self) -> str:
        return (self.question or self.q or "").strip()
    def sid(self) -> str:
        return (self.session_id or "default").strip() or "default"
```
Pydantic model that accepts either `question` or `q` (backward compat). `sid()` always returns a non-empty string so Redis never gets an empty key.

### Lines 76–95: Endpoints
```python
@app.get("/health")
def health():
    return {"ok": True}
```
Simple liveness probe. The startup script polls this to know when the server is ready.

```python
@app.post("/chat")
async def chat(req: Ask):
    async def token_stream():
        async for chunk in rag_stream(prompt, session_id=sid):
            yield chunk if isinstance(chunk, str) else str(chunk)
    return StreamingResponse(token_stream(), media_type="text/plain")
```
The main endpoint. It wraps `rag_stream` (the async generator from `rag_pipeline.py`) in another generator and returns it as a `StreamingResponse`. The client receives a continuous `text/plain` stream of tokens as they arrive from OpenAI.

### Lines 104–119: `setup_telemetry` startup event
```python
@app.on_event("startup")
async def setup_telemetry():
    tp = trace.get_tracer_provider()
    real_tp = getattr(tp, "_provider", tp)
    if hasattr(real_tp, "add_span_processor"):
        real_tp.add_span_processor(_SanitizingProcessor())
```
`opentelemetry-instrument` wraps the real `TracerProvider` in a `ProxyTracerProvider`. `getattr(tp, "_provider", tp)` unwraps it to get the real provider so we can call `add_span_processor`. This registers `_SanitizingProcessor` to run on every span.

```python
    from opentelemetry.util.genai.handler import get_telemetry_handler
    _pipeline._genai_handler = get_telemetry_handler(
        meter_provider=metrics.get_meter_provider()
    )
```
Attempts to reinitialize the Splunk GenAI handler with the fully-configured `MeterProvider`. Note: `TelemetryHandler` is a singleton — if already initialized this call is a no-op (the SDK logs a warning). The `meter_provider` argument was added to enable `force_flush()` on each `stop_agent()` call.

---

## `app/rag_pipeline.py` — RAG Pipeline

### Lines 1–23: Imports and handler init
```python
load_dotenv()
```
Called again here because `rag_pipeline.py` can be imported directly in tests without going through `main.py`.

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
```
LangChain components: embeddings wrapper, document type, message types for streaming chunk detection, prompt builder, runnable primitives, OpenAI chat wrapper, and the cross-encoder for reranking.

```python
from opentelemetry import trace, context as otel_context
from opentelemetry.util.genai.types import (
    Workflow, AgentInvocation, RetrievalInvocation,
    InputMessage, OutputMessage, Text, Step,
)
from opentelemetry.util.genai.handler import get_telemetry_handler
```
OTel core (`trace`, `context`) plus the Splunk GenAI SDK types and handler. `context as otel_context` is aliased to avoid name collision with Python's built-in.

```python
_genai_handler = get_telemetry_handler()
```
Gets the process-wide singleton `TelemetryHandler`. At this point `opentelemetry-instrument` has already configured the global `TracerProvider` and `MeterProvider` via `sitecustomize.py`, so instruments created here bind to the real OTLP exporters.

### Lines 38–71: Embedding model and Milvus search

```python
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
```
Loads BGE-M3 at module import time (once, not per request). BGE-M3 is a 570 M-parameter multilingual model that produces 1 024-dimensional vectors.

```python
retrieval_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "100"))
rerank_TOP_K = int(os.getenv("RERANK_TOP_K", "70"))
```
Controls how many vectors are fetched from Milvus (`RETRIEVAL_TOP_K`) and how many survive reranking (`RERANK_TOP_K`). Fetching more candidates than you keep improves recall at the cost of reranking time.

```python
def _get_milvus_client():
    global _milvus_client
    if _milvus_client is None:
        from pymilvus import MilvusClient
        _milvus_client = MilvusClient(uri=MILVUS_URI)
    return _milvus_client
```
Lazy singleton for the Milvus gRPC client. Deferred import avoids a startup error if Milvus is not yet up when the module loads.

```python
def _milvus_search(question: str, k: int = retrieval_TOP_K) -> List[Document]:
    vec = emb.embed_query(question)
```
Embeds the user question into a 1 024-d vector using BGE-M3.

```python
    results = client.search(
        collection_name=COLL,
        data=[vec],
        limit=k,
        output_fields=["text", "source"],
        search_params={"metric_type": "IP", "params": {"nprobe": 100}},
    )[0]
```
Inner-product (IP) similarity search — equivalent to cosine similarity on unit-norm vectors. `nprobe=100` means the HNSW index probes 100 candidate clusters per query; higher values improve recall at the cost of latency. `[0]` takes the results for the first (and only) query vector.

```python
    return [
        Document(page_content=hit["entity"].get("text", ""),
                 metadata={"source": hit["entity"].get("source", "")})
        for hit in results
    ]
```
Converts raw Milvus result dicts into LangChain `Document` objects.

### Lines 73–92: Cross-encoder reranking

```python
try:
    _cross = CrossEncoder("BAAI/bge-reranker-v2-m3")
except Exception:
    _cross = None
```
Loads the BGE reranker at startup. Falls back to `None` if it fails (e.g. model not cached); the pipeline degrades gracefully to top-K truncation.

```python
def _rerank_impl(question: str, docs: List[Document]) -> List[Document]:
    pairs = [[question, d.page_content] for d in docs]
    scores = _cross.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:rerank_TOP_K]
    return [d for d, _ in ranked]
```
The cross-encoder scores each `(question, chunk)` pair jointly — unlike the bi-encoder it sees both texts at once, making it more accurate but slower. Results are sorted descending by score and truncated to `rerank_TOP_K`.

```python
Rerank = RunnableLambda(
    lambda x: {"question": x["question"], "docs": _rerank_impl(x["question"], x["docs"])}
).with_config({"run_name": "bge-reranker"})
```
Wraps the rerank function as a LangChain `Runnable` so it can be chained with `|`. `run_name` appears as the span name in LangChain traces.

### Lines 94–99: Context formatting

```python
def _format_ctx(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)

FormatContext = RunnableLambda(
    lambda x: {"question": x["question"], "docs": x["docs"], "context": _format_ctx(x["docs"])}
).with_config({"run_name": "FormatContext"})
```
Concatenates all reranked chunks into one string separated by double newlines. The `docs` list is passed through unchanged so it's available downstream (e.g. for evaluation context).

### Lines 104–121: Prompt template and LLM

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a document Q&A assistant. ..."
               "Context:\n{context}"),
    ("placeholder", "{history}"),
    ("human", "{question}"),
])
```
Three-part prompt:
- `system` — injects retrieved context and sets guardrails (only answer from docs; block unrelated questions)
- `{history}` placeholder — expands to the full Redis conversation history for this session
- `human` — the user's current question

```python
llm = ChatOpenAI(model=CHAT_MODEL, temperature=TEMPERATURE, stream_usage=True, streaming=True)
```
`stream_usage=True` tells LangChain to include a token usage chunk at the end of the stream. `streaming=True` enables SSE-style streaming from OpenAI. The OpenAI auto-instrumentor captures this call and creates an `openai.chat` span with token counts and content.

```python
retrieve_stage = RunnableLambda(lambda x: {"docs": _milvus_search(x), "question": x})
```
Wraps `_milvus_search` as a Runnable. Takes a plain string question and returns `{"docs": [...], "question": "..."}` — the dict format the rest of the chain expects.

```python
llm_chain_stream = prompt | llm
```
The minimal LLM chain: fill the prompt template then call OpenAI. Does not include history yet — that's added by `RunnableWithMessageHistory` below.

### Lines 136–161: Redis conversation history

```python
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_MEMORY_TTL_SEC = int(os.getenv("MEMORY_TTL_SEC", "604800"))  # 7 days
```
Messages expire from Redis after 7 days of inactivity.

```python
def _get_history(session_id: str) -> ChatMessageHistory:
    if REDIS_URL:
        return RedisChatMessageHistory(
            session_id=session_id, url=REDIS_URL,
            ttl=_MEMORY_TTL_SEC, key_prefix=REDIS_KEY_PREFIX,
        )
    hist = _inmem_histories.get(session_id)
    ...
```
Returns a `RedisChatMessageHistory` if Redis is available, otherwise falls back to an in-memory dict. Each session gets its own Redis key (`rag:msgs:<session_id>`).

```python
llm_chain_stream_with_mem = RunnableWithMessageHistory(
    llm_chain_stream,
    _get_history,
    input_messages_key="question",
    history_messages_key="history"
)
```
Wraps `llm_chain_stream` so that before each invocation it fetches the session's history from Redis and injects it at the `{history}` placeholder, and after each invocation it appends the new human+AI message pair back to Redis.

### Lines 196–283: `stream_generate` — the main async generator

This is the entry point called by `main.py`. It is an `async def` generator (uses `yield`) — FastAPI's `StreamingResponse` iterates it to stream tokens to the client.

#### Workflow span (lines 197–211)

```python
workflow = Workflow(
    name="rag-pipeline",
    workflow_type="rag",
    input_messages=[InputMessage(role="user", parts=[Text(content=question)])],
    conversation_id=session_id,
    agent_name="rag-pipeline",
)
_genai_handler.start_workflow(workflow)
```
Creates and starts a Splunk GenAI `Workflow` span. This becomes the root span for the entire RAG request in Splunk's trace view and AI dashboards. `agent_name` annotates the workflow with the agent identity.

```python
_wf_token = None
if getattr(workflow, "span", None) is not None:
    _wf_token = otel_context.attach(trace.set_span_in_context(workflow.span))
```
**Critical async workaround.** The SDK's `_push_current_span` skips `context_api.attach()` in async contexts to avoid cross-task detach errors. Without manual attachment, the subsequent auto-instrumented spans (`openai.chat`, Milvus gRPC) would parent to the HTTP request span instead of the workflow span. We manually attach the workflow span to the OTel context so all child spans nest correctly.

#### Step 1 — Milvus retrieval (lines 213–229)

```python
retrieval = RetrievalInvocation(retriever_type="milvus", query=question, top_k=retrieval_TOP_K)
_genai_handler.start_retrieval(retrieval)
_ret_token = otel_context.attach(trace.set_span_in_context(retrieval.span))
try:
    raw = await retrieve_stage.ainvoke(question)
    retrieval.documents_retrieved = len(raw["docs"])
finally:
    otel_context.detach(_ret_token)
    _genai_handler.stop_retrieval(retrieval)
```
Creates a `retrieval` span under the workflow. The Milvus gRPC call (`milvus.search`) triggered by `retrieve_stage.ainvoke()` nests inside the retrieval span because we attach the retrieval context first. `documents_retrieved` is recorded on the span so Splunk shows how many chunks were fetched.

#### Step 2 — Reranking (lines 231–238)

```python
rerank_step = Step(name="reranking", step_type="execution",
                   objective=f"Cross-encoder rerank top {rerank_TOP_K}")
_genai_handler.start_step(rerank_step)
try:
    reranked = await Rerank.ainvoke(raw)
finally:
    _genai_handler.stop_step(rerank_step)
```
Wraps the cross-encoder call in a `Step` span. A `Step` is a generic execution unit in Splunk's model — useful for timing non-LLM work like CPU-bound reranking.

#### Step 3 — Prompt augmentation (lines 240–247)

```python
aug_step = Step(name="prompt-augmentation", ...)
_genai_handler.start_step(aug_step)
try:
    vals = await FormatContext.ainvoke(reranked)
finally:
    _genai_handler.stop_step(aug_step)
```
Wraps the context-formatting step. `vals` now contains `{"question": ..., "docs": [...], "context": "..."}` — all three keys needed for the LLM call.

#### Step 4 — LLM call inside AgentInvocation (lines 249–278)

```python
agent = AgentInvocation(
    name="rag-pipeline",
    agent_type="rag",
    provider="openai",
    conversation_id=session_id,
    input_messages=[InputMessage(role="user", parts=[Text(content=question)])],
)
_genai_handler.start_agent(agent)
```
Creates an `invoke_agent` span. Splunk's AI Agents page uses `invoke_agent` spans with `gen_ai.agent.name` to populate the agent table and track request counts, latency, and tokens.

```python
_agent_token = otel_context.attach(trace.set_span_in_context(agent.span))
```
Same async workaround as the workflow — manually attaches agent context so the `openai.chat` span (created by the OpenAI auto-instrumentor) becomes a child of the `invoke_agent` span. This is what allows Splunk to attribute token counts to the agent.

```python
async for chunk in llm_chain_stream_with_mem.astream(
    {"question": vals["question"], "context": vals["context"]},
    config={"configurable": {"session_id": session_id}}
):
    text = chunk.content if isinstance(chunk, AIMessageChunk) else str(chunk)
    if text:
        buf.append(text)
        yield text
```
Streams tokens from OpenAI via LangChain. Each `AIMessageChunk` has a `.content` string. The `yield text` sends the token to FastAPI's `StreamingResponse` which forwards it to the browser immediately. `session_id` in `configurable` tells `RunnableWithMessageHistory` which Redis key to use for this conversation.

```python
agent.output_messages.append(
    OutputMessage(role="assistant", parts=[Text(content="".join(buf))])
)
```
After all tokens are received, appends the complete response to the agent's output messages. Splunk uses this to display the full answer in the AI details panel.

```python
finally:
    otel_context.detach(_agent_token)
    _genai_handler.stop_agent(agent)
```
Detaches context and closes the agent span. `stop_agent()` records the `gen_ai.agent.duration` histogram metric and (if `meter_provider` is set) force-flushes it to the OTLP exporter.

#### Workflow cleanup (lines 280–283)

```python
finally:
    if _wf_token is not None:
        otel_context.detach(_wf_token)
    _genai_handler.stop_workflow(workflow)
```
The outer `finally` runs even if an exception occurs mid-stream (e.g. OpenAI timeout, client disconnect). This guarantees the workflow span is always closed so Splunk never sees an open/hanging trace.

---

## `streamlit_app.py` — Chat UI

### Lines 1–13: Setup
```python
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
HTTPXClientInstrumentor().instrument()
```
Instruments all `httpx` calls made by Streamlit so the outgoing `/chat` request gets a span. This links the browser-side latency trace to the backend FastAPI trace.

```python
if "sid" not in st.session_state:
    st.session_state.sid = f"ui-{uuid.uuid4()}"
```
Generates a random UUID-based session ID per browser tab. Persists in `session_state` so it doesn't change on reruns. This ID is sent with every `/chat` request and is used as the Redis key for conversation history.

### Lines 17–70: Sidebar
```python
api_base = st.sidebar.text_input("Backend base URL", value=default_api)
```
Lets you point the UI at a different backend (e.g. remote server) without code changes.

```python
uploaded = st.sidebar.file_uploader("Upload PDFs / MD / TXT / HTML", ...)
if uploaded:
    docs_dir = pathlib.Path("data/docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    for up in uploaded:
        dest = docs_dir / up.name
        dest.write_bytes(up.getbuffer())
```
Writes uploaded files to `data/docs/`. The indexer (`python -m index.indexer`) reads from this directory.

```python
if st.sidebar.button("Rebuild Index"):
    proc = subprocess.run(["python", "-m", "index.indexer"], ...)
```
Runs the indexer as a subprocess. Captures stdout/stderr and displays them in the sidebar so you can see progress or errors inline.

### Lines 72–117: Chat loop
```python
if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"assistant","content":"Hi! ..."}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
```
Persists full conversation history in `session_state`. On every rerun (triggered by new input or `st.rerun()`) it re-renders all past messages.

```python
def stream_from_backend(question: str):
    url = api_base.rstrip("/") + "/chat"
    with httpx.stream("POST", url, json={"question": question, "session_id": st.session_state.sid}, timeout=None) as resp:
        for chunk in resp.iter_text():
            if chunk:
                yield chunk
```
Uses `httpx` in streaming mode — keeps the connection open and yields text chunks as they arrive. `timeout=None` prevents the request from being killed if the LLM is slow.

```python
final = st.write_stream(stream_from_backend(prompt))
```
`st.write_stream` accepts a generator and renders each chunk into the chat bubble in real time. Returns the complete concatenated text when the stream ends.

```python
cites = sorted(set(re.findall(r"\[([^\[\]\n]+)\]", final)))
if cites:
    with st.expander("Sources (parsed from answer)"):
        for c in cites[:30]:
            st.write("•", c)
```
Parses `[source]` citations the LLM may include in its answer and displays them in a collapsible expander.

```python
st.session_state.messages.append({"role":"assistant","content":final})
st.rerun()
```
Appends the answer to history and forces a full rerun so Streamlit re-renders the complete conversation with proper avatar styling.

---

## How a single request flows end-to-end

```
1. User types a question → Streamlit sends POST /chat {question, session_id}
2. FastAPI receives it → calls stream_generate(question, session_id)
3. stream_generate opens a Workflow span (root of the trace)
4. Milvus retrieval: question → BGE-M3 embedding → vector search → top-20 chunks
5. Reranking: cross-encoder scores all 20 (question, chunk) pairs → keeps top-8
6. Prompt augmentation: 8 chunks are joined into the context string
7. AgentInvocation span opens; agent context attached to OTel context
8. LangChain fetches Redis history for this session
9. OpenAI gpt-3.5-turbo streams the answer token by token
   └── openai.chat span (auto-instrumented) nests inside the AgentInvocation span
10. Each token is yielded through FastAPI → httpx → Streamlit → browser
11. Stream ends; output is recorded on the agent span; agent span closes
12. Workflow span closes
13. OTel SDK batches all spans + metrics → gRPC to local collector → HTTPS to Splunk
14. Splunk platform-side evaluator scans gen_ai.prompt/completion attrs on openai.chat
    → produces Relevance/Sentiment/Hallucination/Toxicity/Bias quality scores
```

---

## How to start everything

```bash
# Everything (containers + backend + frontend) with one command:
./run.sh

# Or step by step:
./run.sh infra    # 1. OTel collector, Milvus, Redis, Attu
./run.sh index    # 2. (first run) build the Milvus index from data/docs
./run.sh app      # 3. backend (no --reload) + Streamlit UI
```

> **Why no `--reload`?** Uvicorn's `--reload` spawns a child worker process. The OTel SDK exporters are initialized in the parent process and do not get re-initialized in the child, so no traces reach Splunk.
