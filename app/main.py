# app/main.py
import os
import logging
from dotenv import load_dotenv
load_dotenv()

# `opentelemetry-instrument` installs an OTel LoggingHandler on the root logger, which
# ships records to the collector but prints nothing. uvicorn only configures its own
# loggers. Net effect: app-level warnings/errors (Galileo, Agent Control) were invisible
# on the console. Add a StreamHandler alongside the OTel one — checking for a
# StreamHandler specifically, since root is never handler-free under instrumentation.
_root = logging.getLogger()
if not any(type(h) is logging.StreamHandler for h in _root.handlers):
    _sh = logging.StreamHandler()
    _sh.setFormatter(logging.Formatter("%(levelname)-8s %(name)s - %(message)s"))
    _root.addHandler(_sh)
_root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
from typing import Optional
from fastapi import FastAPI, HTTPException
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import SpanProcessor
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.rag_pipeline import stream_generate as rag_stream
import app.rag_pipeline as _pipeline

def _sanitize(v):
    if isinstance(v, str):
        return v.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(v, (list, tuple)):
        return type(v)(_sanitize(x) for x in v)
    return v

class _SanitizingProcessor(SpanProcessor):
    """Scrub lone surrogates from every span attribute before the batch exporter ships it to Splunk."""
    def on_start(self, span, parent_context=None): pass
    def on_end(self, span):
        try:
            attrs = span._attributes
            if attrs:
                for k, v in list(attrs.items()):
                    attrs[k] = _sanitize(v)
        except Exception:
            pass
    def shutdown(self): pass
    def force_flush(self, timeout_millis=30000): return True

app = FastAPI(title="RAG Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Ask(BaseModel):
    question: Optional[str] = None
    q: Optional[str] = None
    session_id: Optional[str] = None

    def text(self) -> str:
        return (self.question or self.q or "").strip()
    def sid(self) -> str:
        return (self.session_id or "default").strip() or "default"

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
async def chat(req: Ask):
    prompt = req.text()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'question'")
    sid = req.sid()

    async def token_stream():
        try:
            async for chunk in rag_stream(prompt, session_id=sid):
                yield chunk if isinstance(chunk, str) else str(chunk)
        except Exception as e:
            yield f"\n[stream-error] {e}\n"

    return StreamingResponse(token_stream(), media_type="text/plain")

@app.on_event("startup")
async def setup_telemetry():
    tp = trace.get_tracer_provider()
    real_tp = getattr(tp, "_provider", tp)
    if hasattr(real_tp, "add_span_processor"):
        real_tp.add_span_processor(_SanitizingProcessor())

    # NOTE: galileo.otel (GalileoSpanProcessor -> POST /otel/traces) was evaluated as a
    # replacement for the manual g.add_*_span calls in rag_pipeline.py. It ingests fine,
    # but Galileo types spans from its own `galileo.*` attributes, not OTel GenAI semconv:
    # a span carrying gen_ai.operation.name="retrieval" lands as type `workflow`, not
    # `retriever`. Chunk metrics (chunk_relevance / context_precision / precision_at_k)
    # only score typed retriever spans carrying documents, so the OTLP path would silently
    # drop them. Keep the manual retriever spans.

    from opentelemetry.util.genai.handler import get_telemetry_handler
    _pipeline._genai_handler = get_telemetry_handler(
        meter_provider=metrics.get_meter_provider()
    )

    _mlog = logging.getLogger(__name__)
    try:
        # Use canonical ScorerName values (golden demo pattern — same enum works for
        # both log stream metrics and experiments, values match the Protect API snake_case)
        from galileo_core.schemas.shared.scorers.scorer_name import ScorerName as GalileoScorers
        from galileo.log_streams import enable_metrics

        project_name    = os.getenv("GALILEO_PROJECT", "nkim-chatbot")
        log_stream_name = os.getenv("GALILEO_LOG_STREAM", "production")

        wanted = [
            # RAG quality
            GalileoScorers.context_adherence,
            GalileoScorers.completeness,
            GalileoScorers.context_relevance,
            # chunk_relevance must stay enabled: Context Precision and Precision@K are
            # composite metrics built on it, and the API rejects the upsert with
            # "Cannot disable ... required by composite metrics" if it is dropped.
            GalileoScorers.chunk_relevance,
            # Replaces chunk_attribution_utilization, which the API now rejects outright:
            # "Cannot enable deprecated metrics: chunk_attribution_utilization."
            # That single entry used to fail the whole call, so NO metric got enabled.
            GalileoScorers.context_precision,
            GalileoScorers.precision_at_k,

            # Safety
            GalileoScorers.input_pii,
            GalileoScorers.output_pii,
            GalileoScorers.input_toxicity_luna,
            GalileoScorers.output_toxicity_luna,
            GalileoScorers.prompt_injection_luna,
        ]

        from galileo.log_stream import LogStream
        _ls = LogStream.get(name=log_stream_name, project_name=project_name)

        # The upsert REPLACES the scorer set, so merge with what is already configured
        # in the Galileo console instead of silently clobbering it.
        existing = list(_ls.get_metrics())
        merged = existing + [m.value for m in wanted if m.value not in existing]
        enable_metrics(project_name=project_name, log_stream_name=log_stream_name, metrics=merged)

        # Read back: a metric being requested is not proof it is on.
        actual = set(LogStream.get(name=log_stream_name, project_name=project_name).get_metrics())
        missing = [m.value for m in wanted if m.value not in actual]
        if missing:
            _mlog.error("Galileo metrics requested but NOT enabled server-side: %s", missing)
        _mlog.warning("Galileo metrics enabled on %s/%s (%d): %s",
                      project_name, log_stream_name, len(actual), sorted(actual))
    except Exception:
        # Never swallow this again — a single bad metric fails the entire upsert.
        _mlog.exception("Galileo enable_metrics failed")

    # Agent Control init (per docs.galileo.ai/how-to-guides/agent-control/initialize-and-configure-agent-control)
    try:
        import httpx as _httpx
        _log = logging.getLogger(__name__)
        server_url  = os.getenv("AGENT_CONTROL_URL")
        agent_name  = os.getenv("AGENT_CONTROL_AGENT_NAME", "rag-pipeline")
        api_key     = os.getenv("GALILEO_API_KEY")
        api_key_hdr = os.getenv("AGENT_CONTROL_API_KEY_HEADER", "Galileo-API-Key")
        target_type = os.getenv("AGENT_CONTROL_TARGET_TYPE", "log_stream")

        _ac_available = False
        if server_url and api_key:
            try:
                probe = _httpx.get(f"{server_url}/api/v1/agents", timeout=3,
                                   headers={api_key_hdr: api_key})
                _ac_available = probe.status_code not in (404, 502, 503)
            except Exception:
                pass

        if _ac_available:
            import agent_control
            # Resolve project/log_stream IDs from the pipeline's GalileoLogger (docs pattern)
            g = _pipeline._get_galileo()
            if g is None:
                raise RuntimeError("GalileoLogger not initialized — cannot init Agent Control")

            # Docs: set env vars so SDK can resolve IDs internally
            if g.project_id:
                os.environ["GALILEO_PROJECT_ID"] = str(g.project_id)
            if g.log_stream_id:
                os.environ["GALILEO_LOG_STREAM_ID"] = str(g.log_stream_id)

            agent_control.init(
                agent_name=agent_name,
                agent_description="Splunk RAG chat pipeline with Milvus retrieval",
                server_url=server_url,
                api_key=api_key,
                api_key_header=api_key_hdr,
                observability_enabled=True,
                observability_sink_name="registered",
                target_type=target_type,
                target_id=str(g.log_stream_id),
            )
            _log.warning("Agent Control initialized for agent '%s' → log_stream_id=%s", agent_name, g.log_stream_id)
        else:
            _log.warning("Agent Control server not reachable at %s — skipping init, @control calls will fall back", server_url)
    except Exception:
        logging.getLogger(__name__).exception("Agent Control init failed")
