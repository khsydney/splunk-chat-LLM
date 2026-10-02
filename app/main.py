# app/main.py
import logging
import os
from dotenv import load_dotenv
load_dotenv()
from typing import Optional
from fastapi import FastAPI, HTTPException
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import SpanProcessor
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.rag_pipeline import stream_generate as rag_stream
import app.rag_pipeline as _pipeline
from app.jev import telemetry as jev_telemetry
from app.jev import splunk_ao as jev_splunk_ao
from app.jev.config import settings as jev_settings
from app.jev.turn import traced_stream

log = logging.getLogger("app.main")

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
    allow_origins=["http://localhost:8502", "http://127.0.0.1:8502"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Ask(BaseModel):
    question: Optional[str] = None
    q: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None  # optional; shows up as the Langfuse user

    def text(self) -> str:
        return (self.question or self.q or "").strip()
    def sid(self) -> str:
        return (self.session_id or "default").strip() or "default"
    def uid(self) -> str:
        return (self.user_id or "anonymous").strip() or "anonymous"

@app.get("/health")
def health():
    return {
        "ok": True,
        "jev": "mock" if jev_settings.mock_jev else ("on" if jev_settings.jev_enabled else "off"),
        "langfuse": jev_settings.langfuse_enabled,
        "splunk_ao": jev_settings.splunk_ao_enabled,
    }

@app.post("/chat")
async def chat(req: Ask):
    prompt = req.text()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'question'")
    sid = req.sid()

    async def token_stream():
        try:
            # Jev guard → existing RAG stream → Jev evals, all inside one trace (see app/jev/turn.py)
            async for chunk in traced_stream(prompt, sid, rag_stream, user_id=req.uid()):
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

    # Dual export: Langfuse (+Jev) and Splunk Agent Observability (+Luna-2) join the SAME provider
    # that opentelemetry-instrument already exports to the collector. No-ops without credentials.
    jev_telemetry.attach(real_tp)
    jev_splunk_ao.init_agent_control()

    try:
        from opentelemetry.util.genai.handler import get_telemetry_handler
        _pipeline._genai_handler = get_telemetry_handler(
            meter_provider=metrics.get_meter_provider()
        )
    except Exception as e:  # keep serving even if the GenAI handler cannot be rebuilt
        log.warning("util-genai telemetry handler not (re)initialised: %s", e)

@app.on_event("shutdown")
async def flush_telemetry():
    jev_telemetry.flush()
