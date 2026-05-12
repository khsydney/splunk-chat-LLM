# app/main.py
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

    from opentelemetry.util.genai.handler import get_telemetry_handler
    _pipeline._genai_handler = get_telemetry_handler(
        meter_provider=metrics.get_meter_provider()
    )
