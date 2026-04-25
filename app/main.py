# app/main.py
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
# from openinference.instrumentation.langchain import LangChainInstrumentor
# from opentelemetry.instrumentation.langchain import LangChainInstrumentor
# from opentelemetry.instrumentation.redis import RedisInstrumentor

# # from traceloop.sdk import Traceloop
# # from traceloop.sdk.instruments import Instruments

# LangChainInstrumentor().instrument()
# OpenAIInstrumentor().instrument() # This is under openinference
# RedisInstrumentor().instrument()

# import both the non-streaming and streaming pipeline fns
from app.rag_pipeline import (
    stream_generate as rag_stream,
    generate as rag_generate,
)

# Traceloop.init(
#     app_name=os.getenv("OTEL_SERVICE_NAME", "chat-rag"),
#     resource_attributes={"deployment.environment": os.getenv("DEPLOY_ENV", "Nick-LLM")},
#     # instruments={Instruments.LANGCHAIN, Instruments.OPENAI, Instruments.MILVUS},
#     disable_batch=True,
# )

app = FastAPI(title="RAG Server")

# Allow the Streamlit app (localhost:8501) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Ask(BaseModel):
    # accept either "question" (preferred) or "q" (fallback)
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

# Output streaming endpoint for the Streamlit UI
@app.post("/chat")
async def chat(req: Ask):
    prompt = req.text()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'question'")
    sid = req.sid()

    async def token_stream():
        try:
            async for chunk in rag_stream(prompt, session_id=sid):
                # stream back plain text tokens
                yield chunk if isinstance(chunk, str) else str(chunk)
        except Exception as e:
            # surface errors to the UI as part of the stream
            yield f"\n[stream-error] {e}\n"

    return StreamingResponse(token_stream(), media_type="text/plain")


# JSON endpoint (non-streaming) if you need full result & score at once
@app.post("/generate")
async def generate_endpoint(req: Ask):
    prompt = req.text()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'question'")
    return await rag_generate(prompt)


# Optional: warm up heavy models so first call from Streamlit doesn't lag
@app.on_event("startup")
async def warmup():
    try:
        await rag_generate("warmup")
    except Exception:
        # ignore warmup failures; real requests will still run
        pass
