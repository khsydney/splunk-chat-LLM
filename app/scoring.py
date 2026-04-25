import numpy as np
from sentence_transformers import SentenceTransformer
from opentelemetry.metrics import get_meter

# lazy-load to keep startup snappy
_model = None
def _emb():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def answer_context_similarity(answer: str, contexts: list[str]) -> float:
    text = " ".join(contexts)[:8000]
    em = _emb()
    a, c = em.encode([answer, text], normalize_embeddings=True)
    return float(np.clip(np.dot(a, c), -1.0, 1.0))

_meter = get_meter(__name__)
hist = _meter.create_histogram(
    "rag.score.answer_context_similarity",
    unit="1",
    description="Cosine similarity between model answer and retrieved context",
)
def record_similarity(value: float, attrs: dict):
    hist.record(value, attributes=attrs)
