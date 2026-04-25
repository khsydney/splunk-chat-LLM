# app/deepeval_eval.py
import os
import asyncio
from typing import List

from langchain_core.documents import Document

EVAL_MODEL = os.getenv("EVAL_MODEL", "gpt-4o-mini")


def _make_test_case(question: str, answer: str, docs: List[Document]):
    from deepeval.test_case import LLMTestCase
    return LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=[d.page_content for d in docs],
    )


async def run_deepeval_metrics(question: str, answer: str, docs: List[Document]) -> dict:
    """Evaluate a RAG response with DeepEval metrics (async).

    Returns a dict keyed by metric name, each with:
      score  – float 0-1 (None on error)
      passed – bool (score >= threshold)
      reason – explanation string from the LLM judge
    """
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        ContextualRelevancyMetric,
    )

    test_case = _make_test_case(question, answer, docs)

    metrics = {
        "answer_relevancy": AnswerRelevancyMetric(model=EVAL_MODEL, threshold=0.5),
        "faithfulness": FaithfulnessMetric(model=EVAL_MODEL, threshold=0.5),
        "contextual_relevancy": ContextualRelevancyMetric(model=EVAL_MODEL, threshold=0.5),
    }

    async def _measure(name: str, metric):
        try:
            await metric.a_measure(test_case)
            return name, {
                "score": round(float(metric.score), 4) if metric.score is not None else None,
                "passed": metric.is_successful(),
                "reason": getattr(metric, "reason", None),
            }
        except Exception as exc:
            return name, {"score": None, "passed": False, "reason": str(exc)}

    outcomes = await asyncio.gather(*[_measure(n, m) for n, m in metrics.items()])
    return dict(outcomes)
