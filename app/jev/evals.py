"""Post-response evaluation with Jev, pushed to Langfuse as scores.

Runs after the last token has been streamed to the user (so it adds no
user-visible latency), inside the same trace, as a Langfuse `evaluator`
observation with one score per question. Langfuse's native decision-model
evaluator (server-side, sampled, backfillable — see
scripts/langfuse_decision_evaluator.json) can score the same observation with
the same questions, zero code, for comparison.

Context handling: the pipeline hands us the reranked documents; we send the
top `JEV_EVAL_TOP_DOCS` (default 10) capped at `JEV_CONTEXT_MAX_CHARS`. Jev's
docs warn that accuracy falls as irrelevant state grows, and its hard limit is
32k tokens of state — 70 reranked chunks would be both slow and worse.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .client import JevClient, JevResult
from .config import settings
from .questions import EVAL_QUESTIONS, eval_state
from .scoring import record_jev_scores
from .telemetry import current_span, langfuse, set_openinference_io

log = logging.getLogger("jev.evals")

LOW_FAITHFULNESS_NOTE = (
    "\n\n[Note: this answer could not be fully verified against the retrieved documents.]"
)


@dataclass
class EvalOutcome:
    result: JevResult
    low_faithfulness: bool
    flagged: list[str]


def build_context(docs: list[dict] | None, fallback: str = "") -> str:
    """Top-N reranked passages, capped in size. `docs` items: {"source", "text"}."""
    if not docs:
        text = fallback
    else:
        parts = [f"[{d.get('source') or i}] {d.get('text', '')}" for i, d in enumerate(docs[: settings.jev_eval_top_docs])]
        text = "\n\n".join(parts)
    if len(text) > settings.jev_context_max_chars:
        text = text[: settings.jev_context_max_chars] + "\n\n[context truncated]"
    return text


_client: JevClient | None = None


def _jev() -> JevClient:
    global _client
    if _client is None:
        _client = JevClient()
    return _client


def evaluate_turn(user_message: str, retrieved_context: str, assistant_answer: str) -> EvalOutcome:
    lf = langfuse()
    state = eval_state(user_message, retrieved_context, assistant_answer)
    with lf.start_as_current_observation(as_type="evaluator", name="jev-response-eval", input=state) as obs:
        span = current_span()
        result = _jev().evaluate(state, EVAL_QUESTIONS)
        t = settings.thresholds
        flagged: list[str] = []
        if result.p("faithful") < t.faithful_min:
            flagged.append("low_faithfulness")
        if result.p("pii_in_output") >= t.pii_flag:
            flagged.append("pii_in_output")
        if result.p("leaks_instructions") >= t.block:
            flagged.append("leaks_instructions")
        summary = {
            "model": result.model, "latency_ms": round(result.latency_ms, 1),
            "faithful": round(result.p("faithful"), 3),
            "answers_question": round(result.p("answers_question"), 3),
            "pii_in_output": round(result.p("pii_in_output"), 3),
            "leaks_instructions": round(result.p("leaks_instructions"), 3),
            "helpfulness": round(result.score("helpfulness"), 2),
            "answer_tone": result.choice("answer_tone"),
            "flagged": flagged,
        }
        obs.update(output=summary, metadata={"jev_request_id": result.request_id, "usage": result.usage},
                   level="WARNING" if flagged else "DEFAULT")
        set_openinference_io(span, kind="EVALUATOR", input_value=state, output_value=summary,
                             **{"evaluator.provider": "typesafe-jev", "evaluator.latency_ms": round(result.latency_ms, 1)})
        trace_id, obs_id = lf.get_current_trace_id(), lf.get_current_observation_id()
        record_jev_scores(lf, result, EVAL_QUESTIONS, prefix="eval", trace_id=trace_id, observation_id=obs_id)
        for flag in flagged:
            lf.create_score(name=f"eval.flag.{flag}", value=1, data_type="BOOLEAN", comment="Jev post-check",
                            trace_id=trace_id, observation_id=obs_id, score_id=f"{obs_id}-flag-{flag}")
        log.info("jev eval faithful=%.2f flagged=%s in %.0f ms", result.p("faithful"), flagged, result.latency_ms)
        return EvalOutcome(result=result, low_faithfulness="low_faithfulness" in flagged, flagged=flagged)
