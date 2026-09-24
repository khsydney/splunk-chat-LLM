"""Turn a Jev result into Langfuse scores.

Mapping (identical to Langfuse's native decision-model evaluator, see
https://langfuse.com/docs/evaluation/evaluation-methods/jev-as-a-judge):

    Noul   -> NUMERIC     value = P(true) in [0, 1]
    Score  -> NUMERIC     value = expected level in [0, levels-1]
    Choice -> CATEGORICAL value = selected option

Each score carries a human-readable `comment` and `metadata.typesafe` with the
question id, type, model version, probabilities, confidence and legend, so a
dashboard can show *why* a number is what it is even though Jev itself never
returns a rationale.

`score_id` is deterministic (observation id + question) so re-running an
evaluation updates the score instead of duplicating it.
"""
from __future__ import annotations

from .client import JevResult


def record_jev_scores(lf, result: JevResult, questions: dict[str, dict], *, prefix: str,
                      trace_id: str | None, observation_id: str | None) -> list[dict]:
    """Create one Langfuse score per question. Returns the score payloads (for tests/logging)."""
    created: list[dict] = []
    for qid, q in questions.items():
        ans = result.answers[qid]
        name = f"{prefix}.{qid}"
        meta = {
            "typesafe": {
                "questionId": qid,
                "type": ans["type"],
                "model": result.model,
                "requestId": result.request_id,
                "probabilities": ans.get("probabilities"),
                "confidence": ans.get("confidence"),
                "legend": ans.get("legend"),
                "mock": result.mock,
            }
        }
        if ans["type"] == "noul":
            payload = dict(name=name, value=float(ans["noul"]), data_type="NUMERIC",
                           comment=f"P(true)={ans['noul']:.2f}")
        elif ans["type"] == "score":
            probs = ans.get("probabilities") or {}
            best = max(probs, key=probs.get) if probs else None
            payload = dict(name=name, value=float(ans["score"]), data_type="NUMERIC",
                           comment=f"expected level {ans['score']:.2f}; confidence {ans.get('confidence', 0):.2f}; "
                                   f"most likely level {best}")
        elif ans["type"] == "choice":
            probs = ans.get("probabilities") or {}
            ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            runner_up = f"; runner-up {ranked[1][0]} ({ranked[1][1]:.2f})" if len(ranked) > 1 else ""
            payload = dict(name=name, value=str(ans["choice"]), data_type="CATEGORICAL",
                           comment=f"{ans['choice']} (p={probs.get(ans['choice'], 0):.2f}); "
                                   f"confidence {ans.get('confidence', 0):.2f}{runner_up}")
        else:
            continue
        payload.update(trace_id=trace_id, observation_id=observation_id, metadata=meta,
                       score_id=f"{observation_id or trace_id}-{prefix}-{qid}")
        lf.create_score(**payload)
        created.append(payload)
    return created
