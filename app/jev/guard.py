"""Pre-response guardrail powered by Jev, recorded in the trace.

One System One request with the GUARD_QUESTIONS battery (~100–250 ms) →
block | redirect | review | pass. The call is recorded as a Langfuse
`guardrail` observation with one score per question (same mapping as
Langfuse's native Jev evaluator), and the same span reaches the collector and
Splunk AO through the shared TracerProvider.

`redirect` reuses the app's existing out-of-scope sentence so the user-facing
behaviour is unchanged — the difference is that the LLM is never called.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .client import JevClient, JevResult
from .config import settings
from .questions import GUARD_QUESTIONS, guard_state
from .scoring import record_jev_scores
from .telemetry import current_span, langfuse, set_openinference_io

log = logging.getLogger("jev.guard")

BLOCK_MESSAGE = "I can't help with that request."
REDIRECT_MESSAGE = "I can only answer questions related to the documents in my knowledge base."


@dataclass
class GuardDecision:
    action: str  # "pass" | "review" | "redirect" | "block"
    reason: str
    result: JevResult
    response: str | None = None

    @property
    def blocked(self) -> bool:
        return self.action in {"block", "redirect"}


_client: JevClient | None = None


def _jev() -> JevClient:
    global _client
    if _client is None:
        _client = JevClient()
    return _client


def decide(result: JevResult) -> tuple[str, str]:
    """Map probabilities to an action. Precedence: block > redirect > review > pass."""
    t = settings.thresholds
    p_inj, p_harm, p_tox = result.p("prompt_injection"), result.p("harmful_request"), result.p("toxic_input")
    severity = result.score("severity")
    if severity >= t.severity_block or max(p_inj, p_harm, p_tox) >= t.block:
        worst = max((p_inj, "prompt_injection"), (p_harm, "harmful_request"), (p_tox, "toxic_input"))
        return "block", f"{worst[1]}={worst[0]:.2f}, severity={severity:.2f}"
    if result.p("off_topic") >= t.block:
        return "redirect", f"off_topic={result.p('off_topic'):.2f}"
    if max(p_inj, p_harm, p_tox, result.p("pii_in_input")) >= t.review:
        return "review", "at least one signal in the review band"
    return "pass", "all signals below review threshold"


def guard_input(user_message: str) -> GuardDecision:
    lf = langfuse()
    state = guard_state(user_message)
    with lf.start_as_current_observation(as_type="guardrail", name="jev-input-guard", input=state) as obs:
        span = current_span()
        result = _jev().evaluate(state, GUARD_QUESTIONS)
        action, reason = decide(result)
        response = BLOCK_MESSAGE if action == "block" else REDIRECT_MESSAGE if action == "redirect" else None
        summary = {
            "action": action, "reason": reason, "model": result.model,
            "latency_ms": round(result.latency_ms, 1),
            "probabilities": {q: round(result.p(q), 3) for q, d in GUARD_QUESTIONS.items() if d["type"] == "noul"},
            "severity": round(result.score("severity"), 2),
        }
        obs.update(output=summary, metadata={"jev_request_id": result.request_id, "usage": result.usage},
                   level="WARNING" if action != "pass" else "DEFAULT")
        set_openinference_io(span, kind="GUARDRAIL", input_value=user_message, output_value=action,
                             **{"guardrail.action": action, "guardrail.provider": "typesafe-jev",
                                "guardrail.latency_ms": round(result.latency_ms, 1)})
        trace_id, obs_id = lf.get_current_trace_id(), lf.get_current_observation_id()
        record_jev_scores(lf, result, GUARD_QUESTIONS, prefix="guard", trace_id=trace_id, observation_id=obs_id)
        lf.create_score(name="guard.action", value=action, data_type="CATEGORICAL", comment=reason,
                        trace_id=trace_id, observation_id=obs_id, score_id=f"{obs_id}-guard-action")
        log.info("jev guard %s (%s) in %.0f ms", action, reason, result.latency_ms)
        return GuardDecision(action=action, reason=reason, result=result, response=response)
