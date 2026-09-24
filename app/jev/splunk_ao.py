"""The Splunk Agent Observability (formerly Galileo) half of the comparison.

Tracing needs nothing here — telemetry.py already exports every span to the
Agent Stream. What this module adds:

1. `enable_luna_evaluators()` — turn on the Luna-2 evaluators (and their
   LLM-judge twins, if you want rationales next to the Luna scores) on the
   Agent Stream. Evaluators run server-side on sampled traces; nothing in the
   request path changes.

   Gating (research/luna2.md §6): Luna-2 is Enterprise-only. Splunk
   Observability Cloud SaaS exposes only Prompt Injection, Toxicity, Sexism and
   PII on Luna; Context Adherence / Completeness / Tool metrics on Luna need an
   on-prem or standalone Enterprise tenant. Selecting an evaluator the tenant
   does not license may be rejected or ignored at enable time — check the API
   response and the Agent Stream's evaluator page.

2. `guard_with_agent_control(...)` — OPTIONAL runtime guardrail via Agent
   Control (`pip install agent-control-sdk`, Python 3.12+). Pre-stage controls
   attached to the Agent Stream in the Splunk AO UI (e.g. `galileo.luna2`
   Prompt Injection > 0.5 → Deny) are evaluated before the LLM call and Post
   controls after. This is the like-for-like counterpart of jev_guardrail.py.

3. `protect_invoke(...)` — the legacy Protect REST call (`/v2/protect/invoke`).
   Protect is deprecated since June 2026 but the endpoint is still in the API
   reference; kept here only for tenants that have not moved to Agent Control.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable

from .config import settings

log = logging.getLogger("demo.splunk")

# Luna twins of the metrics Jev is asked in questions.py, plus their LLM-judge siblings.
LUNA_EVALUATORS = [
    "prompt_injection_luna",     # ≈ guard.prompt_injection
    "input_toxicity_luna",       # ≈ guard.toxic_input
    "output_toxicity_luna",
    "input_pii_luna",            # ≈ guard.pii_in_input
    "output_pii_luna",           # ≈ eval.pii_in_output
    "context_adherence_luna",    # ≈ eval.faithful          [on-prem / Enterprise only]
    "completeness_luna",         # ≈ eval.answers_question  [on-prem / Enterprise only]
    "input_tone_luna",           # ≈ eval.answer_tone       [on-prem / Enterprise only]
    "output_tone_luna",
]
LLM_JUDGE_TWINS = ["context_adherence", "prompt_injection", "input_toxicity", "instruction_adherence"]


def enable_luna_evaluators(*, include_llm_judge_twins: bool = True, only: list[str] | None = None) -> list[str]:
    """Enable evaluators on the configured Agent Stream. Returns what was requested."""
    from splunk_ao.agent_streams import enable_evaluators
    from splunk_ao.schema.metrics import SplunkAOEvaluators

    names = list(only or LUNA_EVALUATORS)
    if include_llm_judge_twins and not only:
        names += LLM_JUDGE_TWINS
    metrics = [getattr(SplunkAOEvaluators, n) for n in names]
    # NOTE: this call REPLACES the evaluator set currently enabled on the stream.
    enable_evaluators(project_name=settings.splunk_ao_project,
                      agent_stream_name=settings.splunk_ao_agent_stream, metrics=metrics)
    log.info("enabled %d evaluators on %s/%s", len(metrics), settings.splunk_ao_project,
             settings.splunk_ao_agent_stream)
    return names


# ----------------------------------------------------------------------------
# Runtime guardrail via Agent Control (optional)
# ----------------------------------------------------------------------------
_agent_control_ready = False


def init_agent_control() -> bool:
    """Initialise the Agent Control SDK against this app's Agent Stream. Returns True if active."""
    global _agent_control_ready
    if not settings.agent_control_enabled:
        return False
    try:
        # NOT exercised in this repo's tests (agent-control-sdk needs Python 3.12+ and a
        # control server). Verified against splunk-ao 0.4.0's helpers; re-check the
        # how-to at agent-observability-docs.splunk.com/how-to-guides/agent-control.
        import agent_control
        from splunk_ao import get_agent_control_target, setup_agent_control_bridge, splunk_ao_context

        splunk_ao_context.init(project=settings.splunk_ao_project, agent_stream=settings.splunk_ao_agent_stream)
        target = get_agent_control_target(target_type="log_stream")  # resolves the Agent Stream id from context
        agent_control.init(agent_name=settings.app_name, server_url=os.environ["AGENT_CONTROL_URL"],
                           target_type=target.target_type, target_id=target.target_id)
        setup_agent_control_bridge(splunk_ao_context.get_logger_instance())  # control spans -> Agent Stream
        _agent_control_ready = True
        log.info("Agent Control initialised")
    except Exception as exc:
        log.warning("Agent Control not active: %s", exc)
        _agent_control_ready = False
    return _agent_control_ready


def guard_with_agent_control(fn: Callable[..., str]) -> Callable[..., str]:
    """Wrap the LLM call with Agent Control's @control() when it is active, else return fn unchanged.

    The documented `@control()` example wraps an `async def`; this demo's chat
    turn is synchronous (FastAPI runs it in a worker thread), so if the decorated
    call returns an awaitable we drive it to completion here.
    """
    if not _agent_control_ready:
        return fn
    import asyncio
    import inspect

    from agent_control import control

    controlled = control()(fn)

    def _call(*args, **kwargs):
        result = controlled(*args, **kwargs)
        if inspect.isawaitable(result):
            return asyncio.run(result)
        return result

    return _call


class AgentControlViolation(Exception):
    """Re-export so callers do not import agent_control directly."""


def violation_types() -> tuple[type[BaseException], ...]:
    try:
        from agent_control import ControlViolationError

        return (ControlViolationError,)
    except Exception:
        return tuple()


# ----------------------------------------------------------------------------
# Legacy Protect REST helper (deprecated API; kept for older tenants)
# ----------------------------------------------------------------------------
def protect_invoke(text: str, *, stage_name: str, metric: str = "prompt_injection", operator: str = "gt",
                   target_value: float = 0.5, override_text: str = "Sorry, I cannot help with that.",
                   timeout: float = 5.0) -> dict[str, Any]:
    import httpx

    api_url = os.environ.get("SPLUNK_AO_API_URL") or os.environ.get("SPLUNK_AO_CONSOLE_URL")
    if not api_url:
        raise RuntimeError("SPLUNK_AO_API_URL / SPLUNK_AO_CONSOLE_URL not set")
    payload = {
        "payload": {"input": text},
        "project_name": settings.splunk_ao_project,
        "stage_name": stage_name,
        "prioritized_rulesets": [{
            "rules": [{"metric": metric, "operator": operator, "target_value": target_value}],
            "action": {"type": "OVERRIDE", "choices": [override_text]},
        }],
        "timeout": timeout,
    }
    headers = {"Splunk-AO-API-Key": os.environ["SPLUNK_AO_API_KEY"], "Content-Type": "application/json"}
    r = httpx.post(f"{api_url.rstrip('/')}/v2/protect/invoke", json=payload, headers=headers, timeout=timeout + 2)
    r.raise_for_status()
    return r.json()
