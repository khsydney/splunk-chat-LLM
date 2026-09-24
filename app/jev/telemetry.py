"""Attach Langfuse and Splunk Agent Observability to the TracerProvider this app ALREADY has.

The app is started with `opentelemetry-instrument` (Splunk distro), which owns the
global TracerProvider and exports to the local OpenTelemetry Collector →
Observability Cloud (APM). We do not touch that. We add two more span processors
to the same provider:

    existing provider ──┬─► OTLP → collector → Observability Cloud APM     (unchanged)
                        ├─► LangfuseSpanProcessor → Langfuse             (Jev evals + scores live here)
                        └─► SplunkAOSpanProcessor → Splunk Agent Observability  (Luna-2 evaluators here)

so every span — the FastAPI server span, the util-genai Workflow/Retrieval/Agent
spans, the LangChain/OpenAI spans, and the new Jev guardrail/evaluator spans —
carries ONE trace id into all three backends.

Conversation identity:
* Langfuse reads `session.id` / `user.id` (stamped on every span by `propagate_attributes`).
* Splunk AO reads `gen_ai.conversation.id`, which `SplunkAOSpanProcessor.on_start` stamps
  from the execution-local session context. The util-genai Workflow already sets the same
  attribute on its own spans (`conversation_id=session_id`), so the two agree.

Message content: both backends parse `gen_ai.input.messages` / `gen_ai.output.messages`
on LLM spans. The util-genai handler only writes them when
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` — set it, or the judges see no prompts.
"""
from __future__ import annotations

import contextlib
import json
import logging
from typing import Iterator

from opentelemetry import trace
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from .config import settings

log = logging.getLogger("jev.telemetry")

_langfuse = None
_splunk_processor = None
_attached_to = None


class _NoopExporter(SpanExporter):
    def export(self, spans):  # noqa: D401
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def attach(tracer_provider, *, langfuse_span_exporter: SpanExporter | None = None,
           splunk_exporter: SpanExporter | None = None) -> None:
    """Register both vendor processors on an existing SDK TracerProvider. Idempotent.

    The keyword arguments exist for tests (in-memory exporters, no network).
    """
    global _langfuse, _splunk_processor, _attached_to
    if _attached_to is tracer_provider:
        return
    if not hasattr(tracer_provider, "add_span_processor"):
        log.warning("tracer provider %r has no add_span_processor — is opentelemetry-instrument active?",
                    type(tracer_provider).__name__)
        return

    # ---- Splunk Agent Observability ----------------------------------------
    if settings.splunk_ao_enabled or splunk_exporter is not None:
        try:
            from splunk_ao.otel import SplunkAOSpanProcessor

            if splunk_exporter is not None:
                _splunk_processor = SplunkAOSpanProcessor(_exporter=splunk_exporter)
            else:
                _splunk_processor = SplunkAOSpanProcessor(project=settings.splunk_ao_project,
                                                          agentstream=settings.splunk_ao_agent_stream)
            tracer_provider.add_span_processor(_splunk_processor)
            log.info("Splunk AO export attached (project=%s, agent_stream=%s)",
                     settings.splunk_ao_project, settings.splunk_ao_agent_stream)
        except Exception as exc:
            log.warning("Splunk AO export disabled: %s", exc)
    else:
        log.info("Splunk AO export off (no SPLUNK_AO_* credentials)")

    # ---- Langfuse ------------------------------------------------------------
    from langfuse import Langfuse

    if settings.langfuse_enabled and langfuse_span_exporter is None:
        _langfuse = Langfuse(tracer_provider=tracer_provider, environment=settings.environment,
                             blocked_instrumentation_scopes=list(settings.langfuse_blocked_scopes) or None)
        log.info("Langfuse export attached")
    else:
        # No credentials (or a test exporter): keep the Langfuse observation wrappers
        # recording real OTel spans — so the guardrail/evaluator spans still reach the
        # collector and Splunk AO — but export them nowhere and drop scores.
        _langfuse = Langfuse(tracer_provider=tracer_provider, environment=settings.environment,
                             public_key="pk-lf-disabled", secret_key="sk-lf-disabled",
                             base_url="http://127.0.0.1:9",
                             span_exporter=langfuse_span_exporter or _NoopExporter())
        if langfuse_span_exporter is None:
            _langfuse.create_score = lambda **_kw: None  # type: ignore[method-assign]
            log.info("Langfuse export off (no LANGFUSE_* credentials); spans still flow to other backends")
    _attached_to = tracer_provider


def langfuse():
    if _langfuse is None:
        raise RuntimeError("jev.telemetry.attach() has not been called (startup hook missing?)")
    return _langfuse


@contextlib.contextmanager
def session_scope(session_id: str, user_id: str, tags: list[str] | None = None) -> Iterator[None]:
    """Propagate conversation identity to BOTH backends for everything inside."""
    from langfuse import propagate_attributes

    try:
        from splunk_ao.session_context import get_session_selection, restore_session_selection, set_session_context
    except Exception:
        get_session_selection = restore_session_selection = set_session_context = None  # type: ignore

    previous = get_session_selection() if get_session_selection else None
    if set_session_context:
        set_session_context(session_id)
    try:
        with propagate_attributes(session_id=session_id, user_id=user_id, tags=tags or []):
            yield
    finally:
        if restore_session_selection:
            restore_session_selection(previous)


def set_openinference_io(span, *, kind: str | None = None, input_value=None, output_value=None, **attrs) -> None:
    """Stamp OpenInference attributes on an OTel span so Splunk AO sees typed I/O.

    Splunk AO's safety evaluators (Prompt Injection, Toxicity, PII, Sexism, Tone) run on
    the trace-root input/output; the Langfuse wrappers use `langfuse.*` attributes that
    Splunk ignores, so we add the OpenInference ones explicitly.
    """
    if kind:
        span.set_attribute("openinference.span.kind", kind)
    if input_value is not None:
        span.set_attribute("input.value", input_value if isinstance(input_value, str) else _json(input_value))
        span.set_attribute("input.mime_type", "text/plain" if isinstance(input_value, str) else "application/json")
    if output_value is not None:
        span.set_attribute("output.value", output_value if isinstance(output_value, str) else _json(output_value))
        span.set_attribute("output.mime_type", "text/plain" if isinstance(output_value, str) else "application/json")
    for key, value in attrs.items():
        span.set_attribute(key, value)


def flush() -> None:
    if _langfuse is not None:
        with contextlib.suppress(Exception):
            _langfuse.flush()
    if _splunk_processor is not None:
        with contextlib.suppress(Exception):
            _splunk_processor.force_flush()


def current_span():
    return trace.get_current_span()


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
