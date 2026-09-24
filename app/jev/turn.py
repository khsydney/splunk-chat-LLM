"""One traced chat turn around the existing streaming RAG pipeline.

    POST /chat (FastAPI server span, from opentelemetry-instrument)
    └── chat-turn                     Langfuse root observation; OpenInference CHAIN root I/O for Splunk AO
        ├── jev-input-guard  [guardrail]   Jev battery #1 → block / redirect / review / pass  (+ scores)
        ├── rag-pipeline     [workflow]    the app's existing util-genai spans: retrieval → rerank → agent → LLM
        └── jev-response-eval [evaluator]  Jev battery #2 → faithfulness / PII / leak / helpfulness / tone (+ scores)

The RAG pipeline is untouched except for an optional `capture` dict through
which it hands back the reranked documents (needed as the evaluation context).
"""
from __future__ import annotations

import logging
from typing import AsyncGenerator, Callable

from . import evals, guard
from .config import settings
from .telemetry import current_span, langfuse, session_scope, set_openinference_io

log = logging.getLogger("jev.turn")

RagStream = Callable[..., AsyncGenerator[str, None]]


async def traced_stream(question: str, session_id: str, rag_stream: RagStream, *,
                        user_id: str = "anonymous") -> AsyncGenerator[str, None]:
    """Yield the answer tokens exactly like `rag_stream`, with guard + evals around it."""
    lf = langfuse()
    with session_scope(session_id, user_id, tags=["jev-vs-luna"]):
        with lf.start_as_current_observation(
            as_type="span", name="chat-turn",
            input={"user_message": question},
            metadata={"session_id": session_id, "user_id": user_id, "app": settings.app_name,
                      "jev": "on" if settings.jev_enabled else "off"},
        ) as root:
            root_span = current_span()
            guard_action = "disabled"

            # 1. Pre-response guardrail (Jev) ------------------------------------------
            if settings.jev_enabled and settings.jev_guard_enabled:
                try:
                    decision = guard.guard_input(question)
                except Exception as exc:  # fail OPEN: a judge outage must not take the chatbot down
                    log.warning("jev guard failed open: %s", exc)
                    decision = None
                if decision is not None:
                    guard_action = decision.action
                    if decision.blocked:
                        root.update(output={"answer": decision.response, "guard": decision.action},
                                    metadata={"guard_reason": decision.reason}, level="WARNING")
                        set_openinference_io(root_span, kind="CHAIN", input_value=question,
                                             output_value=decision.response, **{"guard.action": decision.action})
                        yield decision.response
                        return

            # 2. The existing RAG pipeline, streamed through unchanged --------------------
            capture: dict = {}
            buf: list[str] = []
            async for chunk in rag_stream(question, session_id=session_id, capture=capture):
                text = chunk if isinstance(chunk, str) else str(chunk)
                buf.append(text)
                yield text
            answer = "".join(buf)
            context = evals.build_context(capture.get("docs"), fallback=capture.get("context", ""))

            # 3. Post-response evaluation (Jev) ------------------------------------------
            flagged: list[str] = []
            if settings.jev_enabled and settings.jev_eval_mode == "sync" and answer.strip():
                try:
                    outcome = evals.evaluate_turn(question, context, answer)
                    flagged = outcome.flagged
                    if settings.jev_post_block and outcome.low_faithfulness:
                        # tokens are already on the wire; the honest option is to append a note
                        yield evals.LOW_FAITHFULNESS_NOTE
                        guard_action = f"{guard_action}+post-note"
                except Exception as exc:
                    log.warning("jev eval failed (answer already delivered): %s", exc)

            # 4. Close out the root observation for both backends -------------------------
            root.update(output={"answer": answer},
                        metadata={"retrieved_context": context, "docs_retrieved": len(capture.get("docs") or []),
                                  "flagged": flagged, "guard_action": guard_action},
                        level="WARNING" if flagged else "DEFAULT")
            set_openinference_io(root_span, kind="CHAIN", input_value=question, output_value=answer,
                                 **{"guard.action": guard_action, "eval.flagged": ",".join(flagged) or "-"})
