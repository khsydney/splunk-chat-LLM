"""Configuration for the Jev guardrail/evals and the dual observability export.

All values come from environment variables (see .env.example, section
"Jev vs Luna-2"). Nothing here changes how the existing RAG pipeline or the
existing OpenTelemetry Collector export behave.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Thresholds:
    """Decision thresholds for Jev probabilities.

    0.5 is NOT a safe default: Arize measured 76% accuracy at 0.5 vs 87% at 0.80
    for a faithfulness Noul on RAGTruth; TypeSafe's guardrail cookbook uses
    0.35 (review) / 0.70 (act). Fit these on a few hundred labelled turns.
    """

    block: float = float(os.getenv("JEV_BLOCK_THRESHOLD", "0.70"))
    review: float = float(os.getenv("JEV_REVIEW_THRESHOLD", "0.35"))
    severity_block: float = float(os.getenv("JEV_SEVERITY_BLOCK", "2.0"))  # Score 0..3
    faithful_min: float = float(os.getenv("JEV_FAITHFUL_MIN", "0.80"))
    pii_flag: float = float(os.getenv("JEV_PII_FLAG", "0.50"))


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("OTEL_SERVICE_NAME", "splunk-chat-llm")
    environment: str = os.getenv("APP_ENV", "demo")

    # what the assistant is for (used by the off-topic Noul); describe YOUR documents here
    kb_topics: str = os.getenv(
        "KB_TOPICS",
        "vendor technical proposals for SPF (Cisco Sanwat, Splunk AppDynamics, WatchDog Purplex: "
        "monitoring features, architecture, retention, security), the Monopoly board game rules, "
        "and Korean AI-industry reports (2024 AI industry survey, SPRI AI brief Dec 2023)",
    )

    # --- Jev / TypeSafe ---
    typesafe_api_key: str | None = os.getenv("TYPESAFE_API_KEY") or None
    mock_jev: bool = _flag("MOCK_JEV")
    jev_model: str = os.getenv("JEV_MODEL", "jev-1.13.0")  # pin; jev-latest can shift calibration silently
    jev_guard_enabled: bool = _flag("JEV_GUARD_ENABLED", "1")
    jev_eval_mode: str = os.getenv("JEV_EVAL_MODE", "sync")  # sync | off
    jev_post_block: bool = _flag("JEV_POST_BLOCK", "0")  # append a warning to low-faithfulness answers
    jev_eval_top_docs: int = int(os.getenv("JEV_EVAL_TOP_DOCS", "10"))  # reranked docs sent to Jev as context
    jev_context_max_chars: int = int(os.getenv("JEV_CONTEXT_MAX_CHARS", "40000"))  # ≈10k tokens, well under Jev's 32k
    thresholds: Thresholds = field(default_factory=Thresholds)

    @property
    def jev_enabled(self) -> bool:
        """Jev runs when a key is present or mock mode is on; otherwise guard/evals are skipped."""
        return self.mock_jev or bool(self.typesafe_api_key)

    # --- Langfuse ---
    langfuse_enabled: bool = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
    langfuse_blocked_scopes: tuple[str, ...] = tuple(
        s.strip() for s in os.getenv("LANGFUSE_BLOCKED_SCOPES", "").split(",") if s.strip()
    )

    # --- Splunk Agent Observability (formerly Galileo) ---
    splunk_ao_enabled: bool = bool(os.getenv("SPLUNK_AO_API_KEY") or os.getenv("SPLUNK_AO_O11Y_TOKEN"))
    splunk_ao_project: str = os.getenv("SPLUNK_AO_PROJECT", "jev-vs-luna")
    splunk_ao_agent_stream: str = os.getenv("SPLUNK_AO_AGENT_STREAM", "splunk-chat-llm")
    agent_control_enabled: bool = _flag("AGENT_CONTROL_ENABLED", "0")


settings = Settings()
