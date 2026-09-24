"""Thin wrapper around TypeSafe's System One API (Jev) with an offline mock.

Real mode uses the official `typesafe-sdk` (`pip install typesafe-sdk`):
    POST https://api.typesafe.ai/v1/systemone  {model, state, questions}
    -> {model, answers: {id: NoulAnswer|ChoiceAnswer|ScoreAnswer}, usage}

Mock mode (MOCK_JEV=1) returns keyword-driven probabilities so the whole
pipeline — traces, guardrail decisions, Langfuse scores — can be exercised
without an API key. Mock answers are labelled `model="mock-jev"` so they can
never be mistaken for real Jev output in a dashboard.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .config import settings

log = logging.getLogger("demo.jev")


@dataclass
class JevResult:
    model: str
    answers: dict[str, dict[str, Any]]  # normalised: {"type","noul"|"choice"|"score",...}
    usage: dict[str, int]
    latency_ms: float
    request_id: str | None = None
    mock: bool = False
    raw: Any = field(default=None, repr=False)

    # convenience accessors ------------------------------------------------
    def p(self, question_id: str) -> float:
        """Probability for a Noul question."""
        return float(self.answers[question_id]["noul"])

    def score(self, question_id: str) -> float:
        return float(self.answers[question_id]["score"])

    def choice(self, question_id: str) -> str:
        return str(self.answers[question_id]["choice"])

    def confidence(self, question_id: str) -> float | None:
        return self.answers[question_id].get("confidence")


class JevClient:
    def __init__(self, *, model: str | None = None, mock: bool | None = None) -> None:
        self.model = model or settings.jev_model
        self.mock = settings.mock_jev if mock is None else mock
        self._client = None
        if not self.mock:
            from typesafe_sdk import TypeSafeClient  # reads TYPESAFE_API_KEY

            self._client = TypeSafeClient(model=self.model, timeout=10)

    # ------------------------------------------------------------------
    def evaluate(self, state: dict | str | list, questions: dict[str, dict]) -> JevResult:
        started = time.perf_counter()
        if self.mock:
            result = _mock_answers(state, questions)
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result

        response = self._client.system_one(state=state, questions=_to_sdk_questions(questions))
        latency_ms = (time.perf_counter() - started) * 1000
        answers = {qid: ans.model_dump() for qid, ans in response.answers.items()}
        return JevResult(
            model=response.model,
            answers=answers,
            usage=response.usage.model_dump() if hasattr(response.usage, "model_dump") else dict(response.usage),
            latency_ms=latency_ms,
            request_id=getattr(response, "request_id", None),
            raw=response,
        )


def _to_sdk_questions(questions: dict[str, dict]) -> dict:
    """Convert plain-dict questions (the JSON wire format) into SDK helper objects."""
    from typesafe_sdk import Choice, Noul, Score

    out = {}
    for qid, q in questions.items():
        kind = q["type"]
        if kind == "noul":
            out[qid] = Noul(instructions=q.get("instructions"), criteria=q.get("criteria"))
        elif kind == "choice":
            out[qid] = Choice(instructions=q.get("instructions"), criteria=q["criteria"])
        elif kind == "score":
            out[qid] = Score(instructions=q.get("instructions"), criteria=q["criteria"])
        else:
            raise ValueError(f"unknown question type {kind!r} for {qid}")
    return out


# ----------------------------------------------------------------------
# Mock: keyword heuristics, deliberately crude, clearly labelled
# ----------------------------------------------------------------------
_INJECTION = re.compile(r"ignore (all |your |the )?(previous|prior|above) instructions|system prompt|reveal|api key|password|jailbreak|developer mode", re.I)
_HARM = re.compile(r"\b(bomb|weapon|kill|hack into|malware|ransomware)\b", re.I)
_TOXIC = re.compile(r"\b(idiot|stupid|useless|hate you|shut up|moron)\b", re.I)
_PII = re.compile(r"\b[STFG]\d{7}[A-Z]\b|\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b|\b\+?\d{2,3}[ -]?\d{4}[ -]?\d{4}\b|[\w.+-]+@[\w-]+\.[\w.]+", re.I)
_TOPIC = re.compile(r"splunk|observab|apm|rum|otel|opentelemetry|collector|trace|metric|log|monitor|kubernetes|agent|luna|galileo|dashboard|alert", re.I)
_OFFTOPIC = re.compile(r"\b(poem|recipe|homework|joke|song|weather|football|stock tip)\b", re.I)
_DONT_KNOW = re.compile(r"(don't|do not|doesn't) (know|have|contain)|not (in|covered by|part of) (the )?(provided )?context|check the official docs", re.I)


def _mock_answers(state: Any, questions: dict[str, dict]) -> JevResult:
    text = str(state)
    user = str(state.get("user_message", "")) if isinstance(state, dict) else text
    answer = str(state.get("assistant_answer", "")) if isinstance(state, dict) else ""
    context = str(state.get("retrieved_context", "")) if isinstance(state, dict) else ""

    injection = 0.96 if _INJECTION.search(user) else 0.03
    harm = 0.92 if _HARM.search(user) else 0.02
    toxic = 0.90 if _TOXIC.search(user) else 0.02
    pii_in = 0.94 if _PII.search(user) else 0.02
    off_topic = 0.88 if (_OFFTOPIC.search(user) and not _TOPIC.search(user)) else (0.12 if _TOPIC.search(user) else 0.55)
    worst = max(injection, harm, toxic)
    severity_probs = _severity_distribution(worst, off_topic, pii_in)

    # post-response heuristics
    if _DONT_KNOW.search(answer):
        faithful = 0.93
    else:
        # crude "grounding": share of answer sentences with ≥3 words also present in context
        faithful = _overlap_faithfulness(answer, context) if answer else 0.5
    answers_q = 0.85 if answer and not _DONT_KNOW.search(answer) else (0.45 if answer else 0.1)
    pii_out = 0.90 if _PII.search(answer) else 0.02
    leaks = 0.90 if re.search(r"system prompt|instructions are|api key|sk-[a-z0-9]{6,}", answer, re.I) else 0.03
    help_probs = _helpfulness_distribution(faithful, answers_q)
    tone_probs = {"neutral": 0.72, "friendly": 0.2, "curt": 0.05, "other": 0.03}

    catalogue = {
        "prompt_injection": {"type": "noul", "noul": injection},
        "harmful_request": {"type": "noul", "noul": harm},
        "toxic_input": {"type": "noul", "noul": toxic},
        "pii_in_input": {"type": "noul", "noul": pii_in},
        "off_topic": {"type": "noul", "noul": off_topic},
        "severity": _score_answer(severity_probs, [q for q in questions.get("severity", {}).get("criteria", [])]),
        "faithful": {"type": "noul", "noul": faithful},
        "answers_question": {"type": "noul", "noul": answers_q},
        "pii_in_output": {"type": "noul", "noul": pii_out},
        "leaks_instructions": {"type": "noul", "noul": leaks},
        "helpfulness": _score_answer(help_probs, [q for q in questions.get("helpfulness", {}).get("criteria", [])]),
        "answer_tone": _choice_answer(tone_probs),
    }
    answers = {qid: catalogue.get(qid, {"type": q["type"], "noul": 0.5}) for qid, q in questions.items()}
    return JevResult(model="mock-jev", answers=answers, usage={"input_tokens": len(text) // 4, "output_tokens": 0},
                     latency_ms=0.0, request_id="mock", mock=True)


def _overlap_faithfulness(answer: str, context: str) -> float:
    ctx = set(re.findall(r"[a-z0-9]{4,}", context.lower()))
    words = re.findall(r"[a-z0-9]{4,}", answer.lower())
    if not words or not ctx:
        return 0.5
    inside = sum(1 for w in words if w in ctx) / len(words)
    return round(min(0.98, max(0.04, inside * 1.15)), 3)


def _severity_distribution(worst: float, off_topic: float, pii: float) -> dict[int, float]:
    if worst > 0.8:
        return {0: 0.02, 1: 0.05, 2: 0.33, 3: 0.60}
    if pii > 0.8:
        return {0: 0.05, 1: 0.25, 2: 0.62, 3: 0.08}
    if off_topic > 0.8:
        return {0: 0.15, 1: 0.75, 2: 0.08, 3: 0.02}
    return {0: 0.90, 1: 0.08, 2: 0.015, 3: 0.005}


def _helpfulness_distribution(faithful: float, answers_q: float) -> dict[int, float]:
    if faithful > 0.8 and answers_q > 0.7:
        return {0: 0.04, 1: 0.16, 2: 0.80}
    if faithful < 0.4:
        return {0: 0.55, 1: 0.35, 2: 0.10}
    return {0: 0.15, 1: 0.60, 2: 0.25}


def _score_answer(probs: dict[int, float], legend: list[str]) -> dict:
    expected = sum(i * p for i, p in probs.items())
    n = len(probs)
    peak = max(probs.values())
    confidence = (n * peak - 1) / (n - 1) if n > 1 else 1.0  # TypeSafe's published formula
    return {"type": "score", "score": round(expected, 3), "confidence": round(confidence, 3),
            "probabilities": probs, "legend": {i: (legend[i] if i < len(legend) else str(i)) for i in probs}}


def _choice_answer(probs: dict[str, float]) -> dict:
    n = len(probs)
    best = max(probs, key=probs.get)
    confidence = (n * probs[best] - 1) / (n - 1) if n > 1 else 1.0
    return {"type": "choice", "choice": best, "confidence": round(confidence, 3), "probabilities": probs}
