# Chat Application with RAG and OpenAI Integration

<img width="100%" alt="architecture diagram" src="https://github.com/user-attachments/assets/aa83730f-da05-48a4-87d7-327612111d6a" />

This repository contains the **high-level architecture and setup** for a **chat application** that leverages **RAG (Retrieval-Augmented Generation)** with **OpenAI**.  

It integrates with:
- **Milvus (Vector Database)** for embeddings storage
- **Redis** for caching and chat history
- **Attu (Web UI for Milvus)**
- **Redis Insight (Web UI for Redis)**
- **OpenTelemetry Collector** for full observability into LLM-powered workflows

---

## 📖 Table of Contents
1. [Setup](#-setup)
   - [Create Virtual Environment](#1-create-and-activate-a-python-virtual-environment)
   - [Install Dependencies](#2-install-dependencies)
   - [Run Required Docker Containers](#3-run-required-docker-containers)
2. [Run the Application](#-run-the-application)
   - [Backend (FastAPI + Uvicorn)](#backend-fastapi-with-uvicorn)
   - [Frontend (Streamlit)](#frontend-streamlit-ui)
3. [Observability](#-observability)
4. [Components](#-components)
5. [Next Steps](#-next-steps)
6. [Notes](#-notes)

---

## 🚀 Setup

### 1. Create and activate a Python virtual environment
```bash
python3 -m venv <venv_name>
source <your_full_path_to>/<venv_name>/bin/activate
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Edit .env File to configure OTEL backend API key
```bash
SPLUNK_API_KEY=<Splunk O11y Cloud Ingestion Key>
AppD_API_Key=<App Dynamics Api Key>
```
### 4. Run required Docker containers
#### A) OpenTelemetry Collector
```bash
docker run -d --name otelcol \
  -p 4317:4317 -p 4318:4318 \
  -v "$PWD/collector.yaml":/etc/otelcol/config.yaml \
  otel/opentelemetry-collector-contrib:latest \
  --config /etc/otelcol/config.yaml
```
#### B) Attu (Web UI for Milvus Vector DB)
```bash
docker run -d -p 8001:3000 --name attu zilliz/attu:latest
```
#### C) Milvus
```bash
docker compose -f docker/milvus-compose.yaml up -d
```
#### D) Redis & Redis Insight (Web UI for Redis)
```bash
docker compose -f docker/redis/docker-compose.yaml up -d
```

## ▶️ Run the Application

We use opentelemetry-instrument to auto-instrument the Python app for observability.

### Backend (FastAPI with Uvicorn)
```Bash
opentelemetry-instrument uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
### Frontend (Streamlit UI)
```Bash
opentelemetry-instrument streamlit run streamlit_app.py
```

## 📊 Observability

This application is instrumented with OpenTelemetry.

All traces, metrics, and logs are collected by the OpenTelemetry Collector.

Data can be exported to Splunk Observability Cloud or other compatible backends.

This enables:

Tracing RAG workflows (retrieval, reranking, LLM responses)

Monitoring Milvus queries and Redis cache performance

LLM evaluation scoring observability

## 🧩 Components

**FastAPI** — Backend API

**Streamlit** — Frontend chat interface

**Milvus** — Vector Database for embeddings

**Redis** — Cache and chat history store

**Attu** — Web UI for Milvus

**Redis Insight** — Web UI for Redis

**OpenTelemetry Collector** — Observability pipeline

## 🖼️ Screenshots / Demo

Streamlit UI (Chat Interface)
<img width="1512" height="823" alt="image" src="https://github.com/user-attachments/assets/4869e725-0be0-487f-9876-7f184c628623" />

Redis Insight
<img width="1512" height="823" alt="image" src="https://github.com/user-attachments/assets/f33e36cb-373f-48e6-af76-d42c44adb674" />
<img width="1512" height="823" alt="image" src="https://github.com/user-attachments/assets/176fcc7b-584b-409b-8e28-ab206936282a" />

Attu (Milvus Web UI)
<img width="1512" height="823" alt="image" src="https://github.com/user-attachments/assets/74724979-ecc6-428b-a9c4-664fa81cf130" />
<img width="1512" height="823" alt="image" src="https://github.com/user-attachments/assets/368076fd-83d4-4917-b465-a020d009a52a" />

## 🧪 Jev vs Luna-2: dual observability (Langfuse + Jev | Splunk Agent Observability + Luna-2)

Branch `Chat-LLM-Cleaned` adds `app/jev/`, which hangs **two more span processors** on the
TracerProvider that `opentelemetry-instrument` already creates, and wraps every `/chat` turn with a
**Jev** guardrail (before the LLM) and a **Jev** evaluation (after the last token). The existing
collector → Observability Cloud path is untouched, so one trace id lands in three places:

```
POST /chat (FastAPI span)
└── chat-turn                          ← Langfuse root observation; OpenInference root I/O for Splunk AO
    ├── jev-input-guard   [guardrail]  ← Jev: prompt_injection / harmful / toxic / pii / off_topic + severity Score
    ├── rag-pipeline      [workflow]   ← unchanged util-genai spans: retrieval → rerank → agent → LLM
    └── jev-response-eval [evaluator]  ← Jev: faithful / answers_question / pii / leaks / helpfulness Score / tone Choice
```

Langfuse shows the Jev probabilities as scores on those observations (and can run the same
questions server-side with its native *decision model evaluator*, spec in
`scripts/langfuse_decision_evaluator.json`). Splunk Agent Observability scores the same trace with
its **Luna-2** evaluators (Context Adherence, Prompt Injection, PII, Toxicity …) enabled on the Agent Stream.

### Run it

```bash
pip install -r requirements.txt                       # adds langfuse, typesafe-sdk, splunk-ao
cp .env.example .env                                  # fill in: TYPESAFE_API_KEY, LANGFUSE_*, SPLUNK_AO_* (see below)
MOCK_JEV=1 python -m pytest -q tests                  # offline check of the whole wiring (no keys needed)

# start exactly as before — collector, Milvus, Redis, then:
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true \
opentelemetry-instrument uvicorn app.main:app --host 0.0.0.0 --port 8000
opentelemetry-instrument streamlit run streamlit_app.py     # optional UI

python scripts/run_scenarios.py                       # 15 turns: benign, Korean, hallucination bait, injection, PII, toxic, off-topic, multi-turn
curl -s localhost:8000/health                         # {"ok":true,"jev":"on","langfuse":true,"splunk_ao":true}
```

Then open the **same trace id** in both consoles (filter on tag/session `jev-vs-luna`) and walk the
comparison checklist in the JEV project (`demo/compare-checklist.md`).

### What each key switches on

| `.env` | Effect when set | When empty |
|---|---|---|
| `TYPESAFE_API_KEY` | real Jev guard + evals (`JEV_MODEL=jev-1.13.0`, pinned) | guard/evals skipped (`MOCK_JEV=1` → heuristic stand-in for plumbing tests) |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` | Langfuse export + scores | Langfuse off; guardrail/evaluator spans still go to the collector and Splunk AO |
| `SPLUNK_AO_API_KEY` + `SPLUNK_AO_CONSOLE_URL` (on-prem) **or** `SPLUNK_AO_REALM` + `SPLUNK_AO_O11Y_TOKEN` (SaaS) | spans also exported to the Agent Stream `SPLUNK_AO_PROJECT/SPLUNK_AO_AGENT_STREAM`; then `python scripts/enable_luna_evaluators.py` (`--saas` on Observability Cloud) | Splunk AO off |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` | prompts/completions on the LLM spans — **required**, both judges read them | judges see no content |

Langfuse one-time UI setup for the zero-code path: *Settings → LLM Connections → typesafe* (paste the
TypeSafe key), then *Evaluators → New decision model evaluator* following `scripts/langfuse_decision_evaluator.json`.

### Notes and traps

* **Thresholds are fitted, not assumed** (`JEV_*_THRESHOLD`): Arize measured 76 % accuracy at Jev's 0.5 vs 87 % at 0.80 on
  RAGTruth faithfulness. Defaults follow TypeSafe's guardrail cookbook (0.35 review / 0.70 act). Label ~200 turns and re-fit.
* **Streaming**: the guard runs before the first token (~110–250 ms, from a US-hosted API); the evaluation runs after the
  last token, so it adds no user-visible latency. `JEV_POST_BLOCK=1` can only *append* a verification note — tokens are already on the wire.
* **Context size**: Jev gets the top `JEV_EVAL_TOP_DOCS` reranked passages (capped at `JEV_CONTEXT_MAX_CHARS`), not all 70 —
  Jev's state limit is 32k tokens and its docs say accuracy falls with irrelevant state.
* **Jev is steerable by adversarial content** (TypeSafe: an injected instruction "can move the answer") — test injections placed *inside* documents.
* **Fail open**: a Jev outage logs a warning and the chatbot keeps answering.
* **Tier gating**: Luna-2 is Enterprise-only; Observability Cloud SaaS exposes only Prompt Injection / Toxicity / Sexism / PII on Luna.
  Context Adherence on Luna needs an on-prem/standalone tenant.
* **Double billing**: identical spans are billed by Langfuse (units) and Splunk AO (spans / Luna tokens), plus Jev tokens.
* **Korean**: the two Korean reports are in the KB; Jev documents CJK as "supported but tested accuracy varies" — the `korean` scenario is there to measure that.
* `collector.yaml` should not contain a live ingest token in a shared repo — move it to an env var (`${SPLUNK_ACCESS_TOKEN}`) and rotate it.

## 📌 Notes

**Attu UI** → http://localhost:8001

**Redis Insight UI** → http://localhost:5540

**FastAPI Backend** → http://localhost:8000

**Streamlit UI** → http://localhost:8502


