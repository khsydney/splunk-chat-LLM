# Chat Application with RAG, OpenAI and Galileo Integration

<img width="100%" alt="architecture diagram" src="https://github.com/user-attachments/assets/aa83730f-da05-48a4-87d7-327612111d6a" />

This branch (**Chat-LLM-Galileo**) contains the **high-level architecture and setup** for a **chat application** that leverages **RAG (Retrieval-Augmented Generation)** with **OpenAI**, with **Galileo** logging and **Agent Control** guardrails added on top of the Splunk observability stack.

It integrates with:
- **Milvus (Vector Database)** for embeddings storage
- **Redis** for caching and chat history
- **Attu (Web UI for Milvus)**
- **Redis Insight (Web UI for Redis)**
- **OpenTelemetry Collector** for full observability into LLM-powered workflows
- **Galileo** for LLM trace logging and evaluation metrics
- **Galileo Agent Control** for runtime input/output guardrails

---

## 📖 Table of Contents
1. [Quick Start (one command)](#-quick-start-one-command)
2. [Setup](#-setup)
   - [Create Virtual Environment](#1-create-and-activate-a-python-virtual-environment)
   - [Install Dependencies](#2-install-dependencies)
   - [Configure .env](#3-configure-the-env-file)
   - [Run Required Docker Containers](#4-run-required-docker-containers)
3. [Run the Application](#️-run-the-application)
   - [Backend (FastAPI + Uvicorn)](#backend-fastapi-with-uvicorn)
   - [Frontend (Streamlit)](#frontend-streamlit-ui)
   - [Build the Vector Index](#build-the-vector-index)
4. [run.sh Command Reference](#-runsh-command-reference)
5. [Observability](#-observability)
6. [Galileo & Agent Control](#-galileo--agent-control)
7. [Voice & Image (Multimodal)](#-voice--image-multimodal)
8. [Components](#-components)
9. [Troubleshooting](#-troubleshooting)
10. [Notes](#-notes)

---

## ⚡ Quick Start (one command)

Prerequisites: **Docker Desktop** running, **Python 3.11+**, an **OpenAI API key**, and a **Galileo API key**.

```bash
git clone -b Chat-LLM-Galileo https://github.com/khsydney/splunk-chat-LLM.git
cd splunk-chat-LLM

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then fill in OPENAI_API_KEY, GALILEO_API_KEY, ...

./run.sh                    # starts all containers + backend + frontend
```

`./run.sh` will:
1. Start the **OpenTelemetry Collector**, **Milvus** (etcd + MinIO + standalone), **Redis + Redis Insight** and **Attu**
2. Wait until Milvus is healthy
3. Load `.env`, activate the virtualenv (`.venv`, `venv`, or `$VENV`)
4. Start the **FastAPI backend** and **Streamlit frontend** with `opentelemetry-instrument`
5. Print all URLs and stream both logs to the terminal

Press **Ctrl+C** to stop the backend and frontend (containers keep running). Use `./run.sh stop` to stop everything.

> First run only: build the vector index from the PDFs in `data/docs` with `./run.sh index`.

---

## 🚀 Setup

The steps below are what `./run.sh` does for you. Use them if you prefer to start each piece by hand.

### 1. Create and activate a Python virtual environment
```bash
python3 -m venv <venv_name>
source <your_full_path_to>/<venv_name>/bin/activate
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Configure the .env file
Copy the template and fill in the values marked `<...>`:
```bash
cp .env.example .env
```
Key settings:
```bash
OPENAI_API_KEY=<OpenAI API key>
GALILEO_API_KEY=<Galileo API key>
GALILEO_PROJECT=<Galileo project name — must already exist in Galileo>
GALILEO_LOG_STREAM=production
AGENT_CONTROL_URL=<Agent Control server URL>
```
The Splunk Observability Cloud ingest token and endpoints are configured in `collector.yaml` (`exporters.otlphttp/splunk`).

### 4. Run required Docker containers
All of these are started by `./run.sh infra`.
#### A) OpenTelemetry Collector
```bash
docker run -d --name otelcol \
  -p 4317:4317 -p 4318:4318 \
  -v "$PWD/collector.yaml":/etc/otelcol/config.yaml \
  otel/opentelemetry-collector-contrib:latest \
  --config /etc/otelcol/config.yaml
```
#### B) Milvus
```bash
docker compose -f docker/milvus-compose.yaml up -d
```
#### C) Attu (Web UI for Milvus Vector DB)
```bash
docker run -d -p 8001:3000 --network milvus --name attu zilliz/attu:latest
```
#### D) Redis & Redis Insight (Web UI for Redis)
```bash
docker compose -f docker/redis/docker-compose.yaml up -d
```

## ▶️ Run the Application

We use `opentelemetry-instrument` to auto-instrument the Python app for observability. Both commands are started by `./run.sh app`.

Load the environment first (run from the project root):
```bash
set -a && source .env && set +a
```

### Backend (FastAPI with Uvicorn)
```Bash
OTEL_SERVICE_NAME=chat-rag opentelemetry-instrument uvicorn app.main:app --host 0.0.0.0 --port 8000
```
> **Do not use `--reload`.** Uvicorn's reloader runs the app in a child process where the OTel exporters are not re-initialized, so no traces reach Splunk.

### Frontend (Streamlit UI)
```Bash
opentelemetry-instrument streamlit run streamlit_app.py --server.port 8501
```

### Build the Vector Index
Embeds every document in `data/docs` and loads it into the Milvus `rag_chunks` collection. Run once, and again whenever you add documents (you can also upload docs from the Streamlit sidebar).
```bash
./run.sh index        # or: python -m index.indexer
```

## 🛠 run.sh Command Reference

| Command | What it does |
|---|---|
| `./run.sh` | Start all containers, then backend + frontend (Ctrl+C stops the apps) |
| `./run.sh infra` | Start only the Docker containers |
| `./run.sh app` | Start only backend + frontend (containers already running) |
| `./run.sh index` | (Re)build the Milvus index from `data/docs` |
| `./run.sh status` | Show container and app status |
| `./run.sh stop` | Stop backend, frontend and all containers (data is kept) |

Optional overrides: `VENV=/path/to/venv`, `BACKEND_PORT=8000`, `FRONTEND_PORT=8501`, `OTEL_ALT_GRPC_PORT=14317`, `OTEL_ALT_HTTP_PORT=14318`.

If ports 4317/4318 are already taken by another OpenTelemetry Collector, `run.sh` starts this app's collector on 14317/14318 and sets `OTEL_EXPORTER_OTLP_ENDPOINT` for the backend and frontend so telemetry still flows through `collector.yaml`. `./run.sh stop` only stops the processes this script started.
Logs are written to `logs/backend.log` and `logs/frontend.log`.

## 📊 Observability

This application is instrumented with OpenTelemetry.

All traces, metrics, and logs are collected by the OpenTelemetry Collector.

Data can be exported to Splunk Observability Cloud or other compatible backends.

This enables:

- Tracing RAG workflows (retrieval, reranking, LLM responses)
- Monitoring Milvus queries and Redis cache performance
- LLM evaluation scoring observability (DeepEval via the Splunk GenAI SDK)

## 🛡 Galileo & Agent Control

**Galileo logging** — every chat turn is logged to Galileo with `GalileoLogger` (project `GALILEO_PROJECT`, log stream `GALILEO_LOG_STREAM`). Each Streamlit session maps to a Galileo session, and the backend enables Galileo's built-in metrics on the log stream at startup.

**Agent Control** — at startup the backend registers the `rag-pipeline` agent with the Agent Control server (`AGENT_CONTROL_URL`, authenticated with `GALILEO_API_KEY`). Controls configured in Galileo are enforced around the pipeline:
- a **deny** control blocks the request (`ControlViolationError`)
- a **steer** control returns guidance instead of the LLM answer (`ControlSteerError`)

Galileo runs alongside Splunk — OpenTelemetry traces still go to Splunk through the collector.

## 🎤 Voice & Image (Multimodal)

The chat accepts a **voice recording** and/or an **image** in addition to typed text, and logs them to Galileo as multimodal content — the audio shows up as an inline waveform player and the image renders inline in the Log stream, so you can inspect exactly what the model received (see [Galileo multimodal observability](https://docs.galileo.ai/concepts/logging/multimodal-observability)).

**How to use it:** in the Streamlit UI open the **"🎤 Voice & 🖼️ image (multimodal)"** panel above the chat box, record a question with the microphone and/or attach a `png`/`jpg`, optionally type some text, then click **Send voice / image**.

**How a multimodal turn flows**

```
Streamlit  ──multipart──▶  FastAPI  POST /chat/multimodal
                               │
                               ├─ start_trace(input=[Text, Data(audio), Data(image)])   ← attachments logged here
                               ├─ speech-to-text   (tool span,  whisper-1)      transcript  → becomes the question
                               ├─ image-understanding (LLM span, gpt-4o-mini)  description → appended to the question
                               ├─ milvus-raw → bge-reranker → milvus-reranked → prompt-augmentation   (unchanged)
                               └─ openai-chat (LLM span) → conclude → flush
```

| Input | Accepted formats | What happens to it |
|---|---|---|
| Voice | `wav` (Streamlit recorder), `mp3` | Transcribed with `TRANSCRIBE_MODEL`; the transcript is used as the question and echoed back to the user (`🎤 You said: …`) |
| Image | `png`, `jpeg` | Described with `VISION_MODEL`; the description is added to the question so retrieval can search for it (`🖼️ Image understood as: …`) |
| Text | — | Optional; combined with either attachment |

**Scoring multimodal content in Galileo.** Out-of-the-box multimodal metrics (`interruption_detection`, `visual_quality`, `*_audio` variants) and custom LLM-as-a-judge metrics with the *Audio* or *Image/PDF* capability need a model integration first (Gemini for audio, GPT-5+ for images). Once configured in the console, enable them at startup with `GALILEO_EXTRA_METRICS=interruption_detection,visual_quality` in `.env`. They return N/A on traces without a matching attachment.

> Note: Galileo multimodal attachments are only supported via `GalileoLogger` (what this app uses), not via the OpenTelemetry path — so the Splunk/OTel trace for the same request stays text-only.

## 🧩 Components

**FastAPI** — Backend API

**Streamlit** — Frontend chat interface

**Milvus** — Vector Database for embeddings

**Redis** — Cache and chat history store

**Attu** — Web UI for Milvus

**Redis Insight** — Web UI for Redis

**OpenTelemetry Collector** — Observability pipeline

**Galileo** — LLM logging, metrics and Agent Control guardrails

## 🖼️ Screenshots / Demo

Streamlit UI (Chat Interface)
<img width="1512" height="823" alt="image" src="https://github.com/user-attachments/assets/4869e725-0be0-487f-9876-7f184c628623" />

Redis Insight
<img width="1512" height="823" alt="image" src="https://github.com/user-attachments/assets/f33e36cb-373f-48e6-af76-d42c44adb674" />
<img width="1512" height="823" alt="image" src="https://github.com/user-attachments/assets/176fcc7b-584b-409b-8e28-ab206936282a" />

Attu (Milvus Web UI)
<img width="1512" height="823" alt="image" src="https://github.com/user-attachments/assets/74724979-ecc6-428b-a9c4-664fa81cf130" />
<img width="1512" height="823" alt="image" src="https://github.com/user-attachments/assets/368076fd-83d4-4917-b465-a020d009a52a" />

## 🧯 Troubleshooting

| Symptom | Fix |
|---|---|
| `Port 4317 is used by another collector — running otelcol on 14317/14318` | Not an error. Another project's collector owns 4317, so this app's collector runs side by side on 14317/14318 and the backend/frontend are pointed at it automatically. Change with `OTEL_ALT_GRPC_PORT` / `OTEL_ALT_HTTP_PORT`. |
| `Milvus did not become healthy` | `docker logs milvus-standalone`; first start can take ~90s. |
| `Port 8000/8501 is in use` | `./run.sh stop`, or set `BACKEND_PORT` / `FRONTEND_PORT`. |
| `Project "<name>" not found` (Galileo) | Create the project in the Galileo UI first, or fix `GALILEO_PROJECT`. |
| `429 insufficient_quota` | The OpenAI account has no credits — add billing or use another key. |
| Chat answers are empty / no sources | Run `./run.sh index` to build the Milvus collection. |

## 📌 Notes

**Streamlit UI** → http://localhost:8501

**FastAPI Backend** → http://localhost:8000 (Swagger: http://localhost:8000/docs)

**Attu UI** → http://localhost:8001 (connect to `milvus-standalone:19530`)

**Redis Insight UI** → http://localhost:5540

**MinIO Console** → http://localhost:9001
