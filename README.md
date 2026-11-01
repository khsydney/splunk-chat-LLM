# Chat Application with RAG, OpenAI and Galileo Integration

<img width="100%" alt="architecture diagram" src="https://github.com/user-attachments/assets/aa83730f-da05-48a4-87d7-327612111d6a" />

This branch (**Chat-LLM-Galileo-Multimodal**) contains the **high-level architecture and setup** for a **chat application** that leverages **RAG (Retrieval-Augmented Generation)** with **OpenAI**, with **Galileo** logging and **Agent Control** guardrails added on top of the Splunk observability stack. On top of typed questions it accepts **voice and image input**, and its Streamlit UI has a warm paper theme.

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
8. [Chat UI](#-chat-ui)
9. [Deploy on EC2](#-deploy-on-ec2)
10. [Components](#-components)
11. [Troubleshooting](#-troubleshooting)
12. [Notes](#-notes)

---

## ⚡ Quick Start (one command)

Prerequisites: **Docker Desktop** running, **Python 3.11+**, an **OpenAI API key**, and a **Galileo API key**.

```bash
git clone -b Chat-LLM-Galileo-Multimodal https://github.com/khsydney/splunk-chat-LLM.git
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

Optional settings:
```bash
TORCH_DEVICE=cpu            # device for the embedding model and reranker (default cpu, see Troubleshooting)
UI_USER_NAME=nick           # name shown in the UI greeting and sidebar
HF_HUB_OFFLINE=1            # skip Hugging Face Hub checks once the models are downloaded
```

**Behind a TLS-inspecting proxy** (for example Cisco Secure Access), Python can't verify OpenAI, Hugging Face or Galileo, because its `certifi` bundle doesn't include the proxy's root certificate. Point Python at a bundle that does. Homebrew's bundle includes the macOS keychain roots:
```bash
SSL_CERT_FILE=/opt/homebrew/etc/ca-certificates/cert.pem
REQUESTS_CA_BUNDLE=/opt/homebrew/etc/ca-certificates/cert.pem
```

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

**How to use it:** in the reply box at the bottom of the Streamlit UI, click **＋** to attach a `png`/`jpg` and/or the **microphone** to record a question, optionally type some text, then send. Text-only messages go to `POST /chat`; anything with an attachment goes to `POST /chat/multimodal`.

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

## 💬 Chat UI

`streamlit_app.py` has its own styling. The theme (paper background, clay accent) lives in `.streamlit/config.toml`; layout and fonts are CSS inside the app.

- **Sidebar** — **New chat**, a chat search box, and **Recents**. Each chat has its own backend `session_id`, so it keeps its own Redis memory and Galileo session. The **Knowledge base** section uploads documents to `data/docs` and rebuilds the index; **Settings** holds the backend URL, a backend check, a side-panel toggle and **Clear this chat**.
- **Conversation** — your messages appear as bubbles on the right, answers as serif text with the response time and model underneath. A pulsing indicator shows while a request is in flight. Bracketed citations in an answer are listed under it as **Sources**.
- **Welcome screen** — a greeting with suggested questions about the indexed documents.
- **Reply box** — text, **＋** for an image and a **microphone** for voice.
- **Side card** — lists the documents in `data/docs` and what the session uses (Milvus collection, Redis memory turns, chat model). Hidden on windows narrower than 1300px.

## ☁️ Deploy on EC2

The `deploy/` folder runs the app on one Ubuntu 24.04 instance. It differs from `./run.sh` on a laptop in these ways:

| | Laptop (`run.sh`) | EC2 (`deploy/`) |
|---|---|---|
| Milvus storage | MinIO (`docker/milvus-compose.yaml`) | Local disk (`deploy/milvus-compose.yaml`), because MinIO images are no longer published |
| Python packages | `requirements.txt` | `requirements.lock.txt`, the exact versions tested, with CPU-only torch |
| Processes | Foreground script | systemd units that restart on failure and start at boot |
| Public access | `localhost` only | nginx on 80/443. Every other port is bound to loopback |
| TLS | None | Let's Encrypt IP certificate. Browsers only allow the microphone on HTTPS pages |

**1. Launch the instance.** Recommended: `m7i.2xlarge` (8 vCPUs), 100 GB gp3 disk. Four people asking at once takes answers to about 60-75 s, against 10-14 s for one. Open 80 and 443 to your users, and 22 to your own IP. Port 80 must stay reachable by Let's Encrypt for certificate renewal. Use `deploy/user-data.sh` as user data, replacing `__SSH_PUBLIC_KEY__` and `__BRANCH__`. The branch must already be pushed, because the instance clones it from GitHub. Setup is finished when `/var/lib/chat-llm-setup.done` exists; its log is `/var/log/chat-llm-setup.log`.

**2. Copy the settings and start the app.** `push-env.sh` leaves out the Mac-only proxy certificate paths and turns off offline model loading.
```bash
deploy/push-env.sh <public-ip>
ssh -i ~/.ssh/chat-llm-key.pem ubuntu@<public-ip> '~/splunk-chat-LLM/deploy/first-start.sh <public-ip>'
```
`first-start.sh` starts the containers, builds the index if it's empty, starts both apps, enables HTTPS and asks a test question. Then open `https://<public-ip>`.

**Day to day:**
```bash
sudo systemctl status chat-llm-backend chat-llm-frontend chat-llm-infra
journalctl -u chat-llm-backend -f
git pull && sudo systemctl restart chat-llm-backend chat-llm-frontend    # deploy a change
```
Redis Insight and Attu are only on loopback, so reach them with an SSH tunnel, for example `ssh -L 8001:localhost:8001 ubuntu@<public-ip>`.

> **The app has no login.** Anyone who can reach the URL can spend your OpenAI quota and upload documents to the index, so limit port 443 in the security group where you can.
>
> **The IP is tied to the certificate.** Without an Elastic IP, stopping the instance changes its address. Run `sudo deploy/enable-https.sh <new-ip>` afterwards.

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
| `docker compose` can't pull `minio/minio` on a new machine | MinIO no longer publishes its images, so `./run.sh` only works where the image is already cached. Use `deploy/milvus-compose.yaml`, which stores Milvus data on local disk. |
| `[stream-error] <MilvusException: ... Fail connecting to server on localhost:19530>` | `milvus-standalone` exited, usually after the Mac slept: it loses its etcd lease (`requested lease not found` in `docker logs milvus-standalone`). The compose file restarts it automatically (`restart: unless-stopped`) and the backend reconnects and retries for ~20s, so the next question normally works. If it doesn't, run `docker start milvus-standalone`. |
| Backend dies with `failed assertion _status < MTLCommandBufferStatusCommitted` | PyTorch on the Mac GPU (MPS) can't be used by two threads at once. Keep `TORCH_DEVICE=cpu` (the default). |
| `CERTIFICATE_VERIFY_FAILED` in the backend log | A TLS-inspecting proxy is re-signing HTTPS. Set `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` (see [Configure .env](#3-configure-the-env-file)). |
| `ImportError: cannot import name 'httpx' from 'huggingface_hub.utils'` | `huggingface_hub` got downgraded below what `datasets` needs: `pip install "huggingface_hub>=1.31,<2"`. |
| `Port 8000/8501 is in use` | `./run.sh stop`, or set `BACKEND_PORT` / `FRONTEND_PORT`. |
| `Project "<name>" not found` (Galileo) | Create the project in the Galileo UI first, or fix `GALILEO_PROJECT`. If it's followed by `422 ... already exists`, the project exists but `GALILEO_API_KEY` belongs to a user who can't see it — use a key from the project's owner. |
| `401 Unauthorized` from Agent Control | The key isn't allowed to use Agent Control. The app logs it, skips the guardrails and keeps answering. |
| `429 insufficient_quota` | The OpenAI account has no credits — add billing or use another key. |
| Chat answers are empty / no sources | Run `./run.sh index` to build the Milvus collection. |
| `/health` is OK but answers fail | `/health` doesn't touch Milvus or OpenAI, and `/chat` reports failures as `[stream-error] ...` inside a 200 response. Ask a real question to check. |

## 📌 Notes

**Streamlit UI** → http://localhost:8501

**FastAPI Backend** → http://localhost:8000 (Swagger: http://localhost:8000/docs)

**Attu UI** → http://localhost:8001 (connect to `milvus-standalone:19530`). If an `attu` container already runs from another clone, use its port instead; `./run.sh` prints the real one.

**Redis Insight UI** → http://localhost:5540

**MinIO Console** → http://localhost:9001
