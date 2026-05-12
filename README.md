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

## 📌 Notes

**Attu UI** → http://localhost:8001

**Redis Insight UI** → http://localhost:5540

**FastAPI Backend** → http://localhost:8000

**Streamlit UI** → http://localhost:8501


