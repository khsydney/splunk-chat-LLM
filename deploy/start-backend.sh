#!/usr/bin/env bash
# No --reload: its child process doesn't re-initialize the OTel exporters.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; source .env; set +a
export OTEL_SERVICE_NAME="${OTEL_SERVICE_NAME:-chat-rag}" PATH="$PWD/.venv/bin:$PATH"
exec opentelemetry-instrument uvicorn app.main:app --host 127.0.0.1 --port 8000
