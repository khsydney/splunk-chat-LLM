#!/usr/bin/env bash
# One-command launcher for the Chat-LLM (Galileo) stack.
#
#   ./run.sh            start containers + backend + frontend (Ctrl+C stops the apps)
#   ./run.sh infra      start only the Docker containers
#   ./run.sh app        start only backend + frontend (containers already running)
#   ./run.sh index      (re)build the Milvus index from data/docs
#   ./run.sh status     show container / app status
#   ./run.sh stop       stop backend, frontend and all containers
#
# Optional env overrides: VENV=/path/to/venv  BACKEND_PORT=8000  FRONTEND_PORT=8501
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-8501}"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

MILVUS_COMPOSE="docker/milvus-compose.yaml"
REDIS_COMPOSE="docker/redis/docker-compose.yaml"

c_blue=$'\033[1;34m'; c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_red=$'\033[1;31m'; c_off=$'\033[0m'
info() { echo "${c_blue}==>${c_off} $*"; }
ok()   { echo "${c_green}✔${c_off}  $*"; }
warn() { echo "${c_yellow}!${c_off}  $*"; }
die()  { echo "${c_red}✘${c_off}  $*" >&2; exit 1; }

port_in_use() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

# ---------------------------------------------------------------- prerequisites
check_docker() {
  command -v docker >/dev/null || die "Docker is not installed."
  docker info >/dev/null 2>&1 || die "Docker daemon is not running. Start Docker Desktop and retry."
}

load_env() {
  [[ -f .env ]] || die ".env not found. Copy .env.example to .env and fill in your keys."
  set -a; # shellcheck disable=SC1091
  source .env
  set +a
}

activate_venv() {
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then return; fi
  local candidate
  for candidate in "${VENV:-}" "$ROOT/.venv" "$ROOT/venv"; do
    if [[ -n "$candidate" && -f "$candidate/bin/activate" ]]; then
      # shellcheck disable=SC1091
      source "$candidate/bin/activate"
      ok "Using virtualenv $candidate"
      return
    fi
  done
  die "No Python virtualenv active. Activate yours first, or set VENV=/path/to/venv."
}

# ---------------------------------------------------------------- containers
# Start (or create) a single named container started with `docker run`.
run_container() {
  local name="$1" port="$2"; shift 2
  if docker ps --format '{{.Names}}' | grep -qx "$name"; then
    ok "$name already running"
  elif port_in_use "$port"; then
    warn "Port $port is already in use by another process — skipping $name."
  elif docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
    docker start "$name" >/dev/null && ok "$name started"
  else
    docker run -d --name "$name" "$@" >/dev/null && ok "$name created"
  fi
}

start_infra() {
  check_docker
  info "Starting OpenTelemetry Collector (ports 4317/4318)"
  run_container otelcol 4317 \
    -p 4317:4317 -p 4318:4318 \
    -v "$ROOT/collector.yaml":/etc/otelcol/config.yaml \
    otel/opentelemetry-collector-contrib:latest \
    --config /etc/otelcol/config.yaml

  info "Starting Milvus (etcd + MinIO + standalone)"
  docker compose -f "$MILVUS_COMPOSE" up -d

  info "Starting Redis + Redis Insight"
  docker compose -f "$REDIS_COMPOSE" up -d

  info "Starting Attu (Milvus Web UI on :8001)"
  run_container attu 8001 -p 8001:3000 --network milvus zilliz/attu:latest

  wait_for_milvus
}

wait_for_milvus() {
  info "Waiting for Milvus to become healthy (first start can take ~90s)"
  local i
  for i in $(seq 1 60); do
    if curl -sf http://localhost:9091/healthz >/dev/null 2>&1; then
      ok "Milvus is healthy"; return
    fi
    sleep 3
  done
  die "Milvus did not become healthy. Check: docker logs milvus-standalone"
}

stop_infra() {
  check_docker
  info "Stopping containers"
  docker stop otelcol attu >/dev/null 2>&1 || true
  docker compose -f "$REDIS_COMPOSE" stop
  docker compose -f "$MILVUS_COMPOSE" stop
  ok "Containers stopped (data is kept)"
}

# ---------------------------------------------------------------- apps
PIDS=()

stop_apps() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  # Also catch instances started by a previous ./run.sh
  pkill -f "uvicorn app.main:app" 2>/dev/null || true
  pkill -f "streamlit run streamlit_app.py" 2>/dev/null || true
}

start_apps() {
  load_env
  activate_venv
  command -v opentelemetry-instrument >/dev/null || die "opentelemetry-instrument not found. Run: pip install -r requirements.txt"

  port_in_use "$BACKEND_PORT"  && die "Port $BACKEND_PORT is in use. Run ./run.sh stop or set BACKEND_PORT."
  port_in_use "$FRONTEND_PORT" && die "Port $FRONTEND_PORT is in use. Run ./run.sh stop or set FRONTEND_PORT."

  # No --reload: the reload child process would not re-initialize the OTel exporters.
  info "Starting backend (FastAPI) on :$BACKEND_PORT  → logs/backend.log"
  OTEL_SERVICE_NAME="${OTEL_SERVICE_NAME:-chat-rag}" \
    opentelemetry-instrument uvicorn app.main:app --host 0.0.0.0 --port "$BACKEND_PORT" \
    >"$LOG_DIR/backend.log" 2>&1 &
  PIDS+=($!)

  info "Starting frontend (Streamlit) on :$FRONTEND_PORT  → logs/frontend.log"
  API_BASE="${API_BASE:-http://localhost:$BACKEND_PORT}" \
    opentelemetry-instrument streamlit run streamlit_app.py \
    --server.port "$FRONTEND_PORT" --server.headless true \
    >"$LOG_DIR/frontend.log" 2>&1 &
  PIDS+=($!)

  trap 'echo; info "Shutting down backend and frontend"; stop_apps; exit 0' INT TERM

  info "Waiting for backend"
  local i
  for i in $(seq 1 60); do
    curl -sf "http://localhost:$BACKEND_PORT/docs" >/dev/null 2>&1 && break
    kill -0 "${PIDS[0]}" 2>/dev/null || { tail -n 30 "$LOG_DIR/backend.log"; die "Backend exited. See logs/backend.log"; }
    sleep 2
  done

  print_urls
  echo "Press Ctrl+C to stop the backend and frontend (containers keep running)."
  echo
  tail -n 0 -F "$LOG_DIR/backend.log" "$LOG_DIR/frontend.log" &
  PIDS+=($!)
  wait "${PIDS[0]}" || true
  warn "Backend exited — see logs/backend.log"
  stop_apps
}

build_index() {
  load_env
  activate_venv
  info "Building Milvus index from data/docs"
  python -m index.indexer
}

print_urls() {
  echo
  ok "Everything is up:"
  echo "   Streamlit UI      → http://localhost:$FRONTEND_PORT"
  echo "   FastAPI backend   → http://localhost:$BACKEND_PORT/docs"
  local attu_port
  attu_port="$(docker port attu 3000/tcp 2>/dev/null | head -n1 | sed 's/.*://')"
  echo "   Attu (Milvus UI)  → http://localhost:${attu_port:-8001}  (connect to milvus-standalone:19530)"
  echo "   Redis Insight     → http://localhost:5540"
  echo "   MinIO console     → http://localhost:9001"
  echo
}

status() {
  check_docker
  docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' \
    | grep -E 'NAMES|otelcol|attu|milvus-|rag_redis' || true
  echo
  port_in_use "$BACKEND_PORT"  && ok "Backend listening on :$BACKEND_PORT"   || warn "Backend not running"
  port_in_use "$FRONTEND_PORT" && ok "Frontend listening on :$FRONTEND_PORT" || warn "Frontend not running"
}

case "${1:-up}" in
  up)     start_infra; start_apps ;;
  infra)  start_infra ;;
  app)    start_apps ;;
  index)  build_index ;;
  status) status ;;
  stop)   stop_apps; stop_infra ;;
  *)      sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
