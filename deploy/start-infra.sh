#!/usr/bin/env bash
# Containers for the server. Same set as `./run.sh infra`, without MinIO and with every port on loopback.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DOCKER_VOLUME_DIRECTORY="$HOME/chat-llm-data"
mkdir -p "$DOCKER_VOLUME_DIRECTORY"

docker compose -f "$ROOT/deploy/milvus-compose.yaml" up -d
docker compose -f "$ROOT/docker/redis/docker-compose.yaml" up -d

start() { # name, then docker run arguments
  local name="$1"; shift
  if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then docker start "$name" >/dev/null
  else docker run -d --name "$name" --restart unless-stopped "$@" >/dev/null; fi
}
start otelcol -p 127.0.0.1:4317:4317 -p 127.0.0.1:4318:4318 \
  -v "$ROOT/collector.yaml":/etc/otelcol/config.yaml \
  otel/opentelemetry-collector-contrib:latest --config /etc/otelcol/config.yaml
start attu -p 127.0.0.1:8001:3000 --network milvus zilliz/attu:latest

for _ in $(seq 1 60); do
  curl -sf http://localhost:9091/healthz >/dev/null && { echo "Milvus healthy"; exit 0; }
  sleep 3
done
echo "Milvus did not become healthy: docker logs milvus-standalone" >&2
exit 1
