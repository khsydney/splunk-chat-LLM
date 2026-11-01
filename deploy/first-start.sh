#!/usr/bin/env bash
# Runs on the server after .env is in place: deploy/first-start.sh <public-ip>
set -euo pipefail
IP="${1:?usage: first-start.sh <public-ip>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -f .env ]] || { echo ".env is missing: run deploy/push-env.sh from the Mac" >&2; exit 1; }

sudo systemctl enable --now chat-llm-infra

set -a; source .env; set +a
count="$(.venv/bin/python - <<'PY'
from pymilvus import MilvusClient
c = MilvusClient(uri="http://localhost:19530")
print(c.get_collection_stats("rag_chunks")["row_count"] if c.has_collection("rag_chunks") else 0)
PY
)"
if [[ "$count" == "0" ]]; then
  echo "Building the index (the first run downloads the embedding model)"
  .venv/bin/python -m index.indexer 2>&1 | tail -3
else
  echo "Index already has $count rows"
fi

sudo systemctl enable --now chat-llm-backend chat-llm-frontend
sudo "$ROOT/deploy/enable-https.sh" "$IP"

echo "Waiting for the backend (it downloads the reranker on first start)"
for _ in $(seq 1 120); do curl -sf http://localhost:8000/health >/dev/null && break; sleep 5; done
# /health doesn't touch Milvus or OpenAI, so ask a real question.
curl -sN -m 240 -X POST http://localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"question":"how many dollars does each player start with in monopoly","session_id":"deploy-check"}' | head -c 300
echo
