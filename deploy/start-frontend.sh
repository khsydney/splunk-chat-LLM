#!/usr/bin/env bash
# nginx serves this publicly; Streamlit itself only listens on loopback.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; source .env; set +a
export API_BASE="${API_BASE:-http://localhost:8000}" PATH="$PWD/.venv/bin:$PATH"
exec opentelemetry-instrument streamlit run streamlit_app.py --server.port 8501 --server.address 127.0.0.1 \
  --server.headless true --browser.gatherUsageStats false --server.fileWatcherType none
