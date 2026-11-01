#!/usr/bin/env bash
# Runs on the Mac: deploy/push-env.sh <public-ip> [ssh-key]
# Copies .env to the server without the settings that only make sense on this Mac:
# the Cisco proxy certificate paths, and offline model loading (the server has no models yet).
set -euo pipefail
IP="${1:?usage: push-env.sh <public-ip> [ssh-key]}"
KEY="${2:-$HOME/.ssh/chat-llm-key.pem}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

grep -vE '^(SSL_CERT_FILE|REQUESTS_CA_BUNDLE|TRANSFORMERS_OFFLINE)=' .env \
  | sed -E 's/^HF_HUB_OFFLINE=.*/HF_HUB_OFFLINE=0/' \
  | ssh -i "$KEY" "ubuntu@$IP" 'umask 077; cat > ~/splunk-chat-LLM/.env'
echo "Copied .env to $IP"
