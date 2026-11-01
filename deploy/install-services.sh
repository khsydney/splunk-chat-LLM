#!/usr/bin/env bash
# Runs as root. Installs the systemd units and the plain-HTTP nginx site. Nothing starts until first-start.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
install -m 644 systemd/chat-llm-*.service /etc/systemd/system/
systemctl daemon-reload
mkdir -p /var/www/certbot
install -m 644 nginx-http.conf /etc/nginx/sites-available/chat-llm
ln -sf /etc/nginx/sites-available/chat-llm /etc/nginx/sites-enabled/chat-llm
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx
