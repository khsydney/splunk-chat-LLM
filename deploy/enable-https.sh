#!/usr/bin/env bash
# Runs on the server: sudo deploy/enable-https.sh <public-ip>
# Browsers only allow the microphone on HTTPS pages, so voice input needs this.
# Let's Encrypt IP certificates last ~6 days and renew automatically through the certbot timer.
set -euo pipefail
IP="${1:?usage: enable-https.sh <public-ip>}"
cd "$(dirname "${BASH_SOURCE[0]}")"

certbot certonly --webroot -w /var/www/certbot --ip-address "$IP" --required-profile shortlived \
  --agree-tos --register-unsafely-without-email --non-interactive --deploy-hook 'systemctl reload nginx'

R="/etc/letsencrypt/renewal/$IP.conf"
grep -q '^renew_before_expiry' "$R" || sed -i '1a renew_before_expiry = 2 days' "$R"

sed "s/__IP__/$IP/g" nginx-https.conf > /etc/nginx/sites-available/chat-llm
nginx -t
systemctl reload nginx
echo "HTTPS enabled: https://$IP"
