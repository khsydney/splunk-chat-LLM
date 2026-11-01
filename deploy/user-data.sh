#!/usr/bin/env bash
# EC2 user data for Ubuntu 24.04. Replace the two __PLACEHOLDERS__ before launching.
# The SSH key is injected here because the cisco-runon role can't import key pairs.
set -euxo pipefail
exec > >(tee -a /var/log/chat-llm-setup.log) 2>&1

SSH_PUBLIC_KEY="__SSH_PUBLIC_KEY__"
BRANCH="__BRANCH__"
REPO_URL="https://github.com/khsydney/splunk-chat-LLM.git"

install -d -m 700 -o ubuntu -g ubuntu /home/ubuntu/.ssh
echo "$SSH_PUBLIC_KEY" >> /home/ubuntu/.ssh/authorized_keys
chown ubuntu:ubuntu /home/ubuntu/.ssh/authorized_keys
chmod 600 /home/ubuntu/.ssh/authorized_keys

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y git curl nginx docker.io docker-compose-v2 python3-venv python3-pip \
  libmagic1 poppler-utils tesseract-ocr libgl1 libglib2.0-0t64
snap wait system seed.loaded
snap install --classic certbot
ln -sf /snap/bin/certbot /usr/local/bin/certbot

# Rotate container logs: the collector's debug exporter is noisy.
cat > /etc/docker/daemon.json <<'JSON'
{ "log-driver": "json-file", "log-opts": { "max-size": "50m", "max-file": "3" } }
JSON
systemctl restart docker
systemctl enable docker
usermod -aG docker ubuntu

sudo -u ubuntu git clone -b "$BRANCH" "$REPO_URL" /home/ubuntu/splunk-chat-LLM
sudo -u ubuntu bash /home/ubuntu/splunk-chat-LLM/deploy/setup-server.sh
bash /home/ubuntu/splunk-chat-LLM/deploy/install-services.sh

touch /var/lib/chat-llm-setup.done
