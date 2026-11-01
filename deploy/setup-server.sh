#!/usr/bin/env bash
# Runs as ubuntu. Installs the exact package versions tested on the Mac (Python 3.12).
# --no-deps reproduces that set as-is; torch comes from the CPU index to skip the CUDA packages.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip

pin() { sed -n "s/^$1==//p" requirements.lock.txt; }
.venv/bin/pip install --no-deps --index-url https://download.pytorch.org/whl/cpu \
  "torch==$(pin torch)+cpu" "torchvision==$(pin torchvision)+cpu"
# milvus has no Linux build and openai-whisper needs triton; the app imports neither.
grep -vE '^(torch|torchvision|milvus|openai-whisper)==' requirements.lock.txt > /tmp/requirements.notorch.txt
.venv/bin/pip install --no-deps -r /tmp/requirements.notorch.txt

.venv/bin/python -c "import torch, galileo, agent_control, streamlit, sentence_transformers, pymilvus, langchain_milvus, opentelemetry.util.genai.handler; print('imports ok; torch', torch.__version__)"
