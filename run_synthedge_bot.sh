#!/usr/bin/env bash
# SynthEdge Alert Bot launcher.

set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

source .venv/bin/activate

if [ -z "${SYNTHDATA_API_KEY:-}" ]; then
    echo "ERROR: SYNTHDATA_API_KEY is not set."
    echo "Run: export SYNTHDATA_API_KEY='your_key_here'"
    exit 1
fi

CONFIG_PATH="${1:-configs/bot.json}"

echo "=== SynthEdge Alert Bot ==="
echo "Python: $(which python)"
echo "Config: ${CONFIG_PATH}"
echo ""

exec python src/bot.py --config "${CONFIG_PATH}"
