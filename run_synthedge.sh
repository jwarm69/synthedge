#!/usr/bin/env bash
# SynthEdge Dashboard Launcher
# Activates .venv (which has xgboost, lightgbm, catboost, joblib)
# so the local ensemble loads correctly.

set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

source .venv/bin/activate

export SYNTHDATA_API_KEY="${SYNTHDATA_API_KEY:-a5ec1862b1113d668519652d1901e19f8cdbf172eecff624}"

echo "=== SynthEdge Dashboard ==="
echo "Python: $(which python)"
echo "Streamlit: $(which streamlit)"
echo "Port: 8503"
echo ""

exec streamlit run src/dashboard_v2.py --server.port 8503
