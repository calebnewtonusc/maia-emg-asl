#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtualenv if present
if [[ -d ".venv" ]]; then
    source .venv/bin/activate
elif [[ -d "venv" ]]; then
    source venv/bin/activate
fi

# Generate baseline ONNX model if none exists
ONNX_PATH="${ONNX_MODEL_PATH:-models/asl_emg_classifier.onnx}"
if [[ ! -f "$ONNX_PATH" ]]; then
    echo "No ONNX model found — generating synthetic baseline..."
    python scripts/train_lstm_baseline.py --export-onnx
fi

export ONNX_MODEL_PATH="$ONNX_PATH"
export MAIA_DISABLE_AUTH="${MAIA_DISABLE_AUTH:-true}"

PORT="${PORT:-8000}"
echo "Starting MAIA inference server on port $PORT..."
uvicorn src.api.main:app --host 0.0.0.0 --port "$PORT" --workers 1
