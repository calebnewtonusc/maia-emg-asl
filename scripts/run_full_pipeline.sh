#!/usr/bin/env bash
# run_full_pipeline.sh -- One command to go from zero to running server
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
step() { echo -e "\n${BLUE}--- Step $1: $2 ${NC}"; }
ok()   { echo -e "${GREEN}[OK] $*${NC}"; }
warn() { echo -e "${YELLOW}[WARN] $*${NC}"; }

# --- Step 1: Dependencies ---
step 1 "Dependencies"
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}ERROR: python3 not found${NC}"; exit 1
fi
PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)"; then
    ok "Python $PYTHON_VER"
else
    warn "Python $PYTHON_VER detected; 3.11+ recommended"
fi

if [[ ! -d ".venv" ]]; then
    echo "Creating virtualenv..."
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -q -r requirements.txt
ok "Dependencies installed"

# --- Step 2: Train baseline model ---
step 2 "Baseline model training"
ONNX_PATH="models/asl_emg_classifier.onnx"
if [[ -f "$ONNX_PATH" ]]; then
    ok "Model already exists: $ONNX_PATH"
else
    echo "Training LSTM baseline on synthetic data (CPU, ~2 min)..."
    python scripts/train_lstm_baseline.py --epochs 100
    ok "Model trained: $ONNX_PATH"
fi

# --- Step 3: Run tests ---
step 3 "Test suite"
if python -m pytest tests/ -q --tb=short 2>&1; then
    ok "All tests passed"
else
    warn "Some tests failed -- check output above"
fi

# --- Step 4: Start server ---
step 4 "Start inference server"
export ONNX_MODEL_PATH="$ONNX_PATH"
export MAIA_DISABLE_AUTH=true
PORT="${PORT:-8000}"
echo "Starting server on port $PORT..."
echo "  REST:      http://localhost:$PORT/health"
echo "  WebSocket: ws://localhost:$PORT/ws/emg"
echo "  Docs:      http://localhost:$PORT/docs"
echo ""
echo "Press Ctrl+C to stop"
uvicorn src.api.main:app --host 0.0.0.0 --port "$PORT" --reload
