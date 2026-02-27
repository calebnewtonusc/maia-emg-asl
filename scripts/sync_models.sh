#!/usr/bin/env bash
# sync_models.sh — Sync trained models between R2 and local
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; RED='\033[0;31m'; NC='\033[0m'

UPLOAD=false
STATUS_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --upload)      UPLOAD=true; shift ;;
        --status)      STATUS_ONLY=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--upload] [--status]"
            echo "  (no flags)  Download latest models from R2"
            echo "  --upload    Upload local .onnx models to R2"
            echo "  --status    Show R2 artifact list without downloading"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Check R2 credentials
if [[ -z "${R2_ACCOUNT_ID:-}" || -z "${R2_ACCESS_KEY_ID:-}" || -z "${R2_SECRET_ACCESS_KEY:-}" ]]; then
    echo -e "${RED}[ERROR]${NC} R2 credentials not set. Export R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY"
    exit 1
fi

if $STATUS_ONLY; then
    echo -e "${BLUE}[INFO]${NC} Listing R2 artifacts..."
    python scripts/download_artifact.py --list
    exit 0
fi

if $UPLOAD; then
    echo -e "${BLUE}[INFO]${NC} Uploading local ONNX models to R2..."
    for f in models/*.onnx; do
        [[ -f "$f" ]] || continue
        echo -e "${BLUE}[INFO]${NC} Uploading $f..."
        python scripts/upload_artifact.py --file "$f" --set-latest
    done
    echo -e "${GREEN}[OK]${NC} Upload complete"
else
    echo -e "${BLUE}[INFO]${NC} Downloading latest models from R2..."
    python scripts/download_artifact.py --all-latest --dest models/
    echo -e "${GREEN}[OK]${NC} Models synced to models/"
fi
