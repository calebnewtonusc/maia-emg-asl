#!/usr/bin/env bash
# deploy_railway.sh — One-command Railway deployment for MAIA EMG-ASL
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# --------------------------------------------------------------------------
# Parse args
# --------------------------------------------------------------------------
SETUP=false
ENV_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --setup) SETUP=true; shift ;;
        --env)   ENV_FILE="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--setup] [--env <file>]"
            echo "  --setup    Initialize Railway project and set env vars"
            echo "  --env FILE Load env vars from file (default: railway.env.example)"
            exit 0 ;;
        *) error "Unknown argument: $1" ;;
    esac
done

# --------------------------------------------------------------------------
# Prerequisites
# --------------------------------------------------------------------------
if ! command -v railway &>/dev/null; then
    error "Railway CLI not found. Install with: npm install -g @railway/cli"
fi
if ! railway whoami &>/dev/null; then
    error "Not logged in to Railway. Run: railway login"
fi

cd "$PROJECT_ROOT"

# --------------------------------------------------------------------------
# Setup mode — initialize project and set environment variables
# --------------------------------------------------------------------------
if $SETUP; then
    info "Setting up Railway project..."
    railway init --name maia-emg-asl 2>/dev/null || warn "Project may already exist, continuing..."

    ENV_FILE="${ENV_FILE:-railway.env.example}"
    if [[ ! -f "$ENV_FILE" ]]; then
        warn "Env file not found: $ENV_FILE — skipping env var setup"
    else
        info "Loading env vars from $ENV_FILE..."
        while IFS= read -r line; do
            # Skip comments and empty lines
            [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
            key="${line%%=*}"
            val="${line#*=}"
            # Skip placeholder values
            [[ "$val" == *"your-"* || "$val" == *"XXXX"* || -z "$val" ]] && continue
            info "  Setting $key"
            railway variables set "$key=$val"
        done < "$ENV_FILE"
        success "Environment variables configured"
    fi
    success "Setup complete! Now run: $0 to deploy"
    exit 0
fi

# --------------------------------------------------------------------------
# Deploy
# --------------------------------------------------------------------------
info "Deploying MAIA EMG-ASL to Railway..."
railway up --detach

# --------------------------------------------------------------------------
# Wait for health check
# --------------------------------------------------------------------------
info "Waiting for deployment to become healthy..."

RAILWAY_URL=$(railway status --json 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
for svc in data.get('services', []):
    for env in svc.get('environments', {}).values():
        url = env.get('url') or env.get('serviceUrl', '')
        if url:
            print(url)
            break
" 2>/dev/null || echo "")

if [[ -z "$RAILWAY_URL" ]]; then
    warn "Could not auto-detect Railway URL. Check the dashboard for your deployment URL."
    success "Deployment triggered! Monitor at: https://railway.app/dashboard"
    exit 0
fi

# Normalize to https
RAILWAY_URL="${RAILWAY_URL#wss://}"
RAILWAY_URL="https://${RAILWAY_URL#https://}"

info "Polling $RAILWAY_URL/health..."
MAX_ATTEMPTS=15
for i in $(seq 1 $MAX_ATTEMPTS); do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$RAILWAY_URL/health" 2>/dev/null || echo "000")
    if [[ "$STATUS" == "200" ]]; then
        success "Server is healthy!"
        break
    fi
    warn "Attempt $i/$MAX_ATTEMPTS — HTTP $STATUS — waiting 20s..."
    sleep 20
done

WSS_URL="wss://${RAILWAY_URL#https://}/ws/emg"

echo ""
success "=== MAIA EMG-ASL deployed to Railway ==="
echo -e "  ${GREEN}REST API:${NC}   $RAILWAY_URL"
echo -e "  ${GREEN}WebSocket:${NC}  $WSS_URL"
echo -e "  ${GREEN}Health:${NC}     $RAILWAY_URL/health"
echo -e "  ${GREEN}Info:${NC}       $RAILWAY_URL/info"
echo ""
info "Update your mobile app .env:"
echo "  EXPO_PUBLIC_SERVER_URL=$RAILWAY_URL"
echo "  EXPO_PUBLIC_RAILWAY_URL=$RAILWAY_URL"
