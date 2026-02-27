# Railway Deployment Guide — MAIA EMG-ASL

The MAIA inference server runs on Railway with automatic deploys triggered by every push to `main` via native GitHub integration.

**Live server:** `https://maia-emg-asl-production.up.railway.app`

---

## Architecture

```
IYA Nvidia Lab  (GPU training)
    │
    │  SLURM job completes
    │  upload_artifact.py ──────────► Cloudflare R2
    │                                  (model storage · free 10GB · no egress fees)
    │
    ▼
GitHub  (calebnewtonusc/maia-emg-asl)
    │
    │  git push origin main
    │  Railway GitHub webhook fires
    │
    ▼
Railway  (inference server)
    │  Docker build (python:3.11-slim)
    │  ONNX model bundled in image  OR  downloaded from R2 at startup
    │  FastAPI + WebSocket on $PORT
    │
    ▼
iPhone App  (React Native)
    On-device ONNX (primary, < 15ms) ──► Railway WebSocket (fallback, ~50ms)
```

---

## Current Setup (Already Done)

The project is live. Every push to `main` auto-deploys. If you're re-reading this for reference:

```bash
curl https://maia-emg-asl-production.up.railway.app/health
# → {"status": "ok", "model_loaded": true}
```

Environment variables currently set on Railway:

| Variable | Value | Notes |
|----------|-------|-------|
| `MAIA_DISABLE_AUTH` | `true` | Dev mode — no API key required |
| `ONNX_MODEL_PATH` | `models/asl_emg_classifier.onnx` | Bundled in Docker image |
| `LOG_LEVEL` | `INFO` | structlog level |
| `ORT_NUM_THREADS` | `2` | ONNX Runtime threads |
| `PORT` | auto | Set by Railway |

---

## Fresh Project Setup (If Starting Over)

### Step 1: Create Railway project from GitHub

1. [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
2. Select `calebnewtonusc/maia-emg-asl`
3. Railway detects the `Dockerfile` automatically
4. **Variables** tab → **Raw Editor** → paste:

```
MAIA_DISABLE_AUTH=true
ONNX_MODEL_PATH=models/asl_emg_classifier.onnx
LOG_LEVEL=INFO
ORT_NUM_THREADS=2
```

5. **Settings → Networking** → **Generate Domain**

Done. Railway deploys on every push to `main` from this point forward.

### Step 2: Verify

```bash
# Health check
curl https://your-domain.up.railway.app/health
# → {"status": "ok", "model_loaded": true}

# Info (auth disabled in dev)
curl https://your-domain.up.railway.app/info

# WebSocket smoke test
python scripts/test_websocket.py --url wss://your-domain.up.railway.app/ws/emg
```

### Step 3: Update mobile app

Edit `mobile/react-native/.env`:
```bash
EXPO_PUBLIC_SERVER_URL=https://your-domain.up.railway.app
EXPO_PUBLIC_RAILWAY_URL=https://your-domain.up.railway.app
EXPO_PUBLIC_FALLBACK_TO_SERVER=true
```

---

## Cloudflare R2 Model Storage (Production)

Once real GPU-trained models exist, store them on R2 instead of bundling in the Docker image.

### R2 Setup

1. [dash.cloudflare.com](https://dash.cloudflare.com) → **R2 Object Storage** → **Create bucket**
2. Name: `maia-emg-asl`
3. Enable **Public access** (for Railway to pull at startup)
4. **Manage R2** → **Create API Token** → Object Read & Write
5. Note: Account ID, Access Key ID, Secret Access Key

```bash
# Add to Railway Variables:
R2_ACCOUNT_ID=your_account_id
R2_ACCESS_KEY_ID=your_access_key
R2_SECRET_ACCESS_KEY=your_secret_key
R2_BUCKET_NAME=maia-emg-asl
R2_PUBLIC_URL=https://pub-XXXX.r2.dev
R2_MODEL_URL=https://pub-XXXX.r2.dev/latest/asl_emg_classifier.onnx
```

### Upload a model

```bash
# After GPU training completes:
python scripts/upload_artifact.py \
    --file models/asl_emg_classifier.onnx \
    --set-latest
# → Uploads to R2 and marks as latest
# → Railway picks it up on next deploy (or restart)
```

### Switching from bundled to R2

When the R2 URL is set, comment out `ONNX_MODEL_PATH` in Railway variables and set `R2_MODEL_URL` instead. The server checks `R2_MODEL_URL` first, falls back to `ONNX_MODEL_PATH`.

---

## Production Auth Setup

When you're ready to lock down the API:

```bash
# Railway Variables → update:
MAIA_DISABLE_AUTH=false
MAIA_API_KEYS=your-prod-key-here,optional-second-key
```

All endpoints except `/health` then require the header:
```
X-API-Key: your-prod-key-here
```

Update the mobile app:
```bash
# mobile/react-native/.env
EXPO_PUBLIC_API_KEY=your-prod-key-here
```

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Auto | — | Set by Railway — do not override |
| `MAIA_DISABLE_AUTH` | No | `false` | Set `true` for development |
| `MAIA_API_KEYS` | Prod only | — | Comma-separated valid API keys |
| `ONNX_MODEL_PATH` | No | `models/asl_emg_classifier.onnx` | Local model path (bundled) |
| `R2_MODEL_URL` | No | — | R2 public URL — takes precedence over local path |
| `R2_ACCOUNT_ID` | If R2 | — | Cloudflare account ID |
| `R2_ACCESS_KEY_ID` | If R2 | — | R2 Access Key ID |
| `R2_SECRET_ACCESS_KEY` | If R2 | — | R2 Secret Access Key |
| `R2_BUCKET_NAME` | If R2 | `maia-emg-asl` | R2 bucket name |
| `LOG_LEVEL` | No | `INFO` | structlog level (DEBUG/INFO/WARNING) |
| `ORT_NUM_THREADS` | No | `2` | ONNX Runtime CPU thread count |

---

## Dockerfile Overview

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE $PORT
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "$PORT"]
```

Railway sets `$PORT` automatically. The `railway.toml` configures the healthcheck:
```toml
[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 30
```

---

## Cost

| Service | Free Tier | Notes |
|---------|-----------|-------|
| Railway | $5 credit/month (Hobby) | Easily covers low-traffic inference |
| Cloudflare R2 | 10 GB storage · 10M reads/month | No egress fees — ideal for model weights |
| IYA Nvidia Lab | Free (USC) | GPU training via SLURM |

---

## Troubleshooting

**`model_loaded: false` after deploy**
- Check Railway build logs: model ONNX wasn't bundled or R2 URL is wrong
- Verify: `curl https://maia-emg-asl-production.up.railway.app/health`
- If using R2: `curl $R2_MODEL_URL -o /dev/null -w "%{http_code}"` must return 200

**WebSocket won't connect from app**
- App uses `wss://` in production — `serverConfig.ts` auto-converts `https://` → `wss://`
- Test manually: `npx wscat -c wss://maia-emg-asl-production.up.railway.app/ws/emg`
- Check Railway logs: `railway logs --tail`

**Auth 401 errors**
- Set `MAIA_DISABLE_AUTH=true` in Railway Variables for dev
- Or send header: `X-API-Key: your-key`

**Deploy not triggering on push**
- Confirm Railway GitHub integration is connected: **Project Settings → Source Repo**
- Check Railway dashboard for failed builds
- `railway logs` to see build output

**Out of memory on Railway**
- Reduce `ORT_NUM_THREADS` to `1`
- Upgrade Railway plan if needed (Conformer model uses more RAM than LSTM)
