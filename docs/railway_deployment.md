# Railway Deployment Guide — MAIA EMG-ASL

This guide deploys the MAIA inference server to Railway (free hobby tier, ~$0/month for light usage).

## Architecture

```
IYA Nvidia Lab (GPU training)
    ↓ SLURM job completes
    ↓ upload_artifact.py → Cloudflare R2 (model storage, free 10GB)
    ↓
Railway (inference server)
    ↓ Downloads ONNX from R2 at startup
    ↓ FastAPI + WebSocket
    ↓
iPhone App (React Native)
    On-device ONNX (primary) → Railway WebSocket (fallback)
```

## Prerequisites

| Tool | Install |
|------|---------|
| Railway CLI | `npm install -g @railway/cli` |
| Cloudflare account | cloudflare.com (free) |
| Trained ONNX model | `python scripts/train_lstm_baseline.py --export-onnx` |

## Step 1: Cloudflare R2 Setup

1. Go to [Cloudflare R2](https://dash.cloudflare.com) → R2 Object Storage → Create bucket
2. Name it `maia-emg-asl`
3. Enable **Public access** (or use custom domain)
4. Create API token: Manage R2 → Create API Token → Object Read & Write
5. Note your: Account ID, Access Key ID, Secret Access Key

```bash
export R2_ACCOUNT_ID=your_account_id
export R2_ACCESS_KEY_ID=your_access_key
export R2_SECRET_ACCESS_KEY=your_secret_key
export R2_BUCKET_NAME=maia-emg-asl
export R2_PUBLIC_URL=https://pub-XXXX.r2.dev
```

## Step 2: Upload Your Model

```bash
# Generate a baseline model if you don't have one yet
python scripts/train_lstm_baseline.py --export-onnx

# Upload to R2
python scripts/upload_artifact.py \
    --file models/asl_emg_classifier.onnx \
    --set-latest

# Note the public URL
python scripts/upload_artifact.py --url asl_emg_classifier.onnx
# → https://pub-XXXX.r2.dev/latest/asl_emg_classifier.onnx
```

## Step 3: Deploy to Railway

```bash
# Login to Railway
railway login

# First-time setup (creates project, sets env vars)
./scripts/deploy_railway.sh --setup

# Edit railway.env.example with your real values, then re-setup:
# Update MAIA_API_KEYS, R2_MODEL_URL, etc.
railway variables set MAIA_API_KEYS=your-secret-key
railway variables set R2_MODEL_URL=https://pub-XXXX.r2.dev/latest/asl_emg_classifier.onnx
railway variables set MAIA_DISABLE_AUTH=false

# Deploy!
./scripts/deploy_railway.sh
```

## Step 4: Verify Deployment

```bash
# Get your Railway URL
railway status

# Test health endpoint
curl https://your-app.railway.app/health
# → {"status": "ok", "model_loaded": true}

# Test info endpoint
curl -H "X-API-Key: your-secret-key" https://your-app.railway.app/info
```

## Step 5: Configure Mobile App

Edit `mobile/react-native/.env`:
```bash
EXPO_PUBLIC_RAILWAY_URL=https://your-app.railway.app
EXPO_PUBLIC_SERVER_URL=https://your-app.railway.app
EXPO_PUBLIC_API_KEY=your-secret-key
EXPO_PUBLIC_FALLBACK_TO_SERVER=true
```

## Cost Table

| Service | Free Tier | Paid |
|---------|-----------|------|
| Railway | $5 credit/month (hobby) | ~$0 for low traffic |
| Cloudflare R2 | 10GB storage, 10M reads | $0.015/GB after |
| IYA Nvidia Lab | Free (university) | — |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MAIA_API_KEYS` | Yes | Comma-separated API keys |
| `R2_MODEL_URL` | Yes | Public URL to ONNX model on R2 |
| `ONNX_MODEL_PATH` | No | Local model path (default: `models/asl_emg_classifier.onnx`) |
| `MAIA_DISABLE_AUTH` | No | Set `true` for dev (default: false) |
| `PORT` | Auto | Set by Railway |
| `ORT_NUM_THREADS` | No | ONNX Runtime threads (default: 2) |

## Troubleshooting

**Model not loading at startup:**
- Check `railway logs` for download errors
- Verify `R2_MODEL_URL` is publicly accessible: `curl $R2_MODEL_URL -o /dev/null -w "%{http_code}"`
- Make sure R2 bucket has public read access

**WebSocket not connecting:**
- Mobile app uses `wss://` (not `ws://`) in production
- `serverConfig.ts` auto-converts `https://` → `wss://`
- Test with: `wscat -c wss://your-app.railway.app/ws/emg`

**Auth errors:**
- Add header: `X-API-Key: your-secret-key`
- Or set `MAIA_DISABLE_AUTH=true` for debugging
