# MAIA EMG-ASL

> Real-time American Sign Language recognition from surface EMG signals using the MAIA Biotech wrist-worn neural band.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1+-orange.svg)](https://pytorch.org)
[![Expo](https://img.shields.io/badge/expo-53-black.svg)](https://expo.dev)
[![Railway](https://img.shields.io/badge/deployed-railway-blueviolet.svg)](https://maia-emg-asl-production.up.railway.app/health)
[![Tests](https://img.shields.io/badge/tests-21%20passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Current State & Next Steps

### What's done right now

| Component | Status | Notes |
|-----------|--------|-------|
| Python ML pipeline | **Done** | Signal processing, 4 model architectures, ONNX export |
| Baseline ONNX model | **Done** | 8.2KB LSTM, trained on synthetic data, 21 tests passing |
| Railway inference server | **Live** | `https://maia-emg-asl-production.up.railway.app/health` |
| Auto-deploy on push | **Done** | GitHub → Railway native integration |
| iOS simulator app | **Running** | Expo 53, expo-router, 3 screens, mock EMG frames |
| Training scripts | **Done** | SLURM jobs for IYA cluster: LSTM, Conformer, cross-modal, HPO |
| Cloudflare R2 pipeline | **Done** | Upload/download scripts ready; bucket not yet created |
| Vision teacher (WLASL) | **Done** | MediaPipe Hands + CLIP-style cross-modal embedding |
| Documentation | **Done** | README, API ref, model card, hardware guide, data protocol, CHANGELOG |

### What's blocked on hardware

| Task | Blocked by | Script ready? |
|------|-----------|---------------|
| Real sEMG data collection | MAIA Band arriving | `calibrate_user.py`, `auto_label_session.py` |
| Retrain with real data | Real sEMG data | `train_lstm_baseline.py`, `train_gpu_ddp.py` |
| BLE connection in app | MAIA Band arriving | `src/inference/` (needs real BLE UUID) |
| CoreML on physical device | iPhone + real model | `export_coreml.py` |

### What's blocked on IYA Lab access

| Task | Blocked by | Script ready? |
|------|-----------|---------------|
| GPU training (Conformer) | IYA cluster SSH access | `scripts/slurm/train_conformer.sh` |
| Hyperparameter search | IYA cluster + real data | `scripts/slurm/optuna_hpo.sh` |
| WLASL cross-modal pre-training | IYA cluster + WLASL download | `scripts/slurm/train_cross_modal.sh` |

### Immediate next steps (unblocked, do these now)

1. **Create Cloudflare R2 bucket** — free, takes 5 minutes → enables auto model storage after GPU training
2. **Get IYA Lab SSH access** → run `sbatch scripts/slurm/train_conformer.sh` for the best model
3. **TestFlight build** — `npx expo build:ios` once you have a physical device to test BLE on

---

## How It Works

```
MAIA Neural Band  (8-channel sEMG · 200Hz · BLE 5.0)
        │
        │  16-byte BLE packets  (8 × int16 big-endian)
        ▼
iPhone App  (React Native / Expo 53)
        │
        ├─► EMGWindowBuffer → 200ms sliding window (50% overlap)
        │
        ├─► On-device ONNX  ──────────────────────  < 15ms  ◄── primary path
        │       └── CoreML Neural Engine (iPhone)
        │
        └─► Railway WebSocket  ───────────────────  ~ 50ms  ◄── fallback
                └── FastAPI + ONNX Runtime CPU
        │
        ▼
ASL letter prediction  (A–Z · confidence score · history)
```

---

## Architecture

| Layer | Details |
|-------|---------|
| Hardware | 8-channel sEMG · 200 Hz · BLE 5.0 · dry Ag/AgCl electrodes |
| Signal processing | Bandpass 20–450 Hz → notch 60 Hz → rectify → 200ms window |
| Features | 80-dim vector: RMS, MAV, WL, ZC, SSC, VAR, AR(4), IEMG, kurtosis, MNF × 8 channels |
| On-device model | LSTM (246K params) → ONNX opset 17 → CoreML `.mlpackage` |
| Server model | Conformer (~1.5M params) trained on IYA Nvidia Lab via SLURM + DDP |
| Transfer learning | CLIP-style EMG↔vision embedding from WLASL (21K ASL videos) |
| Classes | 26 ASL static letters (A–Z) |

### End-to-End Latency

| Stage | Latency |
|-------|---------|
| BLE packet decode | ~1 ms |
| Signal preprocessing | ~10 ms |
| ONNX inference (on-device) | ~15 ms |
| WebSocket RTT to Railway | ~50 ms |
| **Total (on-device path)** | **~26 ms** |

---

## Quickstart

### Python environment

```bash
git clone https://github.com/calebnewtonusc/maia-emg-asl.git
cd maia-emg-asl
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Train baseline model + run tests

```bash
# Train LSTM on synthetic data, export ONNX (< 30s on CPU)
python scripts/train_lstm_baseline.py

# Run full test suite (21 tests)
pytest tests/ -v
```

### Local inference server

```bash
uvicorn src.api.main:app --reload --port 8000
curl http://localhost:8000/health
# → {"status": "ok", "model_loaded": true}
```

### WebSocket smoke test

```bash
python scripts/test_websocket.py --url ws://localhost:8000/ws/emg
```

---

## iOS Simulator Setup

### Prerequisites

- Xcode 15+ (Mac App Store)
- Node.js 18+

### Run

```bash
cd mobile/react-native
npm install
npx expo start --ios
```

Expo automatically boots an iPhone 17 Pro simulator and opens the app.

### App screens

| Screen | Description |
|--------|-------------|
| **ASL Live** | Real-time predictions via WebSocket; mock EMG generator when no hardware connected |
| **Demo** | Alphabet cycling at 0.8s/letter; server health check on mount |
| **Settings** | Server URL, API key, BLE device name, debug toggle |

### Environment

`mobile/react-native/.env` is pre-configured to hit the live Railway server:

```bash
EXPO_PUBLIC_SERVER_URL=https://maia-emg-asl-production.up.railway.app
EXPO_PUBLIC_BLE_DEVICE_NAME=MAIA-Band
EXPO_PUBLIC_FALLBACK_TO_SERVER=true
EXPO_PUBLIC_DEBUG_EMG=false
```

---

## Cloud Deployment (Railway)

The inference server auto-deploys to Railway on every push to `main` via native GitHub integration (no GitHub Actions needed).

### Live server

```
https://maia-emg-asl-production.up.railway.app
```

```bash
curl https://maia-emg-asl-production.up.railway.app/health
# → {"status": "ok", "model_loaded": true}
```

### First-time project setup

1. [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
2. Select `calebnewtonusc/maia-emg-asl`
3. Railway detects the `Dockerfile` automatically and begins building
4. **Variables** tab → paste all at once:

```
MAIA_DISABLE_AUTH=true
ONNX_MODEL_PATH=models/asl_emg_classifier.onnx
LOG_LEVEL=INFO
ORT_NUM_THREADS=2
```

5. **Settings → Networking** → Generate domain

Every subsequent `git push origin main` triggers an automatic redeploy.

See [docs/railway_deployment.md](docs/railway_deployment.md) for the full guide including production auth, R2 model storage, and troubleshooting.

---

## GPU Training (IYA Nvidia Lab)

```bash
# SSH to IYA cluster, clone repo, activate venv, then:
sbatch scripts/slurm/train_lstm.sh          # LSTM baseline      (1 GPU  · ~2 hrs)
sbatch scripts/slurm/train_conformer.sh     # Conformer best     (4 GPUs · ~6 hrs)
sbatch scripts/slurm/train_cross_modal.sh   # EMG↔vision embed   (4 GPUs · ~4 hrs)
sbatch scripts/slurm/optuna_hpo.sh          # Hyperparameter HPO  (8 GPUs · ~8 hrs)

# Monitor
squeue -u $USER
tail -f logs/slurm_*_conformer.log
```

Models auto-upload to Cloudflare R2 on job completion. Railway pulls from R2 on next deploy.

See [docs/gpu_training_guide.md](docs/gpu_training_guide.md) for cluster setup, DDP tips, and W&B integration.

---

## Vision-to-EMG Transfer Learning

We bootstrap recognition before hardware arrives using public ASL video data:

1. **WLASL** (21,083 ASL videos, 2,000 word classes) — `scripts/download_wlasl.py`
2. **MediaPipe Hands** extracts 63-dim wrist-relative landmarks per frame
3. **CLIP-style cross-modal embedding** aligns the EMG feature space with visual landmarks via symmetric InfoNCE loss and a learnable temperature
4. **Auto-labeling** — when the band arrives, `scripts/auto_label_session.py` uses the vision teacher to label synchronized EMG recordings automatically

```bash
python scripts/download_wlasl.py --output data/wlasl/
python scripts/train_cross_modal.py
```

---

## Data Collection (Hardware Phase)

Once the MAIA Neural Band arrives, collect real sEMG data in ~20-minute sessions:

```bash
# 1. Per-user calibration
python scripts/calibrate_user.py --user-id caleb_001

# 2. Record session via MAIA app → Settings → "Start Recording Session"

# 3. Auto-label with vision teacher
python scripts/auto_label_session.py \
    --video session_001_video.mp4 \
    --emg data/train/session_001.npz

# 4. Validate quality
python scripts/validate_session.py --file data/train/session_001.npz

# 5. Retrain with real data
python scripts/train_lstm_baseline.py --data-dir data/train/
```

See [docs/data-collection-protocol.md](docs/data-collection-protocol.md) for electrode placement diagrams, multi-session tips, quality thresholds, and IRB guidance.

---

## Project Structure

```
maia-emg-asl/
├── src/
│   ├── constants.py                    # Single source of truth for signal + model constants
│   ├── signal/
│   │   ├── filters.py                  # Bandpass, notch, full-wave rectify
│   │   └── features.py                 # 80-dim feature extraction (10 features × 8 channels)
│   ├── data/
│   │   ├── loader.py                   # EMGDataset, DataLoader, train/val split
│   │   ├── augmentation.py             # Time-warp, channel dropout, Gaussian noise, scaling
│   │   └── vision_teacher.py           # MediaPipe Hands → 63-dim wrist-relative landmarks
│   ├── models/
│   │   ├── lstm_classifier.py          # LSTM (246K params) — on-device default
│   │   ├── cnn_lstm_classifier.py      # 1D CNN + LSTM hybrid
│   │   ├── conformer_classifier.py     # Conformer (~1.5M params) — server / best accuracy
│   │   ├── svm_classifier.py           # SVM baseline (scikit-learn)
│   │   └── cross_modal_embedding.py    # CLIP-style EMGEncoder + VisionEncoder
│   └── api/
│       ├── auth.py                     # X-API-Key middleware
│       └── main.py                     # FastAPI: /predict, /ws/emg, /health, /info
├── scripts/
│   ├── train_lstm_baseline.py          # Train LSTM on synthetic data + export ONNX
│   ├── train_conformer_baseline.py     # Train Conformer on synthetic data
│   ├── train_gpu_ddp.py                # DDP multi-GPU training (IYA cluster)
│   ├── train_cross_modal.py            # Cross-modal EMG↔vision embedding
│   ├── optuna_hpo.py                   # Optuna hyperparameter search
│   ├── validate_session.py             # Session quality checker + --fix mode
│   ├── calibrate_user.py               # Per-user baseline calibration
│   ├── auto_label_session.py           # Vision teacher auto-labeling pipeline
│   ├── export_coreml.py                # ONNX → CoreML .mlpackage (iOS Neural Engine)
│   ├── benchmark_models.py             # Latency + accuracy comparison across architectures
│   ├── test_websocket.py               # WebSocket smoke test
│   ├── download_wlasl.py               # Download + extract WLASL dataset
│   ├── upload_artifact.py              # Upload model artifact to Cloudflare R2
│   ├── download_artifact.py            # Download model artifact from R2
│   ├── sync_models.sh                  # Sync all models R2 ↔ local
│   ├── deploy_railway.sh               # Legacy CLI deploy helper
│   ├── run_full_pipeline.sh            # End-to-end: data → train → validate → deploy
│   └── slurm/
│       ├── train_lstm.sh               # SLURM: LSTM DDP job (1 GPU)
│       ├── train_conformer.sh          # SLURM: Conformer DDP job (4 GPUs)
│       ├── train_cross_modal.sh        # SLURM: cross-modal job (4 GPUs)
│       └── optuna_hpo.sh               # SLURM: Optuna HPO job (8 GPUs)
├── mobile/react-native/
│   ├── app/
│   │   ├── _layout.tsx                 # Root layout (dark theme, status bar)
│   │   └── (tabs)/
│   │       ├── _layout.tsx             # Bottom tab bar (ASL Live / Demo / Settings)
│   │       ├── index.tsx               # ASL Live: WebSocket + mock EMG generator
│   │       ├── demo.tsx                # Demo: alphabet cycling + health check
│   │       └── settings.tsx            # Settings screen
│   ├── src/
│   │   ├── inference/                  # EMGWindowBuffer + ONNXInference
│   │   ├── config/                     # serverConfig.ts (URL → wss:// auto-convert)
│   │   └── screens/                    # SettingsScreen.tsx
│   ├── assets/                         # icon.png (1024×1024), splash-icon.png
│   ├── app.json                        # Expo config (bundle ID, BLE permissions)
│   └── package.json                    # Expo 53 · RN 0.79 · TypeScript strict
├── configs/
│   ├── lstm_default.yaml               # LSTM training config (CPU/single GPU)
│   └── conformer_gpu.yaml              # Conformer AMP + gradient accumulation config
├── models/
│   └── asl_emg_classifier.onnx         # Trained LSTM baseline (8.2KB · ONNX opset 17)
├── tests/
│   ├── conftest.py                     # pytest fixtures (synthetic batches, mock model)
│   └── test_pipeline.py                # 21 end-to-end tests (signal → feature → model → API)
├── docs/
│   ├── api_reference.md                # Full REST + WebSocket API reference
│   ├── railway_deployment.md           # Railway + R2 deployment guide
│   ├── gpu_training_guide.md           # IYA Nvidia Lab + SLURM guide
│   ├── model_card.md                   # Model specs, benchmarks, limitations
│   ├── data-collection-protocol.md     # sEMG recording protocol + electrode placement
│   └── hardware-setup.md               # MAIA Band setup, BLE spec, calibration
├── Dockerfile                          # Railway container (python:3.11-slim)
├── railway.toml                        # Railway build + healthcheck config
├── .railwayignore                      # Files excluded from Railway Docker builds
├── requirements.txt                    # Runtime Python dependencies
└── requirements-dev.txt                # Dev + test dependencies
```

---

## API Summary

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | None | Server + model load status |
| `GET` | `/info` | API Key | Model metadata, constants, version |
| `POST` | `/predict` | API Key | Single-window REST inference |
| `WS` | `/ws/emg` | API Key | Streaming real-time inference |

Full schema, request/response examples, and error codes: [docs/api_reference.md](docs/api_reference.md)

---

## Hardware (Arriving Soon)

| Spec | Value |
|------|-------|
| Channels | 8 sEMG · dry Ag/AgCl electrodes |
| Sample rate | 200 Hz |
| Connectivity | BLE 5.0 |
| Packet format | 16 bytes = 8 × int16 big-endian (±1.0V range) |
| BLE Service UUID | `12345678-1234-5678-1234-56789abcdef1` |
| BLE Notify UUID | `12345678-1234-5678-1234-56789abcdef0` |

See [docs/hardware-setup.md](docs/hardware-setup.md) for electrode placement, pairing, firmware flash, and calibration.

---

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.11+ |
| PyTorch | 2.1+ |
| Node.js | 18+ |
| Xcode | 15+ (iOS simulator / build) |
| Expo | 53 |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch conventions, code style, and the PR process.

## License

MIT — see [LICENSE](LICENSE).
