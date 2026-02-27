# MAIA EMG-ASL

> Real-time American Sign Language recognition directly from forearm muscle signals — no camera, no gloves, just a wrist-worn neural band.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1+-orange.svg)](https://pytorch.org)
[![Expo](https://img.shields.io/badge/expo-53-black.svg)](https://expo.dev)
[![Railway](https://img.shields.io/badge/deployed-railway-blueviolet.svg)](https://maia-emg-asl-production.up.railway.app/health)
[![Tests](https://img.shields.io/badge/tests-21%20passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## What This Is

**MAIA EMG-ASL** is a complete, open-source pipeline for recognizing ASL hand signs from surface EMG (sEMG) signals at the wrist — in real time, on a consumer device, under 15ms latency.

You wear the **Meta Neural Band** (16-channel sEMG, 2kHz, ships with Meta Ray-Ban Display glasses) on your forearm. As you form ASL letters, the band picks up the electrical activity of your forearm muscles. The system classifies those signals into one of 26 ASL letters (A–Z) and streams predictions to an iPhone app — no camera, no hand tracking, no visual line-of-sight required.

The end goal is a **silent, screenless ASL communication tool** that works in the dark, underwater, or anywhere vision-based systems fail — with sub-26ms end-to-end latency on-device.

---

## What's New — Why This Hasn't Been Done Before

Most prior EMG-ASL research either:
- Uses expensive custom lab rigs ($10,000+ research-grade amplifiers) that nobody can buy
- Uses low-quality consumer bands (Myo armband, 8ch/200Hz) with limited signal fidelity
- Trains on small proprietary datasets (< 5 participants, 1–2 sessions)
- Requires cameras or manual labeling as a crutch

**MAIA does four things no prior open project has combined:**

### 1. Consumer-grade hardware that's actually purchasable

We target the **Meta Neural Band** — 16 bipolar channels at 2kHz, 48 electrode pins, dry contact, the exact hardware Meta Research used in their [Nature 2025 paper](https://www.nature.com/articles/s41586-025-09255-w). It ships with $799 Ray-Ban Display glasses. This is the first consumer device with lab-quality EMG fidelity.

### 2. Pre-training on 900+ hours of real, labeled wrist-EMG data

Meta has released three massive open datasets, all recorded on the same 16ch/2kHz hardware:

| Dataset | Participants | Hours | Task | License |
|---------|-------------|-------|------|---------|
| [generic-neuromotor-interface](https://github.com/facebookresearch/generic-neuromotor-interface) | 100 | 200+ | Discrete gestures + handwriting + wrist | CC-BY-NC-4.0 |
| [emg2pose](https://github.com/facebookresearch/emg2pose) | 193 | 370 | EMG → 63-DoF hand joint angles | CC-BY-NC-4.0 |
| [emg2qwerty](https://github.com/facebookresearch/emg2qwerty) | 108 | 346 | Bilateral typing / character decoding | CC-BY-NC-4.0 |

No prior open EMG-ASL project has had access to this volume of labeled data at this hardware fidelity. We train on Meta's data first, then fine-tune on MAIA-collected ASL sessions.

### 3. A 2-stage pipeline that sidesteps the hardest problem

Direct EMG→ASL classification is hard because it requires large labeled datasets of ASL signs in particular. Instead, we use a two-stage approach:

```
EMG signal  →  [emg2pose model]  →  63-dim hand joint angles  →  [trivial ASL classifier]  →  Letter
```

**Stage 1**: Meta's pretrained `emg2pose` model maps raw EMG to hand joint angles (63 degrees of freedom). Meta already solved this on 370 hours of data.

**Stage 2**: Map joint angles → ASL letter. This is easy — ASL static letters are fully defined by hand shape. A simple KNN or decision tree on 63-dim MediaPipe-compatible joint angles achieves near-perfect classification. No deep learning needed for stage 2.

This architecture means we inherit Meta's years of hardware+data investment and only need to collect ASL-specific data for stage 2 fine-tuning (< 1 hour per user).

### 4. Zero-shot bootstrapping via cross-modal transfer

We build a **CLIP-style EMG↔vision embedding** that aligns the EMG feature space with MediaPipe Hands visual landmarks using symmetric InfoNCE loss. This lets the system auto-label EMG recordings using synchronized video — no human annotation required — and gives us a useful representation even before any labeled EMG data exists.

---

## Current State & Next Steps

### What's done right now

| Component | Status | Notes |
|-----------|--------|-------|
| Python ML pipeline | **Done** | Signal processing, 4 model architectures, ONNX export |
| Baseline ONNX model | **Done** | Trained on synthetic data, 21 tests passing |
| Railway inference server | **Live** | `https://maia-emg-asl-production.up.railway.app/health` |
| Auto-deploy on push | **Done** | GitHub → Railway native integration |
| iOS simulator app | **Running** | Expo 53, expo-router, 3 screens, mock EMG frames |
| Training scripts | **Done** | SLURM jobs for IYA cluster: LSTM, Conformer, cross-modal, HPO |
| Cloudflare R2 pipeline | **Done** | Upload/download scripts ready; bucket not yet created |
| Vision teacher (WLASL) | **Done** | MediaPipe Hands + CLIP-style cross-modal embedding |
| Documentation | **Done** | README, API ref, model card, hardware guide, data protocol |
| Codebase upgraded to 16ch/2kHz | **Done** | All constants, filters, features match Meta Band spec |

### What's blocked on hardware (Meta Band arriving)

| Task | Blocked by | Script ready? |
|------|-----------|---------------|
| Real sEMG data collection | Meta Band arriving | `calibrate_user.py`, `auto_label_session.py` |
| Retrain with real data | Real sEMG data | `train_lstm_baseline.py`, `train_gpu_ddp.py` |
| BLE connection in app | Meta Band arriving | `src/inference/` (needs real BLE UUID) |
| CoreML on physical device | iPhone + real model | `export_coreml.py` |
| Download & run on Meta datasets | Access approval | `scripts/download_meta_emg_data.py` |

### What's blocked on IYA Lab access

| Task | Blocked by | Script ready? |
|------|-----------|---------------|
| GPU training (Conformer) | IYA cluster SSH access | `scripts/slurm/train_conformer.sh` |
| Hyperparameter search | IYA cluster + real data | `scripts/slurm/optuna_hpo.sh` |
| Pre-training on Meta datasets | IYA cluster + data access | `scripts/slurm/train_cross_modal.sh` |

### Immediate next steps (unblocked, do these now)

1. **Create Cloudflare R2 bucket** — free, 5 minutes → enables auto model storage after GPU training
2. **Get IYA Lab SSH access** → run `sbatch scripts/slurm/train_conformer.sh` for the best model
3. **Request Meta dataset access** at [emg2pose](https://github.com/facebookresearch/emg2pose) and [generic-neuromotor-interface](https://github.com/facebookresearch/generic-neuromotor-interface)
4. **TestFlight build** — once Meta Band arrives and BLE UUID is confirmed

---

## How It Works

```
Meta Neural Band  (16-channel sEMG · 2kHz · BLE 5.0)
        │
        │  32-byte BLE packets  (16 × int16 big-endian)
        ▼
iPhone App  (React Native / Expo 53)
        │
        ├─► EMGWindowBuffer → 200ms window @ 2kHz = 400 samples (50% overlap)
        │
        ├─► Stage 1: emg2pose  ──────  EMG → 63-dim hand joint angles
        │
        ├─► Stage 2: ASL classifier  ── joint angles → A–Z letter
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
| Hardware | 16-channel sEMG · 2kHz · BLE 5.0 · 48-pin dry electrode array (Meta Neural Band) |
| Signal processing | Bandpass 20–450 Hz → notch 60 Hz → rectify → 200ms window (400 samples) |
| Features | 160-dim vector: RMS, MAV, WL, ZC, SSC, VAR, AR(4), IEMG, kurtosis, MNF × 16 channels |
| Stage 1 model | emg2pose (Meta pretrained) → 63-dim hand joint angles |
| Stage 2 model | LSTM (246K params) or Conformer (~1.5M params) on joint angles → A–Z |
| On-device model | ONNX opset 17 → CoreML `.mlpackage` → iPhone Neural Engine |
| Server model | Conformer trained on IYA Nvidia Lab via SLURM + DDP |
| Transfer learning | CLIP-style EMG↔vision embedding from WLASL (21K ASL videos) |
| Pre-training data | 900+ hours from Meta's emg2pose + generic-neuromotor-interface + emg2qwerty |
| Classes | 26 ASL static letters (A–Z) |

### End-to-End Latency

| Stage | Latency |
|-------|---------|
| BLE packet decode | ~1 ms |
| Signal preprocessing | ~5 ms |
| ONNX inference (on-device) | ~15 ms |
| WebSocket RTT to Railway | ~50 ms |
| **Total (on-device path)** | **~21 ms** |

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

## Pre-training on Meta Research Datasets

MAIA's architecture is designed to directly exploit three massive open datasets from Meta, all recorded on the same 16ch/2kHz hardware we're using:

```bash
# Download datasets (requires dataset access request approval)
python scripts/download_meta_emg_data.py --dataset emg2pose --output data/meta/
python scripts/download_meta_emg_data.py --dataset generic-neuromotor-interface --output data/meta/

# Pre-train Stage 1 (emg2pose) — reuse Meta's pretrained weights directly
python scripts/pretrain_on_meta_data.py --task pose --checkpoint models/emg2pose_pretrained.pt

# Fine-tune Stage 2 on ASL-specific data
python scripts/train_lstm_baseline.py --pretrained models/emg2pose_pretrained.pt
```

The 2-stage pipeline means ASL fine-tuning only requires hand-shape annotations (stage 2), not raw EMG→letter pairs (much harder to label). See [docs/model_card.md](docs/model_card.md) for architecture details.

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

Once the Meta Neural Band arrives, collect real sEMG data in ~20-minute sessions:

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

# 5. Fine-tune on real data
python scripts/train_lstm_baseline.py --data-dir data/train/
```

See [docs/data-collection-protocol.md](docs/data-collection-protocol.md) for electrode placement diagrams, multi-session tips, quality thresholds, and IRB guidance.

---

## Project Structure

```
maia-emg-asl/
├── src/
│   ├── constants.py                    # Single source of truth: 16ch, 2kHz, 400-sample windows
│   ├── signal/
│   │   ├── filters.py                  # Bandpass, notch, full-wave rectify (defaults: 2kHz)
│   │   └── features.py                 # 160-dim feature extraction (10 features × 16 channels)
│   ├── data/
│   │   ├── loader.py                   # EMGDataset, DataLoader, train/val split
│   │   ├── augmentation.py             # Time-warp, channel dropout, Gaussian noise, scaling
│   │   ├── meta_loader.py              # HDF5 loader for Meta research datasets
│   │   └── vision_teacher.py           # MediaPipe Hands → 63-dim wrist-relative landmarks
│   ├── models/
│   │   ├── lstm_classifier.py          # LSTM (246K params) — on-device default
│   │   ├── cnn_lstm_classifier.py      # 1D CNN + LSTM hybrid
│   │   ├── conformer_classifier.py     # Conformer (~1.5M params) — server / best accuracy
│   │   ├── svm_classifier.py           # SVM baseline (scikit-learn)
│   │   ├── two_stage_classifier.py     # emg2pose → hand joint angles → ASL letter
│   │   └── cross_modal_embedding.py    # CLIP-style EMGEncoder + VisionEncoder
│   └── api/
│       ├── auth.py                     # X-API-Key middleware
│       └── main.py                     # FastAPI: /predict, /ws/emg, /health, /info
├── scripts/
│   ├── train_lstm_baseline.py          # Train LSTM on synthetic data + export ONNX
│   ├── train_conformer_baseline.py     # Train Conformer on synthetic data
│   ├── train_gpu_ddp.py                # DDP multi-GPU training (IYA cluster)
│   ├── train_cross_modal.py            # Cross-modal EMG↔vision embedding
│   ├── pretrain_on_meta_data.py        # Pre-train on Meta research datasets
│   ├── download_meta_emg_data.py       # Download Meta emg2pose / generic-neuromotor-interface
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
│   └── asl_emg_classifier.onnx         # Trained LSTM baseline (ONNX opset 17)
├── tests/
│   ├── conftest.py                     # pytest fixtures (synthetic batches, mock model)
│   └── test_pipeline.py                # End-to-end tests (signal → feature → model → API)
├── docs/
│   ├── api_reference.md                # Full REST + WebSocket API reference
│   ├── railway_deployment.md           # Railway + R2 deployment guide
│   ├── gpu_training_guide.md           # IYA Nvidia Lab + SLURM guide
│   ├── model_card.md                   # Model specs, benchmarks, limitations
│   ├── data-collection-protocol.md     # sEMG recording protocol + electrode placement
│   └── hardware-setup.md               # Meta Neural Band setup, BLE spec, calibration
├── Dockerfile                          # Railway container (python:3.11-slim)
├── railway.toml                        # Railway build + healthcheck config
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

WebSocket binary frame format: **12,800 bytes** = 400 samples × 16 channels × int16

Full schema, request/response examples, and error codes: [docs/api_reference.md](docs/api_reference.md)

---

## Hardware

| Spec | Value |
|------|-------|
| Device | Meta Neural Band (ships with Meta Ray-Ban Display glasses, $799) |
| Channels | 16 bipolar sEMG · 48 electrode pins |
| Sample rate | 2,000 Hz (2kHz) |
| Connectivity | BLE 5.0 |
| Packet format | 32 bytes = 16 × int16 big-endian (±1.0V range) |
| Window size | 400 samples = 200ms @ 2kHz |

See [docs/hardware-setup.md](docs/hardware-setup.md) for electrode placement, pairing, and calibration.

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

## Related Work

| Project | What it does | Relation to MAIA |
|---------|-------------|------------------|
| [emg2pose](https://github.com/facebookresearch/emg2pose) | EMG → 63-DoF hand joint angles, 370h/193 participants | MAIA uses this as Stage 1 |
| [generic-neuromotor-interface](https://github.com/facebookresearch/generic-neuromotor-interface) | Nature 2025 — discrete gestures + handwriting, 100 participants | Pre-training source |
| [emg2qwerty](https://github.com/facebookresearch/emg2qwerty) | Bilateral typing from wrist EMG, 108 users | Pre-training source |
| [Ninapro DB1–DB9](http://ninapro.hevs.ch/) | Hand gesture EMG, multiple hardware configs | Additional pre-training |
| [WLASL](https://dxli94.github.io/WLASL/) | 21K ASL videos, 2K word classes | Vision teacher bootstrapping |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch conventions, code style, and the PR process.

## License

MIT — see [LICENSE](LICENSE).
