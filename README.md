# MAIA EMG-ASL

Real-time American Sign Language recognition from surface EMG (sEMG) signals using MAIA Biotech's wrist-worn neural band.

```
BLE Neural Band (8-ch, 200Hz)
        | 16-byte packets
iPhone (React Native)
        | EMGWindowBuffer -> 200ms sliding window
        | On-device ONNX (primary, <15ms)
        | Railway WebSocket (fallback, ~50ms)
        |
ASL letter prediction (A-Z, confidence score)
```

## Architecture

| Layer | Details |
|-------|---------|
| Hardware | 8-channel sEMG, 200Hz, BLE 5.0 |
| Signal | Bandpass 20-450Hz -> notch 60Hz -> rectify |
| Features | 80-dim: RMS, MAV, WL, ZC, SSC, VAR, AR4, IEMG, kurtosis, MNF per channel |
| Model (on-device) | LSTM (246K params) -> ONNX -> CoreML |
| Model (server) | Conformer 1.5M params trained on IYA Nvidia Lab |
| Inference | On-device <15ms, Railway ~50ms |
| Classes | 26 ASL letters (A-Z) |

## End-to-End Latency

| Stage | Time |
|-------|------|
| BLE packet decode | ~1ms |
| Signal preprocessing | ~10ms |
| ONNX inference (on-device) | ~15ms |
| WebSocket RTT to Railway | ~50ms |
| **Total (on-device)** | **~26ms** |

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate baseline model
python scripts/train_lstm_baseline.py

# 3. Start local inference server
./start-server.sh

# 4. Test with WebSocket client
python scripts/test_websocket.py
```

## Cloud Deployment (Railway)

```bash
npm install -g @railway/cli && railway login
./scripts/deploy_railway.sh --setup
./scripts/deploy_railway.sh
```
See [docs/railway_deployment.md](docs/railway_deployment.md) for full guide.

## GPU Training (IYA Nvidia Lab)

```bash
# SSH to cluster, then:
sbatch scripts/slurm/train_lstm.sh        # LSTM baseline
sbatch scripts/slurm/train_conformer.sh   # Conformer (best accuracy)
```
Models auto-upload to Cloudflare R2. Railway pulls them on next deploy.
See [docs/gpu_training_guide.md](docs/gpu_training_guide.md).

## Vision-to-EMG Transfer Learning

We use WLASL (21K ASL videos) to bootstrap EMG classification before hardware arrives:

1. MediaPipe Hands extracts 63-dim landmarks from WLASL videos
2. CLIP-style cross-modal embedding aligns EMG features with visual landmarks
3. During data collection: vision teacher auto-labels synchronized EMG

## Project Structure

```
maia-emg-asl/
├── src/
│   ├── constants.py          # Signal/model constants
│   ├── signal/               # Filters + feature extraction
│   ├── data/                 # Loader, augmentation, vision teacher
│   ├── models/               # LSTM, CNN-LSTM, Conformer, SVM, cross-modal
│   └── api/                  # FastAPI + WebSocket inference server
├── scripts/
│   ├── train_lstm_baseline.py
│   ├── train_conformer_baseline.py
│   ├── train_gpu_ddp.py      # DDP multi-GPU training
│   ├── optuna_hpo.py         # Hyperparameter search
│   ├── validate_session.py   # Session quality checker
│   ├── upload_artifact.py    # R2 artifact upload
│   ├── download_artifact.py  # R2 artifact download
│   ├── sync_models.sh        # Sync models to/from R2
│   ├── deploy_railway.sh     # One-command Railway deploy
│   └── slurm/                # SLURM job scripts
├── mobile/react-native/
│   ├── src/inference/        # EMGWindowBuffer + ONNXInference
│   ├── src/config/           # serverConfig.ts
│   └── src/screens/          # SettingsScreen.tsx
├── configs/                  # Training YAML configs
├── tests/                    # pytest suite
├── docs/                     # Deployment + training guides
├── railway.toml              # Railway deployment config
└── Dockerfile                # Container for Railway
```

## Hardware (Arriving Soon)

- MAIA Neural Band: 8-ch sEMG, dry electrodes, BLE 5.0
- BLE UUID service: `12345678-1234-5678-1234-56789abcdef1`
- Packet format: 16 bytes = 8 x int16 big-endian (+-1.0V range)
- Electrode placement: forearm flexors/extensors pattern

## Requirements

- Python 3.11+
- PyTorch 2.1+
- React Native + Expo (mobile)
- Railway CLI (deployment)
- Cloudflare R2 account (model storage)
