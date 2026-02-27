# GPU Training Guide — IYA Nvidia Lab

Train MAIA models on the IYA Nvidia GPU cluster at USC using SLURM and PyTorch DDP.

---

## Cluster Overview

| Resource | Details |
|----------|---------|
| Cluster | IYA Nvidia Lab (USC) |
| Scheduler | SLURM |
| GPUs | Nvidia A100 / A6000 (check `sinfo` for current availability) |
| Storage | Shared NFS — use `/scratch/$USER/` for large data |
| Python | 3.11 (module or venv) |
| Login node | `ssh <usc_id>@iya-cluster.usc.edu` |

---

## One-Time Setup

```bash
# 1. SSH to cluster
ssh <usc_id>@iya-cluster.usc.edu

# 2. Clone repo
git clone https://github.com/calebnewtonusc/maia-emg-asl.git
cd maia-emg-asl

# 3. Create virtualenv (use module Python if available)
module load python/3.11   # or: python3.11 -m venv .venv
python3.11 -m venv .venv
source .venv/bin/activate

# 4. Install dependencies (CPU build is fine on login node)
pip install --upgrade pip
pip install -r requirements.txt

# 5. Configure Cloudflare R2 for auto-upload after training
#    Add to ~/.bashrc or pass as SLURM --export:
export R2_ACCOUNT_ID=your_account_id
export R2_ACCESS_KEY_ID=your_access_key
export R2_SECRET_ACCESS_KEY=your_secret_key
export R2_BUCKET_NAME=maia-emg-asl
```

### Verify GPU access

```bash
sinfo                          # list partitions and node status
srun --gres=gpu:1 nvidia-smi   # interactive GPU test (exits immediately)
```

---

## Training Workflows

### 1. LSTM Baseline (CPU · no SLURM needed)

Quick sanity check on the login node:

```bash
python scripts/train_lstm_baseline.py --epochs 100
# Trains on synthetic data, exports ONNX to models/asl_emg_classifier.onnx
# Takes ~30 seconds on CPU
```

### 2. LSTM DDP — SLURM (1–4 GPUs · ~2 hours)

```bash
sbatch scripts/slurm/train_lstm.sh
squeue -u $USER                           # monitor job status
tail -f logs/slurm_<jobid>_lstm.log       # stream logs
```

What the job does:
- Launches `torchrun --nproc_per_node=4` for DDP across 4 GPUs
- Trains for 200 epochs with cosine LR schedule
- Saves best checkpoint to `models/lstm_best.pt`
- Exports ONNX and uploads to R2 (`latest/asl_emg_classifier.onnx`)

### 3. Conformer DDP — SLURM (4 GPUs · ~6 hours)

```bash
sbatch scripts/slurm/train_conformer.sh
```

The Conformer uses:
- AMP (automatic mixed precision) via `torch.cuda.amp`
- Gradient accumulation (4 steps) — effective batch = 1024 × 4 × 4 GPUs = 16,384
- Macaron-style feedforward, GLU depthwise convolution, relative positional encoding
- ~8 GB VRAM per GPU process (A100 recommended)

### 4. Cross-Modal EMG↔Vision Embedding — SLURM (4 GPUs · ~4 hours)

```bash
sbatch scripts/slurm/train_cross_modal.sh
```

Requires WLASL data (download first — see below). The job trains the CLIP-style dual encoder:
- `EMGEncoder`: 3-layer Transformer on 80-dim feature windows
- `VisionEncoder`: 2-layer MLP on 63-dim MediaPipe landmarks
- Symmetric InfoNCE loss with learnable log-temperature
- Saves `models/cross_modal_emg_encoder.pt` and `models/cross_modal_vision_encoder.pt`

### 5. Optuna Hyperparameter Search — SLURM (8 GPUs · ~8 hours)

```bash
sbatch scripts/slurm/optuna_hpo.sh
```

Searches over: hidden size, num layers, dropout, learning rate, batch size, weight decay. Results stored in `optuna_maia.db`. Best config printed at job end.

---

## WLASL Dataset

WLASL (21,083 ASL videos, 2,000 word classes) is required for cross-modal training.

```bash
# Download and extract (~50 GB)
python scripts/download_wlasl.py --output /scratch/$USER/wlasl/

# Symlink into project data dir
ln -s /scratch/$USER/wlasl data/wlasl
```

Download takes ~2 hours on the cluster. Run as a SLURM job if the login node has time limits:

```bash
sbatch --time=04:00:00 --mem=16G --wrap="python scripts/download_wlasl.py --output /scratch/$USER/wlasl/"
```

---

## Monitoring Training

```bash
# Live job status
squeue -u $USER

# Live log streaming
tail -f logs/slurm_<jobid>_conformer.log

# GPU utilization on allocated node
srun --jobid=<jobid> --pty nvidia-smi dmon -s u -d 5

# Cancel a job
scancel <jobid>
```

### W&B integration (optional)

```bash
pip install wandb
wandb login
```

Then pass `--wandb` flag to any training script:
```bash
python scripts/train_gpu_ddp.py --config configs/conformer_gpu.yaml --wandb
```

Runs appear at `wandb.ai/<your-username>/maia-emg-asl`.

---

## After Training: Push to R2

Models auto-upload when SLURM jobs complete (if R2 env vars are set). To upload manually:

```bash
# Upload ONNX and mark as latest
python scripts/upload_artifact.py \
    --file models/asl_emg_classifier.onnx \
    --set-latest

# Upload a PyTorch checkpoint (for future fine-tuning)
python scripts/upload_artifact.py \
    --file models/conformer_best.pt \
    --key checkpoints/conformer_best.pt

# Verify upload
python scripts/upload_artifact.py --list
```

Railway pulls the latest ONNX from R2 on the next deploy. To force a redeploy after uploading:
```bash
git commit --allow-empty -m "chore: trigger Railway redeploy for new model" && git push
```

---

## SLURM Script Reference

All SLURM scripts are in `scripts/slurm/`. Key parameters you might want to adjust:

```bash
#SBATCH --gres=gpu:4          # Number of GPUs (change to 1 for debugging)
#SBATCH --time=06:00:00       # Wall time limit
#SBATCH --mem=64G             # Memory per node
#SBATCH --cpus-per-task=8     # CPU cores (for DataLoader workers)
#SBATCH --partition=gpu       # Change to match your cluster's GPU partition name
```

Check available partitions:
```bash
sinfo -o "%P %G %t %N"
```

---

## AMP + Batch Size Guide

| Model | GPU | Batch | Grad accum | Effective batch | VRAM |
|-------|-----|-------|-----------|----------------|------|
| LSTM | A100 40GB | 2048 | 1 | 2,048 | ~4 GB |
| CNN-LSTM | A100 40GB | 1024 | 2 | 2,048 | ~6 GB |
| Conformer | A100 40GB | 1024 | 4 | 16,384 × 4 GPUs | ~8 GB |
| Conformer | A6000 48GB | 1024 | 4 | 16,384 × 4 GPUs | ~10 GB |

AMP halves memory usage — always use it for Conformer training:
```python
# Already enabled in train_gpu_ddp.py via --config configs/conformer_gpu.yaml
# config flag: amp: true
```

---

## Troubleshooting

**NCCL hangs on multi-GPU start**
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1        # disable InfiniBand if not available
export NCCL_SOCKET_IFNAME=eth0  # use correct network interface

# Check GPU topology
nvidia-smi topo -m
```

**OOM (Out of Memory)**
- Reduce `batch_size` in the config YAML by half
- Increase `gradient_accumulation` to compensate
- Switch from Conformer to LSTM for the first runs

**Job exits immediately with exit code 1**
```bash
cat logs/slurm_<jobid>_*.log   # always check the log first
# Common causes: Python import error, missing data file, CUDA version mismatch
```

**`torchrun` port conflict**
```bash
export MASTER_PORT=$((29500 + RANDOM % 1000))   # randomize port in job script
```

**Module not found**
```bash
# Make sure .venv is activated in the SLURM script:
source /home/$USER/maia-emg-asl/.venv/bin/activate
```
