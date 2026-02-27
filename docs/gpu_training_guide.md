# GPU Training Guide - IYA Nvidia Lab

Train MAIA models on the IYA Nvidia GPU cluster using SLURM.

## Setup on IYA Lab

```bash
# SSH into the cluster
ssh username@iya-cluster.usc.edu

# Clone repo
git clone https://github.com/calebnewtonusc/maia-emg-asl.git
cd maia-emg-asl

# Create virtualenv (use cluster Python)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure R2 credentials (for auto-upload after training)
export R2_ACCOUNT_ID=your_account_id
export R2_ACCESS_KEY_ID=your_key
export R2_SECRET_ACCESS_KEY=your_secret
export R2_BUCKET_NAME=maia-emg-asl
```

## Training Workflows

### 1. Quick LSTM Baseline (synthetic data, no GPU needed)
```bash
python scripts/train_lstm_baseline.py --epochs 100
```

### 2. LSTM DDP (4 GPUs via SLURM)
```bash
sbatch scripts/slurm/train_lstm.sh
squeue -u $USER  # monitor job
tail -f logs/slurm_*_lstm.log
```

### 3. Conformer DDP (4 GPUs, 6 hrs)
```bash
sbatch scripts/slurm/train_conformer.sh
```

### 4. Hyperparameter Search
```bash
sbatch scripts/slurm/optuna_hpo.sh
```

## After Training: Push to R2

Models are auto-uploaded to R2 when SLURM jobs complete (if credentials are set).
To manually upload:
```bash
# Upload ONNX to R2 and set as latest
python scripts/upload_artifact.py \
    --file models/asl_emg_classifier.onnx \
    --set-latest

# Then Railway will pick it up on next deployment (or restart)
```

## AMP + Batch Size Tips

- Use `--config configs/conformer_gpu.yaml` for AMP + gradient accumulation
- Effective batch = batch_size x gradient_accumulation x world_size
- Default conformer config: 1024 x 4 x 4 = 16,384 effective batch size
- Memory: Conformer (d_model=256, 6 layers) approx 8GB VRAM per process

## Troubleshooting NCCL

```bash
# If NCCL hangs on multi-node:
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # disable InfiniBand if not available
export NCCL_SOCKET_IFNAME=eth0

# Check GPU peer access
nvidia-smi topo -m
```

## W&B Integration (optional)

```bash
pip install wandb
wandb login
# Add to train_gpu_ddp.py: wandb.init(project="maia-emg-asl")
```
