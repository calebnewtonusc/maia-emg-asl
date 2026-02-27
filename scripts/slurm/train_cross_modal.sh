#!/bin/bash
#SBATCH --job-name=maia-xmodal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_%j_crossmodal.log

set -euo pipefail

echo "=== MAIA Cross-Modal Training ==="
echo "Job ID: $SLURM_JOB_ID"

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

# Check for WLASL landmarks
if [[ ! -f "data/wlasl/landmarks.npz" ]]; then
    echo "Extracting WLASL landmarks first..."
    python scripts/download_wlasl.py --extract-landmarks-only \
        --video-dir data/wlasl/videos \
        --output data/wlasl/landmarks.npz
fi

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29502

python scripts/train_cross_modal.py \
    --epochs 200 \
    --embed-dim 256 \
    --batch-size 512 \
    --output-dir models/

echo "Cross-modal training complete"

if [[ -n "${R2_ACCOUNT_ID:-}" && "$SLURM_PROCID" == "0" ]]; then
    echo "Uploading to R2..."
    python scripts/upload_artifact.py \
        --file models/cross_modal_embedding.pt \
        --tag cross-modal --set-latest
fi
