#!/bin/bash
#SBATCH --job-name=maia-lstm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%j_lstm.log

set -euo pipefail

echo "=== MAIA LSTM Training ==="
echo "Job ID: $SLURM_JOB_ID | Nodes: $SLURM_NODELIST"

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN

srun torchrun \
    --nproc_per_node=4 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    scripts/train_gpu_ddp.py \
    --model lstm \
    --config configs/lstm_default.yaml \
    --data-dir data/ \
    --output-dir models/

echo "Training complete"

# Upload to R2 if credentials available
if [[ -n "${R2_ACCOUNT_ID:-}" && -n "${R2_ACCESS_KEY_ID:-}" && "$SLURM_PROCID" == "0" ]]; then
    echo "Uploading artifacts to R2..."
    python scripts/upload_artifact.py --file models/asl_emg_classifier.onnx --set-latest
    echo "R2 upload complete"
fi
