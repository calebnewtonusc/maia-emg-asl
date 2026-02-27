#!/bin/bash
#SBATCH --job-name=maia-hpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%j_hpo.log

set -euo pipefail

echo "=== MAIA Optuna HPO ==="
echo "Job ID: $SLURM_JOB_ID"

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

pip install -q optuna

# Run HPO for both models
echo "LSTM HPO..."
python scripts/optuna_hpo.py --model lstm --n-trials 50 --train-best

echo "CNN-LSTM HPO..."
python scripts/optuna_hpo.py --model cnn_lstm --n-trials 30 --train-best

echo "HPO complete. Best params saved to models/"
