#!/bin/bash
#SBATCH --job-name=ol-v9b-bridge
#SBATCH --partition=training-large-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/v9b_songunet_bridge_%j.out
#SBATCH --error=logs/v9b_songunet_bridge_%j.err

set -euo pipefail

cd /scratch/emboulaalam/OceanLens_git
mkdir -p logs

source /softs/Anaconda3/2024.06/etc/profile.d/conda.sh
conda activate oceanlens

export PYTHONPATH=src
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "=== Training v9b_songunet_bridge FM ==="
echo "Start: $(date)"

srun python scripts/train.py \
  --variant v9b_songunet_bridge \
  --phase fm \
  --cno_ckpt /scratch/emboulaalam/OceanLens_git/runs/v2_loggrad/cno/checkpoints/last.ckpt

echo "End: $(date)"
