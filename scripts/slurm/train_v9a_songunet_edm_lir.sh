#!/bin/bash
#SBATCH --job-name=ol-v9a-edm
#SBATCH --partition=training-large-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/v9a_songunet_edm_%j.out
#SBATCH --error=logs/v9a_songunet_edm_%j.err

set -euo pipefail

cd /scratch/emboulaalam/OceanLens_git
mkdir -p logs

source /softs/Anaconda3/2024.06/etc/profile.d/conda.sh
conda activate oceanlens

export PYTHONPATH=src
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "=== Training v9a_songunet_edm FM ==="
echo "Start: $(date)"

srun python scripts/train.py \
  --variant v9a_songunet_edm \
  --phase fm \
  --cno_ckpt /scratch/emboulaalam/OceanLens_git/runs/v2_loggrad/cno/checkpoints/last.ckpt

echo "End: $(date)"
