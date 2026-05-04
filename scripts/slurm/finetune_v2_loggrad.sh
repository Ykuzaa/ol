#!/bin/bash
#SBATCH --job-name=oceanlens-v2-loggrad
#SBATCH --partition=training-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/v2_loggrad_%j.out
#SBATCH --error=logs/v2_loggrad_%j.err

set -euo pipefail

echo "=== Fine-tuning OceanLens V2 with log-gradient temperature loss ==="
echo "Host: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start: $(date)"

module load anaconda3/2024.06
eval "$(conda shell.bash hook)"
conda activate oceanlens

WORKDIR=${WORKDIR:-/scratch/emboulaalam/OceanLens_git}
V2_CNO_CKPT=${V2_CNO_CKPT:-${WORKDIR}/runs/v2/cno/checkpoints/last.ckpt}

cd "${WORKDIR}"
mkdir -p logs

echo "Workdir: ${WORKDIR}"
echo "Starting weights: ${V2_CNO_CKPT}"
echo "Output run: ${WORKDIR}/runs/v2_loggrad/cno"

python scripts/train.py \
    --variant v2_loggrad \
    --phase cno \
    --gpu 0 \
    --cno_ckpt "${V2_CNO_CKPT}"

echo "Done: $(date)"
