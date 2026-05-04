#!/bin/bash
#SBATCH --job-name=oceanlens-v3-fm
#SBATCH --partition=training-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/v3_fm_%j.out
#SBATCH --error=logs/v3_fm_%j.err

set -euo pipefail

echo "=== Fine-tuning OceanLens V3 FM on the V2-loggrad mu ==="
echo "Host: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start: $(date)"

module load anaconda3/2024.06
eval "$(conda shell.bash hook)"
conda activate oceanlens

WORKDIR=${WORKDIR:-/scratch/emboulaalam/OceanLens_git}
V2_LOGGRAD_CNO_CKPT=${V2_LOGGRAD_CNO_CKPT:-${WORKDIR}/runs/v2_loggrad/cno/checkpoints/last.ckpt}
V2_FM_CKPT=${V2_FM_CKPT:-${WORKDIR}/runs/v2/fm/checkpoints/last.ckpt}

cd "${WORKDIR}"
mkdir -p logs

echo "Workdir: ${WORKDIR}"
echo "CNO mu checkpoint: ${V2_LOGGRAD_CNO_CKPT}"
echo "FM starting checkpoint: ${V2_FM_CKPT}"
echo "Output run: ${WORKDIR}/runs/v3/fm"

python scripts/train.py \
    --variant v3 \
    --phase fm \
    --gpu 0 \
    --cno_ckpt "${V2_LOGGRAD_CNO_CKPT}" \
    --fm_ckpt "${V2_FM_CKPT}"

echo "Done: $(date)"
