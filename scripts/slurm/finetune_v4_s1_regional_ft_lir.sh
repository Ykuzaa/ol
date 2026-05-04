#!/bin/bash
#SBATCH --job-name=v4s1-reg-ft
#SBATCH --partition=training-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/v4_s1_regional_ft_%j.out
#SBATCH --error=logs/v4_s1_regional_ft_%j.err

set -euo pipefail

source /softs/Anaconda3/2024.06/etc/profile.d/conda.sh
conda activate oceanlens

WORKDIR=${WORKDIR:-/scratch/emboulaalam/OceanLens_git}
CNO_CKPT=${CNO_CKPT:-${WORKDIR}/runs/v2_loggrad/cno/checkpoints/last.ckpt}
FM_CKPT=${FM_CKPT:-${WORKDIR}/runs/v4_s1_logit_t/fm/checkpoints/last.ckpt}

cd "${WORKDIR}"
mkdir -p logs

python scripts/train.py \
  --variant v4_s1_logit_t_regional_ft \
  --phase fm \
  --gpu 0 \
  --cno_ckpt "${CNO_CKPT}" \
  --fm_ckpt "${FM_CKPT}"
