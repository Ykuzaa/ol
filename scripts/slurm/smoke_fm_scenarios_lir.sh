#!/bin/bash
set -euo pipefail

cd /scratch/emboulaalam/OceanLens_git
mkdir -p logs

CNO_CKPT="${CNO_CKPT:-/scratch/emboulaalam/OceanLens_git/runs/v2_loggrad/cno/checkpoints/v2_loggrad-cno-epoch=024.ckpt}"
variants=(
  v4_s1_logit_t
  v4_s2_independent
  v4_s3_no_attn
  v4_s4_grad_mu
  v5_fm_only
)

source /softs/Anaconda3/2024.06/etc/profile.d/conda.sh
conda activate oceanlens

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

for variant in "${variants[@]}"; do
  echo "[smoke] ${variant}"
  if [[ "${variant}" == v5* ]]; then
    python scripts/train.py \
      --variant "${variant}" \
      --phase fm \
      --gpu 0 \
      --fast_dev_run 1
  else
    python scripts/train.py \
      --variant "${variant}" \
      --phase fm \
      --gpu 0 \
      --cno_ckpt "${CNO_CKPT}" \
      --fast_dev_run 1
  fi
done

echo "[smoke] all scenarios passed"
