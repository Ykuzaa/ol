#!/bin/bash
#SBATCH --job-name=ol-v8a-dit-ov128
#SBATCH --partition=inference-large
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/v8a_dit_pixel_overlap128_%j.out
#SBATCH --error=logs/v8a_dit_pixel_overlap128_%j.err

set -euo pipefail

cd /scratch/emboulaalam/OceanLens_git
mkdir -p logs

source /softs/Anaconda3/2024.06/etc/profile.d/conda.sh
conda activate oceanlens

export PYTHONPATH=src

FM_CKPT=${FM_CKPT:-/scratch/emboulaalam/OceanLens_git/runs/v8a_dit_pixel/fm/checkpoints/v8a_dit_pixel-fm-epoch=132.ckpt}
OUT_DIR=${OUT_DIR:-/scratch/emboulaalam/OceanLens_git/results/v8a_dit_pixel_inference/heun50_sigma10_ens1_tile256_overlap128}

python scripts/infer_v3_minimal.py \
  --variant v8a_dit_pixel \
  --mode cno_fm \
  --cno_ckpt /scratch/emboulaalam/OceanLens_git/runs/v2_loggrad/cno/checkpoints/last.ckpt \
  --fm_ckpt "${FM_CKPT}" \
  --output_dir "${OUT_DIR}" \
  --year 2004 \
  --day_index 0 \
  --solver heun \
  --n_steps 50 \
  --noise_sigma 1.0 \
  --ensemble_members 1 \
  --tile_size 256 \
  --tile_overlap 128 \
  --device cuda

python scripts/plot_thetao_residuals.py \
  --npz "${OUT_DIR}/day_2004-01-01.npz" \
  --output "${OUT_DIR}/thetao_residuals.png" \
  --title "v8a DiT | Heun 50 | sigma=1.0 | ens=1 | tile=256 overlap=128 | 2004-01-01"
