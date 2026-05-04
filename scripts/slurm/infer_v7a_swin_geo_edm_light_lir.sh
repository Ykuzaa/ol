#!/bin/bash
#SBATCH --job-name=ol-v7a-swin-inf
#SBATCH --partition=inference-large
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/v7a_swin_geo_edm_light_infer_%j.out
#SBATCH --error=logs/v7a_swin_geo_edm_light_infer_%j.err

set -euo pipefail

cd /scratch/emboulaalam/OceanLens_git
mkdir -p logs

source /softs/Anaconda3/2024.06/etc/profile.d/conda.sh
conda activate oceanlens

export PYTHONPATH=src

FM_CKPT=${FM_CKPT:-/scratch/emboulaalam/OceanLens_git/runs/v7a_swin_geo_edm_light/fm/checkpoints/v7a_swin_geo_edm_light-fm-epoch=086.ckpt}
OUT_DIR=${OUT_DIR:-/scratch/emboulaalam/OceanLens_git/results/v7a_swin_geo_edm_light_inference/heun50_sigma10_ens1}

python scripts/infer_v3_minimal.py \
  --variant v7a_swin_geo_edm_light \
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
  --tile_size 512 \
  --tile_overlap 64 \
  --device cuda

python scripts/plot_thetao_residuals.py \
  --npz "${OUT_DIR}/day_2004-01-01.npz" \
  --output "${OUT_DIR}/thetao_residuals.png" \
  --title "v7a Swin | Heun 50 | sigma=1.0 | ens=1 | 2004-01-01"
