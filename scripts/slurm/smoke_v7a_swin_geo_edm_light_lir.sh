#!/bin/bash
#SBATCH --job-name=ol-v7a-smoke
#SBATCH --partition=training-large-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/v7a_swin_geo_edm_light_smoke_%j.out
#SBATCH --error=logs/v7a_swin_geo_edm_light_smoke_%j.err

set -eo pipefail

cd /scratch/emboulaalam/OceanLens_git
mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

CNO_CKPT=/scratch/emboulaalam/OceanLens_git/runs/v2_loggrad/cno/checkpoints/last.ckpt

/scratch/emboulaalam/conda/envs/oceanlens/bin/python scripts/train.py \
  --variant v7a_swin_geo_edm_light \
  --phase fm \
  --cno_ckpt "${CNO_CKPT}" \
  --fast_dev_run 1
