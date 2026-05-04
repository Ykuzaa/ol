#!/bin/bash
#SBATCH --job-name=ol-v6-geo
#SBATCH --partition=training-large-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/v6_full_geo_phys_%j.out
#SBATCH --error=logs/v6_full_geo_phys_%j.err

set -eo pipefail

cd /scratch/emboulaalam/OceanLens_git
mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

CNO_CKPT=/scratch/emboulaalam/OceanLens_git/runs/v2_loggrad/cno/checkpoints/last.ckpt

/scratch/emboulaalam/conda/envs/oceanlens/bin/python scripts/train.py \
  --variant v6_full_geo_phys \
  --phase fm \
  --cno_ckpt "${CNO_CKPT}"
