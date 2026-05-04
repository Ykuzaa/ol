#!/bin/bash
#SBATCH --job-name=ol-v4-fm
#SBATCH --partition=training-large-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/v4_fm_%j.out
#SBATCH --error=logs/v4_fm_%j.err

set -eo pipefail

cd /scratch/emboulaalam/OceanLens_git

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

/scratch/emboulaalam/conda/envs/oceanlens/bin/python scripts/train.py \
--variant v4 \
--phase fm \
--gpu 0 \
--cno_ckpt "/scratch/emboulaalam/OceanLens_git/runs/v2_loggrad/cno/checkpoints/v2_loggrad-cno-epoch=024.ckpt"
