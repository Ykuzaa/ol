#!/bin/bash
#SBATCH --job-name=oceanlens-ablation
#SBATCH --partition=training-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err

echo "=== Training FM ablation (no CNO) on $(hostname) ==="
echo "Start: $(date)"

module load anaconda3/2024.06
eval "$(conda shell.bash hook)"
conda activate oceanlens

WORKDIR=/homelocal/emboulaalam/OceanLens_git
cd ${WORKDIR}
mkdir -p logs

python scripts/train.py \
    --variant ablation \
    --phase fm \
    --gpu 0

echo "Done: $(date)"
