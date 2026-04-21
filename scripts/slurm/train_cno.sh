#!/bin/bash
#SBATCH --job-name=oceanlens-cno
#SBATCH --partition=training-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cno_%A_%a.out
#SBATCH --error=logs/cno_%A_%a.err
#SBATCH --array=0-1

# --- Map array index to variant ---
VARIANTS=(v1 v2)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}

echo "=== Training CNO for ${VARIANT} on $(hostname) ==="
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Start: $(date)"

# --- Environment ---
module load anaconda3/2024.06
eval "$(conda shell.bash hook)"
conda activate oceanlens

# --- Paths ---
WORKDIR=/homelocal/emboulaalam/OceanLens_git
cd ${WORKDIR}
mkdir -p logs

# --- Train ---
python scripts/train.py \
    --variant ${VARIANT} \
    --phase cno \
    --gpu 0

echo "Done: $(date)"
