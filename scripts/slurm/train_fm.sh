#!/bin/bash
#SBATCH --job-name=oceanlens-fm
#SBATCH --partition=training-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/fm_%A_%a.out
#SBATCH --error=logs/fm_%A_%a.err
#SBATCH --array=0-1

# --- Map array index to variant ---
VARIANTS=(v1 v2)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}

echo "=== Training FM for ${VARIANT} on $(hostname) ==="
echo "Start: $(date)"

# --- Environment ---
module load anaconda3/2024.06
eval "$(conda shell.bash hook)"
conda activate oceanlens

# --- Paths ---
WORKDIR=/homelocal/emboulaalam/OceanLens_git
RUNS=/scratch/emboulaalam/OceanLens_git/runs
cd ${WORKDIR}
mkdir -p logs

# --- Find best CNO checkpoint ---
CNO_CKPT=${RUNS}/${VARIANT}/cno/checkpoints/last.ckpt
if [ ! -f "${CNO_CKPT}" ]; then
    echo "ERROR: CNO checkpoint not found at ${CNO_CKPT}"
    exit 1
fi

# --- Train ---
python scripts/train.py \
    --variant ${VARIANT} \
    --phase fm \
    --cno_ckpt ${CNO_CKPT} \
    --gpu 0

echo "Done: $(date)"
