#!/bin/bash
#SBATCH --job-name=ol-v8a-scen
#SBATCH --partition=inference-large
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=logs/v8a_dit_pixel_scenario_%x_%j.out
#SBATCH --error=logs/v8a_dit_pixel_scenario_%x_%j.err

set -euo pipefail

cd /scratch/emboulaalam/OceanLens_git
mkdir -p logs

source /softs/Anaconda3/2024.06/etc/profile.d/conda.sh
conda activate oceanlens

export PYTHONPATH=src

TILE_SIZE=${TILE_SIZE:?TILE_SIZE is required}
TILE_OVERLAP=${TILE_OVERLAP:?TILE_OVERLAP is required}
TAG=${TAG:-tile${TILE_SIZE}_overlap${TILE_OVERLAP}}
N_STEPS=${N_STEPS:-50}
NOISE_SIGMA=${NOISE_SIGMA:-1.0}
ENSEMBLE_MEMBERS=${ENSEMBLE_MEMBERS:-1}

FM_CKPT=${FM_CKPT:-/scratch/emboulaalam/OceanLens_git/runs/v8a_dit_pixel/fm/checkpoints/v8a_dit_pixel-fm-epoch=132.ckpt}
OUT_DIR=${OUT_DIR:-/scratch/emboulaalam/OceanLens_git/results/v8a_dit_pixel_inference/heun${N_STEPS}_sigma10_ens${ENSEMBLE_MEMBERS}_${TAG}}

echo "=== v8a scenario ==="
echo "FM_CKPT=${FM_CKPT}"
echo "OUT_DIR=${OUT_DIR}"
echo "TILE_SIZE=${TILE_SIZE}"
echo "TILE_OVERLAP=${TILE_OVERLAP}"
echo "N_STEPS=${N_STEPS}"
echo "NOISE_SIGMA=${NOISE_SIGMA}"
echo "ENSEMBLE_MEMBERS=${ENSEMBLE_MEMBERS}"
echo "Start: $(date)"

python scripts/infer_v3_minimal.py \
  --variant v8a_dit_pixel \
  --mode cno_fm \
  --cno_ckpt /scratch/emboulaalam/OceanLens_git/runs/v2_loggrad/cno/checkpoints/last.ckpt \
  --fm_ckpt "${FM_CKPT}" \
  --output_dir "${OUT_DIR}" \
  --year 2004 \
  --day_index 0 \
  --solver heun \
  --n_steps "${N_STEPS}" \
  --noise_sigma "${NOISE_SIGMA}" \
  --ensemble_members "${ENSEMBLE_MEMBERS}" \
  --tile_size "${TILE_SIZE}" \
  --tile_overlap "${TILE_OVERLAP}" \
  --device cuda

for variable in thetao so zos uo vo; do
  python scripts/plot_one_variable_residuals.py \
    --npz "${OUT_DIR}/day_2004-01-01.npz" \
    --output "${OUT_DIR}/${variable}_residuals.png" \
    --variable "${variable}" \
    --title "v8a DiT | ${variable} | Heun ${N_STEPS} | sigma=${NOISE_SIGMA} | ens=${ENSEMBLE_MEMBERS} | tile=${TILE_SIZE} overlap=${TILE_OVERLAP} | 2004-01-01"
done

echo "End: $(date)"
