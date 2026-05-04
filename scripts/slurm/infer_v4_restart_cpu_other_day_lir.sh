#!/bin/bash
#SBATCH --job-name=ol-v4-cpu-infer
#SBATCH --partition=multi4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/v4_cpu_infer_%j.out
#SBATCH --error=logs/v4_cpu_infer_%j.err

set -eo pipefail

source /softs/Anaconda3/2024.06/etc/profile.d/conda.sh
conda activate oceanlens

cd /scratch/emboulaalam/OceanLens_git
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

OUT_DIR="${OUT_DIR:-/scratch/emboulaalam/OceanLens_git/results/v4_restart_day30_heun30_cpu}"
CNO_CKPT="${CNO_CKPT:-/scratch/emboulaalam/OceanLens_git/runs/v2_loggrad/cno/checkpoints/v2_loggrad-cno-epoch=024.ckpt}"
FM_CKPT="${FM_CKPT:-/scratch/emboulaalam/OceanLens_git/runs/v4/fm/checkpoints/v4-fm-epoch=140.ckpt}"
YEAR="${YEAR:-2004}"
DAY_INDEX="${DAY_INDEX:-30}"
N_STEPS="${N_STEPS:-30}"
TILE_SIZE="${TILE_SIZE:-256}"
TILE_OVERLAP="${TILE_OVERLAP:-32}"
SOLVER="${SOLVER:-heun}"

python scripts/infer_v3_minimal.py \
  --variant v4 \
  --mode cno_fm \
  --output_dir "${OUT_DIR}" \
  --year "${YEAR}" \
  --day_index "${DAY_INDEX}" \
  --n_steps "${N_STEPS}" \
  --solver "${SOLVER}" \
  --ensemble_members 1 \
  --tile_size "${TILE_SIZE}" \
  --tile_overlap "${TILE_OVERLAP}" \
  --device cpu \
  --cno_ckpt "${CNO_CKPT}" \
  --fm_ckpt "${FM_CKPT}"

NPZ=$(ls -1 "${OUT_DIR}"/day_*.npz | tail -1)
python scripts/plot_thetao_residuals.py \
  --npz "${NPZ}" \
  --output "${OUT_DIR}/thetao_residuals.png"

echo "Outputs:"
echo "${OUT_DIR}"
ls -lh "${OUT_DIR}"
