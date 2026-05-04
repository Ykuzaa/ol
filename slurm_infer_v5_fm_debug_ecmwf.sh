#!/bin/bash
#SBATCH --job-name=ol-v5-infer
#SBATCH --partition=gpu_debug
#SBATCH --qos=dg
#SBATCH --output=logs/v5_infer_%j.out
#SBATCH --error=logs/v5_infer_%j.err
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -eo pipefail

module load python3/3.11.10-01
source /ec/res4/scratch/fra0606/work/envs/oceanlens/bin/activate

cd /ec/res4/scratch/fra0606/work/OceanLens_git
mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUT_DIR="${OUT_DIR:-/ec/res4/scratch/fra0606/work/OceanLens_git/results/v5_fm_day0}"
INPUT_NC="${INPUT_NC:-/ec/res4/scratch/fra0606/work/OceanLens/data/raw_monthly/2004/glorys_2004_01.nc}"
FM_CKPT="${FM_CKPT:-/ec/res4/scratch/fra0606/work/OceanLens_git/runs/v5/fm/checkpoints/last.ckpt}"
TIME_INDEX="${TIME_INDEX:-0}"
N_STEPS="${N_STEPS:-20}"
TILE_SIZE="${TILE_SIZE:-256}"
TILE_OVERLAP="${TILE_OVERLAP:-32}"
SOLVER="${SOLVER:-euler}"

python scripts/infer_v5_monthly_single_day.py \
  --variant v5 \
  --output_dir "${OUT_DIR}" \
  --input_nc "${INPUT_NC}" \
  --time_index "${TIME_INDEX}" \
  --n_steps "${N_STEPS}" \
  --solver "${SOLVER}" \
  --ensemble_members 1 \
  --tile_size "${TILE_SIZE}" \
  --tile_overlap "${TILE_OVERLAP}" \
  --fm_ckpt "${FM_CKPT}"

NPZ=$(ls -1 "${OUT_DIR}"/day_*.npz | tail -1)
python scripts/plot_thetao_residuals.py \
  --npz "${NPZ}" \
  --output "${OUT_DIR}/thetao_residuals.png" \
  --title "v5 ECMWF | thetao residual diagnostics | monthly day 0"

echo "Outputs:"
echo "${OUT_DIR}"
ls -lh "${OUT_DIR}"
