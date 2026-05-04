#!/bin/bash
#SBATCH --job-name=ol-fm-infer-all
#SBATCH --partition=training-large-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/fm_infer_all_%j.out
#SBATCH --error=logs/fm_infer_all_%j.err

set -eo pipefail

source /softs/Anaconda3/2024.06/etc/profile.d/conda.sh
conda activate oceanlens

cd /scratch/emboulaalam/OceanLens_git
mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

ROOT_RESULTS="${ROOT_RESULTS:-/scratch/emboulaalam/OceanLens_git/results/fm_variant_compare_day0_heun25_ens16}"
CNO_CKPT="${CNO_CKPT:-/scratch/emboulaalam/OceanLens_git/runs/v2_loggrad/cno/checkpoints/last.ckpt}"
YEAR="${YEAR:-2004}"
DAY_INDEX="${DAY_INDEX:-0}"
N_STEPS="${N_STEPS:-25}"
ENSEMBLE_MEMBERS="${ENSEMBLE_MEMBERS:-16}"
TILE_SIZE="${TILE_SIZE:-512}"
TILE_OVERLAP="${TILE_OVERLAP:-64}"
SOLVER="${SOLVER:-heun}"

mkdir -p "${ROOT_RESULTS}"

declare -a WITH_CNO_VARIANTS=(
  "v4"
  "v4_s1_logit_t"
  "v4_s2_independent"
  "v4_s3_no_attn"
  "v4_s4_grad_mu"
)

for VARIANT in "${WITH_CNO_VARIANTS[@]}"; do
  OUT_DIR="${ROOT_RESULTS}/${VARIANT}"
  FM_CKPT="/scratch/emboulaalam/OceanLens_git/runs/${VARIANT}/fm/checkpoints/last.ckpt"
  echo "[infer] ${VARIANT} -> ${OUT_DIR}"
  python scripts/infer_v3_minimal.py \
    --variant "${VARIANT}" \
    --mode cno_fm \
    --output_dir "${OUT_DIR}" \
    --year "${YEAR}" \
    --day_index "${DAY_INDEX}" \
    --n_steps "${N_STEPS}" \
    --solver "${SOLVER}" \
    --ensemble_members "${ENSEMBLE_MEMBERS}" \
    --tile_size "${TILE_SIZE}" \
    --tile_overlap "${TILE_OVERLAP}" \
    --cno_ckpt "${CNO_CKPT}" \
    --fm_ckpt "${FM_CKPT}"

  NPZ=$(ls -1 "${OUT_DIR}"/day_*.npz | tail -1)
  python scripts/plot_thetao_residuals.py \
    --npz "${NPZ}" \
    --output "${OUT_DIR}/thetao_residuals.png" \
    --title "${VARIANT} | thetao residual diagnostics | ${YEAR} day_index=${DAY_INDEX}"
done

V5_OUT="${ROOT_RESULTS}/v5_fm_only"
echo "[infer] v5_fm_only -> ${V5_OUT}"
python scripts/infer_ablation_minimal.py \
  --variant v5_fm_only \
  --output_dir "${V5_OUT}" \
  --year "${YEAR}" \
  --day_index "${DAY_INDEX}" \
  --n_steps "${N_STEPS}" \
  --solver "${SOLVER}" \
  --ensemble_members "${ENSEMBLE_MEMBERS}" \
  --tile_size "${TILE_SIZE}" \
  --tile_overlap "${TILE_OVERLAP}" \
  --fm_ckpt "/scratch/emboulaalam/OceanLens_git/runs/v5_fm_only/fm/checkpoints/last.ckpt"

V5_NPZ=$(ls -1 "${V5_OUT}"/day_*.npz | tail -1)
python scripts/plot_thetao_residuals.py \
  --npz "${V5_NPZ}" \
  --output "${V5_OUT}/thetao_residuals.png" \
  --title "v5_fm_only | thetao residual diagnostics | ${YEAR} day_index=${DAY_INDEX}"

python scripts/plot_thetao_residuals_summary.py \
  --items \
    "v4 baseline=${ROOT_RESULTS}/v4/day_2004-01-01.npz" \
    "v4_s1_logit_t=${ROOT_RESULTS}/v4_s1_logit_t/day_2004-01-01.npz" \
    "v4_s2_independent=${ROOT_RESULTS}/v4_s2_independent/day_2004-01-01.npz" \
    "v4_s3_no_attn=${ROOT_RESULTS}/v4_s3_no_attn/day_2004-01-01.npz" \
    "v4_s4_grad_mu=${ROOT_RESULTS}/v4_s4_grad_mu/day_2004-01-01.npz" \
    "v5_fm_only=${ROOT_RESULTS}/v5_fm_only/day_2004-01-01.npz" \
  --output "${ROOT_RESULTS}/thetao_residuals_all_variants.png" \
  --title "Thetao residual comparison | 2004-01-01 | Heun 25 | ens16 | tile512 ov64"

echo "Outputs:"
find "${ROOT_RESULTS}" -maxdepth 2 -type f | sort
