#!/bin/bash
#SBATCH --job-name=ol-v4s1-reco-inf
#SBATCH --partition=inference-large
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/v4_s1_reco_infer_%j.out
#SBATCH --error=logs/v4_s1_reco_infer_%j.err

set -eo pipefail

source /softs/Anaconda3/2024.06/etc/profile.d/conda.sh
conda activate oceanlens

cd /scratch/emboulaalam/OceanLens_git
mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

CNO_CKPT=/scratch/emboulaalam/OceanLens_git/runs/v2_loggrad/cno/checkpoints/last.ckpt
FM_CKPT=/scratch/emboulaalam/OceanLens_git/runs/v4_s1_logit_t_recoarsen_ft/fm/checkpoints/last.ckpt
OUT_DIR=/scratch/emboulaalam/OceanLens_git/results/v4_s1_recoarsen_ft_inference/heun25_sigma10_ens32

mkdir -p "${OUT_DIR}"

python scripts/infer_v3_minimal.py \
  --variant v4_s1_logit_t_recoarsen_ft \
  --mode cno_fm \
  --output_dir "${OUT_DIR}" \
  --year 2004 \
  --day_index 0 \
  --n_steps 25 \
  --solver heun \
  --ensemble_members 32 \
  --noise_sigma 1.0 \
  --tile_size 512 \
  --tile_overlap 128 \
  --cno_ckpt "${CNO_CKPT}" \
  --fm_ckpt "${FM_CKPT}"

NPZ=$(ls -1 "${OUT_DIR}"/day_*.npz | tail -1)
python scripts/plot_thetao_residuals.py \
  --npz "${NPZ}" \
  --output "${OUT_DIR}/thetao_residuals.png" \
  --title "v4_s1_logit_t_recoarsen_ft | Heun25 | sigma=1.0 | ens=32 | tile512 ov128"

echo "Output directory: ${OUT_DIR}"
find "${OUT_DIR}" -maxdepth 1 -type f | sort
