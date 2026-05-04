#!/bin/bash
#SBATCH --job-name=ol-v6-inf
#SBATCH --partition=inference-large
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/v6_full_geo_phys_infer_%j.out
#SBATCH --error=logs/v6_full_geo_phys_infer_%j.err

set -eo pipefail

source /softs/Anaconda3/2024.06/etc/profile.d/conda.sh
conda activate oceanlens

cd /scratch/emboulaalam/OceanLens_git
mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

OUT=/scratch/emboulaalam/OceanLens_git/results/v6_full_geo_phys_epoch078_inference/heun25_sigma10_ens1
CNO_CKPT=/scratch/emboulaalam/OceanLens_git/runs/v2_loggrad/cno/checkpoints/last.ckpt
FM_CKPT=/scratch/emboulaalam/OceanLens_git/runs/v6_full_geo_phys/fm/checkpoints/v6_full_geo_phys-fm-epoch=078.ckpt

mkdir -p "${OUT}"

python scripts/infer_v3_minimal.py \
  --variant v6_full_geo_phys \
  --mode cno_fm \
  --output_dir "${OUT}" \
  --year 2004 \
  --day_index 0 \
  --n_steps 25 \
  --solver heun \
  --ensemble_members 1 \
  --noise_sigma 1.0 \
  --tile_size 512 \
  --tile_overlap 128 \
  --cno_ckpt "${CNO_CKPT}" \
  --fm_ckpt "${FM_CKPT}"

NPZ=$(ls -1 "${OUT}"/day_*.npz | tail -1)

python scripts/plot_all_residuals.py \
  --npz "${NPZ}" \
  --output "${OUT}/all_residuals.png" \
  --title "v6_full_geo_phys epoch078 | Heun25 | sigma=1.0 | ens=1"

python scripts/plot_thetao_residuals.py \
  --npz "${NPZ}" \
  --output "${OUT}/thetao_residuals.png" \
  --title "v6_full_geo_phys epoch078 | thetao | Heun25 | sigma=1.0 | ens=1"

find "${OUT}" -maxdepth 1 -type f | sort
