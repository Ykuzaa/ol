#!/bin/bash
#SBATCH --job-name=ol-v4s1-cfg-sweep
#SBATCH --partition=inference-large
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/v4_s1_cfg_sweep_%j.out
#SBATCH --error=logs/v4_s1_cfg_sweep_%j.err

set -eo pipefail

source /softs/Anaconda3/2024.06/etc/profile.d/conda.sh
conda activate oceanlens

cd /scratch/emboulaalam/OceanLens_git
mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

CNO_CKPT=/scratch/emboulaalam/OceanLens_git/runs/v2_loggrad/cno/checkpoints/last.ckpt
FM_CKPT=/scratch/emboulaalam/OceanLens_git/runs/v4_s1_logit_t_cfg_ft/fm/checkpoints/last.ckpt
ROOT_RESULTS=/scratch/emboulaalam/OceanLens_git/results/v4_s1_cfg_ft_sweep

mkdir -p "${ROOT_RESULTS}"

run_case () {
  local scale="$1"
  local tag="cfg${scale//./p}_heun50_sigma10_ens1"
  local out_dir="${ROOT_RESULTS}/${tag}"
  mkdir -p "${out_dir}"
  echo "[infer] ${tag}"

  python scripts/infer_v3_minimal.py \
    --variant v4_s1_logit_t_cfg_ft \
    --mode cno_fm \
    --output_dir "${out_dir}" \
    --year 2004 \
    --day_index 0 \
    --n_steps 50 \
    --solver heun \
    --ensemble_members 1 \
    --noise_sigma 1.0 \
    --cfg_scale "${scale}" \
    --tile_size 512 \
    --tile_overlap 128 \
    --cno_ckpt "${CNO_CKPT}" \
    --fm_ckpt "${FM_CKPT}"

  NPZ=$(ls -1 "${out_dir}"/day_*.npz | tail -1)
  python scripts/plot_thetao_residuals.py \
    --npz "${NPZ}" \
    --output "${out_dir}/thetao_residuals.png" \
    --title "v4_s1_logit_t_cfg_ft | cfg=${scale} | Heun50 | sigma=1.0 | ens=1"
}

run_case 1.0
run_case 1.5
run_case 2.0
run_case 3.0

find "${ROOT_RESULTS}" -maxdepth 2 -type f | sort
