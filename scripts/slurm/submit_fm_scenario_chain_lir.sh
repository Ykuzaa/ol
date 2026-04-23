#!/bin/bash
set -euo pipefail

cd /scratch/emboulaalam/OceanLens_git
mkdir -p logs

PARTITION="${PARTITION:-training-large-1}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
CPUS="${CPUS:-16}"
MEM="${MEM:-64G}"
GPU="${GPU:-1}"
CNO_CKPT="${CNO_CKPT:-/scratch/emboulaalam/OceanLens_git/runs/v2_loggrad/cno/checkpoints/v2_loggrad-cno-epoch=024.ckpt}"

variants=(
  v4_s1_logit_t
  v4_s2_independent
  v4_s3_no_attn
  v4_s4_grad_mu
  v5_fm_only
)

previous_job=""
for variant in "${variants[@]}"; do
  job_script="logs/${variant}_job.sh"
  cat > "${job_script}" <<EOF
#!/bin/bash
#SBATCH --job-name=${variant}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${GPU}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=logs/${variant}_%j.out
#SBATCH --error=logs/${variant}_%j.err

set -eo pipefail

source /softs/Anaconda3/2024.06/etc/profile.d/conda.sh
conda activate oceanlens

cd /scratch/emboulaalam/OceanLens_git

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
EOF

  if [[ "${variant}" == v5* ]]; then
    cat >> "${job_script}" <<EOF

python scripts/train.py \\
  --variant ${variant} \\
  --phase fm \\
  --gpu 0
EOF
  else
    cat >> "${job_script}" <<EOF

python scripts/train.py \\
  --variant ${variant} \\
  --phase fm \\
  --gpu 0 \\
  --cno_ckpt ${CNO_CKPT}
EOF
  fi

  chmod +x "${job_script}"

  if [[ -n "${previous_job}" ]]; then
    job_id=$(sbatch --parsable --dependency=afterok:${previous_job} "${job_script}")
  else
    job_id=$(sbatch --parsable "${job_script}")
  fi
  echo "${variant}: ${job_id}"
  previous_job="${job_id}"
done
