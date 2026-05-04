# OceanLens

Unified codebase for deterministic and probabilistic ocean downscaling.

OceanLens compares three trained downscaling strategies:

- `v1`: CNO direct field, then Flow Matching conditioned on `[mu, LR, mask]`
- `v2`: CNO residual field, then Flow Matching conditioned on `[mu]`
- `v4`: CNO residual field with log-gradient loss, then Flow Matching conditioned on `[mu]`
- `v5`: Flow Matching residual field conditioned on `[LR]`, with prediction `LR + FM`
- `ablation`: Flow Matching without CNO, conditioned only on `[LR]`

The code stays close to two reference ideas:

- Convolutional Neural Operator (CNO) for the deterministic branch
- Flow Matching for the stochastic residual branch

The goal is to keep the core models simple, reproducible, and easy to compare
with the same data splits and training protocol.

## Repository Layout

```text
OceanLens_git/
├── configs/
│   ├── base.yaml
│   └── variants/
│       ├── v1.yaml
│       ├── v2.yaml
│       ├── v2_loggrad.yaml
│       ├── v3.yaml
│       └── ablation.yaml
├── scripts/
│   ├── train.py
│   ├── validate.py
│   ├── compute_metrics.py
│   ├── compare_lr_v2.py
│   ├── infer_compare.py
│   ├── plot_ke_spectra.py
│   ├── plot_report.py
│   ├── launch_parallel.sh
│   └── slurm/
│       ├── train_ablation.sh
│       ├── train_cno.sh
│       ├── finetune_v2_loggrad.sh
│       ├── finetune_v3_fm.sh
│       ├── evaluate_comparison.sh
│       └── train_fm.sh
└── src/oceanlens/
    ├── data/
    │   ├── datamodule.py
    │   └── dataset.py
    ├── eval/
    │   ├── __init__.py
    │   └── metrics.py
    ├── losses/
    │   ├── cno.py
    │   └── fm.py
    ├── models/
    │   ├── cno.py
    │   ├── fm_unet.py
    │   ├── system.py
    │   └── variants.py
    └── utils.py
```

## Variants

| Variant | CNO mode | FM condition | FM channels |
| --- | --- | --- | --- |
| `v1` | direct | `[x_t, mu, LR, mask]` | 16 |
| `v2` | residual | `[x_t, mu]` | 10 |
| `v2_loggrad` | residual | `[x_t, mu]` | 10 |
| `v3` | residual | `[x_t, mu]` | 10 |
| `v4` | residual + loggrad | `[x_t, mu]` | 10 |
| `v5` | none | `[x_t, LR]` | 10 |
| `ablation` | none | `[x_t, LR]` | 10 |

`v2_loggrad` is a fine-tuning variant of V2 that adds a log-gradient temperature loss during the CNO phase.

`v3` keeps the `v2_loggrad` CNO frozen and fine-tunes only the FM branch from
the original V2 FM checkpoint. The FM loss is unchanged.

`v4` trains the CNO to predict the residual `HR - LR_up` with the log-gradient
temperature loss enabled. Its deterministic field is `mu = LR_up + CNO(LR_up)`.
The FM branch is conditioned on `mu` and learns the residual `HR - mu`.

`v5` removes the CNO. The FM branch is conditioned on `LR_up`, learns the
residual `HR - LR_up`, and samples are reconstructed as `LR_up + FM_residual`.

## Training

Train CNO first for CNO-based variants:

```bash
python scripts/train.py --variant v1 --phase cno --gpu 0
python scripts/train.py --variant v2 --phase cno --gpu 0
```

Then train Flow Matching using the corresponding CNO checkpoint:

```bash
python scripts/train.py \
  --variant v1 \
  --phase fm \
  --cno_ckpt /scratch/emboulaalam/OceanLens_git/runs/v1/cno/checkpoints/last.ckpt \
  --gpu 0
```

Train the ablation without CNO:

```bash
python scripts/train.py --variant ablation --phase fm --gpu 0
```

For a quick two-GPU interactive run:

```bash
bash scripts/launch_parallel.sh cno v1 v2
bash scripts/launch_parallel.sh fm v1 v2
```

SLURM launchers are available under `scripts/slurm/`.

## V2 Log-Gradient Fine-Tuning

This is the current follow-up experiment.

The idea is not to change the V2 architecture. We start from the already
optimized V2 CNO checkpoint and add a second loss term that focuses on thermal
fronts:

```text
V2:
mu = LR + CNO(LR)
prediction = mu + FM residual
```

For the first quick test, only the deterministic CNO branch is fine-tuned. The
Flow Matching checkpoint is kept unchanged. This isolates the effect of the new
temperature-front loss.

The new objective is:

```text
L_total = L_CNO + lambda * L_loggradT
```

with:

```text
L_loggradT =
| log(|grad(thetao_pred)| + eps) - log(|grad(thetao_HR)| + eps) |
```

Current config:

```text
variant: v2_loggrad
base checkpoint: runs/v2/cno/checkpoints/last.ckpt
lr: 5e-5
max_epochs: 25
thetao weight: 1.5
log-gradient weight: 0.05
```

### Launch on an interactive GPU node

From the allocated GPU shell:

```bash
cd /scratch/emboulaalam/OceanLens_git
module load anaconda3/2024.06
eval "$(conda shell.bash hook)"
conda activate oceanlens
```

Then run:

```bash
python scripts/train.py \
  --variant v2_loggrad \
  --phase cno \
  --gpu 0 \
  --cno_ckpt /scratch/emboulaalam/OceanLens_git/runs/v2/cno/checkpoints/last.ckpt
```

The new checkpoint is written to:

```text
/scratch/emboulaalam/OceanLens_git/runs/v2_loggrad/cno/checkpoints/last.ckpt
```

### Launch with SLURM

From `/scratch/emboulaalam/OceanLens_git`:

```bash
sbatch scripts/slurm/finetune_v2_loggrad.sh
```

Monitor with:

```bash
squeue -u emboulaalam
tail -f logs/v2_loggrad_<JOBID>.out
tail -f logs/v2_loggrad_<JOBID>.err
```

### Compare V2 vs V2-loggrad

After fine-tuning, run inference for the original V2 and the fine-tuned CNO:

```bash
python scripts/infer_compare.py \
  --variants v2 v2_loggrad \
  --max_days 1 \
  --start_index 0 \
  --n_steps 20 \
  --ensemble_members 1 \
  --tile_size 128 \
  --tile_overlap 16 \
  --model_upsample_mode nearest \
  --baseline_upsample_mode bilinear \
  --device cuda \
  --save_npz \
  --output_dir /scratch/emboulaalam/OceanLens_git/results/v2_vs_v2_loggrad \
  --cno_ckpt v2=/scratch/emboulaalam/OceanLens_git/runs/v2/cno/checkpoints/last.ckpt \
  --fm_ckpt v2=/scratch/emboulaalam/OceanLens_git/runs/v2/fm/checkpoints/last.ckpt \
  --cno_ckpt v2_loggrad=/scratch/emboulaalam/OceanLens_git/runs/v2_loggrad/cno/checkpoints/last.ckpt \
  --fm_ckpt v2_loggrad=/scratch/emboulaalam/OceanLens_git/runs/v2/fm/checkpoints/last.ckpt
```

Then create plots:

```bash
python scripts/plot_report.py \
  --comparison_dir /scratch/emboulaalam/OceanLens_git/results/v2_vs_v2_loggrad \
  --variables thetao zos speed \
  --projection auto
```

And kinetic-energy spectra:

```bash
python scripts/plot_ke_spectra.py \
  --comparison_dir /scratch/emboulaalam/OceanLens_git/results/v2_vs_v2_loggrad \
  --variants hr v2 v2_loggrad
```

Main files to inspect:

```text
results/v2_vs_v2_loggrad/summary_by_variable.csv
results/v2_vs_v2_loggrad/summary_currents.csv
results/v2_vs_v2_loggrad/figures/ranking_table.md
results/v2_vs_v2_loggrad/figures/summary_rmse_by_variable.png
results/v2_vs_v2_loggrad/figures/errors_thetao_2019-01-01.png
results/v2_vs_v2_loggrad/figures/ke_spectra/ke_spectrum_mean.png
```

Interpretation:

- `thetao` RMSE/MAE should decrease if the fine-tuning helps.
- `thetao` error maps should show better frontal placement.
- speed, vorticity and KE spectra are secondary checks to see if improving
  thermal fronts also helps the dynamics.
- If this CNO-only fine-tuning is useful, the next step is to fine-tune the FM
  branch with the new `v2_loggrad` CNO frozen.

## V3 FM Fine-Tuning

V3 adapts the FM to the new deterministic field produced by the fine-tuned CNO:

```text
V2:
mu_old = LR + CNO_v2(LR)
FM_v2 learns HR - mu_old

V3:
mu_new = LR + CNO_v2_loggrad(LR)
FM_v3 starts from FM_v2 and fine-tunes on HR - mu_new
```

The FM objective is not changed. This isolates the effect of adapting the
stochastic residual model to the new `mu`.

Current config:

```text
variant: v3
CNO checkpoint: runs/v2_loggrad/cno/checkpoints/last.ckpt
FM starting checkpoint: runs/v2/fm/checkpoints/last.ckpt
FM lr: 5e-5
FM max_epochs: 50
loss: standard OT-CFM
```

### Copy code from local to LIR

From the local machine:

```bash
rsync -avz ~/OceanLens_git/configs/variants/v3.yaml \
  emboulaalam@lir:/scratch/emboulaalam/OceanLens_git/configs/variants/v3.yaml

rsync -avz \
  ~/OceanLens_git/scripts/train.py \
  ~/OceanLens_git/scripts/plot_report.py \
  ~/OceanLens_git/scripts/plot_ke_spectra.py \
  emboulaalam@lir:/scratch/emboulaalam/OceanLens_git/scripts/

rsync -avz ~/OceanLens_git/src/oceanlens/models/system.py \
  emboulaalam@lir:/scratch/emboulaalam/OceanLens_git/src/oceanlens/models/system.py

rsync -avz ~/OceanLens_git/scripts/slurm/finetune_v3_fm.sh \
  emboulaalam@lir:/scratch/emboulaalam/OceanLens_git/scripts/slurm/finetune_v3_fm.sh
```

### Launch V3 FM fine-tuning on LIR

From LIR:

```bash
cd /scratch/emboulaalam/OceanLens_git
sbatch scripts/slurm/finetune_v3_fm.sh
```

Interactive GPU alternative:

```bash
cd /scratch/emboulaalam/OceanLens_git
conda activate oceanlens

python scripts/train.py \
  --variant v3 \
  --phase fm \
  --gpu 0 \
  --cno_ckpt /scratch/emboulaalam/OceanLens_git/runs/v2_loggrad/cno/checkpoints/last.ckpt \
  --fm_ckpt /scratch/emboulaalam/OceanLens_git/runs/v2/fm/checkpoints/last.ckpt
```

The V3 FM checkpoint is written to:

```text
/scratch/emboulaalam/OceanLens_git/runs/v3/fm/checkpoints/last.ckpt
```

### Inference: V2 vs V2-loggrad vs V3

The naming convention is:

```text
v2         = original V2: CNO_v2 + FM_v2
v2_loggrad = CNO_v2_loggrad + unchanged FM_v2
v3         = CNO_v2_loggrad + fine-tuned FM_v3
```

Clean the previous result folder if needed:

```bash
rm -rf /scratch/emboulaalam/OceanLens_git/results/v2_v2loggrad_v3
```

Run inference. For a quick smoke test use `--max_days 1`; for the current
multi-day validation use `--max_days 8`:

```bash
python scripts/infer_compare.py \
  --variants v2 v2_loggrad v3 \
  --max_days 8 \
  --start_index 0 \
  --n_steps 20 \
  --ensemble_members 1 \
  --tile_size 128 \
  --tile_overlap 16 \
  --model_upsample_mode nearest \
  --baseline_upsample_mode bilinear \
  --device cuda \
  --save_npz \
  --output_dir /scratch/emboulaalam/OceanLens_git/results/v2_v2loggrad_v3 \
  --cno_ckpt v2=/scratch/emboulaalam/OceanLens_git/runs/v2/cno/checkpoints/last.ckpt \
  --fm_ckpt v2=/scratch/emboulaalam/OceanLens_git/runs/v2/fm/checkpoints/last.ckpt \
  --cno_ckpt v2_loggrad=/scratch/emboulaalam/OceanLens_git/runs/v2_loggrad/cno/checkpoints/last.ckpt \
  --fm_ckpt v2_loggrad=/scratch/emboulaalam/OceanLens_git/runs/v2/fm/checkpoints/last.ckpt \
  --cno_ckpt v3=/scratch/emboulaalam/OceanLens_git/runs/v2_loggrad/cno/checkpoints/last.ckpt \
  --fm_ckpt v3=/scratch/emboulaalam/OceanLens_git/runs/v3/fm/checkpoints/last.ckpt
```

This writes the usual RMSE/correlation diagnostics and the direct thermal-front
metric:

```text
metrics_loggrad.csv
summary_loggrad.csv
```

The log-gradient metric is computed on `thetao` as:

```text
G(T) = log(|grad T| + eps)
RMSE_loggrad = RMSE(G(thetao_pred), G(thetao_HR))
skill_loggrad = 1 - RMSE_loggrad(model) / RMSE_loggrad(LR)
```

It is meant to answer whether the model improves thermal-front sharpness and
placement, not only pointwise temperature RMSE.

Create field/error plots and ranking tables:

```bash
python scripts/plot_report.py \
  --comparison_dir /scratch/emboulaalam/OceanLens_git/results/v2_v2loggrad_v3 \
  --variables thetao zos speed \
  --variants lr v2 v2_loggrad v3 \
  --projection auto
```

If `summary_loggrad.csv` exists, `plot_report.py` also writes:

```text
figures/summary_loggrad_thetao_rmse.png
figures/summary_loggrad_thetao_skill.png
```

Create kinetic-energy spectra:

```bash
python scripts/plot_ke_spectra.py \
  --comparison_dir /scratch/emboulaalam/OceanLens_git/results/v2_v2loggrad_v3 \
  --variants hr lr v2 v2_loggrad v3
```

Inspect the main outputs on LIR:

```bash
cat /scratch/emboulaalam/OceanLens_git/results/v2_v2loggrad_v3/figures/ranking_table.md
cat /scratch/emboulaalam/OceanLens_git/results/v2_v2loggrad_v3/summary_by_variable.csv
cat /scratch/emboulaalam/OceanLens_git/results/v2_v2loggrad_v3/summary_currents.csv
cat /scratch/emboulaalam/OceanLens_git/results/v2_v2loggrad_v3/summary_loggrad.csv
cat /scratch/emboulaalam/OceanLens_git/results/v2_v2loggrad_v3/figures/ke_spectra/ke_spectrum_summary.md
```

### Copy results from LIR to local without large NPZ files

Dry run:

```bash
rsync -avzn \
  --exclude '*.npz' \
  emboulaalam@lir:/scratch/emboulaalam/OceanLens_git/results/v2_v2loggrad_v3/ \
  ~/OceanLens_results/v2_v2loggrad_v3/
```

Actual copy:

```bash
rsync -avz \
  --exclude '*.npz' \
  emboulaalam@lir:/scratch/emboulaalam/OceanLens_git/results/v2_v2loggrad_v3/ \
  ~/OceanLens_results/v2_v2loggrad_v3/
```

This keeps the CSV, Markdown summaries, metadata, and figures, but skips the
large `day_*.npz` inference arrays.

## Clean V3 Inference Workflow for Meeting

This section is the recommended clean workflow for checking what the V3 CNO and
FM branches are doing.

Notation:

```text
LR_up                      = low-resolution input interpolated to the HR grid
CNO correction             = mu_CNO - LR_up
mu_CNO                     = LR_up + CNO(LR_up)
FM residual correction     = output of the FM branch
final CNO+FM prediction    = mu_CNO + FM residual correction
final correction           = CNO+FM prediction - LR_up
```

The FM branch is trained to predict the residual on top of `mu_CNO`. Therefore,
the most important diagnostic is:

```text
FM target       = HR - mu_CNO
FM prediction   = FM residual correction
```

If the FM is useful, `FM prediction` should be spatially aligned with
`HR - mu_CNO`, and the final error should decrease:

```text
RMSE(CNO+FM, HR) < RMSE(mu_CNO, HR)
```

### Copy the clean scripts to LIR

From the local/px machine:

```bash
rsync -avP \
  /homelocal/emboulaalam/OceanLens_git/scripts/run_v3_inference_flexible.sh \
  /homelocal/emboulaalam/OceanLens_git/scripts/plot_simple_inference.py \
  /homelocal/emboulaalam/OceanLens_git/scripts/plot_cno_fm_decomposition.py \
  /homelocal/emboulaalam/OceanLens_git/scripts/plot_fm_output_vs_hr_lr.py \
  emboulaalam@lir:/scratch/emboulaalam/OceanLens_git/scripts/
```

### Archive old inference results without touching checkpoints

On LIR:

```bash
cd /scratch/emboulaalam/OceanLens_git

mkdir -p results_archive_before_meeting
find results -mindepth 1 -maxdepth 1 -type d -exec mv {} results_archive_before_meeting/ \;
```

This only moves old inference outputs from `results/`. It does not touch:

```text
runs/v2_loggrad/cno/checkpoints/last.ckpt
runs/v3/fm/checkpoints/last.ckpt
```

If you really want to delete old results instead of archiving them:

```bash
rm -rf results/*
```

### Run normal V3 CNO+FM inference, ensemble size 1

On LIR:

```bash
cd /scratch/emboulaalam/OceanLens_git
conda activate oceanlens

MODE=cno_fm \
RESULT_NAME=meeting_v3_cno_fm_ens1 \
MAX_DAYS=1 \
START_INDEX=0 \
N_STEPS=20 \
ENSEMBLE_MEMBERS=1 \
TILE_SIZE=128 \
TILE_OVERLAP=16 \
bash scripts/run_v3_inference_flexible.sh
```

This writes:

```text
/scratch/emboulaalam/OceanLens_git/results/meeting_v3_cno_fm_ens1
```

### Run normal V3 CNO+FM inference, ensemble mean 4

On LIR:

```bash
cd /scratch/emboulaalam/OceanLens_git
conda activate oceanlens

MODE=cno_fm \
RESULT_NAME=meeting_v3_cno_fm_ens4 \
MAX_DAYS=1 \
START_INDEX=0 \
N_STEPS=20 \
ENSEMBLE_MEMBERS=4 \
TILE_SIZE=128 \
TILE_OVERLAP=16 \
bash scripts/run_v3_inference_flexible.sh
```

This writes:

```text
/scratch/emboulaalam/OceanLens_git/results/meeting_v3_cno_fm_ens4
```

### Plot residual decomposition

Use this to see how the full correction is split between CNO and FM:

```bash
RESULT_DIR=/scratch/emboulaalam/OceanLens_git/results/meeting_v3_cno_fm_ens1

python scripts/plot_simple_inference.py \
  --comparison_dir "$RESULT_DIR" \
  --variant v3 \
  --variables thetao \
  --panels hr-lr cno-lr fm pred-lr \
  --output_dir "$RESULT_DIR/figures/simple_residuals"
```

The four panels are:

```text
Target correction: HR - LR
CNO correction: mu_CNO - LR
FM residual correction
Final correction: CNO+FM - LR
```

### Plot whether FM learns its true residual target

Use this to check if the FM output is aligned with the true remaining CNO error:

```bash
RESULT_DIR=/scratch/emboulaalam/OceanLens_git/results/meeting_v3_cno_fm_ens1

python scripts/plot_simple_inference.py \
  --comparison_dir "$RESULT_DIR" \
  --variant v3 \
  --variables thetao \
  --panels hr-mu fm err-before-fm err-after-fm \
  --output_dir "$RESULT_DIR/figures/fm_diagnostic"
```

The four panels are:

```text
FM target: HR - mu_CNO
FM residual correction
Error before FM: HR - mu_CNO
Error after FM: HR - CNO+FM
```

If the FM is useful, the `FM residual correction` should resemble
`HR - mu_CNO`, and `Error after FM` should be smaller than `Error before FM`.

### Plot physical fields

Use this to inspect the final maps in physical units:

```bash
RESULT_DIR=/scratch/emboulaalam/OceanLens_git/results/meeting_v3_cno_fm_ens1

python scripts/plot_simple_inference.py \
  --comparison_dir "$RESULT_DIR" \
  --variant v3 \
  --variables thetao \
  --panels lr hr mu pred \
  --output_dir "$RESULT_DIR/figures/fields"
```

The four panels are:

```text
LR input
HR truth
mu_CNO
CNO+FM prediction
```

### Repeat plots for the ensemble mean 4 result

For the ensemble mean run, only change `RESULT_DIR`:

```bash
RESULT_DIR=/scratch/emboulaalam/OceanLens_git/results/meeting_v3_cno_fm_ens4

python scripts/plot_simple_inference.py \
  --comparison_dir "$RESULT_DIR" \
  --variant v3 \
  --variables thetao \
  --panels hr-lr cno-lr fm pred-lr \
  --output_dir "$RESULT_DIR/figures/simple_residuals"

python scripts/plot_simple_inference.py \
  --comparison_dir "$RESULT_DIR" \
  --variant v3 \
  --variables thetao \
  --panels hr-mu fm err-before-fm err-after-fm \
  --output_dir "$RESULT_DIR/figures/fm_diagnostic"

python scripts/plot_simple_inference.py \
  --comparison_dir "$RESULT_DIR" \
  --variant v3 \
  --variables thetao \
  --panels lr hr mu pred \
  --output_dir "$RESULT_DIR/figures/fields"
```

### Inspect metrics on LIR

```bash
cat /scratch/emboulaalam/OceanLens_git/results/meeting_v3_cno_fm_ens1/summary_by_variable.csv
cat /scratch/emboulaalam/OceanLens_git/results/meeting_v3_cno_fm_ens4/summary_by_variable.csv

cat /scratch/emboulaalam/OceanLens_git/results/meeting_v3_cno_fm_ens1/metadata.json
cat /scratch/emboulaalam/OceanLens_git/results/meeting_v3_cno_fm_ens4/metadata.json
```

The CSV compares `lr` and `v3`. To compare CNO alone against CNO+FM, use the
saved arrays:

```text
v3_mu         = CNO-only deterministic prediction
v3_fm_output  = FM residual correction
v3            = final CNO+FM prediction
```

### Copy results from LIR to local/px

From the local/px machine:

```bash
rsync -avP emboulaalam@lir:/scratch/emboulaalam/OceanLens_git/results/meeting_v3_cno_fm_ens1/ \
  /homelocal/emboulaalam/meeting_v3_cno_fm_ens1/

rsync -avP emboulaalam@lir:/scratch/emboulaalam/OceanLens_git/results/meeting_v3_cno_fm_ens4/ \
  /homelocal/emboulaalam/meeting_v3_cno_fm_ens4/
```

The key files to check are:

```text
day_2019-01-01.npz
metadata.json
infer_time_memory.txt
summary_by_variable.csv
figures/simple_residuals/
figures/fm_diagnostic/
figures/fields/
```

### Suggested interpretation for the meeting

Start from these checks:

```text
1. Is RMSE(mu_CNO, HR) lower than RMSE(LR, HR)?
2. Is RMSE(CNO+FM, HR) lower than RMSE(mu_CNO, HR)?
3. Is corr(FM residual correction, HR - mu_CNO) positive and meaningful?
4. Does ensemble mean 4 improve over ensemble size 1?
```

Current single-member diagnostics showed:

```text
CNO is strong and explains most of HR - LR.
FM outputs a non-zero residual, but on the checked day it was weakly correlated
with HR - mu_CNO and slightly degraded RMSE compared to CNO alone.
```

Therefore the next decision should be based on `ens4` and more days before
changing architecture.

## Validation

```bash
python scripts/validate.py \
  --variant v1 \
  --phase fm \
  --ckpt /scratch/emboulaalam/OceanLens_git/runs/v1/fm/checkpoints/last.ckpt \
  --cno_ckpt /scratch/emboulaalam/OceanLens_git/runs/v1/cno/checkpoints/last.ckpt
```

## Minimal V3 Inference + Notebook Analysis

Use this workflow for interactive discussion/debugging. The inference script is
kept minimal and reproducible; the notebook is used only for plotting and
analysis.

The actual V3 inference pipeline is:

```text
1. LR is interpolated to the HR grid: LR_up
2. CNO inference:
   cno_residual = CNO(LR_up)
   mu_CNO = LR_up + cno_residual
3. FM inference:
   fm_residual = FM(mu_CNO)
4. Final composition:
   pred_cno_fm = mu_CNO + fm_residual
```

The last step is an addition, not a concatenation.

### Copy files to LIR

From local/px:

```bash
rsync -avP \
  /homelocal/emboulaalam/OceanLens_git/scripts/infer_v3_minimal.py \
  emboulaalam@lir:/scratch/emboulaalam/OceanLens_git/scripts/

rsync -avP \
  /homelocal/emboulaalam/OceanLens_git/notebooks/analyze_v3_outputs.ipynb \
  emboulaalam@lir:/scratch/emboulaalam/OceanLens_git/notebooks/
```

If `notebooks/` does not exist on LIR:

```bash
ssh emboulaalam@lir 'mkdir -p /scratch/emboulaalam/OceanLens_git/notebooks'
```

### Run normal V3 CNO+FM inference

On a GPU node on LIR:

```bash
cd /scratch/emboulaalam/OceanLens_git
conda activate oceanlens

RESULT_DIR=/scratch/emboulaalam/OceanLens_git/results/demo_v3_cno_fm_ens1

python scripts/infer_v3_minimal.py \
  --mode cno_fm \
  --output_dir "$RESULT_DIR" \
  --day_index 0 \
  --n_steps 20 \
  --solver euler \
  --ensemble_members 1 \
  --tile_size 128 \
  --tile_overlap 16 \
  --device cuda \
  --cno_ckpt /scratch/emboulaalam/OceanLens_git/runs/v2_loggrad/cno/checkpoints/last.ckpt \
  --fm_ckpt /scratch/emboulaalam/OceanLens_git/runs/v3/fm/checkpoints/last.ckpt
```

This saves:

```text
day_2019-01-01.npz
metadata.json
metrics.csv
```

The `.npz` contains:

```text
hr
lr
mu_cno
cno_residual
fm_residual
pred_cno_fm
target_total_residual
target_fm_residual
error_before_fm
error_after_fm
mask
lat
lon
```

### Run ensemble mean 4

```bash
RESULT_DIR=/scratch/emboulaalam/OceanLens_git/results/demo_v3_cno_fm_ens4

python scripts/infer_v3_minimal.py \
  --mode cno_fm \
  --output_dir "$RESULT_DIR" \
  --day_index 0 \
  --n_steps 20 \
  --solver euler \
  --ensemble_members 4 \
  --tile_size 128 \
  --tile_overlap 16 \
  --device cuda \
  --cno_ckpt /scratch/emboulaalam/OceanLens_git/runs/v2_loggrad/cno/checkpoints/last.ckpt \
  --fm_ckpt /scratch/emboulaalam/OceanLens_git/runs/v3/fm/checkpoints/last.ckpt
```

### Notebook analysis

Open:

```text
notebooks/analyze_v3_outputs.ipynb
```

Set:

```python
RESULT_DIR = Path("/scratch/emboulaalam/OceanLens_git/results/demo_v3_cno_fm_ens1")
```

The notebook can plot:

```text
LR, HR, mu_CNO, CNO+FM
HR - LR
mu_CNO - LR
FM residual
CNO+FM - LR
HR - mu_CNO
HR - CNO+FM
```

The key diagnostic is:

```text
corr(FM residual, HR - mu_CNO)
```

If this correlation is weak and `RMSE(CNO+FM, HR)` is worse than
`RMSE(mu_CNO, HR)`, then the current FM checkpoint is not adding useful
pointwise skill over CNO.

## Inference and Comparison

Generate predictions and simple metrics for `v1`, `v2`, and `ablation`:

```bash
python scripts/infer_compare.py \
  --max_days 4 \
  --tile_size 128 \
  --tile_overlap 16 \
  --baseline_upsample_mode bilinear \
  --model_upsample_mode nearest \
  --save_npz \
  --output_dir results/comparison
```

This writes:

- `metrics_by_variable.csv`
- `metrics_currents.csv`
- `summary_by_variable.csv`
- `summary_currents.csv`
- `metadata.json`
- `day_*.npz` files when `--save_npz` is enabled

Create report figures and a ranking table:

```bash
python scripts/plot_report.py \
  --comparison_dir results/comparison \
  --variables zos thetao speed \
  --projection auto
```

Create a focused LR baseline vs V2 report:

```bash
python scripts/compare_lr_v2.py \
  --comparison_dir results/comparison \
  --variables zos thetao speed \
  --projection auto
```

Create kinetic-energy spectra for HR, LR, and all model variants:

```bash
python scripts/plot_ke_spectra.py \
  --comparison_dir results/comparison \
  --variants hr lr v1 v2 ablation
```

The report focuses on a small set of interpretable diagnostics:

- `MAE` and `RMSE` by variable
- skill score against the upsampled LR baseline
- spatial correlation
- current-speed RMSE
- kinetic-energy RMSE
- vorticity correlation

Positive skill means the model improves over the LR baseline.

On LIR, run the full GPU evaluation job with:

```bash
sbatch scripts/slurm/evaluate_comparison.sh
```

Optional overrides:

```bash
sbatch --export=ALL,MAX_DAYS=8,N_STEPS=30,ENSEMBLE_MEMBERS=4 \
  scripts/slurm/evaluate_comparison.sh
```

The FM U-Net uses attention, so global inference is tiled by default to avoid
GPU out-of-memory errors. Keep `TILE_SIZE=128` unless you have validated a
larger value on the target GPU.

The job writes results by default to:

```text
/scratch/emboulaalam/OceanLens_git/results/comparison_${SLURM_JOB_ID}
```

The final folder contains CSV metrics, `day_*.npz` inference files, figures,
`figures/ranking_table.md`, `figures/lr_vs_v2/lr_vs_v2_summary.md`,
`figures/ke_spectra/ke_spectrum_summary.md`, and `JOB_SUMMARY.md`.

## Current Artifacts

The repository currently contains trained checkpoints under `runs/` for:

- `v1/cno`
- `v1/fm`
- `v2/cno`
- `v2/fm`
- `ablation/fm`

Debug logs are stored under `debug_logs/`.

## Notes

- The evaluation package is intentionally minimal right now.
- `scripts/compute_metrics.py` is an older draft. The current comparison
  workflow is `infer_compare.py` + `plot_report.py` + `plot_ke_spectra.py`.
- `scripts/infer_compare.py` and `scripts/plot_report.py` provide the current
  lightweight evaluation workflow.

## References

- Raonic et al., NeurIPS 2023 - Convolutional Neural Operator
- Lipman et al., 2023 - Flow Matching
- Schusterbauer et al., CVPR 2025 - Diff2Flow
