# OceanLens Website Plot Requirements

This file tracks the figures needed to replace the placeholders in the website.

## Per Version

For each trained version, export the same diagnostics for every variable:

- `presentation.png`: HR / CNO / CNO+FM
- `residuals.png`: HR-LR / CNO residual / FM residual
- `errors.png`: CNO error / FM error
- `loss.png`: train and validation curves from TensorBoard
- `metrics.csv`: RMSE, bias, residual std ratio, EKE score, re-coarsening error

Variables:

- `thetao`
- `so`
- `zos`
- `uo`
- `vo`

Versions:

- `v4`
- `v4_s1_logit_t`
- `v4_s2_independent`
- `v4_s3_no_attn`
- `v4_s4_grad_mu`
- `v4_s1_recoarsen_ft`
- `v4_s1_cfg_ft`
- `v4_s1_regional_ft`
- `v5_fm_only`
- `v6_full_geo_phys`
- `v7a_swin`
- `v8a_dit_pixel`
- `v9a_songunet_edm`
- `v9b_songunet_bridge`
- `v9c_dit_cfg_recoarsen`

## Web Export Style

- PNG width: 2800-3400 px.
- DPI: 180 or higher.
- Use consistent color limits per variable and diagnostic type.
- Keep titles explicit: version, variable, date, solver, steps, sigma, ensemble, tile, overlap.
- Avoid tiny global maps inside large whitespace; map panels must fill most of the canvas.
- Save into:

```text
website/assets/figures/<version>/<variable>/
```

## Local Inventory On 2026-05-06

Already available locally:

- v4 family: `thetao` residual plots for v4, S1, S2, S3, S4, v5.
- fine-tuning: `thetao` residual plots for recoarsen, CFG and regional FT.
- v6: `thetao` residual plots.
- v7: `thetao` residual plot.
- v8: residual plots for `thetao`, `so`, `zos`, `uo`, `vo` for multiple tile/overlap scenarios.

Missing locally:

- HR/CNO/CNO+FM presentation plots for most versions.
- CNO error / FM error plots for most versions.
- TensorBoard loss curves.
- Metrics table for all versions on a common day/domain.
- v9 final plots and metrics.

LIR access note:

- `ssh lir` timed out from this machine on 2026-05-06, so the remote inventory still needs to be rerun when the frontal is reachable.
