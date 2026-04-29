# OceanLens On-Demand Super-Resolution

Fast EDITO demo process for OceanLens surface super-resolution.

The process reads one LR NetCDF from S3, runs CNO `v2_loggrad` + FM `v4_s1_logit_t`, and writes a super-resolved NetCDF plus selected PNG thumbnails.

## Inputs

Environment variables:

| Variable | Default | Description |
|---|---:|---|
| `LR_FILE_URL` | required | S3 or HTTPS URL of the LR NetCDF |
| `S3_OUTPUT_FOLDER` | `AWS_BUCKET_NAME/OceanLens` | Output bucket/prefix |
| `DOMAIN` | `ibi` | One of `ibi`, `med`, `global` |
| `PLOT_VARIABLES` | `thetao,so,zos,speed` | Comma-separated thumbnail variables |
| `N_STEPS` | `20` | Euler steps for the FM solver |
| `UPSCALE_FACTOR` | `15` | LR-to-HR spatial factor |
| `TILE_SIZE` | `512` | Inference tile size |
| `TILE_OVERLAP` | `64` | Tile overlap used for blending |

The LR NetCDF must contain the five surface variables:

```text
thetao, so, zos, uo, vo
```

## Domains

```text
ibi:    lat 26..56,   lon -20..13
med:    lat 30..46.5, lon -6..37
global: full input domain
```

## Outputs

The process writes:

```text
OceanLens_<date>_<domain>.nc
thetao.png
so.png
zos.png
speed.png
metadata.json
```

Only the PNGs requested in `PLOT_VARIABLES` are generated.

## Model Assets

Weights and config are downloaded at runtime from:

```text
s3://project-moi-ai/OceanLens/demo/v1/
```

Expected layout:

```text
checkpoints/cno_v2_loggrad.ckpt
checkpoints/fm_v4_s1_logit_t.ckpt
configs/base.yaml
configs/variants/v4_s1_logit_t.yaml
norm_stats.json
```

## Example

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...
export AWS_S3_ENDPOINT=minio.dive.edito.eu

export LR_FILE_URL=s3://my-bucket/OceanLens/input_lr.nc
export S3_OUTPUT_FOLDER=project-moi-ai/OceanLens/demo-runs
export DOMAIN=ibi
export PLOT_VARIABLES=thetao,so,zos,speed

python run_oceanlens_inference.py
```
