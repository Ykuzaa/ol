#!/usr/bin/env bash
# Download GLORYS12 daily surface fields from Copernicus Marine month by month.
#
# Required: copernicusmarine CLI configured with valid credentials.

set -euo pipefail

OUT_DIR="${OUT_DIR:-/ec/res4/scratch/fra0606/work/OceanLens/data/raw_monthly}"
DATASET_ID="${DATASET_ID:-cmems_mod_glo_phy_my_0.083deg_P1D-m}"
START_YEAR="${START_YEAR:-1994}"
END_YEAR="${END_YEAR:-2004}"
JOBS="${JOBS:-4}"

# Empty bounds means global download. Set these env vars to subset a region.
LON_MIN="${LON_MIN:-}"
LON_MAX="${LON_MAX:-}"
LAT_MIN="${LAT_MIN:-}"
LAT_MAX="${LAT_MAX:-}"

mkdir -p "${OUT_DIR}"

download_month() {
  local year="$1"
  local month="$2"
  local month2
  month2="$(printf "%02d" "${month}")"

  local start="${year}-${month2}-01T00:00:00"
  local next_year="${year}"
  local next_month=$((month + 1))
  if [[ "${next_month}" -eq 13 ]]; then
    next_month=1
    next_year=$((year + 1))
  fi
  local next_month2
  next_month2="$(printf "%02d" "${next_month}")"
  local end="${next_year}-${next_month2}-01T00:00:00"

  local year_dir="${OUT_DIR}/${year}"
  local output_file="glorys_${year}_${month2}.nc"
  local output_path="${year_dir}/${output_file}"
  mkdir -p "${year_dir}"

  if [[ -s "${output_path}" ]]; then
    echo "[skip] ${output_path}"
    return 0
  fi

  echo "[download] ${year}-${month2} -> ${output_path}"
  local cmd=(
    copernicusmarine subset
    --dataset-id "${DATASET_ID}"
    --variable thetao
    --variable so
    --variable zos
    --variable uo
    --variable vo
    --minimum-depth 0
    --maximum-depth 1
    --start-datetime "${start}"
    --end-datetime "${end}"
    --output-directory "${year_dir}"
    --output-filename "${output_file}"
    --overwrite
  )

  if [[ -n "${LON_MIN}" && -n "${LON_MAX}" && -n "${LAT_MIN}" && -n "${LAT_MAX}" ]]; then
    cmd+=(
      --minimum-longitude "${LON_MIN}"
      --maximum-longitude "${LON_MAX}"
      --minimum-latitude "${LAT_MIN}"
      --maximum-latitude "${LAT_MAX}"
    )
  fi

  "${cmd[@]}"
}

export -f download_month
export OUT_DIR DATASET_ID LON_MIN LON_MAX LAT_MIN LAT_MAX

tasks_file="$(mktemp)"
trap 'rm -f "${tasks_file}"' EXIT

for year in $(seq "${START_YEAR}" "${END_YEAR}"); do
  for month in $(seq 1 12); do
    printf "%s %s\n" "${year}" "${month}" >> "${tasks_file}"
  done
done

xargs -n 2 -P "${JOBS}" bash -c 'download_month "$0" "$1"' < "${tasks_file}"

echo "[done] downloads in ${OUT_DIR}"
