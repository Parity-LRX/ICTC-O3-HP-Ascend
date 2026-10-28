#!/usr/bin/env bash
# Native reciprocal q/mu/Q training; execute inside an Ascend allocation.
set -euo pipefail
if [[ $# -lt 2 ]]; then
    printf 'Usage: bash train_hco_long_range.sh DATA_DIR CHECKPOINT [extra arguments]\n' >&2
    exit 2
fi
data_dir="$1"; checkpoint="$2"
shift 2
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_hco.sh" "$data_dir" "$checkpoint" \
    --long-range-mode reciprocal-spectral-v1 --long-range-boundary periodic \
    --long-range-reciprocal-backend mesh_fft --long-range-assignment pcs \
    --long-range-mesh-fft-full-ewald --long-range-mesh-fft-reciprocal-only \
    --long-range-smearing-sigma 1.0 --long-range-mesh-size 32 \
    --long-range-max-multipole-l 2 --electrostatics-backend ascend "$@"
