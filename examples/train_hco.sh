#!/usr/bin/env bash
# Train a short-range H/C/O potential inside an allocated Ascend job.
# Usage: bash train_hco.sh DATA_DIR CHECKPOINT [extra ictc-ascend-train arguments]
set -euo pipefail
if [[ $# -lt 2 ]]; then
    printf 'Usage: bash train_hco.sh DATA_DIR CHECKPOINT [extra arguments]\n' >&2
    exit 2
fi
data_dir="$1"; checkpoint="$2"
shift 2
e0_csv="${E0_CSV:-$data_dir/fitted_E0.csv}"
mapfile -t e0 < <(python - "$e0_csv" <<'PY'
import csv
import sys
with open(sys.argv[1], newline='') as stream:
    rows = list(csv.DictReader(stream))
if not rows:
    raise ValueError('Expected a nonempty CSV with Atom,E0 columns')
print(','.join(str(int(row['Atom'])) for row in rows))
print(','.join(str(float(row['E0'])) for row in rows))
PY
)
[[ ${#e0[@]} -eq 2 ]] || { printf 'Could not read E0 from %s\n' "$e0_csv" >&2; exit 2; }
mkdir -p "$(dirname "$checkpoint")"
exec ictc-ascend-train \
    --data-dir "$data_dir" --checkpoint "$checkpoint" \
    --atomic-energy-keys="${e0[0]}" --atomic-energy-values="${e0[1]}" \
    --channels 64 --lmax 2 --max-ell 2 --num-interaction 2 --correlation 3 \
    --max-radius 4.5 --polynomial-cutoff-p 5 \
    --product-backend ictd-bridge-u --angular-basis ictd --freeze-conv-tp-weight \
    --device npu:0 --dtype float32 --ascendc --ascendc-mode performance \
    --operator-plan generated --generated-layout fused_pack \
    --optimizer adamw --lr 1e-3 --loss mse --energy-weight 1 --force-weight 10 \
    --batch-size 4 --epochs 100 --num-workers 2 --seed 42 \
    --long-range-mode none --long-range-dispersion-mode none "$@"
