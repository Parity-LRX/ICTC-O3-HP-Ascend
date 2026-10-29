#!/usr/bin/env bash
#SBATCH --job-name=ictc-ascend
#SBATCH --partition=HW_910b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=npu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=ictc-%j.log
# Site example. Override partition/account/resources with sbatch arguments.
set -euo pipefail
: "${ICTC_ENV:?Set ICTC_ENV to the absolute Python environment path}"
: "${ICTC_SITE_ENV:?Set ICTC_SITE_ENV to an absolute shell script that sources CANN and site libraries}"
source "$ICTC_SITE_ENV"
export PATH="$ICTC_ENV/bin:$PATH"
if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    export ASCEND_RT_VISIBLE_DEVICES="$SLURM_JOB_GPUS"
fi
export OMP_NUM_THREADS="${ICTC_OMP_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache/ictc}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"
if [[ $# -eq 0 ]]; then
    printf 'Pass a command and its arguments after the batch filename.\n' >&2
    exit 2
fi
exec "$@"
