#!/bin/bash
# Stage 1: produce every checkpoint a grid needs, one per array element.
#
# Indexes over `grid.unique_models()`, not over cells, so each network is trained
# exactly once no matter how many analysis cells share it. Running this to
# completion before stage 2 is what keeps the analysis array from racing to write
# the same .pt file.
#
#   sbatch --array=0-$(($(nproc_models a1)-1)) hpc/train.slurm a1
#
# but prefer ./hpc/submit.sh, which computes the bounds from the grid for you.
#
# Usage: sbatch hpc/train.slurm <grid> [stride]

#SBATCH --job-name=arc-train
#SBATCH --output=hpc/logs/train-%A_%a.out
#SBATCH --error=hpc/logs/train-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
# CONFIRM BEFORE FIRST SUBMISSION: account and partition are site-specific and
# the job will be rejected outright if they are wrong.
#SBATCH --account=CHANGEME
#SBATCH --partition=CHANGEME

set -euo pipefail

GRID="${1:?usage: sbatch hpc/train.slurm <grid> [stride]}"
STRIDE="${2:-0}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p hpc/logs

echo "host      $(hostname)"
echo "grid      ${GRID}"
echo "array id  ${SLURM_ARRAY_TASK_ID:-<none>}"
echo "started   $(date -Is)"

# Fail loudly rather than silently training on CPU for two hours.
uv run python -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('CUDA unavailable — see the torch index note in pyproject.toml')
print('device   ', torch.cuda.get_device_name(0))
"

uv run python scripts/run_sweep.py train \
    --grid "${GRID}" \
    --stride "${STRIDE}" \
    --device cuda

echo "finished  $(date -Is)"
