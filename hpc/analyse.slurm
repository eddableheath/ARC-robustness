#!/bin/bash
# Stage 2: analyse one cell per array element.
#
# Assumes stage 1 has already written every checkpoint. Cells whose result exists
# are skipped, so this script is safe to resubmit verbatim after a partial
# failure.
#
# Usage: sbatch hpc/analyse.slurm <grid> [stride]

#SBATCH --job-name=arc-analyse
#SBATCH --output=hpc/logs/analyse-%A_%a.out
#SBATCH --error=hpc/logs/analyse-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
# No --gpus: the attack and feature extraction are a few seconds on CPU, while
# Ollivier curvature solves an exact LP per edge and dominates the runtime.
# Holding a GH200 idle for hours of scipy would waste the allocation.
#SBATCH --account=CHANGEME
#SBATCH --partition=CHANGEME

set -euo pipefail

GRID="${1:?usage: sbatch hpc/analyse.slurm <grid> [stride]}"
STRIDE="${2:-0}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p hpc/logs

echo "host      $(hostname)"
echo "grid      ${GRID}"
echo "array id  ${SLURM_ARRAY_TASK_ID:-<none>}"
echo "cpus      ${SLURM_CPUS_PER_TASK:-1}"
echo "started   $(date -Is)"

# BLAS threads must be pinned to 1. Every worker in the Ollivier pool would
# otherwise spawn its own thread pool, oversubscribing the node and running
# *slower* than a single-threaded run.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

uv run python scripts/run_sweep.py run \
    --grid "${GRID}" \
    --stride "${STRIDE}" \
    --device cpu \
    --n-jobs "${SLURM_CPUS_PER_TASK:-1}"

echo "finished  $(date -Is)"
