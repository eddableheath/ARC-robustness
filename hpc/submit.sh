#!/bin/bash
# Submit both stages of a grid, with the analysis array held until training
# succeeds.
#
# Array bounds are read from the grid itself rather than written by hand, so they
# cannot drift out of step with a grid definition that has gained cells.
#
# Usage: ./hpc/submit.sh <grid> [max_array_size]

set -euo pipefail

GRID="${1:?usage: ./hpc/submit.sh <grid> [max_array_size]}"
MAX_ARRAY="${2:-1000}"

cd "$(dirname "$0")/.."
mkdir -p hpc/logs

read -r N_CELLS N_MODELS <<<"$(uv run python -c "
from arc_robustness.experiments import build_grid
grid = build_grid('${GRID}')
print(len(grid), len(grid.unique_models()))
")"

echo "grid ${GRID}: ${N_CELLS} cells, ${N_MODELS} checkpoints"

# Write the manifest before submitting: it is the only record of which index
# meant which config, and it must exist even if the sweep is later interrupted.
uv run python scripts/run_sweep.py manifest --grid "${GRID}"

# Chunk with --stride when a stage exceeds the cluster's MaxArraySize.
if (( N_CELLS > MAX_ARRAY )); then
    CELL_STRIDE="${MAX_ARRAY}"
    CELL_UPPER=$(( MAX_ARRAY - 1 ))
    echo "  ${N_CELLS} cells > MaxArraySize ${MAX_ARRAY}: striding by ${MAX_ARRAY}"
else
    CELL_STRIDE=0
    CELL_UPPER=$(( N_CELLS - 1 ))
fi

TRAIN_JOB=$(sbatch --parsable \
    --array="0-$(( N_MODELS - 1 ))" \
    hpc/train.slurm "${GRID}" 0)
echo "  stage 1 (train)   job ${TRAIN_JOB}  array 0-$(( N_MODELS - 1 ))"

# afterok, not afterany: analysing against a missing or half-written checkpoint
# would produce results that look valid and are not.
ANALYSE_JOB=$(sbatch --parsable \
    --dependency="afterok:${TRAIN_JOB}" \
    --array="0-${CELL_UPPER}" \
    hpc/analyse.slurm "${GRID}" "${CELL_STRIDE}")
echo "  stage 2 (analyse) job ${ANALYSE_JOB}  array 0-${CELL_UPPER}  held on ${TRAIN_JOB}"

echo
echo "watch:   squeue -u \$USER"
echo "results: ls results/${GRID}/*.npz | wc -l   # of ${N_CELLS}"
