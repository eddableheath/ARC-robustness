# Running a sweep on Isambard-AI

Two stages, in this order. The split is not cosmetic: many analysis cells share
one checkpoint, so training inside the analysis array would have several elements
racing to write the same `.pt` file.

```
stage 1   train.slurm     array over grid.unique_models()   GPU-bound, minutes
stage 2   analyse.slurm   array over grid cells             CPU-bound, longer
```

## First time on a new machine

```bash
uv sync --extra dev
uv run python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

If that prints `False` on a GH200 node, the aarch64 CPU-only wheels were
installed instead of the CUDA ones. See the `[tool.uv.sources]` comment in
`pyproject.toml`; confirm the cluster's CUDA module matches the pinned `cu128`
index before the first large submission. Nothing errors in this failure mode —
jobs simply run on CPU — so check it explicitly.

## Submitting

```bash
# What am I about to run?
uv run python scripts/run_sweep.py list
uv run python scripts/run_sweep.py manifest --grid a1 --show

# Both stages, with stage 2 held until stage 1 succeeds.
./hpc/submit.sh a1
```

`submit.sh` sizes each array from the grid itself, so the array bounds cannot
drift out of step with the grid definition.

## Resuming a partially-failed sweep

Resubmit the identical command. Cells whose `.npz` exists are skipped, so the
cost is only the missing ones. There is no ledger to reconcile — the result
filename *is* the content hash of the config.

The one thing that breaks this: editing a grid builder between submissions.
Cell identity is a content hash, so changing a cell changes its filename (the old
result is orphaned rather than overwritten), and inserting or reordering cells
renumbers every array index after the edit. Append to grids; do not reorder them.

## Array size limits

If a grid is larger than `MaxArraySize`, use `--stride`:

```bash
sbatch --array=0-49 hpc/analyse.slurm a4 50   # element i handles cells i, i+50, …
```

## Checking on a running sweep

Each finished cell writes a JSON sidecar next to its `.npz`, so a sweep can be
surveyed without loading any arrays:

```bash
ls results/a1/*.npz | wc -l
jq -r '[.scalars.frac_rho_negative, .description] | @tsv' results/a1/*.json
```
