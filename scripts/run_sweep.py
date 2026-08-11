#!/usr/bin/env python
"""
Sweep driver — the command line the scheduler calls.

Four subcommands, in the order a sweep uses them::

    python scripts/run_sweep.py list
    python scripts/run_sweep.py manifest --grid a2
    python scripts/run_sweep.py train    --grid a2 --all
    python scripts/run_sweep.py run      --grid a2 --all

**Why training is a separate stage.** Many analysis cells share one checkpoint —
every estimator variant, every graph seed, the whole rescaling ladder. Lazy
training inside analysis jobs would have several array elements racing to write
the same file. The ``train`` stage indexes over ``grid.unique_models()``, so each
checkpoint is produced exactly once, and the ``run`` stage then finds every
checkpoint already on disk.

Cell selection is the same for both stages:

``--index N``       one cell; defaults to ``SLURM_ARRAY_TASK_ID`` when present.
``--stride S``      with ``--index N``, also do ``N+S``, ``N+2S``, … — for grids
                    larger than the cluster's array-size limit.
``--all``           every cell, sequentially. Fine for ``smoke``, and for
                    running a real grid overnight on one machine.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

# Allow running as a plain script from the repo root without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc_robustness.cell import run_cell  # noqa: E402
from arc_robustness.config import RESULTS_DIR, WEIGHTS_DIR, ExperimentConfig  # noqa: E402
from arc_robustness.experiments import build_grid, list_grids  # noqa: E402
from arc_robustness.training.train import get_or_train, pick_device  # noqa: E402


def _selected_indices(args: argparse.Namespace, n_cells: int) -> list[int]:
    """Resolve the cell selection flags into a list of indices."""
    if args.all:
        return list(range(n_cells))

    index = args.index
    if index is None:
        env = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env is None:
            raise SystemExit(
                "specify --index N or --all (or run under a Slurm array, which "
                "supplies SLURM_ARRAY_TASK_ID)"
            )
        index = int(env)

    if not 0 <= index < n_cells:
        raise SystemExit(f"index {index} out of range for {n_cells} cells")

    stride = args.stride
    return list(range(index, n_cells, stride)) if stride else [index]


def cmd_list(args: argparse.Namespace) -> int:
    print(f"{'grid':<8} {'cells':>7} {'checkpoints':>12}")
    for name, n_cells, n_models in list_grids():
        print(f"{name:<8} {n_cells:>7} {n_models:>12}")
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    grid = build_grid(args.grid)
    out = (
        Path(args.out)
        if args.out
        else (RESULTS_DIR if args.results_dir is None else Path(args.results_dir))
        / args.grid
        / "manifest.json"
    )
    grid.write_manifest(out)
    print(f"{len(grid)} cells, {len(grid.unique_models())} checkpoints -> {out}")
    if args.show:
        for i, config in enumerate(grid):
            print(f"{i:>4}  {config.uid}  {config.describe()}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    grid = build_grid(args.grid)
    models = grid.unique_models()
    device = pick_device(args.device)
    weights_dir = Path(args.weights_dir) if args.weights_dir else None

    failures = 0
    for i in _selected_indices(args, len(models)):
        config = models[i]
        print(f"[{i}/{len(models)}] {config.model_uid}  {config.describe()}", flush=True)
        if config.checkpoint_path(weights_dir).exists() and not args.overwrite:
            print("  skip (checkpoint exists)", flush=True)
            continue
        try:
            get_or_train(config, weights_dir=weights_dir, device=device)
        except Exception:  # noqa: BLE001 — one bad cell must not kill the array
            failures += 1
            traceback.print_exc()
    return 1 if failures else 0


def cmd_run(args: argparse.Namespace) -> int:
    grid = build_grid(args.grid)
    device = pick_device(args.device)
    weights_dir = Path(args.weights_dir) if args.weights_dir else None
    results_dir = Path(args.results_dir) if args.results_dir else None

    failures = 0
    for i in _selected_indices(args, len(grid)):
        config: ExperimentConfig = grid[i]
        print(f"[{i}/{len(grid)}]", end=" ", flush=True)
        try:
            run_cell(
                config,
                weights_dir=weights_dir,
                results_dir=results_dir,
                device=device,
                n_jobs=args.n_jobs,
                overwrite=args.overwrite,
            )
        except Exception:  # noqa: BLE001
            # Keep going: a single failing cell in a 200-cell array should cost
            # that cell, not the sweep. The non-zero exit code makes the failure
            # visible in the Slurm accounting rather than only in the log.
            failures += 1
            traceback.print_exc()

    if failures:
        print(f"{failures} cell(s) failed", file=sys.stderr)
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser, with_selection: bool = True) -> None:
        sub.add_argument("--grid", required=True, help="grid name, e.g. a2")
        sub.add_argument("--weights-dir", default=None)
        sub.add_argument("--results-dir", default=None)
        sub.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
        sub.add_argument("--overwrite", action="store_true")
        if with_selection:
            sub.add_argument("--index", type=int, default=None)
            sub.add_argument("--stride", type=int, default=0)
            sub.add_argument("--all", action="store_true")

    p_list = subparsers.add_parser("list", help="registered grids and their sizes")
    p_list.set_defaults(func=cmd_list)

    p_manifest = subparsers.add_parser(
        "manifest", help="write the index -> config mapping for a grid"
    )
    p_manifest.add_argument("--grid", required=True)
    p_manifest.add_argument("--results-dir", default=None)
    p_manifest.add_argument("--out", default=None)
    p_manifest.add_argument("--show", action="store_true", help="also print each cell")
    p_manifest.set_defaults(func=cmd_manifest)

    p_train = subparsers.add_parser("train", help="produce the grid's checkpoints")
    add_common(p_train)
    p_train.set_defaults(func=cmd_train)

    p_run = subparsers.add_parser("run", help="analyse cells")
    add_common(p_run)
    p_run.add_argument(
        "--n-jobs", type=int, default=1, help="parallel workers for Ollivier curvature"
    )
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"train", "run"}:
        print(f"weights -> {Path(args.weights_dir) if args.weights_dir else WEIGHTS_DIR}")
        print(f"results -> {Path(args.results_dir) if args.results_dir else RESULTS_DIR}")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
