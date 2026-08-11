#!/usr/bin/env python
"""
Sweep progress watcher.

    uv run python scripts/watch_sweep.py                  # snapshot, all grids
    uv run python scripts/watch_sweep.py --watch          # refresh every 10s
    uv run python scripts/watch_sweep.py --grid a1 -w -n 5
    uv run python scripts/watch_sweep.py --grid a1 --cells   # per-cell detail

Reads only the filesystem — the result shards, their JSON sidecars and the
checkpoint directory — compared against the grid definitions. It never imports
the analysis stack or opens an ``.npz``, so it is safe to run against a sweep in
flight, costs nothing, and works regardless of where the job's stdout went.

That choice matters for a resumable sweep: progress is a property of what is on
disk, not of a log file that may belong to an earlier, abandoned submission.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc_robustness.config import RESULTS_DIR, WEIGHTS_DIR  # noqa: E402
from arc_robustness.experiments import GRID_BUILDERS, build_grid  # noqa: E402

BAR_WIDTH = 24

#: Scalars surfaced for recently-finished cells. ``train_acc`` leads because it
#: is the audit column: the A1 memorisation arms are only controls if they
#: actually memorised, and that is the one thing worth watching land.
WATCH_SCALARS = ("train_acc", "clean_accuracy", "frac_rho_negative")


def _colour(text: str, code: str, enabled: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if enabled else text


def _bar(done: int, total: int, width: int = BAR_WIDTH) -> str:
    if total == 0:
        return " " * width
    filled = round(width * done / total)
    return "█" * filled + "░" * (width - filled)


def _format_age(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


class GridStatus:
    """What is on disk for one grid, versus what the grid asks for."""

    def __init__(self, name: str, results_dir: Path, weights_dir: Path) -> None:
        grid = build_grid(name)
        self.name = name
        self.total = len(grid)

        root = results_dir / name
        self.done_paths = [
            path
            for path in (root / f"{config.uid}.npz" for config in grid)
            if path.exists()
        ]
        self.done = len(self.done_paths)
        self.missing = [
            index
            for index, config in enumerate(grid)
            if not (root / f"{config.uid}.npz").exists()
        ]

        checkpoints = weights_dir / "sweep"
        wanted = grid.unique_models()
        self.checkpoints_total = len(wanted)
        checkpoint_paths = [
            checkpoints / f"{config.model_uid}.pt" for config in wanted
        ]
        self.checkpoints_done = sum(path.exists() for path in checkpoint_paths)
        self.checkpoint_mtimes = sorted(
            path.stat().st_mtime for path in checkpoint_paths if path.exists()
        )

        self.mtimes = sorted(path.stat().st_mtime for path in self.done_paths)

    @property
    def complete(self) -> bool:
        return self.done == self.total

    #: How recently something must have been written for the grid to count as
    #: in flight. Generous, because a single memorisation checkpoint takes ~10
    #: minutes and produces nothing on disk until it finishes — a tighter window
    #: would report an actively-training grid as idle.
    ACTIVE_WINDOW = 15 * 60

    @property
    def active(self) -> bool:
        """A cell finished recently."""
        return bool(self.mtimes) and (time.time() - self.mtimes[-1]) < 120

    @property
    def training(self) -> bool:
        """Checkpoints are incomplete *and* one was written recently.

        Both halves are needed. Incompleteness alone is the normal state of every
        grid nobody has started yet — reporting those as "training" is how a
        status display teaches you to ignore it.
        """
        if self.checkpoints_done >= self.checkpoints_total:
            return False
        return bool(self.checkpoint_mtimes) and (
            time.time() - self.checkpoint_mtimes[-1]
        ) < self.ACTIVE_WINDOW

    @property
    def in_flight(self) -> bool:
        return self.active or self.training

    def rate_per_minute(self, window: int = 10) -> float | None:
        """Recent completion rate, from result mtimes.

        Measured over the last *window* completions rather than the whole run, so
        the estimate reflects the current phase. A sweep whose cells differ in
        cost by 3× — ``a4`` spans 200 to 2000 vertices — has no single rate, and a
        run-long average would mispredict whichever phase is in progress.
        """
        if len(self.mtimes) < 2:
            return None
        recent = self.mtimes[-window:]
        elapsed = recent[-1] - recent[0]
        if elapsed <= 0:
            return None
        return (len(recent) - 1) / (elapsed / 60)

    def eta(self) -> str:
        """Projected time to finish, or ``—`` when there is nothing to project.

        Deliberately blank unless the grid is actually in flight. A rate computed
        from cells written during an unrelated earlier run is not a forecast of
        anything, and a confident-looking "~22m" against an idle grid is worse
        than no number at all.
        """
        if self.complete:
            return "done"
        if not self.in_flight:
            return "idle" if self.done else "—"
        rate = self.rate_per_minute()
        if not rate:
            return "training" if self.training else "—"
        return f"~{_format_age(len(self.missing) / rate * 60)}"

    def recent(self, results_dir: Path, limit: int = 5) -> list[dict]:
        """Sidecar scalars for the most recently written cells."""
        newest = sorted(self.done_paths, key=lambda p: p.stat().st_mtime)[-limit:]
        out = []
        for path in reversed(newest):
            sidecar = path.with_suffix(".json")
            if not sidecar.exists():
                continue
            try:
                blob = json.loads(sidecar.read_text())
            except json.JSONDecodeError:
                continue  # being written right now
            out.append(
                {
                    "description": blob.get("description", path.stem),
                    "scalars": blob.get("scalars", {}),
                    "age": time.time() - path.stat().st_mtime,
                }
            )
        return out


def render(
    grids: list[str],
    results_dir: Path,
    weights_dir: Path,
    colour: bool,
    show_cells: bool,
) -> str:
    lines: list[str] = []
    stamp = datetime.now(UTC).astimezone().strftime("%H:%M:%S")
    lines.append(_colour(f"ARC sweep status   {stamp}", "1", colour))
    lines.append("")
    lines.append(
        f"{'grid':<7} {'ckpt':>7}  {'cells':>8}  {'progress':<{BAR_WIDTH}}"
        f" {'':>5}  {'eta':>7}  last"
    )

    statuses = [GridStatus(name, results_dir, weights_dir) for name in grids]

    for status in statuses:
        percent = 100 * status.done / status.total if status.total else 0.0
        bar = _bar(status.done, status.total)
        if status.complete:
            bar = _colour(bar, "32", colour)
        elif status.active:
            bar = _colour(bar, "36", colour)

        last = (
            _format_age(time.time() - status.mtimes[-1]) + " ago"
            if status.mtimes
            else "—"
        )
        marker = "*" if status.active else " "
        lines.append(
            f"{status.name:<7}"
            f" {status.checkpoints_done:>3}/{status.checkpoints_total:<3}"
            f" {status.done:>4}/{status.total:<4} {bar} {percent:>4.0f}%"
            f"  {status.eta():>7}  {last}{marker}"
        )

    active = [s for s in statuses if s.in_flight] or [
        s for s in statuses if 0 < s.done < s.total
    ]
    for status in active[:1]:
        recent = status.recent(results_dir)
        if not recent:
            continue
        lines.append("")
        lines.append(_colour(f"recent cells — {status.name}", "1", colour))
        for entry in recent:
            scalars = entry["scalars"]
            bits = [
                f"{key.replace('_', ' ')} {scalars[key]:.3f}"
                for key in WATCH_SCALARS
                if isinstance(scalars.get(key), (int, float))
            ]
            lines.append(f"  {_format_age(entry['age']):>4} ago  {entry['description']}")
            if bits:
                lines.append(f"            {'   '.join(bits)}")

    if show_cells:
        for status in statuses:
            if status.complete or not status.missing:
                continue
            lines.append("")
            lines.append(_colour(f"outstanding indices — {status.name}", "1", colour))
            preview = ", ".join(str(i) for i in status.missing[:30])
            if len(status.missing) > 30:
                preview += f", … (+{len(status.missing) - 30})"
            lines.append(f"  {preview}")

    total_done = sum(s.done for s in statuses)
    total_cells = sum(s.total for s in statuses)
    lines.append("")
    lines.append(f"{total_done}/{total_cells} cells complete")

    # Training writes no result files, so an actively-training grid sits at 0
    # cells for many minutes while making real progress. Say so explicitly rather
    # than looking hung — but only for grids that really are training.
    training = [s for s in statuses if s.training and not s.active]
    if training:
        names = ", ".join(
            f"{s.name} ({s.checkpoints_done}/{s.checkpoints_total})" for s in training
        )
        lines.append(f"training checkpoints: {names} — no cells land until done")

    idle = [s for s in statuses if not s.in_flight and not s.complete and s.done]
    if idle:
        names = ", ".join(f"{s.name} ({len(s.missing)} left)" for s in idle)
        lines.append(f"started but idle: {names}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Watch sweep progress.")
    parser.add_argument(
        "--grid", action="append", default=None, help="limit to a grid (repeatable)"
    )
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("-w", "--watch", action="store_true", help="refresh until done")
    parser.add_argument(
        "-n", "--interval", type=float, default=10.0, help="refresh seconds"
    )
    parser.add_argument(
        "--cells", action="store_true", help="list outstanding grid indices"
    )
    parser.add_argument("--no-colour", action="store_true")
    args = parser.parse_args(argv)

    grids = args.grid or list(GRID_BUILDERS)
    unknown = [name for name in grids if name not in GRID_BUILDERS]
    if unknown:
        raise SystemExit(f"unknown grid(s) {unknown}; available: {sorted(GRID_BUILDERS)}")

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    weights_dir = Path(args.weights_dir) if args.weights_dir else WEIGHTS_DIR
    colour = sys.stdout.isatty() and not args.no_colour

    if not args.watch:
        print(render(grids, results_dir, weights_dir, colour, args.cells))
        return 0

    try:
        while True:
            frame = render(grids, results_dir, weights_dir, colour, args.cells)
            # Home the cursor and clear to end of screen, rather than clearing
            # first: a full clear flickers, and on a slow redraw leaves the
            # terminal blank for a visible moment.
            sys.stdout.write("\033[H\033[J" + frame + "\n")
            sys.stdout.flush()
            if all(
                GridStatus(name, results_dir, weights_dir).complete for name in grids
            ):
                print("\nall selected grids complete")
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
