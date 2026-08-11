"""
Turning a directory of result shards into tables.

The sweep writes one ``.npz`` per cell; every reported number in the paper is an
aggregate over some subset of them. This module is the only place that reduction
happens, so the rule about *what may be averaged over what* is stated once.

That rule, from plan §0.6: **seeds are the unit of replication, vertices are
not.** The ρ signal lives in a single shared per-layer trend, so the ``N``
vertices inside one cell are not ``N`` independent observations and an error bar
computed across them is meaningless — it would report the sampling noise of one
number as though it were a population. Every uncertainty produced here is
therefore taken across *cells* that differ only in seed, never across vertices.
:func:`summarise` enforces this by construction: it can only ever see the
per-cell scalars, because the per-vertex arrays are reduced before they reach it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from arc_robustness.cell import load_record
from arc_robustness.config import RESULTS_DIR, ExperimentConfig

#: Scalars carried into the cell-level table. Anything per-layer or per-vertex is
#: reduced explicitly by the caller, so a table can never silently average a
#: quantity whose sampling unit is wrong.
CELL_SCALARS: tuple[str, ...] = (
    "accuracy",
    "clean_accuracy",
    "flipped_frac",
    "train_acc",
    "test_acc",
    "frac_rho_negative",
    "frac_rho_centred_negative",
    "global_trend_correlation",
    "eta_between_layer_share",
    "curvature_between_layer_share",
    "modularity_ceiling",
)

#: Per-layer arrays kept in the layer-level table.
LAYER_METRICS: tuple[str, ...] = (
    "modularity",
    "normalised_cut",
    "mean_ollivier",
    "mean_forman",
    "mean_af",
    "curvature_gap_ollivier",
    "curvature_gap_forman",
    "curvature_gap_af",
    "gap_length_matched",
    "mean_pairwise_distance",
    "n_components",
    "largest_component_frac",
    "fiedler_largest_component",
    "normalised_gap",
    "degree_mean",
    "degree_std",
    "degree_frac_at_floor",
    "knn_gap_min",
    "spectral_norm_max",
    "spectral_norm_min",
)


def _config_columns(config: ExperimentConfig) -> dict[str, Any]:
    """Flatten the config fields a table might group or facet by."""
    return {
        "uid": config.uid,
        "model_uid": config.model_uid,
        "experiment": config.experiment,
        "tag": config.tag,
        "dataset": config.data.label(),
        "label_mode": config.data.label_mode,
        "activation": config.model.activation,
        "norm": config.model.norm,
        "depth": config.model.depth,
        "trained": config.train.trained,
        "train_seed": config.train.seed,
        "epochs": config.train.epochs,
        "k": config.graph.k,
        "n_per_class": config.graph.n_per_class,
        "graph_type": config.graph.graph_type,
        "metric": config.graph.metric,
        "subsample_seed": config.graph.subsample_seed,
        "ollivier_norm": config.estimator.ollivier_norm,
        "eta_mode": config.estimator.eta_mode,
        "feature_norm": config.estimator.feature_norm,
        "invariant": config.estimator.is_invariant,
        "attack": config.attack.kind,
        "epsilon": config.attack.epsilon,
        "rescale": config.rescale,
    }


def load_results(
    grid: str, results_dir: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read every result shard for *grid*.

    Returns ``(cells, layers)`` — one row per cell, and one row per cell × layer.
    The split is deliberate: almost every mistake available here involves mixing
    the two, e.g. averaging a per-layer curvature across layers as though depth
    were a replicate.

    Missing cells are simply absent from the frame. Because the sweep is
    idempotent and content-addressed, a partial directory is a valid input — the
    caller checks completeness against the grid, and :func:`missing_cells` says
    which are absent.
    """
    root = (RESULTS_DIR if results_dir is None else Path(results_dir)) / grid
    cell_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []

    for path in sorted(root.glob("*.npz")):
        record = load_record(path)
        config = ExperimentConfig.from_dict(json.loads(record["config_json"]))
        columns = _config_columns(config)

        cell_rows.append(
            {
                **columns,
                **{key: record.get(key, np.nan) for key in CELL_SCALARS},
                # Reduced here, once, so no downstream code is tempted to treat
                # vertices as replicates.
                "rho_mean": float(np.nanmean(record["rho"]))
                if "rho" in record
                else np.nan,
                "rho_median": float(np.nanmedian(record["rho"]))
                if "rho" in record
                else np.nan,
                "n_vertices": int(record["n_vertices"]),
            }
        )

        layer_names = [str(name) for name in record["layer_names"]]
        for index, name in enumerate(layer_names):
            row = {**columns, "layer": name, "layer_index": index}
            for key in LAYER_METRICS:
                values = record.get(key)
                row[key] = (
                    float(values[index])
                    if values is not None and np.ndim(values) == 1
                    else np.nan
                )
            # r_layer is defined on transitions, so it has L-1 entries and is
            # attached to the layer the transition starts from.
            r_layer = record.get("r_layer")
            row["r_layer"] = (
                float(r_layer[index])
                if r_layer is not None and index < len(np.atleast_1d(r_layer))
                else np.nan
            )
            layer_rows.append(row)

    return pd.DataFrame(cell_rows), pd.DataFrame(layer_rows)


def missing_cells(grid: str, results_dir: Path | None = None) -> list[int]:
    """Grid indices with no result on disk — what a resubmission would recompute."""
    from arc_robustness.experiments import build_grid

    root = (RESULTS_DIR if results_dir is None else Path(results_dir)) / grid
    return [
        index
        for index, config in enumerate(build_grid(grid))
        if not (root / f"{config.uid}.npz").exists()
    ]


def summarise(
    cells: pd.DataFrame,
    by: Sequence[str],
    metrics: Sequence[str],
    replicate: str = "train_seed",
) -> pd.DataFrame:
    """Mean, sd and n over *replicate* within each group of *by*.

    ``sd`` is the spread across seeds, and ``n`` is reported alongside it so a
    "±" in a table can always be traced to how many networks it came from. A
    group with ``n = 1`` gets ``sd = NaN`` rather than 0 — one seed is not
    evidence of stability, and a zero there would read as though it were.
    """
    if replicate not in cells.columns:
        raise KeyError(f"replicate column {replicate!r} not in frame")
    grouped = cells.groupby(list(by), dropna=False)

    frames = []
    for metric in metrics:
        stats = grouped[metric].agg(["mean", "std", "count"])
        stats.columns = [f"{metric}_mean", f"{metric}_sd", f"{metric}_n"]
        frames.append(stats)
    return pd.concat(frames, axis=1).reset_index()


def falsification_table(cells: pd.DataFrame) -> pd.DataFrame:
    """The A1 table: one row per control arm × estimator convention.

    Reads as the paper's central control. ``train_acc`` is included not as a
    result but as an audit column: the two memorisation arms are only controls if
    they actually memorised, and a shuffled-label network sitting at chance is an
    undertrained network masquerading as a null result.
    """
    metrics = [
        "train_acc",
        "clean_accuracy",
        "frac_rho_negative",
        "rho_mean",
        "global_trend_correlation",
    ]
    table = summarise(cells, by=["tag", "invariant"], metrics=metrics)
    return table.sort_values(["invariant", "tag"]).reset_index(drop=True)


def layer_profile(
    layers: pd.DataFrame, metric: str, by: Sequence[str] = ("tag", "invariant")
) -> pd.DataFrame:
    """``metric`` as a function of layer, averaged over seeds within each group.

    Layers are a *coordinate*, never a replicate — hence the pivot rather than a
    mean over depth.
    """
    grouped = (
        layers.groupby([*by, "layer_index"], dropna=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    return grouped.pivot_table(
        index=list(by), columns="layer_index", values="mean"
    ).reset_index()
