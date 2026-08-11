"""
The per-cell analysis pipeline: features in, metric record out.

One call to :func:`analyse_features` is exactly what one Slurm array element
computes. It is deliberately pure — features and configs in, a flat dict of
numbers and arrays out — so it can be unit-tested, run locally on a laptop, and
run under a scheduler without any change in behaviour.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

from arc_robustness.analysis.community import (
    connectivity_metrics,
    curvature_gap,
    curvature_gap_decomposition,
    degree_summary,
    modularity,
    modularity_ceiling,
    normalised_cut,
)
from arc_robustness.analysis.evolution import local_ricci_evolution
from arc_robustness.analysis.graph_utils import (
    build_graph,
    knn_gap,
    pairwise_distance_matrix,
)
from arc_robustness.analysis.invariance import normalise_features
from arc_robustness.analysis.ricci import (
    augmented_forman_ricci,
    forman_ricci,
    ollivier_ricci,
    vertex_curvature,
)
from arc_robustness.config import EstimatorConfig, GraphConfig


def analyse_features(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    graph_config: GraphConfig,
    estimator_config: EstimatorConfig,
    layer_names: list[str] | None = None,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Compute every Phase-1 metric for one feature set.

    Parameters
    ----------
    features
        Layer name → ``(N, D_ℓ)`` activations, already subsampled.
    labels
        ``(N,)`` class labels, aligned with the feature rows.
    layer_names
        Explicit layer order. Passed rather than inferred from dict order or
        sorting, so a 12-layer network cannot silently get ``relu10`` placed
        before ``relu2``.

    Returns
    -------
    A flat dict mixing scalars, ``(L,)`` per-layer arrays and ``(N,)`` or
    ``(L, N)`` per-vertex arrays. Keys are stable; consumers index by name.
    """
    names = list(features.keys()) if layer_names is None else list(layer_names)
    missing = [n for n in names if n not in features]
    if missing:
        raise KeyError(f"features missing layers {missing}")

    features = normalise_features(
        OrderedDict((n, features[n]) for n in names), estimator_config.feature_norm
    )

    n_layers = len(names)
    n_vertices = len(labels)

    distances: dict[str, np.ndarray] = {}
    adjacencies: dict[str, Any] = {}
    vertex_ollivier: dict[str, np.ndarray] = {}

    per_layer: dict[str, list[float]] = {
        key: [] for key in (
            "modularity", "normalised_cut", "algebraic_connectivity_legacy",
            "curvature_gap_forman", "curvature_gap_af", "curvature_gap_ollivier",
            "mean_forman", "mean_af", "mean_ollivier",
            "mean_pairwise_distance", "n_edges",
            "n_components", "largest_component_frac",
            "fiedler_largest_component", "normalised_gap",
            "cheeger_lower", "cheeger_upper",
            "degree_mean", "degree_std", "degree_min", "degree_max",
            "degree_frac_at_floor",
            "knn_gap_min", "knn_gap_median",
            "gap_length_matched", "mean_length_intra", "mean_length_inter",
        )
    }
    vertex_arrays: dict[str, list[np.ndarray]] = {
        "vertex_forman": [], "vertex_af": [], "vertex_ollivier": []
    }

    for name in names:
        pts = features[name]
        dist = pairwise_distance_matrix(pts, graph_config.metric)
        adj = build_graph(pts, graph_config, distances=dist)
        distances[name] = dist
        adjacencies[name] = adj

        off_diagonal = ~np.eye(n_vertices, dtype=bool)
        per_layer["mean_pairwise_distance"].append(float(dist[off_diagonal].mean()))
        per_layer["n_edges"].append(float(adj.sum() / 2))

        forman = forman_ricci(adj)
        af = augmented_forman_ricci(adj)
        v_forman = vertex_curvature(forman, n_vertices)
        v_af = vertex_curvature(af, n_vertices)
        vertex_arrays["vertex_forman"].append(v_forman)
        vertex_arrays["vertex_af"].append(v_af)
        per_layer["mean_forman"].append(float(np.nanmean(v_forman)))
        per_layer["mean_af"].append(float(np.nanmean(v_af)))
        per_layer["curvature_gap_forman"].append(curvature_gap(forman, labels))
        per_layer["curvature_gap_af"].append(curvature_gap(af, labels))

        if estimator_config.compute_ollivier:
            ollivier = ollivier_ricci(
                adj,
                pts,
                normalisation=estimator_config.ollivier_norm,
                metric=graph_config.metric,
                distances=dist,
                n_jobs=n_jobs,
            )
            v_ollivier = vertex_curvature(ollivier, n_vertices)
            per_layer["curvature_gap_ollivier"].append(curvature_gap(ollivier, labels))
            decomposition = curvature_gap_decomposition(ollivier, labels, dist)
            per_layer["gap_length_matched"].append(decomposition["gap_length_matched"])
            per_layer["mean_length_intra"].append(decomposition["mean_length_intra"])
            per_layer["mean_length_inter"].append(decomposition["mean_length_inter"])
        else:
            v_ollivier = np.full(n_vertices, np.nan)
            for key in (
                "curvature_gap_ollivier", "gap_length_matched",
                "mean_length_intra", "mean_length_inter",
            ):
                per_layer[key].append(np.nan)

        vertex_ollivier[name] = v_ollivier
        vertex_arrays["vertex_ollivier"].append(v_ollivier)
        per_layer["mean_ollivier"].append(float(np.nanmean(v_ollivier)))

        per_layer["modularity"].append(modularity(adj, labels))
        per_layer["normalised_cut"].append(normalised_cut(adj, labels))

        connectivity = connectivity_metrics(adj)
        for key in (
            "n_components", "largest_component_frac",
            "fiedler_largest_component", "normalised_gap",
            "cheeger_lower", "cheeger_upper",
        ):
            per_layer[key].append(connectivity[key])
        # The legacy whole-graph Fiedler value is exactly the giant-component
        # value when connected, and 0 otherwise; recording both makes the A5
        # replacement auditable rather than asserted.
        per_layer["algebraic_connectivity_legacy"].append(
            connectivity["fiedler_largest_component"]
            if connectivity["n_components"] == 1
            else 0.0
        )

        degree = degree_summary(adj)
        for key, value in degree.items():
            per_layer[f"degree_{key}"].append(value)

        gaps = knn_gap(dist, graph_config.k)
        per_layer["knn_gap_min"].append(float(gaps.min()))
        per_layer["knn_gap_median"].append(float(np.median(gaps)))

    record: dict[str, Any] = {
        "layer_names": np.array(names),
        "labels": labels,
        "n_vertices": n_vertices,
        "n_layers": n_layers,
        "modularity_ceiling": modularity_ceiling(labels),
        "estimator_is_invariant": estimator_config.is_invariant,
    }
    for key, values in per_layer.items():
        record[key] = np.array(values, dtype=np.float64)
    for key, arrays in vertex_arrays.items():
        record[key] = np.stack(arrays, axis=0)

    # Across-layer evolution statistics.
    if estimator_config.compute_ollivier and n_layers >= 3:
        evolution = local_ricci_evolution(
            distances,
            adjacencies,
            vertex_ollivier,
            layer_names=names,
            eta_mode=estimator_config.eta_mode,
        )
        record.update(
            {
                "eta": evolution["eta"],
                "eta_curvature": evolution["curvature"],
                "rho": evolution["rho"],
                "rho_centred": evolution["rho_centred"],
                "r_layer": evolution["r_layer"],
                "global_trend_correlation": evolution["global_trend_correlation"],
                "frac_rho_negative": evolution["frac_rho_negative"],
                "frac_rho_centred_negative": evolution["frac_rho_centred_negative"],
                "eta_between_layer_share": evolution["eta_variance"][
                    "between_layer_share"
                ],
                "curvature_between_layer_share": evolution["curvature_variance"][
                    "between_layer_share"
                ],
            }
        )

    return record
