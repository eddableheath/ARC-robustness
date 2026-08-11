"""
Inductive geometric features for a single query point.

A reference graph is built once from clean held-out activations. Each query point
is then scored *against that fixed backdrop* without being added to it, so:

* queries are scored independently of one another — an attacker cannot influence
  another query's score by submitting a batch;
* the null distribution of each statistic can be estimated once from clean
  held-out points and reused;
* the reference geometry is not contaminated by the queries themselves, which
  would blunt exactly the signal we are looking for.

The statistics are deliberately of three kinds, so the probe can report *which*
kind of geometry carries signal rather than just whether some combination does:

**Density / position** — how far the query sits from the clean manifold.
Cheap, and overlaps with what Mahalanobis and kernel density already capture.

**Community bridging** — whether the query's neighbours come from more than one
class. Directly the "adversarial points bridge communities" intuition, and the
one most closely tied to the modularity story.

**Curvature** — the local Ollivier curvature of the edges attaching the query to
the reference graph, which is the only part that is genuinely *this project's*
contribution rather than a re-derivation of an existing detector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import pairwise_distances

from arc_robustness.analysis.graph_utils import build_graph, neighbour_lists
from arc_robustness.config import GraphConfig


@dataclass
class ReferenceGraph:
    """A fixed clean-data graph against which queries are scored."""

    points: np.ndarray  # (M, D) reference activations
    labels: np.ndarray  # (M,) reference class labels
    neighbours: list[np.ndarray]
    knn_distance: np.ndarray  # (M,) distance to the k-th reference neighbour
    k: int
    metric: str

    @property
    def size(self) -> int:
        return len(self.points)


def reference_graph_features(
    points: np.ndarray, labels: np.ndarray, config: GraphConfig
) -> ReferenceGraph:
    """Build the clean reference graph for one layer."""
    adjacency = build_graph(points, config)
    distances = pairwise_distances(points, metric=config.metric)
    np.fill_diagonal(distances, np.inf)
    ordered = np.sort(distances, axis=1)
    return ReferenceGraph(
        points=points,
        labels=labels,
        neighbours=neighbour_lists(adjacency),
        knn_distance=ordered[:, config.k - 1],
        k=config.k,
        metric=config.metric,
    )


def _uniform_w1(cost: np.ndarray) -> float:
    """Exact ``W₁`` between uniform distributions over the two index sets."""
    import ot

    a = np.full(cost.shape[0], 1.0 / cost.shape[0])
    b = np.full(cost.shape[1], 1.0 / cost.shape[1])
    return float(ot.emd2(a, b, np.ascontiguousarray(cost, dtype=np.float64)))


def detection_features(
    queries: np.ndarray,
    reference: ReferenceGraph,
    predicted_labels: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Per-query geometric statistics against the reference graph.

    Parameters
    ----------
    queries
        ``(N, D)`` query activations at the same layer as ``reference.points``.
    predicted_labels
        The model's predicted class for each query. Supplied because a deployed
        detector has no access to true labels — every statistic that uses a label
        must use the *prediction*, or the evaluation leaks information it would
        not have and reports an optimistic number.

    Returns
    -------
    Dict of ``(N,)`` arrays. NaN where a statistic is undefined (for instance
    curvature when a query has no valid neighbourhood).
    """
    distances = pairwise_distances(queries, reference.points, metric=reference.metric)
    k = reference.k
    order = np.argsort(distances, axis=1)[:, :k]
    n = len(queries)

    knn_distances = np.take_along_axis(distances, order, axis=1)

    out: dict[str, np.ndarray] = {
        # -- density / position --------------------------------------------
        "mean_knn_distance": knn_distances.mean(axis=1),
        "max_knn_distance": knn_distances.max(axis=1),
        # Ratio of the query's k-NN radius to its neighbours' own k-NN radii.
        # Scale-free, so comparable across layers without normalisation —
        # a value >> 1 means the query sits in a sparser place than the clean
        # points it is nearest to.
        "density_ratio": np.full(n, np.nan),
        # -- community bridging -------------------------------------------
        "neighbour_label_entropy": np.full(n, np.nan),
        "neighbour_agreement": np.full(n, np.nan),
        # -- curvature ----------------------------------------------------
        "mean_query_curvature": np.full(n, np.nan),
        "min_query_curvature": np.full(n, np.nan),
        "curvature_spread": np.full(n, np.nan),
    }

    n_classes = int(reference.labels.max()) + 1

    for i in range(n):
        nbrs = order[i]
        local_radii = reference.knn_distance[nbrs]
        median_radius = float(np.median(local_radii))
        if median_radius > 0:
            out["density_ratio"][i] = float(knn_distances[i].mean() / median_radius)

        # Community bridging, measured on reference labels — these are ground
        # truth for the *reference* set, which a deployed detector does have.
        neighbour_labels = reference.labels[nbrs]
        counts = np.bincount(neighbour_labels, minlength=n_classes).astype(np.float64)
        shares = counts / counts.sum()
        nonzero = shares[shares > 0]
        out["neighbour_label_entropy"][i] = float(-(nonzero * np.log(nonzero)).sum())
        if predicted_labels is not None:
            out["neighbour_agreement"][i] = float(
                (neighbour_labels == predicted_labels[i]).mean()
            )

        # Curvature of the edges attaching the query to the reference graph.
        # μ_query is uniform over the query's k nearest reference points;
        # μ_v is uniform over v's reference neighbours.
        query_cloud = reference.points[nbrs]
        curvatures = []
        for v in nbrs:
            v_nbrs = reference.neighbours[v]
            if len(v_nbrs) == 0:
                continue
            edge_length = distances[i, v]
            if edge_length <= 0:
                continue
            cost = pairwise_distances(
                query_cloud, reference.points[v_nbrs], metric=reference.metric
            )
            # Distance-normalised, matching the repaired κ̂ of T1.4 so the
            # detector inherits the invariance rather than reintroducing the
            # scale dependence the estimator work removed.
            curvatures.append(1.0 - _uniform_w1(cost) / edge_length)

        if curvatures:
            arr = np.asarray(curvatures)
            out["mean_query_curvature"][i] = float(arr.mean())
            out["min_query_curvature"][i] = float(arr.min())
            out["curvature_spread"][i] = float(arr.max() - arr.min())

    return out


def stack_layer_features(
    per_layer: dict[str, dict[str, np.ndarray]], layer_names: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Flatten ``{layer: {stat: (N,)}}`` into an ``(N, F)`` design matrix.

    Returns the matrix and the column names (``"<layer>.<stat>"``), so a fitted
    detector's coefficients remain interpretable — which layer and which kind of
    geometry is doing the work is the interesting part of the result.
    """
    columns: list[np.ndarray] = []
    names: list[str] = []
    for layer in layer_names:
        for stat, values in sorted(per_layer[layer].items()):
            columns.append(values)
            names.append(f"{layer}.{stat}")
    return np.stack(columns, axis=1), names
