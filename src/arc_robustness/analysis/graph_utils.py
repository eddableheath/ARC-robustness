"""
Graph construction on activation point clouds.

The A4 sensitivity axes live here. Three construction rules, two metrics:

``symmetric`` (OR)
    ``A = (K + Kᵀ) > 0``. Every vertex has ``deg >= k``. This floor is what makes
    Forman curvature nearly constant and hence uninformative (T7) — the measured
    degree at ``relu6`` was ``7.02 ± 1.10`` against a floor of ``k = 6``.

``mutual`` (AND)
    ``A = K & Kᵀ``. Degrees are ``<= k``; the graph fragments readily, which
    makes it a sharper test of whether conclusions survive a different
    connectivity regime.

``eps_ball``
    Connect all pairs closer than a quantile of the pairwise distance
    distribution. Degree becomes genuinely feature-dependent, so Forman
    curvature stops being degenerate — the one construction under which the
    Forman results could carry geometric information.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph

from arc_robustness.config import GraphConfig


def pairwise_distance_matrix(pts: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Full ``(N, N)`` distance matrix.

    Computed once per layer and threaded through the curvature routines. For the
    ``N <= 2000`` sizes in the plan this is at most 32 MB in float64 and saves
    recomputing distances inside every edge's optimal-transport solve.
    """
    return pairwise_distances(pts, metric=metric)


def build_graph(
    pts: np.ndarray, config: GraphConfig, distances: np.ndarray | None = None
) -> sp.csr_matrix:
    """Build the adjacency matrix described by *config*."""
    if config.graph_type == "eps_ball":
        if distances is None:
            distances = pairwise_distance_matrix(pts, config.metric)
        return _eps_ball_graph(distances, config.eps_quantile)

    knn = kneighbors_graph(
        pts,
        n_neighbors=config.k,
        mode="connectivity",
        metric=config.metric,
        include_self=False,
    )
    if config.graph_type == "symmetric":
        adj = ((knn + knn.T) > 0).astype(np.float64)
    elif config.graph_type == "mutual":
        adj = (knn.multiply(knn.T) > 0).astype(np.float64)
    else:
        raise ValueError(f"unknown graph_type {config.graph_type!r}")

    adj = sp.csr_matrix(adj)
    adj.setdiag(0.0)
    adj.eliminate_zeros()
    return adj


def _eps_ball_graph(distances: np.ndarray, quantile: float) -> sp.csr_matrix:
    """Connect pairs within the given quantile of the off-diagonal distances."""
    n = distances.shape[0]
    off_diagonal = distances[~np.eye(n, dtype=bool)]
    radius = float(np.quantile(off_diagonal, quantile))
    adj = (distances <= radius).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    return sp.csr_matrix(adj)


def build_knn_graph(pts: np.ndarray, k: int) -> sp.csr_matrix:
    """Symmetric unweighted k-NN adjacency.

    Retained as the historical entry point so the original scripts keep working;
    new code should use :func:`build_graph` with a :class:`GraphConfig`.
    """
    return build_graph(pts, GraphConfig(k=k, graph_type="symmetric"))


def edges(adj: sp.csr_matrix) -> list[tuple[int, int]]:
    """Undirected edges ``(u, v)`` with ``u < v``."""
    rows, cols = sp.triu(adj, k=1).nonzero()
    return [(int(u), int(v)) for u, v in zip(rows, cols)]


def edge_array(adj: sp.csr_matrix) -> np.ndarray:
    """Undirected edges as an ``(E, 2)`` integer array, for vectorised use."""
    rows, cols = sp.triu(adj, k=1).nonzero()
    return np.stack([rows, cols], axis=1).astype(np.int64)


def neighbour_lists(adj: sp.csr_matrix) -> list[np.ndarray]:
    """Neighbour index array per vertex.

    Built once from the CSR structure. The original code called
    ``adj.toarray()`` and then ``np.where`` inside a per-edge loop, which is
    ``O(N)`` per lookup and dominated the Ollivier runtime for larger ``N``.
    """
    adj = adj.tocsr()
    return [
        adj.indices[adj.indptr[i] : adj.indptr[i + 1]].astype(np.int64)
        for i in range(adj.shape[0])
    ]


def degrees(adj: sp.csr_matrix) -> np.ndarray:
    """Vertex degrees as a dense float array."""
    return np.asarray(adj.sum(axis=1)).ravel()


def knn_gap(distances: np.ndarray, k: int) -> np.ndarray:
    """Per-vertex ``d_{k+1}(u) - d_k(u)``, the k-NN stability gap of Lemma T4.1.

    A displacement of every point by at most ``δ`` changes each pairwise
    distance by at most ``2δ``, so the k-NN graph is provably unchanged when
    ``min_u gap(u) > 2δ``. Returned per vertex rather than reduced to the
    minimum, because T4's refinement (a probabilistic "most vertices stable"
    version) needs the whole distribution.
    """
    n = distances.shape[0]
    masked = distances.copy()
    np.fill_diagonal(masked, np.inf)
    ordered = np.sort(masked, axis=1)
    if k + 1 > n - 1:
        raise ValueError(f"need at least k+1={k + 1} neighbours, have {n - 1}")
    return ordered[:, k] - ordered[:, k - 1]
