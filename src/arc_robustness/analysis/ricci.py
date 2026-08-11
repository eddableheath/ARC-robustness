"""
Discrete Ricci curvature on graphs built from activation point clouds.

Three notions, matching arXiv:2509.22362:

======================  ===============================================
Forman-Ricci            ``F(u,v)  = 4 - deg(u) - deg(v)``
Augmented Forman        ``AF(u,v) = 4 - deg(u) - deg(v) + 3|N(u) ∩ N(v)|``
Ollivier-Ricci          ``κ(u,v)  = 1 - W₁(μ_u, μ_v) / d(u,v)``
======================  ===============================================

The Ollivier normalisation is the substantive choice, and this module implements
all three conventions so A2 can report them side by side.

**Why the original convention is not a curvature.** The repo's first
implementation set ``d(u,v) = 1`` — the graph hop distance between adjacent
vertices — while computing ``W₁`` with a *Euclidean feature-space* ground cost.
The units do not match: ``W₁`` is in activation units, so ``1 - W₁`` is an
affine function of local neighbourhood scale. The measured consequence was a
monotone per-layer mean curvature

    relu1 … relu6:  -8.075  -6.219  -4.010  -2.036  -0.659  +0.297

tracking a mean pairwise distance of ``21.5, 25.3, 50.5, 73.8, 64.5, 29.9``.
The apparent "curvature increases with depth" trend is substantially "local
neighbourhood diameter shrinks, reported in raw activation units".

Under ``ollivier_norm="distance"`` the estimator becomes ``1 - W₁/d(u,v)`` with
both terms in the same units, which is scale-free and therefore invariant under
the per-layer rescaling group ``G`` of T1 — see :mod:`arc_robustness.analysis.invariance`.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import scipy.sparse as sp

from arc_robustness.analysis.graph_utils import (
    degrees,
    edge_array,
    neighbour_lists,
    pairwise_distance_matrix,
)


# ---------------------------------------------------------------------------
# Forman
# ---------------------------------------------------------------------------


def forman_ricci(adj: sp.csr_matrix) -> dict[tuple[int, int], float]:
    """``F(u,v) = 4 - deg(u) - deg(v)``.

    Note T7: on a symmetric k-NN graph this is a function of the degree sequence
    alone. It is *measurable with respect to the graph*, hence constant on the
    fibre of the map (features → graph), and so cannot see feature geometry
    given the graph. Any Forman-based conclusion is a conclusion about hub
    structure.
    """
    deg = degrees(adj)
    e = edge_array(adj)
    values = 4.0 - deg[e[:, 0]] - deg[e[:, 1]]
    return {(int(u), int(v)): float(val) for (u, v), val in zip(e, values)}


def augmented_forman_ricci(adj: sp.csr_matrix) -> dict[tuple[int, int], float]:
    """``AF(u,v) = 4 - deg(u) - deg(v) + 3|N(u) ∩ N(v)|``.

    The shared-neighbour term does see local clustering, so unlike plain Forman
    this is not degree-determined. T7 recommends keeping it as the cheap proxy
    and verifying its rank correlation against ``κ̂``.
    """
    deg = degrees(adj)
    e = edge_array(adj)
    binary = (adj > 0).astype(np.float64)
    # (A²)[u,v] counts shared one-hop neighbours for a loop-free binary matrix.
    shared = np.asarray((binary @ binary)[e[:, 0], e[:, 1]]).ravel()
    values = 4.0 - deg[e[:, 0]] - deg[e[:, 1]] + 3.0 * shared
    return {(int(u), int(v)): float(val) for (u, v), val in zip(e, values)}


# ---------------------------------------------------------------------------
# Ollivier
# ---------------------------------------------------------------------------


def _w1_for_edges(
    edge_chunk: np.ndarray,
    neighbours: list[np.ndarray],
    distances: np.ndarray,
) -> np.ndarray:
    """Exact ``W₁`` between uniform neighbour distributions, for a chunk of edges.

    Solved as a transport LP per edge with the POT library. The ground cost is
    the *precomputed* feature-space distance matrix restricted to the two
    neighbourhoods, which avoids recomputing ``|N(u)|×|N(v)|`` norms per edge.
    """
    import ot

    out = np.empty(len(edge_chunk), dtype=np.float64)
    for i, (u, v) in enumerate(edge_chunk):
        nu, nv = neighbours[u], neighbours[v]
        if len(nu) == 0 or len(nv) == 0:
            out[i] = np.nan
            continue
        a = np.full(len(nu), 1.0 / len(nu))
        b = np.full(len(nv), 1.0 / len(nv))
        cost = np.ascontiguousarray(distances[np.ix_(nu, nv)], dtype=np.float64)
        out[i] = float(ot.emd2(a, b, cost))
    return out


def ollivier_ricci(
    adj: sp.csr_matrix,
    pts: np.ndarray,
    normalisation: str = "distance",
    metric: str = "euclidean",
    distances: np.ndarray | None = None,
    n_jobs: int = 1,
) -> dict[tuple[int, int], float]:
    """Ollivier-Ricci curvature under the chosen normalisation.

    Parameters
    ----------
    normalisation
        ``"none"``
            ``κ = 1 - W₁``, the repo's original convention with an implicit
            ``d(u,v) = 1``. Mismatched units; **not** scale-free. Retained so the
            paper can report what the original convention yields.
        ``"distance"``
            ``κ̂ = 1 - W₁/d(u,v)`` with ``d`` the Euclidean feature distance.
            Matched units, scale-free, invariant under ``G``. The repaired
            estimator of Theorem T1.4 and the default.
        ``"layer_scale"``
            ``1 - W₁/s`` with ``s`` the layer's mean pairwise distance. Also
            invariant, but normalises *globally* rather than per edge. Included
            to separate "the units were wrong" from "the locality was wrong":
            if ``distance`` and ``layer_scale`` agree, the fix is purely
            dimensional; if they disagree, local scale variation matters too.
    n_jobs
        Processes for the transport solves. The work is embarrassingly parallel
        over edges and is the runtime bottleneck of the whole pipeline, so this
        is where a high-core-count node earns its keep.

    Returns
    -------
    Mapping from undirected edge ``(u, v)``, ``u < v``, to curvature.
    """
    if normalisation not in {"none", "distance", "layer_scale"}:
        raise ValueError(f"unknown normalisation {normalisation!r}")

    if distances is None:
        distances = pairwise_distance_matrix(pts, metric)

    e = edge_array(adj)
    if len(e) == 0:
        return {}
    neighbours = neighbour_lists(adj)

    if n_jobs == 1:
        w1 = _w1_for_edges(e, neighbours, distances)
    else:
        chunks = np.array_split(e, max(n_jobs * 4, 1))
        worker = partial(_w1_for_edges, neighbours=neighbours, distances=distances)
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            w1 = np.concatenate(list(pool.map(worker, chunks)))

    if normalisation == "none":
        denominator = np.ones(len(e))
    elif normalisation == "distance":
        denominator = distances[e[:, 0], e[:, 1]]
    else:
        n = distances.shape[0]
        mean_distance = float(distances[~np.eye(n, dtype=bool)].mean())
        denominator = np.full(len(e), mean_distance)

    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = 1.0 - w1 / denominator
    # A zero denominator means coincident points — duplicate activations, which
    # do occur after ReLU saturation. Curvature is undefined there; NaN
    # propagates honestly rather than producing a spurious infinity.
    kappa[~np.isfinite(denominator) | (denominator == 0)] = np.nan

    return {(int(u), int(v)): float(val) for (u, v), val in zip(e, kappa)}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def vertex_curvature(
    edge_curvatures: dict[tuple[int, int], float], n: int
) -> np.ndarray:
    """Per-vertex mean curvature over incident edges.

    Isolated vertices get ``NaN``. NaN edge values are excluded from the mean
    rather than poisoning it, so a single coincident pair does not wipe out a
    whole vertex's curvature.
    """
    totals = np.zeros(n)
    counts = np.zeros(n)
    for (u, v), kappa in edge_curvatures.items():
        if not np.isfinite(kappa):
            continue
        totals[u] += kappa
        counts[u] += 1
        totals[v] += kappa
        counts[v] += 1
    out = np.full(n, np.nan)
    mask = counts > 0
    out[mask] = totals[mask] / counts[mask]
    return out


def edge_curvature_array(
    edge_curvatures: dict[tuple[int, int], float],
) -> tuple[np.ndarray, np.ndarray]:
    """Split a curvature dict into an ``(E, 2)`` edge array and an ``(E,)`` value array."""
    if not edge_curvatures:
        return np.empty((0, 2), dtype=np.int64), np.empty(0)
    items = sorted(edge_curvatures.items())
    e = np.array([k for k, _ in items], dtype=np.int64)
    values = np.array([v for _, v in items], dtype=np.float64)
    return e, values
