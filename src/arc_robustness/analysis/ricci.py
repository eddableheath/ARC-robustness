"""
Discrete Ricci curvature on unweighted, undirected graphs.

Three notions are implemented, matching arXiv:2509.22362:

  Forman-Ricci       F(u,v)  = 4 - deg(u) - deg(v)
  Aug. Forman-Ricci  AF(u,v) = 4 - deg(u) - deg(v) + 3|N(u) ∩ N(v)|
  Ollivier-Ricci     O(u,v)  = 1 - W₁(μ_u, μ_v) / d(u,v)

For Ollivier, μ_u is the uniform distribution over the neighbours of u,
d(u,v) = 1 (graph distance for adjacent vertices), and W₁ is computed
using Euclidean distances in the original feature space as the ground cost.

All functions accept a scipy CSR adjacency matrix and return a dict mapping
undirected edge (u, v) with u < v to the curvature value.
"""

import numpy as np
import scipy.sparse as sp


def forman_ricci(adj: sp.csr_matrix) -> dict[tuple[int, int], float]:
    """F(u,v) = 4 - deg(u) - deg(v). O(|E|) to compute."""
    deg = np.array(adj.sum(axis=1)).flatten()
    return {
        (int(u), int(v)): 4.0 - deg[u] - deg[v]
        for u, v in zip(*adj.nonzero())
        if u < v
    }


def augmented_forman_ricci(adj: sp.csr_matrix) -> dict[tuple[int, int], float]:
    """AF(u,v) = 4 - deg(u) - deg(v) + 3|N(u) ∩ N(v)|.

    |N(u) ∩ N(v)| = (A²)[u,v] for a binary adjacency matrix without
    self-loops, which counts shared one-hop neighbours.
    """
    deg = np.array(adj.sum(axis=1)).flatten()
    shared = (adj @ adj).toarray()
    return {
        (int(u), int(v)): 4.0 - deg[u] - deg[v] + 3.0 * shared[u, v]
        for u, v in zip(*adj.nonzero())
        if u < v
    }


def ollivier_ricci(
    adj: sp.csr_matrix,
    pts: np.ndarray,
) -> dict[tuple[int, int], float]:
    """O(u,v) = 1 - W₁(μ_u, μ_v) using Euclidean feature-space costs.

    Wasserstein-1 between the uniform distributions on N(u) and N(v) is
    solved exactly via linear programming (scipy EMD via the POT library).
    Graph distance d(u,v) = 1 for all adjacent vertices.

    Parameters
    ----------
    adj : symmetric binary CSR adjacency
    pts : (N, D) array of feature vectors (high-dimensional activations,
          not UMAP projections)
    """
    import ot  # POT: Python Optimal Transport

    adj_dense = adj.toarray()
    curvatures: dict[tuple[int, int], float] = {}

    for u, v in zip(*adj.nonzero()):
        if u >= v:
            continue
        nu = np.where(adj_dense[u] > 0)[0]
        nv = np.where(adj_dense[v] > 0)[0]

        a = np.ones(len(nu), dtype=np.float64) / len(nu)
        b = np.ones(len(nv), dtype=np.float64) / len(nv)

        # Ground cost: pairwise Euclidean distances between neighbours' features
        M = np.linalg.norm(pts[nu][:, None] - pts[nv][None, :], axis=-1)
        M = np.ascontiguousarray(M, dtype=np.float64)

        w1 = float(ot.emd2(a, b, M))
        curvatures[(int(u), int(v))] = 1.0 - w1  # d(u,v) = 1

    return curvatures


def vertex_curvature(
    edge_curvatures: dict[tuple[int, int], float],
    n: int,
) -> np.ndarray:
    """Return per-vertex mean curvature (average over incident edges).

    Parameters
    ----------
    edge_curvatures : dict from edge (u,v) with u<v to curvature
    n               : number of vertices
    """
    totals = np.zeros(n)
    counts = np.zeros(n)
    for (u, v), kappa in edge_curvatures.items():
        totals[u] += kappa
        counts[u] += 1
        totals[v] += kappa
        counts[v] += 1
    mask = counts > 0
    result = np.full(n, np.nan)
    result[mask] = totals[mask] / counts[mask]
    return result
