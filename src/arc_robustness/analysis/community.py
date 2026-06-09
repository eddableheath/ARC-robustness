"""
Community-strength metrics for graphs with known class partitions.

All metrics match the definitions in Appendix A.2.3 of arXiv:2509.22362.
Communities are taken to be the ground-truth class labels — no community
detection is performed.

  Modularity Q         Newman (2004)
  Normalised Cut NCut  Shi & Malik (2000)
  Algebraic Connect.   Second-smallest eigenvalue of the graph Laplacian
  Curvature Gap ΔO     (O_intra - O_inter) / pooled_std
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigvalsh


def modularity(adj: sp.csr_matrix, labels: np.ndarray) -> float:
    """Q = 1/(2|E|) · Σ_{u,v} [A_{uv} - k_u k_v / (2|E|)] · δ(C_u, C_v).

    Values above 0.3 indicate strong community structure.
    """
    m = float(adj.sum()) / 2.0
    if m == 0.0:
        return 0.0
    k = np.array(adj.sum(axis=1)).flatten()
    Q = 0.0
    for cls in np.unique(labels):
        mask = labels == cls
        e_c = float(adj[mask, :][:, mask].sum()) / 2.0
        a_c = float(k[mask].sum())
        Q += e_c / m - (a_c / (2.0 * m)) ** 2
    return float(Q)


def normalised_cut(adj: sp.csr_matrix, labels: np.ndarray) -> float:
    """NCut = ½ · Σ_i cut(C_i) / vol(C_i).

    cut(C_i) = edges crossing the boundary; vol(C_i) = sum of degrees.
    Lower NCut → cleaner separation.
    """
    k = np.array(adj.sum(axis=1)).flatten()
    ncut = 0.0
    for cls in np.unique(labels):
        mask = labels == cls
        vol = float(k[mask].sum())
        if vol == 0.0:
            continue
        cut = float(adj[mask, :][:, ~mask].sum())
        ncut += cut / vol
    return float(ncut / 2.0)


def algebraic_connectivity(adj: sp.csr_matrix) -> float:
    """Second-smallest eigenvalue of the unnormalised Laplacian (Fiedler value).

    > 0 iff the graph is connected; larger values indicate stronger connectivity.
    Uses dense eigensolver — suitable for the graph sizes used here (~400 nodes).
    """
    A = adj.toarray().astype(np.float64)
    L = np.diag(A.sum(axis=1)) - A
    # Request only the two smallest eigenvalues
    vals = eigvalsh(L, subset_by_index=[0, 1])
    return float(vals[1])


def curvature_gap(
    edge_curvatures: dict[tuple[int, int], float],
    labels: np.ndarray,
) -> float:
    """ΔO = (O_intra - O_inter) / σ_pooled.

    σ_pooled = sqrt(½ (σ²_intra + σ²_inter)).
    Positive ΔO → intra-community edges are more positively curved.
    """
    intra, inter = [], []
    for (u, v), kappa in edge_curvatures.items():
        (intra if labels[u] == labels[v] else inter).append(kappa)
    if not intra or not inter:
        return float("nan")
    sigma = np.sqrt(0.5 * (np.var(intra) + np.var(inter)))
    if sigma < 1e-10:
        return float("nan")
    return float((np.mean(intra) - np.mean(inter)) / sigma)
