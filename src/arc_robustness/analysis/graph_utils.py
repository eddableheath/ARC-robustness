"""
Utilities for building k-NN graphs from activation arrays.
All graphs are symmetric, unweighted (binary adjacency), and stored as
scipy CSR sparse matrices.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph


def build_knn_graph(pts: np.ndarray, k: int) -> sp.csr_matrix:
    """Return the symmetric unweighted k-NN adjacency matrix for *pts*.

    Symmetrisation ensures u->v and v->u are both present even when the
    k-NN relationship is not mutually reciprocal.
    """
    G = kneighbors_graph(pts, n_neighbors=k, mode="connectivity", include_self=False)
    G_sym = ((G + G.T) > 0).astype(np.float64)
    return sp.csr_matrix(G_sym)


def edges(adj: sp.csr_matrix) -> list[tuple[int, int]]:
    """Return a list of undirected edges (u, v) with u < v."""
    rows, cols = adj.nonzero()
    return [(int(u), int(v)) for u, v in zip(rows, cols) if u < v]
