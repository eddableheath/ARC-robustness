"""
Local Ricci Evolution Coefficient ρ(x) — arXiv:2509.22362, Section 3.

For each vertex x, ρ(x) is the Pearson correlation (across layers ℓ = 1…L-1)
between:

  η_ℓ(x) = (1/deg_ℓ(x)) · Σ_{y ∈ N_ℓ(x)} [d_{ℓ+1}(x,y) − d_ℓ(x,y)]
            mean change in Euclidean distance to layer-ℓ neighbours

  O_ℓ(x) = mean Ollivier-Ricci curvature over edges incident to x at layer ℓ

A negative ρ(x) means that positive-curvature regions contract and
negative-curvature regions expand — the hallmark of Ricci flow dynamics.
The paper reports 73–98% of vertices with ρ < 0 in trained networks.
"""

import numpy as np
import scipy.sparse as sp


def local_ricci_evolution(
    features: dict[str, np.ndarray],
    adjs: dict[str, sp.csr_matrix],
    ollivier: dict[str, dict[tuple[int, int], float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ρ(x) for every vertex and the per-layer Pearson r_layer.

    Parameters
    ----------
    features  : layer_name -> (N, D) activation array
    adjs      : layer_name -> symmetric binary CSR adjacency
    ollivier  : layer_name -> edge-curvature dict from ricci.ollivier_ricci

    Returns
    -------
    rho     : (N,) per-vertex Pearson correlation across all layers
    r_layer : (L-1,) per-layer Pearson(O_ℓ, η_ℓ) across vertices
    """
    layer_names = list(features.keys())
    N = next(iter(features.values())).shape[0]
    L = len(layer_names)

    eta = np.zeros((N, L - 1))      # mean neighbourhood distance change
    O_local = np.zeros((N, L - 1))  # mean local Ollivier curvature

    for ell in range(L - 1):
        name_l = layer_names[ell]
        name_l1 = layer_names[ell + 1]
        adj_l = adjs[name_l]
        pts_l = features[name_l]
        pts_l1 = features[name_l1]
        ollivier_l = ollivier[name_l]
        adj_arr = adj_l.toarray()

        for x in range(N):
            nbrs = np.where(adj_arr[x] > 0)[0]
            if len(nbrs) == 0:
                continue

            # η: mean distance change to layer-ℓ neighbours
            d_l = np.linalg.norm(pts_l[x] - pts_l[nbrs], axis=1)
            d_l1 = np.linalg.norm(pts_l1[x] - pts_l1[nbrs], axis=1)
            eta[x, ell] = float(np.mean(d_l1 - d_l))

            # O_local: mean Ollivier curvature on incident edges
            kappas = [
                ollivier_l[(min(x, int(y)), max(x, int(y)))]
                for y in nbrs
                if (min(x, int(y)), max(x, int(y))) in ollivier_l
            ]
            if kappas:
                O_local[x, ell] = float(np.mean(kappas))

    # Per-vertex Pearson correlation across all layers
    rho = np.zeros(N)
    for x in range(N):
        ex, ox = eta[x], O_local[x]
        if np.std(ex) < 1e-10 or np.std(ox) < 1e-10:
            rho[x] = 0.0
        else:
            rho[x] = float(np.corrcoef(ex, ox)[0, 1])

    # Per-layer Pearson: corr(O_ℓ(x), η_ℓ(x)) across vertices at each layer ℓ
    r_layer = np.full(L - 1, np.nan)
    for ell in range(L - 1):
        o, e = O_local[:, ell], eta[:, ell]
        mask = np.isfinite(o) & np.isfinite(e)
        if mask.sum() > 2 and np.std(o[mask]) > 1e-10 and np.std(e[mask]) > 1e-10:
            r_layer[ell] = float(np.corrcoef(o[mask], e[mask])[0, 1])

    return rho, r_layer
