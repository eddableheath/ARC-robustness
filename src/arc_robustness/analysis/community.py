"""
Community-strength and connectivity metrics.

Definitions follow Appendix A.2.3 of arXiv:2509.22362, with communities taken to
be ground-truth class labels (no community detection).

Two departures, both driven by measured degeneracies in the stored results:

**A5 — algebraic connectivity is uninformative past layer 3.** The stored
Fiedler values were ``[0.320, 0.065, 0.006, 0.000, -0.000, -0.000]``: the graph
disconnects (4 components at ``relu6``) and a Fiedler value of 0 then means only
"disconnected", carrying no further signal. :func:`connectivity_metrics` reports
component count, the Fiedler value of the *largest component*, and the
normalised-Laplacian spectral gap, all of which stay informative after
fragmentation and are what T5's Cheeger-type bounds actually need.

**Modularity saturates immediately.** ``Q = [0.327, 0.419, 0.433, 0.424, 0.425,
0.424]`` is at ~85% of its two-community ceiling (~0.5) by layer 2 and flat
after. As a clean-network progress measure it has almost no dynamic range, and
the "Q increases across layers" narrative rests on a single transition. It may
still be a good *adversarial* measure, since it can fall.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.linalg import eigh, eigvalsh

from arc_robustness.analysis.graph_utils import degrees


# ---------------------------------------------------------------------------
# Partition quality
# ---------------------------------------------------------------------------


def modularity(adj: sp.csr_matrix, labels: np.ndarray) -> float:
    """``Q = 1/(2m) Σ_{uv} [A_uv - k_u k_v / 2m] δ(C_u, C_v)``."""
    m = float(adj.sum()) / 2.0
    if m == 0.0:
        return 0.0
    k = degrees(adj)
    q = 0.0
    for cls in np.unique(labels):
        mask = labels == cls
        e_c = float(adj[mask][:, mask].sum()) / 2.0
        a_c = float(k[mask].sum())
        q += e_c / m - (a_c / (2.0 * m)) ** 2
    return float(q)


def modularity_ceiling(labels: np.ndarray) -> float:
    """Maximum attainable ``Q`` for this partition, given perfect separation.

    With a fully separated graph the cut term vanishes and
    ``Q = 1 - Σ_c (vol_c / 2m)²``. For two equal communities that is 0.5.
    Reported alongside ``Q`` so a value can be read as a fraction of what is
    achievable rather than on an absolute scale that suggests false headroom.
    """
    _, counts = np.unique(labels, return_counts=True)
    shares = counts / counts.sum()
    return float(1.0 - np.sum(shares**2))


def normalised_cut(adj: sp.csr_matrix, labels: np.ndarray) -> float:
    """``NCut = ½ Σ_i cut(C_i) / vol(C_i)``. Lower is cleaner separation."""
    k = degrees(adj)
    total = 0.0
    for cls in np.unique(labels):
        mask = labels == cls
        vol = float(k[mask].sum())
        if vol == 0.0:
            continue
        total += float(adj[mask][:, ~mask].sum()) / vol
    return float(total / 2.0)


def curvature_gap(
    edge_curvatures: dict[tuple[int, int], float], labels: np.ndarray
) -> float:
    """``ΔO = (O_intra - O_inter) / σ_pooled``, with ``σ_pooled = sqrt(½(σ²_in + σ²_out))``.

    Positive means intra-class edges are more positively curved. The stored
    results are *negative* for layers 2–6 under the unnormalised estimator
    (``[0.393, -0.621, -0.773, -0.562, -0.405, -0.363]``), contradicting the
    slide narrative; A6 exists to decide whether that is an artefact of the
    normalisation or real k-NN geometry. See :func:`curvature_gap_decomposition`.
    """
    intra, inter = [], []
    for (u, v), kappa in edge_curvatures.items():
        if not np.isfinite(kappa):
            continue
        (intra if labels[u] == labels[v] else inter).append(kappa)
    if not intra or not inter:
        return float("nan")
    sigma = float(np.sqrt(0.5 * (np.var(intra) + np.var(inter))))
    if sigma < 1e-12:
        return float("nan")
    return float((np.mean(intra) - np.mean(inter)) / sigma)


def curvature_gap_decomposition(
    edge_curvatures: dict[tuple[int, int], float],
    labels: np.ndarray,
    distances: np.ndarray,
) -> dict[str, float]:
    """Diagnose ΔO by conditioning on edge length (experiment A6).

    A6 tests two competing explanations for the negative gap:

    1. *Artefact of the normalisation* — the unnormalised ``κ = 1 - W₁``
       penalises edges in high-scale regions, and if inter-class edges sit in
       tighter neighbourhoods they will look spuriously positive.
    2. *Real k-NN geometry* — in a well-separated graph the few surviving
       inter-class edges live in the dense overlap region where neighbourhoods
       genuinely are tight, so they *should* be more positively curved. If so,
       ΔO on a k-NN graph is a poor separability proxy, which is worth saying in
       print.

    The discriminator is whether the gap survives conditioning on edge length.
    Under (1) matching intra- and inter-class edges by length should collapse the
    gap; under (2) it should persist. Returns the raw gap, the gap computed
    within length deciles then averaged, and the mean edge length per class type.
    """
    intra_mask, values, lengths = [], [], []
    for (u, v), kappa in edge_curvatures.items():
        if not np.isfinite(kappa):
            continue
        intra_mask.append(labels[u] == labels[v])
        values.append(kappa)
        lengths.append(distances[u, v])

    intra_mask = np.array(intra_mask)
    values = np.array(values)
    lengths = np.array(lengths)

    if intra_mask.all() or (~intra_mask).all() or len(values) == 0:
        return {
            "gap_raw": np.nan,
            "gap_length_matched": np.nan,
            "mean_length_intra": np.nan,
            "mean_length_inter": np.nan,
            "n_intra": int(intra_mask.sum()),
            "n_inter": int((~intra_mask).sum()),
        }

    sigma = float(np.sqrt(0.5 * (np.var(values[intra_mask]) + np.var(values[~intra_mask]))))
    gap_raw = (
        float((values[intra_mask].mean() - values[~intra_mask].mean()) / sigma)
        if sigma > 1e-12
        else np.nan
    )

    # Stratify on edge length and average the within-stratum gaps, weighting by
    # the number of inter-class edges available in each stratum.
    deciles = np.quantile(lengths, np.linspace(0, 1, 11))
    gaps, weights = [], []
    for lo, hi in zip(deciles[:-1], deciles[1:]):
        in_bin = (lengths >= lo) & (lengths <= hi)
        a, b = values[in_bin & intra_mask], values[in_bin & ~intra_mask]
        if len(a) < 2 or len(b) < 2:
            continue
        s = float(np.sqrt(0.5 * (np.var(a) + np.var(b))))
        if s < 1e-12:
            continue
        gaps.append((a.mean() - b.mean()) / s)
        weights.append(len(b))

    gap_matched = (
        float(np.average(gaps, weights=weights)) if gaps else np.nan
    )

    return {
        "gap_raw": gap_raw,
        "gap_length_matched": gap_matched,
        "mean_length_intra": float(lengths[intra_mask].mean()),
        "mean_length_inter": float(lengths[~intra_mask].mean()),
        "n_intra": int(intra_mask.sum()),
        "n_inter": int((~intra_mask).sum()),
    }


# ---------------------------------------------------------------------------
# Connectivity (A5)
# ---------------------------------------------------------------------------


def algebraic_connectivity(adj: sp.csr_matrix) -> float:
    """Fiedler value of the unnormalised Laplacian on the whole graph.

    Retained for continuity with the original results, but see A5: this is 0 for
    any disconnected graph and so goes dead from layer 4 onward. Prefer
    :func:`connectivity_metrics`.
    """
    dense = adj.toarray().astype(np.float64)
    lap = np.diag(dense.sum(axis=1)) - dense
    if lap.shape[0] < 2:
        return float("nan")
    return float(eigvalsh(lap, subset_by_index=[0, 1])[1])


def connectivity_metrics(adj: sp.csr_matrix) -> dict[str, float]:
    """Connectivity summary that stays informative after fragmentation.

    Returns
    -------
    ``n_components``
        Number of connected components. The stored results reach 4 at ``relu6``.
    ``largest_component_frac``
        Fraction of vertices in the giant component.
    ``fiedler_largest_component``
        Fiedler value restricted to the giant component — the quantity the
        original panel was trying to report.
    ``normalised_gap``
        Second-smallest eigenvalue of the symmetric normalised Laplacian on the
        giant component. Bounded in ``[0, 2]`` and scale-free in the degree
        sequence, so it is comparable across layers with different densities in
        a way the unnormalised Fiedler value is not.
    ``cheeger_lower/upper``
        Cheeger inequality bracket ``λ/2 <= h <= sqrt(2λ)`` on the conductance,
        from ``normalised_gap``. Needed for T5.2's curvature-derived certificate
        on how many edges must be cut to detach a point from its class cluster.
    """
    n = adj.shape[0]
    n_components, membership = connected_components(adj, directed=False)

    sizes = np.bincount(membership)
    largest = int(np.argmax(sizes))
    mask = membership == largest
    sub = adj[mask][:, mask]

    result = {
        "n_components": float(n_components),
        "largest_component_frac": float(sizes[largest] / n),
        "fiedler_largest_component": np.nan,
        "normalised_gap": np.nan,
        "cheeger_lower": np.nan,
        "cheeger_upper": np.nan,
    }

    if sub.shape[0] < 3:
        return result

    dense = sub.toarray().astype(np.float64)
    unnormalised = np.diag(dense.sum(axis=1)) - dense
    result["fiedler_largest_component"] = float(
        eigvalsh(unnormalised, subset_by_index=[0, 1])[1]
    )

    norm_lap = laplacian(sp.csr_matrix(dense), normed=True).toarray()
    eigenvalues = eigh(norm_lap, eigvals_only=True)
    gap = float(eigenvalues[1])
    # Numerical noise can push the second eigenvalue slightly negative.
    gap = max(gap, 0.0)
    result["normalised_gap"] = gap
    result["cheeger_lower"] = gap / 2.0
    result["cheeger_upper"] = float(np.sqrt(2.0 * gap))
    return result


def degree_summary(adj: sp.csr_matrix) -> dict[str, float]:
    """Degree distribution summary, supporting the T7 argument about Forman.

    On a symmetric k-NN graph ``deg >= k`` by construction, and in deeper layers
    degrees concentrate at exactly ``k`` (measured: ``7.02 ± 1.10`` at ``relu6``
    with ``k = 6``). Forman curvature is then close to the constant ``4 - 2k``,
    so ``frac_at_floor`` is the direct quantitative support for the claim that
    Forman measures hub-ness only.
    """
    deg = degrees(adj)
    floor = float(deg.min()) if len(deg) else np.nan
    return {
        "mean": float(deg.mean()),
        "std": float(deg.std()),
        "min": floor,
        "max": float(deg.max()),
        "frac_at_floor": float((deg == floor).mean()),
    }
