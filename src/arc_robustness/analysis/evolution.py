"""
Layer-to-layer evolution statistics: η, ρ(x), and the repaired invariant forms.

The original statistic (arXiv:2509.22362 §3) is, per vertex ``x``, the Pearson
correlation across layers between

    ``η_ℓ(x) = mean_{y ∈ N_ℓ(x)} [ d_{ℓ+1}(x,y) - d_ℓ(x,y) ]``
    ``O_ℓ(x) = mean Ollivier curvature over edges incident to x at layer ℓ``

and a negative ρ is read as Ricci-flow-like behaviour.

Three defects were measured in this repo's results (plan §0.6) and each has a
corresponding repair here:

**(a) Not a property of the learned function.** ``η`` is a difference of
distances at two layers, so the per-layer rescaling group ``G`` of T1 acts on it
by an arbitrary positive factor per term. Applying ``c_ℓ = M^ℓ`` moves the
fraction of vertices with ρ<0 across the entire range [0%, 100%] while leaving
predictions, accuracy and adversarial examples bit-identical — and ``M ∈
[0.8, 1.25]``, within seed-to-seed variation, already spans 36.5% to 83.75%.
Repair: ``eta_mode="log_centred"`` computes ``η̂``, on which ``G`` acts as an
additive per-layer constant that the centring removes.

**(b) A global statistic reported as N local ones.** 54.5% of the variance in
``O_ℓ(x)`` is between-layer. Correlating the five *layer means* alone — zero
per-vertex information — gives ρ = −0.80, stronger than the mean of the
per-vertex distribution (−0.283). Repair: :func:`variance_decomposition` reports
the between-layer share explicitly, so the shared trend can never be silently
passed off as N independent local measurements.

**(c) Five-point sampling.** Each ρ(x) is a Pearson correlation over ``L-1 = 5``
points, where ``sd(r) = 0.50`` under an exchangeable null and 39.1% of random
draws have ``|r| > 0.5``. Individual ρ(x) values carry almost no information.
Repair: :func:`r_layer` correlates across *vertices* at fixed layer (N points,
well conditioned) and is the quantity to report.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from arc_robustness.analysis.graph_utils import neighbour_lists


# ---------------------------------------------------------------------------
# η
# ---------------------------------------------------------------------------


def neighbourhood_change(
    distances_l: np.ndarray,
    distances_next: np.ndarray,
    adj: sp.csr_matrix,
    mode: str = "log_centred",
) -> np.ndarray:
    """Per-vertex neighbourhood change ``η_ℓ(x)`` between consecutive layers.

    Parameters
    ----------
    distances_l, distances_next
        Full pairwise distance matrices at layers ``ℓ`` and ``ℓ+1``.
    adj
        Layer-``ℓ`` adjacency: the neighbourhood ``N_ℓ(x)`` is taken at layer ``ℓ``
        and tracked forward, so this measures what happens to a *fixed* set of
        neighbours rather than comparing two different neighbourhoods.
    mode
        ``"raw"``
            ``mean_y [d_{ℓ+1}(x,y) - d_ℓ(x,y)]``, as originally implemented.
            Provably sign-manipulable under ``G`` (T1.2): all distances are
            non-negative, so a large enough ``c_{ℓ+1}/c_ℓ`` forces ``η > 0`` at
            every vertex at once, and a small enough ratio forces ``η < 0``.
            Verified: all ``2^(L-1)`` sign patterns are realisable.
        ``"log_centred"``
            ``η̂``: the mean log ratio ``mean_y log(d_{ℓ+1}/d_ℓ)``, then centred
            across vertices. Rescaling contributes
            ``log(c_{ℓ+1}/c_ℓ)`` — the *same* additive constant at every vertex —
            which the centring annihilates. Invariant under ``G``.

    Returns
    -------
    ``(N,)`` array; ``NaN`` for isolated vertices.
    """
    if mode not in {"raw", "log_centred"}:
        raise ValueError(f"unknown mode {mode!r}")

    neighbours = neighbour_lists(adj)
    n = distances_l.shape[0]
    out = np.full(n, np.nan)

    for x in range(n):
        nbrs = neighbours[x]
        if len(nbrs) == 0:
            continue
        d_l = distances_l[x, nbrs]
        d_next = distances_next[x, nbrs]

        if mode == "raw":
            out[x] = float(np.mean(d_next - d_l))
        else:
            # Coincident points give a zero distance and an undefined ratio.
            # Dropping those pairs is the honest choice; if it drops everything
            # the vertex is NaN rather than silently zero.
            valid = (d_l > 0) & (d_next > 0)
            if not valid.any():
                continue
            out[x] = float(np.mean(np.log(d_next[valid]) - np.log(d_l[valid])))

    if mode == "log_centred":
        finite = np.isfinite(out)
        if finite.any():
            out[finite] -= out[finite].mean()

    return out


# ---------------------------------------------------------------------------
# ρ and r_layer
# ---------------------------------------------------------------------------


def _safe_pearson(a: np.ndarray, b: np.ndarray, min_points: int = 3) -> float:
    """Pearson correlation, returning NaN rather than 0.0 when undefined.

    The original code returned ``0.0`` for a degenerate (zero-variance) input.
    That is not a neutral choice: 0.0 counts as "not negative" and so shifts the
    headline ``frac(ρ < 0)`` statistic. NaN is excluded from downstream means
    instead of biasing them.
    """
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < min_points:
        return np.nan
    a_m, b_m = a[mask], b[mask]
    if np.std(a_m) < 1e-12 or np.std(b_m) < 1e-12:
        return np.nan
    return float(np.corrcoef(a_m, b_m)[0, 1])


def rho_per_vertex(eta: np.ndarray, curvature: np.ndarray) -> np.ndarray:
    """ρ(x): per-vertex correlation across layers.

    Parameters
    ----------
    eta, curvature
        ``(N, L-1)`` arrays of ``η_ℓ(x)`` and ``O_ℓ(x)``.

    Reported only with the §0.6(c) caveat attached: with ``L-1 = 5`` points the
    null standard deviation of ``r`` is 0.50 and ``P(|r| > 0.5) = 0.391``, so the
    observed median of −0.53 is an ordinary value for pure noise. Use
    :func:`r_layer` for inference and treat this as a descriptive distribution.
    """
    return np.array(
        [_safe_pearson(eta[x], curvature[x]) for x in range(eta.shape[0])]
    )


def r_layer(eta: np.ndarray, curvature: np.ndarray) -> np.ndarray:
    """``r̂_ℓ``: per-layer correlation across vertices.

    The advantage over :func:`rho_per_vertex` is **statistical**: ``N`` points per
    estimate rather than ``L-1``, so each ``r̂_ℓ`` is well conditioned where an
    individual ρ(x) is not.

    It is *not* a free route to invariance, and it is worth being precise about
    why. Pearson correlation is invariant to positive affine transformations of
    either argument, but ``G`` does not act on raw ``η`` affinely: it sends
    ``η_ℓ = mean_y[d_{ℓ+1} - d_ℓ]`` to ``mean_y[c_{ℓ+1} d_{ℓ+1} - c_ℓ d_ℓ]``, a
    *re-weighted difference of two vectors*, which is not a function of the
    original ``η_ℓ`` at all. Measured on the synthetic fixture, ``r̂_1`` moves from
    ``+0.254`` to ``-0.197`` — across zero — as ``c_ℓ = M^ℓ`` sweeps ``M`` over
    ``[0.02, 50]``.

    So ``r̂_ℓ`` is invariant exactly when its *inputs* are: ``eta_mode="log_centred"``
    together with a normalised ``κ̂``, which is how T1.4 states it.
    """
    return np.array(
        [
            _safe_pearson(curvature[:, ell], eta[:, ell])
            for ell in range(eta.shape[1])
        ]
    )


def global_trend_correlation(eta: np.ndarray, curvature: np.ndarray) -> float:
    """Correlation of the *layer means* — the component that carries the signal.

    Five numbers against five numbers, with all per-vertex information
    discarded. On the stored results this gives −0.80, stronger than the mean of
    the per-vertex ρ distribution (−0.283, sd 0.614). Reporting it explicitly is
    what stops that shared trend being mistaken for ``N`` independent local
    measurements; its effective sample size is nearer ``L-1`` than ``N``.

    **Returns NaN under ``eta_mode="log_centred"``, by construction.** The T1.4
    centring subtracts each layer's across-vertex mean, so every layer mean of
    ``η̂`` is exactly zero and the correlation has no variance to work with. That
    is not a failed computation but the sharpest form of §0.6(b)'s finding: the
    global trend *is* the non-invariant component, so under an invariant
    estimator the statistic does not exist. Report it for the raw convention,
    where it is the honest single number the ``ρ(x)`` distribution was standing
    in for, and record its absence for the repaired one.
    """
    return _safe_pearson(np.nanmean(eta, axis=0), np.nanmean(curvature, axis=0))


def variance_decomposition(values: np.ndarray) -> dict[str, float]:
    """Split an ``(N, L-1)`` array's variance into between- and within-layer parts.

    ``between_layer_share`` is the fraction of total variance explained by the
    per-layer means. A high share means the quantity is mostly a global trend
    wearing a per-vertex costume.
    """
    finite = np.isfinite(values)
    if not finite.any():
        return {"between_layer_share": np.nan, "total_variance": np.nan}

    grand_mean = np.nanmean(values)
    layer_means = np.nanmean(values, axis=0)
    counts = finite.sum(axis=0)

    between = float(np.nansum(counts * (layer_means - grand_mean) ** 2))
    total = float(np.nansum((values[finite] - grand_mean) ** 2))
    return {
        "between_layer_share": between / total if total > 0 else np.nan,
        "total_variance": total / max(finite.sum() - 1, 1),
    }


def centre_per_layer(values: np.ndarray) -> np.ndarray:
    """Subtract each layer's across-vertex mean.

    The §0.6(b) diagnostic: applying this to the stored results moves ρ from
    mean −0.283 to +0.105 and ``frac(ρ<0)`` from 72.75% to 41.5% (chance = 50%),
    i.e. the signature vanishes and mildly reverses.

    Note this removes the additive per-layer offset but **not** the
    multiplicative ``c_ℓ``, so it is a diagnostic and not the repair. The repair
    is ``eta_mode="log_centred"``, under which rescaling *is* additive and is
    therefore genuinely removed.
    """
    out = values.astype(np.float64).copy()
    for ell in range(out.shape[1]):
        column = out[:, ell]
        finite = np.isfinite(column)
        if finite.any():
            column[finite] -= column[finite].mean()
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def local_ricci_evolution(
    distances: dict[str, np.ndarray],
    adjacencies: dict[str, sp.csr_matrix],
    vertex_curvature: dict[str, np.ndarray],
    layer_names: list[str],
    eta_mode: str = "log_centred",
) -> dict[str, np.ndarray | float | dict]:
    """Compute the full evolution summary across layers.

    Parameters
    ----------
    distances
        Layer name → full pairwise distance matrix.
    adjacencies
        Layer name → adjacency.
    vertex_curvature
        Layer name → ``(N,)`` per-vertex mean curvature (from
        :func:`~arc_robustness.analysis.ricci.vertex_curvature`).
    layer_names
        Explicit layer order. Passed rather than inferred: ``sorted()`` puts
        ``relu10`` before ``relu2``, which would scramble the layer axis of every
        across-layer statistic once Tier C's 12-layer networks arrive.

    Returns
    -------
    Dict with ``eta``/``curvature`` ``(N, L-1)`` matrices, ``rho``, ``r_layer``,
    the global trend correlation, variance decompositions, and the per-layer
    centred ρ diagnostic.
    """
    n_transitions = len(layer_names) - 1
    if n_transitions < 2:
        raise ValueError("need at least three layers to correlate across transitions")

    n = distances[layer_names[0]].shape[0]
    eta = np.full((n, n_transitions), np.nan)
    curvature = np.full((n, n_transitions), np.nan)

    for ell in range(n_transitions):
        name, next_name = layer_names[ell], layer_names[ell + 1]
        eta[:, ell] = neighbourhood_change(
            distances[name], distances[next_name], adjacencies[name], mode=eta_mode
        )
        curvature[:, ell] = vertex_curvature[name]

    rho = rho_per_vertex(eta, curvature)
    rho_centred = rho_per_vertex(centre_per_layer(eta), centre_per_layer(curvature))

    finite_rho = rho[np.isfinite(rho)]
    return {
        "eta": eta,
        "curvature": curvature,
        "rho": rho,
        "rho_centred": rho_centred,
        "r_layer": r_layer(eta, curvature),
        "global_trend_correlation": global_trend_correlation(eta, curvature),
        "frac_rho_negative": float((finite_rho < 0).mean()) if finite_rho.size else np.nan,
        "frac_rho_centred_negative": float(
            (rho_centred[np.isfinite(rho_centred)] < 0).mean()
        )
        if np.isfinite(rho_centred).any()
        else np.nan,
        "eta_variance": variance_decomposition(eta),
        "curvature_variance": variance_decomposition(curvature),
        "eta_mode": eta_mode,
    }
