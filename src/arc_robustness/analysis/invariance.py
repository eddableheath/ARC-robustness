"""
The reparameterisation group ``G`` of T1, acting directly on features.

``G`` is the set of per-layer positive rescalings ``c = (c_1, …, c_L) ∈ ℝ^L_{>0}``
induced by ReLU positive homogeneity. On the *weights* it is implemented by
:meth:`~arc_robustness.training.architectures.MLP.rescale_layers`; its entire
effect on the analysed objects is to multiply layer-``ℓ`` activations by ``c_ℓ``.

That makes :func:`rescale_features` the cheap and exact way to test invariance:
no retraining, no forward passes, and no confounding from float error in the
weight updates. An estimator is invariant under ``G`` iff it returns the same
value on ``features`` and on ``rescale_features(features, c)`` for every positive
``c`` — which is directly checkable, and is what the test suite asserts.

What is and is not invariant (T1.1, T1.2, T1.4):

============================================  =========
Quantity                                      Invariant
============================================  =========
k-NN graph, ``Q``, NCut, components           yes
Laplacian spectrum up to scale                yes
Forman, Augmented Forman                      yes (graph-determined)
``κ̂ = 1 - W₁/d``                              yes
``κ = 1 - W₁``                                **no**
``η`` raw                                     **no**
``η̂`` log-centred                             yes
``ρ`` from raw ``η``                           **no**
``r̂_ℓ`` (across vertices)                     yes
============================================  =========
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np


def rescale_features(
    features: dict[str, np.ndarray],
    factors: dict[str, float] | list[float] | tuple[float, ...],
    layer_names: list[str] | None = None,
) -> OrderedDict[str, np.ndarray]:
    """Apply ``c ∈ G`` to a feature dict: layer ``ℓ`` is multiplied by ``c_ℓ``.

    Exactly reproduces the effect of
    :meth:`~arc_robustness.training.architectures.MLP.rescale_layers` on the
    activations, without touching a model.
    """
    names = list(features.keys()) if layer_names is None else layer_names
    if isinstance(factors, dict):
        lookup = factors
    else:
        if len(factors) != len(names):
            raise ValueError(
                f"need one factor per layer ({len(names)}), got {len(factors)}"
            )
        lookup = dict(zip(names, factors))

    if any(c <= 0 for c in lookup.values()):
        raise ValueError("rescaling factors must be strictly positive")

    return OrderedDict((name, features[name] * lookup[name]) for name in names)


def geometric_rescaling(layer_names: list[str], base: float) -> dict[str, float]:
    """The one-parameter family ``c_ℓ = base^ℓ`` used in the §0.6(a) diagnostic.

    Sweeping ``base`` over ``[0.02, 50]`` moved the stored ``frac(ρ<0)`` across
    the full ``[0%, 100%]`` range while leaving the classifier bit-identical. The
    interval ``base ∈ [0.8, 1.25]`` alone — a ±25% change in per-layer scale
    ratios, within seed-to-seed variation — already spans 36.5% to 83.75%, which
    is what makes this a practical objection rather than a pathological one.
    """
    return {name: base ** (i + 1) for i, name in enumerate(layer_names)}


def normalise_features(
    features: dict[str, np.ndarray], mode: str = "none"
) -> OrderedDict[str, np.ndarray]:
    """Per-layer feature normalisation.

    ``mode="unit_mean_distance"`` divides each layer by its mean pairwise
    distance, forcing a canonical representative of each ``G``-orbit. This is the
    brute-force route to invariance: it should make the *raw* estimators agree
    with the repaired ones, which is a useful internal consistency check on T1 —
    if they still disagree after this, something other than scale is going on.

    Uses a subsample for the mean when ``N`` is large, since the mean pairwise
    distance converges quickly and the full ``O(N²)`` matrix is not needed here.
    """
    if mode == "none":
        return OrderedDict(features)
    if mode != "unit_mean_distance":
        raise ValueError(f"unknown normalisation mode {mode!r}")

    out: OrderedDict[str, np.ndarray] = OrderedDict()
    for name, arr in features.items():
        scale = _mean_pairwise_distance(arr)
        out[name] = arr / scale if scale > 0 else arr.copy()
    return out


def _mean_pairwise_distance(
    arr: np.ndarray, max_points: int = 500, seed: int = 0
) -> float:
    if len(arr) > max_points:
        rng = np.random.default_rng(seed)
        arr = arr[rng.choice(len(arr), size=max_points, replace=False)]
    diff = arr[:, None, :] - arr[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    n = len(arr)
    if n < 2:
        return 0.0
    return float(distances[~np.eye(n, dtype=bool)].mean())


def invariance_error(before: np.ndarray | float, after: np.ndarray | float) -> float:
    """Max absolute difference between two estimator outputs, ignoring NaNs.

    The number the test suite thresholds on. Returns ``inf`` if the finite
    support differs, since an estimator that becomes undefined under rescaling is
    not invariant either.
    """
    a = np.atleast_1d(np.asarray(before, dtype=np.float64))
    b = np.atleast_1d(np.asarray(after, dtype=np.float64))
    if a.shape != b.shape:
        return float("inf")
    finite_a, finite_b = np.isfinite(a), np.isfinite(b)
    if not np.array_equal(finite_a, finite_b):
        return float("inf")
    if not finite_a.any():
        return 0.0
    return float(np.abs(a[finite_a] - b[finite_b]).max())
