"""
Baseline adversarial detectors.

These exist so the curvature probe can be judged rather than merely demonstrated.
Every one of them is cheaper than the curvature statistics, and LID in particular
is a *local geometric* statistic computed on the same feature spaces — so it is
the direct competitor. "Our detector achieves AUROC 0.9" means nothing without
"and LID achieves 0.89 at a fraction of the cost".

References
----------
Feinman et al. 2017, *Detecting Adversarial Samples from Artifacts* — kernel
density and Bayesian uncertainty.
Ma et al. 2018, *Characterizing Adversarial Subspaces Using Local Intrinsic
Dimensionality*.
Lee et al. 2018, *A Simple Unified Framework for Detecting Out-of-Distribution
Samples and Adversarial Attacks* — class-conditional Mahalanobis.
Papernot & McDaniel 2018, *Deep k-Nearest Neighbors*.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances


def margin_score(logits: np.ndarray) -> np.ndarray:
    """Negative logit margin: ``-(top1 - top2)``.

    The baseline that most often defeats elaborate detectors, and the one the
    plan's B4 singles out. Higher means less confident, hence more suspicious.
    Free — it needs no reference data at all.
    """
    ordered = np.sort(logits, axis=1)
    return -(ordered[:, -1] - ordered[:, -2])


def lid_score(
    queries: np.ndarray,
    reference: np.ndarray,
    k: int = 20,
    metric: str = "euclidean",
) -> np.ndarray:
    """Local intrinsic dimensionality, maximum-likelihood estimator.

    ``LID(x) = -[ (1/k) Σ_i log(r_i / r_k) ]^{-1}`` over the ``k`` nearest
    reference distances. Adversarial points are argued to lie in subspaces of
    higher local intrinsic dimension, so *larger* is more suspicious.

    Returns NaN where the estimate is undefined (a zero ``r_k``, i.e. a query
    coinciding with ``k`` reference points).
    """
    distances = pairwise_distances(queries, reference, metric=metric)
    nearest = np.sort(distances, axis=1)[:, :k]

    r_k = nearest[:, -1]
    out = np.full(len(queries), np.nan)
    valid = r_k > 0
    if not valid.any():
        return out

    ratios = nearest[valid] / r_k[valid][:, None]
    # Guard the log against exact-zero nearest distances.
    with np.errstate(divide="ignore"):
        logs = np.log(np.clip(ratios, 1e-12, None))
    mean_log = logs.mean(axis=1)
    nonzero = mean_log < 0
    result = np.full(valid.sum(), np.nan)
    result[nonzero] = -1.0 / mean_log[nonzero]
    out[valid] = result
    return out


def mahalanobis_score(
    queries: np.ndarray,
    reference: np.ndarray,
    reference_labels: np.ndarray,
    shrinkage: float = 1e-3,
) -> np.ndarray:
    """Minimum class-conditional Mahalanobis distance, with a shared covariance.

    Fits one Gaussian per class with a tied covariance estimated on the reference
    set, and returns the distance to the nearest class centroid. Larger is more
    suspicious.

    The activation dimension often exceeds the reference count (512 units against
    400 points), so the empirical covariance is singular. ``shrinkage`` applies
    ridge regularisation to the diagonal; without it this silently returns
    garbage rather than failing.
    """
    classes = np.unique(reference_labels)
    dimension = reference.shape[1]

    centred = []
    means = {}
    for cls in classes:
        block = reference[reference_labels == cls]
        mean = block.mean(axis=0)
        means[cls] = mean
        centred.append(block - mean)
    pooled = np.vstack(centred)

    covariance = np.cov(pooled, rowvar=False)
    covariance = np.atleast_2d(covariance)
    trace_scale = np.trace(covariance) / max(dimension, 1)
    covariance += shrinkage * trace_scale * np.eye(dimension)
    precision = np.linalg.pinv(covariance)

    scores = np.full((len(queries), len(classes)), np.inf)
    for j, cls in enumerate(classes):
        delta = queries - means[cls]
        scores[:, j] = np.einsum("ij,jk,ik->i", delta, precision, delta)
    return scores.min(axis=1)


def kernel_density_score(
    queries: np.ndarray,
    reference: np.ndarray,
    reference_labels: np.ndarray,
    predicted_labels: np.ndarray,
    bandwidth: float | None = None,
) -> np.ndarray:
    """Negative log kernel density under the predicted class's reference points.

    Following Feinman et al.: density is evaluated only against reference points
    of the *predicted* class, so an input the model confidently assigns to a class
    it does not resemble scores as suspicious. Higher is more suspicious.

    ``bandwidth=None`` uses the median pairwise reference distance, a robust
    scale-free default that avoids hand-tuning per layer.
    """
    if bandwidth is None:
        sample = reference[: min(len(reference), 400)]
        pairwise = pairwise_distances(sample)
        n = len(sample)
        bandwidth = float(np.median(pairwise[~np.eye(n, dtype=bool)]))
    if bandwidth <= 0:
        return np.full(len(queries), np.nan)

    out = np.full(len(queries), np.nan)
    for cls in np.unique(predicted_labels):
        mask = predicted_labels == cls
        block = reference[reference_labels == cls]
        if len(block) == 0 or not mask.any():
            continue
        distances = pairwise_distances(queries[mask], block)
        # log-sum-exp for stability, then negate so larger = more suspicious.
        scaled = -((distances / bandwidth) ** 2)
        peak = scaled.max(axis=1, keepdims=True)
        log_density = (
            peak.ravel()
            + np.log(np.exp(scaled - peak).sum(axis=1))
            - np.log(len(block))
        )
        out[mask] = -log_density
    return out
