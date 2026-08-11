"""Shared fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from arc_robustness.config import EstimatorConfig, GraphConfig


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(20260601)


@pytest.fixture(scope="session")
def two_cluster_features() -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Synthetic 5-layer features with two progressively separating clusters.

    Deliberately mimics the qualitative shape of a trained network: the class
    means pull apart with depth while the within-class scatter shrinks, and the
    overall scale is *not* constant across layers — which is exactly the
    condition under which the unnormalised estimators misbehave.
    """
    generator = np.random.default_rng(7)
    n_per_class, dim = 40, 12
    labels = np.repeat([0, 1], n_per_class)

    features: dict[str, np.ndarray] = {}
    for layer in range(5):
        separation = 1.0 + 1.5 * layer
        scatter = 1.0 / (1.0 + 0.4 * layer)
        scale = 2.0**layer  # per-layer scale drift
        centre = np.zeros(dim)
        centre[0] = separation
        block_a = generator.normal(-centre, scatter, size=(n_per_class, dim))
        block_b = generator.normal(centre, scatter, size=(n_per_class, dim))
        features[f"relu{layer + 1}"] = scale * np.vstack([block_a, block_b])

    return features, labels


@pytest.fixture
def graph_config() -> GraphConfig:
    return GraphConfig(k=5, n_per_class=40, graph_type="symmetric")


@pytest.fixture
def raw_estimator() -> EstimatorConfig:
    """The original convention: unnormalised curvature, raw η."""
    return EstimatorConfig(ollivier_norm="none", eta_mode="raw")


@pytest.fixture
def repaired_estimator() -> EstimatorConfig:
    """The T1.4 repair: distance-normalised curvature, log-centred η."""
    return EstimatorConfig(ollivier_norm="distance", eta_mode="log_centred")
