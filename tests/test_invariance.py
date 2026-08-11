"""
Theorem T1 as a test suite.

These tests are the paper's first contribution in executable form, and they are
two-sided on purpose:

* the **repaired** estimators must be invariant under the rescaling group ``G``;
* the **raw** estimators must *fail* to be.

The second half matters as much as the first. A test suite that only checked the
repair would pass just as happily if ``G`` were being applied incorrectly, or if
the synthetic fixture happened to be scale-free. Asserting that the raw
estimators visibly move — and that their *sign* can be driven either way —
confirms the group action is real and the objection has teeth.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from arc_robustness.analysis.evolution import (
    global_trend_correlation,
    neighbourhood_change,
    r_layer,
    rho_per_vertex,
)
from arc_robustness.analysis.graph_utils import build_graph, pairwise_distance_matrix
from arc_robustness.analysis.invariance import (
    geometric_rescaling,
    invariance_error,
    normalise_features,
    rescale_features,
)
from arc_robustness.analysis.pipeline import analyse_features
from arc_robustness.analysis.ricci import ollivier_ricci, vertex_curvature
from arc_robustness.config import EstimatorConfig

TOL = 1e-9


# ---------------------------------------------------------------------------
# T1.1 — what the group leaves alone
# ---------------------------------------------------------------------------


def test_graph_is_invariant(two_cluster_features, graph_config):
    """T1.1: rescaling is a homothety, so it preserves distance *ordering*."""
    features, _ = two_cluster_features
    factors = geometric_rescaling(list(features), base=3.7)
    rescaled = rescale_features(features, factors)

    for name in features:
        before = build_graph(features[name], graph_config)
        after = build_graph(rescaled[name], graph_config)
        assert (before != after).nnz == 0, f"graph changed at {name}"


def test_combinatorial_metrics_are_invariant(
    two_cluster_features, graph_config, repaired_estimator
):
    """Q, NCut, component structure and the degree sequence are graph-determined."""
    features, labels = two_cluster_features
    rescaled = rescale_features(features, geometric_rescaling(list(features), 0.15))

    before = analyse_features(features, labels, graph_config, repaired_estimator)
    after = analyse_features(rescaled, labels, graph_config, repaired_estimator)

    for key in (
        "modularity", "normalised_cut", "n_components",
        "degree_mean", "mean_forman", "mean_af",
        "curvature_gap_forman", "curvature_gap_af",
    ):
        assert invariance_error(before[key], after[key]) < TOL, f"{key} not invariant"


# ---------------------------------------------------------------------------
# T1.2 — what the group moves
# ---------------------------------------------------------------------------


def test_unnormalised_ollivier_is_not_invariant(two_cluster_features, graph_config):
    """``κ = 1 - W₁`` mixes activation units with a unit hop distance."""
    features, _ = two_cluster_features
    name = "relu2"
    pts = features[name]
    adj = build_graph(pts, graph_config)

    before = vertex_curvature(ollivier_ricci(adj, pts, normalisation="none"), len(pts))
    after = vertex_curvature(
        ollivier_ricci(adj, pts * 5.0, normalisation="none"), len(pts)
    )

    assert invariance_error(before, after) > 1.0, (
        "unnormalised Ollivier should move substantially under rescaling"
    )


def test_normalised_ollivier_is_invariant(two_cluster_features, graph_config):
    """``κ̂ = 1 - W₁/d`` is a ratio of quantities that scale identically."""
    features, _ = two_cluster_features
    for name in features:
        pts = features[name]
        adj = build_graph(pts, graph_config)
        before = vertex_curvature(
            ollivier_ricci(adj, pts, normalisation="distance"), len(pts)
        )
        after = vertex_curvature(
            ollivier_ricci(adj, pts * 12.5, normalisation="distance"), len(pts)
        )
        assert invariance_error(before, after) < 1e-8, f"κ̂ not invariant at {name}"


def test_raw_eta_sign_is_manipulable(two_cluster_features, graph_config):
    """T1.2 constructive claim, the sharp form.

    For *any* target sign pattern ``s ∈ {±1}^{L-1}`` there is a ``c ∈ G`` making
    ``sign(η_ℓ(x)) = s_ℓ`` at **every** vertex simultaneously — while the network
    computes an identical function. Verified for all ``2^(L-1)`` patterns.
    """
    features, _ = two_cluster_features
    names = list(features)
    n_transitions = len(names) - 1
    big = 1e6

    for pattern in itertools.product([1, -1], repeat=n_transitions):
        factors = [1.0]
        for sign in pattern:
            factors.append(factors[-1] * (big if sign > 0 else 1.0 / big))
        rescaled = rescale_features(features, factors)
        distances = {n: pairwise_distance_matrix(rescaled[n]) for n in names}

        for ell, want in enumerate(pattern):
            eta = neighbourhood_change(
                distances[names[ell]],
                distances[names[ell + 1]],
                build_graph(rescaled[names[ell]], graph_config),
                mode="raw",
            )
            finite = eta[np.isfinite(eta)]
            achieved = np.sign(finite)
            assert np.all(achieved == want), (
                f"pattern {pattern}, transition {ell}: wanted all {want:+d}, "
                f"got {np.unique(achieved)}"
            )


def test_log_centred_eta_is_invariant(two_cluster_features, graph_config):
    """``η̂``: rescaling acts additively on ``log d``, and centring removes it."""
    features, _ = two_cluster_features
    names = list(features)
    rescaled = rescale_features(features, geometric_rescaling(names, 0.03))

    d_before = {n: pairwise_distance_matrix(features[n]) for n in names}
    d_after = {n: pairwise_distance_matrix(rescaled[n]) for n in names}

    for ell in range(len(names) - 1):
        adj = build_graph(features[names[ell]], graph_config)
        before = neighbourhood_change(
            d_before[names[ell]], d_before[names[ell + 1]], adj, mode="log_centred"
        )
        after = neighbourhood_change(
            d_after[names[ell]], d_after[names[ell + 1]], adj, mode="log_centred"
        )
        assert invariance_error(before, after) < 1e-9, f"η̂ not invariant at {ell}"


# ---------------------------------------------------------------------------
# T1.3 / T1.5 — the headline statistic
# ---------------------------------------------------------------------------


def _rho_and_frac(features, labels, graph_config, estimator):
    record = analyse_features(features, labels, graph_config, estimator)
    return record["rho"], record["frac_rho_negative"]


def test_frac_rho_negative_is_manipulable_under_raw_estimator(
    two_cluster_features, graph_config, raw_estimator
):
    """Corollary T1.3: the headline statistic is not a property of the function.

    Two feature sets related by ``c ∈ G`` — identical network, identical
    predictions — must be able to report grossly different ``frac(ρ<0)``.
    """
    features, labels = two_cluster_features
    names = list(features)

    _, frac_contract = _rho_and_frac(
        rescale_features(features, geometric_rescaling(names, 0.05)),
        labels, graph_config, raw_estimator,
    )
    _, frac_expand = _rho_and_frac(
        rescale_features(features, geometric_rescaling(names, 20.0)),
        labels, graph_config, raw_estimator,
    )

    assert abs(frac_contract - frac_expand) > 0.4, (
        "raw ρ should swing widely under G; got "
        f"{frac_contract:.3f} vs {frac_expand:.3f}"
    )


def test_r_layer_is_invariant_under_repaired_estimator(
    two_cluster_features, graph_config, repaired_estimator
):
    """``r̂_ℓ`` is invariant when built from ``κ̂`` and ``η̂``, as T1.4 states."""
    features, labels = two_cluster_features
    rescaled = rescale_features(features, geometric_rescaling(list(features), 7.0))

    before = analyse_features(features, labels, graph_config, repaired_estimator)
    after = analyse_features(rescaled, labels, graph_config, repaired_estimator)
    assert invariance_error(before["r_layer"], after["r_layer"]) < 1e-8


def test_r_layer_is_not_invariant_under_raw_estimator(
    two_cluster_features, graph_config, raw_estimator
):
    """Correlating across vertices does **not** confer invariance by itself.

    Pearson correlation is invariant to positive affine maps of its arguments,
    but ``G`` does not act affinely on raw ``η``: it re-weights the two terms of
    a difference, giving something that is not a function of the original ``η`` at
    all. So ``r̂_ℓ`` built on raw ``η`` moves, and can cross zero.

    This test exists because the opposite claim is an easy and tempting error —
    it was made, and this is what refuted it.
    """
    features, labels = two_cluster_features
    names = list(features)

    contracted = analyse_features(
        rescale_features(features, geometric_rescaling(names, 0.02)),
        labels, graph_config, raw_estimator,
    )["r_layer"]
    expanded = analyse_features(
        rescale_features(features, geometric_rescaling(names, 50.0)),
        labels, graph_config, raw_estimator,
    )["r_layer"]

    assert invariance_error(contracted, expanded) > 0.1, (
        f"raw r_layer should move under G: {contracted} vs {expanded}"
    )
    assert np.nanmin(contracted * expanded) < 0, (
        "expected at least one layer's r to change sign under G"
    )


def test_repaired_rho_is_invariant(
    two_cluster_features, graph_config, repaired_estimator
):
    """Corollary T1.5: an invariant across-layer ρ exists, built from ``η̂``."""
    features, labels = two_cluster_features
    rescaled = rescale_features(features, geometric_rescaling(list(features), 0.4))

    before = analyse_features(features, labels, graph_config, repaired_estimator)
    after = analyse_features(rescaled, labels, graph_config, repaired_estimator)

    assert invariance_error(before["rho"], after["rho"]) < 1e-8
    assert abs(before["frac_rho_negative"] - after["frac_rho_negative"]) < 1e-9


# ---------------------------------------------------------------------------
# Feature normalisation as an alternative route
# ---------------------------------------------------------------------------


def test_feature_normalisation_confers_invariance_on_raw_estimators(
    two_cluster_features, graph_config
):
    """Normalising to unit mean pairwise distance picks a canonical orbit rep.

    So even the *raw* estimators become invariant. This is the internal
    consistency check on T1: if the raw estimators still moved after this, the
    non-invariance would be coming from something other than scale.
    """
    features, labels = two_cluster_features
    estimator = EstimatorConfig(
        ollivier_norm="none", eta_mode="raw", feature_norm="unit_mean_distance"
    )
    rescaled = rescale_features(features, geometric_rescaling(list(features), 6.25))

    before = analyse_features(features, labels, graph_config, estimator)
    after = analyse_features(rescaled, labels, graph_config, estimator)

    assert invariance_error(before["mean_ollivier"], after["mean_ollivier"]) < 1e-7
    assert abs(before["frac_rho_negative"] - after["frac_rho_negative"]) < 1e-9


def test_layer_scale_equals_none_on_normalised_features(
    two_cluster_features, graph_config
):
    """On unit-mean-distance features ``s_ℓ = 1``, so the two must coincide exactly.

    ``layer_scale`` computes ``1 − W₁/s_ℓ`` and ``none`` computes ``1 − W₁``, so
    normalising the features away makes them the *same function*. An exact
    identity like this is worth a test precisely because it is exact: any drift
    means the layer scale is being computed from something other than the mean
    pairwise distance it is documented to use.
    """
    features, labels = two_cluster_features
    common = {"eta_mode": "raw", "feature_norm": "unit_mean_distance"}
    plain = analyse_features(
        features, labels, graph_config, EstimatorConfig(ollivier_norm="none", **common)
    )
    scaled = analyse_features(
        features,
        labels,
        graph_config,
        EstimatorConfig(ollivier_norm="layer_scale", **common),
    )

    assert invariance_error(plain["mean_ollivier"], scaled["mean_ollivier"]) < 1e-12
    assert plain["frac_rho_negative"] == pytest.approx(
        scaled["frac_rho_negative"], abs=1e-12
    )


def test_invariance_does_not_imply_agreement(two_cluster_features, graph_config):
    """Two invariant estimators may still disagree, and the paper must say so.

    ``1 − W₁`` on normalised features normalises *globally* (one scale per layer);
    ``1 − W₁/d(u,v)`` normalises *locally* (per edge). Both are invariant under
    ``G``; neither is thereby canonical. On the a2 results they give 0.14 and
    0.60 for the same network, so treating either as *the* repaired number would
    be a choice presented as a consequence.
    """
    features, labels = two_cluster_features
    global_norm = analyse_features(
        features,
        labels,
        graph_config,
        EstimatorConfig(
            ollivier_norm="none", eta_mode="raw", feature_norm="unit_mean_distance"
        ),
    )
    local_norm = analyse_features(
        features,
        labels,
        graph_config,
        EstimatorConfig(ollivier_norm="distance", eta_mode="log_centred"),
    )
    assert global_norm["frac_rho_negative"] != pytest.approx(
        local_norm["frac_rho_negative"], abs=1e-6
    )


def test_normalise_features_gives_unit_mean_distance(two_cluster_features):
    features, _ = two_cluster_features
    normalised = normalise_features(features, "unit_mean_distance")
    for name, arr in normalised.items():
        distances = pairwise_distance_matrix(arr)
        n = len(arr)
        mean = distances[~np.eye(n, dtype=bool)].mean()
        assert mean == pytest.approx(1.0, rel=0.05), name


# ---------------------------------------------------------------------------
# Reporting obligations from plan section 0.6
# ---------------------------------------------------------------------------


def test_global_trend_is_reported_and_can_exceed_per_vertex_mean(
    two_cluster_features, graph_config, raw_estimator
):
    """§0.6(b): the layer-mean correlation is a distinct, stronger statistic.

    It must be reported separately so a single shared trend is never presented as
    ``N`` independent local measurements.
    """
    features, labels = two_cluster_features
    record = analyse_features(features, labels, graph_config, raw_estimator)

    assert "global_trend_correlation" in record
    assert "eta_between_layer_share" in record
    assert np.isfinite(record["global_trend_correlation"])

    direct = global_trend_correlation(record["eta"], record["eta_curvature"])
    assert direct == pytest.approx(record["global_trend_correlation"], abs=1e-12)


def test_degenerate_correlation_is_nan_not_zero():
    """A zero-variance input must not be silently counted as "not negative".

    Returning 0.0 there — as the original implementation did — biases
    ``frac(ρ<0)`` by whatever fraction of vertices are degenerate.
    """
    eta = np.zeros((4, 5))
    curvature = np.tile(np.arange(5.0), (4, 1))
    rho = rho_per_vertex(eta, curvature)
    assert np.all(np.isnan(rho))

    r = r_layer(eta, curvature)
    assert np.all(np.isnan(r))
