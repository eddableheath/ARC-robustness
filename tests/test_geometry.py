"""
Correctness of the curvature and graph primitives on cases with known answers.

The invariance suite checks that estimators respect a symmetry; these check that
they compute the right thing in the first place. Both are needed — an estimator
that always returns zero is perfectly invariant.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from arc_robustness.analysis.community import (
    connectivity_metrics,
    curvature_gap,
    degree_summary,
    modularity,
    modularity_ceiling,
    normalised_cut,
)
from arc_robustness.analysis.graph_utils import (
    build_graph,
    edge_array,
    knn_gap,
    neighbour_lists,
    pairwise_distance_matrix,
)
from arc_robustness.analysis.ricci import (
    augmented_forman_ricci,
    forman_ricci,
    ollivier_ricci,
    vertex_curvature,
)
from arc_robustness.config import GraphConfig


def _adjacency_from_edges(n: int, pairs: list[tuple[int, int]]) -> sp.csr_matrix:
    rows = [u for u, v in pairs] + [v for u, v in pairs]
    cols = [v for u, v in pairs] + [u for u, v in pairs]
    data = np.ones(len(rows))
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


# ---------------------------------------------------------------------------
# Forman
# ---------------------------------------------------------------------------


def test_forman_on_path_graph():
    """Path 0-1-2: degrees 1,2,1, so F(0,1) = 4-1-2 = 1 and F(1,2) = 1."""
    adj = _adjacency_from_edges(3, [(0, 1), (1, 2)])
    forman = forman_ricci(adj)
    assert forman == {(0, 1): 1.0, (1, 2): 1.0}


def test_forman_on_triangle():
    """Triangle: every degree is 2, so F = 4-2-2 = 0 on all edges."""
    adj = _adjacency_from_edges(3, [(0, 1), (1, 2), (0, 2)])
    assert set(forman_ricci(adj).values()) == {0.0}


def test_augmented_forman_adds_shared_neighbours():
    """On a triangle each edge shares exactly one neighbour: AF = 0 + 3·1 = 3."""
    adj = _adjacency_from_edges(3, [(0, 1), (1, 2), (0, 2)])
    assert set(augmented_forman_ricci(adj).values()) == {3.0}


def test_augmented_forman_equals_forman_when_triangle_free():
    """A path has no shared neighbours, so AF must coincide with F."""
    adj = _adjacency_from_edges(4, [(0, 1), (1, 2), (2, 3)])
    assert forman_ricci(adj) == augmented_forman_ricci(adj)


# ---------------------------------------------------------------------------
# Ollivier
# ---------------------------------------------------------------------------


def test_ollivier_normalised_on_equilateral_triangle():
    """Symmetric configuration: all edges must share one curvature value."""
    angles = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    pts = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    adj = _adjacency_from_edges(3, [(0, 1), (1, 2), (0, 2)])

    values = list(ollivier_ricci(adj, pts, normalisation="distance").values())
    assert len(values) == 3
    assert np.allclose(values, values[0])


def test_ollivier_normalisation_scales_as_expected():
    """``κ̂`` is scale-free while ``1 - W₁`` is not, on the same graph."""
    generator = np.random.default_rng(3)
    pts = generator.normal(size=(20, 4))
    adj = build_graph(pts, GraphConfig(k=4))

    raw_1 = np.array(list(ollivier_ricci(adj, pts, normalisation="none").values()))
    raw_2 = np.array(list(ollivier_ricci(adj, pts * 3, normalisation="none").values()))
    # 1 - 3·W₁ vs 1 - W₁, so the deviation from 1 triples exactly.
    assert np.allclose(1.0 - raw_2, 3.0 * (1.0 - raw_1))

    hat_1 = np.array(list(ollivier_ricci(adj, pts, normalisation="distance").values()))
    hat_2 = np.array(
        list(ollivier_ricci(adj, pts * 3, normalisation="distance").values())
    )
    assert np.allclose(hat_1, hat_2)


def test_ollivier_parallel_matches_serial():
    generator = np.random.default_rng(11)
    pts = generator.normal(size=(40, 5))
    adj = build_graph(pts, GraphConfig(k=5))

    serial = ollivier_ricci(adj, pts, n_jobs=1)
    parallel = ollivier_ricci(adj, pts, n_jobs=2)
    assert serial.keys() == parallel.keys()
    assert np.allclose(
        [serial[k] for k in serial], [parallel[k] for k in serial], equal_nan=True
    )


def test_vertex_curvature_excludes_nan_edges():
    """A single undefined edge must not wipe out an entire vertex's curvature."""
    values = {(0, 1): 1.0, (0, 2): np.nan, (1, 2): 3.0}
    out = vertex_curvature(values, 3)
    assert out[0] == pytest.approx(1.0)  # only the finite incident edge counts
    assert out[1] == pytest.approx(2.0)
    assert out[2] == pytest.approx(3.0)


def test_vertex_curvature_isolated_vertex_is_nan():
    out = vertex_curvature({(0, 1): 1.0}, 3)
    assert np.isnan(out[2])


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def test_symmetric_graph_degree_floor_is_k():
    """The OR rule guarantees ``deg >= k`` — the basis of the T7 argument."""
    generator = np.random.default_rng(5)
    pts = generator.normal(size=(60, 6))
    for k in (3, 5, 8):
        adj = build_graph(pts, GraphConfig(k=k, graph_type="symmetric"))
        assert degree_summary(adj)["min"] >= k


def test_mutual_graph_degree_ceiling_is_k():
    """The AND rule gives ``deg <= k``, the opposite regime."""
    generator = np.random.default_rng(5)
    pts = generator.normal(size=(60, 6))
    for k in (3, 5, 8):
        adj = build_graph(pts, GraphConfig(k=k, graph_type="mutual"))
        assert degree_summary(adj)["max"] <= k


def test_graphs_are_symmetric_and_loop_free():
    generator = np.random.default_rng(5)
    pts = generator.normal(size=(40, 4))
    for graph_type in ("symmetric", "mutual", "eps_ball"):
        adj = build_graph(pts, GraphConfig(k=5, graph_type=graph_type))
        assert (adj != adj.T).nnz == 0, graph_type
        assert adj.diagonal().sum() == 0, graph_type


def test_neighbour_lists_match_dense_lookup():
    generator = np.random.default_rng(9)
    pts = generator.normal(size=(30, 3))
    adj = build_graph(pts, GraphConfig(k=4))
    dense = adj.toarray()
    for i, nbrs in enumerate(neighbour_lists(adj)):
        assert np.array_equal(np.sort(nbrs), np.where(dense[i] > 0)[0])


def test_edge_array_is_upper_triangular():
    generator = np.random.default_rng(9)
    pts = generator.normal(size=(25, 3))
    e = edge_array(build_graph(pts, GraphConfig(k=4)))
    assert np.all(e[:, 0] < e[:, 1])


def test_knn_gap_matches_definition():
    """``gap(u) = d_{k+1}(u) - d_k(u)`` on an explicit 1-D configuration."""
    pts = np.array([[0.0], [1.0], [3.0], [7.0]])
    distances = pairwise_distance_matrix(pts)
    # From vertex 0 the sorted neighbour distances are 1, 3, 7.
    gaps = knn_gap(distances, k=1)
    assert gaps[0] == pytest.approx(2.0)  # d_2 - d_1 = 3 - 1
    gaps2 = knn_gap(distances, k=2)
    assert gaps2[0] == pytest.approx(4.0)  # d_3 - d_2 = 7 - 3


def test_knn_gap_rejects_impossible_k():
    distances = pairwise_distance_matrix(np.array([[0.0], [1.0], [2.0]]))
    with pytest.raises(ValueError, match="at least"):
        knn_gap(distances, k=5)


# ---------------------------------------------------------------------------
# Community metrics
# ---------------------------------------------------------------------------


def test_modularity_of_two_disconnected_cliques_hits_ceiling():
    """Perfectly separated equal communities attain ``Q = 1 - Σ(vol share)² = 0.5``."""
    pairs = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]
    adj = _adjacency_from_edges(6, pairs)
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert modularity(adj, labels) == pytest.approx(0.5)
    assert modularity_ceiling(labels) == pytest.approx(0.5)


def test_normalised_cut_is_zero_when_separated_and_positive_otherwise():
    labels = np.array([0, 0, 1, 1])
    separated = _adjacency_from_edges(4, [(0, 1), (2, 3)])
    assert normalised_cut(separated, labels) == pytest.approx(0.0)

    bridged = _adjacency_from_edges(4, [(0, 1), (2, 3), (1, 2)])
    assert normalised_cut(bridged, labels) > 0.0


def test_curvature_gap_sign_follows_intra_vs_inter():
    labels = np.array([0, 0, 1, 1])
    # Intra-class edges more positively curved -> positive gap.
    values = {(0, 1): 1.0, (2, 3): 1.2, (1, 2): -1.0, (0, 3): -0.8}
    assert curvature_gap(values, labels) > 0
    flipped = {k: -v for k, v in values.items()}
    assert curvature_gap(flipped, labels) < 0


def test_curvature_gap_is_nan_without_both_edge_types():
    labels = np.array([0, 0, 1, 1])
    assert np.isnan(curvature_gap({(0, 1): 1.0}, labels))


# ---------------------------------------------------------------------------
# Connectivity (A5)
# ---------------------------------------------------------------------------


def test_connectivity_detects_components():
    """Two disjoint triangles: the legacy Fiedler value is 0 and says nothing more.

    The replacement metrics stay informative — this is the whole point of A5.
    """
    adj = _adjacency_from_edges(6, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)])
    metrics = connectivity_metrics(adj)
    assert metrics["n_components"] == 2
    assert metrics["largest_component_frac"] == pytest.approx(0.5)
    assert metrics["fiedler_largest_component"] > 0
    assert metrics["normalised_gap"] > 0


def test_connected_graph_reports_single_component():
    adj = _adjacency_from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    metrics = connectivity_metrics(adj)
    assert metrics["n_components"] == 1
    assert metrics["largest_component_frac"] == pytest.approx(1.0)


def test_normalised_gap_is_bounded_and_brackets_cheeger():
    """``λ ∈ [0, 2]`` for the normalised Laplacian, and ``λ/2 <= h <= sqrt(2λ)``."""
    generator = np.random.default_rng(4)
    pts = generator.normal(size=(50, 5))
    metrics = connectivity_metrics(build_graph(pts, GraphConfig(k=5)))
    assert 0.0 <= metrics["normalised_gap"] <= 2.0
    assert metrics["cheeger_lower"] <= metrics["cheeger_upper"]


def test_degree_summary_reports_floor_concentration():
    """``frac_at_floor`` is the quantitative support for T7."""
    generator = np.random.default_rng(6)
    pts = generator.normal(size=(80, 3))
    summary = degree_summary(build_graph(pts, GraphConfig(k=6)))
    assert summary["min"] >= 6
    assert 0.0 <= summary["frac_at_floor"] <= 1.0
