"""
Grid definitions.

The properties tested here are the ones a sweep silently depends on. A grid that
builds is not enough: if ordering drifts, ``SLURM_ARRAY_TASK_ID`` addresses a
different cell after a resubmission and a resumed array corrupts its own results;
if a grid's checkpoint count is wrong, the training stage races the analysis
stage; if a control arm is missing, the falsification table has a hole in it that
no downstream code will notice.
"""

from __future__ import annotations

import pytest

from arc_robustness.config import ExperimentConfig
from arc_robustness.experiments import (
    CONTROL_ARMS,
    ESTIMATOR_PAIR,
    GRID_BUILDERS,
    RESCALE_LADDER,
    build_grid,
    list_grids,
)

GRID_NAMES = sorted(GRID_BUILDERS)


@pytest.mark.parametrize("name", GRID_NAMES)
def test_every_grid_builds_and_is_non_empty(name):
    grid = build_grid(name)
    assert len(grid) > 0
    assert all(isinstance(c, ExperimentConfig) for c in grid)


@pytest.mark.parametrize("name", GRID_NAMES)
def test_grid_ordering_is_reproducible(name):
    """Two builds must agree cell for cell, or array indices are meaningless."""
    first = [c.uid for c in build_grid(name)]
    second = [c.uid for c in build_grid(name)]
    assert first == second


@pytest.mark.parametrize("name", GRID_NAMES)
def test_grid_uids_are_unique(name):
    """Duplicates would have two array elements writing the same result file."""
    uids = [c.uid for c in build_grid(name)]
    assert len(set(uids)) == len(uids)


@pytest.mark.parametrize("name", GRID_NAMES)
def test_every_cell_declares_its_own_experiment(name):
    """Cells must land in ``results/<name>/``, not in a sibling grid's directory."""
    assert all(c.experiment == name for c in build_grid(name))


def test_unknown_grid_names_are_rejected():
    with pytest.raises(KeyError, match="unknown grid"):
        build_grid("a99")


def test_list_grids_reports_cells_and_checkpoints():
    reported = dict((name, (cells, models)) for name, cells, models in list_grids())
    assert set(reported) == set(GRID_BUILDERS)
    for name, (cells, models) in reported.items():
        assert 0 < models <= cells, name


# ---------------------------------------------------------------------------
# A1 — the falsification table
# ---------------------------------------------------------------------------


def test_a1_covers_every_control_arm_under_both_estimators():
    """The point of A1 is coverage, so a missing arm must fail loudly here."""
    grid = build_grid("a1")
    for arm in CONTROL_ARMS:
        cells = [c for c in grid if c.tag == f"control:{arm}"]
        assert cells, f"control arm {arm!r} missing from a1"
        estimators = {c.estimator.is_invariant for c in cells}
        assert estimators == {True, False}, arm


def test_a1_control_arms_are_genuinely_different_networks():
    """Each arm must differ from the trained arm in the *weights*, not just a tag.

    A tag-only difference would produce a falsification table whose rows are
    identical numbers — a failure mode that looks like a null result.
    """
    grid = build_grid("a1")
    trained = {c.model_uid for c in grid if c.tag == "control:trained"}
    for arm in CONTROL_ARMS:
        if arm == "trained":
            continue
        arm_uids = {c.model_uid for c in grid if c.tag == f"control:{arm}"}
        assert not (arm_uids & trained), arm


def test_a1_memorisation_arms_have_an_accuracy_target():
    """A shuffled-label net that never fit is not a memorisation control.

    The 400-epoch floor is empirical, not a round number: a 150-epoch budget was
    measured to leave the shuffled arm at 82% train accuracy and still climbing
    at ~0.001/epoch. Early stopping means a larger budget is free once the target
    is reached, so the floor guards against a future edit quietly reintroducing
    an undertrained control.
    """
    grid = build_grid("a1")
    for arm in ("shuffled_labels", "random_data"):
        cells = [c for c in grid if c.tag == f"control:{arm}"]
        assert all(c.train.target_train_acc is not None for c in cells), arm
        assert all(c.train.epochs >= 400 for c in cells), arm


def test_a1_untrained_arm_does_not_train():
    cells = [c for c in build_grid("a1") if c.tag == "control:untrained"]
    assert cells and all(not c.train.trained for c in cells)


# ---------------------------------------------------------------------------
# A2 — estimators and the rescaling sweep
# ---------------------------------------------------------------------------


def test_a2_rescale_sweep_reuses_a_single_checkpoint():
    """The whole point of T1.3: one network, many reported values.

    If the rescale cells trained separate networks, the sweep would no longer
    demonstrate that *the same classifier* reports different statistics — it
    would just show seed variation.
    """
    cells = [c for c in build_grid("a2") if c.tag == "rescale-sweep"]
    assert {c.rescale for c in cells} == set(RESCALE_LADDER)
    assert len({c.model_uid for c in cells}) == 1


def test_a2_rescale_sweep_cells_are_distinct_results():
    """Same checkpoint, so the uid must still separate them."""
    cells = [c for c in build_grid("a2") if c.tag == "rescale-sweep"]
    assert len({c.uid for c in cells}) == len(cells)


def test_a2_estimator_cross_is_complete():
    cells = [c for c in build_grid("a2") if c.tag == "estimator-cross"]
    combinations = {
        (c.estimator.ollivier_norm, c.estimator.eta_mode, c.estimator.feature_norm)
        for c in cells
    }
    assert len(combinations) == 3 * 2 * 2


def test_a2_includes_the_brute_force_invariance_check():
    """``feature_norm`` must appear *with* the raw estimators, or it checks nothing.

    Normalising features to unit mean pairwise distance kills the symmetry by
    force, so raw-on-normalised should agree with repaired-on-raw. That
    cross-check only exists if the combination is in the grid.
    """
    cells = build_grid("a2").configs
    assert any(
        c.estimator.feature_norm == "unit_mean_distance"
        and c.estimator.ollivier_norm == "none"
        and c.estimator.eta_mode == "raw"
        for c in cells
    )


# ---------------------------------------------------------------------------
# A3 / A4 / B1
# ---------------------------------------------------------------------------


def test_a3_crosses_training_and_subsample_seeds():
    """Both axes, fully crossed — otherwise the two noise sources are conflated."""
    grid = build_grid("a3")
    pairs = {(c.train.seed, c.graph.subsample_seed) for c in grid}
    train_seeds = {p[0] for p in pairs}
    subsample_seeds = {p[1] for p in pairs}
    assert len(train_seeds) >= 5 and len(subsample_seeds) >= 5
    assert len(pairs) == len(train_seeds) * len(subsample_seeds)


def test_a3_has_one_checkpoint_per_training_seed():
    grid = build_grid("a3")
    assert len(grid.unique_models()) == len({c.train.seed for c in grid})


def test_a4_varies_one_graph_axis_at_a_time():
    """Marginal sweep, not a product: every cell differs from the main arm in
    exactly one of k / n / graph_type / metric."""
    from arc_robustness.experiments import MAIN_GRAPH

    for config in build_grid("a4"):
        differences = sum(
            (
                config.graph.k != MAIN_GRAPH.k,
                config.graph.n_per_class != MAIN_GRAPH.n_per_class,
                config.graph.graph_type != MAIN_GRAPH.graph_type,
                config.graph.metric != MAIN_GRAPH.metric,
            )
        )
        assert differences <= 1, config.describe()


def test_a4_reuses_one_checkpoint():
    """Graph construction is an analysis choice; it must not retrain anything."""
    assert len(build_grid("a4").unique_models()) == 1


def test_b1_pairs_every_attack_epsilon_with_a_noise_control():
    """B2's non-negotiable: no ε at which an attack has no perturbative control."""
    grid = build_grid("b1")
    adversarial_eps = {c.attack.epsilon for c in grid if c.attack.is_adversarial}
    control_eps = {
        c.attack.epsilon for c in grid if c.attack.kind in {"gaussian", "uniform"}
    }
    assert adversarial_eps
    assert adversarial_eps <= control_eps


def test_b1_samples_below_the_accuracy_collapse():
    """Plan §0.8: the old sweep started at ε=0.03, already below chance.

    At least three budgets must sit strictly inside (0, 0.03) or the geometric
    transition is again unsampled.
    """
    epsilons = {
        c.attack.epsilon for c in build_grid("b1") if c.attack.is_adversarial
    }
    assert len([e for e in epsilons if 0 < e < 0.03]) >= 3


def test_b1_has_a_clean_anchor():
    """A sweep with no ε=0 cell has nothing to measure Δmetric against."""
    assert any(c.attack.kind == "none" for c in build_grid("b1"))


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def test_smoke_matches_the_main_arm_depth():
    """ρ is a correlation over ``L-1`` transitions.

    At depth 3 it is undefined and the smoke run would pass while never
    exercising the project's central statistic — so depth is the one dimension
    the smoke grid must not shrink.
    """
    from arc_robustness.experiments import MAIN_MODEL

    for config in build_grid("smoke"):
        assert config.model.depth == MAIN_MODEL.depth


def test_smoke_exercises_every_runner_path():
    """Clean, attacked, noise-controlled, and both estimator conventions."""
    grid = build_grid("smoke")
    kinds = {c.attack.kind for c in grid}
    assert {"none", "fgsm", "uniform"} <= kinds
    assert {c.estimator.is_invariant for c in grid} == {True, False}


def test_smoke_is_small_enough_to_be_a_smoke_test():
    grid = build_grid("smoke")
    assert len(grid) <= 6
    assert len(grid.unique_models()) == 1
    assert all(c.train.epochs <= 3 for c in grid)


def test_estimator_pair_is_one_raw_and_one_invariant():
    """Every grid reports side by side, so the pair must actually be a contrast."""
    assert {e.is_invariant for e in ESTIMATOR_PAIR} == {True, False}
