"""
Config identity and grid addressing.

These properties are what make the HPC workflow safe. If ``uid`` were unstable,
resubmitting an array would write results to new filenames and silently
duplicate work; if grid ordering were unstable, ``SLURM_ARRAY_TASK_ID`` would
point at a different cell on resubmission and a partially-failed sweep could not
be resumed.
"""

from __future__ import annotations

import pytest

from arc_robustness.config import (
    AttackConfig,
    DataConfig,
    EstimatorConfig,
    ExperimentConfig,
    GraphConfig,
    Grid,
    ModelConfig,
    TrainConfig,
)


def test_uid_is_stable_across_construction_order():
    a = ExperimentConfig(experiment="a2", graph=GraphConfig(k=6, n_per_class=200))
    b = ExperimentConfig(experiment="a2", graph=GraphConfig(n_per_class=200, k=6))
    assert a.uid == b.uid


def test_uid_changes_with_any_field():
    base = ExperimentConfig(experiment="a2")
    variants = [
        base.with_(experiment="a1"),
        base.with_(graph=GraphConfig(k=10)),
        base.with_(estimator=EstimatorConfig(eta_mode="raw")),
        base.with_(attack=AttackConfig(kind="fgsm", epsilon=0.03)),
        base.with_(tag="second-arm"),
    ]
    uids = {base.uid, *(v.uid for v in variants)}
    assert len(uids) == len(variants) + 1


def test_round_trip_through_dict_preserves_uid():
    """Tuples must survive JSON's list coercion, or reloaded configs mismatch."""
    original = ExperimentConfig(
        experiment="a3",
        data=DataConfig(classes=(1, 7)),
        model=ModelConfig(widths=(64, 32, 16)),
        train=TrainConfig(checkpoint_epochs=(1, 5, 10)),
    )
    restored = ExperimentConfig.from_dict(original.to_dict())
    assert restored == original
    assert restored.uid == original.uid


def test_model_uid_ignores_analysis_only_fields():
    """Cells differing only in graph/estimator/attack must share a checkpoint.

    This is what keeps the sweep tractable: training once per network rather than
    once per analysis cell.
    """
    base = ExperimentConfig(experiment="a2")
    same_model = [
        base.with_(graph=GraphConfig(k=15)),
        base.with_(estimator=EstimatorConfig(ollivier_norm="none")),
        base.with_(attack=AttackConfig(kind="pgd_linf", epsilon=0.01)),
        base.with_(tag="anything"),
    ]
    assert {c.model_uid for c in same_model} == {base.model_uid}


def test_model_uid_changes_with_training_seed():
    a = ExperimentConfig(experiment="a3", train=TrainConfig(seed=0))
    b = ExperimentConfig(experiment="a3", train=TrainConfig(seed=1))
    assert a.model_uid != b.model_uid


def test_grid_is_ordered_and_deduplicated():
    grid = Grid("test")
    first = ExperimentConfig(experiment="a1")
    second = ExperimentConfig(experiment="a1", graph=GraphConfig(k=10))
    grid.add(first)
    grid.add(second)
    grid.add(first)  # duplicate, dropped

    assert len(grid) == 2
    assert grid[0].uid == first.uid
    assert grid[1].uid == second.uid


def test_grid_index_is_reproducible():
    def build() -> Grid:
        grid = Grid("test")
        for k in (3, 5, 6, 10):
            for seed in (0, 1):
                grid.add(
                    ExperimentConfig(
                        experiment="a4",
                        graph=GraphConfig(k=k),
                        train=TrainConfig(seed=seed),
                    )
                )
        return grid

    assert [c.uid for c in build()] == [c.uid for c in build()]


def test_grid_unique_models_collapses_shared_checkpoints():
    grid = Grid("test")
    for estimator in ("none", "distance", "layer_scale"):
        for k in (5, 6):
            grid.add(
                ExperimentConfig(
                    experiment="a2",
                    estimator=EstimatorConfig(ollivier_norm=estimator),
                    graph=GraphConfig(k=k),
                )
            )
    assert len(grid) == 6
    assert len(grid.unique_models()) == 1


def test_result_path_is_namespaced_by_experiment(tmp_path):
    config = ExperimentConfig(experiment="a5")
    path = config.result_path(tmp_path)
    assert path.parent.name == "a5"
    assert path.name == f"{config.uid}.npz"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: DataConfig(dataset="imagenet"), "unknown dataset"),
        (lambda: DataConfig(label_mode="scrambled"), "unknown label_mode"),
        (lambda: DataConfig(classes=(2,)), "at least two classes"),
        (lambda: ModelConfig(activation="gelu"), "unknown activation"),
        (lambda: ModelConfig(norm="group"), "unknown norm"),
        (lambda: ModelConfig(widths=()), "at least one hidden layer"),
        (lambda: GraphConfig(graph_type="delaunay"), "unknown graph_type"),
        (lambda: GraphConfig(k=0), "k must be positive"),
        (lambda: EstimatorConfig(ollivier_norm="magic"), "unknown ollivier_norm"),
        (lambda: AttackConfig(kind="deepfool"), "unknown attack kind"),
        (lambda: AttackConfig(kind="fgsm", epsilon=-1), "non-negative"),
    ],
)
def test_invalid_configs_are_rejected(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()


def test_is_invariant_flag_matches_theory():
    """The flag drives figure labelling, so it must track T1.4 exactly."""
    assert EstimatorConfig(ollivier_norm="distance", eta_mode="log_centred").is_invariant
    assert EstimatorConfig(ollivier_norm="layer_scale", eta_mode="log_centred").is_invariant
    assert not EstimatorConfig(ollivier_norm="none", eta_mode="log_centred").is_invariant
    assert not EstimatorConfig(ollivier_norm="distance", eta_mode="raw").is_invariant
    # Feature normalisation confers invariance regardless of the other choices.
    assert EstimatorConfig(
        ollivier_norm="none", eta_mode="raw", feature_norm="unit_mean_distance"
    ).is_invariant


def test_layer_names_are_ordered_by_depth():
    assert ModelConfig(widths=(8,) * 12).layer_names[9] == "relu10"
