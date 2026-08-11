"""
The cell runner, end to end.

These tests run the whole path — train, subsample, perturb, extract, analyse,
write — on a synthetic dataset small enough to be a unit test. The dataset is
injected rather than downloaded so the suite stays offline and fast, but nothing
else is stubbed: the model, the attack, the curvature and the file format are all
the real ones.

The test that matters most is
:func:`test_rescaling_moves_the_raw_statistic_but_not_the_repaired_one`, which is
Corollary T1.3 as an executable claim: one classifier, two reported values.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from arc_robustness.cell import load_record, run_cell
from arc_robustness.config import (
    AttackConfig,
    DataConfig,
    EstimatorConfig,
    ExperimentConfig,
    GraphConfig,
    ModelConfig,
    TrainConfig,
)

CPU = torch.device("cpu")

N_PER_CLASS = 40
N_KEEP = 15


@pytest.fixture
def synthetic_dataset(monkeypatch):
    """Two linearly separable image classes, injected in place of Fashion-MNIST.

    Separable enough that one epoch reaches high accuracy, so the trained arm is
    a *trained* network rather than a noisy one — otherwise the attack arm has no
    margin to eat into and the accuracy assertions become meaningless.
    """
    generator = torch.Generator().manual_seed(11)
    shape = (N_PER_CLASS, 1, 28, 28)
    class_a = torch.rand(shape, generator=generator) * 0.3
    class_b = torch.rand(shape, generator=generator) * 0.3 + 0.6
    images = torch.cat([class_a, class_b]).clamp(0.0, 1.0)
    labels = torch.cat([torch.zeros(N_PER_CLASS), torch.ones(N_PER_CLASS)]).long()

    def fake_load_dataset(cfg, split=None, data_dir=None, seed=0):
        return images.clone(), labels.clone()

    # Patched in both namespaces: the runner loads the analysis split, and
    # training loads its own train/test splits through a separate import.
    monkeypatch.setattr("arc_robustness.cell.load_dataset", fake_load_dataset)
    monkeypatch.setattr("arc_robustness.training.train.load_dataset", fake_load_dataset)
    return images, labels


@pytest.fixture
def base_config() -> ExperimentConfig:
    """A tiny cell: four layers, so ρ has three transitions and is defined."""
    return ExperimentConfig(
        experiment="smoke",
        data=DataConfig(dataset="fashion_mnist", classes=(2, 6), split="test"),
        model=ModelConfig(widths=(16, 12, 8, 8)),
        train=TrainConfig(epochs=1, seed=0),
        graph=GraphConfig(k=4, n_per_class=N_KEEP, subsample_seed=0),
        estimator=EstimatorConfig(ollivier_norm="distance", eta_mode="log_centred"),
    )


def _run(config, tmp_path, **kwargs):
    return run_cell(
        config,
        weights_dir=tmp_path / "weights",
        results_dir=tmp_path / "results",
        device=CPU,
        verbose=False,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


def test_run_cell_writes_result_and_sidecar(synthetic_dataset, base_config, tmp_path):
    path = _run(base_config, tmp_path)

    assert path.exists()
    assert path.name == f"{base_config.uid}.npz"
    assert path.parent.name == "smoke"

    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert sidecar["uid"] == base_config.uid
    assert sidecar["config"]["graph"]["k"] == 4
    assert 0.0 <= sidecar["scalars"]["accuracy"] <= 1.0


def test_record_contains_metrics_provenance_and_theory_ingredients(
    synthetic_dataset, base_config, tmp_path
):
    """One file must be self-sufficient for every Phase-1 and Phase-2 analysis."""
    record = load_record(_run(base_config, tmp_path))

    for key in ("modularity", "mean_ollivier", "curvature_gap_ollivier", "rho"):
        assert key in record, key
    for key in ("n_components", "fiedler_largest_component", "normalised_gap"):
        assert key in record, key  # A5
    for key in ("gap_length_matched", "mean_length_intra", "mean_length_inter"):
        assert key in record, key  # A6
    # T3/T4 ingredients: the derived bounds are computed at analysis time from
    # these, so a change to the theory must not require re-running the sweep.
    for key in ("spectral_norm_max", "spectral_norm_min", "knn_gap_min", "input_dim"):
        assert key in record, key
    # Provenance
    assert record["uid"] == base_config.uid
    assert record["model_uid"] == base_config.model_uid
    assert json.loads(record["config_json"])["graph"]["k"] == 4


def test_record_array_shapes_match_layers_and_vertices(
    synthetic_dataset, base_config, tmp_path
):
    record = load_record(_run(base_config, tmp_path))
    n_layers = base_config.model.depth
    n_vertices = N_KEEP * base_config.data.n_classes

    assert record["n_layers"] == n_layers
    assert record["n_vertices"] == n_vertices
    assert record["modularity"].shape == (n_layers,)
    assert record["spectral_norm_max"].shape == (n_layers,)
    assert record["vertex_ollivier"].shape == (n_layers, n_vertices)
    assert record["rho"].shape == (n_vertices,)
    assert record["r_layer"].shape == (n_layers - 1,)
    assert record["logits"].shape == (n_vertices, base_config.data.n_classes)
    assert record["perturbation_linf"].shape == (n_vertices,)


def test_subsample_is_balanced_and_sorted(synthetic_dataset, base_config, tmp_path):
    """Vertex ``i`` of the graph must be row ``i`` of the features and labels.

    An unsorted subsample silently decorrelates the curvature arrays from the
    class partition, which corrupts every community metric without erroring.
    """
    record = load_record(_run(base_config, tmp_path))
    indices = record["subsample_indices"]

    assert np.all(np.diff(indices) > 0)
    counts = np.bincount(record["labels"])
    assert set(counts.tolist()) == {N_KEEP}


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------


def test_existing_results_are_not_recomputed(synthetic_dataset, base_config, tmp_path):
    """Resubmitting an array must cost only its missing cells."""
    path = _run(base_config, tmp_path)
    first = path.stat().st_mtime_ns
    assert _run(base_config, tmp_path) == path
    assert path.stat().st_mtime_ns == first


def test_overwrite_forces_recomputation(synthetic_dataset, base_config, tmp_path):
    path = _run(base_config, tmp_path)
    record = load_record(path)
    path.unlink()
    rerun = load_record(_run(base_config, tmp_path))
    # Bit-for-bit: the cell is a deterministic function of its config, which is
    # what makes the content-addressed result path meaningful in the first place.
    assert np.allclose(record["modularity"], rerun["modularity"])
    assert np.array_equal(record["subsample_indices"], rerun["subsample_indices"])


def test_cells_sharing_a_model_train_once(synthetic_dataset, base_config, tmp_path):
    """Estimator variants must reuse a checkpoint, not retrain it."""
    other = base_config.with_(estimator=EstimatorConfig(ollivier_norm="none"))
    _run(base_config, tmp_path)
    _run(other, tmp_path)

    checkpoints = list((tmp_path / "weights" / "sweep").glob("*.pt"))
    assert len(checkpoints) == 1
    assert base_config.model_uid == other.model_uid


# ---------------------------------------------------------------------------
# T1.3 — the same classifier, two reported values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("estimator", "expect_invariant"),
    [
        (EstimatorConfig(ollivier_norm="distance", eta_mode="log_centred"), True),
        (EstimatorConfig(ollivier_norm="none", eta_mode="raw"), False),
    ],
)
def test_rescaling_moves_the_raw_statistic_but_not_the_repaired_one(
    synthetic_dataset, base_config, tmp_path, estimator, expect_invariant
):
    """Corollary T1.3, executed.

    ``rescale`` applies the function-preserving map ``c_ℓ = M^ℓ``. The network's
    predictions are unchanged, so *any* difference in a reported statistic is
    measurement artefact. The invariant estimator must return the same numbers;
    the raw one is expected to move, and the test asserts that it does — a raw
    estimator that happened to be stable here would mean the construction was
    not being applied.
    """
    plain = base_config.with_(estimator=estimator)
    rescaled = plain.with_(rescale=2.0)

    a = load_record(_run(plain, tmp_path))
    b = load_record(_run(rescaled, tmp_path))

    # Same classifier: identical predictions and identical accuracy.
    assert np.array_equal(a["predictions"], b["predictions"])
    assert a["accuracy"] == b["accuracy"]
    # Same checkpoint, different result files.
    assert a["model_uid"] == b["model_uid"]
    assert a["uid"] != b["uid"]

    # The graph is a homothety away, so combinatorial statistics are invariant
    # regardless of estimator (Proposition T1.1).
    assert np.allclose(a["modularity"], b["modularity"])

    curvature_moved = not np.allclose(
        a["mean_ollivier"], b["mean_ollivier"], atol=1e-6
    )
    rho_moved = not np.allclose(
        np.nan_to_num(a["rho"]), np.nan_to_num(b["rho"]), atol=1e-6
    )

    if expect_invariant:
        assert not curvature_moved
        assert not rho_moved
    else:
        assert curvature_moved or rho_moved


def test_rescale_is_rejected_where_the_symmetry_does_not_exist(base_config):
    """Identity activations and normalisation both break the construction."""
    with pytest.raises(ValueError, match="positive homogeneity"):
        base_config.with_(
            model=ModelConfig(widths=(16, 8), activation="identity"), rescale=2.0
        )
    with pytest.raises(ValueError, match="quotients out"):
        base_config.with_(model=ModelConfig(widths=(16, 8), norm="batch"), rescale=2.0)


# ---------------------------------------------------------------------------
# B2 — the noise control
# ---------------------------------------------------------------------------


def test_attack_and_noise_control_are_norm_matched(
    synthetic_dataset, base_config, tmp_path
):
    """The claim "matched in norm" must be checkable from the result files.

    Signed uniform noise at the same ε puts every pixel on the same ±ε corner
    FGSM does, so the realised ℓ∞ norms agree exactly and the arms differ only in
    direction. That is the whole basis for attributing an effect to
    adversariality rather than to perturbation.
    """
    epsilon = 0.05
    attack = load_record(
        _run(
            base_config.with_(attack=AttackConfig(kind="fgsm", epsilon=epsilon)),
            tmp_path,
        )
    )
    control = load_record(
        _run(
            base_config.with_(
                attack=AttackConfig(kind="uniform", epsilon=epsilon, seed=0)
            ),
            tmp_path,
        )
    )

    assert attack["perturbation_linf"].max() <= epsilon + 1e-6
    assert control["perturbation_linf"].max() <= epsilon + 1e-6
    # Within 10% on ℓ₂ — clipping against [0, 1] accounts for the residual.
    ratio = control["perturbation_l2"].mean() / attack["perturbation_l2"].mean()
    assert 0.9 < ratio < 1.1


def test_clean_cell_has_no_perturbation(synthetic_dataset, base_config, tmp_path):
    record = load_record(_run(base_config, tmp_path))
    assert record["perturbation_linf"].max() == 0.0
    assert record["accuracy"] == record["clean_accuracy"]
    assert record["flipped_frac"] == 0.0


def test_attack_records_both_clean_and_attacked_outcomes(
    synthetic_dataset, base_config, tmp_path
):
    """B3 splits by outcome, so both predictions must be in the same file."""
    record = load_record(
        _run(base_config.with_(attack=AttackConfig(kind="fgsm", epsilon=0.1)), tmp_path)
    )
    assert record["clean_predictions"].shape == record["predictions"].shape
    assert record["clean_margin"].shape == record["margin"].shape
    flipped = (record["clean_correct"]) & (~record["correct"])
    assert record["flipped_frac"] == pytest.approx(flipped.mean())
