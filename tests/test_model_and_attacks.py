"""
Model, rescaling symmetry, and attack contracts.

The rescaling test here is the *weight-space* counterpart of the feature-space
tests in ``test_invariance.py``: it confirms that ``G`` as implemented on the
network genuinely preserves the function, which is what licenses treating
feature rescaling as a faithful stand-in for it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from arc_robustness.attacks import (
    apply_attack,
    build_perturbation_arms,
    matched_noise,
    perturbation_norms,
)
from arc_robustness.attacks.runner import matching_norm_for
from arc_robustness.config import AttackConfig, DataConfig, ModelConfig
from arc_robustness.features import _layer_sort_key, extract_features
from arc_robustness.training.architectures import init_model


@pytest.fixture
def small_model():
    data = DataConfig(dataset="fashion_mnist", classes=(2, 6))
    return init_model(ModelConfig(widths=(24, 18, 12)), data, seed=0).eval()


@pytest.fixture
def images():
    generator = torch.Generator().manual_seed(1)
    return torch.rand((12, 1, 28, 28), generator=generator)


@pytest.fixture
def labels():
    return torch.tensor([0, 1] * 6)


# ---------------------------------------------------------------------------
# Rescaling symmetry (T1.2) in weight space
# ---------------------------------------------------------------------------


def test_rescaling_preserves_the_function(small_model, images):
    with torch.no_grad():
        before = small_model(images).clone()
    small_model.rescale_layers([1.7, 0.4, 3.1])
    with torch.no_grad():
        after = small_model(images)

    assert torch.allclose(before, after, atol=1e-4)
    assert torch.equal(before.argmax(1), after.argmax(1))


def test_rescaling_scales_each_layer_by_its_own_factor(small_model, images):
    """Each layer is scaled by ``c_ℓ``, not by a running product.

    The compensating division on the *incoming* weight cancels the previous
    layer's factor. Getting this wrong yields a cumulative product, which still
    preserves the function but rescales the layers by the wrong amounts — so the
    η-manipulation experiment would be sweeping a different family than intended.
    """
    factors = [1.5, 0.6, 2.4]
    with torch.no_grad():
        _, before = small_model.forward_features(images)
    small_model.rescale_layers(factors)
    with torch.no_grad():
        _, after = small_model.forward_features(images)

    for factor, name in zip(factors, small_model.layer_names):
        # Threshold relative to the layer's own scale. Post-ReLU values barely
        # above zero carry catastrophic relative error in float32 — an absolute
        # 1e-8 cutoff admits activations whose ratio is accurate to only ~1e-3,
        # which says nothing about the correctness of the rescaling.
        reference = before[name].max()
        active = before[name] > 1e-3 * reference
        if not active.any():
            continue
        ratio = (after[name][active] / before[name][active]).numpy()
        assert np.allclose(ratio, factor, rtol=1e-4), (
            f"{name}: max deviation {np.abs(ratio - factor).max():.2e}"
        )


def test_rescaling_rejects_non_relu_and_normalised_architectures():
    data = DataConfig(classes=(2, 6))
    linear = init_model(ModelConfig(widths=(8, 8), activation="identity"), data, seed=0)
    with pytest.raises(ValueError, match="positive homogeneity"):
        linear.rescale_layers([2.0, 2.0])

    normalised = init_model(ModelConfig(widths=(8, 8), norm="layer"), data, seed=0)
    with pytest.raises(ValueError, match="quotients out"):
        normalised.rescale_layers([2.0, 2.0])


def test_rescaling_rejects_bad_factors(small_model):
    with pytest.raises(ValueError, match="one factor per hidden layer"):
        small_model.rescale_layers([2.0])
    with pytest.raises(ValueError, match="strictly positive"):
        small_model.rescale_layers([1.0, -1.0, 1.0])


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def test_linear_network_still_yields_features():
    """The regression that motivated abandoning ReLU forward hooks.

    A hook-based extractor returns an empty dict for this architecture, so the
    A1 linear control would appear to run while measuring nothing.
    """
    data = DataConfig(classes=(2, 6))
    model = init_model(ModelConfig(widths=(16, 12), activation="identity"), data, seed=0)
    features, logits = extract_features(
        model, torch.rand((8, 1, 28, 28)), torch.device("cpu")
    )
    assert list(features) == ["relu1", "relu2"]
    assert features["relu1"].shape == (8, 16)
    assert logits.shape == (8, 2)


def test_features_are_float64(small_model, images):
    features, _ = extract_features(small_model, images, torch.device("cpu"))
    assert all(arr.dtype == np.float64 for arr in features.values())


def test_layer_sort_key_orders_double_digits():
    names = [f"relu{i}" for i in range(1, 13)]
    assert sorted(reversed(names), key=_layer_sort_key) == names


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "epsilon"),
    [("fgsm", 0.03), ("pgd_linf", 0.03), ("pgd_l2", 1.0), ("uniform", 0.03),
     ("gaussian", 0.02)],
)
def test_attacks_stay_in_pixel_range(small_model, images, labels, kind, epsilon):
    adv = apply_attack(
        AttackConfig(kind=kind, epsilon=epsilon, steps=5),
        small_model, images, labels, torch.device("cpu"),
    )
    assert adv.min() >= 0.0 and adv.max() <= 1.0
    assert adv.shape == images.shape


@pytest.mark.parametrize("kind", ["fgsm", "pgd_linf"])
def test_linf_attacks_respect_their_budget(small_model, images, labels, kind):
    epsilon = 0.03
    adv = apply_attack(
        AttackConfig(kind=kind, epsilon=epsilon, steps=5),
        small_model, images, labels, torch.device("cpu"),
    )
    assert perturbation_norms(images, adv)["linf"].max() <= epsilon + 1e-6


def test_pgd_l2_respects_its_budget(small_model, images, labels):
    epsilon = 0.9
    adv = apply_attack(
        AttackConfig(kind="pgd_l2", epsilon=epsilon, steps=5),
        small_model, images, labels, torch.device("cpu"),
    )
    assert perturbation_norms(images, adv)["l2"].max() <= epsilon + 1e-5


def test_zero_epsilon_is_a_no_op(small_model, images, labels):
    for kind in ("fgsm", "pgd_linf", "pgd_l2", "gaussian"):
        adv = apply_attack(
            AttackConfig(kind=kind, epsilon=0.0),
            small_model, images, labels, torch.device("cpu"),
        )
        assert torch.equal(adv, images), kind


def test_attacks_are_deterministic_given_a_seed(small_model, images, labels):
    config = AttackConfig(kind="pgd_linf", epsilon=0.03, steps=5, seed=42)
    first = apply_attack(config, small_model, images, labels, torch.device("cpu"))
    second = apply_attack(config, small_model, images, labels, torch.device("cpu"))
    assert torch.equal(first, second)


# ---------------------------------------------------------------------------
# The B2 control
# ---------------------------------------------------------------------------


def test_matching_norm_follows_attack_geometry():
    assert matching_norm_for("fgsm") == "linf"
    assert matching_norm_for("pgd_linf") == "linf"
    assert matching_norm_for("pgd_l2") == "l2"


def test_matched_noise_matches_the_requested_norm(small_model, images, labels):
    adv = apply_attack(
        AttackConfig(kind="fgsm", epsilon=0.03),
        small_model, images, labels, torch.device("cpu"),
    )
    noise = matched_noise(images, adv, norm="linf")
    a = perturbation_norms(images, adv)
    b = perturbation_norms(images, noise)
    assert torch.allclose(a["linf"], b["linf"], atol=1e-6)


def test_matched_noise_is_not_the_attack(small_model, images, labels):
    """Same norm, different direction — otherwise it is not a control."""
    adv = apply_attack(
        AttackConfig(kind="fgsm", epsilon=0.03),
        small_model, images, labels, torch.device("cpu"),
    )
    noise = matched_noise(images, adv, norm="linf")
    assert not torch.allclose(adv, noise)


def test_perturbation_arms_include_the_control(small_model, images, labels):
    arms = build_perturbation_arms(
        small_model, images, labels,
        AttackConfig(kind="fgsm", epsilon=0.03), torch.device("cpu"),
    )
    assert set(arms) == {"clean", "attack", "matched_noise"}
    assert torch.equal(arms["clean"], images)
    # For an ℓ∞ attack, auto-matching should align ℓ₂ closely too, since both
    # perturbations sit on a corner of the same ball.
    a = perturbation_norms(images, arms["attack"])["l2"]
    b = perturbation_norms(images, arms["matched_noise"])["l2"]
    assert torch.allclose(a, b, rtol=0.05)
