"""
Attack dispatch and per-example difficulty measurement.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from arc_robustness.config import AttackConfig
from arc_robustness.attacks.fgsm import fgsm
from arc_robustness.attacks.noise import gaussian_noise, matched_noise, uniform_noise
from arc_robustness.attacks.pgd import pgd_l2, pgd_linf


def apply_attack(
    config: AttackConfig,
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Produce perturbed images according to *config*. Returns ``[0, 1]`` images."""
    if config.kind == "none" or config.epsilon == 0.0:
        return images.clone()

    if config.kind == "fgsm":
        return fgsm(model, images, labels, config.epsilon, device)
    if config.kind == "pgd_linf":
        return pgd_linf(
            model, images, labels, config.epsilon, device,
            steps=config.steps, step_size=config.step_size,
            random_start=config.random_start, seed=config.seed,
        )
    if config.kind == "pgd_l2":
        return pgd_l2(
            model, images, labels, config.epsilon, device,
            steps=config.steps, step_size=config.step_size,
            random_start=config.random_start, seed=config.seed,
        )
    if config.kind == "gaussian":
        return gaussian_noise(images, config.epsilon, seed=config.seed)
    if config.kind == "uniform":
        return uniform_noise(images, config.epsilon, seed=config.seed, signed=True)

    raise ValueError(f"unhandled attack kind {config.kind!r}")


def matching_norm_for(kind: str) -> str:
    """Which norm the random control should match, given the attack's geometry.

    Matching the *wrong* norm reintroduces the confound the control exists to
    remove. An ℓ∞ attack moves every pixel by about ``±ε``; a dense isotropic
    direction rescaled to the same ℓ₂ norm puts far more mass on individual
    pixels, so it clips against ``[0, 1]`` much harder and ends up with a
    different realised norm *and* a different sparsity pattern. Measured on
    FGSM at ε=0.03: ℓ₂-matched Gaussian noise lands at ℓ₂ 0.67 against the
    attack's 0.75 (21% worst-case per-example gap) with ℓ∞ inflated to 0.09,
    whereas signed-corner noise matches ℓ₂ to 0.752 exactly and keeps ℓ∞ at
    0.03.
    """
    return "l2" if kind == "pgd_l2" else "linf"


def build_perturbation_arms(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    attack: AttackConfig,
    device: torch.device,
    match_norm: str = "auto",
) -> dict[str, torch.Tensor]:
    """Return ``{"clean", "attack", "matched_noise"}`` image tensors.

    The convenience that makes B2's control arm hard to forget: any analysis
    that asks for an attack gets its norm-matched random counterpart in the same
    call, computed from the attack's *realised* perturbation.

    ``match_norm="auto"`` picks the norm from the attack kind via
    :func:`matching_norm_for`, which is what you want unless you are
    deliberately probing the mismatch.
    """
    adversarial = apply_attack(attack, model, images, labels, device)
    norm = matching_norm_for(attack.kind) if match_norm == "auto" else match_norm
    return {
        "clean": images.clone(),
        "attack": adversarial,
        "matched_noise": matched_noise(
            images, adversarial, norm=norm, seed=attack.seed
        ),
    }


@torch.no_grad()
def _predictions(
    model: nn.Module, images: torch.Tensor, device: torch.device, batch_size: int = 512
) -> torch.Tensor:
    model.eval()
    out = []
    for start in range(0, len(images), batch_size):
        out.append(model(images[start : start + batch_size].to(device)).argmax(1).cpu())
    return torch.cat(out)


def minimum_epsilon(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    kind: str = "pgd_linf",
    grid: np.ndarray | None = None,
    steps: int = 20,
) -> np.ndarray:
    """Per-example ``ε_min``: the smallest grid budget at which the attack flips *x*.

    Sweeps *grid* in increasing order and records, for each example, the first
    ``ε`` at which the prediction becomes wrong. Costs exactly ``len(grid)``
    attack passes. Returns ``0.0`` for examples already misclassified when clean
    and ``inf`` for those never flipped within the grid.

    This is the target variable of B4 — the test of whether clean-network
    geometry predicts *which points are fragile*. Three caveats for the write-up:

    * The result is an **upper bound** on the true ``ε_min``, limited by both
      grid resolution and PGD's incompleteness as a search.
    * Because flipping is not guaranteed monotone in ``ε``, "first flip on the
      grid" is the honest statistic; a per-example bisection would additionally
      assume monotonicity, which PGD does not satisfy.
    * ``ε_min`` measured with a specific attack conflates the example's
      fragility with that attack's strength, so B4 should report it for at
      least ℓ∞ and ℓ₂.
    """
    if grid is None:
        grid = np.concatenate(
            [np.linspace(0.002, 0.03, 15), np.linspace(0.04, 0.3, 14)]
        )
    grid = np.sort(np.asarray(grid, dtype=np.float64))

    n = len(images)
    result = np.full(n, np.inf, dtype=np.float64)

    clean_wrong = (_predictions(model, images, device) != labels).numpy()
    result[clean_wrong] = 0.0
    pending = ~clean_wrong

    for epsilon in grid:
        if not pending.any():
            break
        flipped = _flips(model, images, labels, float(epsilon), device, kind, steps)
        newly = pending & flipped
        result[newly] = epsilon
        pending = pending & ~flipped

    return result


def _flips(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    device: torch.device,
    kind: str,
    steps: int,
) -> np.ndarray:
    cfg = AttackConfig(kind=kind, epsilon=epsilon, steps=steps)
    adv = apply_attack(cfg, model, images, labels, device)
    return (_predictions(model, adv, device) != labels).numpy()
