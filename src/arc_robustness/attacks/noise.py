"""
Random-perturbation controls.

B2 treats these as non-negotiable rather than optional. Every geometric quantity
we measure — curvature, modularity, ρ — is a function of the feature point
cloud, and *any* input perturbation moves that cloud. So "the signature degrades
under attack" is only a claim about adversarial examples if a perturbation of the
same size but no adversarial direction leaves it intact. Without this arm the
result reduces to "perturbing the input perturbs the features", which needs no
Ricci curvature to predict.

:func:`matched_noise` is the arm that actually matters. Calibrating noise to the
nominal ``ε`` is not a fair control, because clamping to ``[0, 1]`` means an
attack's *realised* perturbation norm is usually below its budget — more so at
large ``ε``, and unevenly across examples, since pixels already near 0 or 1
cannot move. Matching per-example realised norms removes that confound.
"""

from __future__ import annotations

import torch


def uniform_noise(
    images: torch.Tensor, epsilon: float, seed: int = 0, signed: bool = False
) -> torch.Tensor:
    """Uniform noise in the ℓ∞ ball of radius *epsilon*.

    With ``signed=True`` every pixel moves by exactly ``±epsilon``, placing the
    perturbation on a random *corner* of the ball. That is the correct
    norm-matched comparison for FGSM, which also lands on a corner — uniform
    interior noise has a much smaller typical ℓ₂ norm and so understates the
    control.
    """
    generator = torch.Generator().manual_seed(seed)
    if signed:
        signs = torch.randint(0, 2, images.shape, generator=generator) * 2.0 - 1.0
        delta = epsilon * signs
    else:
        delta = torch.empty(images.shape).uniform_(-epsilon, epsilon, generator=generator)
    return (images + delta).clamp(0.0, 1.0)


def gaussian_noise(
    images: torch.Tensor, sigma: float, seed: int = 0
) -> torch.Tensor:
    """Isotropic Gaussian noise with standard deviation *sigma*."""
    generator = torch.Generator().manual_seed(seed)
    delta = torch.randn(images.shape, generator=generator) * sigma
    return (images + delta).clamp(0.0, 1.0)


def perturbation_norms(
    clean: torch.Tensor, perturbed: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Per-example realised perturbation norms.

    Reported alongside every attack so the ε-sweep can be plotted against what
    the perturbation actually was rather than what was requested.
    """
    delta = (perturbed - clean).reshape(len(clean), -1)
    return {
        "linf": delta.abs().amax(dim=1),
        "l2": delta.norm(dim=1),
        "l1": delta.abs().sum(dim=1),
        "l0": (delta.abs() > 1e-12).sum(dim=1).float(),
    }


def matched_noise(
    clean: torch.Tensor,
    adversarial: torch.Tensor,
    norm: str = "l2",
    seed: int = 0,
) -> torch.Tensor:
    """Random perturbation matching *adversarial*'s per-example realised norm.

    A random direction is drawn per example and rescaled so its norm equals the
    attack's realised norm in the chosen ``norm``, then clamped to the pixel box.

    Clamping can only shrink the norm, so the control is very slightly *weaker*
    than the attack it matches; :func:`perturbation_norms` on the result records
    the achieved norms so the residual gap can be reported rather than assumed
    away.
    """
    if norm not in {"l2", "linf"}:
        raise ValueError(f"unsupported norm {norm!r}")

    generator = torch.Generator().manual_seed(seed)
    target = perturbation_norms(clean, adversarial)[norm]
    n = len(clean)

    direction = torch.randn(clean.shape, generator=generator).reshape(n, -1)

    if norm == "l2":
        current = direction.norm(dim=1).clamp(min=1e-12)
        scaled = direction * (target / current).view(-1, 1)
    else:
        # For ℓ∞, a random corner of the ball is the right analogue: rescaling a
        # Gaussian to a target ℓ∞ norm would leave most coordinates far inside.
        signs = torch.randint(0, 2, (n, direction.shape[1]), generator=generator) * 2.0 - 1.0
        scaled = signs * target.view(-1, 1)

    delta = scaled.reshape(clean.shape)
    return (clean + delta).clamp(0.0, 1.0)
