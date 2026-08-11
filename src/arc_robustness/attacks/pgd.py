"""
Projected Gradient Descent (Madry et al., 2018), ℓ∞ and ℓ₂ variants.

Required by B2: FGSM is a single-step attack and a weak one, so a geometric
change observed only under FGSM is not evidence about adversarial examples in
general. PGD is the standard strong first-order baseline, and the ℓ∞/ℓ₂ pair
matters because the two norms displace features very differently — ℓ∞ spreads a
small change over every pixel, while ℓ₂ concentrates it, and the k-NN graph
stability argument of T4 is stated in terms of feature displacement ``δ``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _default_step(epsilon: float, steps: int) -> float:
    """Standard heuristic: 2.5·ε/steps, giving enough travel to reach the boundary."""
    return 2.5 * epsilon / max(steps, 1)


def pgd_linf(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    device: torch.device,
    steps: int = 40,
    step_size: float | None = None,
    random_start: bool = True,
    seed: int = 0,
    batch_size: int = 256,
) -> torch.Tensor:
    """ℓ∞-bounded PGD. Returns adversarial images in ``[0, 1]``."""
    if epsilon == 0.0:
        return images.clone()
    alpha = _default_step(epsilon, steps) if step_size is None else step_size

    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    out = torch.empty_like(images)

    for start in range(0, len(images), batch_size):
        x0 = images[start : start + batch_size].to(device)
        y = labels[start : start + batch_size].to(device)

        if random_start:
            delta = (
                torch.empty(x0.shape)
                .uniform_(-epsilon, epsilon, generator=generator)
                .to(device)
            )
            x = (x0 + delta).clamp(0.0, 1.0)
        else:
            x = x0.clone()

        for _ in range(steps):
            x = x.detach().requires_grad_(True)
            loss = F.cross_entropy(model(x), y)
            (grad,) = torch.autograd.grad(loss, x)
            with torch.no_grad():
                x = x + alpha * grad.sign()
                # Project onto the ε-ball around x0, then onto the pixel box.
                x = x0 + (x - x0).clamp(-epsilon, epsilon)
                x = x.clamp(0.0, 1.0)

        out[start : start + batch_size] = x.detach().cpu()

    return out


def pgd_l2(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    device: torch.device,
    steps: int = 40,
    step_size: float | None = None,
    random_start: bool = True,
    seed: int = 0,
    batch_size: int = 256,
) -> torch.Tensor:
    """ℓ₂-bounded PGD. ``epsilon`` is the ℓ₂ radius over the whole image."""
    if epsilon == 0.0:
        return images.clone()
    alpha = _default_step(epsilon, steps) if step_size is None else step_size

    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    out = torch.empty_like(images)

    for start in range(0, len(images), batch_size):
        x0 = images[start : start + batch_size].to(device)
        y = labels[start : start + batch_size].to(device)
        n = x0.shape[0]

        if random_start:
            delta = torch.randn(x0.shape, generator=generator).to(device)
            delta = _renorm_l2(delta, epsilon * torch.rand(n, device=device))
            x = (x0 + delta).clamp(0.0, 1.0)
        else:
            x = x0.clone()

        for _ in range(steps):
            x = x.detach().requires_grad_(True)
            loss = F.cross_entropy(model(x), y)
            (grad,) = torch.autograd.grad(loss, x)
            with torch.no_grad():
                # Normalised gradient step: raw ℓ₂ gradients vary in magnitude by
                # orders of magnitude across examples, so an unnormalised step
                # makes the effective step size example-dependent.
                flat = grad.reshape(n, -1)
                norm = flat.norm(dim=1).clamp(min=1e-12).view(-1, *([1] * (x.dim() - 1)))
                x = x + alpha * grad / norm
                x = x0 + _renorm_l2(x - x0, epsilon)
                x = x.clamp(0.0, 1.0)

        out[start : start + batch_size] = x.detach().cpu()

    return out


def _renorm_l2(delta: torch.Tensor, radius) -> torch.Tensor:
    """Project each example's perturbation onto the ℓ₂ ball of given radius.

    *radius* may be a scalar or a per-example tensor. Perturbations already
    inside the ball are left alone rather than scaled up to the boundary.
    """
    n = delta.shape[0]
    flat = delta.reshape(n, -1)
    norm = flat.norm(dim=1).clamp(min=1e-12)
    if not torch.is_tensor(radius):
        radius = torch.full_like(norm, float(radius))
    factor = (radius / norm).clamp(max=1.0)
    return (flat * factor.view(-1, 1)).reshape(delta.shape)
