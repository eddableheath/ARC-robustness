"""Fast Gradient Sign Method (Goodfellow et al., 2015)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def fgsm(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Single-step ℓ∞ attack. Returns adversarial images in ``[0, 1]``.

    ``epsilon`` is in raw pixel units. The gradient is taken with respect to the
    raw input and the model normalises internally, so this is correct for
    per-channel normalisation statistics as well as scalar ones — unlike the
    original implementation, which exploited ``sign(∂L/∂x_norm) = sign(∂L/∂x_raw)``
    and would have broken silently on CIFAR-10.
    """
    if epsilon == 0.0:
        return images.clone()

    model.eval()
    out = torch.empty_like(images)

    for start in range(0, len(images), batch_size):
        x = images[start : start + batch_size].to(device).clone().requires_grad_(True)
        y = labels[start : start + batch_size].to(device)

        loss = F.cross_entropy(model(x), y)
        (grad,) = torch.autograd.grad(loss, x)

        with torch.no_grad():
            adv = (x + epsilon * grad.sign()).clamp(0.0, 1.0)
        out[start : start + batch_size] = adv.detach().cpu()

    return out
