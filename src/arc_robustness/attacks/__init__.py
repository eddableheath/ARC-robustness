"""
Evasion attacks and their norm-matched random-noise controls.

All attacks operate in **raw pixel space** on images in ``[0, 1]`` and return
images in ``[0, 1]``. Input normalisation happens inside the model, so gradients
flow through it correctly for any per-channel statistics.

The noise generators are not an afterthought. B2 makes the point that without a
perturbation of *matched norm* but no adversarial direction, any observed
geometric change is attributable to perturbation rather than to adversariality.
:func:`build_perturbation_arms` returns the attack and its matched control
together so the control is hard to omit.
"""

from arc_robustness.attacks.fgsm import fgsm
from arc_robustness.attacks.noise import (
    gaussian_noise,
    matched_noise,
    perturbation_norms,
    uniform_noise,
)
from arc_robustness.attacks.pgd import pgd_l2, pgd_linf
from arc_robustness.attacks.runner import (
    apply_attack,
    build_perturbation_arms,
    minimum_epsilon,
)

__all__ = [
    "fgsm",
    "pgd_linf",
    "pgd_l2",
    "gaussian_noise",
    "uniform_noise",
    "matched_noise",
    "perturbation_norms",
    "apply_attack",
    "build_perturbation_arms",
    "minimum_epsilon",
]
