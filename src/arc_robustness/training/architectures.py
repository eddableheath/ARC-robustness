"""
Configurable architectures with explicit feature access.

Two departures from the original ``model.py``:

**Normalisation lives inside the model.** The network consumes raw ``[0, 1]``
images and normalises as its first operation. Attacks therefore differentiate
through the normalisation and can clamp to the valid pixel range without any
of the ``sign(∂L/∂x_norm) == sign(∂L/∂x_raw)`` reasoning the old FGSM
implementation relied on. That identity happens to hold for a positive scalar
std, but it does not hold for per-channel stds (CIFAR-10), so relying on it
would have broken silently when the dataset axis of C2 was added.

**Features are returned, not hooked.** The original extractor registered
forward hooks on every ``nn.ReLU`` module. The linear-net control of A1 has no
``nn.ReLU`` modules at all, so hooks would have yielded an empty feature dict
and the control would have appeared to run successfully while measuring
nothing. :meth:`MLP.forward_features` returns the post-activation
representations explicitly, so a missing layer is a loud error.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from arc_robustness.config import DataConfig, ModelConfig
from arc_robustness.data import normalisation_stats


class Normalise(nn.Module):
    """Channel-wise input normalisation as a layer.

    Registered as buffers so they move with ``.to(device)`` and are saved in
    the state dict — a checkpoint therefore carries its own preprocessing and
    cannot be silently paired with the wrong statistics.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        return (x - self.mean) / self.std


def _make_norm(kind: str, width: int) -> nn.Module | None:
    if kind == "none":
        return None
    if kind == "batch":
        return nn.BatchNorm1d(width)
    if kind == "layer":
        return nn.LayerNorm(width)
    raise ValueError(f"unknown norm {kind!r}")


def _make_activation(kind: str) -> nn.Module:
    if kind == "relu":
        return nn.ReLU()
    if kind == "identity":
        return nn.Identity()
    raise ValueError(f"unknown activation {kind!r}")


class MLP(nn.Module):
    """Fully-connected network with per-layer feature extraction.

    The analysed feature spaces are the *post-activation* representations,
    named ``relu1 … reluL``. The names are kept even when
    ``activation="identity"`` so that result files from the control arm line up
    column-for-column with the main arm; the label refers to the position in
    the network, not to the function applied.

    When ``norm`` is set, the feature is taken *after* normalisation and
    activation. That ordering is what makes T1's prediction testable: the
    normalisation is what quotients out the rescaling symmetry, so it must be
    inside the measured representation.
    """

    def __init__(self, model_cfg: ModelConfig, data_cfg: DataConfig):
        super().__init__()
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg

        mean, std = normalisation_stats(data_cfg.dataset)
        self.normalise = Normalise(mean, std, enabled=data_cfg.normalise)
        self.flatten = nn.Flatten()

        dims = [data_cfg.input_dim, *model_cfg.widths]
        self.linears = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1], bias=model_cfg.bias)
            for i in range(len(model_cfg.widths))
        )
        self.norms = nn.ModuleList(
            _make_norm(model_cfg.norm, w) or nn.Identity() for w in model_cfg.widths
        )
        self.activations = nn.ModuleList(
            _make_activation(model_cfg.activation) for _ in model_cfg.widths
        )
        self.head = nn.Linear(
            model_cfg.widths[-1], data_cfg.n_classes, bias=model_cfg.bias
        )

    @property
    def layer_names(self) -> tuple[str, ...]:
        return self.model_cfg.layer_names

    def forward_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor]]:
        """Return ``(logits, {layer_name: post-activation features})``."""
        h = self.flatten(self.normalise(x))
        feats: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name, linear, norm, act in zip(
            self.layer_names, self.linears, self.norms, self.activations
        ):
            h = act(norm(linear(h)))
            feats[name] = h
        return self.head(h), feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)[0]

    # -- the reparameterisation group of T1 --------------------------------

    @torch.no_grad()
    def rescale_layers(self, factors: list[float] | tuple[float, ...]) -> None:
        """Apply the function-preserving rescaling ``c = (c_1, …, c_L)`` in place.

        For a ReLU network, positive homogeneity ``ReLU(cz) = c·ReLU(z)`` means
        that scaling layer ``ℓ``'s weights and bias by ``c_ℓ`` and dividing the
        *next* map's weights by ``c_ℓ`` leaves the input-output map exactly
        unchanged while multiplying every layer-``ℓ`` activation by ``c_ℓ``.

        Concretely, with ``h_ℓ = ReLU(W_ℓ h_{ℓ-1} + b_ℓ)``, the update
        ``W_ℓ → c_ℓ W_ℓ / c_{ℓ-1}``, ``b_ℓ → c_ℓ b_ℓ`` gives
        ``h'_ℓ = ReLU((c_ℓ W_ℓ / c_{ℓ-1})(c_{ℓ-1} h_{ℓ-1}) + c_ℓ b_ℓ) = c_ℓ h_ℓ``,
        and dividing the head by ``c_L`` restores the logits. Each layer is
        scaled by its *own* ``c_ℓ``, not by a running product — the compensating
        division on the incoming weight cancels the previous layer's factor.

        This is the constructive engine of Theorem T1.2 and the basis of the
        invariance test suite: an estimator claiming invariance must return
        bit-comparable values before and after this call, while the raw
        estimators are expected to move — including across zero.

        Only valid for ``activation="relu"`` and ``norm="none"``. With
        normalisation the activations are rescaled away, so the symmetry is not
        present in the first place; with an identity activation the map is
        affine and the bias handling differs. Both raise.
        """
        if self.model_cfg.activation != "relu":
            raise ValueError(
                "the rescaling symmetry relies on ReLU positive homogeneity; "
                f"activation is {self.model_cfg.activation!r}"
            )
        if self.model_cfg.norm != "none":
            raise ValueError(
                "normalisation quotients out the rescaling symmetry, so there "
                "is nothing to apply; this is precisely T1's prediction for C1"
            )
        if len(factors) != len(self.linears):
            raise ValueError(
                f"need one factor per hidden layer ({len(self.linears)}), "
                f"got {len(factors)}"
            )
        if any(c <= 0 for c in factors):
            raise ValueError("rescaling factors must be strictly positive")

        for i, c in enumerate(factors):
            self.linears[i].weight.mul_(c)
            if self.linears[i].bias is not None:
                self.linears[i].bias.mul_(c)
            # Compensate on the way in to the next map. Scalar multiplications
            # commute, so it is safe that this weight is also multiplied by
            # c_{i+1} on the next iteration.
            next_map = self.head if i == len(factors) - 1 else self.linears[i + 1]
            next_map.weight.div_(c)


def build_model(model_cfg: ModelConfig, data_cfg: DataConfig) -> MLP:
    """Construct the architecture named by *model_cfg*."""
    if model_cfg.arch == "mlp":
        return MLP(model_cfg, data_cfg)
    raise NotImplementedError(
        f"arch {model_cfg.arch!r} is planned for Tier C but not yet implemented"
    )


def init_model(model_cfg: ModelConfig, data_cfg: DataConfig, seed: int) -> MLP:
    """Construct with a seeded initialisation.

    Seeding here rather than globally means the random-init control (A1) and
    the trained model with the same seed start from *identical* weights, so the
    comparison isolates the effect of training rather than of initialisation.
    """
    torch.manual_seed(seed)
    return build_model(model_cfg, data_cfg)
