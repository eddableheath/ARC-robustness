"""
Post-activation feature extraction.

Moved out of ``visualisation/`` — the layer features are the central object of
the whole analysis, not a plotting concern, and every pipeline stage imports
them.

The extractor uses :meth:`MLP.forward_features` rather than forward hooks. See
``training/architectures.py`` for why that distinction is not cosmetic.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from arc_robustness.training.architectures import MLP


@torch.no_grad()
def extract_features(
    model: MLP,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[OrderedDict[str, np.ndarray], np.ndarray]:
    """Return ``({layer_name: (N, D_ℓ) features}, (N, C) logits)``.

    Features come back as float64 numpy arrays. The upcast is deliberate: the
    Ollivier computation solves an exact LP per edge, and float32 round-off in
    the ground-cost matrix produces visible jitter in ``W₁`` for the near-ties
    that are common in a k-NN neighbourhood.
    """
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {}
    logit_chunks: list[np.ndarray] = []

    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        logits, feats = model.forward_features(batch)
        logit_chunks.append(logits.detach().cpu().numpy())
        for name, value in feats.items():
            chunks.setdefault(name, []).append(
                value.detach().cpu().numpy().astype(np.float64)
            )

    if not chunks:
        raise RuntimeError("model returned no features — check forward_features")

    features: OrderedDict[str, np.ndarray] = OrderedDict(
        (name, np.concatenate(parts, axis=0)) for name, parts in chunks.items()
    )
    return features, np.concatenate(logit_chunks, axis=0)


def save_features(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    out_dir: Path,
    logits: np.ndarray | None = None,
) -> None:
    """Write one ``.npy`` per layer plus labels (and optionally logits)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "labels.npy", labels)
    if logits is not None:
        np.save(out_dir / "logits.npy", logits)
    for name, arr in features.items():
        np.save(out_dir / f"{name}.npy", arr)


def load_features(
    feature_dir: Path, layer_names: list[str] | None = None
) -> tuple[OrderedDict[str, np.ndarray], np.ndarray]:
    """Read features written by :func:`save_features`.

    When *layer_names* is given, layers are returned in that order. Relying on
    directory listing order is a latent bug once a network has ten or more
    layers, because ``sorted()`` puts ``relu10`` before ``relu2`` — which would
    silently scramble the layer axis of every across-layer statistic, ρ included.
    """
    labels = np.load(feature_dir / "labels.npy")

    if layer_names is None:
        discovered = [
            p.stem
            for p in feature_dir.glob("*.npy")
            if p.stem not in {"labels", "logits"}
        ]
        layer_names = sorted(discovered, key=_layer_sort_key)

    features: OrderedDict[str, np.ndarray] = OrderedDict()
    for name in layer_names:
        path = feature_dir / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"missing layer file {path}")
        features[name] = np.load(path)
    return features, labels


def _layer_sort_key(name: str) -> tuple[str, int]:
    """Sort ``relu2`` before ``relu10`` by splitting off the trailing integer."""
    prefix = name.rstrip("0123456789")
    suffix = name[len(prefix) :]
    return (prefix, int(suffix) if suffix else -1)
