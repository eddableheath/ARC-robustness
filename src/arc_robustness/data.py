"""
Dataset construction and subsampling.

Replaces four near-identical copies of ``subsample_per_class`` that had drifted
apart across the scripts (two took numpy arrays, two took torch tensors, and
they disagreed about whether to sort the kept indices).  Sorting matters: the
k-NN graph is built on row order, so an unsorted subsample silently permutes
the vertex indexing between the features and the labels.

All loaders return raw images in ``[0, 1]``.  Normalisation is applied inside
the model wrapper rather than the dataset, so that attacks can operate in pixel
space while gradients flow through the normalisation (see ``attacks``).
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

from arc_robustness.config import DATASET_SPECS, DataConfig

_TORCHVISION_DATASETS = {
    "fashion_mnist": torchvision.datasets.FashionMNIST,
    "mnist": torchvision.datasets.MNIST,
    "cifar10": torchvision.datasets.CIFAR10,
}


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalisation_stats(dataset: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(mean, std)`` as tensors broadcastable over ``(B, C, H, W)``."""
    spec = DATASET_SPECS[dataset]
    mean = torch.tensor(spec["mean"], dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.tensor(spec["std"], dtype=torch.float32).view(1, -1, 1, 1)
    return mean, std


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_raw_split(
    cfg: DataConfig, split: str, data_dir
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one split as ``(images in [0,1], original integer labels)``."""
    ctor = _TORCHVISION_DATASETS[cfg.dataset]
    dataset = ctor(
        str(data_dir),
        train=(split == "train"),
        download=True,
        transform=transforms.ToTensor(),
    )

    targets = torch.as_tensor(
        dataset.targets if not isinstance(dataset.targets, list) else dataset.targets
    )
    keep_mask = torch.isin(targets, torch.tensor(cfg.classes))
    keep_idx = torch.where(keep_mask)[0]

    # Index the underlying tensor directly rather than iterating the Dataset;
    # for 12k images the per-item transform path costs seconds for no reason.
    loader = DataLoader(
        torch.utils.data.Subset(dataset, keep_idx.tolist()),
        batch_size=1024,
        shuffle=False,
    )
    images = torch.cat([batch for batch, _ in loader])
    labels = targets[keep_idx]
    return images, labels


def _synthesise_random_data(
    cfg: DataConfig, n_samples: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gaussian-noise inputs with uniformly random labels (A1 control).

    Clamped to ``[0, 1]`` so the pixel-space attack budget means the same thing
    it does for real images.
    """
    shape = (n_samples, *cfg.input_shape)
    images = torch.rand(shape, generator=generator)
    labels = torch.tensor(cfg.classes)[
        torch.randint(len(cfg.classes), (n_samples,), generator=generator)
    ]
    return images, labels


def load_dataset(
    cfg: DataConfig,
    split: str | None = None,
    data_dir=None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(images, labels)`` for *cfg*.

    Images are ``(N, C, H, W)`` floats in ``[0, 1]``. Labels are remapped to
    contiguous indices ``0..n_classes-1``.

    ``label_mode`` is applied here so every caller — training, feature
    extraction, attack generation — sees a consistent view of the control
    conditions.  A shuffled-label control must permute labels *identically*
    at train and analysis time, which is why the permutation is derived from
    ``seed`` rather than drawn fresh.
    """
    from arc_robustness.config import DATA_DIR

    data_dir = DATA_DIR if data_dir is None else data_dir
    split = cfg.split if split is None else split
    generator = torch.Generator().manual_seed(seed)

    if cfg.label_mode == "random_data":
        # Size the synthetic set to match the real one so training dynamics
        # are comparable to the true-label arm.
        n_real = len(_load_raw_split(cfg, "train" if split != "test" else "test", data_dir)[1])
        images, labels_orig = _synthesise_random_data(cfg, n_real, generator)
    elif split == "both":
        train_images, train_labels = _load_raw_split(cfg, "train", data_dir)
        test_images, test_labels = _load_raw_split(cfg, "test", data_dir)
        images = torch.cat([train_images, test_images])
        labels_orig = torch.cat([train_labels, test_labels])
    else:
        images, labels_orig = _load_raw_split(cfg, split, data_dir)

    labels = remap_labels(labels_orig, cfg.classes)

    if cfg.label_mode == "shuffled":
        perm = torch.randperm(len(labels), generator=generator)
        labels = labels[perm]

    return images, labels


def remap_labels(labels: torch.Tensor, classes: tuple[int, ...]) -> torch.Tensor:
    """Map original class values onto contiguous 0-based indices."""
    lookup = torch.full((int(labels.max().item()) + 1,), -1, dtype=torch.long)
    for i, c in enumerate(classes):
        lookup[c] = i
    out = lookup[labels.long()]
    if (out < 0).any():
        raise ValueError("labels contain classes outside the configured set")
    return out


def make_loader(
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
) -> DataLoader:
    """DataLoader over in-memory tensors with a seeded shuffle."""
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        TensorDataset(images, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------


def balanced_subsample_indices(
    labels: np.ndarray | torch.Tensor,
    n_per_class: int,
    seed: int,
) -> np.ndarray:
    """Indices of a class-balanced subsample, **sorted ascending**.

    Sorting keeps vertex ``i`` of the graph aligned with row ``i`` of every
    feature array and with ``labels[i]``. The original scripts disagreed on
    this; an unsorted subsample scrambles the correspondence between the
    curvature arrays and the class partition, which silently corrupts every
    community metric.

    Raises if any class has fewer than ``n_per_class`` members, rather than
    quietly returning an unbalanced sample — an unbalanced graph makes
    modularity incomparable across cells.
    """
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    rng = np.random.default_rng(seed)

    keep: list[int] = []
    for cls in np.unique(labels_np):
        idx = np.where(labels_np == cls)[0]
        if len(idx) < n_per_class:
            raise ValueError(
                f"class {cls} has only {len(idx)} samples, need {n_per_class}"
            )
        keep.extend(rng.choice(idx, size=n_per_class, replace=False).tolist())

    return np.array(sorted(keep), dtype=np.int64)


def subsample_features(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    n_per_class: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Apply a balanced subsample to every layer's features at once.

    Returns ``(features, labels, indices)``. The indices are returned so that
    downstream analysis can trace a vertex back to its dataset row — needed by
    the detection probe, which must pair each adversarial point with its clean
    counterpart.
    """
    idx = balanced_subsample_indices(labels, n_per_class, seed)
    return {name: arr[idx] for name, arr in features.items()}, labels[idx], idx
