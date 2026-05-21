"""
Extract post-ReLU activations from each hidden layer of the trained base model at
inference time, for the purpose of manifold visualisation.

Outputs are written to FEATURES_DIR as NumPy .npy files — one per layer plus a
labels file — so they can be passed directly to the visualisation utilities.

Usage:
    python extract_features.py            # test split (default)
    python extract_features.py --train    # training split
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

from arc_robustness.training.model import (
    CLASSES,
    DATA_DIR,
    FEATURES_DIR,
    WEIGHTS_PATH,
    DigitClassifier,
    filter_to_classes,
    remap_labels,
)

BATCH_SIZE = 256
NORMALISE = transforms.Normalize((0.1307,), (0.3081,))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(device: torch.device) -> DigitClassifier:
    model = DigitClassifier(num_classes=len(CLASSES)).to(device)
    state = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_layer_features(
    model: DigitClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run inference and return post-ReLU activations for every hidden layer.

    Returns
    -------
    features : dict[layer_name, ndarray of shape (N, units)]
        Post-ReLU activations collected over the full dataset.
    labels : ndarray of shape (N,)
        Remapped ground-truth class indices (0, 1, 2).
    """
    activations: dict[str, list[torch.Tensor]] = {}
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):

            def _make_hook(layer_name: str):
                def _hook(_module, _input, output: torch.Tensor) -> None:
                    activations.setdefault(layer_name, []).append(output.detach().cpu())

                return _hook

            hooks.append(module.register_forward_hook(_make_hook(name)))

    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in loader:
            model(images.to(device))
            all_labels.append(remap_labels(labels, CLASSES))

    for hook in hooks:
        hook.remove()

    features = {
        name: torch.cat(tensors).numpy() for name, tensors in activations.items()
    }
    labels_arr = torch.cat(all_labels).numpy()
    return features, labels_arr


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract layer features from the base model."
    )
    parser.add_argument(
        "--train", action="store_true", help="Use the training split instead of test."
    )
    args = parser.parse_args()

    transform = transforms.Compose([transforms.ToTensor(), NORMALISE])
    split = "train" if args.train else "test"

    loader = DataLoader(
        filter_to_classes(
            torchvision.datasets.MNIST(
                DATA_DIR, train=args.train, download=False, transform=transform
            ),
            CLASSES,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(device)

    print(f"Extracting features from {split} split on {device}...")
    features, labels = extract_layer_features(model, loader, device)

    out_dir: Path = FEATURES_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "labels.npy", labels)
    for layer_name, activations in features.items():
        np.save(out_dir / f"{layer_name}.npy", activations)
        print(f"  {layer_name:8s}  shape: {activations.shape}")

    print(f"\nFeatures saved to {out_dir}")


if __name__ == "__main__":
    main()
