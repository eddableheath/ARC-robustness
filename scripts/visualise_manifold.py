"""
For the torchvision MNIST dataset (classes 0, 1, 2), run inference with the trained
model, extract post-ReLU activations at each hidden layer, save them as .npy files,
and produce an animated GIF showing the Isomap 2-D manifold projection evolving
layer by layer through the network.
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from arc_robustness.training.model import (
    CLASSES,
    DATA_DIR,
    FEATURES_DIR,
    filter_to_classes,
)
from arc_robustness.visualisation.extract_features import (
    extract_layer_features,
    load_model,
)
from arc_robustness.visualisation.visualise import visualise_manifold

matplotlib.use("Agg")  # headless backend — must be set before importing pyplot

# Isomap's internal k-NN graph construction triggers this scipy warning; it is
# an implementation detail of sklearn and cannot be avoided from user code.
warnings.filterwarnings("ignore", message="Changing the sparsity structure")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BATCH_SIZE = 256
DEFAULT_SAMPLES_PER_CLASS = 100
NORMALISE = transforms.Normalize((0.1307,), (0.3081,))
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
CMAP = matplotlib.colormaps.get_cmap("tab10").resampled(len(CLASSES))


def subsample_per_class(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    n_per_class: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return at most *n_per_class* randomly chosen indices for each class."""
    keep: list[int] = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        chosen = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        keep.extend(chosen.tolist())
    keep_arr = np.array(sorted(keep))
    sub_features = {name: acts[keep_arr] for name, acts in features.items()}
    return sub_features, labels[keep_arr]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise layer manifolds for MNIST classes 0-2."
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLASS,
        metavar="N",
        help=f"Samples per class used for Isomap / animation (default: {DEFAULT_SAMPLES_PER_CLASS}).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for subsampling."
    )
    args = parser.parse_args()

    transform = transforms.Compose([transforms.ToTensor(), NORMALISE])

    # Combine both splits, filtered to the three target classes
    full_dataset = ConcatDataset(
        [
            filter_to_classes(
                torchvision.datasets.MNIST(
                    DATA_DIR, train=True, download=True, transform=transform
                ),
                CLASSES,
            ),
            filter_to_classes(
                torchvision.datasets.MNIST(
                    DATA_DIR, train=False, download=False, transform=transform
                ),
                CLASSES,
            ),
        ]
    )
    loader = DataLoader(
        full_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(
        f"Running inference on {device}  ",
        f"({len(full_dataset):,} samples, classes {CLASSES})",
    )

    model = load_model(device)
    features, labels = extract_layer_features(model, loader, device)

    # ------------------------------------------------------------------
    # Save raw features
    # ------------------------------------------------------------------
    out_dir = FEATURES_DIR / "full"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "labels.npy", labels)
    for layer_name, activations in features.items():
        np.save(out_dir / f"{layer_name}.npy", activations)
        print(f"  saved {layer_name}: {activations.shape}")
    print(f"Features saved to {out_dir}\n")

    # ------------------------------------------------------------------
    # Subsample for Isomap (keeps full features on disk)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    vis_features, vis_labels = subsample_per_class(
        features, labels, args.samples_per_class, rng
    )
    total_vis = len(vis_labels)
    print(
        f"Subsampled to {args.samples_per_class} per class "
        f"({total_vis} points total) for Isomap.\n"
    )

    # ------------------------------------------------------------------
    # Compute Isomap projections (one per layer)
    # ------------------------------------------------------------------
    layer_names = list(vis_features.keys())
    projections: dict[str, np.ndarray] = {}
    for layer_name in tqdm(layer_names, desc="Isomap projections"):
        projections[layer_name] = visualise_manifold(vis_features[layer_name])

    # ------------------------------------------------------------------
    # Build animation
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    sm = plt.cm.ScalarMappable(
        cmap=CMAP,
        norm=plt.Normalize(vmin=-0.5, vmax=len(CLASSES) - 0.5),
    )
    fig.colorbar(sm, cax=cax, ticks=range(len(CLASSES)), label="digit class")

    def draw_frame(i: int) -> None:
        ax.clear()
        layer_name = layer_names[i]
        proj = projections[layer_name]
        ax.scatter(
            proj[:, 0],
            proj[:, 1],
            c=vis_labels,
            cmap=CMAP,
            vmin=-0.5,
            vmax=len(CLASSES) - 0.5,
            s=3,
            lw=0,
            alpha=0.6,
        )
        ax.set_title(f"Layer {i + 1}/{len(layer_names)}: {layer_name}", fontsize=12)
        ax.axis("off")

    ani = animation.FuncAnimation(
        fig,
        draw_frame,
        frames=len(layer_names),
        interval=1500,
        repeat=True,
    )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    gif_path = OUTPUTS_DIR / "layer_manifolds.gif"
    ani.save(gif_path, writer="pillow", fps=1)
    print(f"\nAnimation saved to {gif_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
