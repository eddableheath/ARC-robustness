"""
Generate FGSM adversarial examples for the trained Fashion-MNIST binary classifier
(Pullover vs Shirt).

For the primary ε (used for Ricci analysis) and a configurable set of comparison
ε values, saves per-epsilon directories:

  outputs/adversarial/eps_{eps:.3f}/
      clean_images.npy   (N, 1, 28, 28) clean test images in [0, 1]
      adv_images.npy     (N, 1, 28, 28) adversarial images in [0, 1]
      labels.npy         (N,)            remapped labels (0 = Pullover, 1 = Shirt)
      {layer_name}.npy   (N, D_ℓ)       per-layer activations for adversarial inputs

Also saves:
  outputs/adversarial/comparison_grid.png  visual grid: rows = examples, cols = ε
  outputs/adversarial/attack_summary.txt   accuracy, L∞ norms, success rates

Usage:
    uv run python scripts/generate_adversarial.py
    uv run python scripts/generate_adversarial.py --epsilon 0.1
    uv run python scripts/generate_adversarial.py --epsilon 0.05 --samples-per-class 300
    uv run python scripts/generate_adversarial.py --compare-epsilons 0.05 0.15 0.3
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from arc_robustness.training.model import (
    CLASS_NAMES,
    CLASSES,
    DATA_DIR,
    filter_to_classes,
    remap_labels,
)
from arc_robustness.visualisation.extract_features import (
    extract_layer_features,
    load_model,
)

matplotlib.use("Agg")

BATCH_SIZE = 256
DEFAULT_SAMPLES_PER_CLASS = 200
DEFAULT_EPSILON = 0.03
DEFAULT_COMPARE_EPSILONS = [0.05, 0.1, 0.2, 0.3]
NORMALISE_MEAN = 0.2860
NORMALISE_STD = 0.3530
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "adversarial"


# ---------------------------------------------------------------------------
# FGSM
# ---------------------------------------------------------------------------

def fgsm_attack(
    model: torch.nn.Module,
    images_raw: torch.Tensor,   # (B, 1, 28, 28) in [0, 1], no normalisation
    labels_model: torch.Tensor,  # (B,) remapped to 0 / 1
    epsilon: float,
    device: torch.device,
) -> torch.Tensor:
    """Single-step FGSM.  Returns adversarial images in [0, 1].

    ε is in raw [0, 1] pixel space.  sign(∂L/∂x_norm) = sign(∂L/∂x_raw)
    because σ > 0, so we can compute the gradient w.r.t. the normalised
    input and apply the step directly in pixel space.
    """
    if epsilon == 0.0:
        return images_raw.clone()

    x_raw = images_raw.to(device)
    x_norm = ((x_raw - NORMALISE_MEAN) / NORMALISE_STD).requires_grad_(True)

    model.zero_grad()
    loss = F.cross_entropy(model(x_norm), labels_model.to(device))
    loss.backward()

    with torch.no_grad():
        x_adv = x_raw + epsilon * x_norm.grad.sign()
        x_adv = x_adv.clamp(0.0, 1.0)

    return x_adv.cpu()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate_accuracy(
    model: torch.nn.Module,
    images_raw: torch.Tensor,
    labels_model: torch.Tensor,
    device: torch.device,
) -> float:
    """Accuracy of model on raw (un-normalised) images."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(labels_model), BATCH_SIZE):
            x = images_raw[start:start + BATCH_SIZE].to(device)
            y = labels_model[start:start + BATCH_SIZE].to(device)
            x_norm = (x - NORMALISE_MEAN) / NORMALISE_STD
            pred = model(x_norm).argmax(dim=1)
            correct += (pred == y).sum().item()
    return correct / len(labels_model)


def subsample_per_class(
    images: torch.Tensor,
    labels_orig: torch.Tensor,
    n_per_class: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    keep = []
    for cls in CLASSES:
        idx = (labels_orig == cls).nonzero(as_tuple=True)[0].numpy()
        chosen = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        keep.extend(chosen.tolist())
    keep = sorted(keep)
    return images[keep], labels_orig[keep]


def save_comparison_grid(
    clean: torch.Tensor,                           # (N, 1, 28, 28)
    adv_by_eps: list[tuple[float, torch.Tensor]],  # [(ε, (N,1,28,28)), ...]
    labels_orig: torch.Tensor,
    n_rows: int,
    out_path: Path,
) -> None:
    """Grid of example images: rows = samples, cols = clean + each ε."""
    rng = np.random.default_rng(0)
    idx: list[int] = []
    per_class = max(1, n_rows // len(CLASSES))
    for cls in CLASSES:
        pool = (labels_orig == cls).nonzero(as_tuple=True)[0].numpy()
        chosen = rng.choice(pool, size=min(per_class, len(pool)), replace=False)
        idx.extend(chosen.tolist())
    idx = sorted(idx[:n_rows])

    cols = [(0.0, clean)] + adv_by_eps
    n_cols = len(cols)
    n_rows_actual = len(idx)

    fig, axes = plt.subplots(
        n_rows_actual, n_cols,
        figsize=(n_cols * 1.5, n_rows_actual * 1.5),
    )
    if n_rows_actual == 1:
        axes = axes[np.newaxis, :]

    for col_i, (eps, imgs) in enumerate(cols):
        for row_i, sample_i in enumerate(idx):
            ax = axes[row_i, col_i]
            ax.imshow(imgs[sample_i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if row_i == 0:
                ax.set_title("clean" if eps == 0.0 else f"ε={eps:.2f}", fontsize=9)
            if col_i == 0:
                cls_idx = CLASSES.index(labels_orig[sample_i].item())
                ax.set_ylabel(
                    CLASS_NAMES[cls_idx], fontsize=8, rotation=0, labelpad=32, va="center",
                )

    fig.suptitle(
        f"FGSM adversarial examples — Fashion-MNIST: {CLASS_NAMES[0]} vs {CLASS_NAMES[1]}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison grid saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate FGSM adversarial examples for Fashion-MNIST binary classifier."
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        metavar="ε",
        help=(
            f"Primary L∞ budget in [0, 1] pixel space; "
            f"used for Ricci feature extraction (default: {DEFAULT_EPSILON})."
        ),
    )
    parser.add_argument(
        "--compare-epsilons",
        type=float,
        nargs="+",
        default=DEFAULT_COMPARE_EPSILONS,
        metavar="ε",
        help=(
            "Additional ε values for visual comparison and stats "
            f"(default: {DEFAULT_COMPARE_EPSILONS})."
        ),
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLASS,
        metavar="N",
        help=f"Test images per class (default: {DEFAULT_SAMPLES_PER_CLASS}).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load test set — raw [0, 1], no normalisation
    # ------------------------------------------------------------------
    test_ds = filter_to_classes(
        torchvision.datasets.FashionMNIST(
            DATA_DIR, train=False, download=True,
            transform=transforms.ToTensor(),
        ),
        CLASSES,
    )
    all_images = torch.stack([img for img, _ in test_ds])
    all_labels_orig = torch.tensor([lbl for _, lbl in test_ds])

    rng = np.random.default_rng(args.seed)
    images, labels_orig = subsample_per_class(
        all_images, all_labels_orig, args.samples_per_class, rng,
    )
    labels_model = remap_labels(labels_orig, CLASSES)  # 0 / 1 for cross-entropy
    N = len(labels_model)
    print(f"{N} images sampled ({args.samples_per_class}/class)")

    model = load_model(device)
    model.eval()

    acc_clean = evaluate_accuracy(model, images, labels_model, device)
    print(f"Clean accuracy: {acc_clean:.1%}")

    # ------------------------------------------------------------------
    # FGSM across all ε values
    # ------------------------------------------------------------------
    all_epsilons = sorted(set(args.compare_epsilons) | {args.epsilon})
    adv_by_eps: dict[float, torch.Tensor] = {}
    acc_by_eps: dict[float, float] = {}

    print()
    for eps in tqdm(all_epsilons, desc="FGSM ε values"):
        batches = []
        loader = DataLoader(
            TensorDataset(images, labels_model),
            batch_size=BATCH_SIZE, shuffle=False,
        )
        for x_batch, y_batch in loader:
            batches.append(fgsm_attack(model, x_batch, y_batch, eps, device))
        adv = torch.cat(batches, dim=0)
        adv_by_eps[eps] = adv
        acc_by_eps[eps] = evaluate_accuracy(model, adv, labels_model, device)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n--- FGSM attack summary ---")
    header = f"{'ε':>8}  {'accuracy':>10}  {'acc drop':>10}  {'mean L∞':>10}"
    print(header)
    print("-" * len(header))
    for eps in all_epsilons:
        l_inf = (adv_by_eps[eps] - images).abs().amax(dim=(1, 2, 3)).mean().item()
        print(
            f"{eps:>8.3f}  {acc_by_eps[eps]:>10.1%}  "
            f"{acc_clean - acc_by_eps[eps]:>10.1%}  {l_inf:>10.4f}"
        )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Visual comparison grid
    compare_eps_nonzero = [
        (eps, adv_by_eps[eps]) for eps in sorted(set(args.compare_epsilons)) if eps > 0
    ]
    save_comparison_grid(
        images, compare_eps_nonzero, labels_orig,
        n_rows=8, out_path=OUTPUTS_DIR / "comparison_grid.png",
    )

    # Attack summary text
    summary_path = OUTPUTS_DIR / "attack_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"FGSM — Fashion-MNIST: {CLASS_NAMES[0]} vs {CLASS_NAMES[1]}\n")
        f.write(f"N={N}, seed={args.seed}, samples_per_class={args.samples_per_class}\n\n")
        f.write(f"{'ε':>8}  {'accuracy':>10}  {'acc drop':>10}  {'mean L∞':>10}\n")
        for eps in all_epsilons:
            l_inf = (adv_by_eps[eps] - images).abs().amax(dim=(1, 2, 3)).mean().item()
            f.write(
                f"{eps:>8.3f}  {acc_by_eps[eps]:>10.1%}  "
                f"{acc_clean - acc_by_eps[eps]:>10.1%}  {l_inf:>10.4f}\n"
            )
    print(f"\nAttack summary saved to {summary_path}")

    # Per-ε: images, labels, and per-layer adversarial features
    normalise = transforms.Normalize((NORMALISE_MEAN,), (NORMALISE_STD,))
    for eps in all_epsilons:
        eps_dir = OUTPUTS_DIR / f"eps_{eps:.3f}"
        eps_dir.mkdir(parents=True, exist_ok=True)

        adv = adv_by_eps[eps]
        np.save(eps_dir / "clean_images.npy", images.numpy())
        np.save(eps_dir / "adv_images.npy", adv.numpy())
        np.save(eps_dir / "labels.npy", labels_model.numpy())

        # Feature extraction: pass normalised adversarial images + original labels
        # so extract_layer_features can remap labels internally
        adv_norm = torch.stack([normalise(img) for img in adv])
        feat_loader = DataLoader(
            TensorDataset(adv_norm, labels_orig),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        )
        print(f"Extracting features for ε={eps:.3f}...")
        features, _ = extract_layer_features(model, feat_loader, device)
        for layer_name, acts in features.items():
            np.save(eps_dir / f"{layer_name}.npy", acts)

        print(f"  → {eps_dir}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
