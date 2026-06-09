"""
Compute Ricci curvatures and community-strength metrics for each hidden layer
of the trained model on Fashion-MNIST (Pullover vs Shirt).

For each layer the script builds a symmetric k-NN graph on the raw (high-
dimensional) activations, then computes:

  Forman-Ricci, Augmented Forman-Ricci, Ollivier-Ricci per edge
  per-vertex mean curvature
  Modularity Q, Normalised Cut, Algebraic Connectivity, Curvature Gap

Across layers it also computes the Local Ricci Evolution Coefficient ρ(x)
(Pearson correlation between η and O per vertex, see arXiv:2509.22362 §3).

Results are saved to outputs/ricci_metrics.npz and are consumed by
scripts/visualise_metrics.py.

Usage:
    uv run python scripts/compute_metrics.py
    uv run python scripts/compute_metrics.py --knn-k 8 --skip-ollivier
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import ConcatDataset, DataLoader

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
from arc_robustness.analysis.graph_utils import build_knn_graph
from arc_robustness.analysis.ricci import (
    forman_ricci,
    augmented_forman_ricci,
    ollivier_ricci,
    vertex_curvature,
)
from arc_robustness.analysis.community import (
    modularity,
    normalised_cut,
    algebraic_connectivity,
    curvature_gap,
)
from scipy.linalg import orthogonal_procrustes

from arc_robustness.analysis.evolution import local_ricci_evolution
from arc_robustness.visualisation.visualise import visualise_manifold

warnings.filterwarnings("ignore", message="Changing the sparsity structure")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BATCH_SIZE = 256
DEFAULT_SAMPLES_PER_CLASS = 200
DEFAULT_KNN_K = 6
NORMALISE = transforms.Normalize((0.2860,), (0.3530,))
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def subsample_per_class(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    n_per_class: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    keep: list[int] = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        chosen = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        keep.extend(chosen.tolist())
    keep_arr = np.array(sorted(keep))
    return {name: acts[keep_arr] for name, acts in features.items()}, labels[keep_arr]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Ricci and community metrics for each hidden layer."
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=DEFAULT_KNN_K,
        metavar="K",
        help=f"k for the symmetric k-NN graph (default: {DEFAULT_KNN_K}).",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLASS,
        metavar="N",
        help=f"Points per class to use (default: {DEFAULT_SAMPLES_PER_CLASS}).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for subsampling."
    )
    parser.add_argument(
        "--skip-ollivier",
        action="store_true",
        help="Skip Ollivier-Ricci (slow) — Forman and AF are always computed.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load / extract features
    # ------------------------------------------------------------------
    feature_dir = FEATURES_DIR / "full"
    label_path = feature_dir / "labels.npy"

    if label_path.exists():
        print("Loading saved features from disk...")
        labels_full = np.load(label_path)
        layer_files = sorted(
            p for p in feature_dir.iterdir() if p.name != "labels.npy"
        )
        features_full = {p.stem: np.load(p) for p in layer_files}
    else:
        print("Saved features not found — extracting from model...")
        transform = transforms.Compose([transforms.ToTensor(), NORMALISE])
        dataset = ConcatDataset([
            filter_to_classes(
                torchvision.datasets.FashionMNIST(
                    DATA_DIR, train=True, download=True, transform=transform
                ),
                CLASSES,
            ),
            filter_to_classes(
                torchvision.datasets.FashionMNIST(
                    DATA_DIR, train=False, download=True, transform=transform
                ),
                CLASSES,
            ),
        ])
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = load_model(device)
        features_full, labels_full = extract_layer_features(model, loader, device)

    rng = np.random.default_rng(args.seed)
    features, labels = subsample_per_class(
        features_full, labels_full, args.samples_per_class, rng
    )
    layer_names = list(features.keys())
    N = len(labels)
    L = len(layer_names)
    print(
        f"\n{N} points ({args.samples_per_class}/class), "
        f"{L} layers, k={args.knn_k}\n"
    )

    # ------------------------------------------------------------------
    # Per-layer graph construction and metric computation
    # ------------------------------------------------------------------
    adjs: dict[str, object] = {}
    forman: dict[str, dict] = {}
    af: dict[str, dict] = {}
    ollivier: dict[str, dict] = {}

    vert_forman: dict[str, np.ndarray] = {}
    vert_af: dict[str, np.ndarray] = {}
    vert_ollivier: dict[str, np.ndarray] = {}

    Q_arr = np.zeros(L)
    ncut_arr = np.zeros(L)
    fiedler_arr = np.zeros(L)
    gap_forman_arr = np.full(L, np.nan)
    gap_ollivier_arr = np.full(L, np.nan)

    for i, name in enumerate(tqdm(layer_names, desc="Layers")):
        pts = features[name]
        adj = build_knn_graph(pts, args.knn_k)
        adjs[name] = adj

        # Curvatures
        fr = forman_ricci(adj)
        afr = augmented_forman_ricci(adj)
        forman[name] = fr
        af[name] = afr

        vert_forman[name] = vertex_curvature(fr, N)
        vert_af[name] = vertex_curvature(afr, N)

        if not args.skip_ollivier:
            orr = ollivier_ricci(adj, pts)
            ollivier[name] = orr
            vert_ollivier[name] = vertex_curvature(orr, N)
        else:
            ollivier[name] = {}
            vert_ollivier[name] = np.full(N, np.nan)

        # Community metrics
        Q_arr[i] = modularity(adj, labels)
        ncut_arr[i] = normalised_cut(adj, labels)
        fiedler_arr[i] = algebraic_connectivity(adj)
        gap_forman_arr[i] = curvature_gap(fr, labels)
        if not args.skip_ollivier:
            gap_ollivier_arr[i] = curvature_gap(ollivier[name], labels)

    # ------------------------------------------------------------------
    # Local Ricci Evolution Coefficient ρ(x)
    # ------------------------------------------------------------------
    if not args.skip_ollivier:
        print("\nComputing ρ(x)...")
        rho, r_layer = local_ricci_evolution(features, adjs, ollivier)
        frac_neg = (rho < 0).mean()
        print(
            f"  mean ρ = {rho.mean():.3f},  "
            f"median ρ = {np.median(rho):.3f},  "
            f"{frac_neg:.1%} of vertices have ρ < 0"
        )
    else:
        rho = np.full(N, np.nan)
        r_layer = np.full(L - 1, np.nan)
        print("\nρ(x) skipped (--skip-ollivier).")

    # ------------------------------------------------------------------
    # 3-D UMAP projections (Procrustes-aligned to final layer)
    # Saved alongside metrics so visualise_metrics.py can load them directly.
    # ------------------------------------------------------------------
    print("\nComputing 3-D UMAP projections...")
    raw_projs: dict[str, np.ndarray] = {}
    for name in tqdm(layer_names, desc="UMAP 3D"):
        raw_projs[name] = visualise_manifold(features[name], n_components=3)

    final_raw = raw_projs[layer_names[-1]]
    ref = final_raw - final_raw.mean(axis=0)
    aligned_projs: dict[str, np.ndarray] = {}
    for name in layer_names:
        p = raw_projs[name] - raw_projs[name].mean(axis=0)
        R, _ = orthogonal_procrustes(p, ref)
        aligned_projs[name] = p @ R

    umap_stack = np.stack([aligned_projs[n] for n in layer_names], axis=0)  # (L, N, 3)

    has_ollivier = not args.skip_ollivier
    mean_kappa = np.array([
        np.nanmean(vert_ollivier[n]) if has_ollivier else np.nanmean(vert_forman[n])
        for n in layer_names
    ])

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "ricci_metrics.npz"

    # Flatten per-layer vertex curvature arrays into 2D (L × N)
    vf = np.stack([vert_forman[n] for n in layer_names], axis=0)
    vaf = np.stack([vert_af[n] for n in layer_names], axis=0)
    vo = np.stack([vert_ollivier[n] for n in layer_names], axis=0)

    np.savez(
        out_path,
        layer_names=np.array(layer_names),
        labels=labels,
        knn_k=args.knn_k,
        # Community metrics (L,)
        modularity=Q_arr,
        normalised_cut=ncut_arr,
        algebraic_connectivity=fiedler_arr,
        curvature_gap_forman=gap_forman_arr,
        curvature_gap_ollivier=gap_ollivier_arr,
        # Per-layer per-vertex curvature (L, N)
        vertex_forman=vf,
        vertex_af=vaf,
        vertex_ollivier=vo,
        # Local Ricci evolution
        rho=rho,
        r_layer=r_layer,        # (L-1,) per-layer Pearson(O_ℓ, η_ℓ)
        # Visualisation data
        umap_projections=umap_stack,  # (L, N, 3) Procrustes-aligned
        mean_kappa=mean_kappa,        # (L,) mean curvature per layer
        used_ollivier=np.bool_(has_ollivier),
    )
    print(f"\nMetrics saved to {out_path}")

    # Quick summary
    print("\n--- Community metrics by layer ---")
    header = f"{'Layer':<12}  {'Q':>7}  {'NCut':>7}  {'Fiedler':>9}  {'ΔO(F)':>8}"
    print(header)
    print("-" * len(header))
    for i, name in enumerate(layer_names):
        print(
            f"{name:<12}  {Q_arr[i]:>7.4f}  {ncut_arr[i]:>7.4f}  "
            f"{fiedler_arr[i]:>9.4f}  {gap_forman_arr[i]:>8.4f}"
        )


if __name__ == "__main__":
    main()
