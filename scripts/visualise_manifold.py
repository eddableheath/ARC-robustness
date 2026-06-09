"""
For the torchvision Fashion-MNIST dataset (Pullover vs Shirt), run inference with the
trained model, extract post-ReLU activations at each hidden layer, save them as .npy
files, and produce an animated GIF showing the UMAP 3-D manifold projection evolving
layer by layer.

Each frame shows per-class convex hulls (optional) and an r-filtered kNN graph.
All projections are Procrustes-aligned to the final layer to remove arbitrary
orientation changes between frames. The final frames show the linear decision boundary;
the last hold adds an edge-on view where the boundary collapses to a line.
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
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import ConvexHull, QhullError
from sklearn.neighbors import kneighbors_graph
from sklearn.svm import LinearSVC
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from arc_robustness.training.model import (
    CLASS_NAMES,
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

warnings.filterwarnings("ignore", message="Changing the sparsity structure")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BATCH_SIZE = 256
DEFAULT_SAMPLES_PER_CLASS = 200
DEFAULT_KNN_K = 6
DEFAULT_HOLD_FRAMES = 3
NORMALISE = transforms.Normalize((0.2860,), (0.3530,))
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

# Sample viridis at well-separated positions for maximum contrast with 2 classes.
_v = matplotlib.colormaps["viridis"]
CMAP = matplotlib.colors.ListedColormap([_v(0.1), _v(0.85)])
_NORM = matplotlib.colors.Normalize(vmin=-0.5, vmax=len(CLASSES) - 0.5)

# Sentinel for frame-sequence view mode
_NORMAL = 0
_EDGE_ON = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    return {name: acts[keep_arr] for name, acts in features.items()}, labels[keep_arr]


def procrustes_align(projections: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Rotate each layer's projection to best match the final layer.

    Uses orthogonal Procrustes (rotation + reflection only, no scale/translation)
    so that the only motion between frames reflects genuine topology change.
    """
    layer_names = list(projections.keys())
    final = projections[layer_names[-1]]
    ref = final - final.mean(axis=0)

    aligned: dict[str, np.ndarray] = {}
    for name in layer_names:
        proj = projections[name]
        p = proj - proj.mean(axis=0)
        R, _ = orthogonal_procrustes(p, ref)
        aligned[name] = p @ R
    return aligned


def build_class_edges(
    proj: np.ndarray,
    labels: np.ndarray,
    k: int,
    r_thresh: float | None,
) -> tuple[list, list]:
    """Return (segments, colors) for kNN edges within each class, optionally filtered
    to pairs whose Euclidean distance is at most r_thresh * overall_diameter."""
    segments: list = []
    colors: list = []
    abs_r: float | None = None
    if r_thresh is not None:
        diameter = float(np.linalg.norm(proj.max(axis=0) - proj.min(axis=0)))
        abs_r = r_thresh * diameter
    for cls in np.unique(labels):
        pts = proj[labels == cls]
        if len(pts) <= k:
            continue
        graph = kneighbors_graph(
            pts, n_neighbors=k, mode="distance", include_self=False
        )
        rows, cols = graph.nonzero()
        color = CMAP(_NORM(cls))
        for src, dst in zip(rows, cols):
            if src < dst:
                if abs_r is None or float(graph[src, dst]) <= abs_r:
                    segments.append([pts[src], pts[dst]])
                    colors.append(color)
    return segments, colors


def draw_class_hull(ax, pts: np.ndarray, color: tuple, alpha: float = 0.12) -> None:
    """Add the convex hull of pts as a transparent triangulated surface to ax."""
    try:
        hull = ConvexHull(pts)
    except QhullError:
        return
    poly = Poly3DCollection(pts[hull.simplices], linewidth=0.2)
    poly.set_facecolor((*color[:3], alpha))
    poly.set_edgecolor((*color[:3], 0.18))
    ax.add_collection3d(poly)


def draw_decision_boundary(
    ax,
    proj: np.ndarray,
    w: np.ndarray,
    b: float,
    margin: float = 0.2,
) -> None:
    """Decision plane as a patch parameterised in the plane's own coordinates.

    Works regardless of orientation — no z-clipping or masking so the boundary
    always renders in full.  The patch is sized to cover the full data extent
    projected onto the plane, plus *margin*.
    """
    w_norm = float(np.linalg.norm(w))
    if w_norm < 1e-10:
        return

    n = w / w_norm  # unit normal

    # Point on the plane nearest to the data centroid
    centroid = proj.mean(axis=0)
    p0 = centroid - (float(n @ centroid) + b / w_norm) * n

    # Two orthogonal unit vectors spanning the plane
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    # Patch radius: cover the full data footprint projected onto the plane
    d = proj - p0
    R = float(max(np.abs(d @ u).max(), np.abs(d @ v).max())) * (1.0 + margin)

    ss, tt = np.meshgrid(np.linspace(-R, R, 50), np.linspace(-R, R, 50))
    pts = p0 + ss[..., np.newaxis] * u + tt[..., np.newaxis] * v  # (50, 50, 3)
    xx, yy, zz = pts[..., 0], pts[..., 1], pts[..., 2]

    ax.plot_surface(xx, yy, zz, color="white", alpha=0.35, linewidth=0, antialiased=True)
    ax.plot_wireframe(xx, yy, zz, color="dimgray", linewidth=0.6, alpha=0.7, rstride=3, cstride=3)

    # Restore data-based axis limits so the patch doesn't push the view out
    pad_ax = 0.1 * (proj.max(axis=0) - proj.min(axis=0))
    ax.set_xlim(proj[:, 0].min() - pad_ax[0], proj[:, 0].max() + pad_ax[0])
    ax.set_ylim(proj[:, 1].min() - pad_ax[1], proj[:, 1].max() + pad_ax[1])
    ax.set_zlim(proj[:, 2].min() - pad_ax[2], proj[:, 2].max() + pad_ax[2])


def compute_edge_on_view(w: np.ndarray) -> tuple[float, float]:
    """Return (elev, azim) so the plane with normal w appears edge-on (as a line).

    At elev=0 the matplotlib camera looks along d = [-sin(a), cos(a), 0].
    Setting d · w_hat = 0 gives azim = atan2(w[1], w[0]).
    """
    w_hat = w / (np.linalg.norm(w) + 1e-10)
    azim = float(np.degrees(np.arctan2(w_hat[1], w_hat[0])))
    return 0.0, azim + 90


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise layer manifolds for Fashion-MNIST Pullover vs Shirt."
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLASS,
        metavar="N",
        help=f"Samples per class used for UMAP / animation (default: {DEFAULT_SAMPLES_PER_CLASS}).",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=DEFAULT_KNN_K,
        metavar="K",
        help=f"Neighbours per point in the within-class kNN graph (default: {DEFAULT_KNN_K}).",
    )
    parser.add_argument(
        "--r-thresh",
        type=float,
        default=None,
        metavar="R",
        help=(
            "Maximum edge length as a fraction of the overall point-cloud diameter "
            "(e.g. 0.15 = 15%%). Disabled by default (pure kNN, no distance filter)."
        ),
    )
    parser.add_argument(
        "--no-hull",
        action="store_true",
        help="Skip drawing convex hulls; show only graph edges and points.",
    )
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=DEFAULT_HOLD_FRAMES,
        metavar="H",
        help=f"Extra frames to hold on the final layer, both normal and edge-on (default: {DEFAULT_HOLD_FRAMES}).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for subsampling."
    )
    args = parser.parse_args()

    transform = transforms.Compose([transforms.ToTensor(), NORMALISE])

    full_dataset = ConcatDataset(
        [
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
        ]
    )
    loader = DataLoader(
        full_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(
        f"Running inference on {device}  "
        f"({len(full_dataset):,} samples, classes {CLASS_NAMES})"
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
    # Subsample for UMAP (keeps full features on disk)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    vis_features, vis_labels = subsample_per_class(
        features, labels, args.samples_per_class, rng
    )
    print(
        f"Subsampled to {args.samples_per_class} per class "
        f"({len(vis_labels)} points total) for UMAP.\n"
    )

    # ------------------------------------------------------------------
    # Compute UMAP projections (one per layer, 3-D)
    # ------------------------------------------------------------------
    layer_names = list(vis_features.keys())
    raw_projections: dict[str, np.ndarray] = {}
    for layer_name in tqdm(layer_names, desc="UMAP projections"):
        raw_projections[layer_name] = visualise_manifold(
            vis_features[layer_name], n_components=3
        )

    # ------------------------------------------------------------------
    # Procrustes-align all layers to the final layer so inter-frame
    # motion reflects genuine topology change, not arbitrary orientation.
    # ------------------------------------------------------------------
    projections = procrustes_align(raw_projections)
    print("Projections Procrustes-aligned to final layer.\n")

    # ------------------------------------------------------------------
    # Pre-compute graph edges and fit the final-layer decision boundary
    # ------------------------------------------------------------------
    layer_edges: dict[str, tuple[list, list]] = {
        name: build_class_edges(
            projections[name], vis_labels, args.knn_k, args.r_thresh
        )
        for name in layer_names
    }

    final_proj = projections[layer_names[-1]]
    svc = LinearSVC(dual="auto", max_iter=5000).fit(final_proj, vis_labels)
    w_final = svc.coef_[0]
    b_final = float(svc.intercept_[0])
    edge_on_elev, edge_on_azim = compute_edge_on_view(w_final)
    print(
        f"Decision boundary fitted. Edge-on view: elev={edge_on_elev:.1f}°, "
        f"azim={edge_on_azim:.1f}°\n"
    )

    # ------------------------------------------------------------------
    # Build frame sequence:
    #   layer-by-layer → hold on final (normal view) → hold on final (edge-on)
    # ------------------------------------------------------------------
    n = len(layer_names)
    frame_sequence: list[tuple[int, int]] = (
        [(i, _NORMAL) for i in range(n)]
        + [(n - 1, _NORMAL)] * args.hold_frames
        + [(n - 1, _EDGE_ON)] * args.hold_frames
    )

    fig = plt.figure(figsize=(9, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0], projection="3d")
    cax = fig.add_subplot(gs[1])

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=_NORM)
    cb = fig.colorbar(sm, cax=cax, ticks=range(len(CLASSES)))
    cb.set_ticklabels(CLASS_NAMES)

    def draw_frame(i: int) -> None:
        ax.clear()
        layer_idx, view_mode = frame_sequence[i]
        layer_name = layer_names[layer_idx]
        proj = projections[layer_name]
        is_final = layer_idx == n - 1

        if not args.no_hull:
            for cls in np.unique(vis_labels):
                draw_class_hull(ax, proj[vis_labels == cls], CMAP(_NORM(cls)))

        segments, colors = layer_edges[layer_name]
        if segments:
            ax.add_collection3d(
                Line3DCollection(segments, colors=colors, linewidths=0.4, alpha=0.3)
            )

        ax.scatter(
            proj[:, 0],
            proj[:, 1],
            proj[:, 2],
            c=vis_labels,
            cmap=CMAP,
            norm=_NORM,
            s=4,
            lw=0,
            alpha=0.5,
        )

        if is_final:
            draw_decision_boundary(ax, proj, w_final, b_final)

        title = f"Layer {layer_idx + 1}/{n}: {layer_name}"
        if view_mode == _EDGE_ON:
            title += "  [edge-on boundary]"
        elif is_final:
            title += "  [decision boundary]"
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("UMAP 1", fontsize=8)
        ax.set_ylabel("UMAP 2", fontsize=8)
        ax.set_zlabel("UMAP 3", fontsize=8)

        if view_mode == _EDGE_ON:
            ax.view_init(elev=edge_on_elev, azim=edge_on_azim)
        else:
            ax.view_init(elev=20, azim=45 + layer_idx * (180 / max(n - 1, 1)))

    ani = animation.FuncAnimation(
        fig,
        draw_frame,
        frames=len(frame_sequence),
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
