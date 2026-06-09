"""
Visualise Ricci and community metrics from outputs/ricci_metrics.npz.

Produces two outputs:

  outputs/community_metrics.png
      Four-panel static figure: Modularity Q, Normalised Cut, Algebraic
      Connectivity, and Curvature Gap ΔO as functions of layer depth.

  outputs/combined_manifold.gif
      Animated GIF with three panels:
        LEFT   — 3-D UMAP of activations, coloured by class (viridis),
                 with per-class convex hulls, kNN graph edges, and the
                 linear decision boundary drawn on the final hold frames.
        TOP RIGHT  — mean Ollivier-Ricci κ̄ per layer, growing as the
                     animation steps through layers.
        BOTTOM RIGHT — per-layer Pearson r(η_ℓ, O_ℓ) across vertices,
                       also growing frame by frame.

Usage:
    uv run python scripts/visualise_metrics.py
    uv run python scripts/visualise_metrics.py --no-gif
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
from matplotlib.lines import Line2D
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

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Changing the sparsity structure")

BATCH_SIZE = 256
DEFAULT_SAMPLES_PER_CLASS = 200
NORMALISE = transforms.Normalize((0.2860,), (0.3530,))
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

# Class colour map — matches visualise_manifold.py
_v = matplotlib.colormaps["viridis"]
CLASS_CMAP = matplotlib.colors.ListedColormap([_v(0.1), _v(0.85)])
CLASS_NORM = matplotlib.colors.Normalize(vmin=-0.5, vmax=len(CLASSES) - 0.5)


def subsample_per_class(features, labels, n_per_class, rng):
    keep: list[int] = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        chosen = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        keep.extend(chosen.tolist())
    keep_arr = np.array(sorted(keep))
    return {n: a[keep_arr] for n, a in features.items()}, labels[keep_arr]


# ---------------------------------------------------------------------------
# Figure 1: static community-metrics panel
# ---------------------------------------------------------------------------

def plot_community_metrics(data: dict, out_path: Path) -> None:
    layer_names = data["layer_names"]
    x = np.arange(len(layer_names))
    tick_labels = [n.replace("relu", "ReLU ") for n in layer_names]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "Community structure evolving through network depth\n"
        f"Fashion-MNIST: {CLASS_NAMES[0]} vs {CLASS_NAMES[1]}",
        fontsize=13,
    )

    ax = axes[0, 0]
    ax.plot(x, data["modularity"], "o-", color="#2196F3", lw=2)
    ax.axhline(0.3, color="gray", ls="--", lw=1, label="Q = 0.3 (strong)")
    ax.axhline(0.0, color="black", ls=":", lw=0.8)
    ax.set_title("Modularity Q")
    ax.set_ylabel("Q")
    ax.legend(fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)

    ax = axes[0, 1]
    ax.plot(x, data["normalised_cut"], "o-", color="#E91E63", lw=2)
    ax.set_title("Normalised Cut (NCut)")
    ax.set_ylabel("NCut")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)

    ax = axes[1, 0]
    ax.plot(x, data["algebraic_connectivity"], "o-", color="#4CAF50", lw=2)
    ax.set_title("Algebraic Connectivity (Fiedler value)")
    ax.set_ylabel("λ₂")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)

    ax = axes[1, 1]
    has_ollivier = not np.all(np.isnan(data["curvature_gap_ollivier"]))
    if has_ollivier:
        ax.plot(x, data["curvature_gap_ollivier"], "o-", color="#FF9800", lw=2, label="Ollivier")
    ax.plot(x, data["curvature_gap_forman"], "s--", color="#9C27B0", lw=1.5, label="Forman")
    ax.axhline(0.0, color="black", ls=":", lw=0.8)
    ax.set_title("Curvature Gap ΔO")
    ax.set_ylabel("ΔO")
    ax.legend(fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)

    for ax in axes.flat:
        ax.set_xlabel("Layer")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Community metrics figure saved to {out_path}")


# ---------------------------------------------------------------------------
# Helpers for the 3-D manifold panel (mirror of visualise_manifold.py)
# ---------------------------------------------------------------------------

def _build_class_edges(
    proj: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> tuple[list, list]:
    """kNN edges within each class in UMAP space (visualisation only)."""
    segments: list = []
    colors: list = []
    for cls in np.unique(labels):
        pts = proj[labels == cls]
        if len(pts) <= k:
            continue
        graph = kneighbors_graph(pts, n_neighbors=k, mode="distance", include_self=False)
        rows, cols = graph.nonzero()
        color = CLASS_CMAP(CLASS_NORM(cls))
        for src, dst in zip(rows, cols):
            if src < dst:
                segments.append([pts[src], pts[dst]])
                colors.append(color)
    return segments, colors


def _draw_hull(ax, pts: np.ndarray, color: tuple, alpha: float = 0.12) -> None:
    """Convex hull as a transparent triangulated surface."""
    try:
        hull = ConvexHull(pts)
    except (QhullError, Exception):
        return
    poly = Poly3DCollection(pts[hull.simplices], linewidth=0.2)
    poly.set_facecolor((*color[:3], alpha))
    poly.set_edgecolor((*color[:3], 0.18))
    ax.add_collection3d(poly)


def _draw_boundary(
    ax,
    proj: np.ndarray,
    w: np.ndarray,
    b: float,
    margin: float = 0.1,
) -> None:
    """Linear SVM decision plane given pre-fitted coefficients w, b."""
    if abs(w[2]) < 1e-6:
        return
    pad = margin * (proj.max(axis=0) - proj.min(axis=0))
    xx, yy = np.meshgrid(
        np.linspace(proj[:, 0].min() - pad[0], proj[:, 0].max() + pad[0], 35),
        np.linspace(proj[:, 1].min() - pad[1], proj[:, 1].max() + pad[1], 35),
    )
    zz = np.clip(
        -(w[0] * xx + w[1] * yy + b) / w[2],
        proj[:, 2].min() - pad[2],
        proj[:, 2].max() + pad[2],
    )
    ax.plot_surface(xx, yy, zz, color="white", alpha=0.35, linewidth=0, antialiased=True)
    ax.plot_wireframe(xx, yy, zz, color="dimgray", linewidth=0.6, alpha=0.7, rstride=4, cstride=4)


def _compute_edge_on_view(w: np.ndarray) -> tuple[float, float]:
    """Return (elev, azim) so the plane with normal w appears edge-on."""
    w_hat = w / (np.linalg.norm(w) + 1e-10)
    azim = float(np.degrees(np.arctan2(w_hat[1], w_hat[0])))
    return 0.0, azim + 90


# View-mode sentinels for frame sequence
_NORMAL = 0
_EDGE_ON = 1


# ---------------------------------------------------------------------------
# Figure 2: combined manifold + metric line-plot animation
# ---------------------------------------------------------------------------

def make_combined_gif(
    data: dict,
    features: dict | None,
    labels: np.ndarray,
    out_path: Path,
    knn_k: int = 6,
    hold_frames: int = 3,
) -> None:
    layer_names = list(data["layer_names"])
    L = len(layer_names)

    # ---- UMAP projections -----------------------------------------------
    if "umap_projections" in data:
        umap_stack = data["umap_projections"]  # (L, N, 3)
        projections = {n: umap_stack[i] for i, n in enumerate(layer_names)}
    else:
        print("UMAP projections not in npz — computing on the fly...")
        raw_projs: dict[str, np.ndarray] = {}
        for name in tqdm(layer_names, desc="UMAP 3D"):
            raw_projs[name] = visualise_manifold(features[name], n_components=3)
        final_raw = raw_projs[layer_names[-1]]
        ref = final_raw - final_raw.mean(axis=0)
        projections = {}
        for name in layer_names:
            p = raw_projs[name] - raw_projs[name].mean(axis=0)
            R, _ = orthogonal_procrustes(p, ref)
            projections[name] = p @ R

    # ---- Metrics for line plots -----------------------------------------
    used_ollivier = bool(data.get("used_ollivier", not np.all(np.isnan(data["vertex_ollivier"]))))
    if "mean_kappa" in data:
        mean_kappa = np.array(data["mean_kappa"], dtype=float)
    elif used_ollivier:
        mean_kappa = np.nanmean(data["vertex_ollivier"], axis=1).astype(float)
    else:
        mean_kappa = np.nanmean(data["vertex_forman"], axis=1).astype(float)
    kappa_label = "mean Ollivier κ̄" if used_ollivier else "mean Forman κ̄"

    # r_layer is only present in npz generated by the updated compute_metrics.py
    r_layer_raw = data.get("r_layer")
    r_layer = (
        np.array(r_layer_raw, dtype=float)
        if r_layer_raw is not None
        else np.full(L - 1, np.nan)
    )
    has_rlayer = not np.all(np.isnan(r_layer))

    # Fallback: full per-vertex rho (available in all versions of the npz)
    rho_raw = data.get("rho")
    rho = np.array(rho_raw, dtype=float) if rho_raw is not None else None
    has_rho = rho is not None and not np.all(np.isnan(rho))

    # ---- Decision boundary (fit once on final layer projection) ----------
    final_proj = projections[layer_names[-1]]
    svc = LinearSVC(dual="auto", max_iter=5000).fit(final_proj, labels)
    w, b = svc.coef_[0], float(svc.intercept_[0])

    # ---- Pre-compute kNN edges for each layer ---------------------------
    layer_edges: dict[str, tuple[list, list]] = {
        name: _build_class_edges(projections[name], labels, knn_k)
        for name in tqdm(layer_names, desc="kNN edges")
    }

    # ---- Frame sequence -------------------------------------------------
    # Each entry is (layer_idx, view_mode)
    frame_seq = (
        [(i, _NORMAL) for i in range(L)]
        + [(L - 1, _NORMAL)] * hold_frames    # hold with boundary
        + [(L - 1, _EDGE_ON)] * hold_frames   # edge-on: boundary collapses to line
    )

    # ---- Figure layout --------------------------------------------------
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[3, 1.2],
        hspace=0.5, wspace=0.3,
        left=0.04, right=0.97,
        top=0.92, bottom=0.09,
    )
    ax_main = fig.add_subplot(gs[:, 0], projection="3d")
    ax_kappa = fig.add_subplot(gs[0, 1])
    ax_rho = fig.add_subplot(gs[1, 1])

    # ---- Static grey background for both line plots ---------------------
    x_k = np.arange(L)
    x_r = np.arange(L - 1)

    ax_kappa.plot(x_k, mean_kappa, "o", color="lightgrey", ms=5, zorder=1)
    ax_kappa.axhline(0, color="black", ls=":", lw=0.8, alpha=0.4)
    ax_kappa.set_title(kappa_label, fontsize=9, pad=4)
    ax_kappa.set_xlabel("Layer", fontsize=8)
    ax_kappa.tick_params(labelsize=7)
    ax_kappa.grid(True, alpha=0.3)
    ax_kappa.set_xlim(-0.5, L - 0.5)
    valid_k = mean_kappa[np.isfinite(mean_kappa)]
    if len(valid_k):
        pad_k = max((valid_k.max() - valid_k.min()) * 0.15, 0.05)
        ax_kappa.set_ylim(valid_k.min() - pad_k, valid_k.max() + pad_k)

    ax_rho.axhline(0, color="black", ls=":", lw=0.8, alpha=0.4)
    ax_rho.tick_params(labelsize=7)
    ax_rho.grid(True, alpha=0.3)
    if has_rlayer:
        ax_rho.plot(x_r, r_layer, "o", color="lightgrey", ms=5, zorder=1)
        ax_rho.set_title("layer r(η_ℓ, O_ℓ)", fontsize=9, pad=4)
        ax_rho.set_xlabel("Transition ℓ→ℓ+1", fontsize=8)
        ax_rho.set_xlim(-0.5, max(L - 2, 0) + 0.5)
        valid_r = r_layer[np.isfinite(r_layer)]
        if len(valid_r):
            pad_r = max((valid_r.max() - valid_r.min()) * 0.15, 0.1)
            ax_rho.set_ylim(valid_r.min() - pad_r, valid_r.max() + pad_r)
    elif has_rho:
        # Old npz: r_layer not yet saved; show full ρ(x) summary instead
        mean_rho = float(np.nanmean(rho))
        std_rho = float(np.nanstd(rho))
        ax_rho.axhline(mean_rho, color="lightgrey", ls="-", lw=3, zorder=1)
        ax_rho.set_title(f"mean ρ(x) = {mean_rho:.3f}", fontsize=9, pad=4)
        ax_rho.set_xlabel("(re-run compute_metrics.py for r_ℓ)", fontsize=7)
        ax_rho.set_xlim(-0.5, L - 0.5)
        ax_rho.set_ylim(mean_rho - 2 * std_rho - 0.1, mean_rho + 2 * std_rho + 0.1)
        ax_rho.text(
            0.05, 0.08,
            f"{(rho < 0).mean():.0%} vertices ρ < 0",
            transform=ax_rho.transAxes, ha="left", va="bottom", fontsize=8, color="gray",
        )
    else:
        ax_rho.set_title("ρ(x)", fontsize=9, pad=4)
        ax_rho.text(
            0.5, 0.5, "no ρ data\n(re-run compute_metrics.py)",
            transform=ax_rho.transAxes, ha="center", va="center",
            fontsize=8, color="gray",
        )

    # Live line artists (data updated each frame via set_data)
    (kappa_line,) = ax_kappa.plot([], [], "o-", color="#FF9800", ms=5, lw=1.5, zorder=2)
    (kappa_cur,) = ax_kappa.plot([], [], "*", color="crimson", ms=9, zorder=3)
    (rho_line,) = ax_rho.plot([], [], "o-", color="#2196F3", ms=5, lw=1.5, zorder=2)
    (rho_cur,) = ax_rho.plot([], [], "*", color="crimson", ms=9, zorder=3)

    # Class legend for 3-D panel
    unique_cls = np.unique(labels)
    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=CLASS_CMAP(CLASS_NORM(cls)),
            markersize=7, label=CLASS_NAMES[cls_idx],
        )
        for cls_idx, cls in enumerate(unique_cls)
    ]

    def draw_frame(i: int) -> None:
        layer_idx, view_mode = frame_seq[i]
        show_boundary = i >= L          # hold and edge-on frames
        is_edge_on = view_mode == _EDGE_ON
        name = layer_names[layer_idx]
        proj = projections[name]

        # ---- 3-D manifold -----------------------------------------------
        ax_main.cla()

        segs, cols = layer_edges[name]
        if segs:
            ax_main.add_collection3d(
                Line3DCollection(segs, colors=cols, linewidths=0.4, alpha=0.35)
            )

        ax_main.scatter(
            proj[:, 0], proj[:, 1], proj[:, 2],
            c=labels, cmap=CLASS_CMAP, norm=CLASS_NORM,
            s=5, lw=0, alpha=0.6,
        )

        if show_boundary:
            _draw_boundary(ax_main, proj, w, b)

        if is_edge_on:
            title_note = "  [edge-on]"
            edge_elev, edge_azim = _compute_edge_on_view(w)
            ax_main.view_init(elev=edge_elev, azim=edge_azim)
        elif show_boundary:
            title_note = "  [decision boundary]"
            azim = 30 + layer_idx * (180 / max(L - 1, 1))
            ax_main.view_init(elev=20, azim=azim)
        else:
            title_note = ""
            azim = 30 + layer_idx * (180 / max(L - 1, 1))
            ax_main.view_init(elev=20, azim=azim)

        ax_main.set_title(
            f"Layer {layer_idx + 1}/{L}: {name}{title_note}",
            fontsize=11, pad=4,
        )
        ax_main.set_xlabel("UMAP 1", fontsize=8, labelpad=0)
        ax_main.set_ylabel("UMAP 2", fontsize=8, labelpad=0)
        ax_main.set_zlabel("UMAP 3", fontsize=8, labelpad=0)
        ax_main.tick_params(labelsize=6)
        ax_main.legend(handles=legend_handles, fontsize=8, loc="upper right")

        # ---- Line plots -------------------------------------------------
        reveal = layer_idx + 1
        kappa_line.set_data(x_k[:reveal], mean_kappa[:reveal])
        kappa_cur.set_data([layer_idx], [mean_kappa[layer_idx]])

        # r_layer[ell] covers transition ell→ell+1; reveal one per layer visited
        r_reveal = layer_idx
        if has_rlayer and r_reveal > 0:
            rho_line.set_data(x_r[:r_reveal], r_layer[:r_reveal])
            rho_cur.set_data([x_r[r_reveal - 1]], [r_layer[r_reveal - 1]])
        else:
            rho_line.set_data([], [])
            rho_cur.set_data([], [])

    ani = animation.FuncAnimation(
        fig, draw_frame, frames=len(frame_seq), interval=1500, repeat=True,
    )
    ani.save(out_path, writer="pillow", fps=1)
    plt.close(fig)
    print(f"Combined manifold animation saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise Ricci metrics from outputs/ricci_metrics.npz."
    )
    parser.add_argument(
        "--no-gif", action="store_true", help="Skip the animated GIF (faster)."
    )
    parser.add_argument(
        "--samples-per-class", type=int, default=DEFAULT_SAMPLES_PER_CLASS, metavar="N"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics_path = OUTPUTS_DIR / "ricci_metrics.npz"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found. Run compute_metrics.py first."
        )

    raw = np.load(metrics_path, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    data["layer_names"] = list(data["layer_names"])

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_community_metrics(data, OUTPUTS_DIR / "community_metrics.png")

    if not args.no_gif:
        labels = data["labels"]
        features = None

        # Only need raw features if UMAP projections are missing from the npz
        if "umap_projections" not in data:
            print("umap_projections not found in npz — loading features for UMAP...")
            feature_dir = FEATURES_DIR / "full"
            label_path = feature_dir / "labels.npy"
            if label_path.exists():
                labels_full = np.load(label_path)
                layer_files = sorted(
                    p for p in feature_dir.iterdir() if p.name != "labels.npy"
                )
                features_full = {p.stem: np.load(p) for p in layer_files}
            else:
                print("Features not on disk — extracting from model...")
                transform = transforms.Compose([transforms.ToTensor(), NORMALISE])
                dataset = ConcatDataset([
                    filter_to_classes(
                        torchvision.datasets.FashionMNIST(
                            DATA_DIR, train=True, download=True, transform=transform,
                        ),
                        CLASSES,
                    ),
                    filter_to_classes(
                        torchvision.datasets.FashionMNIST(
                            DATA_DIR, train=False, download=True, transform=transform,
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

        knn_k = int(data.get("knn_k", 6))
        make_combined_gif(
            data, features, labels,
            OUTPUTS_DIR / "combined_manifold.gif",
            knn_k=knn_k,
        )


if __name__ == "__main__":
    main()
