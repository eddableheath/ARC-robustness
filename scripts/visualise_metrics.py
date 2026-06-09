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

from arc_robustness.analysis.community import curvature_gap, modularity
from arc_robustness.analysis.evolution import local_ricci_evolution
from arc_robustness.analysis.graph_utils import build_knn_graph
from arc_robustness.analysis.ricci import (
    forman_ricci,
    ollivier_ricci,
    vertex_curvature,
)
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
ADV_DIR = OUTPUTS_DIR / "adversarial"

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
    """Decision plane as a patch parameterised in the plane's own coordinates.

    Works regardless of how the plane is oriented — no z-clipping or masking,
    so the boundary always renders in full.  The patch is sized to cover the
    full extent of the projected data plus *margin*.
    """
    w_norm = float(np.linalg.norm(w))
    if w_norm < 1e-10:
        return

    n = w / w_norm  # unit normal to the plane

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

def _compute_combined_umap(
    features_clean: dict[str, np.ndarray],
    features_adv: dict[str, np.ndarray],
    layer_names: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """UMAP on clean + adversarial concatenated per layer, Procrustes-aligned.

    Both sets share the same rotation (derived from clean projections) so the
    clean manifold remains comparable across runs. Adversarial points are
    centred on the clean centroid before rotation, placing them in the same
    coordinate frame.
    """
    N_c = next(iter(features_clean.values())).shape[0]

    raw_clean: dict[str, np.ndarray] = {}
    raw_adv: dict[str, np.ndarray] = {}
    for name in tqdm(layer_names, desc="UMAP (clean+adv)"):
        combined = np.concatenate([features_clean[name], features_adv[name]], axis=0)
        proj = visualise_manifold(combined, n_components=3)
        raw_clean[name] = proj[:N_c]
        raw_adv[name] = proj[N_c:]

    # Procrustes reference: centred final clean layer
    final_c = raw_clean[layer_names[-1]]
    ref = final_c - final_c.mean(axis=0)

    proj_clean: dict[str, np.ndarray] = {}
    proj_adv: dict[str, np.ndarray] = {}
    for name in layer_names:
        centroid = raw_clean[name].mean(axis=0)
        c = raw_clean[name] - centroid
        R, _ = orthogonal_procrustes(c, ref)
        proj_clean[name] = c @ R
        # Centre adversarial on the same clean centroid, apply same rotation
        proj_adv[name] = (raw_adv[name] - centroid) @ R

    return proj_clean, proj_adv


# ---------------------------------------------------------------------------
# Adversarial Ricci analysis
# ---------------------------------------------------------------------------

def compute_layer_metrics(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    knn_k: int,
    skip_ollivier: bool = False,
) -> dict:
    """Run the full Ricci/community pipeline on a feature dict.

    Returns a dict with keys matching ricci_metrics.npz where useful:
      mean_kappa, modularity, gap_ollivier, gap_forman, rho, r_layer, used_ollivier
    """
    layer_names = list(features.keys())
    N = len(labels)
    L = len(layer_names)

    adjs: dict = {}
    forman_curv: dict = {}
    ollivier_curv: dict = {}
    vert_forman: dict = {}
    vert_ollivier: dict = {}
    Q_arr = np.zeros(L)
    gap_forman_arr = np.full(L, np.nan)
    gap_ollivier_arr = np.full(L, np.nan)

    for i, name in enumerate(tqdm(layer_names, desc="  layers", leave=False)):
        pts = features[name]
        adj = build_knn_graph(pts, knn_k)
        adjs[name] = adj

        fr = forman_ricci(adj)
        forman_curv[name] = fr
        vert_forman[name] = vertex_curvature(fr, N)

        if not skip_ollivier:
            orr = ollivier_ricci(adj, pts)
            ollivier_curv[name] = orr
            vert_ollivier[name] = vertex_curvature(orr, N)
        else:
            ollivier_curv[name] = {}
            vert_ollivier[name] = np.full(N, np.nan)

        Q_arr[i] = modularity(adj, labels)
        gap_forman_arr[i] = curvature_gap(fr, labels)
        if not skip_ollivier:
            gap_ollivier_arr[i] = curvature_gap(ollivier_curv[name], labels)

    rho = np.full(N, np.nan)
    r_layer = np.full(L - 1, np.nan)
    if not skip_ollivier:
        rho, r_layer = local_ricci_evolution(features, adjs, ollivier_curv)

    mean_kappa = np.array([
        np.nanmean(vert_ollivier[n]) if not skip_ollivier else np.nanmean(vert_forman[n])
        for n in layer_names
    ])

    return {
        "layer_names": layer_names,
        "mean_kappa": mean_kappa,
        "modularity": Q_arr,
        "gap_ollivier": gap_ollivier_arr,
        "gap_forman": gap_forman_arr,
        "rho": rho,
        "r_layer": r_layer,
        "used_ollivier": not skip_ollivier,
    }


def plot_adversarial_ricci_comparison(
    clean_data: dict,
    adv_metrics: dict,
    out_path: Path,
    epsilon: float,
) -> None:
    """4-panel static figure comparing clean vs adversarial Ricci metrics.

    Panels:
      (0,0) Mean curvature κ̄ per layer
      (0,1) Curvature gap ΔO per layer
      (1,0) Modularity Q per layer
      (1,1) ρ(x) distribution — violin plot
    """
    layer_names = list(clean_data["layer_names"])
    L = len(layer_names)
    x = np.arange(L)
    tick_labels = [n.replace("relu", "L") for n in layer_names]

    used_ollivier = bool(clean_data.get("used_ollivier", False))
    kappa_label = "mean Ollivier κ̄" if used_ollivier else "mean Forman κ̄"

    # Pull clean metrics from the saved npz
    if "mean_kappa" in clean_data:
        clean_kappa = np.array(clean_data["mean_kappa"], dtype=float)
    elif used_ollivier:
        clean_kappa = np.nanmean(clean_data["vertex_ollivier"], axis=1).astype(float)
    else:
        clean_kappa = np.nanmean(clean_data["vertex_forman"], axis=1).astype(float)

    clean_Q = np.array(clean_data["modularity"], dtype=float)
    clean_gap = np.array(
        clean_data.get("curvature_gap_ollivier", clean_data.get("curvature_gap_forman")),
        dtype=float,
    )
    clean_rho = np.array(clean_data["rho"], dtype=float) if "rho" in clean_data else None

    adv_kappa = adv_metrics["mean_kappa"]
    adv_Q = adv_metrics["modularity"]
    adv_gap = adv_metrics["gap_ollivier"] if adv_metrics["used_ollivier"] else adv_metrics["gap_forman"]
    adv_rho = adv_metrics["rho"]

    C_CLEAN = "#1565C0"   # dark blue
    C_ADV   = "#C62828"   # dark red

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Adversarial Ricci analysis  —  clean vs FGSM ε={epsilon:.3f}",
        fontsize=13, fontweight="bold",
    )

    def _line_panel(ax, clean_vals, adv_vals, ylabel, title):
        ax.plot(x, clean_vals, "o-", color=C_CLEAN, lw=2, ms=6, label="clean")
        ax.plot(x, adv_vals,   "s--", color=C_ADV,   lw=2, ms=6, label=f"adv ε={epsilon:.3f}")
        ax.axhline(0, color="black", ls=":", lw=0.8, alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _line_panel(axes[0, 0], clean_kappa, adv_kappa, kappa_label, kappa_label + " per layer")
    _line_panel(axes[0, 1], clean_gap,   adv_gap,   "ΔO",         "Curvature gap ΔO per layer")
    _line_panel(axes[1, 0], clean_Q,     adv_Q,     "Q",          "Modularity Q per layer")

    # ρ(x) violin comparison
    ax_v = axes[1, 1]
    rho_data = []
    rho_positions = []
    rho_colors = []
    rho_labels = []
    if clean_rho is not None and not np.all(np.isnan(clean_rho)):
        clean_finite = clean_rho[np.isfinite(clean_rho)]
        rho_data.append(clean_finite)
        rho_positions.append(1)
        rho_colors.append(C_CLEAN)
        rho_labels.append("clean")
    if not np.all(np.isnan(adv_rho)):
        adv_finite = adv_rho[np.isfinite(adv_rho)]
        rho_data.append(adv_finite)
        rho_positions.append(2)
        rho_colors.append(C_ADV)
        rho_labels.append(f"adv ε={epsilon:.3f}")

    if rho_data:
        parts = ax_v.violinplot(rho_data, positions=rho_positions, showmedians=True)
        for body, color in zip(parts["bodies"], rho_colors):
            body.set_facecolor(color)
            body.set_alpha(0.55)
        parts["cmedians"].set_colors(rho_colors)
        parts["cbars"].set_colors(rho_colors)
        parts["cmins"].set_colors(rho_colors)
        parts["cmaxes"].set_colors(rho_colors)

        ax_v.set_xticks(rho_positions)
        ax_v.set_xticklabels(rho_labels, fontsize=9)
        for pos, vals, color in zip(rho_positions, rho_data, rho_colors):
            frac_neg = (vals < 0).mean()
            ax_v.text(
                pos, ax_v.get_ylim()[0] if ax_v.get_ylim()[0] != ax_v.get_ylim()[1] else -1,
                f"{frac_neg:.0%} ρ<0",
                ha="center", va="bottom", fontsize=8, color=color,
            )
    else:
        ax_v.text(0.5, 0.5, "ρ(x) not available\n(skip_ollivier was set)",
                  ha="center", va="center", transform=ax_v.transAxes, fontsize=9, color="gray")

    ax_v.axhline(0, color="black", ls=":", lw=0.8, alpha=0.4)
    ax_v.set_ylabel("ρ(x)", fontsize=9)
    ax_v.set_title("ρ(x) distribution", fontsize=10)
    ax_v.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Adversarial Ricci comparison saved to {out_path}")


def make_combined_gif(
    data: dict,
    features: dict | None,
    labels: np.ndarray,
    out_path: Path,
    knn_k: int = 6,
    hold_frames: int = 3,
    adv_features: dict | None = None,
    adv_epsilon: float | None = None,
    adv_labels: np.ndarray | None = None,
) -> None:
    layer_names = list(data["layer_names"])
    L = len(layer_names)

    # ---- UMAP projections -----------------------------------------------
    adv_projections: dict[str, np.ndarray] | None = None

    if adv_features is not None:
        # Always recompute UMAP so clean and adversarial share a coordinate frame
        projections, adv_projections = _compute_combined_umap(
            features, adv_features, layer_names,
        )
    elif "umap_projections" in data:
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
    if adv_projections is not None:
        eps_label = f"ε={adv_epsilon:.2f}" if adv_epsilon is not None else "adv"
        for cls_idx, cls in enumerate(np.unique(labels)):
            legend_handles.append(
                Line2D(
                    [0], [0], marker="x", color=CLASS_CMAP(CLASS_NORM(cls)),
                    linestyle="None", markersize=7, markeredgewidth=1.5,
                    label=f"{CLASS_NAMES[cls_idx]} adv ({eps_label})",
                )
            )

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

        if adv_projections is not None:
            adv_proj = adv_projections[name]
            cls_labels = adv_labels if adv_labels is not None else labels
            for cls in np.unique(cls_labels):
                mask = cls_labels == cls
                color = CLASS_CMAP(CLASS_NORM(cls))
                ax_main.scatter(
                    adv_proj[mask, 0], adv_proj[mask, 1], adv_proj[mask, 2],
                    color=color, marker="x", s=20, linewidths=1.2, alpha=0.65,
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
    parser.add_argument(
        "--adv-epsilon",
        type=float,
        default=None,
        metavar="ε",
        help=(
            "Overlay adversarial examples from outputs/adversarial/eps_{ε:.3f}/ "
            "on the manifold animation. Requires generate_adversarial.py to have "
            "been run with this ε first."
        ),
    )
    parser.add_argument(
        "--ricci-analysis",
        action="store_true",
        help=(
            "Compute Ricci/community metrics for adversarial features and produce "
            "a comparison figure (requires --adv-epsilon). Ollivier-Ricci is "
            "computed by default; use --skip-ollivier for a faster Forman-only run."
        ),
    )
    parser.add_argument(
        "--skip-ollivier",
        action="store_true",
        help="Use Forman-Ricci instead of Ollivier-Ricci for --ricci-analysis (much faster).",
    )
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

    # ---- Adversarial Ricci analysis ----------------------------------------
    if args.ricci_analysis:
        if args.adv_epsilon is None:
            parser.error("--ricci-analysis requires --adv-epsilon")
        adv_dir_ra = ADV_DIR / f"eps_{args.adv_epsilon:.3f}"
        if not adv_dir_ra.exists():
            raise FileNotFoundError(
                f"{adv_dir_ra} not found. "
                f"Run: uv run python scripts/generate_adversarial.py --epsilon {args.adv_epsilon}"
            )
        ra_layer_names = data["layer_names"]
        missing_ra = [n for n in ra_layer_names if not (adv_dir_ra / f"{n}.npy").exists()]
        if missing_ra:
            raise FileNotFoundError(
                f"Missing adversarial features for layers {missing_ra} in {adv_dir_ra}."
            )
        adv_feats_ra = {n: np.load(adv_dir_ra / f"{n}.npy") for n in ra_layer_names}
        adv_labels_ra = np.load(adv_dir_ra / "labels.npy")
        print(
            f"\nComputing adversarial Ricci metrics for ε={args.adv_epsilon:.3f} "
            f"({'Ollivier' if not args.skip_ollivier else 'Forman only'})..."
        )
        adv_metrics = compute_layer_metrics(
            adv_feats_ra, adv_labels_ra,
            knn_k=int(data.get("knn_k", 6)),
            skip_ollivier=args.skip_ollivier,
        )
        ra_out = OUTPUTS_DIR / f"adversarial_ricci_eps{args.adv_epsilon:.3f}.png"
        plot_adversarial_ricci_comparison(data, adv_metrics, ra_out, args.adv_epsilon)

    if not args.no_gif:
        labels = data["labels"]
        features = None
        adv_features = None
        adv_labels = None

        # Load adversarial features when requested
        if args.adv_epsilon is not None:
            adv_dir = ADV_DIR / f"eps_{args.adv_epsilon:.3f}"
            if not adv_dir.exists():
                raise FileNotFoundError(
                    f"{adv_dir} not found. "
                    f"Run: uv run python scripts/generate_adversarial.py "
                    f"--epsilon {args.adv_epsilon}"
                )
            layer_names = data["layer_names"]
            missing = [n for n in layer_names if not (adv_dir / f"{n}.npy").exists()]
            if missing:
                raise FileNotFoundError(
                    f"Missing adversarial features for layers {missing} in {adv_dir}."
                )
            adv_features = {n: np.load(adv_dir / f"{n}.npy") for n in layer_names}
            adv_labels = np.load(adv_dir / "labels.npy")
            print(
                f"Loaded adversarial features for ε={args.adv_epsilon:.3f} "
                f"({next(iter(adv_features.values())).shape[0]} points, "
                f"labels: {np.unique(adv_labels).tolist()})"
            )

        # Raw clean features are needed when UMAP must be (re)computed:
        # either because projections aren't saved yet, or because the adversarial
        # overlay forces a combined UMAP refit.
        need_features = "umap_projections" not in data or adv_features is not None
        if need_features:
            reason = "adversarial overlay" if adv_features is not None else "missing umap_projections"
            print(f"Loading clean features for UMAP ({reason})...")
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
        gif_name = (
            f"combined_manifold_adv{args.adv_epsilon:.3f}.gif"
            if args.adv_epsilon is not None
            else "combined_manifold.gif"
        )
        make_combined_gif(
            data, features, labels,
            OUTPUTS_DIR / gif_name,
            knn_k=knn_k,
            adv_features=adv_features,
            adv_epsilon=args.adv_epsilon,
            adv_labels=adv_labels,
        )


if __name__ == "__main__":
    main()
