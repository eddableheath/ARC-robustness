"""
visualise_manifold_smooth.py

Smooth interpolated 3-D UMAP animation of the network's feature manifold.

Improvements over visualise_manifold.py:
  - Cosine-eased morphing between layer projections
  - Per-edge alpha fade: k-NN edges dissolve/appear as the graph topology changes
  - Continuous camera azimuth sweep between layers (no snap)
  - Decision boundary fades in on arriving at the final layer
  - Smooth camera sweep to the edge-on view at the end
  - UMAP projections cached to outputs/umap_projections.npz (fast reruns)

Usage:
  uv run python scripts/visualise_manifold_smooth.py
  uv run python scripts/visualise_manifold_smooth.py --interp-frames 30 --fps 24
  uv run python scripts/visualise_manifold_smooth.py --format mp4  # requires ffmpeg
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.linalg import orthogonal_procrustes
from sklearn.neighbors import kneighbors_graph
from sklearn.svm import LinearSVC
from tqdm import tqdm

from arc_robustness.training.model import CLASS_NAMES, CLASSES
from arc_robustness.visualisation.visualise import visualise_manifold

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Changing the sparsity structure")

# ─── Paths ───────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "features" / "full"
OUTPUTS_DIR  = ROOT / "outputs"
UMAP_CACHE   = OUTPUTS_DIR / "umap_projections.npz"

# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_KNN_K        = 6
DEFAULT_SAMPLES      = 200      # per class
DEFAULT_INTERP       = 20       # interpolation frames between layers
DEFAULT_HOLD         = 8        # hold frames at each layer
DEFAULT_ROTATION     = 40       # frames for the edge-on camera sweep
DEFAULT_FPS          = 20
DEFAULT_DPI          = 90
DEFAULT_SEED         = 42
EDGE_BASE_ALPHA      = 0.35

# ─── Colour map ──────────────────────────────────────────────────────────────

_v         = matplotlib.colormaps["viridis"]
CLASS_CMAP = matplotlib.colors.ListedColormap([_v(0.1), _v(0.85)])
CLASS_NORM = matplotlib.colors.Normalize(vmin=-0.5, vmax=len(CLASSES) - 0.5)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def subsample_per_class(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        chosen = rng.choice(idx, size=min(n, len(idx)), replace=False)
        keep.extend(chosen.tolist())
    return np.array(sorted(keep))


def load_or_compute_projections(
    features_sub: dict[str, np.ndarray],
    layer_names: list[str],
    cache_path: Path,
    force: bool = False,
) -> dict[str, np.ndarray]:
    if not force and cache_path.exists():
        cached = np.load(cache_path)
        if all(n in cached for n in layer_names):
            print(f"  Loaded UMAP projections from cache: {cache_path}")
            return {n: cached[n] for n in layer_names}

    print("  Computing UMAP projections (takes ~30–60 s)…")
    projections: dict[str, np.ndarray] = {}
    for name in tqdm(layer_names, desc="  UMAP"):
        projections[name] = visualise_manifold(features_sub[name], n_components=3)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **projections)
    print(f"  Cached to {cache_path}")
    return projections


def procrustes_align(
    projections: dict[str, np.ndarray], layer_names: list[str]
) -> dict[str, np.ndarray]:
    final = projections[layer_names[-1]]
    ref   = final - final.mean(axis=0)
    aligned: dict[str, np.ndarray] = {}
    for name in layer_names:
        proj = projections[name]
        p    = proj - proj.mean(axis=0)
        R, _ = orthogonal_procrustes(p, ref)
        aligned[name] = p @ R
    return aligned


def build_edge_index_pairs(
    proj: np.ndarray, labels: np.ndarray, k: int
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], np.ndarray]]:
    """Return (set of (i,j) vertex-index pairs, dict mapping pair → RGBA color).

    Pairs are within-class only, sorted so i < j.
    """
    pairs:  set[tuple[int, int]]               = set()
    colors: dict[tuple[int, int], np.ndarray]  = {}
    for cls in np.unique(labels):
        mask = labels == cls
        idx  = np.where(mask)[0]
        pts  = proj[mask]
        if len(pts) <= k:
            continue
        graph = kneighbors_graph(pts, n_neighbors=k,
                                 mode="connectivity", include_self=False)
        rows, cols = graph.nonzero()
        rgba = np.array(CLASS_CMAP(CLASS_NORM(int(cls))))
        for r, c in zip(rows, cols):
            gi, gj = int(idx[r]), int(idx[c])
            pair = (min(gi, gj), max(gi, gj))
            pairs.add(pair)
            colors[pair] = rgba
    return pairs, colors


def build_edge_frame(
    proj: np.ndarray,
    pairs_src: set,
    pairs_dst: set,
    colors_src: dict,
    colors_dst: dict,
    t: float,
) -> tuple[list, list]:
    """Compute edge segments + RGBA colors for the interpolated state at t ∈ [0,1].

    - Edges in both sets: full opacity throughout, positions follow vertices.
    - Edges only in src: fade out (alpha → 0).
    - Edges only in dst: fade in (alpha 0 → EDGE_BASE_ALPHA).
    """
    all_pairs = pairs_src | pairs_dst
    segments: list[np.ndarray] = []
    rgba:     list[tuple]       = []

    for pair in all_pairs:
        i, j = pair
        segments.append([proj[i].tolist(), proj[j].tolist()])

        in_src = pair in pairs_src
        in_dst = pair in pairs_dst

        if in_src and in_dst:
            c     = colors_src[pair]
            alpha = EDGE_BASE_ALPHA
        elif in_src:
            c     = colors_src[pair]
            alpha = EDGE_BASE_ALPHA * (1.0 - t)
        else:
            c     = colors_dst[pair]
            alpha = EDGE_BASE_ALPHA * t

        rgba.append((float(c[0]), float(c[1]), float(c[2]), float(alpha)))

    return segments, rgba


def draw_boundary(
    ax, proj: np.ndarray, w: np.ndarray, b: float,
    surf_alpha: float = 0.35, margin: float = 0.1,
) -> None:
    """Decision plane as a patch in local (u, v) coordinates."""
    w_norm = float(np.linalg.norm(w))
    if w_norm < 1e-10 or surf_alpha < 0.01:
        return

    n        = w / w_norm
    centroid = proj.mean(axis=0)
    p0       = centroid - (float(n @ centroid) + b / w_norm) * n

    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u   = np.cross(n, ref)
    u  /= np.linalg.norm(u)
    v   = np.cross(n, u)

    d = proj - p0
    R = float(max(np.abs(d @ u).max(), np.abs(d @ v).max())) * (1.0 + margin)

    ss, tt = np.meshgrid(np.linspace(-R, R, 50), np.linspace(-R, R, 50))
    pts    = p0 + ss[..., np.newaxis] * u + tt[..., np.newaxis] * v
    xx, yy, zz = pts[..., 0], pts[..., 1], pts[..., 2]

    ax.plot_surface(xx, yy, zz, color="white", alpha=surf_alpha * 0.9,
                    linewidth=0, antialiased=True)
    ax.plot_wireframe(xx, yy, zz, color="dimgray",
                      linewidth=0.6, alpha=surf_alpha * 1.8,
                      rstride=3, cstride=3)

    pad = 0.1 * (proj.max(axis=0) - proj.min(axis=0))
    ax.set_xlim(proj[:, 0].min() - pad[0], proj[:, 0].max() + pad[0])
    ax.set_ylim(proj[:, 1].min() - pad[1], proj[:, 1].max() + pad[1])
    ax.set_zlim(proj[:, 2].min() - pad[2], proj[:, 2].max() + pad[2])


def _ease(t: float) -> float:
    """Cosine ease-in-out: smoothstep from 0 → 1."""
    return (1.0 - np.cos(np.pi * t)) / 2.0


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smooth interpolated UMAP layer animation."
    )
    parser.add_argument("--interp-frames", type=int, default=DEFAULT_INTERP,
                        metavar="N",
                        help=f"Frames interpolated between each layer pair (default {DEFAULT_INTERP}).")
    parser.add_argument("--hold-frames", type=int, default=DEFAULT_HOLD,
                        metavar="H",
                        help=f"Frames to hold at each layer (default {DEFAULT_HOLD}).")
    parser.add_argument("--rotation-frames", type=int, default=DEFAULT_ROTATION,
                        metavar="R",
                        help=f"Frames for the final edge-on camera sweep (default {DEFAULT_ROTATION}).")
    parser.add_argument("--knn-k", type=int, default=DEFAULT_KNN_K)
    parser.add_argument("--samples-per-class", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-cache", action="store_true",
                        help="Force recompute UMAP projections even if cache exists.")
    parser.add_argument("--format", choices=["gif", "mp4"], default="gif",
                        dest="fmt",
                        help="Output format (mp4 requires ffmpeg, default: gif).")
    parser.add_argument("--out", type=Path,
                        default=None,
                        help="Override output path (default: outputs/smooth_manifold.{fmt}).")
    args = parser.parse_args()

    # ── Load features ─────────────────────────────────────────────────────────
    labels_full = np.load(FEATURES_DIR / "labels.npy")
    layer_names = sorted(
        p.stem for p in FEATURES_DIR.iterdir() if p.name != "labels.npy"
    )
    features_full = {n: np.load(FEATURES_DIR / f"{n}.npy") for n in layer_names}

    keep = subsample_per_class(labels_full, args.samples_per_class, args.seed)
    labels_sub   = labels_full[keep]
    features_sub = {n: features_full[n][keep] for n in layer_names}

    L = len(layer_names)
    print(f"\nLayers: {layer_names}")
    print(f"Subsampled {len(keep)} points ({args.samples_per_class}/class)\n")

    # ── UMAP projections (cached) ──────────────────────────────────────────────
    raw_proj = load_or_compute_projections(
        features_sub, layer_names, UMAP_CACHE, force=args.no_cache
    )
    aligned = procrustes_align(raw_proj, layer_names)
    print("Procrustes-aligned to final layer.\n")

    # ── Decision boundary (fit on final-layer UMAP projection) ────────────────
    final_proj = aligned[layer_names[-1]]
    svc        = LinearSVC(dual="auto", max_iter=5000).fit(final_proj, labels_sub)
    w_final    = svc.coef_[0]
    b_final    = float(svc.intercept_[0])

    # Edge-on camera angle
    w_hat          = w_final / (np.linalg.norm(w_final) + 1e-10)
    edge_on_azim   = float(np.degrees(np.arctan2(w_hat[1], w_hat[0]))) + 90.0
    edge_on_elev   = 0.0

    print(f"Decision boundary fitted. Edge-on: elev={edge_on_elev:.1f}°, "
          f"azim={edge_on_azim:.1f}°\n")

    # ── Per-layer edge index sets ──────────────────────────────────────────────
    print("Building k-NN edge sets…")
    edge_pairs:  list[set]  = []
    edge_colors: list[dict] = []
    for name in layer_names:
        p, c = build_edge_index_pairs(aligned[name], labels_sub, args.knn_k)
        edge_pairs.append(p)
        edge_colors.append(c)
    print(f"  Edge counts per layer: {[len(ep) for ep in edge_pairs]}\n")

    # ── Camera azimuths at each layer (evenly spread 45°→225°) ────────────────
    layer_azimuths = [45.0 + i * (180.0 / max(L - 1, 1)) for i in range(L)]
    normal_elev    = 20.0

    # ── Build frame sequence ───────────────────────────────────────────────────
    # Each entry: (src_idx, dst_idx, t_ease, boundary_alpha, elev, azim)
    # t_ease=0 → fully at src; t_ease=1 → fully at dst
    frame_seq: list[tuple[int, int, float, float, float, float]] = []

    for layer_idx in range(L):
        azim   = layer_azimuths[layer_idx]
        is_first = layer_idx == 0
        is_last  = layer_idx == L - 1
        hold   = args.hold_frames if (is_first or is_last) else max(2, args.hold_frames // 2)
        b_alpha = 0.0

        for h in range(hold):
            # Fade in the boundary over the last-layer hold frames
            if is_last:
                b_alpha = _ease(h / max(hold - 1, 1)) * 0.35
            frame_seq.append((layer_idx, layer_idx, 0.0, b_alpha, normal_elev, azim))

        if not is_last:
            azim_next = layer_azimuths[layer_idx + 1]
            for f in range(args.interp_frames):
                t      = (f + 1) / (args.interp_frames + 1)
                te     = _ease(t)
                az_i   = azim + t * (azim_next - azim)   # linear azimuth sweep
                frame_seq.append((layer_idx, layer_idx + 1, te, 0.0, normal_elev, az_i))

    # Smooth camera sweep to edge-on view (boundary fully opaque)
    azim_start = layer_azimuths[-1]
    for f in range(args.rotation_frames):
        t    = f / max(args.rotation_frames - 1, 1)
        te   = _ease(t)
        azim = azim_start + te * (edge_on_azim - azim_start)
        elev = normal_elev + te * (edge_on_elev - normal_elev)
        frame_seq.append((L - 1, L - 1, 0.0, 0.35, elev, azim))

    # Hold at edge-on
    for _ in range(args.hold_frames):
        frame_seq.append((L - 1, L - 1, 0.0, 0.35, edge_on_elev, edge_on_azim))

    total_frames = len(frame_seq)
    duration_s   = total_frames / args.fps
    print(f"Frame sequence: {total_frames} frames at {args.fps} fps "
          f"≈ {duration_s:.1f} s\n")

    # ── Render ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(9, 7))
    gs  = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.05)
    ax  = fig.add_subplot(gs[0], projection="3d")
    cax = fig.add_subplot(gs[1])

    sm = plt.cm.ScalarMappable(cmap=CLASS_CMAP, norm=CLASS_NORM)
    cb = fig.colorbar(sm, cax=cax, ticks=range(len(CLASSES)))
    cb.set_ticklabels(CLASS_NAMES)

    def draw_frame(frame_i: int) -> None:
        ax.clear()
        src_i, dst_i, t_ease, b_alpha, elev, azim = frame_seq[frame_i]

        proj_src  = aligned[layer_names[src_i]]
        proj_dst  = aligned[layer_names[dst_i]]
        proj      = (1.0 - t_ease) * proj_src + t_ease * proj_dst

        # Scatter — morphed positions
        ax.scatter(
            proj[:, 0], proj[:, 1], proj[:, 2],
            c=labels_sub, cmap=CLASS_CMAP, norm=CLASS_NORM,
            s=5, lw=0, alpha=0.65, zorder=3,
        )

        # Edges — per-edge alpha fade
        segs, rgba = build_edge_frame(
            proj,
            edge_pairs[src_i], edge_pairs[dst_i],
            edge_colors[src_i], edge_colors[dst_i],
            t_ease,
        )
        if segs:
            lc = Line3DCollection(segs, linewidths=0.5, zorder=2)
            lc.set_color(rgba)
            ax.add_collection3d(lc)

        # Decision boundary (fades in on arrival at final layer)
        if b_alpha > 0.005:
            draw_boundary(ax, proj, w_final, b_final, surf_alpha=b_alpha)

        # Title
        if src_i == dst_i:
            name = layer_names[src_i]
            if elev < 5.0:
                title = f"Layer {src_i + 1}/{L}: {name}  [edge-on]"
            else:
                title = f"Layer {src_i + 1}/{L}: {name}"
        else:
            pct   = int(t_ease * 100)
            title = f"Layer {src_i + 1} → {dst_i + 1}  ({pct}%)"
        ax.set_title(title, fontsize=12)

        ax.set_xlabel("UMAP 1", fontsize=8, labelpad=0)
        ax.set_ylabel("UMAP 2", fontsize=8, labelpad=0)
        ax.set_zlabel("UMAP 3", fontsize=8, labelpad=0)
        ax.view_init(elev=elev, azim=azim)

    print("Rendering frames…")
    ani = animation.FuncAnimation(
        fig, draw_frame, frames=total_frames, interval=1000 // args.fps, repeat=True,
    )

    out_path = args.out or (OUTPUTS_DIR / f"smooth_manifold.{args.fmt}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.fmt == "mp4":
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=1800)
        ani.save(str(out_path), writer=writer, dpi=args.dpi)
    else:
        ani.save(str(out_path), writer="pillow", fps=args.fps, dpi=args.dpi)

    print(f"\nSaved → {out_path}  ({out_path.stat().st_size / 1024:.0f} kB)")
    plt.close(fig)


if __name__ == "__main__":
    main()
