"""Generate diagrams for the talk slides."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

C_U = "#1565C0"   # dark blue  — node u
C_V = "#B71C1C"   # dark red   — node v
C_TRANSPORT = "#555555"


def _node(ax, xy, color, label, r=0.16):
    ax.add_patch(plt.Circle(xy, r, color=color, zorder=6))
    ax.text(xy[0], xy[1], label, ha="center", va="center",
            fontsize=13, fontweight="bold", color="white", zorder=7)


def _small_node(ax, xy, color, r=0.10):
    ax.add_patch(plt.Circle(xy, r, color=color, alpha=0.85, zorder=5))


def _edge(ax, p1, p2, color="#333333", lw=2.0, alpha=1.0):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
            color=color, lw=lw, alpha=alpha, zorder=2)


def _transport_arrow(ax, p1, p2, lw=1.6, alpha=0.45):
    ann = ax.annotate(
        "", xy=p2, xytext=p1,
        arrowprops=dict(
            arrowstyle="->",
            color=C_TRANSPORT,
            lw=lw,
            connectionstyle="arc3,rad=0.18",
        ),
        zorder=4,
    )
    ann.arrow_patch.set_alpha(alpha)


def _neighborhood_cloud(ax, nbr_pts, color, mu_label, label_below=True):
    pts = np.array(nbr_pts)
    cx, cy = pts.mean(axis=0)
    r = np.linalg.norm(pts - [cx, cy], axis=1).max() + 0.38
    ax.add_patch(plt.Circle(
        (cx, cy), r,
        facecolor=color, edgecolor=color,
        alpha=0.10, linewidth=1.5, linestyle="--", zorder=1,
    ))
    y_off = -r - 0.14 if label_below else r + 0.14
    va = "top" if label_below else "bottom"
    ax.text(cx, cy + y_off, mu_label,
            ha="center", va=va, fontsize=12, color=color,
            style="italic", fontweight="bold")


def _w1_annotation(ax, u_nbrs, v_nbrs, text):
    uc = np.array(u_nbrs).mean(axis=0)
    vc = np.array(v_nbrs).mean(axis=0)
    # bracket arrow between the two cloud centroids
    ax.annotate("", xy=vc, xytext=uc,
                arrowprops=dict(arrowstyle="<->", color="#888888", lw=1.4),
                zorder=3)
    mid = (uc + vc) / 2
    ax.text(mid[0], mid[1] + 0.22, text,
            ha="center", va="bottom", fontsize=10, color="#555555")


# ---------------------------------------------------------------------------
# Slide 5 — Ollivier-Ricci curvature intuition
# ---------------------------------------------------------------------------

def make_slide5_ricci():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("white")

    U = np.array([-1.5, 0.0])
    V = np.array([ 1.5, 0.0])

    # ── Left: positive curvature — neighborhoods cluster toward each other ──
    ax = axes[0]
    ax.set_aspect("equal")
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-1.9, 1.9)
    ax.axis("off")
    ax.set_title("Positive curvature   κ > 0", fontsize=13,
                 fontweight="bold", pad=10, color="#1A237E")

    u_nbrs_hi = [(-0.5,  0.75), (-0.5, -0.75), (-1.0,  0.0)]
    v_nbrs_hi = [( 0.5,  0.75), ( 0.5, -0.75), ( 1.0,  0.0)]

    _edge(ax, U, V)
    for n in u_nbrs_hi:
        _edge(ax, U, n, color=C_U, lw=1.2, alpha=0.55)
    for n in v_nbrs_hi:
        _edge(ax, V, n, color=C_V, lw=1.2, alpha=0.55)

    _neighborhood_cloud(ax, u_nbrs_hi, C_U, "μᵤ", label_below=True)
    _neighborhood_cloud(ax, v_nbrs_hi, C_V, "μᵥ", label_below=True)

    for p1, p2 in zip(u_nbrs_hi, v_nbrs_hi):
        _transport_arrow(ax, p1, p2)

    for n in u_nbrs_hi:
        _small_node(ax, n, C_U)
    for n in v_nbrs_hi:
        _small_node(ax, n, C_V)

    _node(ax, U, C_U, "u")
    _node(ax, V, C_V, "v")

    _w1_annotation(ax, u_nbrs_hi, v_nbrs_hi, "$W_1$ small")

    ax.text(0, -1.75,
            r"$O(u,v) = 1 - W_1(\mu_u,\,\mu_v)\ \approx\ +0.6$",
            ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#E3F2FD", edgecolor="#90CAF9", lw=1.2))

    # ── Right: negative curvature — neighborhoods diverge away ──
    ax = axes[1]
    ax.set_aspect("equal")
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-1.9, 1.9)
    ax.axis("off")
    ax.set_title("Negative curvature   κ < 0", fontsize=13,
                 fontweight="bold", pad=10, color="#B71C1C")

    u_nbrs_lo = [(-2.4,  0.70), (-2.4, -0.70), (-1.5,  1.15)]
    v_nbrs_lo = [( 2.4,  0.70), ( 2.4, -0.70), ( 1.5,  1.15)]

    _edge(ax, U, V)
    for n in u_nbrs_lo:
        _edge(ax, U, n, color=C_U, lw=1.2, alpha=0.55)
    for n in v_nbrs_lo:
        _edge(ax, V, n, color=C_V, lw=1.2, alpha=0.55)

    _neighborhood_cloud(ax, u_nbrs_lo, C_U, "μᵤ", label_below=False)
    _neighborhood_cloud(ax, v_nbrs_lo, C_V, "μᵥ", label_below=False)

    for p1, p2 in zip(u_nbrs_lo, v_nbrs_lo):
        _transport_arrow(ax, p1, p2)

    for n in u_nbrs_lo:
        _small_node(ax, n, C_U)
    for n in v_nbrs_lo:
        _small_node(ax, n, C_V)

    _node(ax, U, C_U, "u")
    _node(ax, V, C_V, "v")

    _w1_annotation(ax, u_nbrs_lo, v_nbrs_lo, "$W_1$ large")

    ax.text(0, -1.75,
            r"$O(u,v) = 1 - W_1(\mu_u,\,\mu_v)\ \approx\ -0.4$",
            ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#FFEBEE", edgecolor="#EF9A9A", lw=1.2))

    fig.tight_layout(pad=1.5)
    out = FIGURES_DIR / "slide5_ollivier_ricci.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Forman vs Ollivier comparison
# ---------------------------------------------------------------------------
# Two example graphs:
#   A — clustered (many shared / close neighbours)
#       Forman: F = 4 − 4 − 4 = −4  (negative, because high degree)
#       Ollivier: O ≈ +0.7           (positive, neighbourhoods overlap)
#   B — diverging (tree-branch structure)
#       Forman: F = 4 − 2 − 2 =  0  (neutral)
#       Ollivier: O ≈ −0.3           (negative, neighbourhoods flee apart)
#
# Punchline: same topology, opposite signs → Forman misses geometry.
# ---------------------------------------------------------------------------

def _degree_badge(ax, xy, deg, color):
    """Small white badge beside a node showing its degree."""
    bx, by = xy[0] + 0.30, xy[1] + 0.35
    ax.text(bx, by, f"deg={deg}", ha="center", va="center",
            fontsize=9, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=color, lw=1.2), zorder=8)


def _result_box(ax, x, y, text, bg, border):
    ax.text(x, y, text, ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.45", facecolor=bg,
                      edgecolor=border, lw=1.4), zorder=8)


def _panel_setup(ax, xlim=(-3.2, 3.2), ylim=(-2.0, 2.0)):
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def _draw_example(ax, U, V, u_nbrs, v_nbrs,
                  show_degrees=False, show_clouds=False):
    """Draw the shared graph structure with optional Forman or Ollivier overlays."""
    # Central edge
    _edge(ax, U, V, lw=2.4)

    # Spoke edges
    for n in u_nbrs:
        _edge(ax, U, n, color=C_U, lw=1.3, alpha=0.55)
    for n in v_nbrs:
        _edge(ax, V, n, color=C_V, lw=1.3, alpha=0.55)

    # Neighbourhood clouds + transport arrows (Ollivier panels)
    if show_clouds:
        _neighborhood_cloud(ax, u_nbrs, C_U, "μᵤ", label_below=True)
        _neighborhood_cloud(ax, v_nbrs, C_V, "μᵥ", label_below=True)
        for p1, p2 in zip(u_nbrs, v_nbrs):
            _transport_arrow(ax, p1, p2)

    # Neighbour nodes
    for n in u_nbrs:
        _small_node(ax, n, C_U)
    for n in v_nbrs:
        _small_node(ax, n, C_V)

    # Main nodes
    _node(ax, U, C_U, "u")
    _node(ax, V, C_V, "v")

    # Degree badges (Forman panels)
    if show_degrees:
        deg_u = len(u_nbrs) + 1   # +1 for the u–v edge itself
        deg_v = len(v_nbrs) + 1
        _degree_badge(ax, U, deg_u, C_U)
        _degree_badge(ax, V, deg_v, C_V)


def make_forman_ollivier_comparison():
    U = np.array([-1.1, 0.0])
    V = np.array([ 1.1, 0.0])

    # Example A — clustered: u & v each have 3 extra neighbours near the centre
    u_A = [(-0.2,  0.85), (-0.2, -0.85), ( 0.0,  0.52)]
    v_A = [( 0.2,  0.85), ( 0.2, -0.85), ( 0.0,  0.52)]
    # Note: third neighbour is shared → W₁ very small for that pair

    # Example B — diverging: one branch each, pointing away
    u_B = [(-2.4, 0.0)]
    v_B = [( 2.4, 0.0)]

    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor("white")

    # Column header strip
    fig.text(0.30, 0.97,
             "Forman-Ricci\n"
             r"$F(u,v) = 4 - \deg(u) - \deg(v)$",
             ha="center", va="top", fontsize=12, fontweight="bold",
             color="#2E7D32",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F1F8E9",
                       edgecolor="#A5D6A7", lw=1.2))
    fig.text(0.72, 0.97,
             "Ollivier-Ricci\n"
             r"$O(u,v) = 1 - W_1(\mu_u,\,\mu_v)$",
             ha="center", va="top", fontsize=12, fontweight="bold",
             color="#1565C0",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#E3F2FD",
                       edgecolor="#90CAF9", lw=1.2))

    # Row labels
    fig.text(0.01, 0.73, "Example A\n(clustered\nneighbours)",
             ha="left", va="center", fontsize=10, color="#444444",
             style="italic")
    fig.text(0.01, 0.27, "Example B\n(diverging\nneighbours)",
             ha="left", va="center", fontsize=10, color="#444444",
             style="italic")

    gs = fig.add_gridspec(2, 2, left=0.08, right=0.97,
                          top=0.88, bottom=0.05,
                          hspace=0.18, wspace=0.08)

    # ── A / Forman ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    _panel_setup(ax)
    ax.set_facecolor("#F9FBE7")
    _draw_example(ax, U, V, u_A, v_A, show_degrees=True, show_clouds=False)
    ax.text(0, 1.75, "high degree → large penalty", ha="center", va="center",
            fontsize=9.5, color="#555")
    _result_box(ax, 0, -1.75,
                r"$F = 4 - 4 - 4 = -4$   ✗ negative",
                "#FFEBEE", "#EF9A9A")

    # ── A / Ollivier ──────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    _panel_setup(ax)
    ax.set_facecolor("#EFF8FF")
    _draw_example(ax, U, V, u_A, v_A, show_degrees=False, show_clouds=True)
    ax.text(0, 1.75, "neighbourhoods overlap → cheap transport",
            ha="center", va="center", fontsize=9.5, color="#555")
    _result_box(ax, 0, -1.75,
                r"$O \approx +0.7$   ✓ positive",
                "#E8F5E9", "#A5D6A7")

    # Disagreement annotation between A panels
    fig.text(0.515, 0.73, "≠", ha="center", va="center",
             fontsize=22, color="#E53935", fontweight="bold")

    # ── B / Forman ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    _panel_setup(ax)
    ax.set_facecolor("#F9FBE7")
    _draw_example(ax, U, V, u_B, v_B, show_degrees=True, show_clouds=False)
    ax.text(0, 1.75, "low degree → small penalty", ha="center", va="center",
            fontsize=9.5, color="#555")
    _result_box(ax, 0, -1.75,
                r"$F = 4 - 2 - 2 = 0$   ~ neutral",
                "#F5F5F5", "#BDBDBD")

    # ── B / Ollivier ──────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    _panel_setup(ax)
    ax.set_facecolor("#EFF8FF")
    _draw_example(ax, U, V, u_B, v_B, show_degrees=False, show_clouds=True)
    ax.text(0, 1.75, "neighbourhoods far apart → expensive transport",
            ha="center", va="center", fontsize=9.5, color="#555")
    _result_box(ax, 0, -1.75,
                r"$O \approx -0.3$   ✗ negative",
                "#FFEBEE", "#EF9A9A")

    # Disagreement annotation between B panels
    fig.text(0.515, 0.27, "≠", ha="center", va="center",
             fontsize=22, color="#E53935", fontweight="bold")

    # Bottom caption
    fig.text(0.52, 0.01,
             "Same topology — opposite geometric meaning.  "
             "Forman captures connectivity; Ollivier captures spatial geometry.",
             ha="center", va="bottom", fontsize=10, color="#444",
             style="italic")

    out = FIGURES_DIR / "forman_vs_ollivier.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Slide 8 — Experimental setup
# ---------------------------------------------------------------------------

def make_slide8_setup():
    import sys
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT / "src"))

    adv_dir = ROOT / "outputs" / "adversarial" / "eps_0.030"
    images = np.load(adv_dir / "clean_images.npy")   # (400, 1, 28, 28) in [0,1]
    labels = np.load(adv_dir / "labels.npy")          # 0=Pullover, 1=Shirt

    # One representative example of each class
    img_pull = images[np.where(labels == 0)[0][3], 0]
    img_shirt = images[np.where(labels == 1)[0][5], 0]

    # ── Architecture data ────────────────────────────────────────────────────
    LAYERS = [
        ("input\n784",     784, "#E0E0E0", "#9E9E9E", False),
        ("relu₁\n512",     512, "#FFF3E0", "#FF9800", True),
        ("relu₂\n512",     512, "#FFF3E0", "#FF9800", True),
        ("relu₃\n256",     256, "#FFF3E0", "#FF9800", True),
        ("relu₄\n256",     256, "#FFF3E0", "#FF9800", True),
        ("relu₅\n128",     128, "#FFF3E0", "#FF9800", True),
        ("relu₆\n64",       64, "#FFF3E0", "#FF9800", True),
        ("output\n2",        2, "#E3F2FD", "#1565C0", False),
    ]

    fig = plt.figure(figsize=(15, 5.5))
    fig.patch.set_facecolor("white")

    # ── Left: example images ─────────────────────────────────────────────────
    ax_pull  = fig.add_axes([0.01, 0.52, 0.08, 0.38])
    ax_shirt = fig.add_axes([0.01, 0.10, 0.08, 0.38])

    for ax, img, cls, color in [
        (ax_pull,  img_pull,  "Pullover", C_U),
        (ax_shirt, img_shirt, "Shirt",    C_V),
    ]:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2.5)
        ax.set_title(cls, fontsize=10, color=color, fontweight="bold", pad=3)

    # Arrow from images into network
    fig.text(0.095, 0.50, "→", ha="center", va="center",
             fontsize=20, color="#555")

    # ── Main axis: network diagram ────────────────────────────────────────────
    ax = fig.add_axes([0.10, 0.00, 0.89, 1.00])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Positions
    xs = np.linspace(0.6, 9.4, len(LAYERS))
    MAX_LOG = np.log2(784)
    BOX_W = 0.55
    CENTER_Y = 3.2

    for i, (label, dim, fc, ec, extract) in enumerate(LAYERS):
        h = (np.log2(max(dim, 2)) / MAX_LOG) * 2.5
        y0 = CENTER_Y - h / 2
        x0 = xs[i] - BOX_W / 2

        box = mpatches.FancyBboxPatch(
            (x0, y0), BOX_W, h,
            boxstyle="round,pad=0.05",
            facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3,
        )
        ax.add_patch(box)

        ax.text(xs[i], CENTER_Y, label,
                ha="center", va="center", fontsize=8.5,
                color="#333", zorder=4, linespacing=1.4)

        # Forward arrow to next layer
        if i < len(LAYERS) - 1:
            x_arrow_start = xs[i] + BOX_W / 2
            x_arrow_end   = xs[i + 1] - BOX_W / 2
            ax.annotate("", xy=(x_arrow_end, CENTER_Y),
                        xytext=(x_arrow_start, CENTER_Y),
                        arrowprops=dict(arrowstyle="->", color="#555",
                                        lw=1.4), zorder=2)

        # Downward extraction arrow for relu layers
        if extract:
            ax.annotate("", xy=(xs[i], 1.1),
                        xytext=(xs[i], y0),
                        arrowprops=dict(arrowstyle="->", color="#FF9800",
                                        lw=1.3, linestyle="dashed"), zorder=2)

    # ── Analysis pipeline below the relu extraction arrows ────────────────────
    # Connector bar linking all 6 extraction arrows
    x_first = xs[1]   # relu1
    x_last  = xs[6]   # relu6
    ax.plot([x_first, x_last], [1.1, 1.1], color="#FF9800", lw=1.4,
            linestyle="dashed", zorder=2)

    # k-NN box
    knn_x = (x_first + x_last) / 2
    knn_box = mpatches.FancyBboxPatch(
        (knn_x - 1.0, 0.38), 2.0, 0.56,
        boxstyle="round,pad=0.05",
        facecolor="#F3E5F5", edgecolor="#7B1FA2", linewidth=1.6, zorder=3,
    )
    ax.add_patch(knn_box)
    ax.annotate("", xy=(knn_x, 0.94), xytext=(knn_x, 1.10),
                arrowprops=dict(arrowstyle="->", color="#FF9800", lw=1.3), zorder=2)
    ax.text(knn_x, 0.66,
            "symmetric k-NN graph  (k = 6)\nOllivier / Forman curvature  ·  ρ(x)  ·  Q",
            ha="center", va="center", fontsize=9, color="#4A148C", zorder=4)

    # UMAP + Procrustes note on the right
    ax.text(9.6, 0.66,
            "UMAP 3-D\n(Procrustes-\naligned)",
            ha="center", va="center", fontsize=9, color="#1565C0",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#E3F2FD",
                      edgecolor="#1565C0", lw=1.4), zorder=4)
    ax.annotate("", xy=(9.25, 0.66), xytext=(knn_x + 1.0, 0.66),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.3), zorder=2)

    # Subsample note
    ax.text(5.0, 4.75,
            "200 samples / class  (400 total)  ·  "
            "Fashion-MNIST: Pullover (class 2)  vs  Shirt (class 6)",
            ha="center", va="center", fontsize=10, color="#444",
            style="italic")

    out = FIGURES_DIR / "slide8_setup.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Slide 12 — ρ(x) distribution
# ---------------------------------------------------------------------------

def make_slide12_rho():
    from scipy.stats import gaussian_kde, norm

    ROOT = Path(__file__).resolve().parent.parent
    data = np.load(ROOT / "outputs" / "ricci_metrics.npz", allow_pickle=True)
    rho = data["rho"].astype(float)
    rho = rho[np.isfinite(rho)]

    mean_rho   = float(rho.mean())
    median_rho = float(np.median(rho))
    frac_neg   = float((rho < 0).mean())

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("white")

    xs = np.linspace(-1.05, 1.05, 400)

    # Filled histogram
    n, bins, patches = ax.hist(
        rho, bins=30, density=True,
        color="#90CAF9", edgecolor="white", linewidth=0.6,
        alpha=0.6, zorder=2, label="trained network (ours)",
    )
    # Colour bars left of zero differently
    for patch, left in zip(patches, bins[:-1]):
        if left < 0:
            patch.set_facecolor("#1565C0")
            patch.set_alpha(0.55)
        else:
            patch.set_facecolor("#EF9A9A")
            patch.set_alpha(0.45)

    # KDE over trained distribution
    kde = gaussian_kde(rho, bw_method=0.25)
    ax.plot(xs, kde(xs), color="#0D47A1", lw=2.5, zorder=4,
            label="KDE (trained)")

    # Reference: random network ≈ N(0, σ_random)
    # Pearson of L−1=5 i.i.d. pairs → σ ≈ 1/√(L−2) = 1/√4 = 0.5
    sigma_rand = 0.50
    ax.plot(xs, norm.pdf(xs, 0, sigma_rand), color="#888", lw=2.0,
            linestyle="--", zorder=3,
            label=r"random network  $\mathcal{N}(0,\,0.5)$")

    # Zero line
    ax.axvline(0, color="#333", lw=1.3, linestyle=":", zorder=3)
    ax.text(0.02, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 1,
            "ρ = 0", ha="left", va="top", fontsize=10, color="#333")

    # Shade ρ < 0 region under KDE
    xs_neg = xs[xs <= 0]
    ax.fill_between(xs_neg, kde(xs_neg), alpha=0.12, color="#1565C0", zorder=1)

    # Annotations: mean and median lines
    ax.axvline(mean_rho,   color="#0D47A1", lw=1.5, linestyle="-.",  zorder=3)
    ax.axvline(median_rho, color="#1565C0", lw=1.5, linestyle=(0,(3,1,1,1)), zorder=3)

    # Stats box (top-left or top-right depending on distribution shape)
    stats_text = (
        f"mean ρ  = {mean_rho:+.3f}\n"
        f"median ρ = {median_rho:+.3f}\n"
        f"{frac_neg:.1%} of vertices  ρ < 0\n\n"
        f"paper range: 73–98%"
    )
    ax.text(0.97, 0.97, stats_text,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD",
                      edgecolor="#90CAF9", lw=1.3))

    ax.set_xlabel("ρ(x)", fontsize=13)
    ax.set_ylabel("density", fontsize=12)
    ax.set_title(
        r"Local Ricci Evolution Coefficient  $\rho(x)$  —  trained network",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(-1.05, 1.05)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = FIGURES_DIR / "slide12_rho_distribution.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Slide 13 — r_layer vs ρ(x): two views of the same phenomenon
# ---------------------------------------------------------------------------

def _compute_r_layer_from_features(root):
    """Re-derive r_layer[ℓ] from saved features + vertex_ollivier in npz.

    Matches the subsampling (seed=42, 200/class) used by compute_metrics.py.
    """
    from sklearn.neighbors import kneighbors_graph

    KNN_K = 6
    N_PER_CLASS = 200
    SEED = 42

    feat_dir = root / "features" / "full"
    labels_full = np.load(feat_dir / "labels.npy")
    layer_files = sorted(p for p in feat_dir.iterdir() if p.name != "labels.npy")
    features_full = {p.stem: np.load(p) for p in layer_files}
    layer_names = list(features_full.keys())

    # Reproduce the subsample
    rng = np.random.default_rng(SEED)
    keep = []
    for cls in np.unique(labels_full):
        idx = np.where(labels_full == cls)[0]
        chosen = rng.choice(idx, size=min(N_PER_CLASS, len(idx)), replace=False)
        keep.extend(chosen.tolist())
    keep = np.array(sorted(keep))

    feats = {n: features_full[n][keep] for n in layer_names}
    N = len(keep)
    L = len(layer_names)

    # Load O_ℓ(x) from the saved npz (L, N)
    npz = np.load(root / "outputs" / "ricci_metrics.npz", allow_pickle=True)
    vo = npz["vertex_ollivier"]  # (L, N)

    # Build symmetric k-NN adjacency for each layer
    adjs = []
    for name in layer_names:
        g = kneighbors_graph(feats[name], n_neighbors=KNN_K,
                             mode="connectivity", include_self=False)
        adj = (g + g.T)
        adj.data[:] = 1.0
        adjs.append(adj.toarray())

    # Compute η_ℓ(x): mean distance change to layer-ℓ neighbours
    eta = np.zeros((N, L - 1))
    for ell in range(L - 1):
        pts_l  = feats[layer_names[ell]]
        pts_l1 = feats[layer_names[ell + 1]]
        adj_arr = adjs[ell]
        for x in range(N):
            nbrs = np.where(adj_arr[x] > 0)[0]
            if len(nbrs) == 0:
                continue
            d_l  = np.linalg.norm(pts_l[x]  - pts_l[nbrs],  axis=1)
            d_l1 = np.linalg.norm(pts_l1[x] - pts_l1[nbrs], axis=1)
            eta[x, ell] = float(np.mean(d_l1 - d_l))

    # r_layer[ℓ] = Pearson(O_ℓ, η_ℓ) across vertices
    r_layer = np.full(L - 1, np.nan)
    for ell in range(L - 1):
        o = vo[ell]
        e = eta[:, ell]
        mask = np.isfinite(o) & np.isfinite(e)
        if mask.sum() > 2 and np.std(o[mask]) > 1e-10 and np.std(e[mask]) > 1e-10:
            r_layer[ell] = float(np.corrcoef(o[mask], e[mask])[0, 1])

    return r_layer, layer_names


def make_slide13_rlayer():
    from scipy.stats import gaussian_kde

    ROOT = Path(__file__).resolve().parent.parent
    npz  = np.load(ROOT / "outputs" / "ricci_metrics.npz", allow_pickle=True)
    rho  = npz["rho"].astype(float)
    layer_names = list(npz["layer_names"])
    L = len(layer_names)

    print("  computing r_layer (builds k-NN graphs + η from saved features)…")
    r_layer, _ = _compute_r_layer_from_features(ROOT)
    print(f"  r_layer = {np.round(r_layer, 3)}")

    trans_labels = [f"L{i+1}→L{i+2}" for i in range(L - 1)]
    xs = np.arange(L - 1)

    # ── Figure: two panels ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw={"width_ratios": [1.1, 1.6]})
    fig.patch.set_facecolor("white")

    # ── Left: schematic matrix showing the two views ─────────────────────────
    ax = axes[0]
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-1.2, 7.5)
    ax.axis("off")
    ax.set_title("Two views of the same data", fontsize=11,
                 fontweight="bold", pad=8)

    NR, NC = 6, 5     # rows = sampled vertices, cols = layer transitions
    cell  = 0.75
    C_NEG = "#90CAF9"
    C_POS = "#FFCDD2"

    # Heatmap cells (stylised — alternating sign to suggest mixed but neg-dominant)
    signs = np.array([
        [-1,-1,-1,-1,-1],
        [-1,+1,-1,-1,-1],
        [-1,-1,-1,+1,-1],
        [-1,-1,-1,-1,+1],
        [-1,-1,+1,-1,-1],
        [-1,-1,-1,-1,-1],
    ])
    for r in range(NR):
        for c in range(NC):
            color = C_NEG if signs[r, c] < 0 else C_POS
            rect = mpatches.FancyBboxPatch(
                (c * cell + 0.04, r * cell + 0.04),
                cell - 0.08, cell - 0.08,
                boxstyle="round,pad=0.04",
                facecolor=color, edgecolor="white", lw=1.0, zorder=2,
            )
            ax.add_patch(rect)

    # Column labels (layer transitions)
    for c in range(NC):
        ax.text(c * cell + cell/2, NR * cell + 0.15,
                f"ℓ{c+1}", ha="center", va="bottom",
                fontsize=8.5, color="#555")

    # Row labels (vertices)
    for r in range(NR):
        ax.text(-0.22, r * cell + cell/2,
                f"x{r+1}", ha="right", va="center",
                fontsize=8.5, color="#555")

    # Axis label above columns
    ax.text(NC * cell / 2, NR * cell + 0.55,
            "layer transition ℓ", ha="center", va="bottom",
            fontsize=9, color="#333", style="italic")

    # Axis label left of rows
    ax.text(-0.55, NR * cell / 2,
            "vertex  x", ha="center", va="center",
            fontsize=9, color="#333", style="italic",
            rotation=90)

    # ρ(x) arrow — along a row
    row_y = 4 * cell + cell/2
    ax.annotate("", xy=(NC * cell + 0.55, row_y),
                xytext=(0, row_y),
                arrowprops=dict(arrowstyle="->", color="#0D47A1", lw=1.8))
    ax.text(NC * cell + 0.62, row_y,
            r"$\rho(x)$", ha="left", va="center",
            fontsize=11, color="#0D47A1", fontweight="bold")
    ax.text(NC * cell + 0.62, row_y - 0.32,
            "per vertex\nacross layers", ha="left", va="top",
            fontsize=8, color="#0D47A1")

    # r_layer arrow — along a column
    col_x = 2 * cell + cell/2
    ax.annotate("", xy=(col_x, -0.72),
                xytext=(col_x, 0),
                arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1.8))
    ax.text(col_x, -0.80,
            r"$r_\ell$", ha="center", va="top",
            fontsize=11, color="#B71C1C", fontweight="bold")
    ax.text(col_x, -1.05,
            "per layer\nacross vertices", ha="center", va="top",
            fontsize=8, color="#B71C1C")

    # ── Right: actual data — r_layer bars + ρ(x) KDE ────────────────────────
    ax2 = axes[1]

    # r_layer bars
    bar_colors = ["#1565C0" if v < 0 else "#C62828" for v in r_layer]
    bars = ax2.bar(xs, r_layer, color=bar_colors, alpha=0.75,
                   width=0.5, zorder=3, label=r"$r_\ell$ (per-layer Pearson)")
    for bar, val in zip(bars, r_layer):
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     val + (0.03 if val > 0 else -0.05),
                     f"{val:.2f}",
                     ha="center", va="bottom" if val > 0 else "top",
                     fontsize=9, color="white" if abs(val) > 0.25 else "#333",
                     fontweight="bold")

    # ρ(x) KDE as a shaded curve on a secondary x-axis (rotated)
    # Show as a simple rug/violin on the side instead
    kde = gaussian_kde(rho, bw_method=0.25)
    rho_xs = np.linspace(-1, 1, 300)
    kde_vals = kde(rho_xs)
    # Normalise to fit as a thin band on the right
    ax2_twin = ax2.twinx()
    ax2_twin.plot(rho_xs, kde_vals, color="#0D47A1", lw=0, alpha=0)
    ax2_twin.set_visible(False)

    # Just annotate mean ρ as a horizontal dashed reference
    mean_rho = float(rho.mean())
    ax2.axhline(mean_rho, color="#0D47A1", lw=1.5, linestyle="--",
                alpha=0.7, zorder=2, label=f"mean ρ(x) = {mean_rho:.3f}")

    ax2.axhline(0, color="#333", lw=1.0, linestyle=":", zorder=2)

    ax2.set_xticks(xs)
    ax2.set_xticklabels(trans_labels, fontsize=10)
    ax2.set_ylabel("Pearson correlation", fontsize=11)
    ax2.set_ylim(-1.05, 0.65)
    ax2.set_title(r"$r_\ell$ at each layer transition  (vs  mean $\rho(x)$)",
                  fontsize=11, fontweight="bold", pad=8)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3, axis="y")

    # Annotation explaining the dashed line
    ax2.text(xs[-1] + 0.05, mean_rho + 0.04,
             r"mean $\rho(x)$", ha="left", va="bottom",
             fontsize=9, color="#0D47A1", style="italic")

    fig.tight_layout(pad=1.5)
    out = FIGURES_DIR / "slide13_rlayer.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    make_slide5_ricci()
    make_forman_ollivier_comparison()
    make_slide8_setup()
    make_slide12_rho()
    make_slide13_rlayer()
