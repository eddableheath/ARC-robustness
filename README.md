# Neural Feature Geometry and Adversarial Robustness

A research implementation recreating and extending [arXiv:2509.22362](https://arxiv.org/abs/2509.22362) — *"Neural Feature Geometry Evolves as Discrete Ricci Flow"* — with a focus on adversarial robustness.

The paper shows that trained neural networks evolve their internal representations in a way that mirrors discrete Ricci flow: curvature and distance changes are anti-correlated layer-by-layer, concentrating class structure. This codebase reproduces that finding on a binary Fashion-MNIST task and adds an original adversarial analysis: do FGSM perturbations disrupt the Ricci-flow signature?

---

## Task

Binary classification on Fashion-MNIST: **Pullover** (class 2) vs **Shirt** (class 6).

Model: 6-layer fully-connected ReLU MLP trained from scratch. Post-ReLU activations at each hidden layer (`relu1`–`relu6`) are the objects of analysis.

---

## Key concepts

**Discrete Ricci curvature** — two notions are computed on the k-NN graph built on layer activations:

- **Ollivier-Ricci** `O(u,v) = 1 − W₁(μ_u, μ_v)` — earth-mover distance between neighbour distributions; slow but geometrically principled.
- **Forman-Ricci** `F(u,v) = 4 − deg(u) − deg(v)` — combinatorial approximation; fast.

**Community metrics** (per layer, matching Appendix A.2.3 of the paper):

- Modularity `Q` — how well the k-NN graph clusters by class
- Normalised Cut — cross-class edge fraction
- Algebraic Connectivity (Fiedler value)
- Curvature Gap `ΔO` — mean curvature difference between same-class and cross-class edges

**Local Ricci Evolution Coefficient ρ(x)** (§3 of the paper) — per vertex, the Pearson correlation across layers between:

- `η_ℓ(x)` — mean change in Euclidean distance to layer-ℓ neighbours
- `O_ℓ(x)` — mean Ollivier curvature on incident edges at layer ℓ

`ρ(x) < 0` means positive-curvature regions contract and negative-curvature regions expand — the hallmark of Ricci flow. The paper reports 73–98% of vertices with `ρ < 0` in trained networks.

`r_layer[ℓ]` is the per-layer analogue: Pearson(`O_ℓ(x)`, `η_ℓ(x)`) across vertices at layer ℓ.

**FGSM adversarial attack** — untargeted single-step Fast Gradient Sign Method:

```text
x_adv = clip(x_raw + ε · sign(∂L/∂x_norm), 0, 1)
```

ε is in raw [0, 1] pixel space; the gradient is computed w.r.t. the normalised input (mean=0.2860, std=0.3530).

---

## Project structure

```text
arc-robustness/
├── src/arc_robustness/
│   ├── analysis/
│   │   ├── graph_utils.py     symmetric k-NN graph construction
│   │   ├── ricci.py           Forman, Augmented Forman, Ollivier-Ricci
│   │   ├── community.py       Q, NCut, Fiedler, curvature gap
│   │   └── evolution.py       ρ(x) and r_layer
│   ├── training/
│   │   ├── model.py           model definition + shared constants
│   │   └── train_base_model.py
│   └── visualisation/
│       ├── extract_features.py   post-ReLU activation extraction
│       └── visualise.py          UMAP wrapper
│
├── scripts/
│   ├── run_training.py            train the base model
│   ├── visualise_manifold.py      3-D UMAP animation (standalone)
│   ├── compute_metrics.py         full Ricci/community pipeline → outputs/ricci_metrics.npz
│   ├── visualise_metrics.py       community figure + combined manifold GIF + adversarial Ricci figure
│   └── generate_adversarial.py    FGSM attack + feature extraction
│
├── outputs/
│   ├── ricci_metrics.npz
│   ├── community_metrics.png
│   ├── combined_manifold.gif
│   ├── adversarial_ricci_eps{ε}.png
│   └── adversarial/
│       └── eps_{ε}/
│           ├── clean_images.npy / adv_images.npy / labels.npy
│           └── relu1.npy … relu6.npy
│
├── features/full/                 cached clean activations
└── weights/base_model.pt
```

---

## Pipeline

### 1. Train

```bash
uv run python scripts/run_training.py
```

Trains the 6-layer MLP on Fashion-MNIST (Pullover vs Shirt) and saves `weights/base_model.pt`.

### 2. Compute Ricci metrics

```bash
uv run python scripts/compute_metrics.py
# faster (Forman only, no ρ(x)):
uv run python scripts/compute_metrics.py --skip-ollivier
```

Builds a symmetric k-NN graph on each layer's activations, computes all curvature and community metrics, runs ρ(x), fits 3-D UMAP projections (Procrustes-aligned to the final layer), and saves everything to `outputs/ricci_metrics.npz`.

Key options: `--knn-k K` (default 6), `--samples-per-class N` (default 200), `--seed`.

### 3. Visualise

```bash
# Community metrics figure + combined manifold GIF:
uv run python scripts/visualise_metrics.py

# GIF with adversarial overlay (requires step 4 first):
uv run python scripts/visualise_metrics.py --adv-epsilon 0.03

# Adversarial Ricci comparison figure:
uv run python scripts/visualise_metrics.py --adv-epsilon 0.03 --ricci-analysis

# Fast Forman-only Ricci comparison (skips Ollivier, no ρ(x)):
uv run python scripts/visualise_metrics.py --adv-epsilon 0.03 --ricci-analysis --skip-ollivier
```

**`community_metrics.png`** — four-panel static figure: Q, NCut, Fiedler, ΔO per layer.

**`combined_manifold.gif`** — animated GIF with three panels: 3-D UMAP manifold (class colours, k-NN graph edges, linear decision boundary on hold frames, edge-on rotation at end), growing mean κ̄ line plot, growing r_layer line plot.

**`adversarial_ricci_eps{ε}.png`** — four-panel comparison (clean vs adversarial): mean κ̄, curvature gap ΔO, modularity Q per layer, and ρ(x) distribution violin.

### 4. Generate adversarial examples

```bash
# Default ε=0.03, visual comparison grid + features for all standard ε values:
uv run python scripts/generate_adversarial.py

# Custom primary ε:
uv run python scripts/generate_adversarial.py --epsilon 0.1
```

Saves per-ε directories under `outputs/adversarial/eps_{ε:.3f}/` containing clean and adversarial images, labels (remapped 0/1), and per-layer post-ReLU activations for the adversarial inputs.

Also saves `comparison_grid.png` and `attack_summary.txt` directly under `outputs/adversarial/`.

---

## Setup

```bash
# Install dependencies (requires uv)
uv sync

# Or with pip:
pip install -e .
```

Requires Python ≥ 3.11. PyTorch is pulled from PyPI; the default macOS wheels include MPS support. For CUDA, add a custom index in `pyproject.toml` pointing at `https://download.pytorch.org/whl/cuXXX`.

---

## Notes on performance

- Ollivier-Ricci uses the POT library's earth-mover distance solver. For 400 points × 6 layers it takes a few minutes on CPU; for large `N` or `K`, `--skip-ollivier` is a useful shortcut.
- All scripts auto-detect MPS (Apple Silicon) or CPU; CUDA support is present but untested.
- UMAP is run at `random_state=42` with `n_components=3`. Procrustes alignment is applied to all layers so inter-frame motion in the GIF reflects genuine topology change rather than arbitrary orientation drift.
