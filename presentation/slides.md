# Neural Feature Geometry and Adversarial Robustness
## Talk slides — ~60 minutes, ML/AI research group

Each slide entry has: **SLIDE** (what appears on screen), **NOTES** (what to say).

---

## Section 1 — Motivation (5 min, 2 slides)

---

### Slide 1 — Title

**SLIDE**
```
Neural Feature Geometry and Adversarial Robustness

[your name]
[date / affiliation]

Based on: arXiv:2509.22362 (Gong et al., 2025)
+ original adversarial extension
```

**NOTES**
This talk is about a geometric lens on how neural networks build their internal representations, and what that lens might tell us about adversarial robustness. The foundation is a recent paper from 2025 that I've been reproducing and extending. I'll walk through the paper's main ideas, our own results, and then spend the second half on an open research question: does adversarial perturbation disrupt the geometric signature that the paper identifies?

---

### Slide 2 — The robustness problem, geometrically

**SLIDE**
```
The robustness problem

• Small perturbations → large prediction changes
• Classifiers learn decision surfaces, not "concepts"
• Gap: why do good representations fail this way?

A geometric question:
  What is the internal structure of a well-trained network's
  feature space — and what does adversarial attack do to it?
```

**NOTES**
We all know the empirical story: you can take an image, add imperceptible noise, and fool a classifier with high confidence. The common framing is that networks learn spurious correlations or overfit texture. But there's a more fundamental geometric question: what does a well-trained network's internal representation space *look like*, how does it evolve layer by layer, and does adversarial perturbation disrupt that structure in a systematic way? That's the question we're going to pursue today.

---

## Section 2 — The paper: discrete Ricci flow in neural networks (12 min, 5 slides)

---

### Slide 3 — The core finding

**SLIDE**
```
arXiv:2509.22362 — Gong et al. 2025

"Neural Feature Geometry Evolves as Discrete Ricci Flow"

Main claim:
  Trained neural networks evolve internal representations
  layer by layer in a way that mirrors discrete Ricci flow.

Evidence:
  • 73–98% of vertices have ρ(x) < 0 across architectures
  • Holds for image classifiers, not random/untrained networks
  • Consistent across VGG, ResNet, ViT
```

**NOTES**
The paper's central claim is that the geometry of a neural network's feature space evolves across layers in a way that's formally analogous to discrete Ricci flow — a process from differential geometry where positive-curvature regions contract and negative-curvature regions expand, causing the manifold to organise. The evidence is a per-vertex coefficient ρ(x) — I'll define this properly in a moment — that is negative for the vast majority of vertices in trained networks but not in random ones. This is the key statistic we're reproducing and then testing under adversarial attack.

---

### Slide 4 — Building the feature graph

**SLIDE**
```
Feature graph construction (per layer ℓ):

1. Extract post-ReLU activations {fₗ(x)} for all inputs x
2. Build a symmetric k-NN graph G_ℓ on those activations
   — nodes = data points, edges = k nearest neighbours
3. Measure the geometry of G_ℓ using discrete curvature

For each layer: one graph, one geometric snapshot
Across layers: a sequence of graphs → evolution
```

**NOTES**
The key object of study is a graph built on the activations at each layer. Take all your input points, push them through the network up to layer ℓ, and build a k-nearest-neighbour graph in that activation space — nodes are data points, edges connect each point to its k closest neighbours. Do this at every layer and you get a sequence of graphs. The paper then asks: how does the geometry of this sequence evolve?

---

### Slide 5 — Discrete Ricci curvature (intuition)

**SLIDE**
```
Ricci curvature — intuition

Riemannian:  positive κ → geodesics converge (sphere)
             negative κ → geodesics diverge (saddle)

Discrete (per edge u–v):
  κ(u,v) measures how "close together" the neighbours
  of u are compared to the neighbours of v

Positive κ: neighbours cluster together → class cohesion
Negative κ: neighbours spread out → across-class mixing
```
*[figures/slide5_ollivier_ricci.png]*

**NOTES**
Before we define things precisely, the intuition. In Riemannian geometry, Ricci curvature on a manifold measures whether nearby geodesics converge or diverge. Positive curvature — like on a sphere — means geodesics focus. Negative curvature — like a saddle — means they spread apart. The discrete analogue attaches a curvature to each *edge* u–v in the graph by asking: how close are the neighbourhoods of u and v to each other? If the neighbourhood of u and the neighbourhood of v largely overlap, curvature is positive. If they're in different parts of the graph, curvature is negative. In a neural network context, positive curvature on an edge means the two points are sitting in a locally cohesive cluster; negative curvature means they're bridging different regions — potentially different classes.

---

### Slide 6 — Two discrete curvature notions

**SLIDE**
```
[figures/forman_vs_ollivier.png]
```

**NOTES**
We use two notions of discrete curvature. Ollivier-Ricci is the principled one: for an edge u–v, define a probability measure on each node's neighbourhood — the uniform distribution over its k neighbours — and measure the Wasserstein-1 distance between those distributions. Subtract from 1 and you get the curvature. Close neighbourhoods means small transport cost means high curvature. This is geometrically well-motivated but requires solving an optimal transport problem per edge, so it's slow. Forman-Ricci is a purely combinatorial approximation that only uses node degrees — it's fast but less sensitive. For the key statistic ρ(x) we need Ollivier; for quick exploration Forman is a useful proxy.

---

### Slide 7 — The local Ricci evolution coefficient ρ(x)

**SLIDE**
```
ρ(x) — per vertex, across layers ℓ = 1…L-1

Two layer-by-layer signals:

  η_ℓ(x) = mean change in distance to layer-ℓ neighbours
            (how much does x's neighbourhood contract/expand?)

  O_ℓ(x) = mean Ollivier curvature on edges incident to x

ρ(x) = Pearson(η(x), O(x))  [across layers]

ρ < 0: high-curvature regions contract, low-curvature expand
      → Ricci flow signature

Paper: 73–98% of vertices have ρ < 0 in trained networks
```

**NOTES**
Now for the key quantity. For each data point x, we track two signals across layers. η_ℓ(x) measures how much x's neighbourhood contracts or expands between layer ℓ and ℓ+1 — the mean change in distance to the same set of neighbours. O_ℓ(x) measures the mean Ollivier curvature of edges incident to x at layer ℓ. ρ(x) is the Pearson correlation between these two signals across all layers. If ρ(x) is negative, it means that when curvature at x is high — neighbours are clustered — the neighbourhood subsequently contracts. And when curvature is low — neighbours are spread — the neighbourhood expands. That's exactly what Ricci flow does: it smooths out curvature by contracting high-curvature regions and expanding low-curvature ones. The paper shows this is the dominant behaviour in trained networks.

---

## Section 3 — Experimental setup (4 min, 2 slides)

---

### Slide 8 — Our setup

**SLIDE**
```
[figures/slide8_setup.png]
```

**NOTES**
Our setup. We use Fashion-MNIST restricted to two classes — Pullover and Shirt — since the paper's binary setting is the clearest to analyse. Our model is a 6-layer fully-connected network with ReLU activations; we extract features at each post-ReLU hidden layer. For the graph we use symmetric k-nearest-neighbours with k=6 on the raw activations, subsampling to 200 points per class for tractability. The 3-D UMAP projections are Procrustes-aligned to the final layer so that the animation shows genuine geometric change rather than arbitrary rotations.

---

### Slide 9 — Analysis pipeline

**SLIDE**
```
Pipeline:

  1. train_base_model.py
     → weights/base_model.pt

  2. compute_metrics.py
     → k-NN graphs, Ollivier/Forman curvatures,
       Q, NCut, ΔO, ρ(x), r_layer, 3-D UMAP
     → outputs/ricci_metrics.npz

  3. generate_adversarial.py   [ε=0.03 … 0.30]
     → FGSM images + adversarial features

  4. visualise_metrics.py
     → community_metrics.png
     → combined_manifold.gif
     → adversarial_ricci_epsε.png
```

**NOTES**
The pipeline is four scripts. Training is one-shot. The metrics script is the expensive one — Ollivier-Ricci involves solving one optimal transport problem per edge per layer, which takes a few minutes on CPU. The adversarial script runs FGSM at multiple ε values, saves both the perturbed images and the corresponding features. The visualisation script produces everything: the static community metrics figure, the animated 3-D manifold GIF, and when given the adversarial flag, the Ricci comparison figure.

---

## Section 4 — Clean results (10 min, 4 slides)

---

### Slide 10 — Manifold animation

**SLIDE**
```
3-D UMAP manifold — layer by layer

[embed outputs/combined_manifold.gif]

  Colour = class (Pullover / Shirt)
  Edges  = k-NN graph within each class
  Final  = linear decision boundary
  Last   = edge-on view (boundary → line)
```

**NOTES**
This is the main visual. We project the activations at each layer into 3-D using UMAP, and align all projections to the final layer via Procrustes. What you're watching is genuine geometric change — at layer 1, the two classes are mixed together; by layer 6, they've separated into two distinct clusters with a clear linear boundary between them. The k-NN graph edges show the local connectivity within each class becoming tighter as we go deeper. The final edge-on view collapses the boundary to a line, showing just how cleanly the two manifolds separate. The right panels show the mean curvature κ̄ and the per-layer Pearson coefficient r_layer growing as the animation steps through layers.

---

### Slide 11 — Community metrics

**SLIDE**
```
[embed outputs/community_metrics.png]

Four panels, one per layer:
  • Modularity Q    — ↑ layers → better class separation
  • Normalised Cut  — ↓ layers → fewer cross-class edges
  • Fiedler value   — ↑ layers → better-connected clusters
  • Curvature gap ΔO — within-class edges more positive
```

**NOTES**
The community metrics tell a consistent story. Modularity increases across layers — the graph becomes more class-separated. The normalised cut decreases — fewer edges cross class boundaries. The Fiedler value (algebraic connectivity) grows — the within-class structure becomes more robustly connected. The curvature gap ΔO measures the mean curvature difference between same-class and cross-class edges; it grows as the network learns to concentrate positive curvature within classes. All of these are consistent with a representation that's progressively organising by class identity.

---

### Slide 12 — ρ(x) distribution

**SLIDE**
```
[figures/slide12_rho_distribution.png]

mean ρ = −0.283   ·   median ρ = −0.531   ·   72.8% of vertices ρ < 0
```

**NOTES**
And here's the key statistic. The violin plot shows the distribution of ρ(x) across all 400 data points. The majority — consistent with the paper's 73–98% range — have ρ < 0. Fill in the exact number when you run compute_metrics.py. The interpretation: most data points are in regions where the geometry is genuinely evolving in the Ricci-flow direction — positive-curvature neighbourhoods contract, negative-curvature neighbourhoods expand. This is the baseline we'll compare against under adversarial attack.

---

### Slide 13 — r_layer: layer-by-layer view

**SLIDE**
```
[figures/slide13_rlayer.png]

r_layer[ℓ] = Pearson(O_ℓ(x), η_ℓ(x))  across vertices

  L1→L2:  +0.059      L2→L3:  −0.314
  L3→L4:  −0.140      L4→L5:  +0.632
  L5→L6:  +0.681

Contrast: ρ(x) correlates *per vertex* across all layers
          r_layer correlates *per layer* across all vertices
```

**NOTES**
We also look at a per-layer version, r_layer, which measures the same correlation but across vertices at a single layer transition. The picture is more nuanced than the aggregate ρ(x): the middle transitions (L2→L3, L3→L4) show the negative Ricci-flow signature, but the final two transitions are strongly positive — when we go from the 256- to the 128- and 64-dimensional layers, vertices with higher curvature actually see their neighbourhoods expand. One interpretation: the final layers are performing a deliberate spreading of the representation to maximise class separation, reversing the geometry in those last steps. The earlier contractive behaviour does the clustering; the final expansive behaviour opens up the decision gap. This is a richer picture than the paper's aggregate ρ(x) statistic, and a natural place to look for disruption under adversarial attack. This decomposition is useful because it lets us ask whether the disruption happens uniformly across layers or is concentrated in specific transitions.

---

## Section 5 — Adversarial attack (5 min, 2 slides)

---

### Slide 14 — FGSM attack

**SLIDE**
```
Fast Gradient Sign Method (untargeted):

  x_adv = clip( x_raw + ε · sign(∂L/∂x_norm), 0, 1 )

  ε = perturbation budget in [0, 1] pixel space

     ε        accuracy      accuracy drop
   0.030        [X]%           [X]%
   0.050        [X]%           [X]%
   0.100        [X]%           [X]%
   0.200        [X]%           [X]%
   0.300        [X]%           [X]%

Primary analysis: ε = 0.03 (imperceptible)
```

**NOTES**
The adversarial attack is FGSM — a single gradient step in the direction that maximises the loss. ε controls the perturbation budget in raw pixel space. Fill in the accuracy numbers from your attack_summary.txt. At ε=0.03 the images are imperceptible to a human but the model's accuracy drops noticeably — that's the regime we're most interested in for the Ricci analysis, because we want to understand what's happening to the representation when the attack is subtle, not when it's just destroying the image.

---

### Slide 15 — Visual examples

**SLIDE**
```
[embed outputs/adversarial/comparison_grid.png]

Rows = example images (Pullover / Shirt)
Cols = clean, ε=0.05, ε=0.10, ε=0.20, ε=0.30

At ε=0.03–0.05: changes invisible to human
At ε=0.20+:    texture noise visible but semantics preserved
```

**NOTES**
The comparison grid shows how the images change across ε values. At our primary analysis ε of 0.03 the changes are genuinely imperceptible — that's important context for the geometry results. The model is being fooled by something a human wouldn't notice. That makes the question of *what* has changed in the representation space much sharper.

---

## Section 6 — Adversarial Ricci analysis (12 min, 4 slides)

---

### Slide 16 — Adversarial manifold overlay

**SLIDE**
```
[embed outputs/combined_manifold_adv0.030.gif — key frames]

× markers = adversarial examples (coloured by original class)
○ markers = clean examples

Layer 1: adv examples close to clean
Layer 6: some displacement; does the separation hold?
```

**NOTES**
The first look at what adversarial perturbation does to the representations. We run the clean and adversarial features through a combined UMAP — fitting the projection on both at once so they share a coordinate frame — and then apply the same Procrustes rotation. The × markers show adversarial examples coloured by their original class. Qualitatively, we can ask: do adversarial examples stay in the same region of the manifold as their clean counterparts? Do they cross the decision boundary? And does the manifold structure itself look different?

---

### Slide 17 — Curvature and community metrics: clean vs adversarial

**SLIDE**
```
[embed outputs/adversarial_ricci_eps0.030.png — top two panels]

Panel 1: Mean κ̄ per layer (clean=blue, adv=red)
Panel 2: Curvature gap ΔO per layer

Questions:
  • Is the curvature profile preserved under attack?
  • Does ΔO (class separation) change?
  • Are the changes uniform or concentrated in specific layers?
```

**NOTES**
Here's the core comparison. The top two panels show mean curvature κ̄ and curvature gap ΔO across layers for clean versus adversarial representations. Fill in your observations from the actual plot. Key things to look for: does the adversarial profile qualitatively follow the same trajectory as clean, just shifted? Or does it show qualitatively different behaviour at certain layers? A drop in ΔO would mean the adversarial representations are less separable by class in curvature terms — the network's geometric class structure is being eroded.

---

### Slide 18 — Modularity: is class structure preserved?

**SLIDE**
```
[embed outputs/adversarial_ricci_eps0.030.png — bottom-left panel]

Modularity Q per layer:
  Clean:      [trajectory description]
  Adversarial:[trajectory description]

Q measures how well the k-NN graph clusters by class.

If Q drops under attack: the k-NN graph mixes class labels
→ the network's internal class structure is being disrupted
  even in layers the adversarial examples "pass through"
```

**NOTES**
Modularity is perhaps the most direct measure of class structure in the graph. A drop in Q under adversarial attack means that in the k-NN graph built on adversarial features, the class labels are less well-clustered — the two classes mix together more. This would be evidence that FGSM is doing something systematic to the internal geometry beyond just moving examples across the decision boundary.

---

### Slide 19 — ρ(x) under adversarial attack

**SLIDE**
```
[embed outputs/adversarial_ricci_eps0.030.png — bottom-right panel]

ρ(x) distribution: clean vs adversarial

  Clean:       [X]% of vertices with ρ < 0
  Adversarial: [X]% of vertices with ρ < 0

  Δ = ?

If the Ricci flow signature is disrupted:
  adversarial ρ should be less negative / more symmetric
  → the network's layer-by-layer organising principle breaks
```

**NOTES**
And the headline result. Does adversarial perturbation disrupt the Ricci flow signature — the ρ(x) distribution? If the attack is purely moving examples across the decision boundary while leaving the representation geometry intact, we'd expect ρ(x) to be largely unchanged. If, on the other hand, the adversarial perturbation is working by disrupting the network's geometric organising principle — by breaking the anti-correlation between curvature and neighbourhood change — we'd expect the fraction of vertices with ρ < 0 to decrease. Fill in the actual numbers here. Whatever the result, it's informative: preservation would suggest the Ricci flow signature is robust to input-space perturbation; disruption would suggest it's a potential attack surface.

---

## Section 7 — Interpretation and open questions (7 min, 3 slides)

---

### Slide 20 — What would it mean?

**SLIDE**
```
Scenario A: ρ(x) preserved under attack

  → Adversarial examples "ride along" with the Ricci flow
  → The decision boundary is moved, but the geometry isn't
  → Attack is surface-level; representation is robust

Scenario B: ρ(x) disrupted under attack

  → Attack works by breaking the geometric organising principle
  → Positive-curvature regions no longer contract
  → The network's internal compass is confused

Both scenarios raise a theoretical question.
```

**NOTES**
Whatever we observe, it raises an interesting theoretical question. If the Ricci flow signature is preserved under attack, that tells us that FGSM is a fairly "shallow" attack — it moves things around in the representation space without fundamentally altering the network's geometric behaviour. If it's disrupted, that's more interesting: it suggests that the anti-correlation between curvature and distance change — the property that makes trained networks organise representations well — is being actively broken. That would point toward a connection between the Ricci flow signature and the robustness of the representation.

---

### Slide 21 — Ricci surgery as a theoretical lens

**SLIDE**
```
Ricci flow with surgery (Hamilton-Perelman):

  • Ricci flow can develop singularities (pinching, necks)
  • Surgery: remove the singular region, continue the flow
  • Perelman used this to prove the Poincaré conjecture

Neural robustness analogy:
  • Adversarial perturbation = singularity in the flow?
  • Robust network = one that performs "surgery" to recover?
  • Hypothesis: architectural features that prevent pinching
    (e.g. residual connections, normalisation) correspond
    to surgical interventions in the Ricci flow picture

[OPEN — no theorems yet]
```

**NOTES**
This is the speculative but motivating theoretical framing. In Riemannian geometry, Ricci flow can develop singularities — regions where curvature blows up. Hamilton and Perelman showed that you can cut out these singular regions and continue the flow; this technique — Ricci surgery — was the key tool in Perelman's proof of the Poincaré conjecture. The analogy to neural robustness is: adversarial perturbation might create singularity-like configurations in the network's geometric flow, and a robust network is one that can recover from them — perform surgery. This is entirely speculative at this stage, but it's a mathematically well-defined setting in which to try to prove something.

---

### Slide 22 — Open questions

**SLIDE**
```
Open questions

1. Is there a layer-specific signature?
   Does adversarial disruption concentrate in early or late layers?

2. Does ρ(x) predict fooling?
   Are examples with ρ > 0 (already "wrong direction") more
   easily fooled? Can it serve as a detector?

3. Does architecture matter?
   ResNets / normalisation layers / attention: different ρ profiles
   under attack?

4. Can you construct attacks that maximally disrupt ρ(x)?
   Curvature-aware adversarial attack as a harder threat model.

5. What does a "Ricci-robust" training objective look like?
   Regularising toward ρ < 0 as a constraint.
```

**NOTES**
Five open questions, in roughly increasing order of difficulty. The first two are directly testable with the current codebase. The third requires running on different architectures. The fourth is a research direction: if the Ricci flow signature is something the network relies on, then an attack that deliberately disrupts it might be harder to defend against than standard FGSM. The fifth is the long-term question: can we write down a training objective that explicitly regularises the network toward Ricci-flow-like behaviour, and does that improve robustness? That's where this could go if the preliminary results are interesting.

---

### Slide 23 — Summary

**SLIDE**
```
Summary

• arXiv:2509.22362: trained networks evolve representations
  layer-by-layer in a way consistent with discrete Ricci flow

• We reproduce this on Fashion-MNIST (Pullover vs Shirt):
  [X]% of vertices have ρ < 0; community metrics grow monotonically

• FGSM at ε=0.03: accuracy drops [X]%; representations shift

• Adversarial Ricci analysis:
  — curvature profile: [preserved / disrupted]
  — modularity Q: [preserved / disrupted]
  — ρ(x) distribution: [X]% → [X]% ρ < 0 under attack

• Open: can the Ricci flow signature predict or detect
  adversarial vulnerability?

Code: github.com/eddableheath/ARC-robustness
```

**NOTES**
Fill in the numbers from your actual runs. The talk makes a simple arc: here's a geometric property that trained networks have; here's what adversarial attacks do to it; here are the questions that opens up. Close by pointing people to the code and inviting questions.

---

## Timing guide

| Section | Slides | Time |
|---------|--------|------|
| Motivation | 1–2 | 5 min |
| Paper background | 3–7 | 12 min |
| Experimental setup | 8–9 | 4 min |
| Clean results | 10–13 | 10 min |
| Adversarial attack | 14–15 | 5 min |
| Adversarial Ricci analysis | 16–19 | 12 min |
| Interpretation + open questions | 20–23 | 7 min |
| Questions | — | 5 min |
| **Total** | **23 slides** | **~60 min** |

---

## Figures to prepare

| Slide | Figure | Source |
|-------|--------|--------|
| 10 | Combined manifold GIF | `outputs/combined_manifold.gif` |
| 11 | Community metrics | `outputs/community_metrics.png` |
| 12 | ρ(x) histogram/violin | extract from ricci_metrics.npz |
| 13 | r_layer line plot | visible in combined GIF right panel |
| 15 | Comparison grid | `outputs/adversarial/comparison_grid.png` |
| 16 | Adversarial overlay GIF | `outputs/combined_manifold_adv0.030.gif` |
| 17–19 | Ricci comparison figure | `outputs/adversarial_ricci_eps0.030.png` |

The ρ(x) violin (slide 12) will also appear in the adversarial comparison figure (slide 19); you can use the clean half of the adversarial Ricci figure for slide 12, or generate a standalone plot.

---

## Placeholder numbers to fill in

Run the following and substitute into the slides:

```bash
# ρ(x) stats (slide 12, 23)
uv run python -c "
import numpy as np
d = np.load('outputs/ricci_metrics.npz')
rho = d['rho']
print(f'mean rho = {rho.mean():.3f}')
print(f'median rho = {np.median(rho):.3f}')
print(f'frac < 0 = {(rho < 0).mean():.1%}')
"

# Attack summary (slides 14, 23)
cat outputs/adversarial/attack_summary.txt
```
