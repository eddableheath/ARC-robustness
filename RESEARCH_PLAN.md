# Research Plan — Geometric Analysis of Adversarial Robustness via Discrete Ricci Flow

Scoping document for turning the current codebase into a coherent paper.

**Target claim of the paper (working thesis):**
> The layer-wise geometric signature identified by arXiv:2509.22362 can be made
> reparameterisation-invariant and statistically well-founded; once it is, it yields
> a *quantitative, falsifiable* account of where and why feature-graph structure
> collapses under adversarial perturbation — including a provable stability radius
> that predicts the layer at which collapse begins.

That thesis is deliberately narrower and harder than "we looked at curvature under
attack". It gives one theorem that predicts one number that one experiment can falsify.

---

## Part 0 — Diagnostics: what the current results already tell us

These come from `outputs/ricci_metrics.npz` and the cached features. They are not
bugs to hide; several are the seed of the paper's contribution.

### 0.1 The Ollivier curvature as implemented is not scale-free

`ricci.ollivier_ricci` computes `κ(u,v) = 1 − W₁(μ_u, μ_v)` where `W₁` uses **Euclidean
feature-space ground cost** but the normalising `d(u,v)` is hard-coded to **1 (graph
hop distance)**. The units do not match: `W₁` is in activation units, so `κ` is an
affine function of local neighbourhood scale, not a curvature.

Observed consequence — mean vertex Ollivier curvature per layer:

```
relu1    relu2    relu3    relu4    relu5    relu6
-8.075   -6.219   -4.010   -2.036   -0.659   +0.297
```

and mean pairwise activation distance:

```
relu1    relu2    relu3    relu4    relu5    relu6
 21.5     25.3     50.5     73.8     64.5     29.9
```

The monotone κ̄ trend is largely "local neighbourhood diameter shrinks", reported in
raw activation units. **Action:** implement both the properly normalised
`κ = 1 − W₁/d_Euclid(u,v)` and a per-layer scale-normalised variant, and check against
the paper's actual convention. Every κ̄ figure gets re-run. See T1.

### 0.2 The curvature gap ΔO is *negative* for layers 2–6

```
curvature_gap_ollivier   [ 0.393  -0.621  -0.773  -0.562  -0.405  -0.363]
curvature_gap_forman     [ 0.373   0.422  -0.312  -0.059   0.025   0.119]
```

Slide 11 and slide 17 both assert ΔO grows as the network concentrates positive
curvature within classes. The data says the opposite from layer 2 onward: **inter-class
edges are more positively curved than intra-class edges.** Two candidate explanations,
both testable:

1. **Artefact of §0.1** — unnormalised κ penalises edges in high-scale regions.
2. **Real k-NN geometry** — in a well-separated k-NN graph the few surviving inter-class
   edges live in the dense overlap region where neighbourhoods are tight, so they
   *should* look positively curved. If so, ΔO on a k-NN graph is a poor separability
   proxy and this is worth saying in print.

The two curvature notions also disagree in sign at layers 2–3, which the paper's framing
does not anticipate. Resolving this is a Tier-A blocker.

### 0.3 Forman-Ricci carries almost no information on a symmetric k-NN graph

`F(u,v) = 4 − deg(u) − deg(v)`, and measured degree spread is:

```
relu1: deg 9.12 ± 3.84  (min 6, max 36)
relu3: deg 7.96 ± 1.65  (min 6, max 13)
relu6: deg 7.02 ± 1.10  (min 6, max 11)
```

By construction `deg ≥ k = 6`, and in deeper layers degrees concentrate at exactly `k`.
So Forman is close to the constant `4 − 2k` and its variation measures hub-ness only —
it is a function of the graph's degree sequence and is **completely blind to the
feature-space geometry given the graph**. The README's framing of Forman as a "fast
approximation" to Ollivier is not supportable here. Make this a stated proposition
(T7) rather than a silent caveat.

### 0.4 Algebraic connectivity is uninformative after layer 3

```
algebraic_connectivity   [0.320  0.065  0.006  0.000  -0.000  -0.000]
```

The graph disconnects (4 components at relu6). A Fiedler value of 0 just means
"disconnected" and carries no further signal. **Action:** report component count,
Fiedler value of the largest component, and the normalised-Laplacian spectral gap
instead.

### 0.5 Modularity saturates immediately

```
modularity   [0.327  0.419  0.433  0.424  0.425  0.424]
```

For two equal-sized communities Q is bounded near 0.5, so this is at ~85% of ceiling
by layer 2 and flat thereafter. As a *clean-network* progress measure it has almost no
dynamic range. It may still be a good *adversarial* measure (it can fall), but the
"Q increases across layers" narrative is supported by exactly one transition.

### 0.6 ρ(x) — three confirmed defects *(measured, not conjectured)*

Verified numerically; the decomposition below reproduces the stored `rho` array to
within 2.2e-6, so these are exact statements about the current results.

**(a) Not a property of the learned function.** Applying the function-preserving ReLU
rescaling `c_ℓ = Mˡ` (see T1) and recomputing:

```
      M    mean rho    frac rho<0
   0.02     +1.0000         0.00%
   0.50     +0.9533         0.50%
   0.80     +0.2856        36.50%
   1.00     -0.2832        72.75%   <- as trained
   1.25     -0.5038        83.75%
   4.00     -0.5958        94.00%
  50.00     -0.9996       100.00%
```

Every row is the same classifier: identical predictions, accuracy, and adversarial
examples. The headline statistic spans [0%, 100%]. Critically, `M ∈ [0.8, 1.25]` — a
±25% change in per-layer scale ratios, well within seed-to-seed variation — already
moves it from 36.5% to 83.75%. This is not a pathological edge case.

**(b) A global statistic reported as 400 local ones.** Variance decomposition:

```
O_l(x):   layer means [-8.075 -6.219 -4.010 -2.036 -0.659]   between-layer share 54.5%
eta_l(x): layer means [ 0.546  2.178 -0.918 -1.807 -1.535]   between-layer share 18.8%
```

Correlating the *layer means alone* — five numbers vs five numbers, zero per-vertex
information — gives **ρ_global = −0.80**, stronger than the mean of the per-vertex
distribution (−0.283, sd 0.614). Removing the shared per-layer trend by centring each
layer across vertices:

```
rho(x) after per-layer centring:  mean +0.105   median +0.163   frac<0 41.5%
```

The signature vanishes and mildly reverses (chance = 50%). The signal lives in the
global trend, which is precisely the non-invariant component from (a). *Note: centring
removes the additive per-layer offset but not the multiplicative `c_ℓ`, so this is a
diagnostic, not the repair — the repair is T1.4, tested in A2.*

**(c) The 5-point sampling problem.** Each ρ(x) is a Pearson correlation over `L−1 = 5`
points. Under an exchangeable null, `P(r<0) = 0.4993` (the sign test is well calibrated)
but `sd(r) = 0.50` and **39.1% of random draws have |r| > 0.5**. The observed median of
−0.53 is an ordinary value for pure noise, so individual ρ(x) values carry almost no
information. The aggregate's nominal 9.1σ assumes 400 independent vertices; since the
signal is one shared trend, the effective sample size for that component is nearer 5.

**What survives.** The *paired within-network comparison* (clean vs adversarial through
the same weights) is still well-posed — `c_ℓ` is held fixed, so non-invariance cancels
in the difference. But the error bar must come from training seeds and bootstrap
resampling, never from the 400 vertices. The violin plot on slide 12 shows sampling
noise around a single global number, not a population of local measurements, and must
be relabelled or replaced. The global trend correlation of −0.80 is a real effect, to
be reported as one number from 5 transitions with the A1 controls doing the work.

Diagnostic script: `scripts/rho_diagnostics.py` (to be added from scratchpad).

### 0.7 Missing control: no untrained / random-label baseline

The paper's central control is "trained networks show this, random ones do not".
Nothing in the repo runs it. This is the single highest-priority missing experiment —
without it there is no result, only a measurement.

### 0.8 The attack sweep misses the interesting regime

```
ε      accuracy   drop
0.030    41.5%    51.5%
0.050    17.5%    75.5%
0.100     1.2%    91.8%
0.200     0.0%    93.0%
0.300     0.0%    93.0%
```

Clean accuracy is 93%; at the "primary analysis" ε = 0.03 the model is already **below
chance** for a binary task. The geometric transition is happening somewhere in
ε ∈ (0, 0.03), which is entirely unsampled. Re-sweep at ε ∈ {0.002, 0.005, 0.01, 0.015,
0.02, 0.025, 0.03} and define the primary operating point as the ε that halves the
margin, not a round number.

---

## Stream 1 — Numerical experiments

Four tiers. Tier A gates everything else: until the estimator is trustworthy, no
adversarial result is interpretable.

### Tier A — Validity, controls, and statistics *(blocking)*

| ID | Experiment | Output |
|----|-----------|--------|
| **A1** | **Controls.** Random-init (untrained) net; net trained on shuffled labels; linear net (ReLU removed); net trained to 100% train acc on random data. All metrics on each. | The falsification table. Every subsequent claim is "relative to these". |
| **A2** | **Invariant estimators.** Implement `κ_norm = 1 − W₁/d_Euclid(u,v)`; per-layer feature normalisation (unit mean pairwise distance); log-ratio η̃; centred variants. Re-run all figures under raw and invariant estimators side by side. | Shows which of the paper's findings survive invariance. Directly pairs with T1. |
| **A3** | **Error bars.** ≥5 training seeds × ≥5 subsample seeds for every reported number. Bootstrap CIs. Graph-aware permutation null for ρ and r_layer. | No number in the paper without a CI. |
| **A4** | **Graph-construction sensitivity.** k ∈ {3,5,6,10,15,20}; N ∈ {200,400,1000,2000}; symmetric-OR vs mutual-AND k-NN vs ε-ball; Euclidean vs cosine. | Does the *sign* of ΔO flip? Does ρ<0 fraction move? Robustness of the whole approach. |
| **A5** | **Connectivity fix.** Component count, largest-component Fiedler, normalised-Laplacian gap, Cheeger bound. | Replaces the dead §0.4 panel; needed for T5. |
| **A6** | **ΔO diagnosis.** Decompose ΔO by edge density / distance to boundary; test the two hypotheses in §0.2. | Resolves the contradiction between the data and the slides. |

### Tier B — Core adversarial experiments *(the contribution)*

| ID | Experiment | Why it matters |
|----|-----------|----------------|
| **B1** | Fine ε-sweep, ε ∈ [0, 0.03], all metrics, all invariant estimators. Identify the geometric transition point ε_geo and compare it to the accuracy transition ε_acc. | If ε_geo < ε_acc, geometry breaks *before* accuracy — the paper's most quotable result. |
| **B2** | **Attack family + norm-matched random-noise control.** FGSM, PGD-ℓ∞, PGD-ℓ₂, C&W, and Gaussian/uniform noise at matched perturbation norm. | Without the noise control you cannot claim anything is *adversarial* rather than *perturbative*. Non-negotiable. |
| **B3** | Split by outcome: fooled vs correctly-classified adversarial examples. Compare ρ, κ̄, local ΔO. | Tests slide-22 Q2. |
| **B4** | **Predictiveness.** Is clean-network ρ(x) (or local curvature) predictive of ε_min(x), the minimum ε that flips x? Report Spearman correlation and detector AUROC against baselines (margin, logit gap, local Lipschitz estimate). | This is the "does the geometry buy you anything the margin doesn't" test. Be prepared for a negative result and report it — a clean negative here is publishable and honest. |
| **B5** | Adversarial graph-design ablation: separate adv graph vs joint clean+adv graph; ground-truth vs predicted labels for community metrics. | Currently one arbitrary choice is baked in. Report all four cells. |
| **B6** | **Layer localisation.** Δmetric as a function of (layer, ε). Which layer degrades first? | Pairs directly with theorem T4's prediction — this is the theory/experiment join. |
| **B7** | **Curvature-aware attack.** Optimise a perturbation to maximally reduce Q / flip r_layer sign, subject to an ℓ∞ budget. Measure fooling rate per unit norm vs PGD; measure transfer. | Tests whether the geometric signature is a genuine attack surface (slide-22 Q4). |
| **B8** | **Adversarially trained model.** PGD-train the same architecture; recompute every metric; recompute the ε-sweep. | The decisive "does geometry track robustness" experiment. If robust models have a measurably different Ricci profile, the whole programme is validated; if not, say so. |

### Tier C — Ablations and edge cases

| ID | Experiment | Note |
|----|-----------|------|
| **C1** | **Architecture.** Depth {3, 6, 12}; width; residual connections; **BatchNorm / LayerNorm**; a small CNN. | BN/LN is the interesting case: normalisation *removes* the positive-rescaling symmetry, so T1's non-invariance objection changes character. Strong theory/experiment synergy. |
| **C2** | **Datasets.** FMNIST 2v6 (hard, current), FMNIST 0v1 (easy), MNIST 1v7 and 6v9 (paper's pairs, for direct comparison), CIFAR-10 cars-vs-planes. Plus a **10-class multiclass** run. | Multiclass changes the community count — does ρ<0 survive? The paper is binary-heavy. |
| **C3** | **Training dynamics.** Metrics at epochs 0, 1, 2, 5, 10, …, final. | Does ρ<0 emerge *with* accuracy, *before* it, or *after*? Makes a strong figure and speaks to whether the signature is cause or consequence. |
| **C4** | **Train vs test points.** All metrics on held-out data; overfit regime. | Generalisation angle; cheap. |
| **C5** | **Depth-matched capacity control.** Match parameter count across depths so depth effects aren't capacity effects. | Guards a likely reviewer objection. |

### Tier D — Visualisation

| ID | Deliverable |
|----|------------|
| **D1** | Re-cut the combined GIF as side-by-side clean/adversarial, and a second animation with **ε as the time axis** at fixed layer (currently layer is the only axis). |
| **D2** | Curvature-coloured k-NN edges on the manifold — show *where* negative curvature concentrates, not just its mean. |
| **D3** | Per-vertex (O_ℓ, η̃_ℓ) trajectory scatter, layer as colour, contrasting high-\|ρ\| and low-\|ρ\| vertices. Makes ρ legible as a picture. |
| **D4** | **Fragility heat map:** layer × ε → Δmetric, with the T4 predicted stability radius overlaid as a contour. This is the paper's headline figure. |
| **D5** | Publication-quality static figure set (GIFs don't go in a paper); consistent style, colourblind-safe palette. |

---

## Stream 2 — Theoretical programme

Ordered by ratio of (contribution) to (difficulty). T1–T4 are the paper's theory
section; T5 is the future-work section; T6–T7 are methodological necessities.

### T1 — Reparameterisation invariance of the Ricci-flow signature ⭐ *highest value, lowest cost*

**Setting.** A ReLU MLP admits an exact function-preserving symmetry from positive
homogeneity, `ReLU(cz) = c·ReLU(z)` for `c > 0`: scaling `(W_ℓ, b_ℓ) → (c W_ℓ, c b_ℓ)`
and `W_{ℓ+1} → W_{ℓ+1}/c` leaves the input–output map unchanged while rescaling every
layer-ℓ activation by `c`. Write `G` for the group of such per-layer rescalings
`c = (c_1, …, c_L) ∈ ℝ_{>0}^L`.

**Proposition T1.1 (what is invariant).** The symmetric k-NN graph `G_ℓ`, and hence
modularity `Q`, normalised cut, component structure and the Laplacian spectrum *up to
scale*, are invariant under `G`. *(Proof: rescaling is a homothety; it preserves the
ordering of pairwise distances.)*

**Theorem T1.2 (what is not).** The unnormalised Ollivier curvature `κ_ℓ = 1 − W₁`,
the neighbourhood-change statistic `η_ℓ(x) = mean_y[d_{ℓ+1}(x,y) − d_ℓ(x,y)]`, and
therefore the local Ricci evolution coefficient `ρ(x)` are **not** invariant under `G`.
Moreover the construction is effective: for any target sign pattern
`s ∈ {±1}^{L−1}` there exists `c ∈ G` such that `sign(η_ℓ(x)) = s_ℓ` for **every**
vertex `x` simultaneously.

*Proof sketch.* Under `c`, `η_ℓ(x) = mean_y[c_{ℓ+1} d_{ℓ+1}(x,y) − c_ℓ d_ℓ(x,y)]`. All
distances are non-negative, so taking `c_{ℓ+1}/c_ℓ` large forces `η_ℓ(x) > 0` for all
`x`, and taking it small forces `η_ℓ(x) < 0` for all `x`. Chain across `ℓ` to realise
any `s`. Since `ρ(x)` is a Pearson correlation over `ℓ` between `η_·(x)` and `O_·(x)`,
its sign is thereby manipulable without changing the network's function. ∎

**Corollary T1.3.** The fraction of vertices with `ρ(x) < 0` — the paper's headline
statistic — is not a property of the learned function. Two networks computing the
identical map can report 0% and 100%.

**Theorem T1.4 (the repaired estimator).** Define
- `κ̂_ℓ(u,v) = 1 − W₁(μ_u, μ_v)/d_ℓ(u,v)` (matched units, scale-free);
- `η̂_ℓ(x) = mean_y log(d_{ℓ+1}(x,y)/d_ℓ(x,y)) − mean_{x'} mean_y log(d_{ℓ+1}(x',y)/d_ℓ(x',y))`
  (log-ratio, centred across vertices at each layer);
- `r̂_ℓ = corr_x(Ô_ℓ(x), η̂_ℓ(x))` (correlation **across vertices**, per layer).

Then `κ̂`, `η̂` and `r̂_ℓ` are invariant under `G`. *(Proof: rescaling acts on
`log d_ℓ` as an additive per-layer constant, killed by the centring; `κ̂` is a ratio of
two quantities that scale identically; Pearson correlation across vertices is invariant
to per-layer affine transformations of each argument.)*

**Corollary T1.5.** An invariant across-layer analogue of `ρ(x)` exists, but only after
per-layer centring — i.e. `ρ` must be built from `η̂`, not `η`.

**Why this matters.** It is a correct, cheap, constructive result that (a) identifies a
genuine flaw in a 2025 arXiv paper's central statistic, (b) supplies the fix, and
(c) generates experiment A2, whose outcome ("does the signature survive?") is
interesting either way. It also predicts C1's result: **BatchNorm/LayerNorm quotient out
the rescaling symmetry, so normalised architectures should show the invariance gap
close.** That is a falsifiable prediction from theory to experiment.

*Caveat to check before writing:* confirm the paper's own normalisation convention
(this repo's `d(u,v) = 1` may be an implementation choice, not the paper's). The
invariance argument stands regardless for `η` and `ρ`; only the `κ` half depends on it.

### T2 — Making the "Ricci flow" correspondence quantitative rather than sign-based

Discrete Ricci flow on a weighted graph is `dw(u,v)/dt = −κ(u,v)·w(u,v)`. The paper's
evidence is only that a correlation is *negative* — a sign test on a rank statistic.

**Derivation target.** Treat layer index as time with step `h` and edge weight
`w_ℓ(u,v) = d_ℓ(u,v)`. The exact one-step discretisation predicts

  `log w_{ℓ+1}(u,v) − log w_ℓ(u,v) = −h · κ̂_ℓ(u,v) + O(h²)`.

Define the **flow residual** `R_ℓ(u,v)` as the deviation from this, and fit the
regression `Δ log w_ℓ = −β_ℓ κ̂_ℓ + α_ℓ + ε` per layer.

**Proposition T2.1.** If layer `ℓ` implemented an exact step of discrete Ricci flow with
step size `h`, then `β_ℓ = h`, `α_ℓ = 0`, and `R² = 1`. Conversely `ρ < 0` is implied by
`β_ℓ > 0` but does not imply it.

So: `ρ < 0` is a *necessary but very weak* consequence of the flow hypothesis. Report
`(β_ℓ, R²_ℓ)` with CIs. This turns "the correlation is negative" into a *model fit with
a falsifiable prediction*, and gives a natural effect size for the adversarial
comparison (does attack change `β` or only `R²`?).

**Extension.** Compare against the alternative hypotheses the sign test cannot
distinguish: (i) pure isotropic contraction (`β_ℓ > 0` but κ̂ uncorrelated with position),
(ii) mean-curvature-flow-like behaviour, (iii) a Lipschitz-bounded linear map with no
curvature dependence. A model-selection table over these is a strong result.

### T3 — What one layer can do: spectral bounds on achievable flow

A layer is `x ↦ σ(Wx + b)` with `σ` 1-Lipschitz.

**Lemma T3.1.** For any `u, v`, `d_{ℓ+1}(u,v) ≤ s_max(W_ℓ)·d_ℓ(u,v)`; and if `σ` were
the identity, `d_{ℓ+1}(u,v) ≥ s_min(W_ℓ)·d_ℓ(u,v)`. ReLU can only contract, so the
lower bound requires care (state it on the subspace where the ReLU pattern is fixed —
each ReLU cell is a linear region and the bound holds within it).

**Proposition T3.2 (flow budget).** The per-layer spread of achievable log-distance
changes, `max_{u,v} Δ log w_ℓ − min_{u,v} Δ log w_ℓ`, is bounded by
`log cond(W_ℓ) = log(s_max/s_min)`. Hence the regression slope of T2 satisfies
`|β_ℓ| · spread(κ̂_ℓ) ≤ log cond(W_ℓ)`.

**Interpretation and payoff.** Ricci-flow-like behaviour — differentially contracting
high-curvature and expanding low-curvature regions — *requires* an ill-conditioned
weight matrix. But robustness certificates want `s_max` small (small Lipschitz
constant). This surfaces a concrete, quantifiable **tension between the Ricci-flow
signature and Lipschitz-based robustness**, and it is the bridge between this
geometric literature and the mainstream certified-robustness literature. Verify
empirically: measure `cond(W_ℓ)` vs `β_ℓ` across the seeds from A3, and across the
adversarially-trained model from B8 (prediction: PGD training shrinks `cond`, hence
shrinks `|β|`, hence *weakens* the Ricci signature — a sharp, checkable prediction).

### T4 — Graph stability radius: the predictive theorem ⭐ *best theorem/experiment pairing*

**Lemma T4.1 (k-NN graph stability).** Fix a point cloud and let `d_k(u)` and
`d_{k+1}(u)` be the distances from `u` to its `k`-th and `(k+1)`-th nearest neighbours.
If every point is displaced by at most `δ` and
  `min_u [d_{k+1}(u) − d_k(u)] > 2δ`,
then the k-NN graph is unchanged. *(Proof: each pairwise distance changes by at most
`2δ`, so the ordering across the k/(k+1) gap is preserved.)*

**Corollary T4.2.** Under the same condition every purely combinatorial graph statistic
— `Q`, NCut, component structure, Forman curvature — is **exactly** unchanged, and
`κ̂` changes by at most `O(δ / d_min)`.

**Theorem T4.3 (layer-wise fragility profile).** An input perturbation of ℓ∞-norm `ε`
displaces layer-ℓ features by at most `δ_ℓ ≤ ε·√d·∏_{j≤ℓ} s_max(W_j)`. Define the
layer-ℓ **stability radius**

  `ε*_ℓ = (1/2)·min_u[d_{k+1}^{(ℓ)}(u) − d_k^{(ℓ)}(u)] / (√d · ∏_{j≤ℓ} s_max(W_j))`.

Then for `ε < ε*_ℓ` the layer-ℓ feature graph and all its combinatorial statistics are
provably unchanged. The predicted **first layer to break** is `argmin_ℓ ε*_ℓ`.

**Why this is the right theorem.** It is elementary (a page of proof), it is *predictive*
rather than descriptive, and it converts directly into experiment **B6/D4**: compute
`ε*_ℓ` from the trained weights and the clean features, then check whether the observed
collapse in `Q(ℓ, ε)` begins at the predicted layer and at an `ε` above the bound.
The bound will be loose (product of spectral norms always is) — quantifying *how* loose,
and tightening it with an empirical local Lipschitz estimate, is itself a result.

**Refinements worth attempting:** replace `∏ s_max` with the empirical local Lipschitz
constant along the attack direction; a probabilistic version (most vertices stable) via
concentration on the k/(k+1) gap distribution; a matching lower bound (an attack that
provably breaks the graph at `ε = C·ε*`).

### T5 — Curvature ⇒ spectral gap ⇒ robustness, and the surgery programme

Rather than proving Riemannian results from scratch, chain established discrete results.

**Proposition T5.1 (imported).** If the layer-ℓ within-class subgraph has Ollivier
curvature `κ̂ ≥ κ₀ > 0` on all edges, then by Ollivier (2009) / Lin–Lu–Yau its
Laplacian spectral gap satisfies `λ₁ ≥ κ₀` (Lichnerowicz) and its diameter satisfies
`diam ≤ 2/κ₀` (discrete Bonnet–Myers).

**Corollary T5.2.** Combined with Cheeger, a positive intra-class curvature floor gives
a lower bound on the number of edges that must be cut to detach a point from its class
cluster — i.e. a curvature-derived robustness certificate for the *graph* structure.
Combine with T4.3 to push it back to input space.

**Surgery (future work, stated as conjecture not theorem).** A discrete neck-pinch is a
bottleneck: a small edge cut separating two dense regions, with curvature diverging on
the bridging edges. Conjecture: adversarial perturbation preferentially creates such
bottlenecks, and architectural devices that prevent them (residual connections,
normalisation) act as surgical interventions. State it precisely enough to be attacked
later — define the discrete singularity, state what "surgery" would mean as an operation
on the layer map, and identify what would have to be proved. Do **not** promise a
theorem here in this paper.

### T6 — Statistics of the estimator

**Needed for the paper to be defensible:**
1. Null distribution of `ρ(x)` for `L−1 = 5` points under an exchangeable null
   (Pearson `r` on 5 points; exact null is available in closed form for Gaussian data).
2. **Dependence structure**: vertices share edges, so the 400 `ρ(x)` values are not
   independent. Derive or estimate an effective sample size; use a graph-block bootstrap
   (resample connected blocks, not vertices) for the CI on "fraction ρ < 0".
3. Power analysis: with 5 layers, what effect size is detectable? This directly bears on
   whether deeper networks (C1) are *required* to make the statistic meaningful — a
   12-layer net gives 11 points and a much better-conditioned correlation.
4. Multiple-comparison correction across the layer × metric × ε grid.

### T7 — Forman-Ricci on k-NN graphs is degree-only

**Proposition T7.1.** On a symmetric k-NN graph, `F(u,v) = 4 − deg(u) − deg(v)` with
`deg ≥ k`, and `deg(u) = k + |{w : u ∈ kNN(w), w ∉ kNN(u)}|` — the reverse-neighbour
excess. Hence Forman curvature is a deterministic function of the in-degree profile of
the directed k-NN relation and is measurable with respect to the graph alone: it is
constant on the fibre of the map (features → graph). Any conclusion drawn from Forman
is a conclusion about hub structure, not feature geometry.

Support with the measured degree concentration (`7.02 ± 1.10` at relu6, floor at `k=6`).
Augmented Forman adds `3|N(u) ∩ N(v)|`, which does see local clustering, so it is the
one worth keeping as a cheap proxy — verify its rank correlation with `κ̂`.

---

## Suggested paper structure

1. **Introduction** — geometric view of representation learning; the robustness question.
2. **Background** — discrete Ricci curvature; the Ricci flow hypothesis (2509.22362).
3. **The signature is not reparameterisation-invariant** (T1 + A2). *First contribution.*
4. **A quantitative flow test** (T2 + A1/A3 controls). *Second contribution.*
5. **Stability radius and the fragility profile** (T4 + B6/D4). *Third contribution — the core.*
6. **Adversarial experiments** (B1–B8): what breaks, when, and whether geometry adds
   information over the margin.
7. **The Lipschitz tension** (T3 + B8): geometry vs certified robustness.
8. **Discussion & future work** — curvature ⇒ spectral gap (T5.1–T5.2); the surgery
   programme as a conjecture.
9. **Appendices** — T6 statistics, T7 Forman, full ablation tables (Tier C).

---

## Sequencing

**Phase 1 — repair (blocks everything).** T1 written up; A2 invariant estimators
implemented; A1 controls; A3 error bars; A5/A6 metric fixes. Re-run every existing
figure. Outcome: a trustworthy baseline and the paper's first contribution.

**Phase 2 — the theorem and its test.** T4 proof; compute `ε*_ℓ` from weights;
B1 fine ε-sweep; B2 noise control; B6 layer localisation; D4 fragility heat map.
Outcome: the paper's core.

**Phase 3 — does it buy anything.** T2/T3 derivations; B4 predictiveness (accept a
negative result); B8 adversarial training; C1 architectures including BN/LN (tests
T1's prediction); C3 training dynamics.

**Phase 4 — breadth and write-up.** C2 datasets; B7 curvature-aware attack; T5/T6/T7;
D5 figure set; Tier C remainder as appendix tables.

## Risk register

| Risk | Mitigation |
|------|-----------|
| The signature vanishes under invariant estimators (T1/A2) | This *is* the paper — a methodological correction with controls. Frame accordingly from the start. |
| ρ(x) carries no information beyond the margin (B4) | Report the negative result; T4's stability radius does not depend on it. |
| T4's bound is vacuously loose | Report the loose bound *and* the empirical local-Lipschitz version; the gap is a quantified result. |
| Everything is an artefact of N=400, k=6 | A4 sensitivity sweep is in Tier A precisely for this. |
| Scope creep into full Riemannian theory | T5 is explicitly conjecture-only; surgery is future work. |
