"""
Geometric detection of evasion attacks (experiment A7, feasibility probe).

The hypothesis, stated so it can fail: **an adversarial input occupies a
distinguishable position in the layer-wise feature graph, and that position is
visible in local curvature.** If a perturbation drags a point across a class
boundary in feature space, it should arrive somewhere structurally unusual — in a
low-density region, bridging two communities, with a neighbourhood that is
inconsistent across layers. Those are curvature-shaped properties.

Two deployment regimes, and the distinction is not cosmetic:

**Transductive (batch).** Build the graph on a batch of test points and look for
distributional change. This is what the existing pipeline naturally does, but it
is a weak threat model: it needs a batch that is substantially adversarial, so it
answers "is this dataset attacked?" rather than "is this input attacked?".

**Inductive (single point).** Insert one query point into a reference graph built
from *clean* held-out data and compute its local geometry against that fixed
backdrop. This is the deployable version and the one worth testing. It also has a
clean statistical story: the reference graph is fixed, so the null distribution of
a query's local curvature can be estimated once from clean data and reused.

:mod:`arc_robustness.detection.features` implements the inductive regime.

Honesty requirements, non-negotiable for this to be publishable
--------------------------------------------------------------

1. **Strong baselines, or the result is meaningless.** Adversarial detection has
   an extensive literature and the obvious cheap signals are strong. At minimum:
   the model's own margin / max softmax, Mahalanobis distance to class-conditional
   feature Gaussians (Lee et al., 2018), kernel density and Bayesian uncertainty
   (Feinman et al., 2017), local intrinsic dimensionality (Ma et al., 2018), and
   deep k-NN (Papernot & McDaniel, 2018). LID in particular is a *local geometric*
   statistic on exactly these feature spaces, so it is the direct competitor —
   curvature must beat it or add to it, not merely work.

2. **Adaptive attacks, or the result will not survive review.** This subfield has
   a specific and repeated failure mode: detectors are proposed, then broken by
   attacks aware of them (Carlini & Wagner, 2017, *Adversarial Examples Are Not
   Easily Detected*; Athalye et al., 2018 on obfuscated gradients; Tramèr et al.,
   2020 on adaptive evaluation). A detector evaluated only against FGSM/PGD that
   do not know it exists reports an *upper bound* that is often wildly optimistic.
   The curvature statistics here are differentiable-ish, and B7's curvature-aware
   attack is the natural adaptive attack — so the probe should be designed knowing
   that its own honest evaluation is planned.

3. **The graph is a shared resource, so detection is not i.i.d.** A query's local
   curvature depends on the reference cloud. Test points cannot be treated as
   independent trials, and AUROC confidence intervals need a bootstrap over
   reference sets, not over query points.

Decision gate for the probe
---------------------------
Run the inductive detector against the baselines on the existing FGSM sweep at a
single ε. Expand into a full tier only if curvature either beats LID and
Mahalanobis, or adds significant AUROC in combination with them. If it does not,
record the negative result — "curvature carries no detection signal beyond LID"
is a genuine and reportable finding, and it costs one experiment rather than a
programme.
"""

from arc_robustness.detection.baselines import (
    kernel_density_score,
    lid_score,
    mahalanobis_score,
    margin_score,
)
from arc_robustness.detection.features import (
    detection_features,
    reference_graph_features,
)

# Still to write: ``detection.evaluate`` — the harness implementing the decision
# gate above (AUROC for each statistic and each baseline, their combination, and
# CIs bootstrapped over *reference sets* rather than query points, per honesty
# requirement 3). The feature extractors and baselines below are complete and
# usable without it; nothing else in the package depends on it.

__all__ = [
    "detection_features",
    "reference_graph_features",
    "lid_score",
    "mahalanobis_score",
    "kernel_density_score",
    "margin_score",
]
