"""
Grid definitions — the version-controlled description of every sweep.

Each builder returns an ordered :class:`~arc_robustness.config.Grid`, so
``SLURM_ARRAY_TASK_ID`` addresses the same cell on every submission and a
partially-failed array can be resumed by index. That guarantee is only as good
as this file's stability: **appending** to a grid is safe, but reordering or
inserting cells renumbers everything after the edit. Once a grid has results in
``results/``, add new cells at the end or define a new grid.

Grids are declared, not computed. Nothing here reads the filesystem, the
environment, or a previous run's output — a grid must be identical on a laptop
and on a login node, before and after any results exist.

Phase-1 coverage (see ``RESEARCH_PLAN.md``):

============  ==========================================================
``smoke``     Four cheap cells for end-to-end verification on a laptop.
``a1``        The falsification table: five control arms × two estimators.
``a2``        Estimator variants, including the T1.3 rescaling sweep.
``a3``        Error bars: training seeds × subsample seeds.
``a4``        Graph-construction sensitivity.
``b1``        Fine ε-sweep with norm-matched noise controls (Phase 2).
============  ==========================================================

A5 (connectivity) and A6 (curvature-gap decomposition) need no grids of their
own: :func:`~arc_robustness.analysis.pipeline.analyse_features` records their
metrics for every cell, so they come along with all of the above.
"""

from __future__ import annotations

from collections.abc import Callable

from arc_robustness.config import (
    AttackConfig,
    DataConfig,
    EstimatorConfig,
    ExperimentConfig,
    GraphConfig,
    Grid,
    ModelConfig,
    TrainConfig,
)

# ---------------------------------------------------------------------------
# The main arm
# ---------------------------------------------------------------------------
# Every grid is a set of departures from this configuration, so that "the
# difference from the main arm" is always exactly one or two fields.

#: Fashion-MNIST pullover-vs-shirt, analysed on the held-out test split. The
#: test split is the analysis default so that no reported geometry is measured
#: on points the network has memorised; C4 revisits train-vs-test deliberately.
MAIN_DATA = DataConfig(dataset="fashion_mnist", classes=(2, 6), split="test")

MAIN_MODEL = ModelConfig(widths=(512, 512, 256, 256, 128, 64))
MAIN_TRAIN = TrainConfig(epochs=20, seed=0)
MAIN_GRAPH = GraphConfig(k=6, n_per_class=200, subsample_seed=0)

#: The original convention of this repo and of the paper's released code:
#: ``κ = 1 − W₁`` with hop-distance normalisation, and raw additive ``η``.
#: Retained in every grid because the paper's claims must be reproducible
#: before they can be corrected.
RAW = EstimatorConfig(ollivier_norm="none", eta_mode="raw")

#: The T1.4 repair: distance-normalised curvature and per-layer-centred
#: log-ratio ``η̂``. Invariant under the rescaling group ``G``.
REPAIRED = EstimatorConfig(ollivier_norm="distance", eta_mode="log_centred")

#: The two estimators that every Phase-1 figure is reported under, side by side.
ESTIMATOR_PAIR = (RAW, REPAIRED)

#: Seeds for the headline arms. A3 extends this to five of each; three is enough
#: to see whether a control arm's effect is larger than seed noise, which is all
#: the falsification table needs.
CONTROL_SEEDS = (0, 1, 2)


def _base(experiment: str, **overrides) -> ExperimentConfig:
    """The main arm, in *experiment*, with top-level sections replaced."""
    fields = {
        "data": MAIN_DATA,
        "model": MAIN_MODEL,
        "train": MAIN_TRAIN,
        "graph": MAIN_GRAPH,
        "estimator": REPAIRED,
        "attack": AttackConfig(),
    }
    fields.update(overrides)
    return ExperimentConfig(experiment=experiment, **fields)


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def build_smoke() -> Grid:
    """Four cells that exercise every path in the runner, in about a minute.

    Small enough to run on a laptop, but not a toy: it covers a trained network,
    both estimator conventions, an attack, and the norm-matched noise control —
    which between them touch training, checkpoint reuse, attack dispatch,
    feature extraction, Ollivier curvature and the evolution statistics. If this
    grid completes, a cluster submission will fail only for reasons of scale.
    """
    grid = Grid("smoke")
    # Narrow but *as deep as the main arm*. Depth is the one dimension the smoke
    # grid must not shrink: ρ(x) is a correlation over ``L−1`` transitions, so a
    # 3-layer network gives 2 points and ρ is undefined (the evolution code
    # correctly returns NaN). A smoke run at depth 3 therefore passes while never
    # executing the statistic the project is about.
    model = ModelConfig(widths=(64, 64, 32, 32, 16, 16))
    train = TrainConfig(epochs=2, seed=0)
    graph = GraphConfig(k=5, n_per_class=30, subsample_seed=0)

    def cell(**overrides) -> ExperimentConfig:
        return _base("smoke", model=model, train=train, graph=graph, **overrides)

    for estimator in ESTIMATOR_PAIR:
        grid.add(cell(estimator=estimator))
    grid.add(
        cell(attack=AttackConfig(kind="fgsm", epsilon=0.03), tag="attack")
    )
    # ℓ∞-matched control: signed uniform noise at the same ε puts every pixel on
    # the same ±ε corner FGSM does, so the two arms differ in *direction* only.
    grid.add(
        cell(
            attack=AttackConfig(kind="uniform", epsilon=0.03, seed=0),
            tag="noise-control",
        )
    )
    return grid


# ---------------------------------------------------------------------------
# A1 — controls
# ---------------------------------------------------------------------------

#: The five arms of the falsification table. Each is a *named departure* from
#: the main arm; the name is carried in the cell's ``tag`` so results can be
#: grouped without re-deriving which arm a uid belongs to.
#:
#: The two memorisation arms get a generous epoch budget with an accuracy
#: target: a shuffled-label or random-data network that has *not* reached ~100%
#: train accuracy is not a memorisation control, it is an undertrained network,
#: and it would weaken the very comparison the control exists to make.
#:
#: The budget is 600 because 150 was measured to be far too small. Fitting
#: shuffled labels is *much* slower than fitting real ones: the main arm reaches
#: 96% train accuracy in 20 epochs, while the shuffled arm climbed roughly
#: linearly at 0.001/epoch — 0.67 at epoch 46, 0.82 at epoch 136 — with test
#: accuracy pinned at 0.50 throughout, which is memorisation proceeding exactly
#: as it should, only slowly. Early stopping on ``target_train_acc`` means the
#: larger budget costs nothing once the target is hit.
#:
#: 0.99 rather than 0.995: the last half-percent is the slowest part of the
#: curve and buys nothing — 99% on shuffled labels is memorisation by any
#: reading, and ``train_acc`` is recorded per cell so the claim is auditable.
CONTROL_ARMS: dict[str, dict] = {
    "trained": {},
    "untrained": {"train": TrainConfig(trained=False, epochs=0)},
    "shuffled_labels": {
        "data": DataConfig(
            dataset="fashion_mnist", classes=(2, 6), split="test",
            label_mode="shuffled",
        ),
        "train": TrainConfig(epochs=600, target_train_acc=0.99),
    },
    "linear": {"model": ModelConfig(widths=MAIN_MODEL.widths, activation="identity")},
    "random_data": {
        "data": DataConfig(
            dataset="fashion_mnist", classes=(2, 6), split="test",
            label_mode="random_data",
        ),
        "train": TrainConfig(epochs=600, target_train_acc=0.99),
    },
}


def build_a1() -> Grid:
    """A1: the falsification table.

    The paper's central control is "trained networks show the signature, others
    do not". Without these arms there is no result, only a measurement — so this
    grid is the one that must run first, and both estimator conventions appear
    in every arm because "which findings survive invariance" is a question about
    the controls too, not only about the main arm.
    """
    grid = Grid("a1")
    for name, overrides in CONTROL_ARMS.items():
        for seed in CONTROL_SEEDS:
            train = overrides.get("train", MAIN_TRAIN)
            for estimator in ESTIMATOR_PAIR:
                grid.add(
                    _base(
                        "a1",
                        **{**overrides, "train": train.with_seed(seed)},
                        estimator=estimator,
                        tag=f"control:{name}",
                    )
                )
    return grid


# ---------------------------------------------------------------------------
# A2 — estimator variants and the rescaling sweep
# ---------------------------------------------------------------------------

#: The rescaling ladder of plan §0.6(a). ``M ∈ [0.8, 1.25]`` is the interesting
#: window — a ±25% change in per-layer scale ratios, well inside seed-to-seed
#: variation — so it is sampled more finely than the extremes, which are there
#: only to show that the statistic saturates at 0% and 100%.
RESCALE_LADDER = (0.02, 0.5, 0.8, 0.9, 1.0, 1.1, 1.25, 4.0, 50.0)


def build_a2() -> Grid:
    """A2: every estimator variant on one fixed network.

    Two blocks, and the split matters for what each can show:

    **Estimator cross.** ``ollivier_norm × eta_mode × feature_norm`` on the
    as-trained network. Separates "the units were wrong" (``distance`` vs
    ``layer_scale``) from "the centring was missing" (``eta_mode``).

    The ``feature_norm`` arm supplies two *exact* identities that act as internal
    consistency checks on the implementation, both verified on the a2 results:

    1. An estimator flagged ``is_invariant`` returns the same value with and
       without feature normalisation (measured: ``distance+log_centred`` gives
       0.6008 either way; ``layer_scale+log_centred`` gives 0.5342 either way).
    2. On unit-mean-distance features the layer scale is ``s_ℓ = 1``, so
       ``layer_scale`` and ``none`` must *coincide exactly* — and they do, at
       0.1417 for raw ``η`` and 0.5342 for ``η̂``.

    What these checks do **not** say is that the different invariant estimators
    agree with each other, and it is worth being explicit because the opposite is
    tempting to assume: invariance is not uniqueness. ``1 − W₁`` on normalised
    features (a *global* per-layer normalisation) and ``1 − W₁/d(u,v)`` on raw
    features (a *local* one) are both invariant under ``G`` and are different
    statistics — measured 0.1417 against 0.6008. Which local geometry to
    normalise by is a modelling choice the invariance argument does not settle,
    so the paper has to report the choice rather than presenting one as *the*
    repaired number.

    **Rescaling sweep.** The same checkpoint under ``c_ℓ = M^ℓ``. Every cell is
    the identical classifier, so any variation across this block is measurement
    artefact by construction — it is the empirical form of Corollary T1.3, and
    the invariant estimator must return the same number in every one of them.
    """
    grid = Grid("a2")

    for ollivier_norm in ("none", "distance", "layer_scale"):
        for eta_mode in ("raw", "log_centred"):
            for feature_norm in ("none", "unit_mean_distance"):
                for seed in CONTROL_SEEDS:
                    grid.add(
                        _base(
                            "a2",
                            train=MAIN_TRAIN.with_seed(seed),
                            estimator=EstimatorConfig(
                                ollivier_norm=ollivier_norm,
                                eta_mode=eta_mode,
                                feature_norm=feature_norm,
                            ),
                            tag="estimator-cross",
                        )
                    )

    for rescale in RESCALE_LADDER:
        for estimator in ESTIMATOR_PAIR:
            grid.add(
                _base("a2", estimator=estimator, rescale=rescale, tag="rescale-sweep")
            )

    return grid


# ---------------------------------------------------------------------------
# A3 — error bars
# ---------------------------------------------------------------------------

TRAIN_SEEDS = (0, 1, 2, 3, 4)
SUBSAMPLE_SEEDS = (0, 1, 2, 3, 4)


def build_a3() -> Grid:
    """A3: five training seeds crossed with five subsample seeds.

    The cross is the point. Plan §0.6 establishes that the ρ signal lives in a
    single shared per-layer trend, so the 400 vertices inside one cell are not
    400 independent observations and cannot supply an error bar. Training seeds
    are the only legitimate source of uncertainty for the global-trend
    statistics; subsample seeds bound the graph-construction noise on top. Both
    axes are needed to say which of the two dominates — and if subsample noise
    turns out to exceed seed noise, every figure in the paper needs re-thinking.
    """
    grid = Grid("a3")
    for train_seed in TRAIN_SEEDS:
        for subsample_seed in SUBSAMPLE_SEEDS:
            for estimator in ESTIMATOR_PAIR:
                grid.add(
                    _base(
                        "a3",
                        train=MAIN_TRAIN.with_seed(train_seed),
                        graph=MAIN_GRAPH.with_subsample_seed(subsample_seed),
                        estimator=estimator,
                    )
                )
    return grid


# ---------------------------------------------------------------------------
# A4 — graph-construction sensitivity
# ---------------------------------------------------------------------------

#: One axis varied at a time from the main arm, rather than a full product.
#: A full cross of k × n × graph_type × metric is 144 cells before seeds, and it
#: would not answer a question the marginals leave open: what A4 has to settle
#: is whether the *sign* of ΔO and the ρ<0 fraction are stable, and a marginal
#: sweep answers that at a quarter of the cost. If any single axis moves a sign,
#: that axis earns a full cross of its own.
GRAPH_AXES: dict[str, tuple[GraphConfig, ...]] = {
    "k": tuple(GraphConfig(k=k, n_per_class=200) for k in (3, 5, 6, 10, 15, 20)),
    # 1000 per class is the whole Fashion-MNIST test split for a class, so this
    # axis tops out here for the test-split analysis.
    "n": tuple(GraphConfig(k=6, n_per_class=n) for n in (100, 200, 500, 1000)),
    "graph_type": tuple(
        GraphConfig(k=6, n_per_class=200, graph_type=t)
        for t in ("symmetric", "mutual", "eps_ball")
    ),
    "metric": tuple(
        GraphConfig(k=6, n_per_class=200, metric=m) for m in ("euclidean", "cosine")
    ),
}


def build_a4() -> Grid:
    """A4: does any conclusion depend on how the graph was built?"""
    grid = Grid("a4")
    for axis, variants in GRAPH_AXES.items():
        for graph in variants:
            for subsample_seed in (0, 1, 2):
                for estimator in ESTIMATOR_PAIR:
                    grid.add(
                        _base(
                            "a4",
                            graph=graph.with_subsample_seed(subsample_seed),
                            estimator=estimator,
                            tag=f"axis:{axis}",
                        )
                    )
    return grid


# ---------------------------------------------------------------------------
# B1 — fine ε-sweep
# ---------------------------------------------------------------------------

#: Plan §0.8: the existing sweep starts at ε = 0.03, where a 93%-accurate binary
#: classifier is already at 41.5% — below chance. The geometric transition is
#: somewhere in ε ∈ (0, 0.03) and was never sampled. The two coarse points are
#: kept only to connect the new sweep to the old figures.
EPSILON_GRID = (0.002, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.1)


def build_b1() -> Grid:
    """B1: the fine ε-sweep, with a norm-matched noise control at every ε.

    The controls are generated in the same loop as the attacks, at the same ε,
    so there is no ε at which an adversarial result exists without its
    perturbative counterpart. That pairing is what licenses the word
    "adversarial" in any statement about the results; leaving it to a separate
    grid is how it ends up missing from half the sweep.

    ``uniform`` with signed corners is the ℓ∞ match (see
    :func:`~arc_robustness.attacks.runner.matching_norm_for`): it reproduces the
    attack's per-pixel magnitude exactly, differing only in direction.
    """
    grid = Grid("b1")
    grid.add(_base("b1", tag="clean"))  # ε = 0 anchor

    for epsilon in EPSILON_GRID:
        arms = (
            AttackConfig(kind="fgsm", epsilon=epsilon),
            AttackConfig(kind="pgd_linf", epsilon=epsilon, steps=20),
            AttackConfig(kind="uniform", epsilon=epsilon, seed=0),
        )
        for attack in arms:
            grid.add(_base("b1", attack=attack, tag=f"eps{epsilon:g}"))
    return grid


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GRID_BUILDERS: dict[str, Callable[[], Grid]] = {
    "smoke": build_smoke,
    "a1": build_a1,
    "a2": build_a2,
    "a3": build_a3,
    "a4": build_a4,
    "b1": build_b1,
}


def build_grid(name: str) -> Grid:
    """Build the named grid, or raise with the list of valid names."""
    if name not in GRID_BUILDERS:
        raise KeyError(f"unknown grid {name!r}; available: {sorted(GRID_BUILDERS)}")
    return GRID_BUILDERS[name]()


def list_grids() -> list[tuple[str, int, int]]:
    """``(name, n_cells, n_checkpoints)`` for every registered grid."""
    out = []
    for name in GRID_BUILDERS:
        grid = build_grid(name)
        out.append((name, len(grid), len(grid.unique_models())))
    return out
