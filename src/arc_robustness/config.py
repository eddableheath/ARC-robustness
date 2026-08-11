"""
Experiment configuration.

Everything an experiment needs is captured in a frozen, hashable
:class:`ExperimentConfig`.  Two properties matter for the HPC workflow:

* **Deterministic ordering.** A grid expands to a list of configs in a fixed
  order, so ``SLURM_ARRAY_TASK_ID`` maps to the same config on every
  submission.  Re-running array element 37 always re-runs the same cell.
* **Content-addressed output.** Each config has a short ``uid`` derived from
  its contents.  Results are written to ``<results>/<experiment>/<uid>.npz``,
  so jobs never collide, and a completed cell can be skipped on resubmission
  without keeping a separate ledger.

Config objects are the *only* place experimental parameters live.  Nothing in
``analysis/`` or ``attacks/`` reads globals.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
FEATURES_DIR: Path = PROJECT_ROOT / "features"
WEIGHTS_DIR: Path = PROJECT_ROOT / "weights"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
RESULTS_DIR: Path = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Dataset descriptors
# ---------------------------------------------------------------------------

#: Per-dataset channel statistics used for input normalisation, and the
#: raw input dimension after flattening.
DATASET_SPECS: dict[str, dict[str, Any]] = {
    "fashion_mnist": {"mean": (0.2860,), "std": (0.3530,), "shape": (1, 28, 28)},
    "mnist": {"mean": (0.1307,), "std": (0.3081,), "shape": (1, 28, 28)},
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "shape": (3, 32, 32),
    },
}

#: Human-readable names for the class pairs used in the paper and in our runs.
CLASS_NAMES: dict[str, dict[int, str]] = {
    "fashion_mnist": {
        0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
        5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot",
    },
    "mnist": {i: str(i) for i in range(10)},
    "cifar10": {
        0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
        5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck",
    },
}


# ---------------------------------------------------------------------------
# Component configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataConfig:
    """Which data the network sees, and how the labels are assigned.

    ``label_mode`` drives two of the A1 controls:

    * ``"shuffled"`` — real images, labels permuted before training. The
      network can only memorise; any Ricci signature it shows is not about
      class structure.
    * ``"random_data"`` — Gaussian noise inputs with random labels, trained to
      fit. Isolates memorisation from natural image statistics entirely.
    """

    dataset: str = "fashion_mnist"
    classes: tuple[int, ...] = (2, 6)
    label_mode: str = "true"  # true | shuffled | random_data
    split: str = "test"  # train | test | both
    normalise: bool = True

    def __post_init__(self) -> None:
        if self.dataset not in DATASET_SPECS:
            raise ValueError(f"unknown dataset {self.dataset!r}")
        if self.label_mode not in {"true", "shuffled", "random_data"}:
            raise ValueError(f"unknown label_mode {self.label_mode!r}")
        if self.split not in {"train", "test", "both"}:
            raise ValueError(f"unknown split {self.split!r}")
        if len(self.classes) < 2:
            raise ValueError("need at least two classes")

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(DATASET_SPECS[self.dataset]["shape"])

    @property
    def input_dim(self) -> int:
        shape = self.input_shape
        return shape[0] * shape[1] * shape[2]

    @property
    def class_names(self) -> tuple[str, ...]:
        names = CLASS_NAMES[self.dataset]
        return tuple(names[c] for c in self.classes)

    def label(self) -> str:
        """Short slug, e.g. ``fashion_mnist-2v6`` or ``mnist-1v7-shuffled``."""
        pair = "v".join(str(c) for c in self.classes)
        slug = f"{self.dataset}-{pair}"
        if self.label_mode != "true":
            slug += f"-{self.label_mode}"
        return slug


@dataclass(frozen=True)
class ModelConfig:
    """Architecture.

    ``activation="identity"`` collapses the network to a composition of affine
    maps — the linear-net control of A1. It is also the one case where the
    positive-rescaling group ``G`` of T1 acts differently, since there is no
    ReLU homogeneity to exploit; the rescaling is then a pure reparameterisation
    of an affine map.

    ``norm`` matters for T1's falsifiable prediction: BatchNorm and LayerNorm
    quotient out the per-layer rescaling symmetry, so normalised architectures
    should show the raw and invariant estimators *agree*.
    """

    arch: str = "mlp"  # mlp | cnn
    widths: tuple[int, ...] = (512, 512, 256, 256, 128, 64)
    activation: str = "relu"  # relu | identity
    norm: str = "none"  # none | batch | layer
    bias: bool = True

    def __post_init__(self) -> None:
        if self.arch not in {"mlp", "cnn"}:
            raise ValueError(f"unknown arch {self.arch!r}")
        if self.activation not in {"relu", "identity"}:
            raise ValueError(f"unknown activation {self.activation!r}")
        if self.norm not in {"none", "batch", "layer"}:
            raise ValueError(f"unknown norm {self.norm!r}")
        if not self.widths:
            raise ValueError("need at least one hidden layer")

    @property
    def depth(self) -> int:
        """Number of hidden layers, i.e. the number of analysed feature spaces."""
        return len(self.widths)

    @property
    def layer_names(self) -> tuple[str, ...]:
        """Names of the analysed post-activation feature spaces."""
        return tuple(f"relu{i + 1}" for i in range(self.depth))

    def label(self) -> str:
        slug = f"{self.arch}-d{self.depth}-w{self.widths[0]}"
        if self.activation != "relu":
            slug += f"-{self.activation}"
        if self.norm != "none":
            slug += f"-{self.norm}norm"
        return slug


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation.

    ``trained=False`` skips training entirely, giving the random-init control
    of A1. ``seed`` drives weight init, data shuffling and label permutation,
    and is the axis over which A3's error bars are computed — the *only*
    legitimate source of uncertainty for the global-trend statistics
    (see plan section 0.6).
    """

    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 0
    trained: bool = True
    #: Stop early once train accuracy reaches this, for the memorisation
    #: controls that must reach 100% to be meaningful. None = run all epochs.
    target_train_acc: float | None = None
    #: Epochs at which to checkpoint, for C3's training-dynamics sweep.
    checkpoint_epochs: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.epochs < 0:
            raise ValueError("epochs must be non-negative")

    def with_seed(self, seed: int) -> TrainConfig:
        """Copy with a different training seed — the A3 error-bar axis."""
        return replace(self, seed=seed)

    def label(self) -> str:
        return "untrained" if not self.trained else f"e{self.epochs}-s{self.seed}"


@dataclass(frozen=True)
class GraphConfig:
    """k-NN graph construction — the A4 sensitivity axes.

    ``graph_type``:

    * ``"symmetric"`` — the OR rule, ``A = (K + Kᵀ) > 0``. Degrees are ``>= k``,
      which is what makes Forman curvature degenerate (T7).
    * ``"mutual"`` — the AND rule, ``A = K & Kᵀ``. Degrees are ``<= k`` and the
      graph fragments more readily.
    * ``"eps_ball"`` — connect all pairs within ``eps_quantile`` of the pairwise
      distance distribution. Degree is then genuinely feature-dependent.
    """

    k: int = 6
    n_per_class: int = 200
    graph_type: str = "symmetric"  # symmetric | mutual | eps_ball
    metric: str = "euclidean"  # euclidean | cosine
    eps_quantile: float = 0.02
    #: Seed for choosing which points enter the graph. Distinct from the
    #: training seed: A3 needs >=5 of each, crossed.
    subsample_seed: int = 0

    def __post_init__(self) -> None:
        if self.graph_type not in {"symmetric", "mutual", "eps_ball"}:
            raise ValueError(f"unknown graph_type {self.graph_type!r}")
        if self.metric not in {"euclidean", "cosine"}:
            raise ValueError(f"unknown metric {self.metric!r}")
        if self.k < 1:
            raise ValueError("k must be positive")

    def with_subsample_seed(self, seed: int) -> GraphConfig:
        """Copy with a different subsample seed.

        Kept separate from the training seed throughout: A3 needs the two
        crossed, and conflating them would make graph-construction noise
        indistinguishable from training noise.
        """
        return replace(self, subsample_seed=seed)

    def label(self) -> str:
        slug = f"k{self.k}-n{self.n_per_class}"
        if self.graph_type != "symmetric":
            slug += f"-{self.graph_type}"
        if self.metric != "euclidean":
            slug += f"-{self.metric}"
        return slug


@dataclass(frozen=True)
class EstimatorConfig:
    """Which estimator variant to compute — the heart of A2.

    ``ollivier_norm``:

    * ``"none"`` — ``κ = 1 - W₁``, reproducing this repo's original
      implementation. Not scale-free: ``W₁`` is in activation units while the
      implied ``d(u,v)`` is 1 hop. Kept so the paper can report what the
      original convention gives.
    * ``"distance"`` — ``κ̂ = 1 - W₁/d_euclid(u,v)``, matched units. The
      repaired estimator of T1.4, invariant under per-layer rescaling.
    * ``"layer_scale"`` — ``1 - W₁/s_ℓ`` where ``s_ℓ`` is the layer's mean
      pairwise distance. Also invariant, but a *global* rather than local
      normalisation; included to separate "units were wrong" from "locality
      was wrong".

    ``eta_mode``:

    * ``"raw"`` — ``η = mean_y[d_{ℓ+1} - d_ℓ]``, as originally implemented.
      Provably sign-manipulable (T1.2).
    * ``"log_centred"`` — ``η̂``, the per-layer-centred mean log ratio of T1.4.
      Invariant.

    ``feature_norm="unit_mean_distance"`` rescales each layer's activations to
    unit mean pairwise distance *before* anything else. This is the brute-force
    way to kill the rescaling symmetry and should make ``raw`` and repaired
    estimators agree — a useful internal consistency check on T1.
    """

    ollivier_norm: str = "distance"  # none | distance | layer_scale
    eta_mode: str = "log_centred"  # raw | log_centred
    feature_norm: str = "none"  # none | unit_mean_distance
    #: Compute Ollivier curvature at all. Forman/AF and community metrics are
    #: cheap; Ollivier dominates runtime.
    compute_ollivier: bool = True

    def __post_init__(self) -> None:
        if self.ollivier_norm not in {"none", "distance", "layer_scale"}:
            raise ValueError(f"unknown ollivier_norm {self.ollivier_norm!r}")
        if self.eta_mode not in {"raw", "log_centred"}:
            raise ValueError(f"unknown eta_mode {self.eta_mode!r}")
        if self.feature_norm not in {"none", "unit_mean_distance"}:
            raise ValueError(f"unknown feature_norm {self.feature_norm!r}")

    @property
    def is_invariant(self) -> bool:
        """True if every reported quantity is invariant under the group G of T1."""
        if self.feature_norm == "unit_mean_distance":
            return True
        return self.ollivier_norm in {"distance", "layer_scale"} and (
            self.eta_mode == "log_centred"
        )

    def label(self) -> str:
        slug = f"o-{self.ollivier_norm}+e-{self.eta_mode}"
        if self.feature_norm != "none":
            slug += f"+f-{self.feature_norm}"
        return slug


@dataclass(frozen=True)
class AttackConfig:
    """Perturbation to apply before feature extraction.

    ``kind="none"`` is the clean baseline. ``kind="gaussian"``/``"uniform"``
    are the norm-matched random-noise controls that B2 requires: without them
    no observed change can be attributed to *adversariality* rather than to
    perturbation per se.
    """

    kind: str = "none"  # none | fgsm | pgd_linf | pgd_l2 | gaussian | uniform
    epsilon: float = 0.0
    steps: int = 1
    step_size: float | None = None  # None = epsilon/steps * 2.5 (standard PGD)
    random_start: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        valid = {"none", "fgsm", "pgd_linf", "pgd_l2", "gaussian", "uniform"}
        if self.kind not in valid:
            raise ValueError(f"unknown attack kind {self.kind!r}")
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")

    @property
    def is_adversarial(self) -> bool:
        return self.kind in {"fgsm", "pgd_linf", "pgd_l2"}

    def label(self) -> str:
        if self.kind == "none":
            return "clean"
        return f"{self.kind}-eps{self.epsilon:g}"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    """A single computational cell: one model, one graph, one estimator, one attack.

    This is the unit of work a Slurm array element performs.
    """

    experiment: str  # a1 | a2 | a3 | a4 | b1 | smoke
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    #: Function-preserving rescaling applied to the *loaded* network, with
    #: ``c_ℓ = rescale**ℓ``. This is the constructive engine of T1.2 turned into
    #: an experimental axis: every value gives the identical classifier — same
    #: predictions, same accuracy, same adversarial examples — while moving the
    #: non-invariant estimators arbitrarily. It is deliberately *outside*
    #: ``model_uid``, so the whole sweep over ``rescale`` reuses one checkpoint;
    #: retraining per value would be both wasteful and misleading, since the
    #: point is that the weights are untouched by anything except the rescaling.
    rescale: float = 1.0
    #: Free-form tag distinguishing cells that are otherwise identical, e.g.
    #: the arm of a control comparison. Participates in the uid.
    tag: str = ""

    def __post_init__(self) -> None:
        if self.rescale <= 0:
            raise ValueError("rescale must be strictly positive")
        if self.rescale != 1.0 and self.model.activation != "relu":
            raise ValueError(
                "the rescaling symmetry relies on ReLU positive homogeneity; "
                f"activation is {self.model.activation!r}"
            )
        if self.rescale != 1.0 and self.model.norm != "none":
            raise ValueError(
                "normalisation quotients out the rescaling symmetry, so there "
                "is nothing to rescale; this is T1's prediction for C1"
            )

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        """Rebuild from :meth:`to_dict` output, coercing lists back to tuples."""

        def _tuples(section: dict[str, Any], keys: Sequence[str]) -> dict[str, Any]:
            out = dict(section)
            for key in keys:
                if key in out and out[key] is not None:
                    out[key] = tuple(out[key])
            return out

        return cls(
            experiment=d["experiment"],
            data=DataConfig(**_tuples(d["data"], ["classes"])),
            model=ModelConfig(**_tuples(d["model"], ["widths"])),
            train=TrainConfig(**_tuples(d["train"], ["checkpoint_epochs"])),
            graph=GraphConfig(**d["graph"]),
            estimator=EstimatorConfig(**d["estimator"]),
            attack=AttackConfig(**d["attack"]),
            rescale=d.get("rescale", 1.0),
            tag=d.get("tag", ""),
        )

    def canonical_json(self) -> str:
        """Stable JSON encoding — the basis of :attr:`uid`."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def uid(self) -> str:
        """Short content hash. Stable across machines and Python versions."""
        digest = hashlib.sha256(self.canonical_json().encode()).hexdigest()
        return digest[:12]

    # -- derived identities -----------------------------------------------

    @property
    def model_uid(self) -> str:
        """Hash of only the parts that determine the trained weights.

        Many cells share a checkpoint — every estimator variant and every
        graph seed reuses the same network. Training once per ``model_uid``
        instead of once per cell is the difference between a tractable sweep
        and a wasteful one.
        """
        payload = json.dumps(
            {
                "data": asdict(self.data),
                "model": asdict(self.model),
                "train": asdict(self.train),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def checkpoint_path(self, weights_dir: Path | None = None) -> Path:
        root = WEIGHTS_DIR if weights_dir is None else weights_dir
        return root / "sweep" / f"{self.model_uid}.pt"

    def result_path(self, results_dir: Path | None = None) -> Path:
        root = RESULTS_DIR if results_dir is None else results_dir
        return root / self.experiment / f"{self.uid}.npz"

    @property
    def rescale_factors(self) -> tuple[float, ...]:
        """Per-layer factors ``c_ℓ = rescale**ℓ`` for :meth:`MLP.rescale_layers`.

        A geometric ladder is the right family to sweep because it makes the
        *ratio* ``c_{ℓ+1}/c_ℓ`` constant at ``rescale``, and it is that ratio —
        not the absolute scale — that sets the sign of the raw ``η_ℓ``. So one
        scalar traverses the whole construction of T1.2.
        """
        return tuple(self.rescale ** (i + 1) for i in range(self.model.depth))

    def describe(self) -> str:
        """One-line human-readable summary for logs."""
        parts = [
            self.experiment,
            self.data.label(),
            self.model.label(),
            self.train.label(),
            self.graph.label(),
            self.estimator.label(),
            self.attack.label(),
        ]
        if self.rescale != 1.0:
            parts.append(f"rescale{self.rescale:g}")
        if self.tag:
            parts.append(self.tag)
        return " | ".join(parts)

    def with_(self, **kwargs: Any) -> ExperimentConfig:
        """Return a copy with top-level fields replaced."""
        return replace(self, **kwargs)


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------


class Grid:
    """An ordered, deduplicated list of configs addressable by integer index.

    The ordering is the insertion order of :meth:`add`, with duplicates (by
    ``uid``) dropped on first-wins. That makes ``grid[i]`` reproducible so long
    as the grid-building code is unchanged — which is why grid definitions live
    in version control and are never generated from mutable state.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._configs: list[ExperimentConfig] = []
        self._seen: set[str] = set()

    def add(self, config: ExperimentConfig) -> None:
        if config.uid not in self._seen:
            self._seen.add(config.uid)
            self._configs.append(config)

    def extend(self, configs: Sequence[ExperimentConfig]) -> None:
        for config in configs:
            self.add(config)

    def __len__(self) -> int:
        return len(self._configs)

    def __getitem__(self, index: int) -> ExperimentConfig:
        return self._configs[index]

    def __iter__(self) -> Iterator[ExperimentConfig]:
        return iter(self._configs)

    @property
    def configs(self) -> list[ExperimentConfig]:
        return list(self._configs)

    def model_uids(self) -> list[str]:
        """Distinct checkpoints this grid needs, in first-appearance order."""
        seen: set[str] = set()
        out: list[str] = []
        for config in self._configs:
            if config.model_uid not in seen:
                seen.add(config.model_uid)
                out.append(config.model_uid)
        return out

    def unique_models(self) -> list[ExperimentConfig]:
        """One representative config per distinct checkpoint."""
        seen: set[str] = set()
        out: list[ExperimentConfig] = []
        for config in self._configs:
            if config.model_uid not in seen:
                seen.add(config.model_uid)
                out.append(config)
        return out

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "index": i,
                "uid": c.uid,
                "model_uid": c.model_uid,
                "description": c.describe(),
                "config": c.to_dict(),
            }
            for i, c in enumerate(self._configs)
        ]

    def write_manifest(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"grid": self.name, "size": len(self), "cells": self.manifest()},
                indent=2,
            )
        )
