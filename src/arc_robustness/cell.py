"""
The cell runner: one :class:`ExperimentConfig` in, one ``.npz`` out.

This is the join between the library and the scheduler. A *cell* is the unit of
work one Slurm array element performs, and :func:`run_cell` is the whole of it:
load or train the network, subsample, perturb, extract features, analyse, write.

Three properties are load-bearing for the HPC workflow:

**Idempotence.** A cell whose result file exists is skipped. Resubmitting a
partially-failed array costs only the missing cells, and there is no ledger to
keep in sync — the filesystem *is* the ledger, keyed by the content hash of the
config.

**Order of operations.** Subsampling happens *before* perturbation, not after.
The attack is then computed only for the points that enter the graph, which for
``n_per_class=200`` against a 2000-point test split is a 5× saving on the most
expensive GPU step. It also makes the adversarial arm's vertex set identical to
the clean arm's, which is what allows a paired comparison at all.

**Provenance over derivation.** The record stores ingredients, not conclusions:
per-layer spectral norms rather than T4's stability radius, per-example logits
rather than an accuracy. Anything derivable is derived at analysis time, so
correcting a formula in the theory does not mean re-running the sweep — which
matters when the sweep is the expensive part and the theory is the part still
being written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from arc_robustness.analysis.pipeline import analyse_features
from arc_robustness.attacks.runner import apply_attack
from arc_robustness.config import ExperimentConfig
from arc_robustness.data import balanced_subsample_indices, load_dataset
from arc_robustness.detection.baselines import margin_score
from arc_robustness.features import extract_features
from arc_robustness.training.architectures import MLP
from arc_robustness.training.train import get_or_train, pick_device

# ---------------------------------------------------------------------------
# Weight geometry
# ---------------------------------------------------------------------------


@torch.no_grad()
def weight_spectra(model: MLP) -> dict[str, np.ndarray]:
    """Extremal singular values of each hidden layer's weight matrix.

    Recorded per cell because two separate strands of the theory need them and
    neither can recover them from the metrics:

    * **T4.3** bounds the layer-``ℓ`` feature displacement by
      ``ε·√d·∏_{j≤ℓ} s_max(W_j)``, so the stability radius is a function of these
      numbers and the k/(k+1) gap that :func:`analyse_features` already records.
    * **T3.2** bounds the achievable spread of log-distance changes by
      ``log cond(W_ℓ) = log(s_max/s_min)`` — the quantity that puts the Ricci
      signature in tension with Lipschitz-based robustness.

    Storing the singular values rather than either derived bound keeps the sweep
    independent of the exact form those bounds end up taking.
    """
    s_max, s_min = [], []
    for linear in model.linears:
        singular = torch.linalg.svdvals(linear.weight.detach().float().cpu())
        s_max.append(float(singular[0]))
        s_min.append(float(singular[-1]))
    return {
        "spectral_norm_max": np.array(s_max, dtype=np.float64),
        "spectral_norm_min": np.array(s_min, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Perturbation bookkeeping
# ---------------------------------------------------------------------------


def perturbation_norms(clean: torch.Tensor, perturbed: torch.Tensor) -> dict[str, Any]:
    """Per-example realised ``ℓ∞`` and ``ℓ₂`` norms of the perturbation.

    *Realised*, not requested: clipping to ``[0, 1]`` means the delivered norm is
    at or below ``ε``, and the gap differs systematically between an ℓ∞ attack
    and an ℓ₂-matched noise control. B2's claim is that two arms are matched in
    norm, and that claim has to be checkable from the results file rather than
    asserted from the config.
    """
    delta = (perturbed - clean).flatten(start_dim=1)
    return {
        "perturbation_linf": delta.abs().amax(dim=1).cpu().numpy().astype(np.float64),
        "perturbation_l2": delta.norm(dim=1).cpu().numpy().astype(np.float64),
    }


@torch.no_grad()
def _forward_logits(
    model: MLP, images: torch.Tensor, device: torch.device, batch_size: int = 512
) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        out.append(model(batch).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# The cell
# ---------------------------------------------------------------------------


def analyse_cell(
    config: ExperimentConfig,
    model: MLP,
    device: torch.device,
    data_dir: Path | None = None,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Compute one cell's full record from an already-loaded model.

    Split out from :func:`run_cell` so the computation can be exercised without
    touching the weights or results directories.
    """
    images, labels = load_dataset(
        config.data, data_dir=data_dir, seed=config.train.seed
    )

    # Subsample first: the attack, the feature extraction and the graph then all
    # operate on the same vertex set, and the expensive steps see 200 points per
    # class rather than the whole split.
    indices = balanced_subsample_indices(
        labels, config.graph.n_per_class, config.graph.subsample_seed
    )
    images = images[indices]
    labels = labels[indices]

    clean_logits = _forward_logits(model, images, device)
    perturbed = apply_attack(config.attack, model, images, labels, device)
    features, logits = extract_features(model, perturbed, device)

    labels_np = labels.numpy().astype(np.int64)
    predictions = logits.argmax(axis=1)
    clean_predictions = clean_logits.argmax(axis=1)

    record = analyse_features(
        features,
        labels_np,
        config.graph,
        config.estimator,
        layer_names=list(config.model.layer_names),
        n_jobs=n_jobs,
    )

    record.update(weight_spectra(model))
    record.update(perturbation_norms(images, perturbed))
    record.update(
        {
            "subsample_indices": indices,
            "input_dim": config.data.input_dim,
            # Per-example, so that B3 can split by outcome and B4 can regress
            # against the margin without another forward pass.
            "logits": logits,
            "clean_logits": clean_logits,
            "predictions": predictions,
            "clean_predictions": clean_predictions,
            "margin": margin_score(logits),
            "clean_margin": margin_score(clean_logits),
            "correct": (predictions == labels_np),
            "clean_correct": (clean_predictions == labels_np),
            # Scalars are conveniences for scanning result files; every one of
            # them is recoverable from the per-example arrays above.
            "accuracy": float((predictions == labels_np).mean()),
            "clean_accuracy": float((clean_predictions == labels_np).mean()),
            "flipped_frac": float(
                ((clean_predictions == labels_np) & (predictions != labels_np)).mean()
            ),
        }
    )
    return record


def _checkpoint_metadata(config: ExperimentConfig, weights_dir: Path | None) -> dict:
    """Training accuracies recorded alongside the checkpoint, if available.

    Carried into every result file because the A1 falsification table has to
    state that the memorisation arms actually memorised. A shuffled-label arm
    sitting at 70% train accuracy is not a control, and that has to be visible
    from the result rather than requiring a look at the checkpoint directory.
    """
    sidecar = config.checkpoint_path(weights_dir).with_suffix(".json")
    if not sidecar.exists():
        return {}
    blob = json.loads(sidecar.read_text())
    result = blob.get("result", {})
    return {
        "train_acc": float(result.get("train_acc", np.nan)),
        "test_acc": float(result.get("test_acc", np.nan)),
        "epochs_run": int(result.get("epochs_run", 0)),
    }


def run_cell(
    config: ExperimentConfig,
    weights_dir: Path | None = None,
    results_dir: Path | None = None,
    data_dir: Path | None = None,
    device: torch.device | None = None,
    n_jobs: int = 1,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Run one cell and write ``<results>/<experiment>/<uid>.npz``.

    Returns the result path, whether it was computed now or already present.
    """
    path = config.result_path(results_dir)
    if path.exists() and not overwrite:
        if verbose:
            print(f"skip (exists) {config.uid}  {config.describe()}", flush=True)
        return path

    device = pick_device() if device is None else device
    if verbose:
        print(f"run  {config.uid}  {config.describe()}  [{device}]", flush=True)

    model = get_or_train(config, weights_dir=weights_dir, device=device, verbose=verbose)
    if config.rescale != 1.0:
        # In place, on a freshly loaded copy: the checkpoint on disk is never
        # modified, so the many rescale cells sharing one model_uid stay
        # independent of each other and of their execution order.
        model.rescale_layers(config.rescale_factors)

    record = analyse_cell(
        config, model, device, data_dir=data_dir, n_jobs=n_jobs
    )
    record.update(_checkpoint_metadata(config, weights_dir))
    record.update(
        {
            "uid": config.uid,
            "model_uid": config.model_uid,
            "experiment": config.experiment,
            "tag": config.tag,
            "description": config.describe(),
            "config_json": config.canonical_json(),
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in record.items()})

    # A JSON sidecar of just the scalars, so a sweep can be surveyed with grep
    # and jq on a login node without loading several hundred npz files.
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "uid": config.uid,
                "model_uid": config.model_uid,
                "description": config.describe(),
                "config": config.to_dict(),
                "scalars": {
                    key: _jsonable(record[key])
                    for key in (
                        "accuracy", "clean_accuracy", "flipped_frac",
                        "train_acc", "test_acc",
                        "frac_rho_negative", "frac_rho_centred_negative",
                        "global_trend_correlation", "modularity_ceiling",
                        "estimator_is_invariant",
                    )
                    if key in record
                },
            },
            indent=2,
        )
    )

    if verbose:
        print(f"  wrote {path}", flush=True)
    return path


def _jsonable(value: Any) -> Any:
    """Coerce numpy scalars and bools to plain JSON types."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_record(path: Path) -> dict[str, Any]:
    """Read a result file back as a plain dict.

    ``np.load`` returns a lazy, file-backed mapping whose entries are all
    zero-dimensional arrays for the scalars; unwrapping them here means analysis
    code can treat a record identically whether it came from
    :func:`analyse_cell` in memory or from disk.
    """
    with np.load(path, allow_pickle=False) as blob:
        out: dict[str, Any] = {}
        for key in blob.files:
            value = blob[key]
            out[key] = value.item() if value.ndim == 0 else value
    return out
