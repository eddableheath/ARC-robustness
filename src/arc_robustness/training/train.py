"""
Config-driven training.

One entry point, :func:`train_model`, covers the main arm and every A1 control:

===========================  ==========================================
Control                      Config
===========================  ==========================================
Trained network (main arm)   defaults
Random init (untrained)      ``TrainConfig(trained=False)``
Shuffled labels              ``DataConfig(label_mode="shuffled")``
Linear network               ``ModelConfig(activation="identity")``
Memorised random data        ``DataConfig(label_mode="random_data")``
===========================  ==========================================

Checkpoints are content-addressed by ``config.model_uid``, so the many cells
that share a network (every estimator variant, every graph seed) train it once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from arc_robustness.config import ExperimentConfig
from arc_robustness.data import load_dataset, make_loader
from arc_robustness.training.architectures import MLP, init_model


def pick_device(prefer: str = "auto") -> torch.device:
    """Select a compute device.

    On Isambard-AI this resolves to CUDA (GH200); on the development Mac to
    MPS. The geometry pipeline is CPU-bound regardless — only training and
    attack generation use the accelerator.
    """
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainResult:
    train_acc: float
    test_acc: float
    final_loss: float
    epochs_run: int
    history: list[dict[str, float]]

    def to_dict(self) -> dict:
        return {
            "train_acc": self.train_acc,
            "test_acc": self.test_acc,
            "final_loss": self.final_loss,
            "epochs_run": self.epochs_run,
            "history": self.history,
        }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> float:
    """Top-1 accuracy on raw ``[0, 1]`` images."""
    model.eval()
    correct = 0
    for start in range(0, len(labels), batch_size):
        x = images[start : start + batch_size].to(device)
        y = labels[start : start + batch_size].to(device)
        correct += (model(x).argmax(dim=1) == y).sum().item()
    return correct / len(labels)


def train_model(
    config: ExperimentConfig,
    device: torch.device | None = None,
    data_dir: Path | None = None,
    verbose: bool = True,
) -> tuple[MLP, TrainResult]:
    """Train (or merely initialise) the network described by *config*."""
    device = pick_device() if device is None else device
    cfg_train = config.train

    model = init_model(config.model, config.data, seed=cfg_train.seed).to(device)

    train_images, train_labels = load_dataset(
        config.data, split="train", data_dir=data_dir, seed=cfg_train.seed
    )
    test_images, test_labels = load_dataset(
        config.data, split="test", data_dir=data_dir, seed=cfg_train.seed
    )

    if not cfg_train.trained:
        # Random-init control: report the accuracy of the untrained network so
        # the falsification table can show it sitting at chance.
        return model, TrainResult(
            train_acc=evaluate(model, train_images, train_labels, device),
            test_acc=evaluate(model, test_images, test_labels, device),
            final_loss=float("nan"),
            epochs_run=0,
            history=[],
        )

    loader = make_loader(
        train_images,
        train_labels,
        batch_size=cfg_train.batch_size,
        shuffle=True,
        seed=cfg_train.seed,
    )
    optimiser = torch.optim.Adam(
        model.parameters(), lr=cfg_train.lr, weight_decay=cfg_train.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    epochs_run = 0
    running_loss = float("nan")

    for epoch in range(cfg_train.epochs):
        model.train()
        total = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimiser.step()
            total += loss.item()
        running_loss = total / len(loader)
        epochs_run = epoch + 1

        train_acc = evaluate(model, train_images, train_labels, device)
        test_acc = evaluate(model, test_images, test_labels, device)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": running_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
            }
        )
        if verbose:
            print(
                f"  epoch {epoch + 1:>3}/{cfg_train.epochs}  "
                f"loss {running_loss:.4f}  train {train_acc:.4f}  test {test_acc:.4f}",
                flush=True,
            )

        if (
            cfg_train.target_train_acc is not None
            and train_acc >= cfg_train.target_train_acc
        ):
            if verbose:
                print(
                    f"  reached target train acc {cfg_train.target_train_acc:.3f}, stopping",
                    flush=True,
                )
            break

    return model, TrainResult(
        train_acc=history[-1]["train_acc"] if history else float("nan"),
        test_acc=history[-1]["test_acc"] if history else float("nan"),
        final_loss=running_loss,
        epochs_run=epochs_run,
        history=history,
    )


def save_checkpoint(
    model: MLP, config: ExperimentConfig, result: TrainResult, path: Path
) -> None:
    """Write weights plus the config that produced them.

    Storing the config alongside the weights means a checkpoint can always be
    reloaded into the right architecture without consulting an external ledger,
    and makes an accidental architecture mismatch impossible to miss.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config.to_dict(),
            "model_uid": config.model_uid,
            "result": result.to_dict(),
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "model_uid": config.model_uid,
                "description": config.describe(),
                "result": result.to_dict(),
            },
            indent=2,
        )
    )


def load_checkpoint(
    path: Path, config: ExperimentConfig, device: torch.device | None = None
) -> MLP:
    """Load weights saved by :func:`save_checkpoint`, verifying provenance."""
    device = pick_device() if device is None else device
    blob = torch.load(path, map_location=device, weights_only=False)
    if blob.get("model_uid") != config.model_uid:
        raise ValueError(
            f"checkpoint {path} was produced by model_uid "
            f"{blob.get('model_uid')!r}, but this config wants "
            f"{config.model_uid!r}"
        )
    model = init_model(config.model, config.data, seed=config.train.seed).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def get_or_train(
    config: ExperimentConfig,
    weights_dir: Path | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> MLP:
    """Return the checkpoint for *config*, training it if absent.

    Idempotent: safe to call from every array element of a Slurm job. Note that
    concurrent first-time calls could race, which is why the sweep runs a
    dedicated training stage before the analysis stage rather than relying on
    lazy training inside analysis jobs.
    """
    path = config.checkpoint_path(weights_dir)
    if path.exists():
        return load_checkpoint(path, config, device)
    model, result = train_model(config, device=device, verbose=verbose)
    save_checkpoint(model, config, result, path)
    return model
