"""
Shared model definition and helpers used by both the training and feature-extraction
scripts.
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Subset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSES: list[int] = [0, 1, 2]

# Paths are anchored to the project root (four levels up from this file):
#   training/ -> arc-robustness/ -> src/ -> ARC-robustness/
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
WEIGHTS_PATH: Path = _PROJECT_ROOT / "weights" / "base_model.pt"
DATA_DIR: Path = _PROJECT_ROOT / "data"
FEATURES_DIR: Path = _PROJECT_ROOT / "features"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DigitClassifier(nn.Module):
    """Four-hidden-layer dense classifier for MNIST digit recognition."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        return self.fc5(x)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def filter_to_classes(dataset, classes: list[int]) -> Subset:
    """Return a Subset of *dataset* containing only samples from *classes*."""
    targets = torch.as_tensor(dataset.targets)
    mask = torch.isin(targets, torch.tensor(classes))
    indices = torch.where(mask)[0].tolist()
    return Subset(dataset, indices)


def remap_labels(labels: torch.Tensor, classes: list[int]) -> torch.Tensor:
    """Map original class values to contiguous 0-based indices."""
    label_map = {c: i for i, c in enumerate(classes)}
    return torch.tensor([label_map[int(i)] for i in labels], dtype=torch.long)
