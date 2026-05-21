"""
Train a simple dense neural model on MNIST (classes 0, 1, 2), to be used as a base
model for testing robustness to adversarial attacks.  Run this script directly to
download MNIST, train for EPOCHS epochs, and save the weights to WEIGHTS_PATH.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from arc_robustness.training.model import (
    CLASSES,
    DATA_DIR,
    WEIGHTS_PATH,
    DigitClassifier,
    filter_to_classes,
    remap_labels,
)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
NORMALISE = transforms.Normalize((0.1307,), (0.3081,))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train() -> None:
    """Train the model and save the weights."""
    transform = transforms.Compose([transforms.ToTensor(), NORMALISE])

    train_loader = DataLoader(
        filter_to_classes(
            torchvision.datasets.MNIST(
                DATA_DIR, train=True, download=True, transform=transform
            ),
            CLASSES,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        filter_to_classes(
            torchvision.datasets.MNIST(
                DATA_DIR, train=False, download=True, transform=transform
            ),
            CLASSES,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on {device}")

    model = DigitClassifier(num_classes=len(CLASSES)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    epoch_bar = tqdm(range(EPOCHS), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0
        batch_bar = tqdm(
            train_loader,
            desc=f"  Train {epoch + 1:>2}/{EPOCHS}",
            leave=False,
            unit="batch",
        )
        for images, labels in batch_bar:
            images = images.to(device)
            labels = remap_labels(labels, CLASSES).to(device)
            optimiser.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in tqdm(
                val_loader,
                desc=f"  Val   {epoch + 1:>2}/{EPOCHS}",
                leave=False,
                unit="batch",
            ):
                images = images.to(device)
                labels = remap_labels(labels, CLASSES).to(device)
                correct += (model(images).argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

        avg_loss = running_loss / len(train_loader)
        val_acc = correct / total
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", val_acc=f"{val_acc:.4f}")
        tqdm.write(
            f"Epoch {epoch + 1:>2}/{EPOCHS} | loss: {avg_loss:.4f} | val acc: {val_acc:.4f}"
        )

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"\nWeights saved to {WEIGHTS_PATH}")


if __name__ == "__main__":
    train()
