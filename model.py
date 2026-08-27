"""Ordinary PyTorch and MNIST code shared by data, train, and inference scripts.

This file does not import W&B. Integration stays in the W&B-facing orchestration
scripts so the boundary is easy to see.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


MNIST_MEAN = 0.1307
MNIST_STANDARD_DEVIATION = 0.3081


class MNISTCNN(nn.Module):
    """A small convolutional network for 28 x 28 grayscale images."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def select_device() -> torch.device:
    """Use CUDA or Apple silicon when available; otherwise use the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_mnist_data(data_dir: Path, split_seed: int) -> tuple[int, int, Path]:
    """Download MNIST and save one deterministic train/validation split."""
    train_dataset = datasets.MNIST(data_dir, train=True, download=True)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True)

    generator = torch.Generator().manual_seed(split_seed)
    shuffled_indices = torch.randperm(len(train_dataset), generator=generator)
    split_path = data_dir / "split_indices.pt"
    torch.save(
        {
            "train": shuffled_indices[:55_000],
            "validation": shuffled_indices[55_000:],
        },
        split_path,
    )
    return len(train_dataset), len(test_dataset), split_path


def build_loaders(
    data_dir: Path,
    batch_size: int,
    split_seed: int,
    download: bool,
    split_indices_path: Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load complete MNIST train, validation, and test splits."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STANDARD_DEVIATION,)),
        ]
    )
    full_train = datasets.MNIST(
        data_dir, train=True, download=download, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=download, transform=transform
    )

    if split_indices_path is None:
        generator = torch.Generator().manual_seed(split_seed)
        shuffled_indices = torch.randperm(len(full_train), generator=generator)
        train_indices = shuffled_indices[:55_000]
        validation_indices = shuffled_indices[55_000:]
    else:
        split_indices = torch.load(
            split_indices_path, map_location="cpu", weights_only=True
        )
        train_indices = split_indices["train"]
        validation_indices = split_indices["validation"]

    train_dataset = Subset(full_train, train_indices.tolist())
    validation_dataset = Subset(full_train, validation_indices.tolist())

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, validation_loader, test_loader


def build_test_loader(data_dir: Path, batch_size: int, download: bool) -> DataLoader:
    """Load the complete MNIST test split."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STANDARD_DEVIATION,)),
        ]
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=download, transform=transform
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one complete pass over the training split."""
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    """Return loss, accuracy, predictions, and labels for one complete split."""
    model.eval()
    total_loss = 0.0
    correct = 0
    all_predictions = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        predictions = logits.argmax(dim=1)

        total_loss += criterion(logits, labels).item() * labels.size(0)
        correct += (predictions == labels).sum().item()
        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())

    sample_count = len(loader.dataset)
    return (
        total_loss / sample_count,
        correct / sample_count,
        torch.cat(all_predictions),
        torch.cat(all_labels),
    )


@torch.inference_mode()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the full test split and keep 16 examples for visualization."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    example_image_batches = []
    example_label_batches = []
    example_prediction_batches = []
    examples_collected = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        predictions = logits.argmax(dim=1)

        total_loss += criterion(logits, labels).item() * labels.size(0)
        correct += (predictions == labels).sum().item()

        if examples_collected < 16:
            examples_needed = 16 - examples_collected
            example_image_batches.append(images[:examples_needed].cpu())
            example_label_batches.append(labels[:examples_needed].cpu())
            example_prediction_batches.append(predictions[:examples_needed].cpu())
            examples_collected += min(examples_needed, labels.size(0))

    sample_count = len(loader.dataset)
    return (
        total_loss / sample_count,
        correct / sample_count,
        torch.cat(example_image_batches),
        torch.cat(example_label_batches),
        torch.cat(example_prediction_batches),
    )


def save_result_plots(
    history: dict[str, list[float]],
    test_labels: torch.Tensor,
    test_predictions: torch.Tensor,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save training curves and a test confusion matrix as PNG files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_numbers = range(1, len(history["train_loss"]) + 1)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epoch_numbers, history["train_loss"], marker="o", label="train")
    axes[0].plot(
        epoch_numbers, history["validation_loss"], marker="o", label="validation"
    )
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        epoch_numbers,
        history["validation_accuracy"],
        marker="o",
        color="tab:green",
    )
    axes[1].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    curves_path = output_dir / "training_curves.png"
    figure.savefig(curves_path, dpi=160)
    plt.close(figure)

    confusion_matrix = torch.zeros((10, 10), dtype=torch.int64)
    for true_label, prediction in zip(test_labels, test_predictions):
        confusion_matrix[int(true_label), int(prediction)] += 1

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(confusion_matrix.numpy(), cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title="Test confusion matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=range(10),
        yticks=range(10),
    )
    figure.tight_layout()
    confusion_path = output_dir / "test_confusion_matrix.png"
    figure.savefig(confusion_path, dpi=160)
    plt.close(figure)

    return curves_path, confusion_path


def save_prediction_plot(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    output_path: Path,
) -> None:
    """Save 16 test images with their true and predicted labels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = (images * MNIST_STANDARD_DEVIATION + MNIST_MEAN).clamp(0, 1)

    figure, axes = plt.subplots(4, 4, figsize=(8, 8))
    for index, axis in enumerate(axes.flat):
        axis.imshow(images[index, 0].numpy(), cmap="gray")
        correct = int(labels[index]) == int(predictions[index])
        axis.set_title(
            f"true={int(labels[index])}, pred={int(predictions[index])}",
            color="green" if correct else "red",
            fontsize=9,
        )
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
