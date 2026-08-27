#!/usr/bin/env python3
"""Optionally publish MNIST as a W&B Dataset Artifact."""

import argparse
import os
from pathlib import Path

import wandb  # [W&B CORE] Import the W&B Python package.

from model import prepare_mnist_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optional step: upload MNIST as a versioned W&B Dataset Artifact."
    )
    parser.add_argument(
        "--project",
        # [W&B OPTIONAL] Otherwise use the Project selected by `wandb init`.
        default=os.environ.get("WANDB_PROJECT"),
        help="Override WANDB_PROJECT or the Project selected by wandb init.",
    )
    parser.add_argument(
        "--entity",
        # [W&B OPTIONAL] Otherwise use the entity selected by `wandb init`.
        default=os.environ.get("WANDB_ENTITY"),
        help="Override WANDB_ENTITY or the entity selected by wandb init.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifact-name", default="mnist-dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_image_count, test_image_count, split_path = prepare_mnist_data(
        args.data_dir, args.seed
    )

    # [W&B CORE] One data-preparation job becomes one W&B Run. The context
    # manager finishes and uploads the Run automatically.
    with wandb.init(
        # [W&B WORKFLOW] Keep data, training, and inference together.
        project=args.project,
        entity=args.entity,  # [W&B OPTIONAL] Otherwise use local W&B settings.
        job_type="prepare-data",  # [W&B RECOMMENDED] Describes the Run's role.
        config={  # [W&B RECOMMENDED] Records how this dataset was prepared.
            "dataset": "MNIST",
            "split_seed": args.seed,
        },
    ) as run:
        # [W&B ARTIFACT OUTPUT] Package the raw data and split definition.
        dataset_artifact = wandb.Artifact(
            name=args.artifact_name,
            type="dataset",
            description=(
                "MNIST data with a deterministic train/validation split; "
                "see MNIST_LICENSE.md for source, attribution, and license"
            ),
            metadata={
                "train_images": 55_000,
                "validation_images": 5_000,
                "test_images": test_image_count,
                "split_seed": args.seed,
                "source": "https://yann.lecun.com/exdb/mnist/",
                "license": "CC BY-SA 3.0",
                "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
            },
        )
        dataset_artifact.add_dir(args.data_dir / "MNIST", name="MNIST")
        dataset_artifact.add_file(split_path, name="split_indices.pt")
        dataset_artifact.add_file(
            Path(__file__).with_name("MNIST_LICENSE.md"), name="MNIST_LICENSE.md"
        )

        # [W&B ARTIFACT OUTPUT] Upload and version the Dataset Artifact.
        logged_dataset = run.log_artifact(dataset_artifact, aliases=["latest"])

        # [W&B ARTIFACT VERSION] Print the immutable input version so training
        # and inference can consume the same exact dataset.
        logged_dataset.wait()

        dataset_collection_ref = logged_dataset.qualified_name.rsplit(":", 1)[0]
        dataset_artifact_ref = f"{dataset_collection_ref}:{logged_dataset.version}"
        print(f"Dataset Artifact: {dataset_artifact_ref}")
        print(
            f"Dataset images: {train_image_count} train/validation, "
            f"{test_image_count} test"
        )


if __name__ == "__main__":
    main()
