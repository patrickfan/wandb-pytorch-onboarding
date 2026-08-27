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
        # [W&B OPTIONAL] The environment variable supplies a reusable default.
        default=os.environ.get("WANDB_PROJECT", "pytorch-mnist-onboarding"),
    )
    parser.add_argument(
        "--entity",
        # [W&B OPTIONAL] Leave this unset to use your default W&B entity.
        default=os.environ.get("WANDB_ENTITY"),
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
        entity=args.entity,  # [W&B OPTIONAL] Omit it to use your default entity.
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
        run.log_artifact(dataset_artifact, aliases=["latest"])

        print(
            f"Uploaded dataset Artifact: {args.artifact_name}:latest "
            f"({train_image_count} train/validation images, "
            f"{test_image_count} test images)"
        )


if __name__ == "__main__":
    main()
