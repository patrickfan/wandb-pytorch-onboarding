#!/usr/bin/env python3
"""Train MNIST and publish model and result Artifacts to W&B."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import wandb  # [W&B CORE] Import the W&B Python package.

from model import (
    MNISTCNN,
    build_loaders,
    evaluate,
    save_result_plots,
    select_device,
    train_one_epoch,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MNIST and track the experiment with W&B."
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
    # [W&B OPTIONAL] Set this only when you want a versioned Dataset Artifact.
    parser.add_argument(
        "--dataset-artifact",
        default=None,
        help="Optional Dataset Artifact, for example mnist-dataset:latest",
    )
    parser.add_argument("--model-artifact-name", default="mnist-cnn")
    parser.add_argument("--results-artifact-name", default="mnist-training-results")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = select_device()

    # [W&B RECOMMENDED] Store the settings needed to understand and reproduce
    # the training Run.
    config = {
        "architecture": "MNISTCNN",
        "dataset": "MNIST",
        "dataset_source": args.dataset_artifact or "TorchVision download",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": "Adam",
        "seed": args.seed,
        "device": str(device),
    }

    # [W&B CORE] Create one W&B Run for the complete training job.
    with wandb.init(
        # [W&B WORKFLOW] Keep training and inference together.
        project=args.project,
        entity=args.entity,  # [W&B OPTIONAL] Otherwise use local W&B settings.
        job_type="train",  # [W&B RECOMMENDED] Describes the Run's role.
        config=config,  # [W&B RECOMMENDED] Makes Runs comparable.
    ) as run:
        if args.dataset_artifact:
            # [W&B ARTIFACT INPUT] OPTIONAL: record and download versioned data.
            dataset_artifact = run.use_artifact(args.dataset_artifact, type="dataset")
            dataset_path = Path(
                dataset_artifact.download(root=args.artifact_dir / "mnist-dataset")
            )
            split_indices_path = dataset_path / "split_indices.pt"
            download_data = False
        else:
            # No Dataset Artifact: TorchVision downloads public MNIST directly.
            dataset_path = args.data_dir
            split_indices_path = None
            download_data = True

        train_loader, validation_loader, test_loader = build_loaders(
            data_dir=dataset_path,
            batch_size=args.batch_size,
            split_seed=args.seed,
            download=download_data,
            split_indices_path=split_indices_path,
        )
        model = MNISTCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        history = {
            "train_loss": [],
            "validation_loss": [],
            "validation_accuracy": [],
        }

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            validation_loss, validation_accuracy, _, _ = evaluate(
                model, validation_loader, criterion, device
            )

            history["train_loss"].append(train_loss)
            history["validation_loss"].append(validation_loss)
            history["validation_accuracy"].append(validation_accuracy)

            # [W&B METRICS] Send custom metrics to the W&B charts.
            run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "validation/loss": validation_loss,
                    "validation/accuracy": validation_accuracy,
                }
            )
            print(
                f"epoch={epoch}/{args.epochs} "
                f"train_loss={train_loss:.4f} "
                f"validation_loss={validation_loss:.4f} "
                f"validation_accuracy={validation_accuracy:.2%}"
            )

        test_loss, test_accuracy, test_predictions, test_labels = evaluate(
            model, test_loader, criterion, device
        )

        curves_path, confusion_path = save_result_plots(
            history, test_labels, test_predictions, args.output_dir
        )
        metrics_path = args.output_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "history": history,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # [W&B METRICS + MEDIA] Log final metrics and both result plots to the UI.
        run.log(
            {
                "test/loss": test_loss,
                "test/accuracy": test_accuracy,
                "results/training_curves": wandb.Image(str(curves_path)),
                "results/test_confusion_matrix": wandb.Image(str(confusion_path)),
            }
        )

        checkpoint_path = args.output_dir / "mnist_cnn_state_dict.pt"
        torch.save(model.to("cpu").state_dict(), checkpoint_path)

        # [W&B ARTIFACT OUTPUT] Upload a versioned model for later inference.
        model_artifact = wandb.Artifact(
            name=args.model_artifact_name,
            type="model",
            description="MNISTCNN PyTorch state dictionary",
            metadata={**config, "test_accuracy": test_accuracy},
        )
        model_artifact.add_file(checkpoint_path, name=checkpoint_path.name)
        logged_model = run.log_artifact(model_artifact, aliases=["latest"])

        # [W&B ARTIFACT VERSION] Wait for the server-assigned immutable vN so
        # the exact source can be promoted to W&B Registry next.
        logged_model.wait()
        model_collection_ref = logged_model.qualified_name.rsplit(":", 1)[0]
        model_artifact_ref = f"{model_collection_ref}:{logged_model.version}"
        # [W&B SUMMARY] Make the exact promotion input visible on the Run.
        run.summary["model/artifact_reference"] = model_artifact_ref

        # [W&B ARTIFACT OUTPUT] Upload plots and metrics as a separate,
        # versioned result bundle.
        results_artifact = wandb.Artifact(
            name=args.results_artifact_name,
            type="evaluation",
            description="Training curves, test confusion matrix, and metrics",
            metadata={"test_loss": test_loss, "test_accuracy": test_accuracy},
        )
        results_artifact.add_file(curves_path, name=curves_path.name)
        results_artifact.add_file(confusion_path, name=confusion_path.name)
        results_artifact.add_file(metrics_path, name=metrics_path.name)
        run.log_artifact(results_artifact, aliases=["latest"])

        print(f"test_loss={test_loss:.4f} test_accuracy={test_accuracy:.2%}")
        print(f"Model Artifact: {model_artifact_ref}")
        # [W&B OPTIONAL] Print the direct link to this Run when one is available.
        if run.url:
            print(f"W&B Run: {run.url}")


if __name__ == "__main__":
    main()
