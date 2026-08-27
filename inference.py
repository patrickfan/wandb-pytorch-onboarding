#!/usr/bin/env python3
"""Download a W&B model Artifact, run MNIST inference, and upload results."""

import argparse
import json
import os
from pathlib import Path

import torch
import wandb  # [W&B CORE] Import the W&B Python package.

from model import (
    MNISTCNN,
    build_test_loader,
    run_inference,
    save_prediction_plot,
    select_device,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference from a W&B Model Artifact."
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
    # [W&B OPTIONAL] Set this only when you want a versioned Dataset Artifact.
    parser.add_argument(
        "--dataset-artifact",
        default=None,
        help="Optional Dataset Artifact, for example mnist-dataset:latest",
    )
    parser.add_argument("--model-artifact", default="mnist-cnn:latest")
    parser.add_argument("--results-artifact-name", default="mnist-inference-results")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inference"))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = select_device()

    # [W&B CORE] Create one W&B Run for the complete inference job.
    with wandb.init(
        # [W&B WORKFLOW] Keep training and inference together.
        project=args.project,
        entity=args.entity,  # [W&B OPTIONAL] Omit it to use your default entity.
        job_type="inference",  # [W&B RECOMMENDED] Describes the Run's role.
        config={  # [W&B RECOMMENDED] Records the inputs used for inference.
            "dataset": "MNIST",
            "dataset_source": args.dataset_artifact or "TorchVision download",
            "model_artifact": args.model_artifact,
            "batch_size": args.batch_size,
            "device": str(device),
        },
    ) as run:
        if args.dataset_artifact:
            # [W&B ARTIFACT INPUT] OPTIONAL: record and download versioned data.
            dataset_artifact = run.use_artifact(args.dataset_artifact, type="dataset")
            dataset_path = Path(
                dataset_artifact.download(root=args.artifact_dir / "mnist-dataset")
            )
            download_data = False
        else:
            # No Dataset Artifact: TorchVision downloads public MNIST directly.
            dataset_path = args.data_dir
            download_data = True

        # [W&B ARTIFACT INPUT] Record and download the versioned trained model.
        model_artifact = run.use_artifact(args.model_artifact, type="model")
        model_path = Path(
            model_artifact.download(root=args.artifact_dir / "mnist-model")
        )

        test_loader = build_test_loader(
            data_dir=dataset_path,
            batch_size=args.batch_size,
            download=download_data,
        )
        model = MNISTCNN().to(device)
        state_dict = torch.load(
            model_path / "mnist_cnn_state_dict.pt",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)

        test_loss, test_accuracy, images, labels, predictions = run_inference(
            model, test_loader, device
        )

        prediction_plot = args.output_dir / "inference_predictions.png"
        save_prediction_plot(images, labels, predictions, prediction_plot)

        metrics_path = args.output_dir / "inference_metrics.json"
        metrics_path.write_text(
            json.dumps(
                {"test_loss": test_loss, "test_accuracy": test_accuracy}, indent=2
            ),
            encoding="utf-8",
        )

        # [W&B METRICS + MEDIA] Send inference metrics and the prediction plot
        # to the W&B UI.
        run.log(
            {
                "inference/test_loss": test_loss,
                "inference/test_accuracy": test_accuracy,
                "inference/predictions": wandb.Image(str(prediction_plot)),
            }
        )

        # [W&B ARTIFACT OUTPUT] Version the inference plot and metrics together.
        results_artifact = wandb.Artifact(
            name=args.results_artifact_name,
            type="evaluation",
            description="MNIST inference predictions and evaluation metrics",
            metadata={"test_loss": test_loss, "test_accuracy": test_accuracy},
        )
        results_artifact.add_file(prediction_plot, name=prediction_plot.name)
        results_artifact.add_file(metrics_path, name=metrics_path.name)
        run.log_artifact(results_artifact, aliases=["latest"])

        print(f"test_loss={test_loss:.4f} test_accuracy={test_accuracy:.2%}")
        # [W&B OPTIONAL] Print the direct link to this Run when one is available.
        if run.url:
            print(f"W&B Run: {run.url}")


if __name__ == "__main__":
    main()
