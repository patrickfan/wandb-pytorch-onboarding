#!/usr/bin/env python3
"""Run a Bayesian W&B hyperparameter Sweep on MNIST."""

import argparse
import os
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import wandb  # [W&B CORE] Import the W&B Python package.

from model import MNISTCNN, build_loaders, evaluate, select_device, train_one_epoch


SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val/accuracy", "goal": "maximize"},
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-1,
        },
        "hidden": {"values": [64, 128, 256, 512]},
        "dropout": {"min": 0.0, "max": 0.5},
    },
}


def train_sweep_trial(args: argparse.Namespace) -> None:
    """Train one model using the hyperparameters assigned by W&B."""
    # [W&B CORE] Every Sweep trial is one context-managed W&B Run.
    with wandb.init(
        project=args.project,
        entity=args.entity,
        job_type="sweep-trial",
    ) as run:
        # [W&B SWEEP] Read the hyperparameters assigned to this trial.
        learning_rate = float(run.config["learning_rate"])
        hidden = int(run.config["hidden"])
        dropout = float(run.config["dropout"])

        # [W&B RECOMMENDED] Add the fixed training settings to this Run's config.
        run.config.update(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "dataset": "MNIST",
            }
        )

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        device = select_device()

        train_loader, validation_loader, _ = build_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            split_seed=args.seed,
            download=True,
        )
        model = MNISTCNN(hidden=hidden, dropout=dropout).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_validation_accuracy = 0.0
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            validation_loss, validation_accuracy, _, _ = evaluate(
                model, validation_loader, criterion, device
            )
            best_validation_accuracy = max(
                best_validation_accuracy, validation_accuracy
            )

            # [W&B METRICS] The metric name exactly matches SWEEP_CONFIG.
            run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": validation_loss,
                    "val/accuracy": validation_accuracy,
                }
            )

        # [W&B SUMMARY] Keep the best epoch visible without changing the Sweep
        # objective, which uses the final logged val/accuracy.
        run.summary["best_val_accuracy"] = best_validation_accuracy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Bayesian W&B Sweep for MNIST hyperparameters."
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT"),
        help="Override WANDB_PROJECT or the Project selected by wandb init.",
    )
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="Override WANDB_ENTITY or the entity selected by wandb init.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()

    # [W&B SWEEP] Create the server-side Bayesian Sweep.
    sweep_id = wandb.sweep(
        SWEEP_CONFIG,
        entity=args.entity,
        project=args.project,
    )

    # [W&B SWEEP] Run a bounded number of trials. Each callback opens one Run.
    wandb.agent(
        sweep_id,
        function=partial(train_sweep_trial, args),
        entity=args.entity,
        project=args.project,
        count=args.count,
    )


if __name__ == "__main__":
    main()
