#!/usr/bin/env python3
"""Link one trained MNIST Model Artifact to W&B Registry."""

import argparse
import shlex

import wandb  # [W&B CORE] Import the W&B Python package.


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote an MNIST Model Artifact to W&B Registry."
    )
    parser.add_argument(
        "--project",
        # [W&B OPTIONAL] Defaults to the source Artifact's Project.
        default=None,
    )
    parser.add_argument(
        "--entity",
        # [W&B OPTIONAL] Defaults to the source Artifact's Team.
        default=None,
    )
    parser.add_argument(
        "--model-artifact",
        required=True,
        help=(
            "Exact Project Model Artifact printed by train.py: "
            "ENTITY/PROJECT/NAME:vN. Aliases are rejected."
        ),
    )
    parser.add_argument(
        "--registry",
        required=True,
        help="Existing Registry name shown in the W&B Registry UI.",
    )
    parser.add_argument(
        "--collection",
        default="mnist-cnn",
        help="Registry collection that will contain the linked model.",
    )
    parser.add_argument(
        "--alias",
        default="candidate",
        help="Registry alias assigned to the linked version.",
    )
    args = parser.parse_args()

    path_parts = args.model_artifact.split("/")
    name_and_version = path_parts[-1].rsplit(":", 1)
    if (
        len(path_parts) != 3
        or any(not part for part in path_parts)
        or len(name_and_version) != 2
        or not name_and_version[0]
        or not name_and_version[1].startswith("v")
        or not name_and_version[1][1:].isdigit()
    ):
        parser.error(
            "--model-artifact must be the exact value printed by train.py: "
            "ENTITY/PROJECT/NAME:vN"
        )

    source_entity, source_project, _ = path_parts
    target_path = f"wandb-registry-{args.registry}/{args.collection}"

    # [W&B CORE] Registry linking requires an online W&B connection. A local
    # WANDB_MODE=offline/disabled safety setting is intentionally not overridden.
    with wandb.init(
        project=args.project or source_project,
        entity=args.entity or source_entity,
        job_type="promote-model",  # [W&B RECOMMENDED] Identifies this Run's role.
        config={  # [W&B RECOMMENDED] Records the promotion decision.
            "source_artifact": args.model_artifact,
            "registry": args.registry,
            "collection": args.collection,
            "registry_alias": args.alias,
        },
    ) as run:
        # [W&B ARTIFACT INPUT] Resolve the source once and keep that exact object.
        source_artifact = run.use_artifact(args.model_artifact, type="model")

        exact_source_ref = source_artifact.qualified_name

        # [W&B REGISTRY] Link the existing object; do not upload another checkpoint.
        try:
            registry_artifact = run.link_artifact(
                source_artifact,
                target_path=target_path,
                aliases=[args.alias],
            )
        except Exception as error:
            raise RuntimeError(
                f"W&B could not link the model to Registry '{args.registry}'. "
                "If that Registry does not exist, open https://wandb.ai/registry/, "
                "create it with support for the 'model' Artifact type, then rerun "
                "this command. "
                f"Original W&B error: {error}"
            ) from error
        exact_registry_ref = registry_artifact.qualified_name
        registry_collection_ref = exact_registry_ref.rsplit(":", 1)[0]
        registry_alias_ref = f"{registry_collection_ref}:{args.alias}"
        next_command = shlex.join(
            [
                "python",
                "inference.py",
                "--entity",
                str(run.entity),
                "--project",
                str(run.project),
                "--model-artifact",
                registry_alias_ref,
            ]
        )

        # [W&B SUMMARY] Preserve both identities because Project and Registry
        # version numbers are separate.
        run.summary["registry/source_artifact"] = exact_source_ref
        run.summary["registry/source_version"] = registry_artifact.source_version
        run.summary["registry/artifact"] = exact_registry_ref
        run.summary["registry/version"] = registry_artifact.version
        run.summary["registry/alias"] = args.alias

        print(f"Source Model Artifact: {exact_source_ref}")
        print(f"Registry Model: {exact_registry_ref}")
        print(f"Registry alias: {registry_alias_ref}")
        print(f"Registry URL: {registry_artifact.url}")
        print(f"Next command: {next_command}")


if __name__ == "__main__":
    main()
