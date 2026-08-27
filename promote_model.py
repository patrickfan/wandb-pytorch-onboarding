#!/usr/bin/env python3
"""Link one trained MNIST Model Artifact to W&B Registry."""

import argparse
import os
import shlex

import wandb  # [W&B CORE] Import the W&B Python package.


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote an MNIST Model Artifact to W&B Registry."
    )
    parser.add_argument(
        "--project",
        # [W&B OPTIONAL] Otherwise use the Project selected by `wandb init`.
        default=None,
        help="Override WANDB_PROJECT or the Project selected by wandb init.",
    )
    parser.add_argument(
        "--entity",
        # [W&B OPTIONAL] Otherwise use the entity selected by `wandb init`.
        default=None,
        help="Override WANDB_ENTITY or the entity selected by wandb init.",
    )
    parser.add_argument(
        "--model-artifact",
        default=None,
        help=(
            "Optional exact Project Model Artifact: ENTITY/PROJECT/NAME:vN. "
            "When omitted, resolve mnist-cnn:latest once in the selected Project. "
            "Explicit aliases are rejected."
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

    if args.model_artifact:
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
                "An explicit --model-artifact must be exact: "
                "ENTITY/PROJECT/NAME:vN"
            )
        source_entity, source_project, _ = path_parts
        requested_model = args.model_artifact
        run_project = args.project or source_project
        run_entity = args.entity or source_entity
    else:
        requested_model = "mnist-cnn:latest"
        run_project = args.project or os.environ.get("WANDB_PROJECT")
        run_entity = args.entity or os.environ.get("WANDB_ENTITY")

    target_path = f"wandb-registry-{args.registry}/{args.collection}"

    # [W&B CORE] Registry linking requires an online W&B connection. A local
    # WANDB_MODE=offline/disabled safety setting is intentionally not overridden.
    with wandb.init(
        project=run_project,
        entity=run_entity,
        job_type="promote-model",  # [W&B RECOMMENDED] Identifies this Run's role.
        config={  # [W&B RECOMMENDED] Records the promotion decision.
            "source_artifact": requested_model,
            "registry": args.registry,
            "collection": args.collection,
            "registry_alias": args.alias,
        },
    ) as run:
        # [W&B ARTIFACT INPUT] Resolve once, then link this exact returned object.
        source_artifact = run.use_artifact(requested_model, type="model")

        # qualified_name may still end in :latest. Replace that alias with the
        # immutable server-assigned version returned by the Artifact object.
        source_collection_ref = source_artifact.qualified_name.rsplit(":", 1)[0]
        exact_source_ref = f"{source_collection_ref}:{source_artifact.version}"

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
        # The linked Artifact name may still show :candidate, so use .version
        # when printing and storing the immutable Registry reference.
        registry_collection_ref = registry_artifact.qualified_name.rsplit(":", 1)[0]
        exact_registry_ref = f"{registry_collection_ref}:{registry_artifact.version}"
        registry_alias_ref = f"{registry_collection_ref}:{args.alias}"
        production_ref = f"{registry_collection_ref}:production"
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
        production_command = shlex.join(
            [
                "python",
                "inference.py",
                "--entity",
                str(run.entity),
                "--project",
                str(run.project),
                "--model-artifact",
                production_ref,
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
        print(f"After assigning production: {production_command}")


if __name__ == "__main__":
    main()
