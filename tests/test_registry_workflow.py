import sys
from pathlib import Path

import pytest
import torch

import inference
import prepare_data
import promote_model
import train
from model import MNISTCNN


ROOT = Path(__file__).resolve().parents[1]


def test_data_preparation_prints_an_exact_artifact_version(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    class LoggedDataset:
        qualified_name = "demo-team/demo-project/mnist-dataset:v2"

        def __init__(self) -> None:
            self.waited = False

        def wait(self) -> None:
            self.waited = True

    class DatasetArtifact:
        def __init__(self, **kwargs) -> None:
            self.files = []

        def add_dir(self, path, name):
            self.files.append((Path(path), name))

        def add_file(self, path, name):
            self.files.append((Path(path), name))

    class FakeRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def log_artifact(self, artifact, aliases):
            return logged_dataset

    logged_dataset = LoggedDataset()
    monkeypatch.setattr(prepare_data.wandb, "init", lambda **kwargs: FakeRun())
    monkeypatch.setattr(prepare_data.wandb, "Artifact", DatasetArtifact)
    monkeypatch.setattr(
        prepare_data,
        "prepare_mnist_data",
        lambda data_dir, seed: (60_000, 10_000, data_dir / "split_indices.pt"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_data.py", "--data-dir", str(tmp_path / "data")],
    )

    prepare_data.main()

    assert logged_dataset.waited
    assert (
        "Dataset Artifact: demo-team/demo-project/mnist-dataset:v2"
        in capsys.readouterr().out
    )


def test_training_prints_the_server_assigned_model_version(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    class LoggedModel:
        qualified_name = "demo-team/demo-project/mnist-cnn:v4"

        def __init__(self) -> None:
            self.waited = False

        def wait(self) -> None:
            self.waited = True

    class OutputArtifact:
        def __init__(self, **kwargs) -> None:
            self.files = []

        def add_file(self, path, name):
            self.files.append((Path(path), name))

    class FakeRun:
        entity = "demo-team"
        project = "demo-project"
        url = None

        def __init__(self, logged_model) -> None:
            self.logged_model = logged_model
            self.summary = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def log(self, values):
            self.logged_values = values

        def log_artifact(self, artifact, aliases):
            return self.logged_model

    logged_model = LoggedModel()
    run = FakeRun(logged_model)
    monkeypatch.setattr(train.wandb, "init", lambda **kwargs: run)
    monkeypatch.setattr(train.wandb, "Image", lambda path: path)
    monkeypatch.setattr(train.wandb, "Artifact", OutputArtifact)
    monkeypatch.setattr(train, "select_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        train, "build_loaders", lambda **kwargs: (object(), object(), object())
    )
    monkeypatch.setattr(train, "train_one_epoch", lambda *args: 0.5)
    monkeypatch.setattr(
        train,
        "evaluate",
        lambda *args: (
            0.2,
            0.9,
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
        ),
    )

    def fake_result_plots(history, labels, predictions, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        curves = output_dir / "training_curves.png"
        confusion = output_dir / "test_confusion_matrix.png"
        curves.write_bytes(b"curves")
        confusion.write_bytes(b"confusion")
        return curves, confusion

    monkeypatch.setattr(train, "save_result_plots", fake_result_plots)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--epochs",
            "1",
            "--output-dir",
            str(tmp_path / "outputs"),
        ],
    )

    train.main()

    assert logged_model.waited
    assert (
        run.summary["model/artifact_reference"] == "demo-team/demo-project/mnist-cnn:v4"
    )
    assert (
        "Model Artifact: demo-team/demo-project/mnist-cnn:v4" in capsys.readouterr().out
    )


def test_promotion_links_the_exact_source_object(monkeypatch, capsys) -> None:
    source = type(
        "SourceArtifact",
        (),
        {"qualified_name": "demo-team/demo-project/mnist-cnn:v7"},
    )()
    linked = type(
        "RegistryArtifact",
        (),
        {
            "qualified_name": "demo-org/wandb-registry-Model Registry/mnist-cnn:v2",
            "version": "v2",
            "source_version": "v7",
            "url": "https://wandb.ai/registry/example",
        },
    )()

    class FakeRun:
        entity = "demo-team"
        project = "demo-project"

        def __init__(self) -> None:
            self.summary = {}
            self.used = None
            self.linked = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def use_artifact(self, reference, type):
            self.used = (reference, type)
            return source

        def link_artifact(self, artifact, target_path, aliases):
            self.linked = (artifact, target_path, aliases)
            return linked

        def log_artifact(self, *args, **kwargs):
            raise AssertionError("Promotion must link, not re-log, the model")

    run = FakeRun()
    init_kwargs = {}

    def fake_init(**kwargs):
        init_kwargs.update(kwargs)
        return run

    monkeypatch.setattr(promote_model.wandb, "init", fake_init)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_model.py",
            "--model-artifact",
            "demo-team/demo-project/mnist-cnn:v7",
            "--registry",
            "Model Registry",
            "--collection",
            "mnist-cnn",
            "--alias",
            "candidate",
        ],
    )

    promote_model.main()

    assert init_kwargs["entity"] == "demo-team"
    assert init_kwargs["project"] == "demo-project"
    assert run.used == ("demo-team/demo-project/mnist-cnn:v7", "model")
    assert run.linked == (
        source,
        "wandb-registry-Model Registry/mnist-cnn",
        ["candidate"],
    )
    assert run.summary["registry/source_version"] == "v7"
    assert run.summary["registry/version"] == "v2"
    output = capsys.readouterr().out
    assert (
        "Registry alias: demo-org/wandb-registry-Model Registry/mnist-cnn:candidate"
        in output
    )
    assert (
        "Next command: python inference.py --entity demo-team --project demo-project "
        "--model-artifact 'demo-org/wandb-registry-Model Registry/mnist-cnn:candidate'"
        in output
    )


def test_promotion_rejects_a_moving_source_alias(monkeypatch) -> None:
    def unexpected_init(**kwargs):
        raise AssertionError("Invalid input must fail before wandb.init")

    monkeypatch.setattr(promote_model.wandb, "init", unexpected_init)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_model.py",
            "--model-artifact",
            "demo-team/demo-project/mnist-cnn:latest",
            "--registry",
            "Models",
        ],
    )

    with pytest.raises(SystemExit) as error:
        promote_model.main()

    assert error.value.code == 2


def test_missing_registry_error_explains_how_to_create_it(monkeypatch) -> None:
    source = type(
        "SourceArtifact",
        (),
        {"qualified_name": "demo-team/demo-project/mnist-cnn:v7"},
    )()
    original_error = RuntimeError("project wandb-registry-Missing was not found")

    class FakeRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def use_artifact(self, reference, type):
            return source

        def link_artifact(self, artifact, target_path, aliases):
            raise original_error

    monkeypatch.setattr(promote_model.wandb, "init", lambda **kwargs: FakeRun())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_model.py",
            "--model-artifact",
            "demo-team/demo-project/mnist-cnn:v7",
            "--registry",
            "Missing",
        ],
    )

    with pytest.raises(RuntimeError) as error:
        promote_model.main()

    message = str(error.value)
    assert "Registry 'Missing'" in message
    assert "https://wandb.ai/registry/" in message
    assert "create it" in message
    assert "rerun this command" in message
    assert str(original_error) in message
    assert error.value.__cause__ is original_error


@pytest.mark.parametrize(
    ("cli_reference", "expected_reference", "is_registry"),
    [
        (None, "mnist-cnn:latest", False),
        (
            "demo-org/wandb-registry-Models/mnist-cnn:candidate",
            "demo-org/wandb-registry-Models/mnist-cnn:candidate",
            True,
        ),
    ],
)
def test_inference_accepts_project_and_registry_models(
    monkeypatch,
    tmp_path: Path,
    cli_reference: str | None,
    expected_reference: str,
    is_registry: bool,
) -> None:
    requested_references = []

    class RegistryArtifact:
        qualified_name = (
            "demo-org/wandb-registry-Models/mnist-cnn:v3"
            if is_registry
            else "demo-team/demo-project/mnist-cnn:v3"
        )
        source_qualified_name = "demo-team/demo-project/mnist-cnn:v7"
        source_version = "v7"
        is_link = is_registry

        def download(self, root):
            model_dir = Path(root)
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(MNISTCNN().state_dict(), model_dir / "mnist_cnn_state_dict.pt")
            return str(model_dir)

    class OutputArtifact:
        def __init__(self, **kwargs) -> None:
            self.files = []

        def add_file(self, path, name):
            self.files.append((Path(path), name))

    class FakeRun:
        url = None

        def __init__(self) -> None:
            self.summary = {}
            self.logged_artifact = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def use_artifact(self, reference, type):
            requested_references.append((reference, type))
            return RegistryArtifact()

        def log(self, values):
            self.logged_values = values

        def log_artifact(self, artifact, aliases):
            self.logged_artifact = (artifact, aliases)

    run = FakeRun()
    monkeypatch.setattr(inference.wandb, "init", lambda **kwargs: run)
    monkeypatch.setattr(inference.wandb, "Image", lambda path: path)
    monkeypatch.setattr(inference.wandb, "Artifact", OutputArtifact)
    monkeypatch.setattr(inference, "select_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(inference, "build_test_loader", lambda **kwargs: object())
    monkeypatch.setattr(
        inference,
        "run_inference",
        lambda model, loader, device: (
            0.1,
            0.95,
            torch.zeros(16, 1, 28, 28),
            torch.arange(16) % 10,
            torch.arange(16) % 10,
        ),
    )

    def fake_prediction_plot(images, labels, predictions, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"plot")

    monkeypatch.setattr(inference, "save_prediction_plot", fake_prediction_plot)
    arguments = [
        "inference.py",
        "--data-dir",
        str(tmp_path / "data"),
        "--artifact-dir",
        str(tmp_path / "artifacts"),
        "--output-dir",
        str(tmp_path / "outputs"),
    ]
    if cli_reference is not None:
        arguments.extend(["--model-artifact", cli_reference])
    monkeypatch.setattr(
        sys,
        "argv",
        arguments,
    )

    inference.main()

    assert requested_references == [(expected_reference, "model")]
    assert run.summary["model/requested_reference"] == expected_reference
    assert run.summary["model/resolved_reference"] == RegistryArtifact.qualified_name
    if is_registry:
        assert run.summary["model/source_reference"].endswith(":v7")
    else:
        assert "model/source_reference" not in run.summary
    assert (tmp_path / "outputs" / "inference_metrics.json").is_file()
    assert run.logged_artifact[1] == ["latest"]


def test_wandb_boundary_and_registry_labels() -> None:
    assert "import wandb" not in (ROOT / "model.py").read_text(encoding="utf-8")
    assert "[W&B REGISTRY]" in (ROOT / "promote_model.py").read_text(encoding="utf-8")
    assert "[W&B MODEL INPUT]" in (ROOT / "inference.py").read_text(encoding="utf-8")
