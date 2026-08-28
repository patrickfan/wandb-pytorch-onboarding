# W&B × PyTorch Onboarding: Train → Artifact → Inference

This beginner example trains a PyTorch model on public MNIST, logs metrics and
plots to W&B, saves the model as a Model Artifact, and downloads that Artifact
in a separate inference Run.

The default path does **not** require W&B Registry. Registry is an optional
organization-sharing and model-governance feature. Uploading MNIST as a Dataset
Artifact is also optional; TorchVision downloads it directly by default.

> [!IMPORTANT]
> Never put a W&B API key in code or Git. In CI or on a cluster, store
> `WANDB_API_KEY` in the platform's secret manager.

## 1. Install and log in

Python 3.10 or newer is required. Run all commands from this directory.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
wandb login
```

On Windows PowerShell, replace the activation command with:

```powershell
.\.venv\Scripts\Activate.ps1
```

This authentication is normally needed only once for one local credential
setup. It does not create a Run.

## 2. Required: choose where W&B uploads (entity + Project)

This step is required in this tutorial because it tells W&B where the Runs and
Artifacts should appear in the Web UI:

```bash
wandb init
```

Choose these values in the prompts:

| Prompt | What it means | What to choose |
|---|---|---|
| **Team / entity** | The account or Team that owns the uploaded data | Your personal account, or an organization Team if you plan to use Registry |
| **Project** | The container for this demo's Runs, charts, and Artifacts | Select or create `pytorch-mnist-onboarding` |

Team and Project are not the same thing. In the URL
`https://wandb.ai/TEAM-OR-USERNAME/pytorch-mnist-onboarding`, the first part is
the owner—called `entity` in the Python SDK—and the second part is the Project.

For organization-owned work, the W&B hierarchy is:

```text
Organization
├── Team
│   └── Project
│       ├── Workspace and Runs
│       └── source Model Artifact
└── Registry
    └── Collection
        └── linked Model Artifact version
```

Registry is a separate branch at the Organization level; it is not inside a
Team or Project. The linked Registry version points to the source Model
Artifact rather than uploading a second copy of the model.

The command saves your choices in this directory's `wandb/settings` file. The
Python scripts then call `wandb.init()` to create each Run and upload it to that
destination. Keep running the remaining commands from this directory.

The commands shown below use that saved destination. An explicit `--entity` or
`--project` flag—or an existing `WANDB_ENTITY` or `WANDB_PROJECT` environment
variable—overrides it.

## 3. Train and run inference

This is the recommended first run. It does not require Registry or a Dataset
Artifact.

```bash
python train.py
python inference.py --model-artifact "mnist-cnn:latest"
```

`train.py`:

- downloads MNIST through TorchVision;
- trains for three epochs;
- evaluates all 10,000 official test images;
- logs metrics and plots;
- uploads the model and training results as Artifacts.

`inference.py` downloads the Project's latest `mnist-cnn` Model Artifact,
rebuilds `MNISTCNN`, evaluates the test set, and uploads inference metrics and
the prediction plot.

Both scripts use the entity and Project selected in Step 2. At the end, W&B
prints **View run** and **View project** links; open them to see metrics, plots,
configuration, files, and Artifact lineage.

## Optional: share the model through W&B Registry

Skip this section unless you want organization-level discovery, lifecycle
aliases, access roles, or model governance. Registry is **not required for
inference**; the two-command workflow above already loads the trained model.

This optional path requires the Team choice in Step 2. A model uploaded under
a personal account cannot be linked into an organization Registry. If you
selected a personal account, repeat Step 2 with a Team and rerun `train.py`
before continuing.

### Step 1 — Create the Registry in the Web UI

Open [W&B Registry](https://wandb.ai/registry/) in the same organization that
contains the selected Team. Create or verify a Registry with:

- **Name:** `Models`
- **Accepted Artifact type:** `model`
- **Visibility:** **Restricted** or **Organization**, depending on the intended
  audience

The Registry must exist before the next command. You do not need to create the
`mnist-cnn` collection; the first successful link creates it when your role
permits that action.

### Step 2 — Link the trained model as `candidate`

```bash
python promote_model.py --registry Models
```

This resolves `mnist-cnn:latest` once in the `pytorch-mnist-onboarding` Project
under the Team selected in Step 2. It prints the exact source `vN` and links
that returned Artifact object without uploading a second checkpoint.

The short command is intended for this sequential demo; another training Run
can later move `latest`.

### Step 3 — Run inference from `candidate`

```bash
python inference.py --model-artifact "wandb-registry-Models/mnist-cnn:candidate"
```

### Step 4 — Promote and use `production`

In Registry, open **Models → mnist-cnn → candidate**, review the version, and
add or move the `production` alias to it. Then run:

```bash
python inference.py --model-artifact "wandb-registry-Models/mnist-cnn:production"
```

`candidate` and `production` are movable aliases. Use the exact Registry `:vN`
shown in the UI when inference must stay pinned to one version.

## Optional: version MNIST as a Dataset Artifact

Use this only when explicit dataset versioning and lineage are useful:

```bash
python prepare_data.py
python train.py --dataset-artifact "mnist-dataset:latest"
python inference.py --dataset-artifact "mnist-dataset:latest" --model-artifact "mnist-cnn:latest"
```

`prepare_data.py` uploads MNIST and a deterministic 55,000/5,000
train/validation split. For fixed lineage, replace `mnist-dataset:latest` with
the exact `Dataset Artifact: ...:vN` printed by the script.

The Dataset Artifact includes `MNIST_LICENSE.md`. Retain that file and review
your organization's data-sharing policy before publishing a dataset copy.

## Optional: tune hyperparameters with a Bayesian Sweep

This online workflow creates one Sweep and starts an agent with a limit of
eight trial Runs in the entity and Project selected earlier:

```bash
python bayesian_sweep.py
```

The script searches this space:

```python
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
```

Each trial gets its values from `run.config`, trains on the MNIST training
split, and reports `val/accuracy` on the validation split. The Sweep optimizes
the final logged validation accuracy; the test split is not used for tuning.

This script demonstrates search and visualization only. It does not upload a
Model Artifact, select a winning checkpoint, or pass the winning `hidden` and
`dropout` values into `train.py`. The normal `train.py` workflow still creates
the baseline model; connecting the winning architecture to Registry is a
separate advanced extension.

## What W&B code is required?

For basic online tracking with custom metrics, the W&B-specific minimum is:

1. install and import `wandb`;
2. make authentication available;
3. call `wandb.init()` once for the job;
4. call `run.log()` for custom metrics;
5. finish the Run, handled here by the `with wandb.init(...) as run` context.

Model Artifacts are required by this tutorial's separate inference workflow,
but not by basic W&B metric tracking. Dataset Artifacts and Registry are
optional. Sweeps and `run.watch()` are also optional.

`model.py` contains the ordinary PyTorch, data, evaluation, and plotting code
and does not import W&B. W&B integration stays in the other scripts.

**In this repository, every Run uses the context-manager form, and Run-scoped
operations go through `run`:**

```python
with wandb.init(...) as run:
    run.log({"loss": loss})
    run.config.update({"epochs": epochs})
    run.summary["best_accuracy"] = best_accuracy
    run.log_artifact(model_artifact)
    source_artifact = run.use_artifact("model:v0")
    run.link_artifact(source_artifact, target_path="wandb-registry-Models/model")
    run.watch(model)  # Optional; adds gradient/parameter tracking overhead.
```

Module-level constructors create W&B objects with `wandb.Image(...)` and
`wandb.Artifact(...)`. Operations on an Artifact use that object's methods,
such as `artifact.add_file(...)`, `artifact.download(...)`, and
`artifact.wait()`; these are object-scoped, not Run-scoped. Sweep control is
also separate: `wandb.sweep(...)` creates the server-side Sweep and
`wandb.agent(...)` runs trials. Each trial then opens its own context-managed
`run`.

The `[W&B ...]` comments in the scripts are only visual markers that make W&B
code easy to find. These are the actual W&B Python APIs used or demonstrated:

| Actual W&B code | What it does | Needed when |
|---|---|---|
| `with wandb.init(...) as run` | Creates one Run in the selected entity and Project; the context manager finishes it | Required for every tracked job |
| `run.log({...})` | Sends custom metrics to the Run history | Required for loss and accuracy charts |
| `wandb.Image(path)` | Converts a saved plot into W&B media | Used for plots in the Web UI |
| `wandb.Artifact(name=..., type=...)` | Creates an Artifact description | Used for model, results, or optional dataset versioning |
| `artifact.add_file(...)` / `artifact.add_dir(...)` | Adds local files to an Artifact | Used before logging an Artifact |
| `run.log_artifact(artifact, aliases=[...])` | Logs a versioned Artifact output | Required for this tutorial's model handoff |
| `run.use_artifact(reference, type=...)` | Resolves an Artifact input and records lineage | Required when training or inference consumes an Artifact |
| `artifact.download(root=...)` | Downloads the resolved Artifact files | Required before loading its dataset or model file |
| `artifact.wait()` and `artifact.version` | Waits for logging and reads the immutable server version such as `v3` | Used when an exact version is needed |
| `run.config[...]` / `run.config.update(...)` | Reads Sweep choices or records configuration | Required by the Sweep trial |
| `run.summary["name"] = value` | Stores final metrics or provenance fields | Recommended, not required |
| `run.watch(model)` | Tracks gradients or parameters | Optional; not enabled by these scripts |
| `run.link_artifact(...)` | Links an exact Model Artifact into W&B Registry | Optional Registry workflow only |
| `wandb.sweep(...)` | Creates a server-side hyperparameter Sweep | Optional Sweep workflow only |
| `wandb.agent(...)` | Requests configurations and runs bounded trials | Optional Sweep workflow only |

## Four terms used in this guide

- **Python API:** functions and methods used in code, such as `wandb.init()` and
  `run.log()`. “API” can also refer to the underlying web-service interface.
- **W&B Python SDK:** the installed `wandb` package that provides that Python
  API and handles communication with W&B.
- **W&B CLI:** commands installed with the package and typed in a terminal.
- **Web UI:** the browser interface for Runs, charts, Artifacts, and Registry.

## Privacy and scope

**Workspace**, **Runs**, and source **Artifacts** belong to a Project. Registry
belongs to an organization, so it is normal for Registry not to appear in a
Project sidebar.

Skipping Registry does not make a Project private. Check Project visibility
before uploading. Project access and Registry access are separate: promotion
requires access to the source Artifact and permission to link into the target
Registry. Restrict both when only selected collaborators should have access.

An Artifact logged under a personal account cannot be linked into an
organization Registry. For strictly personal use, keep a non-public personal
Project and use the direct inference workflow instead. `candidate` and
`production` do not grant or revoke access.

## Files

- `model.py`: W&B-free PyTorch, data, evaluation, and plotting code
- `train.py`: training, metrics, plots, Model Artifact, and results Artifact
- `inference.py`: Project/Registry model download and inference results
- `promote_model.py`: optional Registry link and aliases
- `prepare_data.py`: optional Dataset Artifact publisher
- `bayesian_sweep.py`: optional Bayesian hyperparameter Sweep
- `tests/`: fake-W&B tests that cannot write to W&B Cloud

## Test locally and publish safely

The test suite uses local fakes and does not create Cloud Runs or Registry
links:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Before publishing to GitHub:

- keep credentials out of code, Markdown, workflow YAML, and Git history;
- rotate any key that was ever committed;
- confirm `.gitignore` excludes `.venv/`, `data/`, `outputs/`, `artifacts/`,
  `wandb/`, `.env`, and checkpoints;
- retain `MNIST_LICENSE.md` if publishing the optional dataset workflow;
- add an appropriate code license.

## Official references

- [W&B PyTorch integration](https://docs.wandb.ai/models/integrations/pytorch)
- [W&B Artifacts](https://docs.wandb.ai/models/artifacts)
- [`Run.log_artifact()` and `Run.use_artifact()`](https://docs.wandb.ai/models/ref/python/experiments/run)
- [W&B Registry](https://docs.wandb.ai/models/registry)
- [Registry creation and visibility](https://docs.wandb.ai/models/registry/create_registry)
- [Registry roles](https://docs.wandb.ai/models/registry/configure_registry)
- [W&B Sweeps walkthrough](https://docs.wandb.ai/models/sweeps/walkthrough)
- [Sweep configuration keys](https://docs.wandb.ai/models/sweeps/sweep-config-keys)
- [TorchVision MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
