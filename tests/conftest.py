import wandb  # [W&B TEST SAFETY] Patch every service entry point used by tests.
import pytest


@pytest.fixture(autouse=True)
def prevent_cloud_access(monkeypatch, tmp_path) -> None:
    """Keep every automated test local even if the shell is logged in to W&B."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path / "wandb-cache"))
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "wandb-data"))
    monkeypatch.setenv("WANDB_CONFIG_DIR", str(tmp_path / "wandb-config"))

    def blocked(*args, **kwargs):
        raise AssertionError("Automated tests must not access W&B Cloud")

    monkeypatch.setattr(wandb, "login", blocked)
    monkeypatch.setattr(wandb, "Api", blocked)
    monkeypatch.setattr(wandb, "init", blocked)
