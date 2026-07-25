import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "unsloth" / "notebook_token.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("notebook_token_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_detects_colab_hf_token(monkeypatch):
    module = _load_module()
    monkeypatch.delenv("HF_TOKEN", raising = False)
    monkeypatch.setenv("COLAB_BACKEND_URL", "https://colab.invalid")
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising = False)
    monkeypatch.delenv("KAGGLE_URL_BASE", raising = False)
    monkeypatch.setattr(module, "_read_colab_secret", lambda name: f"  {name}-value  ")

    assert module.detect_notebook_hf_token() == "HF_TOKEN-value"
    assert module.os.environ["HF_TOKEN"] == "HF_TOKEN-value"


def test_detects_kaggle_hf_token_after_missing_colab_secret(monkeypatch):
    module = _load_module()
    monkeypatch.delenv("HF_TOKEN", raising = False)
    monkeypatch.setenv("COLAB_JUPYTER_IP", "127.0.0.1")
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    monkeypatch.setattr(module, "_read_colab_secret", lambda _name: None)
    monkeypatch.setattr(module, "_read_kaggle_secret", lambda name: f"{name}-kaggle")

    assert module.detect_notebook_hf_token() == "HF_TOKEN-kaggle"


def test_explicit_hf_token_is_not_overridden(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("HF_TOKEN", "explicit")
    monkeypatch.setenv("COLAB_BACKEND_URL", "https://colab.invalid")
    monkeypatch.setattr(
        module,
        "_read_colab_secret",
        lambda _name: (_ for _ in ()).throw(AssertionError("secret store was accessed")),
    )

    assert module.detect_notebook_hf_token() == "explicit"


def test_secret_access_failure_is_non_fatal(monkeypatch):
    module = _load_module()
    monkeypatch.delenv("HF_TOKEN", raising = False)
    monkeypatch.setenv("KAGGLE_URL_BASE", "https://www.kaggle.com")
    monkeypatch.setattr(
        module,
        "_read_kaggle_secret",
        lambda _name: (_ for _ in ()).throw(RuntimeError("not granted")),
    )

    assert module.detect_notebook_hf_token() is None
    assert "HF_TOKEN" not in module.os.environ
