from utils import notebook_token


def _clear_platform(monkeypatch):
    for name in (
        "HF_TOKEN",
        "COLAB_BACKEND_URL",
        "COLAB_JUPYTER_IP",
        "KAGGLE_KERNEL_RUN_TYPE",
        "KAGGLE_URL_BASE",
    ):
        monkeypatch.delenv(name, raising = False)


def test_resolves_kaggle_secret_into_backend_environment(monkeypatch):
    _clear_platform(monkeypatch)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    monkeypatch.setattr(notebook_token, "_read_kaggle_secret", lambda name: f" {name}-value ")

    assert notebook_token.resolve_notebook_hf_token() == ("HF_TOKEN-value", "kaggle")
    assert notebook_token.os.environ["HF_TOKEN"] == "HF_TOKEN-value"


def test_existing_notebook_token_is_returned_without_secret_lookup(monkeypatch):
    _clear_platform(monkeypatch)
    monkeypatch.setenv("COLAB_BACKEND_URL", "https://colab.invalid")
    monkeypatch.setenv("HF_TOKEN", "existing")
    monkeypatch.setattr(
        notebook_token,
        "_read_colab_secret",
        lambda _name: (_ for _ in ()).throw(AssertionError("unexpected lookup")),
    )

    assert notebook_token.resolve_notebook_hf_token() == ("existing", "colab")


def test_desktop_environment_token_is_not_returned_to_browser(monkeypatch):
    _clear_platform(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "server-secret")

    assert notebook_token.resolve_notebook_hf_token() == (None, None)
