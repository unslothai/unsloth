import sys
import types
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent / "studio" / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

if "structlog" not in sys.modules:
    class _L:
        def __getattr__(self, n):
            return lambda *a, **k: None
    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger=_L, get_logger=lambda *a, **k: _L(),
    )
if "datasets" not in sys.modules:
    class _NF(Exception):
        pass
    m = types.ModuleType("datasets")
    m.get_dataset_config_names = lambda *a, **k: []
    m.get_dataset_split_names = lambda *a, **k: []
    m.IterableDataset = type("IterableDataset", (), {})
    m.Dataset = type("Dataset", (), {})
    m.load_dataset = lambda *a, **k: None
    sys.modules["datasets"] = m
    e = types.ModuleType("datasets.exceptions")
    e.DatasetNotFoundError = _NF
    sys.modules["datasets.exceptions"] = e

from fastapi import HTTPException  # noqa: E402

from models.datasets import DatasetSplitsRequest  # noqa: E402
import routes.datasets as rd  # noqa: E402


class _CapturingLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg, *a, **k):
        self.messages.append(str(msg))

    def warning(self, msg, *a, **k):
        self.messages.append(str(msg))

    def error(self, msg, *a, **k):
        self.messages.append(str(msg))


SECRET_TOKEN = "hf_SECRETTOKEN123456abcdef"


def test_token_not_logged_on_upstream_exception(monkeypatch):
    import datasets as ds
    cap = _CapturingLogger()
    monkeypatch.setattr(rd, "logger", cap)

    def _raise(*a, **k):
        raise RuntimeError("upstream error details")
    monkeypatch.setattr(ds, "get_dataset_config_names", _raise)

    req = DatasetSplitsRequest(
        dataset_name="owner/x", hf_token=SECRET_TOKEN,
    )
    with pytest.raises(HTTPException):
        rd.get_dataset_splits(req, current_subject="t")

    combined = "\n".join(cap.messages)
    assert SECRET_TOKEN not in combined
    assert "hf_SECRET" not in combined


def test_token_not_logged_on_hfhub_error(monkeypatch):
    import datasets as ds
    from huggingface_hub.utils import HfHubHTTPError
    cap = _CapturingLogger()
    monkeypatch.setattr(rd, "logger", cap)

    class _R:
        status_code = 403

    def _raise(*a, **k):
        e = HfHubHTTPError("forbidden")
        e.response = _R()
        raise e
    monkeypatch.setattr(ds, "get_dataset_config_names", _raise)

    req = DatasetSplitsRequest(
        dataset_name="owner/x", hf_token=SECRET_TOKEN,
    )
    with pytest.raises(HTTPException):
        rd.get_dataset_splits(req, current_subject="t")

    combined = "\n".join(cap.messages)
    assert SECRET_TOKEN not in combined
