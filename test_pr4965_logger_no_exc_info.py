import sys
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent / "studio" / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from fastapi import HTTPException  # noqa: E402

from models.datasets import DatasetSplitsRequest  # noqa: E402
import routes.datasets as rd  # noqa: E402


class _CapturingLogger:
    def __init__(self):
        self.calls = []

    def info(self, *a, **k):
        self.calls.append(("info", a, k))

    def warning(self, *a, **k):
        self.calls.append(("warning", a, k))

    def error(self, *a, **k):
        self.calls.append(("error", a, k))


def _make_http_error(status):
    from huggingface_hub.utils import HfHubHTTPError

    class _R:
        status_code = status

    e = HfHubHTTPError("upstream boom")
    e.response = _R()
    return e


def test_hfhub_error_handler_has_no_exc_info_kwarg(monkeypatch, fake_datasets, fake_structlog):
    import datasets as ds

    cap = _CapturingLogger()
    monkeypatch.setattr(rd, "logger", cap)

    def _raise(*a, **k):
        raise _make_http_error(500)

    monkeypatch.setattr(ds, "get_dataset_config_names", _raise)

    req = DatasetSplitsRequest(dataset_name = "owner/x")
    with pytest.raises(HTTPException):
        rd.get_dataset_splits(req, current_subject = "t")

    error_calls = [c for c in cap.calls if c[0] == "error"]
    assert error_calls, "expected at least one logger.error call"
    for _, args, kwargs in error_calls:
        assert "exc_info" not in kwargs
        for a in args:
            # positional exc_info or traceback objects not allowed
            assert not isinstance(a, bool)


def test_generic_exception_handler_has_no_exc_info_kwarg(monkeypatch, fake_datasets, fake_structlog):
    import datasets as ds

    cap = _CapturingLogger()
    monkeypatch.setattr(rd, "logger", cap)

    def _raise(*a, **k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(ds, "get_dataset_config_names", _raise)

    req = DatasetSplitsRequest(dataset_name = "owner/x")
    with pytest.raises(HTTPException):
        rd.get_dataset_splits(req, current_subject = "t")

    error_calls = [c for c in cap.calls if c[0] == "error"]
    assert error_calls
    for _, args, kwargs in error_calls:
        assert "exc_info" not in kwargs
