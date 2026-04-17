import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

_BACKEND = Path(__file__).resolve().parent / "studio" / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from fastapi import HTTPException  # noqa: E402

from models.datasets import DatasetSplitsRequest  # noqa: E402
import routes.datasets as rd  # noqa: E402


class _CapturingLogger:
    def __init__(self):
        self.messages = []
        self.kwargs = []

    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def error(self, msg, *a, **k):
        self.messages.append(str(msg))
        self.kwargs.append(k)


def _deep_chain_exception():
    def a():
        raise ValueError("deep_ctx_a")

    def b():
        try:
            a()
        except Exception as inner:
            raise RuntimeError("deep_ctx_b") from inner

    try:
        b()
    except Exception as top:
        return top


def test_logger_never_receives_exc_info_true(
    monkeypatch, fake_datasets, fake_structlog
):
    import datasets as ds

    cap = _CapturingLogger()
    monkeypatch.setattr(rd, "logger", cap)

    chained = _deep_chain_exception()

    def _raise(*a, **k):
        raise chained

    monkeypatch.setattr(ds, "get_dataset_config_names", _raise)

    req = DatasetSplitsRequest(dataset_name = "owner/x")
    with pytest.raises(HTTPException):
        rd.get_dataset_splits(req, current_subject = "t")

    # No logger.error call forwards exc_info=True
    for kwargs in cap.kwargs:
        assert kwargs.get("exc_info") is not True


def test_logger_message_has_no_traceback_keywords(
    monkeypatch, fake_datasets, fake_structlog
):
    import datasets as ds

    cap = _CapturingLogger()
    monkeypatch.setattr(rd, "logger", cap)

    def _raise(*a, **k):
        # An exception whose str() does NOT contain "Traceback",
        # ensuring the test catches the logger (not str(e)) leaking the trace.
        raise RuntimeError("plain bounded message")

    monkeypatch.setattr(ds, "get_dataset_config_names", _raise)

    req = DatasetSplitsRequest(dataset_name = "owner/x")
    with pytest.raises(HTTPException):
        rd.get_dataset_splits(req, current_subject = "t")

    combined = "\n".join(cap.messages)
    assert "Traceback" not in combined
    assert 'File "' not in combined
