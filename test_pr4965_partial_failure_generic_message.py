import sys
from pathlib import Path


_BACKEND = Path(__file__).resolve().parent / "studio" / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from models.datasets import DatasetSplitsRequest  # noqa: E402
import routes.datasets as rd  # noqa: E402


def test_partial_failure_message_is_generic(monkeypatch, fake_datasets, fake_structlog, fake_datasets, fake_structlog):
    import datasets as ds
    from huggingface_hub.utils import HfHubHTTPError

    monkeypatch.setattr(
        ds,
        "get_dataset_config_names",
        lambda name, token = None: ["ok", "broken"],
    )

    class _R:
        status_code = 500

    def _splits(name, config_name = None, token = None):
        if config_name == "ok":
            return ["train"]
        e = HfHubHTTPError(
            "https://huggingface.co/datasets/xyz returned 500: internal detail"
        )
        e.response = _R()
        raise e

    monkeypatch.setattr(ds, "get_dataset_split_names", _splits)

    req = DatasetSplitsRequest(dataset_name = "owner/mixed")
    resp = rd.get_dataset_splits(req, current_subject = "t")

    assert resp.partial_failure is not None
    # New generic format
    assert "1 of 2 config(s) could not be loaded" in resp.partial_failure
    assert "Some subset options may be missing" in resp.partial_failure


def test_partial_failure_message_does_not_leak_upstream_detail(monkeypatch, fake_datasets, fake_structlog, fake_datasets, fake_structlog):
    import datasets as ds
    from huggingface_hub.utils import HfHubHTTPError

    monkeypatch.setattr(
        ds,
        "get_dataset_config_names",
        lambda name, token = None: ["ok", "leaky"],
    )

    leak_marker = "https://internal.corp/path?key=sekrit42"

    class _R:
        status_code = 500

    def _splits(name, config_name = None, token = None):
        if config_name == "ok":
            return ["train"]
        e = HfHubHTTPError(f"boom {leak_marker} stacktrace here")
        e.response = _R()
        raise e

    monkeypatch.setattr(ds, "get_dataset_split_names", _splits)

    req = DatasetSplitsRequest(dataset_name = "owner/mixed2")
    resp = rd.get_dataset_splits(req, current_subject = "t")

    assert resp.partial_failure is not None
    assert leak_marker not in resp.partial_failure
    assert "sekrit42" not in resp.partial_failure
