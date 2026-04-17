import sys
from pathlib import Path


_BACKEND = Path(__file__).resolve().parent / "studio" / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from models.datasets import DatasetSplitsRequest  # noqa: E402
import routes.datasets as rd  # noqa: E402


def test_partial_failure_survives_unicode_upstream_error(
    monkeypatch, fake_datasets, fake_structlog
):
    import datasets as ds
    from huggingface_hub.utils import HfHubHTTPError

    monkeypatch.setattr(
        ds,
        "get_dataset_config_names",
        lambda name, token = None: ["ok", "unicode"],
    )

    class _R:
        status_code = 500

    unicode_msg = "\u4e2d\u6587\u30a8\u30e9\u30fc \u00e9\u00e7\u00f1\ud83d\ude80"

    def _splits(name, config_name = None, token = None):
        if config_name == "ok":
            return ["train"]
        e = HfHubHTTPError(unicode_msg)
        e.response = _R()
        raise e

    monkeypatch.setattr(ds, "get_dataset_split_names", _splits)
    req = DatasetSplitsRequest(dataset_name = "owner/unicode")
    resp = rd.get_dataset_splits(req, current_subject = "t")

    assert resp.partial_failure is not None
    # Generic message doesn't include the raw unicode upstream text
    assert unicode_msg not in resp.partial_failure
    # But it's a valid ascii-friendly string that encodes cleanly
    resp.partial_failure.encode("utf-8")


def test_partial_failure_survives_embedded_newlines(
    monkeypatch, fake_datasets, fake_structlog
):
    import datasets as ds
    from huggingface_hub.utils import HfHubHTTPError

    monkeypatch.setattr(
        ds,
        "get_dataset_config_names",
        lambda name, token = None: ["ok", "nl"],
    )

    class _R:
        status_code = 500

    def _splits(name, config_name = None, token = None):
        if config_name == "ok":
            return ["train"]
        e = HfHubHTTPError("line1\nline2\r\nline3\n--END")
        e.response = _R()
        raise e

    monkeypatch.setattr(ds, "get_dataset_split_names", _splits)
    req = DatasetSplitsRequest(dataset_name = "owner/nl")
    resp = rd.get_dataset_splits(req, current_subject = "t")

    assert resp.partial_failure is not None
    # Multi-line upstream content not leaked into banner
    assert "line2" not in resp.partial_failure
    assert "--END" not in resp.partial_failure
