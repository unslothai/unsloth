import sys
from pathlib import Path


_BACKEND = Path(__file__).resolve().parent / "studio" / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from models.datasets import DatasetSplitsRequest  # noqa: E402
import routes.datasets as rd  # noqa: E402


def _run(monkeypatch, configs, fail_set):
    import datasets as ds
    from huggingface_hub.utils import HfHubHTTPError

    monkeypatch.setattr(
        ds,
        "get_dataset_config_names",
        lambda name, token = None: list(configs),
    )

    class _R:
        status_code = 500

    def _splits(name, config_name = None, token = None):
        if config_name in fail_set:
            e = HfHubHTTPError("err")
            e.response = _R()
            raise e
        return ["train"]

    monkeypatch.setattr(ds, "get_dataset_split_names", _splits)
    req = DatasetSplitsRequest(dataset_name = "owner/t")
    return rd.get_dataset_splits(req, current_subject = "t")


def test_partial_failure_count_three_of_five(monkeypatch, fake_datasets, fake_structlog, fake_datasets, fake_structlog):
    resp = _run(monkeypatch, ["a", "b", "c", "d", "e"], {"a", "b", "c"})
    assert resp.partial_failure is not None
    assert "3 of 5 config(s)" in resp.partial_failure


def test_partial_failure_count_one_of_ten(monkeypatch, fake_datasets, fake_structlog, fake_datasets, fake_structlog):
    cfgs = [f"c{i}" for i in range(10)]
    resp = _run(monkeypatch, cfgs, {"c7"})
    assert resp.partial_failure is not None
    assert "1 of 10 config(s)" in resp.partial_failure


def test_partial_failure_none_when_no_failures(monkeypatch, fake_datasets, fake_structlog, fake_datasets, fake_structlog):
    resp = _run(monkeypatch, ["a", "b"], set())
    assert resp.partial_failure is None
