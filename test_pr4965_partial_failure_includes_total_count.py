import sys
import types
from pathlib import Path


_BACKEND = Path(__file__).resolve().parent / "studio" / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

if "structlog" not in sys.modules:

    class _L:
        def __getattr__(self, n):
            return lambda *a, **k: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _L,
        get_logger = lambda *a, **k: _L(),
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


def test_partial_failure_count_three_of_five(monkeypatch):
    resp = _run(monkeypatch, ["a", "b", "c", "d", "e"], {"a", "b", "c"})
    assert resp.partial_failure is not None
    assert "3 of 5 config(s)" in resp.partial_failure


def test_partial_failure_count_one_of_ten(monkeypatch):
    cfgs = [f"c{i}" for i in range(10)]
    resp = _run(monkeypatch, cfgs, {"c7"})
    assert resp.partial_failure is not None
    assert "1 of 10 config(s)" in resp.partial_failure


def test_partial_failure_none_when_no_failures(monkeypatch):
    resp = _run(monkeypatch, ["a", "b"], set())
    assert resp.partial_failure is None
