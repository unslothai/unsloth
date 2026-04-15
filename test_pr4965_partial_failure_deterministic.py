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


def _run_with_error(monkeypatch, upstream_msg):
    import datasets as ds
    from huggingface_hub.utils import HfHubHTTPError

    monkeypatch.setattr(
        ds,
        "get_dataset_config_names",
        lambda name, token = None: ["ok", "bad"],
    )

    class _R:
        status_code = 500

    def _splits(name, config_name = None, token = None):
        if config_name == "ok":
            return ["train"]
        e = HfHubHTTPError(upstream_msg)
        e.response = _R()
        raise e

    monkeypatch.setattr(ds, "get_dataset_split_names", _splits)
    req = DatasetSplitsRequest(dataset_name = "owner/t")
    return rd.get_dataset_splits(req, current_subject = "t").partial_failure


def test_partial_failure_message_is_deterministic_across_errors(monkeypatch):
    m1 = _run_with_error(monkeypatch, "Error A with URL https://a.b/c")
    m2 = _run_with_error(monkeypatch, "Error B completely different text")
    m3 = _run_with_error(monkeypatch, "X" * 4000)
    assert m1 is not None and m2 is not None and m3 is not None
    assert m1 == m2 == m3


def test_partial_failure_message_deterministic_for_same_counts(monkeypatch):
    import datasets as ds
    from huggingface_hub.utils import HfHubHTTPError

    class _R:
        status_code = 500

    def _splits_factory(bad_name):
        def _splits(name, config_name = None, token = None):
            if config_name == bad_name:
                e = HfHubHTTPError("error_of_any_kind")
                e.response = _R()
                raise e
            return ["train"]

        return _splits

    monkeypatch.setattr(
        ds,
        "get_dataset_config_names",
        lambda name, token = None: ["c1", "c2", "c3"],
    )
    monkeypatch.setattr(ds, "get_dataset_split_names", _splits_factory("c1"))
    r1 = rd.get_dataset_splits(
        DatasetSplitsRequest(dataset_name = "owner/t"),
        current_subject = "t",
    ).partial_failure

    monkeypatch.setattr(ds, "get_dataset_split_names", _splits_factory("c3"))
    r2 = rd.get_dataset_splits(
        DatasetSplitsRequest(dataset_name = "owner/t"),
        current_subject = "t",
    ).partial_failure

    assert r1 == r2
