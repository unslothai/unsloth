import sys
import types
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent / "studio" / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


@pytest.fixture(autouse=True)
def _fake_datasets(monkeypatch):
    """Inject a fake datasets module scoped to each test with auto-teardown."""
    if "datasets" in sys.modules:
        yield
        return

    class _NF(Exception):
        pass

    m = types.ModuleType("datasets")
    m.get_dataset_config_names = lambda *a, **k: []
    m.get_dataset_split_names = lambda *a, **k: []
    m.IterableDataset = type("IterableDataset", (), {})
    m.Dataset = type("Dataset", (), {})
    m.load_dataset = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "datasets", m)
    e = types.ModuleType("datasets.exceptions")
    e.DatasetNotFoundError = _NF
    monkeypatch.setitem(sys.modules, "datasets.exceptions", e)
    yield


@pytest.fixture(autouse=True)
def _fake_structlog(monkeypatch):
    """Inject a fake structlog module scoped to each test with auto-teardown."""
    if "structlog" in sys.modules:
        yield
        return

    class _L:
        def __getattr__(self, n):
            return lambda *a, **k: None

    monkeypatch.setitem(sys.modules, "structlog", types.SimpleNamespace(
        BoundLogger = _L,
        get_logger = lambda *a, **k: _L(),
    ))
    yield


from fastapi import HTTPException  # noqa: E402

from models.datasets import DatasetSplitsRequest  # noqa: E402
import routes.datasets as rd  # noqa: E402


def test_empty_configs_skips_fetch_calls(monkeypatch):
    import datasets as ds

    monkeypatch.setattr(ds, "get_dataset_config_names", lambda n, token = None: [])
    calls = {"n": 0}

    def _splits(*a, **k):
        calls["n"] += 1
        return ["train"]

    monkeypatch.setattr(ds, "get_dataset_split_names", _splits)

    req = DatasetSplitsRequest(dataset_name = "owner/empty")
    with pytest.raises(HTTPException) as ei:
        rd.get_dataset_splits(req, current_subject = "t")
    assert ei.value.status_code == 404
    assert calls["n"] == 0


def test_empty_configs_skips_thread_pool_executor(monkeypatch):
    import datasets as ds
    import concurrent.futures as cf

    monkeypatch.setattr(ds, "get_dataset_config_names", lambda n, token = None: [])
    spawns = {"n": 0}

    original = cf.ThreadPoolExecutor

    class _Tracker(original):
        def __init__(self, *a, **k):
            spawns["n"] += 1
            super().__init__(*a, **k)

    monkeypatch.setattr(cf, "ThreadPoolExecutor", _Tracker)

    req = DatasetSplitsRequest(dataset_name = "owner/empty")
    with pytest.raises(HTTPException) as ei:
        rd.get_dataset_splits(req, current_subject = "t")
    assert ei.value.status_code == 404
    assert spawns["n"] == 0
