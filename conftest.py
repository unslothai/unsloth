import sys
import types
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent / "studio" / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


@pytest.fixture
def fake_structlog(monkeypatch):
    """Inject a fake structlog module scoped to each test with auto-teardown."""
    if "structlog" in sys.modules:
        yield
        return

    class _L:
        def __getattr__(self, n):
            return lambda *a, **k: None

    monkeypatch.setitem(
        sys.modules,
        "structlog",
        types.SimpleNamespace(
            BoundLogger=_L,
            get_logger=lambda *a, **k: _L(),
        ),
    )
    yield


@pytest.fixture
def fake_datasets(monkeypatch):
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
