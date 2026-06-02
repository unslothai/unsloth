"""Reingest endpoint tests (Backfill UX).

Full end-to-end reingest needs a running studio + a real embedder; that's
covered manually via the curl smoke flow in the plan. Here we cover the
parts that are testable without external models: payload validation.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_BACKEND = REPO_ROOT / "studio" / "backend"
if str(STUDIO_BACKEND) not in sys.path:
    sys.path.insert(0, str(STUDIO_BACKEND))


def _rag_route():
    """Load ``routes/rag.py`` directly, bypassing the ``routes`` package.

    ``from routes.rag import X`` first runs ``routes/__init__.py``, which eagerly
    imports every router — including the datasets router, whose chain does
    ``from datasets import IterableDataset`` at import time. On a GPU-less CI
    runner the unsloth bootstrap can leave ``datasets`` half-initialized, so that
    eager import raises. These tests only need pure helpers from rag.py, so load
    the file on its own (it has no intra-``routes`` imports).
    """
    mod = sys.modules.get("_rag_route_under_test")
    if mod is None:
        spec = importlib.util.spec_from_file_location(
            "_rag_route_under_test", STUDIO_BACKEND / "routes" / "rag.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules["_rag_route_under_test"] = mod
    return mod


def test_reingest_request_accepts_all_optional_fields():
    ReingestKBRequest = _rag_route().ReingestKBRequest

    empty = ReingestKBRequest()
    assert empty.embedding_model is None

    partial = ReingestKBRequest(embedding_model = "BAAI/bge-small-en-v1.5")
    assert partial.embedding_model == "BAAI/bge-small-en-v1.5"
