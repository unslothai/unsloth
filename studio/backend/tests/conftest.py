# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared pytest configuration for the backend test suite.

Puts the backend root on sys.path (mirrors app launch) and provides a hybrid
``studio_server`` session fixture for end-to-end tests with two modes:
external server (``UNSLOTH_E2E_BASE_URL``/``UNSLOTH_E2E_API_KEY``) for fast
iteration, or a fixture-managed server started/torn down per session for CI.
Model/variant for the managed mode resolve from ``--unsloth-model`` /
``--unsloth-gguf-variant``, then env vars, then ``test_studio_api.py`` defaults.
"""

import os
import sys
from pathlib import Path

import pytest

# Add backend root to sys.path (mirrors app launch)
_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))


# Pytest CLI options


def pytest_addoption(parser):
    group = parser.getgroup(
        "unsloth-e2e",
        "Unsloth Studio end-to-end test options",
    )
    group.addoption(
        "--unsloth-model",
        action = "store",
        default = None,
        help = (
            "GGUF model id used when starting a server for e2e tests. "
            "Ignored if UNSLOTH_E2E_BASE_URL is set. Overrides "
            "UNSLOTH_E2E_MODEL env var. Defaults to test_studio_api.py's "
            "DEFAULT_MODEL."
        ),
    )
    group.addoption(
        "--unsloth-gguf-variant",
        action = "store",
        default = None,
        help = (
            "GGUF variant used when starting a server for e2e tests. "
            "Ignored if UNSLOTH_E2E_BASE_URL is set. Overrides "
            "UNSLOTH_E2E_VARIANT env var. Defaults to test_studio_api.py's "
            "DEFAULT_VARIANT."
        ),
    )


# E2E server fixtures


@pytest.fixture(scope = "session")
def studio_server(request):
    """Yield ``(base_url, api_key)`` for e2e tests.

    Uses ``UNSLOTH_E2E_BASE_URL`` (requires ``UNSLOTH_E2E_API_KEY``) if set,
    else starts/tears down a fresh server via ``_start_server``. Session-scoped
    and lazy so the GGUF load happens at most once and only when requested.
    """
    external_url = os.environ.get("UNSLOTH_E2E_BASE_URL")
    if external_url:
        api_key = os.environ.get("UNSLOTH_E2E_API_KEY")
        if not api_key:
            pytest.skip(
                "UNSLOTH_E2E_BASE_URL is set but UNSLOTH_E2E_API_KEY is "
                "missing — tests that require auth cannot run against an "
                "external server without it.",
            )
        yield external_url, api_key
        return

    # Lazy import; pytest has already loaded test_studio_api, so this is a cache hit.
    import test_studio_api as _e2e

    model = (
        request.config.getoption("--unsloth-model")
        or os.environ.get("UNSLOTH_E2E_MODEL")
        or _e2e.DEFAULT_MODEL
    )
    variant = (
        request.config.getoption("--unsloth-gguf-variant")
        or os.environ.get("UNSLOTH_E2E_VARIANT")
        or _e2e.DEFAULT_VARIANT
    )

    proc, api_key = _e2e._start_server(model, variant)
    try:
        yield f"http://{_e2e.HOST}:{_e2e.PORT}", api_key
    finally:
        _e2e._kill_server(proc)


@pytest.fixture
def base_url(studio_server):
    """Base URL for the e2e Unsloth server (from ``studio_server``)."""
    return studio_server[0]


@pytest.fixture
def api_key(studio_server):
    """API key for the e2e Unsloth server (from ``studio_server``)."""
    return studio_server[1]


# ── RAG fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def rag_home(tmp_path, monkeypatch):
    """Isolate the RAG database under a fresh UNSLOTH_STUDIO_HOME per test.

    Points the storage root at ``tmp_path`` and resets the lazy schema flag so
    each test starts from an empty rag.db. Yields the temp home path.
    """
    from storage import rag_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(rag_db, "_schema_ready", False)
    return tmp_path


@pytest.fixture
def rag_conn(rag_home):
    """A fresh RAG connection bound to the isolated ``rag_home`` database."""
    from storage import rag_db

    conn = rag_db.get_connection()
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def stub_embeddings(monkeypatch):
    """Stub ``core.rag.embeddings`` with deterministic hash-based vectors.

    Lets store / retrieval / ingestion tests run fast without downloading a
    sentence-transformers model. Returns the fixed embedding dimension.
    """
    import hashlib
    import math

    from core.rag import embeddings

    dim = 32

    def _vec(text: str):
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [seed[i % len(seed)] / 255.0 for i in range(dim)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    def fake_encode(
        texts,
        *,
        model_name = None,
        normalize = True,
    ):
        return [_vec(t) for t in texts]

    monkeypatch.setattr(embeddings, "encode", fake_encode)
    monkeypatch.setattr(embeddings, "dim", lambda model_name = None: dim)
    monkeypatch.setattr(
        embeddings,
        "token_counter",
        lambda model_name = None: (lambda t: len(t.split())),
    )
    monkeypatch.setattr(embeddings, "warm", lambda model_name = None: None)
    return dim
