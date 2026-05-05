# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Shared pytest configuration for the backend test suite.

Responsibilities:
1. Put the backend root on sys.path so `from models.inference import ...`
   (and similar flat imports) resolve in test modules — mirrors how the
   app itself is launched.
2. Provide a hybrid ``studio_server`` session fixture for end-to-end tests
   (see ``test_studio_api.py``). The fixture supports two invocation modes:

   a. **External server.** If ``UNSLOTH_E2E_BASE_URL`` is set, tests point
      at an already-running Studio instance. ``UNSLOTH_E2E_API_KEY`` must
      also be set. This is the fast-iteration mode: start the server once
      with ``unsloth studio run ...``, then run pytest against it many
      times with no per-run GGUF load cost.

   b. **Fixture-managed server.** Otherwise, the fixture launches a fresh
      server via ``_start_server`` and tears it down at session end. This
      is the one-shot mode for CI or a clean-slate verification run.

   The model / variant for mode (b) come from ``--unsloth-model`` /
   ``--unsloth-gguf-variant`` pytest options, then ``UNSLOTH_E2E_MODEL`` /
   ``UNSLOTH_E2E_VARIANT`` env vars, then the defaults in
   ``test_studio_api.py``.
"""

import os
import sys
from pathlib import Path

import pytest

# Add backend root to sys.path (mirrors how the app itself is launched)
_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))


# ── Pytest CLI options ───────────────────────────────────────────────


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


# ── E2E server fixtures ──────────────────────────────────────────────


@pytest.fixture(scope = "session")
def studio_server(request):
    """Yield ``(base_url, api_key)`` for e2e tests.

    Resolution order:

    1. If ``UNSLOTH_E2E_BASE_URL`` is set → point at that server,
       require ``UNSLOTH_E2E_API_KEY`` alongside (skip if missing).
    2. Otherwise → start a fresh ``unsloth studio run`` subprocess via
       the existing ``_start_server`` helper in ``test_studio_api.py``
       and tear it down on session teardown.

    Session-scoped so the expensive GGUF load happens at most once per
    pytest invocation. Lazily instantiated — tests that don't request
    the fixture (e.g. the unit tests in ``test_anthropic_messages.py``
    or ``test_help_output``) do not trigger server startup.
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

    # Lazy import: pytest has already loaded test_studio_api into
    # sys.modules by the time any test requests this fixture, so this
    # is a cache hit, not a re-execution.
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
    """Base URL for the e2e Studio server (from ``studio_server``)."""
    return studio_server[0]


@pytest.fixture
def api_key(studio_server):
    """API key for the e2e Studio server (from ``studio_server``)."""
    return studio_server[1]
