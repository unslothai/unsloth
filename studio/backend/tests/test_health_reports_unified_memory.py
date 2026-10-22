# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Health reports apple_silicon, not device_type == mac, since Intel Macs can spill to system RAM."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.hardware as hardware_pkg  # noqa: E402


def _restore_real_logging_modules() -> dict:
    """Drops the loggers/structlog stubs other test files install, since main needs the real packages."""
    removed = {}
    stub = sys.modules.get("loggers")
    if stub is not None and not hasattr(stub, "__path__"):
        for name in [n for n in sys.modules if n == "loggers" or n.startswith("loggers.")]:
            removed[name] = sys.modules.pop(name)
    stub = sys.modules.get("structlog")
    if stub is not None and not hasattr(stub, "BoundLogger"):
        removed["structlog"] = sys.modules.pop("structlog")
    try:
        import structlog  # noqa: F401
    except Exception:
        sys.modules.update(removed)
        pytest.skip("real structlog is required to import main")
    return removed


def _health(
    monkeypatch,
    *,
    apple_silicon: bool,
    authed: bool = True,
) -> dict:
    """Drive /api/health and return the body."""
    _swapped_modules = _restore_real_logging_modules()
    import auth.authentication as _authmod
    import main as main_mod
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    # health_check imports is_apple_silicon at call time, so patch the package attribute.
    monkeypatch.setattr(hardware_pkg, "is_apple_silicon", lambda: apple_silicon)

    hw_mod = main_mod._hw_module
    monkeypatch.setattr(hw_mod, "DEVICE", hw_mod.DeviceType.CPU, raising = False)
    monkeypatch.setattr(hw_mod, "CHAT_ONLY", False, raising = False)
    monkeypatch.setattr(hw_mod, "CHAT_ONLY_REASON", None, raising = False)
    monkeypatch.setattr(main_mod, "_torch_warm_in_progress", lambda: False)
    was_complete = hw_mod.DETECTION_COMPLETE.is_set()
    hw_mod.DETECTION_COMPLETE.set()

    async def _subject(_creds):
        if not authed:
            from fastapi import HTTPException
            raise HTTPException(status_code = 401, detail = "no")
        return "tester"

    monkeypatch.setattr(_authmod, "get_current_subject", _subject)
    app = FastAPI()
    app.add_api_route("/api/health", main_mod.health_check, methods = ["GET"])
    try:
        with TestClient(app) as client:
            headers = {"Authorization": "Bearer probe"} if authed else {}
            return client.get("/api/health", headers = headers).json()
    finally:
        if not was_complete:
            hw_mod.DETECTION_COMPLETE.clear()
        sys.modules.update(_swapped_modules)


def test_apple_silicon_is_reported(monkeypatch):
    body = _health(monkeypatch, apple_silicon = True)
    assert body["apple_silicon"] is True
    assert body["device_type"]


def test_every_other_host_reports_false(monkeypatch):
    """Including an Intel Mac, which is the case device_type cannot distinguish and the
    one the old wording was already right about."""
    body = _health(monkeypatch, apple_silicon = False)
    assert body["apple_silicon"] is False


def test_it_rides_with_device_type(monkeypatch):
    """Both are authed-only. The frontend keys `fetched` on device_type and reads
    apple_silicon on the same terms, so a reply carrying one without the other would make
    the store cache a verdict it was never told."""
    for apple in (True, False):
        body = _health(monkeypatch, apple_silicon = apple)
        assert ("apple_silicon" in body) == ("device_type" in body)


def test_an_unauthenticated_reply_carries_neither(monkeypatch):
    """It fingerprints the host, like device_type, so it stays behind the bearer."""
    body = _health(monkeypatch, apple_silicon = True, authed = False)
    assert "device_type" not in body
    assert "apple_silicon" not in body


def test_the_gate_is_the_one_the_metal_budget_uses(monkeypatch):
    """Not a second definition of Apple Silicon. If these ever diverge, the UI starts
    describing a memory model the loader is not enforcing."""
    import inspect

    import main as main_mod
    from core.inference.llama_cpp import LlamaCppBackend

    assert "from utils.hardware import is_apple_silicon" in inspect.getsource(main_mod.health_check)
    assert "from utils.hardware import is_apple_silicon" in inspect.getsource(
        LlamaCppBackend._apple_metal_memory_budget_bytes
    )
