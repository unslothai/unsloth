# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Contract tests for the confirmation + visible-result llama.cpp update (#7003).

Properties covered:
  - remote authenticated + confirmed  -> update proceeds, result names the host
  - remote authenticated, no confirm   -> refused with a clear message, NO swap
  - local / desktop                    -> identical behavior (no same-machine axis)
  - unauthenticated                    -> refused (401), NO swap
  - stale / expired / replayed token   -> refused, NO swap

routes/llama.py is loaded standalone with stubbed auth / loggers / llama_cpp_update
and the real utils.update_confirm, so no heavy backend deps are needed. Both handler
calls and a real FastAPI TestClient (HTTP + auth gate) are exercised.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from typing import Optional

import pytest

pytest.importorskip("fastapi")

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent


def _install_stubs():
    """Register stub packages so routes/llama.py imports cleanly, plus the real
    update_confirm module under its production name."""
    # auth.authentication.get_current_subject -> a real FastAPI dependency that
    # 401s without a valid bearer, so the HTTP tests can prove the auth gate.
    from fastapi import Header, HTTPException

    def get_current_subject(authorization: Optional[str] = Header(default = None)) -> str:
        if authorization == "Bearer good":
            return "operator"
        raise HTTPException(status_code = 401, detail = "unauthorized")

    auth_pkg = types.ModuleType("auth")
    auth_pkg.__path__ = []
    auth_mod = types.ModuleType("auth.authentication")
    auth_mod.get_current_subject = get_current_subject

    # loggers.get_logger -> no-op structured logger.
    loggers_mod = types.ModuleType("loggers")

    class _NopLogger:
        def info(self, *a, **k):
            pass

        def debug(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

    loggers_mod.get_logger = lambda *_a, **_k: _NopLogger()

    # utils.llama_cpp_update -> monkeypatchable get_update_status / start_update.
    utils_pkg = sys.modules.get("utils")
    if utils_pkg is None:
        utils_pkg = types.ModuleType("utils")
        utils_pkg.__path__ = []
        sys.modules["utils"] = utils_pkg

    lcu_mod = types.ModuleType("utils.llama_cpp_update")

    def _default_status(force_refresh: bool = False):
        return {
            "supported": True,
            "update_available": True,
            "stale": False,
            "installed_tag": "b9860",
            "latest_tag": "b9909",
            "published_repo": "unslothai/llama.cpp",
            "installed_at_utc": None,
            "age_days": None,
            "source_build": False,
            "update_size_bytes": 42_000_000,
            "job": {"state": "idle"},
        }

    def _default_start(expected_tag = None):
        return {"started": True, "reason": None, "job": {"state": "running"}}

    lcu_mod.get_update_status = _default_status
    lcu_mod.start_update = _default_start

    # The real confirmation-token module, under its production import name.
    uc_spec = importlib.util.spec_from_file_location(
        "utils.update_confirm", str(_BACKEND / "utils" / "update_confirm.py")
    )
    uc_mod = importlib.util.module_from_spec(uc_spec)
    uc_spec.loader.exec_module(uc_mod)

    for name, mod in (
        ("auth", auth_pkg),
        ("auth.authentication", auth_mod),
        ("loggers", loggers_mod),
        ("utils.llama_cpp_update", lcu_mod),
        ("utils.update_confirm", uc_mod),
    ):
        sys.modules[name] = mod
    return uc_mod


_uc = _install_stubs()


def _load_route():
    spec = importlib.util.spec_from_file_location(
        "llama_route_under_test", str(_BACKEND / "routes" / "llama.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["llama_route_under_test"] = mod  # pydantic forward-ref resolution
    spec.loader.exec_module(mod)
    return mod


rl = _load_route()


@pytest.fixture(autouse = True)
def _reset(monkeypatch):
    _uc.reset_tokens_for_tests()
    # Restore default status/start each test; individual tests override as needed.
    monkeypatch.setattr(
        rl,
        "get_update_status",
        lambda force_refresh = False: {
            "supported": True,
            "update_available": True,
            "installed_tag": "b9860",
            "latest_tag": "b9909",
            "update_size_bytes": 42_000_000,
            "job": {"state": "idle"},
        },
    )
    yield


def _track_start(monkeypatch):
    calls = {"n": 0, "expected_tags": []}

    def _start(expected_tag = None):
        calls["n"] += 1
        calls["expected_tags"].append(expected_tag)
        return {"started": True, "reason": None, "job": {"state": "running"}}

    monkeypatch.setattr(rl, "start_update", _start)
    return calls


# --------------------------------------------------------------------------- #
# update_confirm token unit tests
# --------------------------------------------------------------------------- #


def test_token_roundtrip_single_use():
    tok, exp = _uc.mint_confirm_token("b9909")
    assert isinstance(tok, str) and tok
    ok, reason = _uc.consume_confirm_token(tok, "b9909")
    assert ok and reason is None
    # Single use: a replay is refused.
    ok2, reason2 = _uc.consume_confirm_token(tok, "b9909")
    assert ok2 is False and reason2 == "invalid_token"


def test_token_stale_target_refused():
    tok, _ = _uc.mint_confirm_token("b9909")
    ok, reason = _uc.consume_confirm_token(tok, "b9910")  # offered build changed
    assert ok is False and reason == "stale_target"


def test_token_expired_refused():
    tok, _ = _uc.mint_confirm_token("b9909", ttl_seconds = 0)
    ok, reason = _uc.consume_confirm_token(tok, "b9909")
    assert ok is False and reason == "expired_token"


def test_token_missing_refused():
    ok, reason = _uc.consume_confirm_token(None, "b9909")
    assert ok is False and reason == "invalid_token"


# --------------------------------------------------------------------------- #
# Handler-level: the swap only runs with an explicit confirmation
# --------------------------------------------------------------------------- #


def test_apply_without_confirmation_is_refused_and_never_swaps(monkeypatch):
    calls = _track_start(monkeypatch)
    out = asyncio.run(rl.llama_update(request = None, current_subject = "operator"))
    assert out.started is False
    assert out.reason == "confirmation_required"
    assert "swap" in out.message.lower() or "confirm" in out.message.lower()
    # The result still names the host + build so the refusal is actionable.
    assert out.machine.hostname
    assert out.latest_tag == "b9909"
    assert calls["n"] == 0  # installer NEVER started


def test_apply_with_confirmed_true_proceeds(monkeypatch):
    calls = _track_start(monkeypatch)
    body = rl.LlamaUpdateRequest(confirmed = True)
    out = asyncio.run(rl.llama_update(request = body, current_subject = "operator"))
    assert out.started is True
    assert out.machine.hostname  # which machine
    assert out.latest_tag == "b9909"  # which version
    assert calls["n"] == 1
    # The confirmed target is threaded into start_update so it installs exactly
    # that build (and aborts if latest moved since this refresh).
    assert calls["expected_tags"] == ["b9909"]


def test_apply_with_valid_token_proceeds(monkeypatch):
    calls = _track_start(monkeypatch)
    tok, _ = _uc.mint_confirm_token("b9909")
    body = rl.LlamaUpdateRequest(confirm_token = tok)
    out = asyncio.run(rl.llama_update(request = body, current_subject = "operator"))
    assert out.started is True
    assert calls["n"] == 1
    # Token is single-use: replaying the same body is refused, no second swap.
    out2 = asyncio.run(rl.llama_update(request = body, current_subject = "operator"))
    assert out2.started is False
    assert out2.reason == "invalid_token"
    assert calls["n"] == 1


def test_apply_with_stale_token_is_refused(monkeypatch):
    calls = _track_start(monkeypatch)
    tok, _ = _uc.mint_confirm_token("b9900")  # bound to an older offered build
    body = rl.LlamaUpdateRequest(confirm_token = tok)
    out = asyncio.run(rl.llama_update(request = body, current_subject = "operator"))
    assert out.started is False
    assert out.reason == "stale_target"
    assert calls["n"] == 0


def test_confirm_endpoint_describes_and_mints(monkeypatch):
    out = asyncio.run(rl.llama_update_confirm(current_subject = "operator"))
    assert out.update_available is True
    assert out.appliable is True
    assert out.installed_tag == "b9860"
    assert out.latest_tag == "b9909"
    assert out.machine.hostname
    assert out.confirm_token
    # The freshly minted token applies the update end to end.
    calls = _track_start(monkeypatch)
    body = rl.LlamaUpdateRequest(confirm_token = out.confirm_token)
    applied = asyncio.run(rl.llama_update(request = body, current_subject = "operator"))
    assert applied.started is True
    assert calls["n"] == 1


def test_confirm_endpoint_up_to_date_offers_no_token(monkeypatch):
    monkeypatch.setattr(
        rl,
        "get_update_status",
        lambda force_refresh = False: {
            "update_available": False,
            "installed_tag": "b9909",
            "latest_tag": "b9909",
            "job": {"state": "idle"},
        },
    )
    out = asyncio.run(rl.llama_update_confirm(current_subject = "operator"))
    assert out.update_available is False
    assert out.appliable is False
    assert out.confirm_token is None


def test_confirm_endpoint_reports_local_link_not_up_to_date(monkeypatch):
    # A --with-llama-cpp-dir tree reports local_link=True with update_available=False.
    # The confirm endpoint must surface reason="local_link", not mask it behind the
    # generic up_to_date refusal.
    monkeypatch.setattr(
        rl,
        "get_update_status",
        lambda force_refresh = False: {
            "update_available": False,
            "local_link": True,
            "installed_tag": "b9909",
            "latest_tag": None,
            "job": {"state": "idle"},
        },
    )
    out = asyncio.run(rl.llama_update_confirm(current_subject = "operator"))
    assert out.reason == "local_link"
    assert out.appliable is False
    assert out.confirm_token is None


def test_apply_revalidates_token_against_refreshed_target(monkeypatch):
    # A token confirmed for b9909; a newer build b9910 publishes before apply.
    # The apply must re-resolve the target and refuse the now-stale token rather
    # than install a build the operator never confirmed.
    def _status(force_refresh: bool = False):
        return {
            "supported": True,
            "update_available": True,
            "installed_tag": "b9860",
            "latest_tag": "b9910" if force_refresh else "b9909",
            "update_size_bytes": 42_000_000,
            "job": {"state": "idle"},
        }

    monkeypatch.setattr(rl, "get_update_status", _status)
    calls = _track_start(monkeypatch)
    tok, _ = _uc.mint_confirm_token("b9909")
    body = rl.LlamaUpdateRequest(confirm_token = tok)
    out = asyncio.run(rl.llama_update(request = body, current_subject = "operator"))
    assert out.started is False
    assert out.reason == "stale_target"
    assert calls["n"] == 0  # never installed the unconfirmed b9910


def test_status_reports_machine():
    out = asyncio.run(rl.llama_update_status(force_refresh = False, current_subject = "operator"))
    assert out.machine.hostname
    assert out.machine.platform
    assert out.latest_tag == "b9909"


# --------------------------------------------------------------------------- #
# start_update pins the confirmed target: a release that publishes in the gap
# after confirmation must not be installed in place of the confirmed build.
# --------------------------------------------------------------------------- #


def _load_real_llama_cpp_update():
    """Load the real utils.llama_cpp_update to exercise start_update's
    confirmed-target guard directly. The module-level stubs shadow only
    utils.llama_cpp_update, so this loads it under a private name (leaving that
    stub in place for the handler tests) and its two real deps under their
    production names."""

    def _load(mod_name: str, filename: str):
        spec = importlib.util.spec_from_file_location(
            mod_name, str(_BACKEND / "utils" / filename)
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    for mod_name, filename in (
        ("utils.llama_cpp_freshness", "llama_cpp_freshness.py"),
        ("utils.process_lifetime", "process_lifetime.py"),
    ):
        if mod_name not in sys.modules:
            _load(mod_name, filename)
    return _load("real_llama_cpp_update", "llama_cpp_update.py")


def _stub_marker_update(monkeypatch, lcu, *, resolved_latest: str):
    """Route start_update down the marker branch with a fixed freshly-resolved
    latest tag; no network, no real install-root probing."""
    monkeypatch.setattr(lcu, "_find_binary", lambda: "/llama.cpp/build/bin/llama-server")
    monkeypatch.setattr(lcu, "_active_install_is_local_link", lambda binary: False)
    monkeypatch.setattr(
        lcu,
        "read_install_marker",
        lambda binary: {
            "tag": "b9860",
            "published_repo": "unslothai/llama.cpp",
            "asset": "cuda-x64.zip",
        },
    )
    monkeypatch.setattr(
        lcu, "_installer_script", lambda: Path("/fake/install_llama_prebuilt.py")
    )
    monkeypatch.setattr(lcu, "_install_dir_for", lambda binary: Path("/llama.cpp/build"))
    monkeypatch.setattr(
        lcu,
        "get_update_status",
        lambda force_refresh = False: {
            "update_available": True,
            "installed_tag": "b9860",
            "latest_tag": resolved_latest,
            "job": {"state": "idle"},
        },
    )


def test_start_update_aborts_when_resolved_latest_differs_from_confirmed(monkeypatch):
    # Operator confirmed b9909, but a newer b9910 publishes before the updater
    # re-resolves latest. The confirmed target must win: abort, never install.
    lcu = _load_real_llama_cpp_update()
    lcu._reset_job_for_tests()
    _stub_marker_update(monkeypatch, lcu, resolved_latest = "b9910")
    installs = {"n": 0}
    monkeypatch.setattr(
        lcu, "_run_update", lambda *a, **k: installs.__setitem__("n", installs["n"] + 1)
    )
    out = lcu.start_update("b9909")
    assert out["started"] is False
    assert out["reason"] == "stale_target"
    assert installs["n"] == 0  # never swapped to the unconfirmed b9910


def test_start_update_proceeds_when_resolved_latest_matches_confirmed(monkeypatch):
    # Confirmed target still matches the freshly-resolved latest: install it and
    # pin the installer to exactly that build.
    lcu = _load_real_llama_cpp_update()
    lcu._reset_job_for_tests()
    _stub_marker_update(monkeypatch, lcu, resolved_latest = "b9909")
    spawned: dict = {}

    class _CapturingThread:
        def __init__(self, *, target = None, args = (), name = None, daemon = None):
            spawned["args"] = args

        def start(self):
            spawned["started"] = True

    monkeypatch.setattr(lcu.threading, "Thread", _CapturingThread)
    out = lcu.start_update("b9909")
    assert out["started"] is True
    assert out["reason"] is None
    assert spawned.get("started") is True
    # args = (install_dir, repo, asset, script, pin_release_tag); the installer
    # is pinned to exactly the confirmed build (a pin is disabled on macOS).
    if sys.platform != "darwin":
        assert spawned["args"][4] == "b9909"
    lcu._reset_job_for_tests()


# --------------------------------------------------------------------------- #
# HTTP-level: auth gate + wiring + backwards-compatible bodyless POST
# --------------------------------------------------------------------------- #


def _client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(rl.router, prefix = "/api/llama")
    return TestClient(app, raise_server_exceptions = True)


def test_http_unauthenticated_update_is_refused(monkeypatch):
    calls = _track_start(monkeypatch)
    client = _client()
    # No Authorization header at all -> 401 from the auth dependency.
    r = client.post("/api/llama/update", json = {"confirmed": True})
    assert r.status_code == 401
    assert calls["n"] == 0  # installer never reached


def test_http_authenticated_bodyless_post_refused_no_swap(monkeypatch):
    # Backwards compat: an OLD frontend posts /update with no body. That must be
    # safe -- refused (confirmation_required), NEVER a silent swap.
    calls = _track_start(monkeypatch)
    client = _client()
    r = client.post("/api/llama/update", headers = {"Authorization": "Bearer good"})
    assert r.status_code == 200
    body = r.json()
    assert body["started"] is False
    assert body["reason"] == "confirmation_required"
    assert body["machine"]["hostname"]
    assert calls["n"] == 0


def test_http_two_step_confirm_then_apply(monkeypatch):
    calls = _track_start(monkeypatch)
    client = _client()
    h = {"Authorization": "Bearer good"}

    step1 = client.post("/api/llama/update/confirm", headers = h)
    assert step1.status_code == 200
    tok = step1.json()["confirm_token"]
    assert tok

    step2 = client.post("/api/llama/update", headers = h, json = {"confirm_token": tok})
    assert step2.status_code == 200
    assert step2.json()["started"] is True
    assert step2.json()["machine"]["hostname"]
    assert calls["n"] == 1


def test_http_local_and_remote_behave_identically(monkeypatch):
    # There is no same-machine axis: the contract depends only on auth + confirm,
    # so a "local" and a "remote" authenticated caller get identical outcomes.
    calls = _track_start(monkeypatch)
    client = _client()
    h = {"Authorization": "Bearer good"}
    # Simulate a remote caller by adding proxy/forwarding headers that the closed
    # PR would have treated as "not host" -- here they change nothing.
    remote_h = {**h, "X-Forwarded-For": "203.0.113.7", "CF-Connecting-IP": "203.0.113.7"}

    local = client.post("/api/llama/update", headers = h, json = {"confirmed": True})
    remote = client.post("/api/llama/update", headers = remote_h, json = {"confirmed": True})
    assert local.json()["started"] == remote.json()["started"] == True
    assert calls["n"] == 2  # both proceeded; location was never the gate
