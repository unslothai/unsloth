# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A download speed limit is stored per install, keeps a limited job on HTTP, and paces the worker's transfer."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import time
import types as _types
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.settings as settings
import storage.studio_db as studio_db
import utils.download_speed_limit as speed_limit
from hub.services import download_lifecycle

from .test_hf_download_pinned_revision import _OLD, _REPO, _WORKER, _FakeHub, _repo_dir
from .test_hf_download_reclaim_race import _worker_env


@pytest.fixture
def store(monkeypatch):
    values: dict = {}
    monkeypatch.setattr(
        studio_db, "get_app_setting", lambda key, fallback = None: values.get(key, fallback)
    )
    monkeypatch.setattr(
        studio_db, "upsert_app_settings", lambda updates: values.update(updates) or values
    )
    return values


def test_nothing_stored_means_no_limit(store):
    assert speed_limit.get_download_speed_limit_mbps() is None
    assert store == {}


def test_a_stored_limit_is_returned(store):
    store[speed_limit.DOWNLOAD_SPEED_LIMIT_SETTING_KEY] = 8
    assert speed_limit.get_download_speed_limit_mbps() == 8.0


@pytest.mark.parametrize("junk", ["fast", -3, 0, True, None])
def test_junk_on_disk_reads_as_no_limit(store, junk):
    store[speed_limit.DOWNLOAD_SPEED_LIMIT_SETTING_KEY] = junk
    assert speed_limit.get_download_speed_limit_mbps() is None


def test_set_validates_and_persists(store):
    assert speed_limit.set_download_speed_limit_mbps(8) == 8.0
    assert store[speed_limit.DOWNLOAD_SPEED_LIMIT_SETTING_KEY] == 8.0
    assert speed_limit.set_download_speed_limit_mbps(None) is None
    assert store[speed_limit.DOWNLOAD_SPEED_LIMIT_SETTING_KEY] is None
    assert speed_limit.set_download_speed_limit_mbps(2.5) == 2.5
    assert speed_limit.set_download_speed_limit_mbps(0) is None
    for junk in ("8", -1, True, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            speed_limit.set_download_speed_limit_mbps(junk)


@pytest.fixture
def client(store):
    app = FastAPI()
    app.include_router(settings.router)
    app.dependency_overrides[settings.get_current_subject] = lambda: "admin"
    return TestClient(app, raise_server_exceptions = False)


def test_route_reports_no_limit_by_default(client):
    assert client.get("/download-speed-limit").json() == {"mbps": None}


def test_route_saves_and_clears_a_limit(client, store):
    assert client.put("/download-speed-limit", json = {"mbps": 8}).json() == {"mbps": 8.0}
    assert store[speed_limit.DOWNLOAD_SPEED_LIMIT_SETTING_KEY] == 8.0
    assert client.put("/download-speed-limit", json = {"mbps": None}).json() == {"mbps": None}
    assert store[speed_limit.DOWNLOAD_SPEED_LIMIT_SETTING_KEY] is None


def test_route_rejects_a_negative_limit(client):
    assert client.put("/download-speed-limit", json = {"mbps": -1}).status_code == 422


def test_a_managed_account_downloads_under_the_owners_limit(monkeypatch):
    from utils.account_context import AccountContext, current_account_id, run_as

    dbs: dict = {}

    def _db() -> dict:
        return dbs.setdefault(current_account_id(), {})

    monkeypatch.setattr(
        studio_db, "get_app_setting", lambda key, fallback = None: _db().get(key, fallback)
    )
    monkeypatch.setattr(
        studio_db, "upsert_app_settings", lambda updates: _db().update(updates) or _db()
    )
    speed_limit.set_download_speed_limit_mbps(8)

    user = AccountContext("alice", "alice")
    assert run_as(user, speed_limit.get_download_speed_limit_mbps) == 8.0
    assert run_as(user, download_lifecycle.resolve_requested_use_xet, "xet", True)[0] is False


def _spawn_env(monkeypatch) -> dict:
    captured = {}

    class _Fake:
        pid = 4242

    def _fake_popen(*_args, **popen_kwargs):
        captured.update(popen_kwargs["env"])
        return _Fake()

    monkeypatch.setattr(download_lifecycle.subprocess, "Popen", _fake_popen)
    download_lifecycle.spawn_worker(["--repo-id", _REPO], None, use_xet = False)
    return captured


def test_spawn_worker_hands_the_limit_to_the_worker(monkeypatch, store):
    monkeypatch.delenv(speed_limit.DOWNLOAD_SPEED_LIMIT_ENV, raising = False)
    assert speed_limit.DOWNLOAD_SPEED_LIMIT_ENV not in _spawn_env(monkeypatch)
    store[speed_limit.DOWNLOAD_SPEED_LIMIT_SETTING_KEY] = 8
    assert _spawn_env(monkeypatch)[speed_limit.DOWNLOAD_SPEED_LIMIT_ENV] == "8.0"


@pytest.mark.parametrize("mode", ["xet", "auto", None])
def test_a_limit_keeps_a_download_on_http(store, mode):
    store[speed_limit.DOWNLOAD_SPEED_LIMIT_SETTING_KEY] = 8
    use_xet, reason = download_lifecycle.resolve_requested_use_xet(mode, True)
    assert use_xet is False
    assert "limit" in reason.lower()


@pytest.mark.parametrize("mode", ["xet", "auto", None])
@pytest.mark.parametrize("limit", [None, 8])
def test_a_file_too_large_for_https_stays_on_xet(store, monkeypatch, mode, limit):
    monkeypatch.setattr(
        download_lifecycle, "resolve_effective_use_xet", lambda requested: requested
    )
    store[speed_limit.DOWNLOAD_SPEED_LIMIT_SETTING_KEY] = limit
    use_xet, reason = download_lifecycle.resolve_requested_use_xet(
        mode, True, largest_file_bytes = 54_400_261_312
    )
    assert use_xet is True
    assert ("limit" in reason.lower()) is (limit is not None)


_BIG = hashlib.sha256(b"paced").digest() * (24 * 1024 * 1024 // 32)


def _timed_run(hub: _FakeHub, root: Path, limit_mbps: str | None) -> float:
    env = _worker_env(hub, root)
    env.pop(speed_limit.DOWNLOAD_SPEED_LIMIT_ENV, None)
    if limit_mbps is not None:
        env[speed_limit.DOWNLOAD_SPEED_LIMIT_ENV] = limit_mbps
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, str(_WORKER), "--repo-id", _REPO, "--transport", "http"],
        cwd = _BACKEND_DIR,
        env = env,
        capture_output = True,
        text = True,
        timeout = 180,
    )
    elapsed = time.monotonic() - started
    assert result.returncode == 0, result.stderr
    assert (_repo_dir(root) / "snapshots" / _OLD / "model.safetensors").read_bytes() == _BIG
    return elapsed


def test_the_worker_paces_the_transfer_to_the_limit(tmp_path):
    hub = _FakeHub()
    hub.commits[_OLD]["model.safetensors"] = _BIG
    try:
        unlimited = _timed_run(hub, tmp_path / "unlimited", None)
        # 24 MB at 32 Mbit/s is six seconds of pacing, less the 0.25 s burst credit.
        limited = _timed_run(hub, tmp_path / "limited", "32")
    finally:
        hub.close()
    assert limited >= 5.7, (unlimited, limited)
    assert limited - unlimited >= 3.0, (unlimited, limited)


@pytest.mark.parametrize(
    "limit, chunk", [("8", 500_000), ("0.1", 64 * 1024), (None, 10 * 1024 * 1024)]
)
def test_a_limit_shrinks_the_write_chunk_so_progress_keeps_moving(monkeypatch, limit, chunk):
    monkeypatch.delenv(speed_limit.DOWNLOAD_SPEED_LIMIT_ENV, raising = False)
    if limit is not None:
        monkeypatch.setenv(speed_limit.DOWNLOAD_SPEED_LIMIT_ENV, limit)
    code = (
        f"import runpy; runpy.run_path({str(_WORKER)!r}, run_name = 'w')['_install_speed_limit']()\n"
        "from huggingface_hub import constants; print(constants.DOWNLOAD_CHUNK_SIZE)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd = _BACKEND_DIR, capture_output = True, text = True, timeout = 120
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.split()[-1]) == chunk


@pytest.mark.parametrize("limit", [None, 8])
def test_capabilities_tell_the_frontend_a_limit_forces_http(store, limit):
    import main

    store[speed_limit.DOWNLOAD_SPEED_LIMIT_SETTING_KEY] = limit
    caps = main.studio_download_transport_capabilities(_current_subject = "admin")
    assert main.TransportCapabilities(**caps).speed_limited is (limit is not None)
