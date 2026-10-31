# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Developer mode's CPU video opt-in (unslothai/unsloth#9458).

Off by default; when on, only a plain CPU-only host (reason ``no_accelerator``) is admitted, as
experimental. Every other refusal stays one. ``UNSLOTH_VIDEO_ALLOW_CPU`` overrides the persisted
owner setting both ways. The setting is installation wide: written only by the owner, read from the
owner's store whoever asks. Hardware probes are mocked, so this runs on any host.
"""

from __future__ import annotations

import sys
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import utils.hardware.hardware as hw
from auth import policy
from auth.authentication import get_current_subject
from utils import video_cpu_settings as vcs
from utils.account_context import (
    OWNER,
    AccountContext,
    bind_account,
    current_account,
    reset_account,
)

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture
def store(monkeypatch):
    """A fake studio_db keyed by the account that is bound when it is called."""
    per_account: dict[str, dict] = {}
    studio_db = types.ModuleType("storage.studio_db")

    def get_app_setting(key, fallback = None):
        return per_account.get(current_account().account_id, {}).get(key, fallback)

    def upsert_app_settings(settings, **_kw):
        per_account.setdefault(current_account().account_id, {}).update(settings)
        return dict(per_account[current_account().account_id])

    studio_db.get_app_setting = get_app_setting
    studio_db.upsert_app_settings = upsert_app_settings
    monkeypatch.setitem(sys.modules, "storage.studio_db", studio_db)
    monkeypatch.delenv(vcs.VIDEO_CPU_ENV, raising = False)
    return per_account


def _host(
    monkeypatch,
    *,
    device = hw.DeviceType.CPU,
    torch = True,
    apple = False,
    system = "Linux",
    chat_only_reason = "no_gpu",
    mps = None,
    torch_import_error = None,
):
    monkeypatch.setattr(hw, "_has_torch", lambda: torch)
    monkeypatch.setattr(hw, "get_device", lambda: device)
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: apple)
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", chat_only_reason)
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", None)
    monkeypatch.setattr(hw, "TORCH_IMPORT_ERROR", torch_import_error)
    monkeypatch.setattr(hw, "current_chat_only_verdict", lambda: (chat_only_reason, None))
    monkeypatch.setattr(hw, "_torch_mps_available", lambda: torch if mps is None else mps)
    monkeypatch.setattr(hw.platform, "system", lambda: system)
    # Independent of the interpreter running the suite (an x86_64 Python on Apple Silicon is Rosetta).
    monkeypatch.setattr(hw, "_running_under_rosetta", lambda: False)


def _opt_in(store, value):
    store.setdefault(OWNER.account_id, {})[vcs.VIDEO_CPU_SETTING_KEY] = value


# -- capability matrix ---------------------------------------------------------------------------


def test_cpu_only_host_stays_refused_by_default(monkeypatch, store):
    _host(monkeypatch)
    cap = hw.video_capability()
    assert cap["video_supported"] is False
    assert cap["video_unsupported_reason"] == "no_accelerator"
    assert cap["video_cpu_experimental"] is False
    # The host is one the opt-in would admit, so Settings can offer it.
    assert cap["video_cpu_opt_in_applicable"] is True
    assert "Developer mode" in cap["video_unsupported_message"]


def test_opt_in_admits_a_cpu_only_host_as_experimental(monkeypatch, store):
    _host(monkeypatch)
    _opt_in(store, True)
    cap = hw.video_capability()
    assert cap == {
        "video_supported": True,
        "video_unsupported_reason": None,
        "video_unsupported_message": None,
        "video_cpu_experimental": True,
        "video_cpu_opt_in_applicable": True,
    }


def test_disabling_restores_the_gate(monkeypatch, store):
    _host(monkeypatch)
    _opt_in(store, True)
    assert hw.video_capability()["video_supported"] is True
    _opt_in(store, False)
    cap = hw.video_capability()
    assert cap["video_supported"] is False
    assert cap["video_unsupported_reason"] == "no_accelerator"


@pytest.mark.parametrize(
    "host, reason",
    [
        (dict(chat_only_reason = "detection_failed"), "detection_failed"),
        (dict(torch = False), "pytorch_not_installed"),
        (dict(system = "Darwin"), "macos_unsupported"),
        (dict(chat_only_reason = "torch_cpu_build"), "torch_cpu_build"),
        (dict(chat_only_reason = "torch_cuda_unavailable"), "torch_cuda_unavailable"),
        (dict(device = hw.DeviceType.MLX, apple = True, system = "Darwin", mps = False), "mps_unavailable"),
        (
            dict(
                device = hw.DeviceType.CPU,
                apple = True,
                system = "Darwin",
                torch = False,
                chat_only_reason = "mlx_unavailable",
                torch_import_error = "OSError('broken')",
            ),
            "detection_failed",
        ),
    ],
)
def test_every_other_refusal_stays_blocked_with_the_opt_in_on(monkeypatch, store, host, reason):
    """Only the missing accelerator is waived: a broken probe, a missing or broken torch, an Intel
    Mac or a GPU torch cannot reach would be silently pushed to the CPU instead of repaired."""
    _host(monkeypatch, **host)
    _opt_in(store, True)
    monkeypatch.setenv(vcs.VIDEO_CPU_ENV, "1")
    cap = hw.video_capability()
    assert cap["video_supported"] is False
    assert cap["video_unsupported_reason"] == reason
    assert cap["video_cpu_experimental"] is False
    assert cap["video_cpu_opt_in_applicable"] is False


@pytest.mark.parametrize("device", [hw.DeviceType.CUDA, hw.DeviceType.XPU])
def test_accelerated_hosts_are_never_marked_experimental(monkeypatch, store, device):
    _host(monkeypatch, device = device, chat_only_reason = None)
    _opt_in(store, True)
    cap = hw.video_capability()
    assert cap["video_supported"] is True
    assert cap["video_cpu_experimental"] is False
    assert cap["video_cpu_opt_in_applicable"] is False


def test_apple_silicon_with_metal_is_not_marked_experimental(monkeypatch, store):
    _host(monkeypatch, device = hw.DeviceType.MLX, apple = True, system = "Darwin", chat_only_reason = None)
    _opt_in(store, True)
    cap = hw.video_capability()
    assert cap["video_supported"] is True
    assert cap["video_cpu_experimental"] is False


def test_an_unreadable_setting_reads_as_off(monkeypatch, store):
    _host(monkeypatch)

    def boom():
        raise RuntimeError("settings store unavailable")

    monkeypatch.setattr(vcs, "video_cpu_enabled", boom)
    cap = hw.video_capability()
    assert cap["video_supported"] is False
    assert cap["video_unsupported_reason"] == "no_accelerator"


# -- env / setting precedence --------------------------------------------------------------------


@pytest.mark.parametrize(
    "env, setting, expected",
    [
        (None, None, False),  # default off
        (None, True, True),
        (None, False, False),
        ("1", None, True),
        ("true", False, True),  # env on beats setting off
        ("0", True, False),  # env off beats setting on
        ("off", True, False),
        ("", True, True),  # empty env is unset
        ("maybe", True, True),  # unrecognised env falls through to the setting
        ("maybe", None, False),
    ],
)
def test_env_overrides_the_setting_both_ways(monkeypatch, store, env, setting, expected):
    _host(monkeypatch)
    if env is not None:
        monkeypatch.setenv(vcs.VIDEO_CPU_ENV, env)
    if setting is not None:
        _opt_in(store, setting)
    assert vcs.video_cpu_enabled() is expected
    assert hw.video_capability()["video_supported"] is expected


def test_the_setting_is_read_from_the_owner_store_for_any_account(monkeypatch, store):
    _host(monkeypatch)
    _opt_in(store, True)
    store.setdefault(ALICE.account_id, {})[vcs.VIDEO_CPU_SETTING_KEY] = False
    token = bind_account(ALICE)
    try:
        assert vcs.get_video_cpu_setting() is True
        assert hw.video_capability()["video_cpu_experimental"] is True
    finally:
        reset_account(token)


def test_setting_rejects_non_booleans(store):
    with pytest.raises(ValueError):
        vcs.set_video_cpu_setting("sometimes")


# -- settings route ------------------------------------------------------------------------------


def _client(account):
    from routes import settings as settings_route

    app = FastAPI()

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    app.include_router(settings_route.router, prefix = "/api/settings")
    return TestClient(app)


@pytest.fixture
def multi_user(monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)


def test_a_managed_account_cannot_enable_cpu_video(monkeypatch, store, multi_user):
    _host(monkeypatch)
    with _client(ALICE) as client:
        response = client.put("/api/settings/developer-mode", json = {"video_cpu": True})
    assert response.status_code == 403
    assert vcs.VIDEO_CPU_SETTING_KEY not in store.get(OWNER.account_id, {})
    assert hw.video_capability()["video_supported"] is False


def test_the_owner_toggles_cpu_video_and_reads_it_back(monkeypatch, store, multi_user):
    _host(monkeypatch)
    with _client(OWNER) as client:
        before = client.get("/api/settings/developer-mode").json()
        on = client.put("/api/settings/developer-mode", json = {"video_cpu": True})
        assert on.status_code == 200
        after_on = hw.video_capability()
        off = client.put("/api/settings/developer-mode", json = {"video_cpu": False}).json()
    assert before["video_cpu"] is False and before["default_video_cpu"] is False
    assert before["video_cpu_applicable"] is True
    assert before["min_ram_bytes"] == vcs.VIDEO_CPU_MIN_RAM_GB * 1024**3
    assert on.json()["video_cpu"] is True and on.json()["video_cpu_setting"] is True
    assert store[OWNER.account_id][vcs.VIDEO_CPU_SETTING_KEY] is False
    assert after_on["video_cpu_experimental"] is True
    assert off["video_cpu"] is False
    assert hw.video_capability()["video_supported"] is False


def test_a_managed_account_can_read_but_not_write(monkeypatch, store, multi_user):
    _host(monkeypatch)
    _opt_in(store, True)
    with _client(ALICE) as client:
        response = client.get("/api/settings/developer-mode")
    assert response.status_code == 200
    assert response.json()["video_cpu"] is True


def test_the_route_reports_the_env_override(monkeypatch, store, multi_user):
    _host(monkeypatch)
    monkeypatch.setenv(vcs.VIDEO_CPU_ENV, "0")
    with _client(OWNER) as client:
        body = client.put("/api/settings/developer-mode", json = {"video_cpu": True}).json()
    # The setting is saved, but the env decides the effective answer, and says so.
    assert body["video_cpu_setting"] is True
    assert body["video_cpu"] is False
    assert body["video_cpu_env_override"] is False


def test_the_route_rejects_a_non_boolean(monkeypatch, store, multi_user):
    _host(monkeypatch)
    with _client(OWNER) as client:
        response = client.put("/api/settings/developer-mode", json = {"video_cpu": "yes"})
    assert response.status_code == 422


def test_gpu_host_reports_the_opt_in_as_not_applicable(monkeypatch, store, multi_user):
    _host(monkeypatch, device = hw.DeviceType.CUDA, chat_only_reason = None)
    with _client(OWNER) as client:
        body = client.get("/api/settings/developer-mode").json()
    assert body["video_cpu_applicable"] is False


# -- wiring (source) -----------------------------------------------------------------------------


def test_health_reports_the_opt_in_only_for_a_no_gpu_verdict():
    """The sidebar greys the Video row on a measured no_gpu host; health carries the opt-in so the
    row stays navigable once the owner enabled it. Read only on that host, so no other poll pays."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "main.py").read_text(encoding = "utf-8")
    block = src[src.index('authed["video_cpu_opt_in"] = False') :][:400]
    assert 'if snapshot[0] and snapshot[1] == "no_gpu":' in block
    assert "video_cpu_enabled()" in block


def test_hardware_endpoints_carry_the_experimental_marker():
    # Both system endpoints spread video_capability(), so the marker rides on both unchanged.
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "main.py").read_text(encoding = "utf-8")
    assert src.count("**video_capability()") >= 2
    hook = (
        Path(__file__).resolve().parent.parent.parent
        / "frontend"
        / "src"
        / "hooks"
        / "use-hardware-info.ts"
    ).read_text(encoding = "utf-8")
    assert "video_cpu_experimental" in hook
