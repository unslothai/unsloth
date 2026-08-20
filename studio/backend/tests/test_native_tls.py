# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the OS-trust-store TLS activation (utils/native_tls.py).

truststore is stubbed: these assert only Unsloth's seam -- the platform defaults,
the UNSLOTH_STUDIO_NATIVE_TLS tri-state, idempotency, and the fail-open-to-certifi
behaviour when truststore is unavailable. CPU-only, no network.
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from utils import native_tls


@pytest.fixture(autouse = True)
def _reset_activation(monkeypatch):
    import os

    monkeypatch.setattr(native_tls, "_activated", False)
    for key in (
        "UNSLOTH_STUDIO_NATIVE_TLS",
        "UNSLOTH_STUDIO_DESKTOP_OWNER_KIND",
        "UV_SYSTEM_CERTS",
        "UV_NATIVE_TLS",
    ):
        monkeypatch.delenv(key, raising = False)
    yield
    # monkeypatch cannot undo vars that were absent, so drop what setdefault added.
    for key in ("UNSLOTH_STUDIO_NATIVE_TLS", "UV_SYSTEM_CERTS", "UV_NATIVE_TLS"):
        os.environ.pop(key, None)


def _fake_truststore(monkeypatch):
    calls = []
    fake = _types.ModuleType("truststore")
    fake.inject_into_ssl = lambda: calls.append("inject")
    monkeypatch.setitem(sys.modules, "truststore", fake)
    return calls


@pytest.mark.parametrize(
    ("platform", "expected"),
    [("darwin", True), ("win32", True), ("linux", False)],
)
def test_platform_defaults(monkeypatch, platform, expected):
    monkeypatch.setattr(sys, "platform", platform)
    assert native_tls.native_tls_enabled() is expected


@pytest.mark.parametrize("value", ["0", "false", "NO", " 0 "])
def test_env_opt_out_wins_on_default_on_platform(monkeypatch, value):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("UNSLOTH_STUDIO_NATIVE_TLS", value)
    assert native_tls.native_tls_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "YES"])
def test_env_opt_in_wins_on_default_off_platform(monkeypatch, value):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("UNSLOTH_STUDIO_NATIVE_TLS", value)
    assert native_tls.native_tls_enabled() is True


def test_activate_injects_once(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    calls = _fake_truststore(monkeypatch)

    assert native_tls.activate_native_tls() is True
    assert native_tls.activate_native_tls() is True
    assert calls == ["inject"]


def test_activate_exports_uv_native_tls(monkeypatch):
    import os

    monkeypatch.setattr(sys, "platform", "darwin")
    _fake_truststore(monkeypatch)

    assert native_tls.activate_native_tls() is True
    assert os.environ["UV_SYSTEM_CERTS"] == "1"
    assert os.environ["UV_NATIVE_TLS"] == "1"


def test_activate_keeps_explicit_uv_override(monkeypatch):
    import os

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("UV_SYSTEM_CERTS", "0")
    _fake_truststore(monkeypatch)

    assert native_tls.activate_native_tls() is True
    assert os.environ["UV_SYSTEM_CERTS"] == "0"
    # uv takes either var as an opt-in, so the legacy name must mirror the opt-out.
    assert os.environ["UV_NATIVE_TLS"] == "0"


def test_activate_mirrors_legacy_uv_override(monkeypatch):
    import os

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("UV_NATIVE_TLS", "0")
    _fake_truststore(monkeypatch)

    assert native_tls.activate_native_tls() is True
    assert os.environ["UV_NATIVE_TLS"] == "0"
    assert os.environ["UV_SYSTEM_CERTS"] == "0"


def test_disabled_does_not_touch_uv_env(monkeypatch):
    import os

    monkeypatch.setattr(sys, "platform", "linux")
    _fake_truststore(monkeypatch)

    assert native_tls.activate_native_tls() is False
    assert "UV_SYSTEM_CERTS" not in os.environ
    assert "UV_NATIVE_TLS" not in os.environ


def test_activate_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    calls = _fake_truststore(monkeypatch)

    assert native_tls.activate_native_tls() is False
    assert calls == []


def test_activate_fails_open_without_truststore(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    # None in sys.modules makes `import truststore` raise ImportError.
    monkeypatch.setitem(sys.modules, "truststore", None)

    assert native_tls.activate_native_tls() is False
    # A later call with truststore available recovers.
    calls = _fake_truststore(monkeypatch)
    assert native_tls.activate_native_tls() is True
    assert calls == ["inject"]


def test_activate_fails_open_when_injection_raises(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    fake = _types.ModuleType("truststore")

    def _boom():
        raise OSError("no cert store")

    fake.inject_into_ssl = _boom
    monkeypatch.setitem(sys.modules, "truststore", fake)

    assert native_tls.activate_native_tls() is False


# ── #9218: the Linux desktop's owned backend trusts the OS store by default ──


@pytest.mark.parametrize(
    ("platform", "desktop_kind", "expected"),
    [
        ("linux", "tauri", True),  # .deb/AppImage desktop: icon launch, no shell profile
        ("linux", "", False),  # headless `unsloth studio`: opt-in stays
        ("linux", "other", False),  # unknown owner kind: not the Tauri handshake
    ],
)
def test_linux_desktop_owner_flips_native_tls_default(
    monkeypatch, platform, desktop_kind, expected
):
    monkeypatch.delenv("UNSLOTH_STUDIO_NATIVE_TLS", raising = False)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("UNSLOTH_STUDIO_DESKTOP_OWNER_KIND", desktop_kind)
    assert native_tls.native_tls_enabled() is expected


def test_linux_explicit_opt_out_wins_over_desktop_owner(monkeypatch):
    # The env override outranks the desktop detection: a user who needs
    # certifi-only verification keeps a way back.
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("UNSLOTH_STUDIO_NATIVE_TLS", "0")
    monkeypatch.setenv("UNSLOTH_STUDIO_DESKTOP_OWNER_KIND", "tauri")
    assert native_tls.native_tls_enabled() is False


# ── the `python -c` probe children resolve the same decision ──


def test_activate_spells_the_resolved_decision_into_the_env(monkeypatch):
    import os

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("UNSLOTH_STUDIO_DESKTOP_OWNER_KIND", "tauri")
    _fake_truststore(monkeypatch)

    assert native_tls.activate_native_tls() is True
    # main.py pops the desktop-owner vars before probes spawn, so the children
    # re-resolve from the tri-state flag: a default that flipped on here must
    # read as on there. setdefault, so an explicit user value is never rewritten.
    assert os.environ["UNSLOTH_STUDIO_NATIVE_TLS"] == "1"


def test_disabled_activation_leaves_the_flag_env_absent(monkeypatch):
    import os

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("UNSLOTH_STUDIO_DESKTOP_OWNER_KIND", "")
    _fake_truststore(monkeypatch)

    assert native_tls.activate_native_tls() is False
    assert "UNSLOTH_STUDIO_NATIVE_TLS" not in os.environ


def _run_inline_gate(
    monkeypatch,
    platform,
    owner_kind = None,
    flag = None,
):
    import os

    calls = _fake_truststore(monkeypatch)
    if owner_kind is not None:
        monkeypatch.setenv("UNSLOTH_STUDIO_DESKTOP_OWNER_KIND", owner_kind)
    if flag is not None:
        monkeypatch.setenv("UNSLOTH_STUDIO_NATIVE_TLS", flag)
    # A child's `sys` is whatever it was handed: only platform and path are
    # read. `import truststore` still consults the real sys.modules, which
    # _fake_truststore holds the stub in.
    child_sys = _types.SimpleNamespace(platform = platform, path = [])
    namespace = {"os": os, "sys": child_sys, "_TRUSTSTORE_VENDOR": "/vendor"}
    exec(native_tls.inline_gate_source(), namespace)
    return calls


def test_inline_gate_injects_for_linux_desktop_child(monkeypatch):
    # A child launched directly by the desktop app sees the handshake marker.
    assert _run_inline_gate(monkeypatch, "linux", owner_kind = "tauri") == ["inject"]


def test_inline_gate_skips_headless_linux_child(monkeypatch):
    assert _run_inline_gate(monkeypatch, "linux", owner_kind = "") == []


def test_inline_gate_opt_out_wins_over_desktop_owner(monkeypatch):
    assert _run_inline_gate(monkeypatch, "linux", owner_kind = "tauri", flag = "0") == []


def test_inline_gate_keeps_platform_and_opt_in_defaults(monkeypatch):
    assert _run_inline_gate(monkeypatch, "darwin", owner_kind = "") == ["inject"]
    assert _run_inline_gate(monkeypatch, "linux", owner_kind = "", flag = "1") == ["inject"]
