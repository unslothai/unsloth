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
    for key in ("UNSLOTH_STUDIO_NATIVE_TLS", "UV_SYSTEM_CERTS", "UV_NATIVE_TLS"):
        monkeypatch.delenv(key, raising = False)
    yield
    # monkeypatch cannot undo vars that were absent, so drop what setdefault added.
    for key in ("UV_SYSTEM_CERTS", "UV_NATIVE_TLS"):
        os.environ.pop(key, None)


def _fake_truststore(monkeypatch):
    calls = []
    fake = _types.ModuleType("truststore")
    fake.inject_into_ssl = lambda: calls.append("inject")
    monkeypatch.setitem(sys.modules, "truststore", fake)
    return calls


@pytest.mark.parametrize(
    ("platform", "expected"),
    [("darwin", True), ("win32", True), ("linux", True), ("freebsd14", False)],
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
    monkeypatch.setattr(sys, "platform", "freebsd14")
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

    monkeypatch.setattr(sys, "platform", "freebsd14")
    _fake_truststore(monkeypatch)

    assert native_tls.activate_native_tls() is False
    assert "UV_SYSTEM_CERTS" not in os.environ
    assert "UV_NATIVE_TLS" not in os.environ


def test_activate_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(sys, "platform", "freebsd14")
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


def test_linux_activation_does_not_export_uv_env(monkeypatch):
    import os

    monkeypatch.setattr(sys, "platform", "linux")
    _fake_truststore(monkeypatch)

    assert native_tls.activate_native_tls() is True
    assert "UV_SYSTEM_CERTS" not in os.environ
    assert "UV_NATIVE_TLS" not in os.environ


def test_linux_explicit_opt_in_keeps_uv_export(monkeypatch):
    """The platform gate is for the default; an opt-in still carries uv with it.

    Linux exported these whenever the opt-in was set, back when the opt-in was the only
    way to get native TLS there. Dropping that with the default-on change would revert
    uv to its bundled roots on exactly the corporate-gateway hosts that set it.
    """
    import os

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("UNSLOTH_STUDIO_NATIVE_TLS", "1")
    _fake_truststore(monkeypatch)

    assert native_tls.activate_native_tls() is True
    assert os.environ["UV_SYSTEM_CERTS"] == "1"
    assert os.environ["UV_NATIVE_TLS"] == "1"


def test_opt_in_platform_keeps_uv_export(monkeypatch):
    import os

    monkeypatch.setattr(sys, "platform", "freebsd14")
    monkeypatch.setenv("UNSLOTH_STUDIO_NATIVE_TLS", "yes")
    _fake_truststore(monkeypatch)

    assert native_tls.activate_native_tls() is True
    assert os.environ["UV_SYSTEM_CERTS"] == "1"
    assert os.environ["UV_NATIVE_TLS"] == "1"


def test_env_opt_out_wins_on_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("UNSLOTH_STUDIO_NATIVE_TLS", "0")
    assert native_tls.native_tls_enabled() is False


def test_python_39_fails_open_but_keeps_uv_export(monkeypatch):
    import os

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "version_info", (3, 9, 18))
    calls = _fake_truststore(monkeypatch)

    assert native_tls.activate_native_tls() is False
    assert calls == []
    assert os.environ["UV_SYSTEM_CERTS"] == "1"
