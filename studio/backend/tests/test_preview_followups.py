# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit coverage for the preview follow-ups: rate limiter, client IP, kill switch."""

from pathlib import Path
import sys
import types as _types


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import utils.preview_rate_limit as rl
from utils.client_ip import client_ip
from utils.preview_sharing_settings import (
    DEFAULT_PREVIEW_SHARING_ENABLED,
    _coerce_bool,
    get_preview_sharing_enabled,
)


# ── Rate limiter ─────────────────────────────────────────────────────────────


def test_rate_limit_per_key(monkeypatch):
    monkeypatch.setattr(rl, "_MAX_REQUESTS", 3)
    rl.reset()
    assert rl.check_rate_limit("ip1") == 0
    assert rl.check_rate_limit("ip1") == 0
    assert rl.check_rate_limit("ip1") == 0
    # 4th request over the ceiling -> positive retry-after seconds.
    assert rl.check_rate_limit("ip1") > 0
    # A different client is unaffected.
    assert rl.check_rate_limit("ip2") == 0


def test_rate_limit_window_rolls_off(monkeypatch):
    monkeypatch.setattr(rl, "_MAX_REQUESTS", 1)
    monkeypatch.setattr(rl, "_WINDOW_SECONDS", 10.0)
    rl.reset()
    t = {"now": 1000.0}
    monkeypatch.setattr(rl.time, "monotonic", lambda: t["now"])
    assert rl.check_rate_limit("ip") == 0
    assert rl.check_rate_limit("ip") > 0  # immediately over
    t["now"] += 11.0  # window elapsed
    assert rl.check_rate_limit("ip") == 0


def test_rate_limit_eviction_does_not_reset_active_bucket(monkeypatch):
    # A flood of distinct keys must not cycle the table and clear a live limit.
    monkeypatch.setattr(rl, "_MAX_REQUESTS", 1)
    monkeypatch.setattr(rl, "_MAX_BUCKETS", 2)
    rl.reset()
    assert rl.check_rate_limit("a") == 0
    assert rl.check_rate_limit("a") > 0  # 'a' throttled (active)
    assert rl.check_rate_limit("b") == 0
    assert rl.check_rate_limit("b") > 0  # 'b' throttled; table now full of actives
    # A new key can't evict an active bucket -> denied (fail closed)...
    assert rl.check_rate_limit("c") > 0
    # ...and the flood did not reset 'a'.
    assert rl.check_rate_limit("a") > 0


# ── Client IP ────────────────────────────────────────────────────────────────


class _Req:
    def __init__(
        self,
        host,
        headers = None,
    ):
        self.client = _types.SimpleNamespace(host = host) if host else None
        self.headers = headers or {}


def test_client_ip_uses_socket_peer_by_default(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_TRUST_FORWARDED", raising = False)
    # Forwarded header is ignored unless the operator opts in.
    req = _Req("203.0.113.9", {"x-forwarded-for": "198.51.100.7"})
    assert client_ip(req) == "203.0.113.9"
    assert client_ip(None) == "_unknown"


def test_client_ip_uses_rightmost_forwarded_when_trusted(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_TRUST_FORWARDED", "1")
    # Leftmost is client-spoofable; the trusted proxy appends the real peer on the
    # right, so the rightmost hop is the one we key on.
    req = _Req("127.0.0.1", {"x-forwarded-for": "1.2.3.4, 198.51.100.7"})
    assert client_ip(req) == "198.51.100.7"


def test_client_ip_uses_cf_connecting_ip_on_loopback(monkeypatch):
    # Managed Cloudflare tunnel terminates at loopback; key by the real visitor.
    monkeypatch.delenv("UNSLOTH_STUDIO_TRUST_FORWARDED", raising = False)
    req = _Req("127.0.0.1", {"cf-connecting-ip": "198.51.100.7"})
    assert client_ip(req) == "198.51.100.7"


def test_client_ip_ignores_cf_header_from_non_loopback(monkeypatch):
    # A direct (non-loopback) caller can't spoof CF-Connecting-IP to skew the limit.
    monkeypatch.delenv("UNSLOTH_STUDIO_TRUST_FORWARDED", raising = False)
    req = _Req("203.0.113.9", {"cf-connecting-ip": "198.51.100.7"})
    assert client_ip(req) == "203.0.113.9"


def test_client_ip_loopback_without_cf_returns_peer(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_TRUST_FORWARDED", raising = False)
    assert client_ip(_Req("127.0.0.1")) == "127.0.0.1"


# ── Kill-switch setting ──────────────────────────────────────────────────────


def test_sharing_defaults_enabled_and_coerces():
    assert DEFAULT_PREVIEW_SHARING_ENABLED is True
    assert _coerce_bool("off") is False
    assert _coerce_bool("on") is True
    assert _coerce_bool(True) is True
    assert _coerce_bool("nonsense") is None


def test_sharing_missing_key_defaults_enabled(monkeypatch):
    import storage.studio_db as sdb
    monkeypatch.setattr(sdb, "get_app_setting", lambda key, fallback = None: None)
    assert get_preview_sharing_enabled() is True


def test_sharing_read_error_fails_closed(monkeypatch):
    # A transient settings-DB failure must not reopen the public surface.
    import storage.studio_db as sdb

    def _boom(*args, **kwargs):
        raise RuntimeError("settings db unavailable")

    monkeypatch.setattr(sdb, "get_app_setting", _boom)
    assert get_preview_sharing_enabled() is False
