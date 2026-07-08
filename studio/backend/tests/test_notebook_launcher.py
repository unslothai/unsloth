# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for the Colab/Kaggle notebook launcher."""

import os
import sys
import types

import colab
from utils import notebook_env


def test_kaggle_environment_detects_kaggle_env(monkeypatch):
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    monkeypatch.setattr(notebook_env.Path, "is_dir", lambda self: False)

    assert colab._is_kaggle_environment() is True


def test_kaggle_environment_ignores_credential_env(monkeypatch):
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising = False)
    monkeypatch.delenv("KAGGLE_URL_BASE", raising = False)
    monkeypatch.setenv("KAGGLE_KEY", "dummy")
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", "/tmp/kaggle")
    monkeypatch.setattr(notebook_env.Path, "is_dir", lambda self: False)

    assert colab._is_kaggle_environment() is False


def test_start_cloudflare_tunnel_allows_bootstrap_pending_only_when_requested(monkeypatch):
    calls = []

    monkeypatch.setattr(colab, "_bootstrap_password_pending", lambda: True)

    def _fake_start(port):
        calls.append(port)
        return "https://ready.trycloudflare.com"

    fake_module = type("FakeTunnelModule", (), {"start_studio_tunnel": staticmethod(_fake_start)})
    monkeypatch.setitem(sys.modules, "cloudflare_tunnel", fake_module)

    assert colab.start_cloudflare_tunnel(8888) is None
    assert colab.start_cloudflare_tunnel(8888, allow_bootstrap_pending = True) == (
        "https://ready.trycloudflare.com"
    )
    assert calls == [8888]


def test_publish_cloudflare_url_suppresses_public_bootstrap_injection(monkeypatch):
    app = types.SimpleNamespace(state = types.SimpleNamespace())
    fake_main = types.SimpleNamespace(app = app)
    monkeypatch.setitem(sys.modules, "main", fake_main)
    monkeypatch.setattr(colab, "_bootstrap_password_pending", lambda: True)

    colab._publish_cloudflare_url("https://ready.trycloudflare.com")

    assert app.state.cloudflare_url == "https://ready.trycloudflare.com"
    assert app.state.suppress_bootstrap_injection_for_public_tunnel is True

    colab._stop_cloudflare_tunnel()

    assert app.state.cloudflare_url is None
    assert app.state.suppress_bootstrap_injection_for_public_tunnel is False


def test_kaggle_start_auto_tunnels_and_marks_hosted(monkeypatch):
    calls = {}
    env = {}
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(colab, "_is_kaggle_environment", lambda: True)
    monkeypatch.setattr(colab, "_is_studio_healthy", lambda port: True)

    def _fake_tunnel(port, *, allow_bootstrap_pending = False):
        calls["tunnel"] = (port, allow_bootstrap_pending)
        return "https://ready.trycloudflare.com"

    monkeypatch.setattr(colab, "start_cloudflare_tunnel", _fake_tunnel)
    monkeypatch.setattr(
        colab, "_publish_cloudflare_url", lambda url: calls.setdefault("published", url)
    )
    monkeypatch.setattr(
        colab,
        "_show_and_embed",
        lambda port, *, cloudflare_url = None: calls.setdefault("embed", (port, cloudflare_url)),
    )
    monkeypatch.setattr(colab, "_stop_cloudflare_tunnel", lambda: calls.setdefault("stopped", True))

    import time

    monkeypatch.setattr(time, "sleep", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    colab.start(8888)

    assert calls["tunnel"] == (8888, True)
    assert calls["published"] == "https://ready.trycloudflare.com"
    assert calls["embed"] == (8888, "https://ready.trycloudflare.com")
    assert calls["stopped"] is True
    assert env["UNSLOTH_STUDIO_HOSTED_NOTEBOOK"] == "1"


def test_kaggle_start_explicit_cloudflare_false_disables_tunnel(monkeypatch):
    calls = {}
    env = {}
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(colab, "_is_kaggle_environment", lambda: True)
    monkeypatch.setattr(colab, "_is_studio_healthy", lambda port: True)
    monkeypatch.setattr(
        colab,
        "start_cloudflare_tunnel",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("tunnel should not start")),
    )
    monkeypatch.setattr(
        colab, "_publish_cloudflare_url", lambda url: calls.setdefault("published", url)
    )
    monkeypatch.setattr(
        colab,
        "_show_and_embed",
        lambda port, *, cloudflare_url = None: calls.setdefault("embed", (port, cloudflare_url)),
    )
    monkeypatch.setattr(colab, "_stop_cloudflare_tunnel", lambda: calls.setdefault("stopped", True))

    import time

    monkeypatch.setattr(time, "sleep", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    colab.start(8888, cloudflare = False)

    assert calls["published"] is None
    assert calls["embed"] == (8888, None)
    assert calls["stopped"] is True
    assert env["UNSLOTH_STUDIO_HOSTED_NOTEBOOK"] == "1"


def test_wait_for_public_url_polls_health_endpoint(monkeypatch):
    calls = []

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return _Response()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    assert colab._wait_for_public_url("https://ready.trycloudflare.com", timeout = 1)
    assert calls[0][0] == "https://ready.trycloudflare.com/api/health"
    assert 0 < calls[0][1] <= 1


def test_wait_for_public_url_failure_respects_timeout(monkeypatch):
    calls = []
    now = {"value": 0.0}
    warnings = []

    def _fake_monotonic():
        now["value"] += 0.25
        return now["value"]

    def _fake_sleep(seconds):
        calls.append(("sleep", seconds))
        now["value"] += seconds

    def _fake_urlopen(request, timeout):
        calls.append(("urlopen", timeout))
        raise OSError("not ready")

    import time
    import urllib.request

    monkeypatch.setattr(time, "monotonic", _fake_monotonic)
    monkeypatch.setattr(time, "sleep", _fake_sleep)
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(colab.logger, "warning", warnings.append)

    assert colab._wait_for_public_url("https://ready.trycloudflare.com", timeout = 0.5) is False
    assert calls == [("urlopen", 0.25)]
    assert "did not become reachable" in warnings[0]


def test_wait_for_public_url_skips_non_https(monkeypatch):
    def _fail_urlopen(*args, **kwargs):
        raise AssertionError("urlopen should not be called")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", _fail_urlopen)

    assert colab._wait_for_public_url("http://localhost:8888", timeout = 1)
