# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for the Colab/Kaggle notebook launcher."""

import colab


def test_kaggle_environment_detects_kaggle_env(monkeypatch):
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    monkeypatch.setattr(colab.Path, "exists", lambda self: False)

    assert colab._is_kaggle_environment() is True


def test_start_cloudflare_tunnel_allows_bootstrap_only_when_requested(monkeypatch):
    calls = []

    monkeypatch.setattr(colab, "_bootstrap_password_pending", lambda: True)

    def _fake_start(port):
        calls.append(port)
        return "https://ready.trycloudflare.com"

    fake_module = type("FakeTunnelModule", (), {"start_studio_tunnel": staticmethod(_fake_start)})
    monkeypatch.setitem(__import__("sys").modules, "cloudflare_tunnel", fake_module)

    assert colab.start_cloudflare_tunnel(8888) is None
    assert colab.start_cloudflare_tunnel(8888, allow_bootstrap_password = True) == (
        "https://ready.trycloudflare.com"
    )
    assert calls == [8888]


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
    assert calls == [("https://ready.trycloudflare.com/api/health", 5)]


def test_wait_for_public_url_skips_non_https(monkeypatch):
    def _fail_urlopen(*args, **kwargs):
        raise AssertionError("urlopen should not be called")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", _fail_urlopen)

    assert colab._wait_for_public_url("http://localhost:8888", timeout = 1)
