# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for the Colab/Kaggle notebook launcher."""

import os
import sys
import types

import colab
from utils import notebook_env


def _capture_display(monkeypatch):
    rendered = []
    fake_ipython = types.ModuleType("IPython")
    fake_display = types.ModuleType("IPython.display")

    class _HTML:
        def __init__(self, data):
            self.data = data

    def _display(obj):
        rendered.append(getattr(obj, "data", obj))

    fake_display.HTML = _HTML
    fake_display.display = _display
    fake_ipython.display = fake_display
    monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", fake_display)
    return rendered


def test_kaggle_environment_detects_kaggle_env(monkeypatch):
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    monkeypatch.setattr(notebook_env.Path, "is_dir", lambda self: False)

    assert colab._is_kaggle_environment() is True


def test_kaggle_environment_detects_kaggle_url_base(monkeypatch):
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising = False)
    monkeypatch.setenv("KAGGLE_URL_BASE", "https://www.kaggle.com")
    monkeypatch.setattr(notebook_env.Path, "is_dir", lambda self: False)

    assert colab._is_kaggle_environment() is True


def test_kaggle_environment_requires_runtime_marker_not_path_only(monkeypatch):
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising = False)
    monkeypatch.delenv("KAGGLE_URL_BASE", raising = False)
    monkeypatch.delenv("KAGGLE_CONTAINER_NAME", raising = False)
    monkeypatch.setattr(
        notebook_env.Path,
        "is_dir",
        lambda self: str(self).replace("\\", "/") == "/kaggle/working",
    )

    assert colab._is_kaggle_environment() is False


def test_kaggle_environment_ignores_credential_env(monkeypatch):
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising = False)
    monkeypatch.delenv("KAGGLE_URL_BASE", raising = False)
    monkeypatch.delenv("KAGGLE_CONTAINER_NAME", raising = False)
    monkeypatch.setenv("KAGGLE_KEY", "dummy")
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", "/tmp/kaggle")
    monkeypatch.setattr(notebook_env.Path, "is_dir", lambda self: False)

    assert colab._is_kaggle_environment() is False


def test_colab_environment_requires_content_and_colab_signal(monkeypatch):
    monkeypatch.setattr(
        notebook_env.Path,
        "is_dir",
        lambda self: str(self).replace("\\", "/") == "/content",
    )
    monkeypatch.setattr(notebook_env.importlib.util, "find_spec", lambda name: None)
    assert notebook_env.is_colab_environment({"COLAB_BACKEND_URL": "http://runtime"}) is True
    assert notebook_env.is_colab_environment({}) is False


def test_hosted_notebook_environment_honours_explicit_override(monkeypatch):
    monkeypatch.setattr(notebook_env.Path, "is_dir", lambda self: False)
    assert notebook_env.is_hosted_notebook_environment({"UNSLOTH_STUDIO_HOSTED_NOTEBOOK": "1"})


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

    assert colab._publish_cloudflare_url("https://ready.trycloudflare.com") is True

    assert app.state.cloudflare_url == "https://ready.trycloudflare.com"
    assert app.state.suppress_bootstrap_injection_for_public_tunnel is True
    assert app.state.trust_cloudflare_client_ip is True
    assert app.state.cloudflare_client_ip_requires_frame_cookie is True

    colab._stop_cloudflare_tunnel()

    assert app.state.cloudflare_url is None
    assert app.state.suppress_bootstrap_injection_for_public_tunnel is False
    assert app.state.trust_cloudflare_client_ip is False
    assert app.state.cloudflare_client_ip_requires_frame_cookie is True


def test_publish_cloudflare_url_reports_state_write_failure(monkeypatch):
    class _State:
        def __setattr__(self, name, value):
            raise RuntimeError("state write failed")

    fake_main = types.SimpleNamespace(app = types.SimpleNamespace(state = _State()))
    monkeypatch.setitem(sys.modules, "main", fake_main)

    assert colab._publish_cloudflare_url("https://ready.trycloudflare.com") is False


def test_start_and_publish_tunnel_fails_closed_when_publish_fails(monkeypatch):
    calls = []
    monkeypatch.setattr(colab, "_bootstrap_password_pending", lambda: True)
    monkeypatch.setattr(colab, "_set_public_tunnel_bootstrap_suppression", lambda enabled: True)
    monkeypatch.setattr(
        colab,
        "start_cloudflare_tunnel",
        lambda port, *, allow_bootstrap_pending = False: "https://ready.trycloudflare.com",
    )
    monkeypatch.setattr(colab, "_publish_cloudflare_url", lambda *args, **kwargs: False)

    def _fake_stop(**kwargs):
        calls.append(("stopped", kwargs))

    monkeypatch.setattr(colab, "_stop_cloudflare_tunnel", _fake_stop)

    assert colab._start_and_publish_cloudflare_tunnel(8888, allow_bootstrap_pending = True) is None
    assert calls == [("stopped", {"expected_url": "https://ready.trycloudflare.com"})]


def test_kaggle_reuse_path_keeps_bootstrap_guard_for_external_server(monkeypatch):
    calls = {}
    env = {}
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(colab, "_OWNED_SERVER_APP", None)
    monkeypatch.setattr(colab, "_OWNED_SERVER_PORT", None)
    monkeypatch.setattr(colab, "_is_kaggle_environment", lambda: True)
    monkeypatch.setattr(colab, "_is_studio_healthy", lambda port: True)
    monkeypatch.setattr(colab, "_bootstrap_password_pending", lambda: True)

    def _fake_tunnel(port, *, allow_bootstrap_pending = False):
        calls["tunnel"] = (port, allow_bootstrap_pending)
        return "https://ready.trycloudflare.com" if allow_bootstrap_pending else None

    monkeypatch.setattr(colab, "start_cloudflare_tunnel", _fake_tunnel)
    monkeypatch.setattr(
        colab,
        "_publish_cloudflare_url",
        lambda url, **kwargs: calls.setdefault("published", (url, kwargs)) and True,
    )
    monkeypatch.setattr(
        colab,
        "_show_and_embed",
        lambda port, *, cloudflare_url = None: calls.setdefault("embed", (port, cloudflare_url)),
    )
    monkeypatch.setattr(colab, "_stop_cloudflare_tunnel", lambda **kwargs: calls.setdefault("stopped", kwargs))

    import time

    monkeypatch.setattr(time, "sleep", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    colab.start(8888)

    assert calls["tunnel"] == (8888, False)
    assert "published" not in calls
    assert calls["embed"] == (8888, None)
    assert "stopped" not in calls
    assert env["UNSLOTH_STUDIO_HOSTED_NOTEBOOK"] == "1"
    assert env["UNSLOTH_STUDIO_NOTEBOOK_FRAME_TOKEN"]


def test_kaggle_reuse_path_allows_bootstrap_pending_for_owned_server(monkeypatch):
    calls = {}
    env = {}
    owned_app = object()
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(colab, "_OWNED_SERVER_APP", owned_app)
    monkeypatch.setattr(colab, "_OWNED_SERVER_PORT", 8888)
    monkeypatch.setattr(colab, "_is_kaggle_environment", lambda: True)
    monkeypatch.setattr(colab, "_is_studio_healthy", lambda port: True)
    monkeypatch.setattr(colab, "_bootstrap_password_pending", lambda: True)
    monkeypatch.setattr(
        colab,
        "_set_public_tunnel_bootstrap_suppression",
        lambda enabled: calls.setdefault("suppressed", enabled) or True,
    )

    def _fake_tunnel(port, *, allow_bootstrap_pending = False):
        calls["tunnel"] = (port, allow_bootstrap_pending)
        return "https://ready.trycloudflare.com"

    def _fake_publish(url, **kwargs):
        calls["published"] = (url, kwargs)
        return True

    monkeypatch.setattr(colab, "start_cloudflare_tunnel", _fake_tunnel)
    monkeypatch.setattr(colab, "_publish_cloudflare_url", _fake_publish)
    monkeypatch.setattr(
        colab,
        "_show_and_embed",
        lambda port, *, cloudflare_url = None: calls.setdefault("embed", (port, cloudflare_url)),
    )
    monkeypatch.setattr(colab, "_stop_cloudflare_tunnel", lambda **kwargs: calls.setdefault("stopped", kwargs))

    import time

    monkeypatch.setattr(time, "sleep", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    colab.start(8888)

    assert calls["suppressed"] is True
    assert calls["tunnel"] == (8888, True)
    assert calls["published"] == (
        "https://ready.trycloudflare.com",
        {"suppress_bootstrap": True},
    )
    assert calls["embed"] == (8888, "https://ready.trycloudflare.com")
    assert calls["stopped"] == {"expected_url": "https://ready.trycloudflare.com"}


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
    monkeypatch.setattr(colab, "_stop_cloudflare_tunnel", lambda **kwargs: calls.setdefault("stopped", kwargs))

    import time

    monkeypatch.setattr(time, "sleep", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    colab.start(8888, cloudflare = False)

    assert "published" not in calls
    assert calls["embed"] == (8888, None)
    assert "stopped" not in calls
    assert env["UNSLOTH_STUDIO_HOSTED_NOTEBOOK"] == "1"


def test_show_and_embed_uses_cloudflare_iframe_for_kaggle(monkeypatch):
    rendered = _capture_display(monkeypatch)
    waits = []
    monkeypatch.setenv("UNSLOTH_STUDIO_NOTEBOOK_FRAME_TOKEN", "frame-token")
    monkeypatch.delenv("UNSLOTH_STUDIO_PUBLIC_URL_WAIT_SECONDS", raising = False)
    monkeypatch.setattr(colab, "_is_kaggle_environment", lambda: True)
    monkeypatch.setattr(
        colab,
        "get_colab_url",
        lambda port: (_ for _ in ()).throw(AssertionError("Colab proxy should not be probed")),
    )
    monkeypatch.setattr(
        colab,
        "_wait_for_public_url",
        lambda url, timeout = 45.0: waits.append((url, timeout)) or True,
    )
    monkeypatch.setattr(colab, "_bootstrap_login_notice_html", lambda: None)

    colab._show_and_embed(8888, cloudflare_url = "https://ready.trycloudflare.com")

    html = "\n".join(rendered)
    assert waits == [("https://ready.trycloudflare.com", 8.0)]
    assert 'src="https://ready.trycloudflare.com?__unsloth_frame=frame-token"' in html
    assert 'href="https://ready.trycloudflare.com"' in html


def test_show_and_embed_skips_cloudflare_iframe_when_public_url_not_ready(monkeypatch):
    rendered = _capture_display(monkeypatch)
    waits = []
    monkeypatch.setenv("UNSLOTH_STUDIO_NOTEBOOK_FRAME_TOKEN", "frame-token")
    monkeypatch.setattr(colab, "_is_kaggle_environment", lambda: True)
    monkeypatch.setattr(
        colab,
        "_wait_for_public_url",
        lambda url, timeout = 45.0: waits.append((url, timeout)) and False,
    )
    monkeypatch.setattr(colab, "_bootstrap_login_notice_html", lambda: None)

    colab._show_and_embed(8888, cloudflare_url = "https://not-ready.trycloudflare.com")

    html = "\n".join(rendered)
    assert waits == [("https://not-ready.trycloudflare.com", 8.0)]
    assert "<iframe" not in html
    assert "not loaded" in html
    assert "https://not-ready.trycloudflare.com" in html


def test_show_and_embed_keeps_colab_proxy_if_available(monkeypatch):
    rendered = _capture_display(monkeypatch)
    monkeypatch.setattr(colab, "_is_kaggle_environment", lambda: False)
    monkeypatch.setattr(
        colab,
        "get_colab_url",
        lambda port: "https://proxy.googleusercontent.com/8888",
    )
    monkeypatch.setattr(
        colab,
        "_wait_for_public_url",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Cloudflare wait should not run")
        ),
    )
    monkeypatch.setattr(colab, "_bootstrap_login_notice_html", lambda: None)

    colab._show_and_embed(8888, cloudflare_url = "https://ready.trycloudflare.com")

    html = "\n".join(rendered)
    assert 'src="https://proxy.googleusercontent.com/8888"' in html
    assert 'href="https://ready.trycloudflare.com"' in html


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


def test_stop_cloudflare_tunnel_preserves_state_when_stop_fails(monkeypatch):
    state = types.SimpleNamespace(
        cloudflare_url = "https://ready.trycloudflare.com",
        suppress_bootstrap_injection_for_public_tunnel = True,
        trust_cloudflare_client_ip = True,
        cloudflare_client_ip_requires_frame_cookie = True,
    )
    monkeypatch.setitem(sys.modules, "main", types.SimpleNamespace(app = types.SimpleNamespace(state = state)))

    def _fail_stop():
        raise RuntimeError("still running")

    monkeypatch.setitem(
        sys.modules,
        "cloudflare_tunnel",
        types.SimpleNamespace(stop_studio_tunnel = _fail_stop),
    )

    assert colab._stop_cloudflare_tunnel(expected_url = "https://ready.trycloudflare.com") is False
    assert state.cloudflare_url == "https://ready.trycloudflare.com"
    assert state.suppress_bootstrap_injection_for_public_tunnel is True
    assert state.trust_cloudflare_client_ip is True


def test_stop_cloudflare_tunnel_skips_stale_cleanup(monkeypatch):
    calls = []
    state = types.SimpleNamespace(
        cloudflare_url = "https://newer.trycloudflare.com",
        suppress_bootstrap_injection_for_public_tunnel = True,
        trust_cloudflare_client_ip = True,
        cloudflare_client_ip_requires_frame_cookie = True,
    )
    monkeypatch.setitem(sys.modules, "main", types.SimpleNamespace(app = types.SimpleNamespace(state = state)))
    monkeypatch.setitem(
        sys.modules,
        "cloudflare_tunnel",
        types.SimpleNamespace(stop_studio_tunnel = lambda: calls.append("stop")),
    )

    assert colab._stop_cloudflare_tunnel(expected_url = "https://old.trycloudflare.com") is False
    assert calls == []
    assert state.cloudflare_url == "https://newer.trycloudflare.com"
