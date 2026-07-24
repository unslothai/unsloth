# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression coverage for Colab iframe embedding (#7344)."""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import colab


def _mock_google_colab_modules(colab_mod):
    """Mock ``google`` and ``google.colab`` for environments without Google packages."""
    google_mod = types.ModuleType("google")
    google_mod.colab = colab_mod
    return {"google": google_mod, "google.colab": colab_mod}


def test_short_colab_url_truncates_proxy_host():
    url = "https://8888-gpu-a100-s-kkb-usc1f0-9hzedjcxrlu8-f.us-central1-0.prod.colab.dev/"
    assert colab._short_colab_url(url, 8888) == "https://8888-gpu-..."


def test_short_colab_url_falls_back_on_unexpected_shape():
    assert colab._short_colab_url("https://example.com", 8888) == "https://example.com"


def test_is_colab_proxy_url_requires_https_proxy():
    assert colab._is_colab_proxy_url("https://8888-test.prod.colab.dev/", 8888) is True
    assert colab._is_colab_proxy_url("http://localhost:8888", 8888) is False
    assert colab._is_colab_proxy_url("http://127.0.0.1:8888", 8888) is False


def test_ready_card_html_does_not_open_colab_proxy_in_new_tab():
    """Colab proxy hosts 404 as top-level tabs (#7349 reporter); never window.open them."""
    html = colab._ready_card_html("https://8888-test.prod.colab.dev/", 8888)
    assert "window.open" not in html
    assert 'href="https://8888-test.prod.colab.dev/"' not in html
    assert "start(cloudflare=True)" in html


def test_ready_card_html_points_to_cloudflare_when_link_ready(monkeypatch):
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: True)
    html = colab._ready_card_html(
        "https://8888-test.prod.colab.dev/",
        8888,
        has_cloudflare_link = True,
    )
    assert "Cloudflare link above" in html


def test_ready_card_html_warns_when_cloudflare_tunnel_missing(monkeypatch):
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: True)
    html = colab._ready_card_html(
        "https://8888-test.prod.colab.dev/",
        8888,
        cloudflare_requested = True,
    )
    assert "Could not open a Cloudflare tunnel" in html


def test_warn_colab_cloudflare_missing_logs_on_colab_without_tunnel(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: True)
    monkeypatch.setattr(colab.logger, "warning", lambda msg, **kwargs: warnings.append(msg))
    colab._warn_colab_cloudflare_missing(use_cloudflare = True, cloudflare_url = None)
    assert warnings
    assert "Cloudflare tunnel unavailable" in warnings[0]


def test_warn_colab_cloudflare_missing_skips_when_tunnel_ready(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: True)
    with caplog.at_level(logging.WARNING):
        colab._warn_colab_cloudflare_missing(
            use_cloudflare = True,
            cloudflare_url = "https://share.trycloudflare.com",
        )
    assert "Cloudflare tunnel unavailable" not in caplog.text


def test_is_colab_runtime_uses_backend_colab_detector(monkeypatch):
    fake_main = types.ModuleType("main")
    fake_main._IS_COLAB = True
    monkeypatch.setitem(sys.modules, "main", fake_main)
    assert colab._is_colab_runtime() is True
    fake_main._IS_COLAB = False
    assert colab._is_colab_runtime() is False


def test_ready_card_html_uses_cloudflare_hint_on_colab_runtime_localhost(monkeypatch):
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: True)
    html = colab._ready_card_html("http://localhost:8888", 8888)
    assert "window.open" not in html
    assert "start(cloudflare=True)" in html


def test_ready_card_html_keeps_open_button_for_localhost_outside_colab(monkeypatch):
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: False)
    html = colab._ready_card_html("http://localhost:8888", 8888)
    assert "window.open" in html
    assert 'href="http://localhost:8888"' in html
    assert "Open Unsloth Studio" in html


def test_embed_kernel_port_iframe_uses_colab_helper(monkeypatch):
    colab_output = MagicMock()
    google_colab = SimpleNamespace(output = colab_output)
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: True)
    with patch.dict("sys.modules", _mock_google_colab_modules(google_colab)):
        assert colab._embed_kernel_port_iframe(8888) is True
    colab_output.serve_kernel_port_as_iframe.assert_called_once_with(
        8888,
        height = colab._COLAB_IFRAME_HEIGHT,
        width = "100%",
    )


def test_embed_kernel_port_iframe_returns_false_without_colab():
    with patch.dict("sys.modules", _mock_google_colab_modules(None)):
        assert colab._embed_kernel_port_iframe(8888) is False


def test_embed_kernel_port_iframe_skips_colabtools_without_runtime(monkeypatch):
    """colabtools can queue JS without appending an iframe; only trust the helper on Colab."""
    colab_output = MagicMock()
    google_colab = SimpleNamespace(output = colab_output)
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: False)
    with patch.dict("sys.modules", _mock_google_colab_modules(google_colab)):
        assert colab._embed_kernel_port_iframe(8888) is False
    colab_output.serve_kernel_port_as_iframe.assert_not_called()


def test_show_and_embed_prefers_kernel_port_iframe(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(colab, "get_colab_url", lambda port: f"https://{port}-test.prod.colab.dev/")
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: True)
    monkeypatch.setattr(
        colab,
        "show_link",
        lambda port,
        *,
        _url = None,
        has_cloudflare_link = False,
        cloudflare_requested = False: calls.append("show_link"),
    )
    monkeypatch.setattr(
        colab,
        "_embed_kernel_port_iframe",
        lambda port: calls.append("kernel_iframe") or True,
    )
    monkeypatch.setattr(
        colab,
        "_embed_html_iframe",
        lambda url, port: calls.append("html_iframe") or True,
    )

    colab._show_and_embed(8888)

    assert calls == ["show_link", "kernel_iframe"]


def test_show_and_embed_falls_back_to_html_iframe(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(colab, "get_colab_url", lambda port: f"https://{port}-test.prod.colab.dev/")
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: False)
    monkeypatch.setattr(
        colab,
        "show_link",
        lambda port, *, _url = None, has_cloudflare_link = False: None,
    )
    monkeypatch.setattr(colab, "_embed_kernel_port_iframe", lambda port: False)
    monkeypatch.setattr(
        colab,
        "_embed_html_iframe",
        lambda url, port: calls.append((url, port)) or True,
    )

    colab._show_and_embed(8888)

    assert calls == [("https://8888-test.prod.colab.dev/", 8888)]


def test_colab_wants_cloudflare_auto_enables_on_runtime(monkeypatch):
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: True)
    assert colab._colab_wants_cloudflare(None) is True
    assert colab._colab_wants_cloudflare(True) is True
    assert colab._colab_wants_cloudflare(False) is False


def test_colab_wants_cloudflare_defaults_off_outside_runtime(monkeypatch):
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: False)
    assert colab._colab_wants_cloudflare(None) is False
    assert colab._colab_wants_cloudflare(True) is True


def test_finalize_colab_admin_password_skips_outside_runtime(monkeypatch):
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: False)
    assert colab._finalize_colab_admin_password() is None


def test_finalize_colab_admin_password_clears_bootstrap_gate(monkeypatch):
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: True)
    monkeypatch.setattr(colab, "_load_colab_login_credentials", lambda: None)
    stored: list[tuple[str, str]] = []
    monkeypatch.setattr(
        colab,
        "_store_colab_login_credentials",
        lambda username, password: stored.append((username, password)),
    )

    storage = SimpleNamespace(
        DEFAULT_ADMIN_USERNAME = "unsloth",
        ensure_default_admin = MagicMock(),
        get_bootstrap_password = MagicMock(return_value = "alpha-beta-gamma"),
        generate_bootstrap_password = MagicMock(return_value = "alpha-beta-gamma"),
        requires_password_change = MagicMock(return_value = True),
        update_password = MagicMock(return_value = True),
    )
    auth_pkg = types.ModuleType("auth")
    auth_pkg.storage = storage
    with patch.dict("sys.modules", {"auth": auth_pkg, "auth.storage": storage}):
        result = colab._finalize_colab_admin_password()

    assert result == ("unsloth", "alpha-beta-gamma")
    storage.ensure_default_admin.assert_called_once()
    storage.update_password.assert_called_once_with("unsloth", "alpha-beta-gamma")
    assert stored == [("unsloth", "alpha-beta-gamma")]


def test_start_skips_finalize_when_cloudflare_disabled(monkeypatch):
    import time

    finalize_calls: list[str] = []
    monkeypatch.setattr(colab, "_is_studio_healthy", lambda port: True)
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: True)
    monkeypatch.setattr(
        colab,
        "_finalize_colab_admin_password",
        lambda: finalize_calls.append("finalize") or ("unsloth", "secret"),
    )
    monkeypatch.setattr(
        colab, "start_cloudflare_tunnel", lambda port: "https://share.trycloudflare.com"
    )
    monkeypatch.setattr(colab, "_publish_cloudflare_url", lambda url: None)
    monkeypatch.setattr(colab, "_show_and_embed", lambda port, **kwargs: None)
    monkeypatch.setattr(colab, "_stop_cloudflare_tunnel", lambda: None)
    monkeypatch.setattr(time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt))

    colab.start(cloudflare = False)

    assert finalize_calls == []


def test_finalize_colab_admin_password_redisplay_on_rerun(monkeypatch):
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: True)
    monkeypatch.setattr(
        colab,
        "_load_colab_login_credentials",
        lambda: ("unsloth", "saved-pass"),
    )

    storage = SimpleNamespace(
        DEFAULT_ADMIN_USERNAME = "unsloth",
        ensure_default_admin = MagicMock(),
        get_bootstrap_password = MagicMock(),
        generate_bootstrap_password = MagicMock(),
        requires_password_change = MagicMock(return_value = False),
        update_password = MagicMock(),
    )
    auth_pkg = types.ModuleType("auth")
    auth_pkg.storage = storage
    with patch.dict("sys.modules", {"auth": auth_pkg, "auth.storage": storage}):
        result = colab._finalize_colab_admin_password()

    assert result == ("unsloth", "saved-pass")
    storage.update_password.assert_not_called()


def test_colab_login_html_includes_credentials():
    html = colab._colab_login_html("unsloth", "alpha-beta-gamma-delta")
    assert "unsloth" in html
    assert "alpha-beta-gamma-delta" in html


def test_show_and_embed_renders_cloudflare_before_colab_login(monkeypatch):
    displayed: list[str] = []
    ipython_display = SimpleNamespace(
        HTML = lambda html: SimpleNamespace(html = html),
        display = lambda html: displayed.append(html.html),
    )

    monkeypatch.setattr(colab, "get_colab_url", lambda port: "https://8888-test.prod.colab.dev/")
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: True)
    monkeypatch.setattr(
        colab,
        "show_link",
        lambda port, *, _url = None, has_cloudflare_link = False, cloudflare_requested = False: None,
    )
    monkeypatch.setattr(colab, "_embed_kernel_port_iframe", lambda port: True)
    with patch.dict("sys.modules", {"IPython.display": ipython_display}):
        colab._show_and_embed(
            8888,
            cloudflare_url = "https://share.trycloudflare.com",
            colab_login = ("unsloth", "secret-pass"),
        )

    assert len(displayed) == 2
    assert "share.trycloudflare.com" in displayed[0]
    assert "secret-pass" in displayed[1]


def test_show_and_embed_skips_iframe_on_colab_when_cloudflare_ready(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(colab, "get_colab_url", lambda port: f"https://{port}-test.prod.colab.dev/")
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: True)
    monkeypatch.setattr(
        colab,
        "show_link",
        lambda port, *, _url = None, has_cloudflare_link = False, cloudflare_requested = False: None,
    )
    monkeypatch.setattr(
        colab,
        "_embed_kernel_port_iframe",
        lambda port: calls.append("kernel_iframe") or True,
    )
    monkeypatch.setattr(
        colab,
        "_embed_html_iframe",
        lambda url, port: calls.append("html_iframe") or True,
    )

    colab._show_and_embed(8888, cloudflare_url = "https://share.trycloudflare.com")

    assert calls == []


def test_show_and_embed_uses_kernel_helper_on_colab_runtime_despite_localhost(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(colab, "get_colab_url", lambda port: f"http://localhost:{port}")
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: True)

    monkeypatch.setattr(
        colab,
        "show_link",
        lambda port,
        *,
        _url = None,
        has_cloudflare_link = False,
        cloudflare_requested = False: calls.append("show_link"),
    )
    monkeypatch.setattr(
        colab,
        "_embed_kernel_port_iframe",
        lambda port: calls.append("kernel_iframe") or True,
    )
    monkeypatch.setattr(
        colab,
        "_embed_html_iframe",
        lambda url, port: calls.append("html_iframe") or True,
    )

    colab._show_and_embed(8888)

    assert calls == ["show_link", "kernel_iframe"]


def test_show_and_embed_skips_kernel_helper_for_localhost_outside_colab(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(colab, "get_colab_url", lambda port: f"http://localhost:{port}")
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: False)

    monkeypatch.setattr(
        colab,
        "show_link",
        lambda port,
        *,
        _url = None,
        has_cloudflare_link = False,
        cloudflare_requested = False: calls.append("show_link"),
    )
    monkeypatch.setattr(
        colab,
        "_embed_kernel_port_iframe",
        lambda port: calls.append("kernel_iframe") or True,
    )
    monkeypatch.setattr(
        colab,
        "_embed_html_iframe",
        lambda url, port: calls.append("html_iframe") or True,
    )

    colab._show_and_embed(8888)

    assert calls == ["show_link", "html_iframe"]


def test_show_and_embed_still_embeds_when_show_link_fails(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(colab, "get_colab_url", lambda port: f"https://{port}-test.prod.colab.dev/")
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: True)
    monkeypatch.setattr(
        colab,
        "show_link",
        lambda port, *, _url = None: (_ for _ in ()).throw(RuntimeError("no display")),
    )
    monkeypatch.setattr(
        colab,
        "_embed_kernel_port_iframe",
        lambda port: calls.append("kernel_iframe") or True,
    )
    monkeypatch.setattr(
        colab,
        "_embed_html_iframe",
        lambda url, port: calls.append("html_iframe") or True,
    )

    colab._show_and_embed(8888)

    assert calls == ["kernel_iframe"]
