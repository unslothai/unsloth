# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression coverage for Colab iframe embedding (#7344)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import colab


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
    assert "Scroll down" in html


def test_is_colab_runtime_requires_release_tag(monkeypatch):
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising = False)
    with patch.dict("sys.modules", {"google.colab": object()}):
        assert colab._is_colab_runtime() is False

    monkeypatch.setenv("COLAB_RELEASE_TAG", "test")
    with patch.dict("sys.modules", {"google.colab": object()}):
        assert colab._is_colab_runtime() is True

    monkeypatch.setenv("COLAB_RELEASE_TAG", "test")
    with patch.dict("sys.modules", {"google.colab": None}):
        assert colab._is_colab_runtime() is False


def test_ready_card_html_uses_scroll_down_on_colab_runtime_localhost(monkeypatch):
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: True)
    html = colab._ready_card_html("http://localhost:8888", 8888)
    assert "window.open" not in html
    assert "Scroll down" in html


def test_ready_card_html_keeps_open_button_for_localhost_outside_colab(monkeypatch):
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: False)
    html = colab._ready_card_html("http://localhost:8888", 8888)
    assert "window.open" in html
    assert 'href="http://localhost:8888"' in html
    assert "Open Unsloth Studio" in html


def test_embed_kernel_port_iframe_uses_colab_helper():
    colab_output = MagicMock()
    google_colab = SimpleNamespace(output = colab_output)
    with patch.dict("sys.modules", {"google.colab": google_colab}):
        assert colab._embed_kernel_port_iframe(8888) is True
    colab_output.serve_kernel_port_as_iframe.assert_called_once_with(
        8888,
        height = colab._COLAB_IFRAME_HEIGHT,
        width = "100%",
    )


def test_embed_kernel_port_iframe_returns_false_without_colab():
    with patch.dict("sys.modules", {"google.colab": None}):
        assert colab._embed_kernel_port_iframe(8888) is False


def test_show_and_embed_prefers_kernel_port_iframe(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(colab, "get_colab_url", lambda port: f"https://{port}-test.prod.colab.dev/")
    monkeypatch.setattr(
        colab,
        "show_link",
        lambda port, *, _url = None: calls.append("show_link"),
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
    monkeypatch.setattr(colab, "show_link", lambda port, *, _url = None: None)
    monkeypatch.setattr(colab, "_embed_kernel_port_iframe", lambda port: False)
    monkeypatch.setattr(
        colab,
        "_embed_html_iframe",
        lambda url, port: calls.append((url, port)) or True,
    )

    colab._show_and_embed(8888)

    assert calls == [("https://8888-test.prod.colab.dev/", 8888)]


def test_show_and_embed_renders_cloudflare_card(monkeypatch):
    displayed: list[str] = []
    ipython_display = SimpleNamespace(
        HTML = lambda html: SimpleNamespace(html = html),
        display = lambda html: displayed.append(html.html),
    )

    monkeypatch.setattr(colab, "get_colab_url", lambda port: "https://8888-test.prod.colab.dev/")
    monkeypatch.setattr(colab, "show_link", lambda port, *, _url = None: None)
    monkeypatch.setattr(colab, "_embed_kernel_port_iframe", lambda port: True)
    with patch.dict("sys.modules", {"IPython.display": ipython_display}):
        colab._show_and_embed(8888, cloudflare_url = "https://share.trycloudflare.com")

    assert len(displayed) == 1
    assert "share.trycloudflare.com" in displayed[0]


def test_show_and_embed_uses_kernel_helper_on_colab_runtime_despite_localhost(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(colab, "get_colab_url", lambda port: f"http://localhost:{port}")
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: True)
    monkeypatch.setattr(colab, "show_link", lambda port, *, _url = None: calls.append("show_link"))
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
    monkeypatch.setattr(colab, "show_link", lambda port, *, _url = None: calls.append("show_link"))
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
