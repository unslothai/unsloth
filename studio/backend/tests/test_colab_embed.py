# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression coverage for Colab iframe embedding (#7344)."""

import inspect
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


def test_start_cloudflare_tunnel_marks_colab_owner(monkeypatch):
    calls = []
    tunnel = types.ModuleType("cloudflare_tunnel")
    tunnel.set_studio_tunnel_url_callback = lambda callback: calls.append(("callback", callback))
    tunnel.start_studio_tunnel = lambda port, **kwargs: (
        calls.append((port, kwargs)) or "https://share.trycloudflare.com"
    )
    monkeypatch.setitem(sys.modules, "cloudflare_tunnel", tunnel)
    monkeypatch.setattr(colab, "_bootstrap_password_pending", lambda: False)

    assert colab.start_cloudflare_tunnel(8891) == "https://share.trycloudflare.com"
    assert calls == [
        ("callback", colab._publish_cloudflare_url),
        (8891, {"managed_by": "colab"}),
    ]


def test_colab_start_does_not_republish_returned_url():
    assert "_publish_cloudflare_url(cf_url)" not in inspect.getsource(colab.start)
    assert "cloudflare_url = None" not in inspect.getsource(colab._stop_cloudflare_tunnel)


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
        lambda url, port, **_kwargs: calls.append("html_iframe") or True,
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
        lambda url, port, **_kwargs: calls.append((url, port)) or True,
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
    monkeypatch.setattr(colab, "_colab_credentials_still_valid", lambda username, password: True)

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


def test_finalize_colab_admin_password_drops_stale_cached_credentials(monkeypatch):
    """After an in-app password change the cached first-run password no longer
    authenticates, so it must not be redisplayed (#7349 Codex review)."""
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: True)
    monkeypatch.setattr(
        colab,
        "_load_colab_login_credentials",
        lambda: ("unsloth", "stale-pass"),
    )
    monkeypatch.setattr(colab, "_colab_credentials_still_valid", lambda username, password: False)
    cleared: list[bool] = []
    monkeypatch.setattr(colab, "_clear_colab_login_credentials", lambda: cleared.append(True))

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

    assert result is None
    assert cleared == [True]
    storage.update_password.assert_not_called()


def test_colab_credentials_still_valid_matches_stored_hash(monkeypatch):
    from auth.hashing import hash_password

    salt, pwd_hash = hash_password("right-pass")
    storage = SimpleNamespace(
        get_user_and_secret = MagicMock(return_value = (salt, pwd_hash, "jwt", False)),
    )
    with patch.dict("sys.modules", {"auth.storage": storage}):
        assert colab._colab_credentials_still_valid("unsloth", "right-pass") is True
        assert colab._colab_credentials_still_valid("unsloth", "wrong-pass") is False


def test_colab_credentials_still_valid_false_when_user_missing(monkeypatch):
    storage = SimpleNamespace(get_user_and_secret = MagicMock(return_value = None))
    with patch.dict("sys.modules", {"auth.storage": storage}):
        assert colab._colab_credentials_still_valid("unsloth", "any") is False


def test_colab_login_html_includes_credentials():
    html = colab._colab_login_html("unsloth", "alpha-beta-gamma-delta")
    assert "unsloth" in html
    assert "alpha-beta-gamma-delta" in html
    # The username is fixed, so it reads inline rather than as its own field.
    assert "Username:" not in html


def test_shareable_link_html_embeds_password_under_the_link():
    """The credential belongs in the same card as the button it unlocks."""
    html = colab._shareable_link_html("https://share.trycloudflare.com", "secret-pass", "unsloth")
    assert "share.trycloudflare.com" in html
    assert "secret-pass" in html
    # Username is stated inline, not as its own labelled field.
    assert "Username:" not in html
    assert "unsloth" in html
    # The password must sit after the link, not above it.
    assert html.index("share.trycloudflare.com") < html.index("secret-pass")


def test_shareable_link_html_renders_the_url_as_a_link():
    """The printed URL is an anchor, using the popup-safe open the button uses."""
    html = colab._shareable_link_html("https://share.trycloudflare.com")
    assert '<a href="https://share.trycloudflare.com"' in html
    assert ">https://share.trycloudflare.com</a>" in html
    assert html.count("window.open(this.href,'_blank')") == 2


def test_shareable_link_html_emphasises_the_password():
    """The password is the one thing to copy, so it is enlarged and underlined."""
    html = colab._shareable_link_html("https://share.trycloudflare.com", "secret-pass", "unsloth")
    pw_tag = html[html.index("Password") : html.index("secret-pass")]
    assert "font-size: 24px" in pw_tag
    assert "text-decoration: underline" in pw_tag


def test_shareable_link_html_password_has_no_adjacent_whitespace():
    """Whitespace beside the password is selected with it on a double click."""
    html = colab._shareable_link_html("https://share.trycloudflare.com", "secret-pass", "unsloth")
    before, after = html.split("secret-pass", 1)
    assert before.endswith(">")
    assert after.startswith("<")
    # Label on its own line, so nothing shares the password's text node.
    assert "Password:" not in html
    # Plain selectable text: user-select overrides break double click to select.
    assert "user-select" not in html


def test_shareable_link_html_omits_login_block_without_password():
    html = colab._shareable_link_html("https://share.trycloudflare.com")
    assert "Password" not in html


def test_show_and_embed_folds_login_into_the_cloudflare_card(monkeypatch):
    """One card, not two: the tunnel card carries the password itself."""
    displayed: list[str] = []
    ipython_display = SimpleNamespace(
        HTML = lambda html: SimpleNamespace(html = html),
        display = lambda html: displayed.append(html.html),
    )
    login_cards: list[tuple] = []

    monkeypatch.setattr(colab, "get_colab_url", lambda port: "https://8888-test.prod.colab.dev/")
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: True)
    monkeypatch.setattr(
        colab,
        "_show_colab_login_credentials",
        lambda *args: login_cards.append(args),
    )
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

    assert len(displayed) == 1
    assert "share.trycloudflare.com" in displayed[0]
    assert "secret-pass" in displayed[0]
    assert login_cards == []


def test_show_and_embed_keeps_separate_login_card_without_tunnel(monkeypatch):
    """No tunnel card to fold into, so the standalone login card still renders."""
    login_cards: list[tuple] = []

    monkeypatch.setattr(colab, "get_colab_url", lambda port: "https://8888-test.prod.colab.dev/")
    monkeypatch.setattr(colab, "_is_colab_runtime", lambda: True)
    monkeypatch.setattr(
        colab,
        "_show_colab_login_credentials",
        lambda *args: login_cards.append(args),
    )
    monkeypatch.setattr(
        colab,
        "show_link",
        lambda port, *, _url = None, has_cloudflare_link = False, cloudflare_requested = False: None,
    )
    monkeypatch.setattr(colab, "_embed_kernel_port_iframe", lambda port: True)
    colab._show_and_embed(8888, colab_login = ("unsloth", "secret-pass"))

    assert login_cards == [("unsloth", "secret-pass")]


def test_show_and_embed_skips_ready_card_when_tunnel_is_up(monkeypatch):
    """The ready card only restates the tunnel card and prints a proxy URL that 404s."""
    calls: list[str] = []

    monkeypatch.setattr(colab, "get_colab_url", lambda port: "https://8888-test.prod.colab.dev/")
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
    monkeypatch.setattr(colab, "_embed_kernel_port_iframe", lambda port: True)
    colab._show_and_embed(8888, cloudflare_url = "https://share.trycloudflare.com")

    assert calls == []


def test_show_and_embed_keeps_ready_card_without_tunnel(monkeypatch):
    """Without a tunnel the ready card is the only guidance, so it must stay."""
    calls: list[str] = []

    monkeypatch.setattr(colab, "get_colab_url", lambda port: "https://8888-test.prod.colab.dev/")
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
    monkeypatch.setattr(colab, "_embed_kernel_port_iframe", lambda port: True)
    colab._show_and_embed(8888)

    assert calls == ["show_link"]


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
        lambda url, port, **_kwargs: calls.append("html_iframe") or True,
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
        lambda url, port, **_kwargs: calls.append("html_iframe") or True,
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
        lambda url, port, **_kwargs: calls.append("html_iframe") or True,
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
        lambda url, port, **_kwargs: calls.append("html_iframe") or True,
    )

    colab._show_and_embed(8888)

    assert calls == ["kernel_iframe"]
