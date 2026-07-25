# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Colab public-launch auto-password + opt-in same-tab link token.

start(cloudflare=True) with no admin password set auto-generates one, shows it in
the cell, and lets the shared link proceed; a supplied password is respected. The
opt-in link token is appended to the SAME-TAB URL only. Imports the backend
directly, so run under the Unsloth venv."""

from __future__ import annotations

import secrets
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import colab  # noqa: E402
from auth import storage  # noqa: E402


@pytest.fixture(autouse = True)
def isolated_auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    yield


# Captured before the autouse stub below replaces it, so the probe itself can be
# exercised for real.
_REAL_DISPLAY_CHANNEL_ACTIVE = colab._display_channel_active


@pytest.fixture(autouse = True)
def _display_channel_available(monkeypatch):
    # Rotation is now gated on a real notebook display channel (pytest has none),
    # so default it to available; the dedicated tests below flip it to exercise the
    # fail-closed refusal.
    monkeypatch.setattr(colab, "_display_channel_active", lambda: True)


def _seed_admin(*, must_change_password: bool) -> str:
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = "bootstrap-secret-123",
        jwt_secret = secrets.token_urlsafe(64),
        must_change_password = must_change_password,
    )
    return storage.DEFAULT_ADMIN_USERNAME


# ── auto-generate on the public path ─────────────────────────────────


def test_auto_generate_sets_password_and_clears_must_change():
    admin = _seed_admin(must_change_password = True)
    assert storage.requires_password_change(admin) is True

    generated = colab._auto_generate_colab_admin_password()

    assert isinstance(generated, str) and len(generated) >= storage.MIN_PASSWORD_LENGTH
    # Committed: must_change cleared and the new password verifies.
    assert storage.requires_password_change(admin) is False
    salt, pwd_hash, _jwt, _mc = storage.get_user_and_secret(admin)
    from auth import hashing

    assert hashing.verify_password(generated, salt, pwd_hash) is True


def test_auto_generate_noop_when_password_already_set():
    admin = _seed_admin(must_change_password = False)
    # A supplied/changed password must NOT be overwritten.
    assert colab._auto_generate_colab_admin_password() is None
    assert storage.requires_password_change(admin) is False
    from auth import hashing

    salt, pwd_hash, _jwt, _mc = storage.get_user_and_secret(admin)
    assert hashing.verify_password("bootstrap-secret-123", salt, pwd_hash) is True


def test_auto_generate_loses_race_to_a_concurrent_password_change(monkeypatch):
    # Another tab can complete /change-password against the reused Colab server
    # between the requires_password_change() read and the rotation. The commit is a
    # compare-and-set on must_change_password, so the user's chosen password
    # survives and no already-replaced generated value is returned for display
    # (publishing the shared link under it would lock everyone out).
    admin = _seed_admin(must_change_password = True)
    real_update = storage.update_password
    real_requires = storage.requires_password_change
    raced = []

    def _racing_requires_password_change(username):
        # Report "still pending", then let the other tab commit before we write.
        if not raced:
            raced.append(True)
            real_update(username, "user-chosen-password-1", revoke_refresh_tokens = True)
        return True

    monkeypatch.setattr(storage, "requires_password_change", _racing_requires_password_change)

    assert colab._auto_generate_colab_admin_password() is None
    assert raced == [True]

    monkeypatch.setattr(storage, "requires_password_change", real_requires)
    from auth import hashing

    salt, pwd_hash, _jwt, _mc = storage.get_user_and_secret(admin)
    # The concurrently chosen password is intact; the generated one never landed.
    assert hashing.verify_password("user-chosen-password-1", salt, pwd_hash) is True
    assert storage.requires_password_change(admin) is False


def test_update_password_compare_and_set_guard():
    # Storage unit: the guarded update writes only while must_change_password = 1.
    admin = _seed_admin(must_change_password = True)
    assert storage.update_password(admin, "first-generated-pw-1", require_must_change = True) is True
    # must_change is now 0, so a second guarded write is refused (rowcount 0)...
    assert (
        storage.update_password(admin, "second-generated-pw-2", require_must_change = True) is False
    )
    from auth import hashing

    salt, pwd_hash, _jwt, _mc = storage.get_user_and_secret(admin)
    assert hashing.verify_password("first-generated-pw-1", salt, pwd_hash) is True
    # ...while the unguarded path (an explicit reset/change) still applies.
    assert storage.update_password(admin, "explicit-change-pw-3") is True
    salt, pwd_hash, _jwt, _mc = storage.get_user_and_secret(admin)
    assert hashing.verify_password("explicit-change-pw-3", salt, pwd_hash) is True


def test_auto_generate_purges_legacy_plaintext_credential_cache():
    # Upgrade path: the pre-#7392 Colab flow persisted the admin username and
    # password in plaintext in .colab_notebook_login and re-read it on every
    # re-run. Nothing calls that flow now, so an upgraded runtime would keep a
    # readable credential on disk forever (CWE-256) while this flow promises the
    # credential is never persisted. Entering the public-auth path purges it, in
    # BOTH branches (rotating, and already-has-a-password).
    _seed_admin(must_change_password = True)
    legacy = colab._colab_login_credentials_path()
    colab._store_colab_login_credentials("unsloth", "legacy-plaintext-pw")
    assert legacy.is_file()

    assert isinstance(colab._auto_generate_colab_admin_password(), str)
    assert not legacy.exists()

    # Already-set branch: still purged.
    colab._store_colab_login_credentials("unsloth", "legacy-plaintext-pw")
    assert colab._auto_generate_colab_admin_password() is None
    assert not legacy.exists()


def test_start_cloudflare_tunnel_autogenerates_then_proceeds(monkeypatch):
    admin = _seed_admin(must_change_password = True)
    shown = {}

    def _fake_display(u, p):
        shown.update(user = u, pw = p)
        return True

    monkeypatch.setattr(colab, "_display_admin_credentials", _fake_display)
    # start_cloudflare_tunnel does `from cloudflare_tunnel import start_studio_tunnel`;
    # patch the module attribute so no real cloudflared is spawned.
    import cloudflare_tunnel

    monkeypatch.setattr(
        cloudflare_tunnel,
        "start_studio_tunnel",
        lambda port: "https://example.trycloudflare.com",
    )

    url = colab.start_cloudflare_tunnel(8888)

    assert url == "https://example.trycloudflare.com"
    assert shown["user"] == admin
    assert storage.requires_password_change(admin) is False


def test_auto_generate_refuses_without_a_display_channel(monkeypatch):
    # display() does not raise outside a notebook: with no InteractiveShell it
    # prints repr(obj) and a terminal shell renders only text/plain, so the HTML
    # card degrades to "<IPython.core.display.HTML object>" and the password is
    # never seen. start(cloudflare=True) is documented to work on any runtime, so
    # resolve the channel BEFORE rotating -- otherwise the seeded recovery
    # credential is destroyed for a password nobody could read.
    admin = _seed_admin(must_change_password = True)
    monkeypatch.setattr(colab, "_display_channel_active", lambda: False)
    rotations = []
    monkeypatch.setattr(storage, "update_password", lambda *a, **k: rotations.append(a) or True)

    assert colab._auto_generate_colab_admin_password() is None

    assert rotations == []  # nothing was rotated
    assert storage.requires_password_change(admin) is True


def test_start_cloudflare_tunnel_refuses_without_a_display_channel(monkeypatch):
    # End to end: no display channel -> no rotation, and the pending-bootstrap
    # backstop then refuses to publish the shared link.
    admin = _seed_admin(must_change_password = True)
    monkeypatch.setattr(colab, "_display_channel_active", lambda: False)
    import cloudflare_tunnel

    started = {"called": False}
    monkeypatch.setattr(
        cloudflare_tunnel,
        "start_studio_tunnel",
        lambda port: started.update(called = True) or "https://example.trycloudflare.com",
    )

    assert colab.start_cloudflare_tunnel(8888) is None
    assert started["called"] is False
    assert storage.requires_password_change(admin) is True


def test_display_channel_active_false_without_a_kernel():
    # The real probe under pytest: IPython may be importable, but there is no
    # ipykernel-backed shell, so the channel must read as inactive.
    assert _REAL_DISPLAY_CHANNEL_ACTIVE() is False


def test_auto_generate_keeps_credential_when_post_commit_cleanup_raises(monkeypatch):
    # update_password commits the row and only then runs its best-effort cleanup
    # (clear_bootstrap_password, clear_desktop_secret -- the latter opens a second
    # SQLite connection that can hit a lock). A raise there used to discard the
    # generated password even though it was already live, and the caller would then
    # publish the link (must_change is 0) under a credential nobody holds.
    admin = _seed_admin(must_change_password = True)
    real_update = storage.update_password

    def _update_then_explode(username, new_password, **kwargs):
        real_update(username, new_password, **kwargs)
        raise OSError("database is locked")  # post-commit cleanup failure

    monkeypatch.setattr(storage, "update_password", _update_then_explode)

    generated = colab._auto_generate_colab_admin_password()

    assert isinstance(generated, str) and generated
    from auth import hashing

    salt, pwd_hash, _jwt, _mc = storage.get_user_and_secret(admin)
    # The returned value is exactly the committed one, so the card shows a
    # password that actually authenticates.
    assert hashing.verify_password(generated, salt, pwd_hash) is True


def test_auto_generate_drops_credential_when_the_commit_itself_fails(monkeypatch):
    # The converse: the write never landed, so nothing may be displayed -- showing
    # it would publish the link under a password that does not authenticate.
    admin = _seed_admin(must_change_password = True)

    def _explode(username, new_password, **kwargs):
        raise OSError("database is locked")

    monkeypatch.setattr(storage, "update_password", _explode)

    assert colab._auto_generate_colab_admin_password() is None
    assert storage.requires_password_change(admin) is True


def test_start_cloudflare_tunnel_refuses_if_autogen_fails(monkeypatch):
    _seed_admin(must_change_password = True)
    # Simulate auto-generation failing (e.g. DB error): the pending gate then still
    # refuses the shared link (fail safe), returning None.
    monkeypatch.setattr(colab, "_auto_generate_colab_admin_password", lambda: None)
    assert colab.start_cloudflare_tunnel(8888) is None


# ── credential display never persists to disk ────────────────────────


@pytest.fixture
def ipython_display(monkeypatch):
    """Stub IPython.display: CI has no IPython, so importing it would error out.

    Mocks the `IPython` parent too, since `from IPython.display import ...`
    resolves the package first.
    """
    import types

    module = types.ModuleType("IPython.display")
    module.HTML = lambda html: types.SimpleNamespace(data = html)
    module.display = lambda *a, **k: None
    package = types.ModuleType("IPython")
    package.display = module
    monkeypatch.setitem(sys.modules, "IPython", package)
    monkeypatch.setitem(sys.modules, "IPython.display", module)
    return module


def test_display_admin_credentials_warns_that_output_is_saved(monkeypatch, ipython_display):
    # The cell is the only surface a notebook has, and a notebook SAVES its output:
    # Colab autosaves to Drive and an exported/shared .ipynb carries the password
    # with it. The card must say so rather than claim the value is never written to
    # disk, so the operator knows to clear the output (or change the password).
    captured = []
    monkeypatch.setattr(ipython_display, "display", lambda *a, **k: captured.append((a, k)))

    colab._display_admin_credentials("unsloth", "Saved-Output-Pw-42")

    rendered = "".join(str(getattr(obj, "data", obj)) for args, _k in captured for obj in args)
    assert "saves cell output" in rendered
    assert "clear this cell's output" in rendered
    assert "not written to disk" not in rendered  # the old, false promise
    assert "reset-password" in rendered


def test_display_admin_credentials_plaintext_fallback_warns_too(monkeypatch, ipython_display):
    # The text/plain fallback carries the same warning; it is the branch a broken
    # HTML renderer falls back to.
    published = []

    def _display(*args, **kwargs):
        if not kwargs.get("raw"):
            raise RuntimeError("HTML render failed")
        published.append(args)

    monkeypatch.setattr(ipython_display, "display", _display)

    assert colab._display_admin_credentials("unsloth", "Fallback-Pw-7") is True
    rendered = "".join(str(a) for args in published for a in args)
    assert "Fallback-Pw-7" in rendered
    assert "saved" in rendered and "clear it before sharing" in rendered
    assert "not saved to disk" not in rendered


def test_display_admin_credentials_never_writes_to_stdout(monkeypatch, capsys, ipython_display):
    # The auto-generated password must reach the notebook cell only through the
    # IPython display channel (iopub display_data), never sys.stdout/stderr or a
    # logger, because the server tees stdout/stderr to a retained on-disk session
    # log (run._setup_server_disk_logging). Writing it there would persist the
    # credential, contradicting the one-time / non-persistent flow.
    captured = []
    monkeypatch.setattr(ipython_display, "display", lambda *a, **k: captured.append((a, k)))

    secret_pw = "Sup3r-Secret-Pw-Token-xyz"
    colab._display_admin_credentials("unsloth", secret_pw)

    out = capsys.readouterr()
    assert secret_pw not in out.out
    assert secret_pw not in out.err

    # It IS shown via the display channel (HTML card carries it in .data; a raw
    # text/plain fallback carries it in the mimebundle dict).
    shown = False
    for args, _kwargs in captured:
        for obj in args:
            data = getattr(obj, "data", obj)
            if secret_pw in str(data):
                shown = True
    assert shown, "credential was not surfaced through the IPython display channel"


def test_display_admin_credentials_no_display_channel_is_silent(monkeypatch, capsys):
    # If IPython is unavailable, we must NOT fall back to stdout/logging (which the
    # server would tee to disk); showing nothing is the safe outcome.
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "IPython.display" or name.startswith("IPython"):
            raise ImportError("simulated: IPython unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    displayed = colab._display_admin_credentials("unsloth", "Another-Secret-Pw-999")
    assert displayed is False  # no channel -> caller must fail closed
    out = capsys.readouterr()
    assert "Another-Secret-Pw-999" not in out.out
    assert "Another-Secret-Pw-999" not in out.err


def test_display_admin_credentials_returns_true_on_success(monkeypatch, ipython_display):
    # A successful publish through the display channel reports True so the caller
    # may proceed to publish the shared link.
    monkeypatch.setattr(ipython_display, "display", lambda *a, **k: None)
    assert colab._display_admin_credentials("unsloth", "Shown-Pw-123") is True


def test_display_admin_credentials_returns_false_when_display_raises(monkeypatch, ipython_display):
    # Both the HTML and the plain-text publish raise (e.g. a broken display hook):
    # nothing was shown, so report False and never fall back to stdout/logging.
    def _boom(*_a, **_k):
        raise RuntimeError("display channel is broken")

    monkeypatch.setattr(ipython_display, "display", _boom)
    assert colab._display_admin_credentials("unsloth", "Never-Shown-Pw-9") is False


def test_start_cloudflare_tunnel_fails_closed_when_display_fails(monkeypatch):
    # The password is rotated and committed, but it cannot be surfaced in the
    # notebook (display returns False). Publishing the shared link would expose it
    # under a password nobody ever saw, so fail closed: no tunnel is started.
    _seed_admin(must_change_password = True)
    monkeypatch.setattr(colab, "_display_admin_credentials", lambda u, p: False)

    import cloudflare_tunnel

    started = {"called": False}

    def _spy(port):
        started["called"] = True
        return "https://example.trycloudflare.com"

    monkeypatch.setattr(cloudflare_tunnel, "start_studio_tunnel", _spy)

    assert colab.start_cloudflare_tunnel(8888) is None
    assert started["called"] is False  # link never published


# ── opt-in same-tab link token ───────────────────────────────────────


def test_link_token_opt_in_env(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_COLAB_LINK_TOKEN", raising = False)
    assert colab._link_token_opt_in(False) is False
    assert colab._link_token_opt_in(True) is True
    monkeypatch.setenv("UNSLOTH_STUDIO_COLAB_LINK_TOKEN", "1")
    assert colab._link_token_opt_in(False) is True


def test_append_link_token_same_tab_only():
    assert colab._append_link_token("https://x/", None) == "https://x/"
    assert colab._append_link_token("https://x/", "") == "https://x/"
    assert colab._append_link_token("https://x/", "tok") == "https://x/?link_token=tok"
    assert colab._append_link_token("https://x/?a=1", "tok") == "https://x/?a=1&link_token=tok"


def test_mint_same_tab_link_token_exchangeable_once():
    admin = _seed_admin(must_change_password = False)
    token = colab._mint_same_tab_link_token()
    assert isinstance(token, str) and token
    from auth.authentication import exchange_link_token

    # It is a real, single-use token bound to the admin.
    assert exchange_link_token(token) == admin
    assert exchange_link_token(token) is None


# ── post-merge: start() routes credentials through the auto-password path ─────


def _patch_start_for_cloudflare(monkeypatch, *, finalize_calls, embed_kwargs):
    """Stub start()'s side effects so it runs to the keepalive loop, then bails.

    Records any call to _finalize_colab_admin_password (the #7349 bootstrap-finalize +
    plaintext disk-cache path) and the kwargs passed to _show_and_embed.
    """
    import time

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
    monkeypatch.setattr(colab, "_mint_same_tab_link_token", lambda: "same-tab-token")
    monkeypatch.setattr(
        colab, "_show_and_embed", lambda port, **kwargs: embed_kwargs.update(kwargs)
    )
    monkeypatch.setattr(colab, "_stop_cloudflare_tunnel", lambda: None)
    monkeypatch.setattr(time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt))


def test_start_cloudflare_never_finalizes_or_caches_credentials(monkeypatch):
    # Post-merge invariant: on the shared-link path start() must NOT call the #7349
    # finalize/disk-cache flow. Doing so would clear the must-change gate before
    # start_cloudflare_tunnel runs (defeating the auto-generate rotation) and would
    # persist the admin password in plaintext on disk, both of which #7392 forbids.
    finalize_calls: list[str] = []
    embed_kwargs: dict = {}
    _patch_start_for_cloudflare(
        monkeypatch, finalize_calls = finalize_calls, embed_kwargs = embed_kwargs
    )

    colab.start(cloudflare = True)

    assert finalize_calls == []  # credential handled by start_cloudflare_tunnel only
    assert "colab_login" not in embed_kwargs or embed_kwargs["colab_login"] is None
    # No opt-in link token requested -> the same-tab URL carries none.
    assert embed_kwargs.get("same_tab_link_token") is None


def test_start_threads_same_tab_link_token_when_opted_in(monkeypatch):
    # The opt-in one-time link token reaches _show_and_embed (same-tab URL only).
    finalize_calls: list[str] = []
    embed_kwargs: dict = {}
    _patch_start_for_cloudflare(
        monkeypatch, finalize_calls = finalize_calls, embed_kwargs = embed_kwargs
    )

    colab.start(cloudflare = True, link_token = True)

    assert finalize_calls == []
    assert embed_kwargs.get("same_tab_link_token") == "same-tab-token"
