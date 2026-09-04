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
    assert (
        storage.update_password(admin, "first-generated-pw-1", require_must_change = True) is not None
    )
    # must_change is now 0, so a second guarded write is refused (rowcount 0)...
    assert storage.update_password(admin, "second-generated-pw-2", require_must_change = True) is None
    from auth import hashing

    salt, pwd_hash, _jwt, _mc = storage.get_user_and_secret(admin)
    assert hashing.verify_password("first-generated-pw-1", salt, pwd_hash) is True
    # ...while the unguarded path (an explicit reset/change) still applies.
    assert storage.update_password(admin, "explicit-change-pw-3") is not None
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
        # **_kwargs: the real start_studio_tunnel takes managed_by/admission, and
        # start_cloudflare_tunnel passes managed_by = "colab".
        lambda port, **_kwargs: "https://example.trycloudflare.com",
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
    # Mirrors the real contract: the rotated JWT secret on success, None on a
    # rejected guard. Returning a bool here would make `False` read as committed.
    monkeypatch.setattr(
        storage, "update_password", lambda *a, **k: rotations.append(a) or "rotated-secret"
    )

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
        lambda port, **_kwargs: started.update(called = True) or "https://example.trycloudflare.com",
    )

    assert colab.start_cloudflare_tunnel(8888) is None
    assert started["called"] is False
    assert storage.requires_password_change(admin) is True


def test_start_cloudflare_tunnel_refuses_after_an_undelivered_rotation(monkeypatch):
    # The retry case. A previous run rotated the password and then failed to
    # render the card, so it refused. must_change_password is 0 now, so every
    # other check in this path passes and the link would go up for an account
    # whose password nobody -- operator included -- has ever seen. The sentinel
    # is the only thing that can still tell.
    admin = _seed_admin(must_change_password = True)
    monkeypatch.setattr(colab, "_display_admin_credentials", lambda *a, **k: False)
    import cloudflare_tunnel

    started = {"called": False}
    monkeypatch.setattr(
        cloudflare_tunnel,
        "start_studio_tunnel",
        lambda port, **_kwargs: started.update(called = True) or "https://example.trycloudflare.com",
    )

    assert colab.start_cloudflare_tunnel(8888) is None       # first run: refuses
    assert storage.requires_password_change(admin) is False  # ...but it committed
    assert storage.credential_undelivered(admin) is True

    assert colab.start_cloudflare_tunnel(8888) is None       # retry: still refuses
    assert started["called"] is False


def test_undelivered_sentinel_clears_on_the_next_password_change(monkeypatch):
    # Self-healing is what keeps the guard from bricking the install: it matches
    # on the committed hash, so `unsloth studio reset-password` releases it even
    # if the sentinel file itself could not be removed.
    admin = _seed_admin(must_change_password = True)
    monkeypatch.setattr(colab, "_display_admin_credentials", lambda *a, **k: False)
    assert colab.start_cloudflare_tunnel(8888) is None
    assert storage.credential_undelivered(admin) is True

    storage.update_password(admin, "operator-chosen-123", revoke_refresh_tokens = True)
    assert storage.credential_undelivered(admin) is False


def test_undelivered_sentinel_ignores_a_hash_it_does_not_match():
    # A leftover file naming a superseded password must not refuse a launch: it
    # no longer describes the live credential, so it is stale, not a warning.
    admin = _seed_admin(must_change_password = False)
    storage.mark_credential_undelivered(admin)
    assert storage.credential_undelivered(admin) is True

    storage.update_password(admin, "operator-chosen-123", revoke_refresh_tokens = True)
    storage.mark_credential_undelivered(admin)
    sentinel = Path(storage._undelivered_credential_path())
    sentinel.write_text("deadbeef" * 8, encoding = "utf-8")
    assert storage.credential_undelivered(admin) is False
    assert sentinel.exists() is False


def test_display_channel_active_false_without_a_kernel():
    # The real probe under pytest: IPython may be importable, but there is no
    # ipykernel-backed shell, so the channel must read as inactive.
    assert _REAL_DISPLAY_CHANNEL_ACTIVE() is False


def test_auto_generate_keeps_credential_when_post_commit_cleanup_raises(monkeypatch):
    # update_password commits the row and only then runs its best-effort cleanup
    # (clear_bootstrap_password, which can still raise on a read-only auth dir or a
    # closed stderr). A raise there used to discard the
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


def test_notebook_text_warns_that_cell_output_is_saved():
    # The runtime card already says the notebook saves its output; the notebook's
    # own intro cell and code comment must not contradict it by promising the
    # credential is never written to disk. Colab autosaves to Drive and a shared or
    # exported copy carries the cell output -- and the password with it.
    import json

    notebook = Path(__file__).resolve().parents[2] / "Unsloth_Studio_Colab.ipynb"
    cells = json.loads(notebook.read_text(encoding = "utf-8"))["cells"]
    text = "".join("".join(cell.get("source", [])) for cell in cells)

    assert "never written to disk" not in text  # the old, false promise
    assert "SAVES its cell output" in text  # intro cell
    assert "This notebook saves cell output" in text  # code-cell comment
    assert text.count("clear the output") >= 2


# ── server-reuse fast path: UNSLOTH_STUDIO_PASSWORD ──────────────────


def test_supplied_password_env_name_matches_the_backend_constant():
    # colab.py hardcodes the name so the strip still runs when the auth import
    # fails; keep it identical to the one run.py/the CLI resolve.
    from auth.terminal_prompt import SUPPLIED_PASSWORD_ENV
    assert colab._SUPPLIED_PASSWORD_ENV == SUPPLIED_PASSWORD_ENV


def test_start_reuse_applies_supplied_password_before_the_tunnel(monkeypatch):
    # Regression (Codex 3651035062, P2): the fast path never calls run_server, so
    # run._apply_supplied_password never runs. The supplied password was ignored
    # while the bootstrap one was still pending (an auto-generated one replaced
    # it), and the plaintext was still in os.environ when cloudflared was spawned,
    # so the child inherited it through the process environment (CWE-214/CWE-526).
    import os

    admin = _seed_admin(must_change_password = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_PASSWORD", "notebook-supplied-pw-1")
    seen: dict = {}
    finalize_calls: list[str] = []
    embed_kwargs: dict = {}
    _patch_start_for_cloudflare(
        monkeypatch, finalize_calls = finalize_calls, embed_kwargs = embed_kwargs
    )

    def _tunnel(port):
        seen["env_at_spawn"] = os.environ.get("UNSLOTH_STUDIO_PASSWORD")
        seen["must_change"] = storage.requires_password_change(admin)
        return "https://share.trycloudflare.com"

    monkeypatch.setattr(colab, "start_cloudflare_tunnel", _tunnel)

    colab.start(cloudflare = True)

    # Stripped BEFORE the subprocess spawn, and applied rather than discarded.
    assert seen["env_at_spawn"] is None
    assert seen["must_change"] is False
    from auth import hashing

    salt, pwd_hash, _jwt, _mc = storage.get_user_and_secret(admin)
    assert hashing.verify_password("notebook-supplied-pw-1", salt, pwd_hash) is True
    assert "UNSLOTH_STUDIO_PASSWORD" not in os.environ


def test_start_reuse_strips_supplied_password_when_one_is_already_set(monkeypatch):
    # The common re-run: a password is already set, so the variable only sets the
    # INITIAL password and must not overwrite it -- but it must still be removed
    # from the environment before anything is spawned.
    import os

    admin = _seed_admin(must_change_password = False)
    monkeypatch.setenv("UNSLOTH_STUDIO_PASSWORD", "notebook-supplied-pw-2")
    seen: dict = {}
    finalize_calls: list[str] = []
    embed_kwargs: dict = {}
    _patch_start_for_cloudflare(
        monkeypatch, finalize_calls = finalize_calls, embed_kwargs = embed_kwargs
    )

    def _tunnel(port):
        seen["env_at_spawn"] = os.environ.get("UNSLOTH_STUDIO_PASSWORD")
        return "https://share.trycloudflare.com"

    monkeypatch.setattr(colab, "start_cloudflare_tunnel", _tunnel)

    colab.start(cloudflare = True)

    assert seen["env_at_spawn"] is None
    from auth import hashing

    salt, pwd_hash, _jwt, _mc = storage.get_user_and_secret(admin)
    # The existing password still authenticates; the supplied one did not win.
    assert hashing.verify_password("bootstrap-secret-123", salt, pwd_hash) is True
    assert hashing.verify_password("notebook-supplied-pw-2", salt, pwd_hash) is False


def test_consume_supplied_password_rejects_invalid_values_but_still_strips(monkeypatch):
    # Too short / whitespace fails the same rules the CLI enforces. In a notebook
    # we cannot exit the process, so the value is ignored (the tunnel path then
    # auto-generates and shows one) -- but never left in the environment.
    import os
    admin = _seed_admin(must_change_password = True)
    for bad in ("short", "has space in it"):
        monkeypatch.setenv("UNSLOTH_STUDIO_PASSWORD", bad)
        colab._consume_supplied_password_on_reuse()
        assert "UNSLOTH_STUDIO_PASSWORD" not in os.environ
        # Left pending, so start_cloudflare_tunnel still secures the link itself.
        assert storage.requires_password_change(admin) is True


def test_consume_supplied_password_never_logs_the_value(monkeypatch, caplog):
    # The password must not reach the logger: the server tees stdout/stderr into a
    # retained session log on disk (run._setup_server_disk_logging).
    import logging
    import os

    _seed_admin(must_change_password = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_PASSWORD", "notebook-supplied-pw-3")
    with caplog.at_level(logging.DEBUG):
        colab._consume_supplied_password_on_reuse()

    assert "notebook-supplied-pw-3" not in caplog.text
    assert "UNSLOTH_STUDIO_PASSWORD" not in os.environ


def test_upgrade_rerun_hands_back_the_cached_credential_before_purging(monkeypatch):
    # An existing Colab user upgrades and re-runs start(). Their password is
    # already set, so nothing is rotated -- but the pre-#7392 flow re-displayed
    # the cached credential on every re-run, and that cache was the only copy a
    # user who cleared their earlier cell output still had. Purging it silently
    # would leave a working password nobody can discover, so hand it back once
    # and THEN remove the file: the CWE-256 fix without the lockout.
    admin = _seed_admin(must_change_password = True)
    existing = "existing-colab-password-from-last-run"
    assert storage.update_password(admin, existing) is not None
    colab._store_colab_login_credentials(admin, existing)
    cache = colab._colab_login_credentials_path()
    assert cache.exists()

    shown: list = []
    monkeypatch.setattr(
        colab,
        "_display_admin_credentials",
        lambda u, p, **kw: shown.append((u, p, kw)) or True,
    )
    monkeypatch.setattr(colab, "_display_channel_active", lambda: True)

    assert colab._auto_generate_colab_admin_password() is None  # nothing rotated
    assert not cache.exists(), "the plaintext cache must still be purged"
    assert len(shown) == 1, shown
    user, pw, kwargs = shown[0]
    assert (user, pw) == (admin, existing)
    assert kwargs.get("final_cached_copy") is True
    # and the credential it handed back is the one that actually works
    from auth import hashing

    salt, pwd_hash, _jwt, _mc = storage.get_user_and_secret(admin)
    assert hashing.verify_password(existing, salt, pwd_hash) is True


def test_upgrade_rerun_purges_a_cached_credential_that_no_longer_works(monkeypatch):
    # Same path, but the cached copy is stale (the user changed the password in
    # the app). Showing it would print a credential that does not authenticate,
    # so it is dropped without being displayed.
    admin = _seed_admin(must_change_password = True)
    colab._store_colab_login_credentials(admin, "stale-cached-password")
    assert storage.update_password(admin, "the-real-current-password") is not None
    cache = colab._colab_login_credentials_path()
    assert cache.exists()

    shown: list = []
    monkeypatch.setattr(
        colab,
        "_display_admin_credentials",
        lambda u, p, **kw: shown.append((u, p, kw)) or True,
    )
    monkeypatch.setattr(colab, "_display_channel_active", lambda: True)

    assert colab._auto_generate_colab_admin_password() is None
    assert not cache.exists()
    assert shown == [], "a credential that no longer authenticates must not be shown"
