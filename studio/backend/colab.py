# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Colab helpers for Unsloth Studio. Uses Colab's built-in proxy."""

import os
from pathlib import Path
import sys

# Seed platform._sys_version_cache before attrs->rich->structlog->platform crash on conda Python.
# See: https://github.com/python/cpython/issues/102396
_backend_dir = str(Path(__file__).parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
import _platform_compat  # noqa: F401


from loggers import get_logger

logger = get_logger(__name__)


def get_colab_url(port: int = 8888) -> str:
    """Get the Colab proxy URL for a port.

    Retries 3x validating a real HTTPS Colab URL; falls back to localhost on failure.
    """
    import time as _time

    fallback = f"http://localhost:{port}"

    try:
        from google.colab.output import eval_js
    except ImportError:
        return fallback

    for attempt in range(3):
        try:
            url = eval_js(f"google.colab.kernel.proxyPort({port})", timeout_sec = 10)
            # Valid proxy URL is https:// and embeds the port.
            if url and isinstance(url, str) and url.startswith("https://") and str(port) in url:
                return url.rstrip("/")
        except Exception as e:
            logger.info(f"Note: Could not get Colab URL (attempt {attempt + 1}/3: {e})")
        if attempt < 2:
            _time.sleep(1)

    logger.warning(
        f"Could not get a valid Colab proxy URL after 3 attempts — using localhost fallback. "
        f"The link/iframe may not work from outside the runtime."
    )
    return fallback


def _short_colab_url(url: str, port: int) -> str:
    """Truncated display form of a Colab proxy URL; falls back to the full URL."""
    try:
        port_prefix = f"{port}-"
        idx = url.index(port_prefix)
        next_dash = url.index("-", idx + len(port_prefix))
        return url[: next_dash + 1] + "..."
    except (ValueError, IndexError):
        return url


def _is_colab_proxy_url(url: str, port: int) -> bool:
    """True when *url* looks like a real Colab kernel proxy, not a localhost fallback."""
    return bool(url and isinstance(url, str) and url.startswith("https://") and str(port) in url)


def _is_colab_runtime() -> bool:
    """True on a hosted Colab notebook kernel.

    Reuses the backend's main Colab detector (``/content`` + Colab env / ``google.colab``)
    instead of a single env var, which is not always present on hosted runtimes.
    """
    try:
        from main import _IS_COLAB
        return bool(_IS_COLAB)
    except Exception:
        return False


def _colab_login_credentials_path() -> Path:
    from auth.storage import DB_PATH
    return DB_PATH.parent / ".colab_notebook_login"


def _store_colab_login_credentials(username: str, password: str) -> None:
    """Persist Colab admin credentials for notebook re-runs after interrupt."""
    path = _colab_login_credentials_path()
    try:
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text(f"{username}\n{password}\n", encoding = "utf-8")
        try:
            import os
            os.chmod(path, 0o600)
        except OSError:
            pass
    except OSError as e:
        logger.info(f"Could not persist Colab login credentials ({e}).")


def _load_colab_login_credentials() -> "tuple[str, str] | None":
    """Return stored Colab admin credentials from a previous ``start()`` run, if any."""
    path = _colab_login_credentials_path()
    try:
        if not path.is_file():
            return None
        lines = path.read_text(encoding = "utf-8").splitlines()
        if len(lines) >= 2 and lines[0] and lines[1]:
            return lines[0], lines[1]
    except (OSError, UnicodeDecodeError) as e:
        logger.info(f"Could not load Colab login credentials ({e}).")
    return None


def _clear_colab_login_credentials() -> None:
    """Drop the cached Colab credentials once they no longer authenticate."""
    path = _colab_login_credentials_path()
    try:
        path.unlink(missing_ok = True)
    except OSError as e:
        logger.info(f"Could not clear Colab login credentials ({e}).")


def _colab_credentials_still_valid(username: str, password: str) -> bool:
    """True when *password* still matches the stored admin hash.

    Guards against redisplaying a cached first-run password after the user has
    changed the admin password through the app, which would print credentials
    that no longer authenticate to the current Cloudflare tunnel.
    """
    try:
        from auth.storage import get_user_and_secret
        from auth.hashing import verify_password
    except Exception as e:
        logger.info(f"Could not load auth to validate cached Colab credentials ({e}).")
        return False
    try:
        row = get_user_and_secret(username)
        if not row:
            return False
        salt, pwd_hash = row[0], row[1]
        return bool(verify_password(password, salt, pwd_hash))
    except Exception as e:
        logger.info(f"Could not validate cached Colab credentials ({e}).")
        return False


def _colab_wants_cloudflare(cloudflare: "bool | None") -> bool:
    """Resolve whether to open a Cloudflare tunnel.

    ``None`` auto-enables on real Colab (the in-cell proxy embed is often blank);
    pass ``False`` to opt out.
    """
    if cloudflare is not None:
        return cloudflare
    return _is_colab_runtime()


def _finalize_colab_admin_password() -> "tuple[str, str] | None":
    """Clear the bootstrap-password gate on Colab so Cloudflare tunnels can start.

    Returns ``(username, password)`` for display in the notebook. On first run the
    random admin password is finalized; on later runs (e.g. after interrupt) the
    stored credentials are re-displayed so the Cloudflare link stays usable.
    Anyone who can read this cell already controls the runtime.
    """
    if not _is_colab_runtime():
        return None
    try:
        from auth.storage import (
            DEFAULT_ADMIN_USERNAME,
            ensure_default_admin,
            generate_bootstrap_password,
            get_bootstrap_password,
            requires_password_change,
            update_password,
        )
    except Exception as e:
        logger.warning(
            f"Could not load auth for Colab setup ({e}); Cloudflare link may be blocked."
        )
        return None

    try:
        ensure_default_admin()
        username = DEFAULT_ADMIN_USERNAME
        if not requires_password_change(username):
            creds = _load_colab_login_credentials()
            if creds is not None and _colab_credentials_still_valid(username, creds[1]):
                return creds
            # The admin password was changed through the app after the first run,
            # so the cached copy is stale; drop it instead of printing dead credentials.
            _clear_colab_login_credentials()
            return None
        password = get_bootstrap_password() or generate_bootstrap_password()
        if not update_password(username, password):
            logger.warning(
                "Could not finalize Colab admin password; Cloudflare link may be blocked."
            )
            return None
        _store_colab_login_credentials(username, password)
        return username, password
    except Exception as e:
        logger.warning(
            f"Could not finalize Colab admin password ({e}); Cloudflare link may be blocked."
        )
        return None


def _colab_login_html(username: str, password: str) -> str:
    """Notebook card with Colab admin credentials (shown once after auto-finalize)."""
    return f"""
    <div style="display: inline-block; padding: 20px; background: #ffffff; border: 2px solid #000000;
                border-radius: 12px; margin: 10px 0; font-family: system-ui, -apple-system, sans-serif;">
        <h2 style="color: #000000; margin: 0 0 12px 0; font-size: 22px; font-weight: 800;">
            Unsloth Studio Login (Colab)
        </h2>
        <p style="color: #333333; margin: 0 0 12px 0; font-size: 14px; font-weight: bold;">
            Log in as <code>{username}</code> with this password. This cell is visible only in
            your notebook session.
        </p>
        <p style="color: #333333; margin: 0; font-size: 14px; font-family: monospace; font-weight: bold;">
            Password: <code>{password}</code>
        </p>
    </div>
    """


def _show_colab_login_credentials(username: str, password: str) -> None:
    """Display Colab admin credentials in the notebook output."""
    from IPython.display import HTML, display

    logger.info(f"🔐 Unsloth Studio login — user: {username}")
    display(HTML(_colab_login_html(username, password)))


def _ready_card_html(
    url: str,
    port: int,
    *,
    has_cloudflare_link: bool = False,
    cloudflare_requested: bool = False,
) -> str:
    """Branded ready card for the in-notebook Unsloth view.

    Colab ``*.prod.colab.dev`` proxy URLs are session-scoped and 404 when opened as a
    top-level tab or on another device, so never ``window.open`` them. On real Colab the
    Cloudflare link is the supported entry point because in-cell proxy embeds often stay blank.
    """
    short_url = _short_colab_url(url, port)
    if _is_colab_runtime() or _is_colab_proxy_url(url, port):
        if has_cloudflare_link:
            embed_note = (
                "Open Unsloth with the Cloudflare link above. In-cell proxy previews on "
                "current Colab often stay blank, so the tunnel link is the supported path."
            )
        elif cloudflare_requested:
            embed_note = (
                "Could not open a Cloudflare tunnel, so Unsloth may be unreachable on Colab. "
                "Check the logs above and re-run this cell. Pass "
                '<code style="background:#f3f3f3;padding:2px 6px;border-radius:4px;">'
                "cloudflare=True</code> after fixing any tunnel errors."
            )
        else:
            embed_note = (
                "Colab proxy links cannot be opened in a new tab (they 404 outside this "
                'notebook). Re-run with <code style="background:#f3f3f3;padding:2px 6px;'
                'border-radius:4px;">start(cloudflare=True)</code> for a working link.'
            )
        return f"""
    <div style="display: inline-block; padding: 20px; background: #ffffff; border: 2px solid #000000;
                border-radius: 12px; margin: 10px 0; font-family: system-ui, -apple-system, sans-serif;">
        <h2 style="color: #000000; margin: 0 0 12px 0; font-size: 26px; font-weight: 800;
                   display: flex; align-items: center; gap: 12px;">
            <img src="https://github.com/unslothai/unsloth/raw/main/studio/frontend/public/unsloth-gem.png"
                 height="48" style="display:block;">
            Unsloth Studio is Ready!
        </h2>
        <p style="color: #333333; margin: 0 0 8px 0; font-size: 15px; font-weight: bold;">
            {embed_note}
        </p>
        <p style="color: #666666; margin: 16px 0 0 0; font-size: 13px; font-family: monospace; font-weight: bold;">
            {short_url}
        </p>
    </div>
    """

    return f"""
    <div style="display: inline-block; padding: 20px; background: #ffffff; border: 2px solid #000000;
                border-radius: 12px; margin: 10px 0; font-family: system-ui, -apple-system, sans-serif;">
        <h2 style="color: #000000; margin: 0 0 12px 0; font-size: 26px; font-weight: 800;
                   display: flex; align-items: center; gap: 12px;">
            <img src="https://github.com/unslothai/unsloth/raw/main/studio/frontend/public/unsloth-gem.png"
                 height="48" style="display:block;">
            Unsloth Studio is Ready!
        </h2>
        <a href="{url}" onclick="var w=window.open(this.href,'_blank');if(!w){{return true;}}return false;"
           style="display: inline-flex; align-items: center; gap: 10px; padding: 14px 28px;
                  background: #000000; color: white; text-decoration: none; border-radius: 8px;
                  font-weight: 800; font-size: 16px; cursor: pointer;">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="white"><polygon points="5,3 19,12 5,21"/></svg>
            Open Unsloth Studio
        </a>
        <p style="color: #333333; margin: 12px 0 0 0; font-size: 14px; font-weight: bold;">
            If the link doesn't work, you can scroll down to view the UI generated directly in Colab.
        </p>
        <p style="color: #333333; margin: 16px 0 0 0; font-size: 13px; font-family: monospace; font-weight: bold;">
            {short_url}
        </p>
    </div>
    """


def show_link(
    port: int = 8888,
    *,
    _url: "str | None" = None,
    has_cloudflare_link: bool = False,
    cloudflare_requested: bool = False,
):
    """Display a styled ready card for the UI.

    Colab proxy URLs are informational only (no new-tab open; they 404 outside the cell);
    non-proxy URLs keep a clickable open button. *_url* is an optional pre-fetched proxy
    URL to avoid a second eval_js round-trip.
    """
    from IPython.display import display, HTML

    url = _url if _url is not None else get_colab_url(port)
    logger.info(f"🌐 Unsloth Studio URL: {url}")
    display(
        HTML(
            _ready_card_html(
                url,
                port,
                has_cloudflare_link = has_cloudflare_link,
                cloudflare_requested = cloudflare_requested,
            )
        )
    )


def _warn_colab_cloudflare_missing(*, use_cloudflare: bool, cloudflare_url: "str | None") -> None:
    """Log a prominent warning when Colab expected a tunnel but none was opened."""
    if not use_cloudflare or cloudflare_url or not _is_colab_runtime():
        return
    logger.warning(
        "Colab Cloudflare tunnel unavailable — Unsloth is unlikely to be reachable in this "
        "notebook. Check the logs above for tunnel or auth errors, then re-run start()."
    )


def _bootstrap_password_pending() -> bool:
    """True while the default admin still owes a bootstrap-password change.

    While pending, a public tunnel GET (no Origin) reads as same-origin and gets the
    injected password, so sharing the link would leak admin access. Fails safe to pending.
    """
    try:
        from auth.storage import requires_password_change, DEFAULT_ADMIN_USERNAME
        return bool(requires_password_change(DEFAULT_ADMIN_USERNAME))
    except Exception as e:
        logger.info(f"Could not check admin password state ({e}); refusing tunnel to be safe.")
        return True


def _display_channel_active() -> bool:
    """True only where IPython.display actually renders a card to the operator.

    display() does NOT raise outside a notebook, so a non-raising call is no proof
    the credential was seen: with no InteractiveShell it just prints repr(obj)
    ("<IPython.core.display.HTML object>") and a terminal shell renders only the
    text/plain repr. Treating that as success would publish the shared link after
    rotating to a password nobody ever read. Only an ipykernel-backed shell
    (Jupyter, and Colab whose google.colab._shell.Shell subclasses it) publishes
    display_data out of band on iopub, so require one: the `kernel` attribute is
    set on the shell by ipykernel and, unlike comparing __class__.__name__ to
    "ZMQInteractiveShell", it is also true under Colab's subclass.
    """
    try:
        from IPython import get_ipython
    except Exception:
        return False
    try:
        shell = get_ipython()
    except Exception:
        return False
    if shell is None:
        return False
    if getattr(shell, "kernel", None) is not None:
        return True
    try:
        from ipykernel.zmqshell import ZMQInteractiveShell
    except Exception:
        return False
    return isinstance(shell, ZMQInteractiveShell)


def _auto_generate_colab_admin_password() -> "str | None":
    """Secure a Colab public (Cloudflare) launch that has no admin password set.

    While the admin still owes its bootstrap-password change, a shared link would
    leak admin access, so today the tunnel is refused. Instead auto-generate a
    strong password and commit it via the normal update path (which clears the
    must-change flag, rotates the JWT secret, revokes refresh tokens, and deletes
    the on-disk bootstrap password), then return it for one-time display in the
    cell. Returns None when a password is already set (nothing to do) or on error.
    The value is never persisted to disk or placed on argv.

    The commit is a compare-and-set on ``must_change_password``: another tab can
    finish /change-password against the reused server between the check below and
    the write, and an unconditional update would either discard the password the
    user just chose or display a generated one that has already been replaced,
    publishing the tunnel under credentials nobody can use. Losing that race means
    a password is now set, so we return None exactly as if one always had been.

    Entering this path also purges the pre-#7392 ``.colab_notebook_login`` cache:
    the removed finalize flow wrote the admin username and password there in
    plaintext, and an upgraded runtime must not keep a readable copy of a
    credential this flow promises is never persisted (CWE-256).

    The display channel is resolved BEFORE rotating (mirroring run.py's
    _one_time_secret_stream preflight): a runtime that cannot render the card is
    refused here, while the seeded recovery credential is still intact.
    """
    try:
        from auth.storage import (
            DEFAULT_ADMIN_USERNAME,
            ensure_default_admin,
            requires_password_change,
            update_password,
        )
    except Exception as e:
        logger.warning(f"Could not load auth storage to secure the public link ({e}).")
        return None
    try:
        ensure_default_admin()
        if not requires_password_change(DEFAULT_ADMIN_USERNAME):
            # A password is already set, so there is nothing to rotate. Before
            # dropping the pre-#7392 plaintext cache, hand its credential back
            # ONCE if it still authenticates.
            #
            # An upgrading runtime is the case that matters. The removed finalize
            # flow re-displayed this cache on every cell re-run, so it was a
            # standing recovery surface; purging it silently would delete the only
            # copy a user who cleared their earlier cell output still had, leaving
            # a working password nobody can discover. Showing it once and then
            # removing it keeps the CWE-256 fix and still leaves the user a way in.
            cached = _load_colab_login_credentials()
            if cached is not None and _colab_credentials_still_valid(*cached):
                _display_admin_credentials(*cached, final_cached_copy = True)
            _clear_colab_login_credentials()
            return None
        _clear_colab_login_credentials()
        if not _display_channel_active():
            # Nowhere to render the one-time card, so rotating would destroy the
            # only recovery credential for a password nobody could read. Refuse
            # BEFORE the write; the caller's pending check then blocks the link.
            logger.warning(
                "No notebook display channel to show a one-time admin password, so the "
                "admin password is left unchanged. Set one with `unsloth studio "
                "reset-password` (or log in locally and change it), then re-run start()."
            )
            return None
        import secrets

        generated = secrets.token_urlsafe(24)
        try:
            # update_password returns the rotated JWT secret, or None when a guard
            # rejected the write. Narrow to a bool here: the except branch below
            # assigns one, and this variable must not hold a secret on one path.
            committed = (
                update_password(
                    DEFAULT_ADMIN_USERNAME,
                    generated,
                    revoke_refresh_tokens = True,
                    require_must_change = True,
                )
                is not None
            )
        except Exception as e:
            # update_password commits the row BEFORE its best-effort cleanup
            # (clear_bootstrap_password, which can still raise -- e.g. printing its
            # own warning to a closed stderr). A raise there leaves the new password
            # live, and discarding it would publish
            # the link -- must_change is already 0 -- under a credential nobody
            # holds. Ask the stored hash which password actually won.
            logger.warning(f"Admin password commit reported an error ({e}); checking what landed.")
            committed = _colab_credentials_still_valid(DEFAULT_ADMIN_USERNAME, generated)
        if not committed:
            # Lost the race: a password was set elsewhere, so ours was never
            # written. Never show it -- it would not authenticate.
            logger.info("An admin password was set concurrently; keeping it for the public link.")
            return None
        # Committed but not yet rendered. Until the caller confirms the card
        # reached the notebook, this password exists only in memory; the sentinel
        # keeps a re-run of the cell from publishing under it. Cleared by
        # start_cloudflare_tunnel once _display_admin_credentials succeeds.
        try:
            from auth.storage import mark_credential_undelivered

            mark_credential_undelivered(DEFAULT_ADMIN_USERNAME)
        except Exception:
            pass
        return generated
    except Exception as e:
        logger.warning(f"Could not auto-generate an admin password for the public link ({e}).")
        return None


def _display_admin_credentials(
    username: str,
    password: str,
    *,
    final_cached_copy: bool = False,
) -> bool:
    """Show an admin credential once, in the notebook cell.

    ``final_cached_copy`` re-displays a credential recovered from the pre-#7392
    ``.colab_notebook_login`` cache on an upgraded runtime, immediately before
    that cache is deleted. It is not newly generated, so the card says so and
    tells the user this is the last time it will appear.

    Renders a branded HTML card with a plain-text fallback. Both paths publish
    through the IPython display channel (iopub display_data), NOT sys.stdout, so
    the credential never reaches the server's tee'd session log on disk (see
    run._setup_server_disk_logging) and is never logged; if no display channel is
    available we intentionally show nothing rather than fall back to
    stdout/logging, which would retain the password in the log file.

    The cell is the only surface a notebook has, and a notebook SAVES its cell
    output: Colab autosaves to Drive, and an exported or shared .ipynb carries the
    output with it, so this password lives in the notebook document until the
    output is cleared or the password is changed. The card says exactly that
    instead of promising the value is never written to disk -- readers who get the
    notebook must be assumed to have the credential.

    Returns True only when the credential was published to the display channel, and
    False when no channel is available or every publish raised, so the caller can
    fail closed rather than expose a shared link under a password nobody ever saw.
    """
    try:
        from IPython.display import HTML, display
    except Exception:
        return False
    _lede = (
        "This is your existing admin password, recovered from the credential file an "
        "older Unsloth cached on this runtime. That file has now been removed, so this "
        "is the last time it will be shown."
        if final_cached_copy
        else "Auto-generated for this public launch."
    )
    try:
        display(
            HTML(f"""
    <div style="display: inline-block; padding: 18px 20px; background: #fff8e1; border: 2px solid #000000;
                border-radius: 12px; margin: 10px 0; font-family: system-ui, -apple-system, sans-serif;">
        <h3 style="color:#000;margin:0 0 8px 0;font-size:18px;font-weight:800;">Unsloth Studio admin login</h3>
        <p style="margin:2px 0;font-size:14px;color:#000;">Username: <b style="font-family:monospace;">{username}</b></p>
        <p style="margin:2px 0;font-size:14px;color:#000;">Password: <b style="font-family:monospace;">{password}</b></p>
        <p style="margin:10px 0 0 0;font-size:12px;color:#333;">
            {_lede} The server never writes it to its logs, but
            this notebook saves cell output: copy the password now, then clear this cell's output
            before you share, export or download the notebook. Change it any time in Settings, or
            with <b style="font-family:monospace;">unsloth studio reset-password</b>.
        </p>
    </div>
    """)
        )
        return True
    except Exception:
        # HTML render failed; show a plain-text copy through the SAME display
        # channel (still iopub, never sys.stdout, so still out of the server log).
        try:
            display(
                {
                    "text/plain": (
                        "Unsloth Studio admin login  "
                        f"username: {username}  password: {password}  "
                        f"({_lede} this cell's output is saved "
                        "with the notebook, so clear it before sharing or exporting)"
                    )
                },
                raw = True,
            )
            return True
        except Exception:
            return False


def _mint_same_tab_link_token() -> "str | None":
    """Opt-in: mint a ONE-TIME link token for the SAME-TAB Colab proxy URL only.

    Never call this for the shared Cloudflare link. Returns None on any failure,
    in which case the same-tab URL simply carries no token and the login page
    still works. The returned token is a bearer credential: it is placed ONLY on
    the private same-tab URL and is never logged.
    """
    try:
        from auth.storage import DEFAULT_ADMIN_USERNAME, ensure_default_admin
        from auth.authentication import create_link_token

        ensure_default_admin()
        return create_link_token(DEFAULT_ADMIN_USERNAME)
    except Exception as e:
        logger.info(f"Could not mint a same-tab link token ({e}); showing the plain URL.")
        return None


def _append_link_token(url: str, token: "str | None") -> str:
    """Append ``?link_token=...`` to the same-tab URL. No-op when token is None."""
    if not token:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}link_token={token}"


def _link_token_opt_in(explicit: bool) -> bool:
    """Whether to mint a same-tab link token: the explicit arg OR the env flag."""
    if explicit:
        return True
    return os.environ.get("UNSLOTH_STUDIO_COLAB_LINK_TOKEN", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


# Literal on purpose: the strip below must run even if importing auth fails, so it
# cannot depend on auth.terminal_prompt.SUPPLIED_PASSWORD_ENV being importable. A
# test asserts the two stay equal.
_SUPPLIED_PASSWORD_ENV = "UNSLOTH_STUDIO_PASSWORD"


def _consume_supplied_password_on_reuse() -> None:
    """Apply-or-strip ``UNSLOTH_STUDIO_PASSWORD`` on start()'s server-reuse path.

    A normal launch goes through ``run_server`` -> ``run._apply_supplied_password``,
    which reads the variable and then unconditionally pops it, so no child process
    inherits the plaintext. The fast path (a re-run cell reusing an already-healthy
    server) never calls ``run_server``, so without this the variable is still set
    when ``start_cloudflare_tunnel`` spawns cloudflared, and every process spawned
    from the kernel afterwards inherits the admin password through its environment
    (CWE-214/CWE-526: readable via ``/proc/<pid>/environ`` and crash dumps for
    anything running as this user, including Studio's own code-execution tools).

    So the variable is popped FIRST and unconditionally, even when the auth imports
    or the update fail. If a value was supplied and the admin still owes its
    bootstrap change, it is then applied through the same compare-and-set update the
    normal path uses, so the password the user asked for is the one that protects
    the shared link instead of being silently replaced by an auto-generated one.
    An already-set password is never overwritten (that is `reset-password`'s job),
    and a value that fails the length/whitespace rules is ignored, leaving
    ``start_cloudflare_tunnel`` to auto-generate and display one. Never logs the
    value, and never exits the process: this runs inside a notebook cell.
    """
    supplied = os.environ.pop(_SUPPLIED_PASSWORD_ENV, None) or None
    if not supplied:
        return
    try:
        from auth.storage import (
            DEFAULT_ADMIN_USERNAME,
            MIN_PASSWORD_LENGTH,
            ensure_default_admin,
            requires_password_change,
            update_password,
        )

        ensure_default_admin()
        if not requires_password_change(DEFAULT_ADMIN_USERNAME):
            logger.info(
                f"An admin password is already set, so {_SUPPLIED_PASSWORD_ENV} was "
                "ignored (it only sets the initial password). Change it with "
                "`unsloth studio reset-password`."
            )
            return
        if len(supplied) < MIN_PASSWORD_LENGTH or any(ch.isspace() for ch in supplied):
            logger.warning(
                f"Ignoring {_SUPPLIED_PASSWORD_ENV}: a password must be at least "
                f"{MIN_PASSWORD_LENGTH} characters and contain no spaces. A strong "
                "one will be generated and shown in this cell instead."
            )
            return
        if update_password(
            DEFAULT_ADMIN_USERNAME,
            supplied,
            revoke_refresh_tokens = True,
            require_must_change = True,
        ):
            logger.info(f"Applied the admin password supplied in {_SUPPLIED_PASSWORD_ENV}.")
        else:
            # Lost the compare-and-set: a password was set elsewhere between the
            # check and the write, so keep theirs (same rule as auto-generation).
            logger.info("An admin password was set concurrently; keeping it.")
    except Exception as e:
        # Never log the value itself. Falling through leaves the bootstrap password
        # pending, so the tunnel path still auto-generates or refuses.
        logger.warning(f"Could not apply the supplied admin password ({e}).")


def start_cloudflare_tunnel(port: int) -> "str | None":
    """Open a shareable Cloudflare quick tunnel to localhost:*port*, or None.

    run_server suppresses the tunnel on Colab by design, so we start it directly.
    When no admin password is set, one is auto-generated and shown in the cell so
    the shareable link is never published under the default bootstrap credential;
    any failure collapses to None and the Colab proxy still works. As a backstop it
    is still refused while the bootstrap password is pending.
    """
    from auth.storage import (
        DEFAULT_ADMIN_USERNAME,
        clear_credential_undelivered,
        credential_undelivered,
    )

    if credential_undelivered(DEFAULT_ADMIN_USERNAME):
        # An earlier run of this cell rotated the password and could not render
        # the card. must_change is 0 now, so nothing below would object, and the
        # link would go up for an account whose password was never shown.
        logger.warning(
            "Cloudflare link not started: the admin password generated by an earlier "
            "run was committed but never shown, so the shared link would be unusable. "
            "Reset it with `unsloth studio reset-password`, then re-run "
            "start(cloudflare=True)."
        )
        return None
    generated = _auto_generate_colab_admin_password()
    if generated is not None and not _display_admin_credentials(DEFAULT_ADMIN_USERNAME, generated):
        # The password was rotated and committed, but it could not be surfaced in
        # this notebook (no IPython display channel, or every publish raised). The
        # shared link would then be live under a password nobody saw, locking every
        # user out. Fail closed: do NOT publish the tunnel, and never fall back to
        # stdout/logging (which the server tees to an on-disk log). Tell the operator
        # to reset the credential.
        logger.warning(
            "Cloudflare link not started: the auto-generated admin password could not "
            "be shown in this notebook, so the shared link would be unusable. Reset it "
            "with `unsloth studio reset-password`, then re-run start(cloudflare=True)."
        )
        return None
    if generated is not None:
        # The card rendered, so the operator has the credential.
        clear_credential_undelivered()
    if _bootstrap_password_pending():
        # Auto-generation is the primary protection; only reached if it failed
        # (e.g. the auth DB could not be read/written). Fail safe: no shared link.
        logger.warning(
            "Cloudflare link not started: the admin account still has its temporary "
            "bootstrap password, which is exposed to anyone who can load the page. "
            "Open Unsloth in this tab, log in and change the admin password, then re-run "
            "start(cloudflare=True) to get the shareable link."
        )
        return None
    try:
        from cloudflare_tunnel import set_studio_tunnel_url_callback, start_studio_tunnel
    except Exception as e:
        logger.info(f"Cloudflare tunnel unavailable ({e}); using Colab proxy only.")
        return None
    try:
        set_studio_tunnel_url_callback(_publish_cloudflare_url)
        url = start_studio_tunnel(port, managed_by = "colab")
    except Exception as e:
        logger.info(f"Cloudflare tunnel failed to start ({e}); using Colab proxy only.")
        return None
    # Success is logged by _show_and_embed; note only misses here.
    if not url:
        logger.info("Cloudflare tunnel did not produce a URL; using Colab proxy only.")
    return url


def _publish_cloudflare_url(cloudflare_url: "str | None") -> None:
    try:
        from main import app as _studio_app
        _studio_app.state.cloudflare_url = cloudflare_url
    except Exception as e:
        logger.info(f"Could not publish Cloudflare URL to /api/health ({e}).")


def _stop_cloudflare_tunnel() -> None:
    """Best-effort teardown of the Cloudflare tunnel started by start_cloudflare_tunnel."""
    try:
        from cloudflare_tunnel import stop_studio_tunnel
        stop_studio_tunnel()
    except Exception:
        pass


def _is_studio_healthy(port: int, timeout: float = 2.0) -> bool:
    """True only if Unsloth Studio (not some other app) answers /api/health on *port*.

    The service-marker check stops the reuse path reusing or tunneling a foreign process.
    """
    import json, urllib.request
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/api/health", timeout = timeout) as r:
            return json.loads(r.read()).get("service") == "Unsloth UI Backend"
    except Exception:
        return False


def _shareable_link_html(
    cloudflare_url: str,
    password: "str | None" = None,
    username: "str | None" = None,
) -> str:
    """Branded card for the shareable Cloudflare link, styled like the show_link banner.

    *password* renders under the link so the credential sits in the card with the button
    it unlocks. The username is always the default admin, so it reads inline.
    """
    login_block = ""
    if password:
        login_block = f"""
        <p style="color: #000000; margin: 16px 0 0 0; font-size: 20px; font-weight: 800;">
            Password
        </p>
        <p style="margin: 6px 0 0 0;"><code style="display: inline-block; font-size: 24px;
            font-weight: 800; text-decoration: underline; background: #f3f3f3;
            padding: 4px 10px; border-radius: 6px;">{password}</code></p>
        <p style="color: #666666; margin: 6px 0 0 0; font-size: 12px;">
            Log in as <code>{username}</code> with this password. Shown only in your
            notebook session, and never included in the shared link.
        </p>"""
    return f"""
    <div style="display: inline-block; padding: 20px; background: #ffffff; border: 2px solid #000000;
                border-radius: 12px; margin: 10px 0; font-family: system-ui, -apple-system, sans-serif;">
        <h2 style="color: #000000; margin: 0 0 12px 0; font-size: 26px; font-weight: 800;
                   display: flex; align-items: center; gap: 12px;">
            <img src="https://github.com/unslothai/unsloth/raw/main/studio/frontend/public/unsloth-gem.png"
                 height="48" style="display:block;">
            Shareable Unsloth Link is Ready!
        </h2>
        <a href="{cloudflare_url}" onclick="var w=window.open(this.href,'_blank');if(!w){{return true;}}return false;"
           style="display: inline-flex; align-items: center; gap: 10px; padding: 14px 28px;
                  background: #000000; color: white; text-decoration: none; border-radius: 8px;
                  font-weight: 800; font-size: 16px; cursor: pointer;">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="white"><polygon points="5,3 19,12 5,21"/></svg>
            Open Unsloth Studio
        </a>
        <p style="color: #333333; margin: 12px 0 0 0; font-size: 14px; font-weight: bold;">
            This Cloudflare HTTPS link works from any device, so you can share it with anyone.
        </p>
        <p style="color: #333333; margin: 16px 0 0 0; font-size: 13px; font-family: monospace; font-weight: bold;">
            🔗 <a href="{cloudflare_url}" onclick="var w=window.open(this.href,'_blank');if(!w){{return true;}}return false;"
                  style="color: #000000; text-decoration: underline; cursor: pointer;">{cloudflare_url}</a>
        </p>{login_block}
    </div>
    """


# Height for serve_kernel_port_as_iframe (~82vh on a 1080p screen, clamped).
_COLAB_IFRAME_HEIGHT = 900


def _embed_kernel_port_iframe(port: int) -> bool:
    """Embed Unsloth via Colab's native kernel-port iframe helper.

    Only trusted on a real Colab runtime: colabtools can import ``google.colab`` and
    queue browser-side JS without appending an iframe, so callers outside Colab must use
    the HTML iframe path instead.
    """
    if not _is_colab_runtime():
        return False
    try:
        from google.colab import output as colab_output
    except ImportError:
        return False
    try:
        colab_output.serve_kernel_port_as_iframe(
            port,
            height = _COLAB_IFRAME_HEIGHT,
            width = "100%",
        )
        return True
    except Exception as e:
        logger.info(f"serve_kernel_port_as_iframe failed ({e}); trying HTML iframe.")
        return False


def _embed_html_iframe(
    url: str,
    port: int,
    *,
    same_tab_url: "str | None" = None,
) -> bool:
    """Fallback embed: raw HTML iframe when the Colab helper is unavailable.

    *same_tab_url* (opt-in) is the token-bearing same-tab proxy URL used ONLY as the
    iframe ``src``; it may carry a one-time ``?link_token=...`` bearer credential and is
    never logged. The visible header still shows the truncated, token-free *url*, and
    when *same_tab_url* is None the plain *url* is used as the src.
    """
    try:
        from IPython.display import HTML, display
    except ImportError:
        return False

    # Header shows the truncated, token-free URL; the token (if any) rides only on the
    # iframe src via *same_tab_url*, never in the visible header or logs.
    short_url = _short_colab_url(url, port)
    iframe_src = same_tab_url or url
    iframe_id = f"unsloth-studio-{port}"
    try:
        display(
            HTML(f"""
<div style="font-family:system-ui,-apple-system,sans-serif;margin:8px 0;
            border-radius:12px;overflow:hidden;box-shadow:0 2px 16px rgba(0,0,0,0.18);">
  <div style="display:flex;align-items:center;gap:10px;padding:10px 16px;background:#000;">
    <img src="https://github.com/unslothai/unsloth/raw/main/studio/frontend/public/unsloth-gem.png"
         height="26" style="display:block;">
    <span style="color:#fff;font-weight:700;font-size:15px;letter-spacing:-0.2px;">Unsloth Studio</span>
    <span style="margin-left:auto;color:#666;font-size:11px;font-family:monospace;">{short_url}</span>
  </div>
  <iframe
    id="{iframe_id}"
    src="{iframe_src}"
    style="width:100%;height:82vh;min-height:600px;max-height:1100px;border:none;display:block;box-sizing:border-box;"
    allow="clipboard-read; clipboard-write"
  ></iframe>
</div>
""")
        )
        return True
    except Exception as e:
        logger.info(f"HTML iframe embed failed ({e}).")
        return False


def _show_and_embed(
    port: int,
    *,
    cloudflare_url: "str | None" = None,
    colab_login: "tuple[str, str] | None" = None,
    cloudflare_requested: bool = False,
    same_tab_link_token: "str | None" = None,
):
    """Render the Unsloth ready card + iframe for *port*.

    Prefer Colab's ``serve_kernel_port_as_iframe`` on real Colab; raw HTML iframe is the
    fallback. Cloudflare cards stay clickable.

    *same_tab_link_token* (opt-in) is appended as ``?link_token=...`` to the SAME-TAB
    proxy URL ONLY (never the shared *cloudflare_url*) and rides solely on the HTML
    iframe ``src``. It is a bearer credential, so it is never logged and never placed on
    the visible header. The kernel-port helper takes only the port and so cannot carry it.

    TODO(frontend): the built UI does not yet read ``?link_token`` from the URL, POST it
    to ``/api/auth/link-exchange``, store the returned JWT, and scrub the query with
    ``history.replaceState``. Until it does, the token is emitted but unused; the same-tab
    login page still works normally. Wire the frontend exchange + scrub to complete the
    one-time auto-login handoff.
    """
    url = get_colab_url(port)
    # Log the token-free URL; the token must never be written to logs.
    logger.info(f"🌐 Unsloth Studio URL: {url}")
    if cloudflare_url:
        logger.info(f"🔗 Shareable Cloudflare link: {cloudflare_url}")

    # Same-tab URL may carry the one-time token; the shared link never does.
    same_tab_url = _append_link_token(url, same_tab_link_token)

    _warn_colab_cloudflare_missing(
        use_cloudflare = cloudflare_requested,
        cloudflare_url = cloudflare_url,
    )

    # Fold the credentials into the link card rather than a second card below it.
    credentials_shown = False
    if cloudflare_url:
        try:
            from IPython.display import HTML, display

            username, password = colab_login if colab_login else (None, None)
            display(HTML(_shareable_link_html(cloudflare_url, password, username)))
            credentials_shown = bool(colab_login)
        except Exception as e:
            logger.info(f"Could not render Cloudflare link card ({e}).")

    if colab_login and not credentials_shown:
        try:
            _show_colab_login_credentials(*colab_login)
        except Exception as e:
            logger.info(f"Could not render Colab login card ({e}).")

    # With a tunnel up the embed below is skipped, so the ready card would only restate
    # the link card and print a proxy URL that 404s outside this tab.
    skip_ready_card = _is_colab_runtime() and bool(cloudflare_url)
    if not skip_ready_card:
        try:
            show_link(
                port,
                _url = url,
                has_cloudflare_link = bool(cloudflare_url),
                cloudflare_requested = cloudflare_requested,
            )
        except Exception as e:
            logger.info(f"Could not render Unsloth link card ({e}).")

    # On Colab with a working tunnel, skip the in-cell proxy embed (often blank).
    if _is_colab_runtime() and cloudflare_url:
        return

    # Real Colab: kernel helper needs only the port (works when eval_js failed). It
    # cannot carry a query token, so the opt-in same-tab link token applies only to the
    # HTML iframe fallback below.
    if _is_colab_runtime():
        if _embed_kernel_port_iframe(port):
            return
    _embed_html_iframe(url, port, same_tab_url = same_tab_url)


def start(
    port: int = 8888,
    *,
    cloudflare: "bool | None" = None,
    link_token: bool = False,
):
    """Start Unsloth Studio in Colab and display the URL.

    Args:
        port: Port to bind/serve on.
        cloudflare: Shareable Cloudflare HTTPS link. ``None`` (default) auto-enables on
            real Colab because the in-cell proxy embed is often blank; pass ``False`` to
            skip the tunnel or ``True`` to force it on other runtimes. The shared link is
            protected: when the admin still owes its bootstrap password one is
            auto-generated and shown in the cell, and the tunnel fails closed if that
            credential cannot be surfaced, so the link is never published under the
            default credential. The cell output is saved with the notebook, so clear
            it before sharing or exporting (the card says so too).
        link_token: Opt in (default OFF; also enabled by
            ``UNSLOTH_STUDIO_COLAB_LINK_TOKEN=1``) to append a ONE-TIME, short-TTL
            link token to the SAME-TAB proxy URL for a one-click login handoff. The
            token is never added to the shared Cloudflare link. See the frontend
            TODO in ``_show_and_embed``: the token is emitted but the UI does not
            consume it yet, so today it is a no-op the login page ignores.

    Usage:
        start()                    # Cloudflare link on Colab (auto); proxy iframe elsewhere
        start(cloudflare=False)    # Colab proxy iframe only (often blank on current Colab)
        start(cloudflare=True)     # force Cloudflare link on any runtime
        start(link_token=True)     # same-tab URL carries a one-time link token
    """
    import time

    logger.info("🦥 Starting Unsloth Studio...")
    use_cloudflare = _colab_wants_cloudflare(cloudflare)
    want_link_token = _link_token_opt_in(link_token)

    # Fast path: already running (cell re-run); re-show link/iframe instead of rebinding the port.
    if _is_studio_healthy(port):
        logger.info(f"   Unsloth is already running on port {port} — reusing existing server.")
        # run_server (and with it run._apply_supplied_password) is skipped here, so
        # apply-or-strip UNSLOTH_STUDIO_PASSWORD ourselves BEFORE spawning
        # cloudflared: otherwise the supplied password is ignored while the
        # bootstrap one is still pending, and the plaintext stays in os.environ for
        # every child process to inherit.
        _consume_supplied_password_on_reuse()
        # try/finally: tear the tunnel down even if interrupted mid-start/render.
        try:
            # start_cloudflare_tunnel owns the shared-link credential: it auto-generates
            # and shows a strong admin password (fail-closed if it cannot be surfaced) and
            # never persists it, so we do NOT finalize/cache the bootstrap password here.
            cf_url = start_cloudflare_tunnel(port) if use_cloudflare else None
            _show_and_embed(
                port,
                cloudflare_url = cf_url,
                cloudflare_requested = use_cloudflare,
                same_tab_link_token = _mint_same_tab_link_token() if want_link_token else None,
            )
            for _ in range(10000):
                time.sleep(300)
                print("=", end = "", flush = True)
        except KeyboardInterrupt:
            logger.info("\nUnsloth Studio keepalive stopped.")
        finally:
            _stop_cloudflare_tunnel()
        return

    logger.info("   Loading backend...")
    from run import run_server

    repo_root = Path(__file__).parent.parent
    frontend_path = repo_root / "frontend" / "dist"

    if not (frontend_path / "index.html").exists():
        logger.info("❌ Frontend not built! Please run the setup cell first.")
        return

    logger.info("   Starting server...")
    try:
        # cloudflare=False: this helper owns the tunnel (via start(cloudflare=...)), so pin it off.
        app = run_server(
            host = "0.0.0.0",
            port = port,
            frontend_path = frontend_path,
            silent = True,
            cloudflare = False,
        )
    except SystemExit as exc:
        logger.error(f"❌ Unsloth Studio failed to start: {exc}")
        return
    except Exception as exc:
        logger.error(f"❌ Unsloth Studio failed to start: {exc}")
        return

    # run_server may auto-increment the port; read back the bound port for the proxy URL/iframe.
    actual_port: int = getattr(getattr(app, "state", None), "server_port", None) or port

    logger.info(f"   Server started on port {actual_port}!")

    # Poll health before showing the link: avoids the race where ready_event fires pre-bind.
    import urllib.request

    server_ready = False
    for _ in range(40):
        try:
            with urllib.request.urlopen(f"http://localhost:{actual_port}/api/health", timeout = 1):
                server_ready = True
                break
        except Exception:
            time.sleep(0.5)

    if not server_ready:
        logger.error(
            f"❌ Unsloth Studio did not become healthy on port {actual_port}. "
            "Check for errors above."
        )
        return

    # Server healthy: secure Colab auth, open the tunnel, publish URL, tear down on interrupt.
    try:
        # start_cloudflare_tunnel owns the shared-link credential: it auto-generates and
        # shows a strong admin password (fail-closed if it cannot be surfaced) and never
        # persists it, so we do NOT finalize/cache the bootstrap password here.
        cf_url = start_cloudflare_tunnel(actual_port) if use_cloudflare else None
        _show_and_embed(
            actual_port,
            cloudflare_url = cf_url,
            cloudflare_requested = use_cloudflare,
            same_tab_link_token = _mint_same_tab_link_token() if want_link_token else None,
        )

        # Keep kernel alive so the daemon server thread runs.
        for _ in range(10000):
            time.sleep(300)
            print("=", end = "", flush = True)
    except KeyboardInterrupt:
        logger.info("\nUnsloth Studio keepalive stopped.")
    finally:
        _stop_cloudflare_tunnel()


if __name__ == "__main__":
    start()
