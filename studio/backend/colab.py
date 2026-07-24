# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Colab helpers for Unsloth Studio. Uses Colab's built-in proxy.
"""

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
    """
    Get the Colab proxy URL for a port.

    Retries up to 3 times, validating the result is a real HTTPS Colab URL.
    Falls back to http://localhost:{port} only when all attempts fail.
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


def show_link(port: int = 8888, *, _url: "str | None" = None):
    """Display a styled clickable link to the UI.

    *_url* is an optional pre-fetched proxy URL; pass it to avoid a second eval_js round-trip.
    """
    from IPython.display import display, HTML

    url = _url if _url is not None else get_colab_url(port)

    # Truncated display URL; try/except so an odd URL shape still renders the link.
    try:
        port_prefix = f"{port}-"
        idx = url.index(port_prefix)
        next_dash = url.index("-", idx + len(port_prefix))
        short_url = url[: next_dash + 1] + "..."
    except (ValueError, IndexError):
        short_url = url

    # Plain-text line so the URL shows even if HTML display fails.
    logger.info(f"🌐 Unsloth Studio URL: {url}")

    html = f"""
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
    display(HTML(html))


def _bootstrap_password_pending() -> bool:
    """True while the default admin still owes a bootstrap-password change.

    While pending, main.py injects that password into same-origin GETs, and a public
    tunnel GET (no Origin) reads as same-origin, so sharing the link would leak admin
    access. Fails safe to pending if the state cannot be read.
    """
    try:
        from auth.storage import requires_password_change, DEFAULT_ADMIN_USERNAME
        return bool(requires_password_change(DEFAULT_ADMIN_USERNAME))
    except Exception as e:
        logger.info(f"Could not check admin password state ({e}); refusing tunnel to be safe.")
        return True


def _auto_generate_colab_admin_password() -> "str | None":
    """Secure a Colab public (Cloudflare) launch that has no admin password set.

    While the admin still owes its bootstrap-password change, a shared link would
    leak admin access, so today the tunnel is refused. Instead auto-generate a
    strong password and commit it via the normal update path (which clears the
    must-change flag, rotates the JWT secret, revokes refresh tokens, and deletes
    the on-disk bootstrap password), then return it for one-time display in the
    cell. Returns None when a password is already set (nothing to do) or on error.
    The value is never persisted to disk or placed on argv.
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
            return None
        import secrets

        generated = secrets.token_urlsafe(24)
        update_password(DEFAULT_ADMIN_USERNAME, generated, revoke_refresh_tokens = True)
        return generated
    except Exception as e:
        logger.warning(f"Could not auto-generate an admin password for the public link ({e}).")
        return None


def _display_admin_credentials(username: str, password: str) -> None:
    """Show an auto-generated admin credential once, in the notebook cell.

    Renders a branded HTML card with a plain-text fallback. Both paths publish
    through the IPython display channel (iopub display_data), NOT sys.stdout, so
    the credential is never written to the server's tee'd session log on disk
    (see run._setup_server_disk_logging). It is never logged or persisted; if no
    display channel is available we intentionally show nothing rather than fall
    back to stdout/logging, which would retain the password in the log file.
    """
    try:
        from IPython.display import HTML, display
    except Exception:
        return
    try:
        display(
            HTML(f"""
    <div style="display: inline-block; padding: 18px 20px; background: #fff8e1; border: 2px solid #000000;
                border-radius: 12px; margin: 10px 0; font-family: system-ui, -apple-system, sans-serif;">
        <h3 style="color:#000;margin:0 0 8px 0;font-size:18px;font-weight:800;">Unsloth Studio admin login</h3>
        <p style="margin:2px 0;font-size:14px;color:#000;">Username: <b style="font-family:monospace;">{username}</b></p>
        <p style="margin:2px 0;font-size:14px;color:#000;">Password: <b style="font-family:monospace;">{password}</b></p>
        <p style="margin:10px 0 0 0;font-size:12px;color:#333;">
            Auto-generated for this public launch and shown once. Save it now; it is not written to disk.
        </p>
    </div>
    """)
        )
    except Exception:
        # HTML render failed; show a plain-text copy through the SAME display
        # channel (still iopub, never sys.stdout, so still not written to disk).
        try:
            display(
                {
                    "text/plain": (
                        "Unsloth Studio admin login  "
                        f"username: {username}  password: {password}  "
                        "(auto-generated for this public launch, shown once, not saved to disk)"
                    )
                },
                raw = True,
            )
        except Exception:
            pass


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


def start_cloudflare_tunnel(port: int) -> "str | None":
    """Open a shareable Cloudflare quick tunnel to localhost:*port*, or None.

    run_server suppresses the tunnel on Colab by design, so we start it directly.
    When no admin password is set, one is auto-generated and shown in the cell so
    the shareable link is never published under the default bootstrap credential;
    any failure collapses to None and the Colab proxy still works.
    """
    from auth.storage import DEFAULT_ADMIN_USERNAME

    generated = _auto_generate_colab_admin_password()
    if generated is not None:
        _display_admin_credentials(DEFAULT_ADMIN_USERNAME, generated)
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
        from cloudflare_tunnel import start_studio_tunnel
    except Exception as e:
        logger.info(f"Cloudflare tunnel unavailable ({e}); using Colab proxy only.")
        return None
    try:
        url = start_studio_tunnel(port)
    except Exception as e:
        logger.info(f"Cloudflare tunnel failed to start ({e}); using Colab proxy only.")
        return None
    # Success is logged by _show_and_embed; note only misses here.
    if not url:
        logger.info("Cloudflare tunnel did not produce a URL; using Colab proxy only.")
    return url


def _publish_cloudflare_url(cloudflare_url: "str | None") -> None:
    """Publish a directly-started tunnel URL onto app.state so /api/health advertises it.

    run_server only sets this when it opens the tunnel itself, which it skips on Colab,
    so we set it here. Otherwise the frontend's API examples fall back to an
    unreachable server_url. Best-effort.
    """
    if not cloudflare_url:
        return
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
    # Stop /api/health advertising a dead tunnel.
    try:
        from main import app as _studio_app
        _studio_app.state.cloudflare_url = None
    except Exception:
        pass


def _is_studio_healthy(port: int, timeout: float = 2.0) -> bool:
    """True only if Unsloth Studio (not some other app) answers /api/health on *port*.

    The service-marker check stops the reuse path reusing or tunneling a foreign
    process that merely serves /api/health.
    """
    import json, urllib.request
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/api/health", timeout = timeout) as r:
            return json.loads(r.read()).get("service") == "Unsloth UI Backend"
    except Exception:
        return False


def _shareable_link_html(cloudflare_url: str) -> str:
    """Branded card for the shareable Cloudflare link, styled like the show_link banner."""
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
            This Cloudflare HTTPS link works from any device — share it with anyone. The Colab view below only works in this tab.
        </p>
        <p style="color: #333333; margin: 16px 0 0 0; font-size: 13px; font-family: monospace; font-weight: bold;">
            🔗 {cloudflare_url}
        </p>
    </div>
    """


def _show_and_embed(
    port: int,
    *,
    cloudflare_url: "str | None" = None,
    same_tab_link_token: "str | None" = None,
):
    """Render the Unsloth header + iframe for *port*, with a shareable-link card above
    when *cloudflare_url* is set. Falls back to serve_kernel_port_as_iframe.

    *same_tab_link_token* (opt-in) is appended as ``?link_token=...`` to the
    same-tab proxy URL ONLY (never the shared *cloudflare_url*). It is a bearer
    credential, so the token-bearing URL is used solely as the iframe/link src and
    is never logged.

    TODO(frontend): the built UI does not yet read ``?link_token`` from the URL,
    POST it to ``/api/auth/link-exchange``, store the returned JWT, and scrub the
    query with ``history.replaceState``. Until it does, the token is emitted but
    unused; the same-tab login page still works normally. Wire the frontend
    exchange + scrub to complete the one-time auto-login handoff.
    """
    url = get_colab_url(port)
    # Log the token-free URL; the token must never be written to logs.
    logger.info(f"🌐 Unsloth Studio URL: {url}")
    if cloudflare_url:
        logger.info(f"🔗 Shareable Cloudflare link: {cloudflare_url}")

    # Same-tab URL may carry the one-time token; the shared link never does.
    same_tab_url = _append_link_token(url, same_tab_link_token)

    try:
        from IPython.display import HTML, display

        iframe_id = f"unsloth-studio-{port}"

        # Truncated header URL — best-effort, falls back to full URL. Built from the
        # token-free URL so no token leaks into the visible header.
        try:
            port_prefix = f"{port}-"
            idx = url.index(port_prefix)
            next_dash = url.index("-", idx + len(port_prefix))
            short_url = url[: next_dash + 1] + "..."
        except (ValueError, IndexError):
            short_url = url

        if cloudflare_url:
            display(HTML(_shareable_link_html(cloudflare_url)))

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
    src="{same_tab_url}"
    style="width:100%;height:82vh;min-height:600px;max-height:1100px;border:none;display:block;box-sizing:border-box;"
    allow="clipboard-read; clipboard-write"
  ></iframe>
</div>
""")
        )
    except Exception:
        # Fallback: Colab's built-in helper.
        try:
            from google.colab import output as colab_output
            colab_output.serve_kernel_port_as_iframe(port, height = 900, width = "100%")
        except ImportError:
            pass


def start(
    port: int = 8888,
    *,
    cloudflare: bool = False,
    link_token: bool = False,
):
    """Start Unsloth Studio in Colab and display the URL.

    Args:
        port: Port to bind/serve on.
        cloudflare: Opt in to a shareable Cloudflare HTTPS link reachable from any
            device (default OFF). It exposes Unsloth's login page beyond Colab, so it
            stays an explicit opt-in; the default shows only the in-tab proxy iframe.
        link_token: Opt in (default OFF; also enabled by
            ``UNSLOTH_STUDIO_COLAB_LINK_TOKEN=1``) to append a ONE-TIME, short-TTL
            link token to the SAME-TAB proxy URL for a one-click login handoff. The
            token is never added to the shared Cloudflare link. See the frontend
            TODO in ``_show_and_embed``: the token is emitted but the UI does not
            consume it yet, so today it is a no-op the login page ignores.

    Usage:
        start()                    # Colab-proxy iframe only (default)
        start(cloudflare=True)     # also open a shareable Cloudflare link
        start(link_token=True)     # same-tab URL carries a one-time link token
    """
    import time

    logger.info("🦥 Starting Unsloth Studio...")
    want_link_token = _link_token_opt_in(link_token)

    # Fast path: Unsloth already running (cell re-run). Re-launching would collide on
    # the port, so just re-show the link and iframe.
    if _is_studio_healthy(port):
        logger.info(f"   Unsloth is already running on port {port} — reusing existing server.")
        # try/finally: tear the tunnel down even if interrupted mid-start/render.
        try:
            cf_url = start_cloudflare_tunnel(port) if cloudflare else None
            _publish_cloudflare_url(cf_url)
            _show_and_embed(
                port,
                cloudflare_url = cf_url,
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

    # Auto-detect frontend path
    repo_root = Path(__file__).parent.parent
    frontend_path = repo_root / "frontend" / "dist"

    if not (frontend_path / "index.html").exists():
        logger.info("❌ Frontend not built! Please run the setup cell first.")
        return

    logger.info("   Starting server...")
    try:
        # cloudflare=False: this helper owns the tunnel (Colab's own
        # start(cloudflare=...) drives it), so pin it off explicitly.
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

    # run_server auto-increments the port if in use; read back the bound port so the
    # proxy URL and iframe point at the right place.
    actual_port: int = getattr(getattr(app, "state", None), "server_port", None) or port

    logger.info(f"   Server started on port {actual_port}!")

    # Poll health endpoint before showing the link — avoids the race where ready_event
    # fires but the process hasn't finished binding.
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

    # Open the tunnel now the server is healthy, publish its URL for /api/health, and
    # tear it down on interrupt (try/finally) rather than orphan the process.
    try:
        cf_url = start_cloudflare_tunnel(actual_port) if cloudflare else None
        _publish_cloudflare_url(cf_url)
        _show_and_embed(
            actual_port,
            cloudflare_url = cf_url,
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
