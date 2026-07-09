# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Notebook helpers for Unsloth Studio.

Colab uses its built-in proxy when available. Kaggle-backed Colab sessions do
not expose that proxy to the browser, so they need a tunnel URL for the inline
Studio frame.
"""

from pathlib import Path
import os
import secrets
import sys

# Seed platform._sys_version_cache before attrs->rich->structlog->platform crash on conda Python.
# See: https://github.com/python/cpython/issues/102396
_backend_dir = str(Path(__file__).parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
import _platform_compat  # noqa: F401


from loggers import get_logger
from utils.notebook_env import is_kaggle_environment
from utils.notebook_frame_auth import (
    NOTEBOOK_FRAME_TOKEN_ENV as _NOTEBOOK_FRAME_TOKEN_ENV,
    NOTEBOOK_FRAME_TOKEN_PARAM as _NOTEBOOK_FRAME_TOKEN_PARAM,
)

logger = get_logger(__name__)

_PUBLIC_URL_WAIT_TIMEOUT_ENV = "UNSLOTH_STUDIO_PUBLIC_URL_WAIT_SECONDS"
_PUBLIC_URL_WAIT_TIMEOUT_SECONDS = 8.0
_OWNED_SERVER_APP = None
_OWNED_SERVER_PORT: int | None = None
_NO_CLOUDFLARE_TUNNEL_TO_STOP = object()
_CLOUDFLARE_STOP_EXPECTED_URL = None


def _is_kaggle_environment() -> bool:
    return is_kaggle_environment()


def _ensure_notebook_frame_token() -> str:
    token = os.environ.get(_NOTEBOOK_FRAME_TOKEN_ENV, "").strip()
    if not token:
        token = secrets.token_urlsafe(32)
        os.environ[_NOTEBOOK_FRAME_TOKEN_ENV] = token
    return token


def _with_notebook_frame_token(url: str) -> str:
    token = os.environ.get(_NOTEBOOK_FRAME_TOKEN_ENV, "").strip()
    if not token:
        return url
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    parts = urlsplit(url)
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values = True)
        if key != _NOTEBOOK_FRAME_TOKEN_PARAM
    ]
    query.append((_NOTEBOOK_FRAME_TOKEN_PARAM, token))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _public_url_wait_timeout() -> float:
    raw = os.environ.get(_PUBLIC_URL_WAIT_TIMEOUT_ENV, "").strip()
    if not raw:
        return _PUBLIC_URL_WAIT_TIMEOUT_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _PUBLIC_URL_WAIT_TIMEOUT_SECONDS


def _remember_owned_server(app, port: int) -> None:
    global _OWNED_SERVER_APP, _OWNED_SERVER_PORT
    _OWNED_SERVER_APP = app
    _OWNED_SERVER_PORT = port


def _can_suppress_reused_server_bootstrap(port: int) -> bool:
    return _OWNED_SERVER_APP is not None and _OWNED_SERVER_PORT == port


def _wait_for_public_url(url: str, timeout: float = 45.0) -> bool:
    """Wait for a public tunnel URL before embedding it.

    cloudflared can print the trycloudflare URL a few seconds before DNS and
    edge routing are ready. If an iframe navigates during that window, the
    browser can keep showing the initial DNS error even though opening the URL
    later works.
    """
    if not url.startswith("https://"):
        return True

    import time
    import urllib.request

    health_url = url.rstrip("/") + "/api/health"
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            req = urllib.request.Request(
                health_url,
                headers = {"User-Agent": "unsloth-studio-notebook/1"},
            )
            with urllib.request.urlopen(req, timeout = min(5.0, max(0.1, remaining))) as resp:
                if 200 <= getattr(resp, "status", 200) < 500:
                    return True
        except Exception as exc:
            last_error = exc
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(1.0, remaining))

    logger.warning(
        f"Notebook tunnel URL did not become reachable before embedding; "
        f"not rendering the iframe yet. Last error: {last_error}"
    )
    return False


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


def start_cloudflare_tunnel(port: int, *, allow_bootstrap_pending: bool = False) -> "str | None":
    """Open a shareable Cloudflare quick tunnel to localhost:*port*, or None.

    run_server suppresses the tunnel on Colab by design, so we start it directly.
    Refused while the bootstrap password is pending unless the caller will suppress
    public bootstrap injection; any failure collapses to None and the Colab proxy
    still works.
    """
    bootstrap_pending = _bootstrap_password_pending()
    if bootstrap_pending and not allow_bootstrap_pending:
        logger.warning(
            "Cloudflare link not started: the admin account still has its temporary "
            "bootstrap password, which is exposed to anyone who can load the page. "
            "Open Studio in this tab, log in and change the admin password, then re-run "
            "start(cloudflare=True) to get the shareable link."
        )
        return None
    if bootstrap_pending:
        logger.warning(
            "Starting the tunnel before the admin password is changed because this "
            "Kaggle-backed notebook has no usable Colab proxy. Bootstrap credentials "
            "will not be injected into the public tunnel page; use the notebook-local "
            "password file to sign in, then change the admin password."
        )
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


def _set_public_tunnel_bootstrap_suppression(enabled: bool) -> bool:
    try:
        from main import app as _studio_app
        _studio_app.state.suppress_bootstrap_injection_for_public_tunnel = bool(enabled)
        return True
    except Exception as e:
        logger.info(f"Could not set public tunnel bootstrap suppression ({e}).")
        return False


def _publish_cloudflare_url(
    cloudflare_url: "str | None", *, suppress_bootstrap: "bool | None" = None
) -> bool:
    """Publish a directly-started tunnel URL onto app.state so /api/health advertises it.

    run_server only sets this when it opens the tunnel itself, which it skips on Colab,
    so we set it here. Otherwise the frontend's API examples fall back to an
    unreachable server_url. Returns False when state could not be updated.
    """
    if not cloudflare_url:
        return True
    try:
        from main import app as _studio_app

        _studio_app.state.cloudflare_url = cloudflare_url
        _studio_app.state.suppress_bootstrap_injection_for_public_tunnel = bool(
            _bootstrap_password_pending() if suppress_bootstrap is None else suppress_bootstrap
        )
        _studio_app.state.trust_cloudflare_client_ip = True
        _studio_app.state.cloudflare_client_ip_requires_frame_cookie = True
        return True
    except Exception as e:
        logger.info(f"Could not publish Cloudflare URL to /api/health ({e}).")
        return False


def _start_and_publish_cloudflare_tunnel(
    port: int, *, allow_bootstrap_pending: bool = False
) -> "str | None":
    bootstrap_pending = _bootstrap_password_pending()
    if bootstrap_pending and allow_bootstrap_pending:
        if not _set_public_tunnel_bootstrap_suppression(True):
            logger.warning(
                "Cloudflare link not started: public bootstrap credential suppression "
                "could not be confirmed."
            )
            return None

    cf_url = start_cloudflare_tunnel(
        port,
        allow_bootstrap_pending = allow_bootstrap_pending,
    )
    if not cf_url:
        if bootstrap_pending and allow_bootstrap_pending:
            _set_public_tunnel_bootstrap_suppression(False)
        return None

    if not _publish_cloudflare_url(
        cf_url,
        suppress_bootstrap = bootstrap_pending and allow_bootstrap_pending,
    ):
        _stop_cloudflare_tunnel_started_for(cf_url)
        return None
    return cf_url


def _stop_cloudflare_tunnel_started_for(expected_url: "str | None") -> bool:
    global _CLOUDFLARE_STOP_EXPECTED_URL
    previous = _CLOUDFLARE_STOP_EXPECTED_URL
    _CLOUDFLARE_STOP_EXPECTED_URL = expected_url or _NO_CLOUDFLARE_TUNNEL_TO_STOP
    try:
        return _stop_cloudflare_tunnel()
    finally:
        _CLOUDFLARE_STOP_EXPECTED_URL = previous


def _stop_cloudflare_tunnel(*, expected_url: "str | None" = None) -> bool:
    """Best-effort teardown of the Cloudflare tunnel started by start_cloudflare_tunnel."""
    if expected_url is None:
        expected_url = _CLOUDFLARE_STOP_EXPECTED_URL
        if expected_url is _NO_CLOUDFLARE_TUNNEL_TO_STOP:
            return False
        if not isinstance(expected_url, str):
            expected_url = None

    current_url = None
    try:
        from main import app as _studio_app
        current_url = getattr(_studio_app.state, "cloudflare_url", None)
        if expected_url and current_url and current_url != expected_url:
            logger.info(
                f"Skipping stale Cloudflare tunnel cleanup for {expected_url}; "
                f"current tunnel is {current_url}."
            )
            return False
    except Exception:
        _studio_app = None

    try:
        from cloudflare_tunnel import stop_studio_tunnel
        stop_studio_tunnel()
    except Exception as exc:
        logger.info(f"Cloudflare tunnel stop failed; preserving published tunnel state ({exc}).")
        return False

    # Stop /api/health advertising a dead tunnel.
    try:
        if _studio_app is None:
            from main import app as _studio_app

        _studio_app.state.cloudflare_url = None
        _studio_app.state.suppress_bootstrap_injection_for_public_tunnel = False
        _studio_app.state.trust_cloudflare_client_ip = False
        _studio_app.state.cloudflare_client_ip_requires_frame_cookie = True
    except Exception as exc:
        logger.info(f"Could not clear Cloudflare tunnel state ({exc}).")
        return False
    return True


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
            Shareable Studio Link is Ready!
        </h2>
        <a href="{cloudflare_url}" onclick="var w=window.open(this.href,'_blank');if(!w){{return true;}}return false;"
           style="display: inline-flex; align-items: center; gap: 10px; padding: 14px 28px;
                  background: #000000; color: white; text-decoration: none; border-radius: 8px;
                  font-weight: 800; font-size: 16px; cursor: pointer;">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="white"><polygon points="5,3 19,12 5,21"/></svg>
            Open Unsloth Studio
        </a>
        <p style="color: #333333; margin: 12px 0 0 0; font-size: 14px; font-weight: bold;">
            This Cloudflare HTTPS link works from any device. Keep it private until the admin password is changed. The notebook view below only works in this tab.
        </p>
        <p style="color: #333333; margin: 16px 0 0 0; font-size: 13px; font-family: monospace; font-weight: bold;">
            🔗 {cloudflare_url}
        </p>
    </div>
    """


def _public_url_not_ready_html(cloudflare_url: str) -> str:
    return f"""
    <div style="display: inline-block; padding: 16px 20px; background: #fff7d6; border: 2px solid #b88700;
                border-radius: 10px; margin: 10px 0; font-family: system-ui, -apple-system, sans-serif;">
        <div style="color:#3b2a00;font-weight:800;font-size:15px;margin-bottom:8px;">Studio link is still starting</div>
        <div style="color:#3b2a00;font-size:13px;line-height:1.45;">
            The Cloudflare URL did not answer the health check yet, so the notebook iframe was not loaded.
            Open the shareable link after a few seconds, or re-run the cell if it does not become reachable.
        </div>
        <div style="color:#3b2a00;margin-top:10px;font-size:13px;font-family:monospace;font-weight:bold;">
            {cloudflare_url}
        </div>
    </div>
    """


def _bootstrap_login_notice_html() -> "str | None":
    """Notebook-local login hint for public tunnels with bootstrap injection off."""
    try:
        from auth import storage

        if not storage.requires_password_change(storage.DEFAULT_ADMIN_USERNAME):
            return None
        bootstrap_path = storage.DB_PATH.parent / ".bootstrap_password"
        return f"""
    <div style="display: inline-block; padding: 16px 20px; background: #fff7d6; border: 2px solid #b88700;
                border-radius: 10px; margin: 10px 0; font-family: system-ui, -apple-system, sans-serif;">
        <div style="color:#3b2a00;font-weight:800;font-size:15px;margin-bottom:8px;">Bootstrap login required</div>
        <div style="color:#3b2a00;font-size:13px;line-height:1.45;">
            Username: <code>{storage.DEFAULT_ADMIN_USERNAME}</code><br>
            Password file: <code>{bootstrap_path}</code><br>
            The password is intentionally not embedded in the public Cloudflare page.
        </div>
    </div>
    """
    except Exception as exc:
        logger.info(f"Could not render bootstrap login notice ({exc}).")
        return None


def _show_and_embed(port: int, *, cloudflare_url: "str | None" = None):
    """Render the Studio header + iframe for *port*, with a shareable-link card above
    when *cloudflare_url* is set. Falls back to serve_kernel_port_as_iframe."""
    is_kaggle = _is_kaggle_environment()
    if cloudflare_url and is_kaggle:
        url = cloudflare_url
        use_cloudflare_iframe = True
    else:
        url = get_colab_url(port)
        use_cloudflare_iframe = bool(cloudflare_url and url.startswith("http://localhost:"))
        if use_cloudflare_iframe:
            url = cloudflare_url
    iframe_url = _with_notebook_frame_token(url) if use_cloudflare_iframe else url
    logger.info(f"🌐 Unsloth Studio URL: {url}")
    if cloudflare_url:
        logger.info(f"🔗 Shareable Cloudflare link: {cloudflare_url}")

    try:
        from IPython.display import HTML, display

        iframe_id = f"unsloth-studio-{port}"

        # Truncated header URL — best-effort, falls back to full URL.
        try:
            port_prefix = f"{port}-"
            idx = url.index(port_prefix)
            next_dash = url.index("-", idx + len(port_prefix))
            short_url = url[: next_dash + 1] + "..."
        except (ValueError, IndexError):
            short_url = url

        if cloudflare_url:
            bootstrap_notice = _bootstrap_login_notice_html()
            if bootstrap_notice:
                display(HTML(bootstrap_notice))
            display(HTML(_shareable_link_html(cloudflare_url)))

        if use_cloudflare_iframe:
            if not _wait_for_public_url(cloudflare_url, timeout = _public_url_wait_timeout()):
                display(HTML(_public_url_not_ready_html(cloudflare_url)))
                return

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
    src="{iframe_url}"
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


def start(port: int = 8888, *, cloudflare: "bool | None" = None):
    """Start Unsloth Studio in Colab and display the URL.

    Args:
        port: Port to bind/serve on.
        cloudflare: Opt in/out of a shareable Cloudflare HTTPS link reachable
            from any device. ``None`` means auto: off on Colab, on in Kaggle
            because the Colab proxy URL is not reachable there. ``False`` is an
            explicit opt-out.

    Usage:
        start()                    # auto: Colab proxy, Kaggle tunnel
        start(cloudflare=False)    # never open a tunnel
        start(cloudflare=True)     # force a shareable Cloudflare link
    """
    import time

    logger.info("🦥 Starting Unsloth Studio...")
    is_kaggle = _is_kaggle_environment()
    effective_cloudflare = is_kaggle if cloudflare is None else bool(cloudflare)
    if is_kaggle:
        os.environ.setdefault("UNSLOTH_STUDIO_HOSTED_NOTEBOOK", "1")
        _ensure_notebook_frame_token()
        if cloudflare is None:
            logger.warning(
                "Kaggle notebooks do not expose Colab's proxy URL to the browser; "
                "opening a Cloudflare quick tunnel by default. Keep the generated "
                "URL private, or pass cloudflare=False to opt out."
            )

    # Fast path: Studio already running (cell re-run). Re-launching would collide on
    # the port, so just re-show the link and iframe.
    if _is_studio_healthy(port):
        logger.info(f"   Studio is already running on port {port} — reusing existing server.")
        # try/finally: tear the tunnel down even if interrupted mid-start/render.
        cf_url = None
        try:
            cf_url = (
                _start_and_publish_cloudflare_tunnel(
                    port,
                    allow_bootstrap_pending = (
                        is_kaggle and _can_suppress_reused_server_bootstrap(port)
                    ),
                )
                if effective_cloudflare
                else None
            )
            _show_and_embed(port, cloudflare_url = cf_url)
            for _ in range(10000):
                time.sleep(300)
                print("=", end = "", flush = True)
        except KeyboardInterrupt:
            logger.info("\nUnsloth Studio keepalive stopped.")
        finally:
            _stop_cloudflare_tunnel_started_for(cf_url)
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
        # cloudflare=False: this helper owns the tunnel. run_server's default True
        # would tunnel this 0.0.0.0 bind if Colab detection fails, breaking the opt-out.
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
    _remember_owned_server(app, actual_port)

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
    cf_url = None
    try:
        cf_url = (
            _start_and_publish_cloudflare_tunnel(
                actual_port,
                allow_bootstrap_pending = is_kaggle,
            )
            if effective_cloudflare
            else None
        )
        _show_and_embed(actual_port, cloudflare_url = cf_url)

        # Keep kernel alive so the daemon server thread runs.
        for _ in range(10000):
            time.sleep(300)
            print("=", end = "", flush = True)
    except KeyboardInterrupt:
        logger.info("\nUnsloth Studio keepalive stopped.")
    finally:
        _stop_cloudflare_tunnel_started_for(cf_url)


if __name__ == "__main__":
    start()
