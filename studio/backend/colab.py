# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Notebook launch helpers for Unsloth Studio."""

from pathlib import Path
import os
import sys
import time

# Seed platform._sys_version_cache before attrs->rich->structlog->platform crash on conda Python.
# See: https://github.com/python/cpython/issues/102396
_backend_dir = str(Path(__file__).parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
import _platform_compat  # noqa: F401


from loggers import get_logger
from utils.notebook_env import is_kaggle_environment as _is_kaggle_environment

logger = get_logger(__name__)

_PUBLIC_URL_WAIT_TIMEOUT_ENV = "UNSLOTH_STUDIO_PUBLIC_URL_WAIT_SECONDS"
_PUBLIC_URL_WAIT_TIMEOUT_SECONDS = 8.0
_OWNED_SERVER_PORT: int | None = None


def _public_url_wait_timeout() -> float:
    raw = os.environ.get(_PUBLIC_URL_WAIT_TIMEOUT_ENV, "").strip()
    if not raw:
        return _PUBLIC_URL_WAIT_TIMEOUT_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _PUBLIC_URL_WAIT_TIMEOUT_SECONDS


def _wait_for_public_url(url: str, timeout: float) -> bool:
    """Wait until a fallback tunnel serves an embeddable Studio document."""
    if not url.startswith("https://"):
        return True

    import urllib.request
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    health_url = urlunsplit((parts.scheme, parts.netloc, "/api/health", "", ""))
    deadline = time.monotonic() + timeout
    last_error: Exception | str | None = None
    headers = {"User-Agent": "unsloth-studio-notebook/1"}

    while (remaining := deadline - time.monotonic()) > 0:
        request_timeout = min(5.0, max(0.1, remaining))
        try:
            request = urllib.request.Request(health_url, headers = headers)
            with urllib.request.urlopen(request, timeout = request_timeout):
                pass

            request = urllib.request.Request(url, headers = headers)
            with urllib.request.urlopen(request, timeout = request_timeout) as response:
                status = getattr(response, "status", 200)
                frame_options = response.headers.get("X-Frame-Options", "").strip()
                csp = response.headers.get("Content-Security-Policy", "")
                if not 200 <= status < 400:
                    last_error = f"iframe document returned HTTP {status}"
                elif frame_options:
                    last_error = f"iframe document set X-Frame-Options: {frame_options}"
                elif "frame-ancestors 'none'" in csp:
                    last_error = "iframe document CSP still denies frame ancestors"
                else:
                    return True
        except Exception as exc:
            last_error = exc
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))

    logger.warning(
        "Notebook tunnel URL did not become reachable before embedding; "
        f"not rendering the iframe yet. Last error: {last_error}"
    )
    return False


def get_colab_url(port: int = 8888) -> str:
    """
    Get the Colab proxy URL for a port.

    Retries up to 3 times, validating the result is a real HTTPS Colab URL.
    Falls back to http://localhost:{port} only when all attempts fail.
    """
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
            time.sleep(1)

    logger.warning(
        f"Could not get a valid Colab proxy URL after 3 attempts — using localhost fallback. "
        f"The link/iframe may not work from outside the runtime."
    )
    return fallback


def _short_url(url: str, port: int) -> str:
    try:
        prefix_end = url.index(f"{port}-") + len(f"{port}-")
        return url[: url.index("-", prefix_end) + 1] + "..."
    except ValueError:
        return url


def _branded_link_html(url: str, *, title: str, note: str, display_url: str) -> str:
    return f"""
    <div style="display: inline-block; padding: 20px; background: #ffffff; border: 2px solid #000000;
                border-radius: 12px; margin: 10px 0; font-family: system-ui, -apple-system, sans-serif;">
        <h2 style="color: #000000; margin: 0 0 12px 0; font-size: 26px; font-weight: 800;
                   display: flex; align-items: center; gap: 12px;">
            <img src="https://github.com/unslothai/unsloth/raw/main/studio/frontend/public/unsloth-gem.png"
                 width="48" height="48" style="display:block;width:48px;height:48px;max-width:48px;max-height:48px;object-fit:contain;flex:0 0 48px;">
            {title}
        </h2>
        <a href="{url}" onclick="var w=window.open(this.href,'_blank');if(!w){{return true;}}return false;"
           style="display: inline-flex; align-items: center; gap: 10px; padding: 14px 28px;
                  background: #000000; color: white; text-decoration: none; border-radius: 8px;
                  font-weight: 800; font-size: 16px; cursor: pointer;">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="white"><polygon points="5,3 19,12 5,21"/></svg>
            Open Unsloth Studio
        </a>
        <p style="color: #333333; margin: 12px 0 0 0; font-size: 14px; font-weight: bold;">
            {note}
        </p>
        <p style="color: #333333; margin: 16px 0 0 0; font-size: 13px; font-family: monospace; font-weight: bold;">
            {display_url}
        </p>
    </div>
    """


def show_link(port: int = 8888, *, _url: "str | None" = None):
    """Display a styled clickable link to the UI.

    *_url* is an optional pre-fetched proxy URL; pass it to avoid a second eval_js round-trip.
    """
    from IPython.display import display, HTML

    url = _url if _url is not None else get_colab_url(port)
    logger.info(f"🌐 Unsloth Studio URL: {url}")
    display(
        HTML(
            _branded_link_html(
                url,
                title = "Unsloth Studio is Ready!",
                note = "If the link doesn't work, scroll down to view Studio in Colab.",
                display_url = _short_url(url, port),
            )
        )
    )


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


def _bootstrap_tunnel_state(allow_bootstrap_pending: bool) -> "bool | None":
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
            "Kaggle notebook has no usable Colab proxy. Bootstrap credentials will "
            "not be injected into the public page; the temporary password is shown "
            "only in this notebook output."
        )
    return bootstrap_pending


def _open_cloudflare_tunnel(port: int) -> "str | None":
    try:
        from cloudflare_tunnel import start_studio_tunnel
        url = start_studio_tunnel(port)
    except Exception as e:
        logger.info(f"Cloudflare tunnel unavailable ({e}); continuing without a public link.")
        return None
    if not url:
        logger.info("Cloudflare tunnel did not produce a URL; continuing without a public link.")
    return url


def start_cloudflare_tunnel(port: int, *, allow_bootstrap_pending: bool = False) -> "str | None":
    """Open and publish a quick tunnel, refusing unsafe bootstrap exposure."""
    bootstrap_pending = _bootstrap_tunnel_state(allow_bootstrap_pending)
    if bootstrap_pending is None:
        return None
    if bootstrap_pending and not _set_public_tunnel_bootstrap_suppression(True):
        logger.warning(
            "Cloudflare link not started: public bootstrap credential suppression "
            "could not be confirmed."
        )
        return None

    cloudflare_url = _open_cloudflare_tunnel(port)
    if not cloudflare_url:
        if bootstrap_pending:
            _set_public_tunnel_bootstrap_suppression(False)
        return None
    if not _publish_cloudflare_url(
        cloudflare_url,
        suppress_bootstrap = bootstrap_pending,
    ):
        _stop_cloudflare_tunnel(cloudflare_url)
        return None
    return cloudflare_url


def _set_public_tunnel_bootstrap_suppression(enabled: bool) -> bool:
    try:
        from main import app as _studio_app
        _studio_app.state.suppress_bootstrap_injection_for_public_tunnel = bool(enabled)
        return True
    except Exception as e:
        logger.info(f"Could not set public tunnel bootstrap suppression ({e}).")
        return False


def _publish_cloudflare_url(cloudflare_url: str, *, suppress_bootstrap: bool) -> bool:
    """Publish notebook-owned tunnel state for health and request middleware."""
    try:
        from main import app as _studio_app

        _studio_app.state.cloudflare_url = cloudflare_url
        _studio_app.state.suppress_bootstrap_injection_for_public_tunnel = suppress_bootstrap
        _studio_app.state.trust_cloudflare_client_ip = True
        return True
    except Exception as e:
        logger.info(f"Could not publish Cloudflare URL to /api/health ({e}).")
        return False


def _stop_cloudflare_tunnel(expected_url: str) -> bool:
    """Stop this session's tunnel without tearing down a newer replacement."""
    try:
        from main import app as _studio_app
        current_url = getattr(_studio_app.state, "cloudflare_url", None)
        if current_url and current_url != expected_url:
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

    try:
        if _studio_app is None:
            from main import app as _studio_app

        _studio_app.state.cloudflare_url = None
        _studio_app.state.suppress_bootstrap_injection_for_public_tunnel = False
        _studio_app.state.trust_cloudflare_client_ip = False
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
    return _branded_link_html(
        cloudflare_url,
        title = "Shareable Studio Link is Ready!",
        note = (
            "This Cloudflare HTTPS link works from any device. Keep it private until "
            "the admin password is changed. The notebook view below only works in this tab."
        ),
        display_url = f"🔗 {cloudflare_url}",
    )


def _kaggle_link_html(cloudflare_url: str) -> str:
    return f"""
    <div style="display:inline-block;padding:16px 20px;background:#ffffff;border:2px solid #000000;
                border-radius:10px;margin:10px 0;font-family:system-ui,-apple-system,sans-serif;">
        <a href="{cloudflare_url}" target="_blank" rel="noopener noreferrer"
           style="display:inline-flex;align-items:center;padding:12px 22px;background:#000000;color:#ffffff;
                  text-decoration:none;border-radius:8px;font-weight:800;font-size:15px;">
            Open Unsloth Studio
        </a>
        <div style="color:#333333;margin-top:12px;font-size:13px;font-family:monospace;font-weight:bold;">
            {cloudflare_url}
        </div>
    </div>
    """


def _public_url_not_ready_html(cloudflare_url: str) -> str:
    return f"""
    <div style="display: inline-block; padding: 16px 20px; background: #fff7d6; border: 2px solid #b88700;
                border-radius: 10px; margin: 10px 0; font-family: system-ui, -apple-system, sans-serif;">
        <div style="color:#3b2a00;font-weight:800;font-size:15px;margin-bottom:8px;">Studio link is still starting</div>
        <div style="color:#3b2a00;font-size:13px;line-height:1.45;">
            The Cloudflare URL did not pass the notebook frame readiness check, so the notebook iframe was not loaded.
            Open the shareable link after a few seconds, or re-run the cell if it does not become reachable.
        </div>
        <div style="color:#3b2a00;margin-top:10px;font-size:13px;font-family:monospace;font-weight:bold;">
            {cloudflare_url}
        </div>
    </div>
    """


def _kaggle_bootstrap_notice_html() -> "str | None":
    """Render Kaggle-only credentials while public-page injection is suppressed."""
    try:
        from html import escape
        from auth import storage

        if not storage.requires_password_change(storage.DEFAULT_ADMIN_USERNAME):
            return None
        bootstrap_path = storage.DB_PATH.parent / ".bootstrap_password"
        bootstrap_password = storage.get_bootstrap_password()
        if not bootstrap_password:
            try:
                bootstrap_password = bootstrap_path.read_text().strip()
            except Exception:
                bootstrap_password = ""
        password_line = (
            f"Temporary password: <code>{escape(bootstrap_password)}</code><br>"
            if bootstrap_password
            else ""
        )
        return f"""
    <div style="display: inline-block; padding: 16px 20px; background: #fff7d6; border: 2px solid #b88700;
                border-radius: 10px; margin: 10px 0; font-family: system-ui, -apple-system, sans-serif;">
        <div style="color:#3b2a00;font-weight:800;font-size:15px;margin-bottom:8px;">Bootstrap login required</div>
        <div style="color:#3b2a00;font-size:13px;line-height:1.45;">
            Username: <code>{escape(storage.DEFAULT_ADMIN_USERNAME)}</code><br>
            {password_line}
            Password file: <code>{escape(str(bootstrap_path))}</code><br>
            This notebook-only password is intentionally not embedded in the public Cloudflare page.
        </div>
    </div>
    """
    except Exception as exc:
        logger.info(f"Could not render bootstrap login notice ({exc}).")
        return None


def _show_and_embed(port: int, *, cloudflare_url: "str | None" = None):
    """Show a link only on Kaggle; otherwise render the Colab inline view."""
    if _is_kaggle_environment():
        if not cloudflare_url:
            logger.warning(
                "Kaggle Studio link is unavailable; no notebook preview will be rendered."
            )
            return
        logger.info(f"🌐 Unsloth Studio URL: {cloudflare_url}")
        logger.info(f"🔗 Shareable Cloudflare link: {cloudflare_url}")
        try:
            from IPython.display import HTML, display

            bootstrap_notice = _kaggle_bootstrap_notice_html()
            if bootstrap_notice:
                display(HTML(bootstrap_notice))
            display(HTML(_kaggle_link_html(cloudflare_url)))
        except Exception as exc:
            logger.info(f"Could not display the Kaggle Studio link ({exc}).")
        return

    proxy_url = get_colab_url(port)
    use_cloudflare_iframe = bool(cloudflare_url and proxy_url.startswith("http://localhost:"))
    url = cloudflare_url if use_cloudflare_iframe else proxy_url
    logger.info(f"🌐 Unsloth Studio URL: {url}")
    if cloudflare_url:
        logger.info(f"🔗 Shareable Cloudflare link: {cloudflare_url}")

    try:
        from IPython.display import HTML, display

        if cloudflare_url:
            display(HTML(_shareable_link_html(cloudflare_url)))

        if use_cloudflare_iframe and not _wait_for_public_url(
            cloudflare_url,
            timeout = _public_url_wait_timeout(),
        ):
            display(HTML(_public_url_not_ready_html(cloudflare_url)))
            return

        display(
            HTML(f"""
<div style="font-family:system-ui,-apple-system,sans-serif;margin:8px 0;
            border-radius:12px;overflow:hidden;box-shadow:0 2px 16px rgba(0,0,0,0.18);">
  <div style="display:flex;align-items:center;gap:10px;padding:10px 16px;background:#000;">
    <img src="https://github.com/unslothai/unsloth/raw/main/studio/frontend/public/unsloth-gem.png"
         width="26" height="26" style="display:block;width:26px;height:26px;max-width:26px;max-height:26px;object-fit:contain;flex:0 0 26px;">
    <span style="color:#fff;font-weight:700;font-size:15px;letter-spacing:-0.2px;">Unsloth Studio</span>
    <span style="margin-left:auto;color:#666;font-size:11px;font-family:monospace;">{_short_url(url, port)}</span>
  </div>
  <iframe
    id="unsloth-studio-{port}"
    src="{url}"
    style="width:100%;height:82vh;min-height:600px;max-height:1100px;border:none;display:block;box-sizing:border-box;"
    allow="clipboard-read; clipboard-write"
  ></iframe>
</div>
""")
        )
    except Exception:
        try:
            from google.colab import output as colab_output
            colab_output.serve_kernel_port_as_iframe(port, height = 900, width = "100%")
        except ImportError:
            pass


def _wait_for_studio(port: int) -> bool:
    import urllib.request
    for _ in range(40):
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/api/health", timeout = 1):
                return True
        except Exception:
            time.sleep(0.5)
    return False


def _run_notebook_session(port: int, *, cloudflare: bool, allow_bootstrap_pending: bool) -> None:
    cloudflare_url = None
    try:
        if cloudflare:
            cloudflare_url = start_cloudflare_tunnel(
                port,
                allow_bootstrap_pending = allow_bootstrap_pending,
            )
        _show_and_embed(port, cloudflare_url = cloudflare_url)
        for _ in range(10000):
            time.sleep(300)
            print("=", end = "", flush = True)
    except KeyboardInterrupt:
        logger.info("\nUnsloth Studio keepalive stopped.")
    finally:
        if cloudflare_url:
            _stop_cloudflare_tunnel(cloudflare_url)


def start(port: int = 8888, *, cloudflare: "bool | None" = None):
    """Start Unsloth Studio in Colab or Kaggle and display its entrypoint.

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
    global _OWNED_SERVER_PORT

    logger.info("🦥 Starting Unsloth Studio...")
    is_kaggle = _is_kaggle_environment()
    effective_cloudflare = is_kaggle if cloudflare is None else bool(cloudflare)
    if is_kaggle:
        if cloudflare is None:
            logger.warning(
                "Kaggle notebooks do not support the embedded Colab preview; opening "
                "a Cloudflare quick tunnel by default. Keep the generated URL private, "
                "or pass cloudflare=False to opt out."
            )

    if _is_studio_healthy(port):
        logger.info(f"   Studio is already running on port {port} — reusing existing server.")
        _run_notebook_session(
            port,
            cloudflare = effective_cloudflare,
            allow_bootstrap_pending = is_kaggle and _OWNED_SERVER_PORT == port,
        )
        return

    logger.info("   Loading backend...")
    from run import run_server

    frontend_path = Path(__file__).parent.parent / "frontend" / "dist"

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
    except (SystemExit, Exception) as exc:
        logger.error(f"❌ Unsloth Studio failed to start: {exc}")
        return

    # run_server auto-increments the port if in use; read back the bound port so the
    # proxy URL and iframe point at the right place.
    actual_port: int = getattr(getattr(app, "state", None), "server_port", None) or port
    _OWNED_SERVER_PORT = actual_port

    logger.info(f"   Server started on port {actual_port}!")

    if not _wait_for_studio(actual_port):
        logger.error(
            f"❌ Unsloth Studio did not become healthy on port {actual_port}. "
            "Check for errors above."
        )
        return

    _run_notebook_session(
        actual_port,
        cloudflare = effective_cloudflare,
        allow_bootstrap_pending = is_kaggle,
    )


if __name__ == "__main__":
    start()
