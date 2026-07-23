# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Colab helpers for Unsloth Studio. Uses Colab's built-in proxy."""

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
    """True on a real Colab notebook kernel (not colabtools-only ``google.colab`` imports).

    ``COLAB_RELEASE_TAG`` is set in notebook kernels but not by colabtools alone, so it
    marks real Colab where the kernel-port helper works even if eval_js fell back to localhost.
    """
    import os

    if "COLAB_RELEASE_TAG" not in os.environ:
        return False
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def _ready_card_html(url: str, port: int) -> str:
    """Branded ready card for the in-notebook Studio view.

    Colab ``*.prod.colab.dev`` proxy URLs are session-scoped and 404 when opened as a
    top-level tab or on another device, so never ``window.open`` them: embed in-cell and
    point users at ``start(cloudflare=True)`` for a shareable link.
    """
    short_url = _short_colab_url(url, port)
    if _is_colab_runtime() or _is_colab_proxy_url(url, port):
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
            Scroll down to use Studio in this Colab cell. Colab proxy links cannot be opened
            in a new tab or on another device (they 404 outside this notebook session).
        </p>
        <p style="color: #333333; margin: 0; font-size: 14px; font-weight: bold;">
            Need a real shareable / new-window URL? Re-run with
            <code style="background:#f3f3f3;padding:2px 6px;border-radius:4px;">start(cloudflare=True)</code>.
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


def show_link(port: int = 8888, *, _url: "str | None" = None):
    """Display a styled ready card for the UI.

    Colab proxy URLs are informational only (no new-tab open; they 404 outside the cell);
    non-proxy URLs keep a clickable open button. *_url* is an optional pre-fetched proxy
    URL to avoid a second eval_js round-trip.
    """
    from IPython.display import display, HTML

    url = _url if _url is not None else get_colab_url(port)
    logger.info(f"🌐 Unsloth Studio URL: {url}")
    display(HTML(_ready_card_html(url, port)))


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


def start_cloudflare_tunnel(port: int) -> "str | None":
    """Open a shareable Cloudflare quick tunnel to localhost:*port*, or None.

    run_server suppresses the tunnel on Colab, so we start it directly. Refused while the
    bootstrap password is pending; any failure collapses to None (Colab proxy still works).
    """
    if _bootstrap_password_pending():
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

    run_server sets this only when it opens the tunnel itself (skipped on Colab), so we
    set it here; otherwise the frontend's API examples fall back to an unreachable
    server_url. Best-effort.
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

    The service-marker check stops the reuse path reusing or tunneling a foreign process.
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


# Height for serve_kernel_port_as_iframe (~82vh on a 1080p screen, clamped).
_COLAB_IFRAME_HEIGHT = 900


def _embed_kernel_port_iframe(port: int) -> bool:
    """Embed Studio via Colab's native kernel-port iframe helper.

    Colab's output sanitizer often strips custom ``<iframe>`` tags without raising,
    leaving a blank cell; the kernel-port helper is the supported embedding path.
    """
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


def _embed_html_iframe(url: str, port: int) -> bool:
    """Fallback embed: raw HTML iframe when the Colab helper is unavailable."""
    try:
        from IPython.display import HTML, display
    except ImportError:
        return False

    short_url = _short_colab_url(url, port)
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
    src="{url}"
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


def _show_and_embed(port: int, *, cloudflare_url: "str | None" = None):
    """Render the Unsloth ready card + iframe for *port*.

    Prefer Colab's ``serve_kernel_port_as_iframe``; raw HTML iframe is the fallback.
    Cloudflare cards stay clickable.
    """
    url = get_colab_url(port)
    logger.info(f"🌐 Unsloth Studio URL: {url}")
    if cloudflare_url:
        logger.info(f"🔗 Shareable Cloudflare link: {cloudflare_url}")

    try:
        show_link(port, _url = url)
    except Exception as e:
        logger.info(f"Could not render Unsloth link card ({e}).")

    if cloudflare_url:
        try:
            from IPython.display import HTML, display
            display(HTML(_shareable_link_html(cloudflare_url)))
        except Exception as e:
            logger.info(f"Could not render Cloudflare link card ({e}).")

    # Real Colab: kernel helper needs only the port (works when eval_js failed).
    # colabtools can import google.colab without COLAB_RELEASE_TAG, so use HTML embed.
    if _is_colab_runtime():
        if _embed_kernel_port_iframe(port):
            return
    elif _is_colab_proxy_url(url, port) and _embed_kernel_port_iframe(port):
        return
    _embed_html_iframe(url, port)


def start(port: int = 8888, *, cloudflare: bool = False):
    """Start Unsloth Studio in Colab and display the URL.

    Args:
        port: Port to bind/serve on.
        cloudflare: Opt in to a shareable Cloudflare HTTPS link reachable from any
            device (default OFF). It exposes Unsloth's login page beyond Colab, so it
            stays an explicit opt-in; the default shows only the in-tab proxy iframe.

    Usage:
        start()                    # Colab-proxy iframe only (default)
        start(cloudflare=True)     # also open a shareable Cloudflare link
    """
    import time

    logger.info("🦥 Starting Unsloth Studio...")

    # Fast path: already running (cell re-run); re-show link/iframe instead of rebinding the port.
    if _is_studio_healthy(port):
        logger.info(f"   Unsloth is already running on port {port} — reusing existing server.")
        # try/finally: tear the tunnel down even if interrupted mid-start/render.
        try:
            cf_url = start_cloudflare_tunnel(port) if cloudflare else None
            _publish_cloudflare_url(cf_url)
            _show_and_embed(port, cloudflare_url = cf_url)
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

    # Server healthy: open the tunnel, publish its URL for /api/health, tear down on interrupt.
    try:
        cf_url = start_cloudflare_tunnel(actual_port) if cloudflare else None
        _publish_cloudflare_url(cf_url)
        _show_and_embed(actual_port, cloudflare_url = cf_url)

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
