# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Colab-specific helpers for running Unsloth Studio.
Uses Colab's built-in proxy - no external tunneling needed!
"""

from pathlib import Path
import sys

# Fix for Anaconda/conda-forge Python: seed platform._sys_version_cache before
# any library imports that trigger attrs -> rich -> structlog -> platform crash.
# See: https://github.com/python/cpython/issues/102396
_backend_dir = str(Path(__file__).parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
import _platform_compat  # noqa: F401


from loggers import get_logger

logger = get_logger(__name__)


def get_colab_url(port: int = 8888) -> str:
    """
    Get the actual Colab proxy URL for a port.

    Retries up to 3 times and validates that the result is a real HTTPS Colab
    URL before returning.  Falls back to http://localhost:{port} only when all
    attempts fail.
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
            # A valid Colab proxy URL starts with https:// and embeds the port.
            if (
                url
                and isinstance(url, str)
                and url.startswith("https://")
                and str(port) in url
            ):
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

    *_url* is an optional pre-fetched Colab proxy URL. When omitted,
    ``get_colab_url(port)`` is called internally. Pass it from
    ``_show_and_embed`` to avoid a second ``eval_js`` round-trip.
    """
    from IPython.display import display, HTML

    url = _url if _url is not None else get_colab_url(port)

    # Build a truncated display URL. Wrap in try/except so an unexpected URL
    # shape never prevents the link from rendering.
    try:
        port_prefix = f"{port}-"
        idx = url.index(port_prefix)
        next_dash = url.index("-", idx + len(port_prefix))
        short_url = url[: next_dash + 1] + "..."
    except (ValueError, IndexError):
        short_url = url

    # Also emit a plain-text line so the URL is visible even if HTML display
    # is suppressed or fails.
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


def _is_studio_healthy(port: int, timeout: float = 2.0) -> bool:
    """Return True if a Studio backend is already answering health checks on *port*."""
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"http://localhost:{port}/api/health", timeout = timeout
        ):
            return True
    except Exception:
        return False


def _show_and_embed(port: int):
    """Embed the Studio inline for *port* with a branded header bar.

    Fetches the Colab proxy URL once (registering the port with Colab's
    reverse-proxy at the same time) then renders a header bar + full-height
    iframe as a single HTML block.

    Falls back to ``serve_kernel_port_as_iframe`` if ``IPython.display.HTML``
    is unavailable for any reason.
    """
    url = get_colab_url(port)
    logger.info(f"🌐 Unsloth Studio URL: {url}")

    try:
        from IPython.display import HTML, display

        iframe_id = f"unsloth-studio-{port}"

        # Truncated URL shown in the header — best-effort, falls back to full URL.
        try:
            port_prefix = f"{port}-"
            idx = url.index(port_prefix)
            next_dash = url.index("-", idx + len(port_prefix))
            short_url = url[: next_dash + 1] + "..."
        except (ValueError, IndexError):
            short_url = url

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
    except Exception:
        # Fallback: Colab's built-in helper (less control, but always works)
        try:
            from google.colab import output as colab_output

            colab_output.serve_kernel_port_as_iframe(port, height = 900, width = "100%")
        except ImportError:
            pass


def start(port: int = 8888):
    """
    Start Unsloth Studio server in Colab and display the URL.

    Usage:
        from colab import start
        start()
    """
    import time

    logger.info("🦥 Starting Unsloth Studio...")

    # --- Fast path: Studio is already running (cell re-run) ---
    # Re-launching would either collide on the port or silently shift to a new
    # port and confuse the user.  Just re-show the link and iframe instead.
    if _is_studio_healthy(port):
        logger.info(
            f"   Studio is already running on port {port} — reusing existing server."
        )
        _show_and_embed(port)
        try:
            for _ in range(10000):
                time.sleep(300)
                print("=", end = "", flush = True)
        except KeyboardInterrupt:
            logger.info("\nUnsloth Studio keepalive stopped.")
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
        app = run_server(
            host = "0.0.0.0", port = port, frontend_path = frontend_path, silent = True
        )
    except SystemExit as exc:
        logger.error(f"❌ Unsloth Studio failed to start: {exc}")
        return
    except Exception as exc:
        logger.error(f"❌ Unsloth Studio failed to start: {exc}")
        return

    # run_server auto-increments the port when the requested one is already in
    # use (e.g. Jupyter occupying 8888). Read back the actual bound port so the
    # Colab proxy URL and iframe always point at the right place.
    actual_port: int = getattr(getattr(app, "state", None), "server_port", None) or port

    logger.info(f"   Server started on port {actual_port}!")

    # Poll health endpoint to confirm the server is truly reachable before
    # showing the link and registering the iframe — avoids the race where
    # ready_event fires but the process hasn't finished binding.
    import urllib.request

    server_ready = False
    for _ in range(40):
        try:
            with urllib.request.urlopen(
                f"http://localhost:{actual_port}/api/health", timeout = 1
            ):
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

    _show_and_embed(actual_port)

    # Keep kernel alive so the daemon server thread stays running.
    # Handle KeyboardInterrupt cleanly so the user gets a readable message
    # rather than a raw traceback when they interrupt the cell.
    try:
        for _ in range(10000):
            time.sleep(300)
            print("=", end = "", flush = True)
    except KeyboardInterrupt:
        logger.info("\nUnsloth Studio keepalive stopped.")


if __name__ == "__main__":
    start()
