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
    """
    try:
        from google.colab.output import eval_js

        # Use Colab's proxy mechanism
        url = eval_js(f"google.colab.kernel.proxyPort({port})", timeout_sec = 5)
        return url if url else f"http://localhost:{port}"
    except Exception as e:
        logger.info(f"Note: Could not get Colab URL ({e})")
        return f"http://localhost:{port}"


def show_link(port: int = 8888):
    """Display a styled clickable link to the UI."""
    from IPython.display import display, HTML

    # Get real Colab proxy URL
    url = get_colab_url(port)

    short_url = (
        url[: url.index("-", url.index(f"{port}-") + len(str(port)) + 1) + 1] + "..."
        if f"{port}-" in url
        else url
    )
    html = f"""
    <div style="display: inline-block; padding: 20px; background: #ffffff; border: 2px solid #000000;
                border-radius: 12px; margin: 10px 0; font-family: system-ui, -apple-system, sans-serif;">
        <h2 style="color: #000000; margin: 0 0 12px 0; font-size: 26px; font-weight: 800;
                   display: flex; align-items: center; gap: 12px;">
            <img src="https://github.com/unslothai/unsloth/raw/main/studio/frontend/public/unsloth-gem.png"
                 height="48" style="display:block;">
            Unsloth Studio is Ready!
        </h2>
        <a href="{url}" target="_blank"
           style="display: inline-flex; align-items: center; gap: 10px; padding: 14px 28px;
                  background: #000000; color: white; text-decoration: none; border-radius: 8px;
                  font-weight: 800; font-size: 16px;">
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


def start(port: int = 8888):
    """
    Start Unsloth Studio server in Colab and display the URL.

    Usage:
        from colab import start
        start()
    """
    import sys

    logger.info("🦥 Starting Unsloth Studio...")

    logger.info("   Loading backend...")
    from run import run_server

    # Auto-detect frontend path
    repo_root = Path(__file__).parent.parent
    frontend_path = repo_root / "frontend" / "dist"

    if not frontend_path.exists():
        logger.info("❌ Frontend not built! Please run the setup cell first.")
        return

    logger.info("   Starting server...")
    # Start server silently
    run_server(host = "0.0.0.0", port = port, frontend_path = frontend_path, silent = True)

    logger.info("   Server started!")

    # Show the clickable link with real URL
    show_link(port)


if __name__ == "__main__":
    start()
