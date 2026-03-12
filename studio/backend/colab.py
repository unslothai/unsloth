# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Colab-specific helpers for running Unsloth Studio.
Uses Colab's built-in proxy - no external tunneling needed!
"""

from pathlib import Path
import sys

# Add backend to path early so local modules like loggers can be imported
backend_path = str(Path(__file__).parent)
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from loggers import get_logger

logger = get_logger(__name__)


def get_colab_url(port: int = 8000) -> str:
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


def show_link(port: int = 8000):
    """Display a styled clickable link to the UI."""
    from IPython.display import display, HTML

    # Get real Colab proxy URL
    url = get_colab_url(port)

    html = f"""
    <div style="padding: 20px; background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
                border-radius: 12px; margin: 10px 0; font-family: system-ui, -apple-system, sans-serif;">
        <h2 style="color: white; margin: 0 0 12px 0; font-size: 24px;">
            🦥 Unsloth Studio is Ready!
        </h2>
        <a href="{url}" target="_blank"
           style="display: inline-block; padding: 14px 28px; background: white; color: #16a34a;
                  text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 16px;
                  box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            🚀 Open Unsloth Studio
        </a>
        <p style="color: rgba(255,255,255,0.9); margin: 16px 0 0 0; font-size: 13px;
                  word-break: break-all; font-family: monospace;">
            {url}
        </p>
    </div>
    """
    display(HTML(html))


def start(port: int = 8000):
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
