# SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
# Copyright © 2025 Unsloth AI

"""
Run script for Unsloth UI Backend.
Works independently and can be moved to any directory.
"""

import os
import sys

# Suppress annoying C-level dependency warnings globally (e.g. SwigPyPacked)
os.environ["PYTHONWARNINGS"] = "ignore"

from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from loggers import get_logger

logger = get_logger(__name__)


def _resolve_external_ip() -> str:
    """
    Resolve the machine's external IP address.

    Tries (in order):
    1. GCE metadata server (instant, works on Google Cloud VMs)
    2. ifconfig.me (works anywhere with internet)
    3. LAN IP via UDP socket trick (fallback)
    """
    import urllib.request
    import socket

    # 1. Try GCE metadata server (responds in <10ms on GCE, times out fast elsewhere)
    try:
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip",
            headers = {"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout = 1) as resp:
            ip = resp.read().decode().strip()
            if ip:
                return ip
    except Exception:
        pass

    # 2. Try public IP service
    try:
        with urllib.request.urlopen("https://ifconfig.me", timeout = 3) as resp:
            ip = resp.read().decode().strip()
            if ip:
                return ip
    except Exception:
        pass

    # 3. Fallback: LAN IP via UDP socket trick
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "0.0.0.0"


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    frontend_path: Path = "studio/frontend/dist",
    silent: bool = False,
):
    """
    Start the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to
        frontend_path: Path to frontend build directory (optional)
        silent: Suppress startup messages
    """
    import nest_asyncio

    nest_asyncio.apply()

    import asyncio
    from threading import Thread
    import time
    import uvicorn

    from main import app, setup_frontend
    from utils.paths import ensure_studio_directories

    # Create all standard directories on startup
    ensure_studio_directories()

    # Setup frontend if path provided
    if frontend_path:
        if setup_frontend(app, frontend_path):
            if not silent:
                print(f"✅ Frontend loaded from {frontend_path}")
        else:
            if not silent:
                print(f"⚠️ Frontend not found at {frontend_path}")

    # Run server
    def _run():
        config = uvicorn.Config(
            app, host = host, port = port, log_level = "info", access_log = False
        )
        server = uvicorn.Server(config)
        asyncio.run(server.serve())

    thread = Thread(target = _run, daemon = True)
    thread.start()
    time.sleep(3)

    if not silent:
        display_host = _resolve_external_ip() if host == "0.0.0.0" else host

        print("")
        print("=" * 50)
        print(f"🦥 Unsloth Studio is running on port {port}")
        print(f"   Local:    http://localhost:{port}")
        print(f"   External: http://{display_host}:{port}")
        print(f"   API:      http://{display_host}:{port}/api")
        print(f"   Health:   http://{display_host}:{port}/api/health")
        print("=" * 50)

    return app


# For direct execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description = "Run Unsloth UI Backend server")
    parser.add_argument("--host", default = "0.0.0.0", help = "Host to bind to")
    parser.add_argument("--port", type = int, default = 8000, help = "Port to bind to")
    parser.add_argument(
        "--frontend",
        type = str,
        default = "studio/frontend/dist",
        help = "Path to frontend build",
    )
    parser.add_argument("--silent", action = "store_true", help = "Suppress output")

    args = parser.parse_args()

    frontend_path = Path(args.frontend) if args.frontend else None
    run_server(
        host = args.host, port = args.port, frontend_path = frontend_path, silent = args.silent
    )

    # Keep running
    import time

    while True:
        time.sleep(1)
