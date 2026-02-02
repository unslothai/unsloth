"""
Run script for Unsloth UI Backend.
Works independently and can be moved to any directory.
"""
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    frontend_path: Path = None,
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
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        asyncio.run(server.serve())

    thread = Thread(target=_run, daemon=True)
    thread.start()
    time.sleep(3)

    if not silent:
        print("")
        print("=" * 50)
        print(f"🦥 Unsloth UI Backend is running on port {port}")
        print(f"   API: http://{host}:{port}/api")
        print(f"   Health: http://{host}:{port}/api/health")
        print("=" * 50)

    return app


# For direct execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Unsloth UI Backend server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--frontend", type=str, default=None, help="Path to frontend build"
    )
    parser.add_argument("--silent", action="store_true", help="Suppress output")

    args = parser.parse_args()

    frontend_path = Path(args.frontend) if args.frontend else None
    run_server(
        host=args.host, port=args.port, frontend_path=frontend_path, silent=args.silent
    )

    # Keep running
    import time

    while True:
        time.sleep(1)

