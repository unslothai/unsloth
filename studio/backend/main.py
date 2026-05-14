# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Main FastAPI application for Unsloth UI Backend
"""

import os
import sys
from pathlib import Path as _Path

# Suppress annoying C-level dependency warnings globally
os.environ["PYTHONWARNINGS"] = "ignore"

# Ensure backend dir is on sys.path so _platform_compat is importable when
# main.py is launched directly (e.g. `uvicorn main:app`).
_backend_dir = str(_Path(__file__).parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

# Fix for Anaconda/conda-forge Python: seed platform._sys_version_cache before
# any library imports that trigger attrs -> rich -> structlog -> platform crash.
# See: https://github.com/python/cpython/issues/102396
import _platform_compat  # noqa: F401

# Direct `uvicorn main:app` launches bypass run.py, so re-export here too
# (mirrors run.py). Required BEFORE the unsloth-zoo import below, since
# its LLAMA_CPP_DEFAULT_DIR binding is import-time.
from utils.paths.storage_roots import studio_root as _studio_root

try:
    _LEGACY_STUDIO_ROOT = (_Path.home() / ".unsloth" / "studio").resolve()
except (OSError, ValueError):
    _LEGACY_STUDIO_ROOT = _Path.home() / ".unsloth" / "studio"
try:
    _STUDIO_ROOT_RESOLVED = _studio_root().resolve()
except (OSError, ValueError):
    _STUDIO_ROOT_RESOLVED = _studio_root()
if _STUDIO_ROOT_RESOLVED != _LEGACY_STUDIO_ROOT:
    if not os.environ.get("UNSLOTH_STUDIO_HOME"):
        os.environ["UNSLOTH_STUDIO_HOME"] = str(_STUDIO_ROOT_RESOLVED)
    if not os.environ.get("UNSLOTH_LLAMA_CPP_PATH"):
        os.environ["UNSLOTH_LLAMA_CPP_PATH"] = str(_STUDIO_ROOT_RESOLVED / "llama.cpp")

import hashlib
import mimetypes
import re as _re
import shutil
import warnings
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version as package_version


_STUDIO_INSTALL_ID_RE = _re.compile(r"^[0-9a-f]{64}$")


def _read_studio_install_id() -> str:
    """Per-install opaque id written by install.sh / install.ps1 at
    $STUDIO_HOME/share/studio_install_id. Returns "" when the file is
    absent (pre-PR install, fresh tree never run through the installer)
    or contains anything other than a 64-char lowercase-hex token --
    in which case /api/health emits "" and the launcher's _check_health
    falls back to the existing "no baked id, accept any healthy
    Unsloth backend" path. This intentionally replaces a previous
    sha256(resolved_install_path) so the field carries no install-path
    information for callers reaching /api/health (relevant when Studio
    is run with -H 0.0.0.0)."""
    try:
        token = (
            (_STUDIO_ROOT_RESOLVED / "share" / "studio_install_id").read_text().strip()
        )
    except (OSError, ValueError):
        return ""
    return token if _STUDIO_INSTALL_ID_RE.fullmatch(token) else ""


_STUDIO_ROOT_ID_CACHE: str = _read_studio_install_id()


def _studio_root_id() -> str:
    """Same-install discriminator for /api/health: a per-install opaque
    token written once by the installer and read once at module import.
    Empty when no installer-written token is present; the launcher
    contract treats "" as "no baked id, accept any healthy backend"."""
    return _STUDIO_ROOT_ID_CACHE


# Fix broken Windows registry MIME types.  Some Windows installs map .js to
# "text/plain" in the registry (HKCR\.js\Content Type).  Python's mimetypes
# module reads from the registry, and FastAPI/Starlette's StaticFiles uses
# mimetypes.guess_type() to set Content-Type headers.  Browsers enforce strict
# MIME checking for ES module scripts (<script type="module">) and will refuse
# to execute .js files served as text/plain — resulting in a blank page.
# Calling add_type() *before* StaticFiles is instantiated ensures the correct
# types are used regardless of the OS registry.
if sys.platform == "win32":
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")

# Suppress annoying dependency warnings in production
if os.getenv("ENVIRONMENT_TYPE", "production") == "production":
    warnings.filterwarnings("ignore")
    # Alternatively, you can be more specific:
    # warnings.filterwarnings("ignore", category=DeprecationWarning)
    # warnings.filterwarnings("ignore", module="triton.*")

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, Response
from pathlib import Path
from datetime import datetime

# Import routers
from routes import (
    auth_router,
    data_recipe_router,
    datasets_router,
    export_router,
    inference_router,
    inference_studio_router,
    models_router,
    providers_router,
    training_history_router,
    training_router,
)
from auth import storage
from auth.authentication import get_current_subject
from utils.hardware import (
    detect_hardware,
    get_device,
    DeviceType,
    get_backend_visible_gpu_info,
)
import utils.hardware.hardware as _hw_module

from utils.cache_cleanup import clear_unsloth_compiled_cache
from utils.native_path_leases import native_path_leases_supported
from utils.update_status import (
    get_studio_install_source_status,
    get_studio_update_status,
)
from utils.studio_version import get_studio_version


def get_unsloth_version() -> str:
    try:
        return package_version("unsloth")
    except PackageNotFoundError:
        pass

    version_file = (
        _Path(__file__).resolve().parents[2] / "unsloth" / "models" / "_utils.py"
    )
    try:
        for line in version_file.read_text(encoding = "utf-8").splitlines():
            if line.startswith("__version__ = "):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return "dev"


UNSLOTH_VERSION = get_unsloth_version()
STUDIO_VERSION = get_studio_version()


def _load_desktop_owner() -> dict[str, str] | None:
    token = os.environ.pop("UNSLOTH_STUDIO_DESKTOP_OWNER_TOKEN", "")
    kind = os.environ.pop("UNSLOTH_STUDIO_DESKTOP_OWNER_KIND", "")
    if kind != "tauri" or not token:
        return None
    return {
        "kind": "tauri",
        "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
    }


_DESKTOP_OWNER = _load_desktop_owner()


def _desktop_owner() -> dict[str, str] | None:
    return _DESKTOP_OWNER


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: detect hardware, seed default admin if needed. Shutdown: clean up compiled cache."""
    # Clean up any stale compiled cache from previous runs
    clear_unsloth_compiled_cache()

    # Remove stale .venv_overlay from previous versions — no longer used.
    # Version switching now uses .venv_t5/ (pre-installed by setup.sh).
    overlay_dir = Path(__file__).resolve().parent.parent.parent / ".venv_overlay"
    if overlay_dir.is_dir():
        shutil.rmtree(overlay_dir, ignore_errors = True)

    # Detect hardware first — sets DEVICE global used everywhere
    detect_hardware()

    from storage.studio_db import cleanup_orphaned_runs

    try:
        cleanup_orphaned_runs()
    except Exception as exc:
        import structlog

        structlog.get_logger(__name__).warning(
            "cleanup_orphaned_runs failed at startup: %s", exc
        )

    # Pre-cache the helper GGUF model for LLM-assisted dataset detection.
    # Runs in a background thread so it doesn't block server startup.
    import threading

    def _precache():
        try:
            from utils.datasets.llm_assist import precache_helper_gguf

            precache_helper_gguf()
        except Exception:
            pass  # non-critical

    threading.Thread(target = _precache, daemon = True).start()

    # Initialize RSA key pair for API key encryption (external providers)
    from core.inference.key_exchange import init_key_pair

    init_key_pair()

    if storage.ensure_default_admin():
        bootstrap_pw = storage.get_bootstrap_password()
        app.state.bootstrap_password = bootstrap_pw

        bootstrap_path = storage.DB_PATH.parent / ".bootstrap_password"
        print("\n" + "=" * 60)
        print("DEFAULT ADMIN ACCOUNT CREATED")
        print(f"    username: {storage.DEFAULT_ADMIN_USERNAME}")
        print(f"    password saved to: {bootstrap_path}")
        print("    Open the Studio UI to sign in and change it.")
        print("=" * 60 + "\n")
    else:
        app.state.bootstrap_password = storage.get_bootstrap_password()
    yield
    # Cleanup
    _hw_module.DEVICE = None
    clear_unsloth_compiled_cache()


# Create FastAPI app
app = FastAPI(
    title = "Unsloth UI Backend",
    version = UNSLOTH_VERSION,
    description = "Backend API for Unsloth UI - Training and Model Management",
    lifespan = lifespan,
)

# Initialize structured logging
from loggers.config import LogConfig
from loggers.handlers import LoggingMiddleware

logger = LogConfig.setup_logging(
    service_name = "unsloth-studio-backend",
    env = os.getenv("ENVIRONMENT_TYPE", "production"),
)

app.add_middleware(LoggingMiddleware)


# Web-search favicons load from *.gstatic.com; everything else is same-origin.
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402
from starlette.requests import Request as _StarletteRequest  # noqa: E402


_CSP_SCRIPT_NONCE_HEADER = "x-internal-script-nonce"


def _build_csp(script_nonce: "str | None" = None) -> str:
    script_src = "script-src 'self'"
    if script_nonce:
        script_src += f" 'nonce-{script_nonce}'"
    return (
        "default-src 'self'; "
        "img-src 'self' data: blob: https://t0.gstatic.com "
        "https://t1.gstatic.com https://t2.gstatic.com "
        "https://t3.gstatic.com; "
        "connect-src 'self' https://huggingface.co https://datasets-server.huggingface.co; "
        "style-src 'self' 'unsafe-inline'; "
        f"{script_src}; "
        "font-src 'self' data:; "
        "frame-ancestors 'none'; "
        "form-action 'self'; "
        "base-uri 'self'"
    )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Set baseline security headers; splice per-response inline-script nonces into CSP."""

    async def dispatch(self, request: _StarletteRequest, call_next):
        response = await call_next(request)
        # Strip the internal nonce hand-off header so it never reaches the client.
        nonce = response.headers.get(_CSP_SCRIPT_NONCE_HEADER)
        if nonce is not None:
            del response.headers[_CSP_SCRIPT_NONCE_HEADER]
        response.headers.setdefault("Content-Security-Policy", _build_csp(nonce))
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=(), interest-cohort=()",
        )
        response.headers["server"] = "unsloth-studio"
        return response


app.add_middleware(SecurityHeadersMiddleware)


# Cap upload body on protected POSTs; default 500 MB, env-tunable.
import json as _json_for_413  # noqa: E402


_MAX_BODY_BYTES = int(os.environ.get("UNSLOTH_STUDIO_MAX_BODY_MB", "500")) * 1024 * 1024
_BODY_PROTECTED_PREFIXES = (
    "/v1/chat/completions",
    "/v1/completions",
    "/api/inference",
    "/api/data-recipe",
    "/api/datasets",
    "/api/train",
    "/api/export",
)


async def _send_413(send, total_bytes: int) -> None:
    payload = _json_for_413.dumps(
        {
            "detail": (
                f"Request body too large "
                f"({total_bytes:,} bytes; max {_MAX_BODY_BYTES:,})."
            )
        },
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload, "more_body": False})


class MaxBodyMiddleware:
    """Reject oversized bodies on protected POST/PUT/PATCH; raw ASGI so chunked uploads cannot bypass the cap."""

    def __init__(self, app, max_bytes: int, protected_prefixes: tuple):
        self.app = app
        self.max_bytes = max_bytes
        self.protected_prefixes = protected_prefixes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        method = scope.get("method", "").upper()
        path = scope.get("path", "")
        if method not in ("POST", "PUT", "PATCH") or not any(
            path.startswith(p) for p in self.protected_prefixes
        ):
            await self.app(scope, receive, send)
            return

        declared = None
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value.decode("latin-1"))
                except (ValueError, UnicodeDecodeError):
                    declared = None
                break
        if declared is not None and declared > self.max_bytes:
            await _send_413(send, declared)
            return

        chunks: list = []
        total = 0
        while True:
            msg = await receive()
            mtype = msg.get("type")
            if mtype == "http.disconnect":
                return
            if mtype != "http.request":
                # Mid-stream unexpected frame: forwarding would corrupt downstream.
                return
            body = msg.get("body", b"") or b""
            if body:
                total += len(body)
                if total > self.max_bytes:
                    await _send_413(send, total)
                    return
                chunks.append(body)
            if not msg.get("more_body", False):
                break

        replayed = {"sent": False}

        async def replay_receive():
            if not replayed["sent"]:
                replayed["sent"] = True
                return {
                    "type": "http.request",
                    "body": b"".join(chunks),
                    "more_body": False,
                }
            # After replay, fall through so http.disconnect still propagates.
            return await receive()

        await self.app(scope, replay_receive, send)


app.add_middleware(
    MaxBodyMiddleware,
    max_bytes = _MAX_BODY_BYTES,
    protected_prefixes = _BODY_PROTECTED_PREFIXES,
)


from starlette.responses import RedirectResponse as _RedirectResponse  # noqa: E402


@app.get("/recipes", include_in_schema = False)
@app.get("/recipes/{rest:path}", include_in_schema = False)
async def _recipes_redirect(rest: str = ""):
    target = "/data-recipes" + (("/" + rest) if rest else "")
    return _RedirectResponse(url = target, status_code = 308)


# CORS middleware
_api_only = os.environ.get("UNSLOTH_API_ONLY") == "1"
_cors_origins = ["*"]
if _api_only:
    _cors_origins = [
        "tauri://localhost",  # Linux/macOS Tauri webview
        "http://tauri.localhost",  # Windows Tauri webview
        "http://localhost",  # dev fallback
        "http://localhost:5173",  # Tauri dev/Vite
        "http://127.0.0.1:5173",  # Tauri dev/Vite fallback
    ]
    _cors_origin_regex = None
else:
    _cors_origin_regex = None

app.add_middleware(
    CORSMiddleware,
    allow_origins = _cors_origins,
    allow_origin_regex = _cors_origin_regex,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# ============ Register API Routes ============

# Register routers
app.include_router(auth_router, prefix = "/api/auth", tags = ["auth"])
app.include_router(training_router, prefix = "/api/train", tags = ["training"])
app.include_router(models_router, prefix = "/api/models", tags = ["models"])
app.include_router(inference_router, prefix = "/api/inference", tags = ["inference"])
# Studio-only inference endpoints (cancel, etc.) are intentionally NOT
# exposed on the /v1 OpenAI-compat prefix below.
app.include_router(inference_studio_router, prefix = "/api/inference", tags = ["inference"])

# OpenAI-compatible endpoints: mount the same inference router at /v1
# so external tools (Open WebUI, SillyTavern, etc.) can use the
# standard /v1/chat/completions path.
app.include_router(inference_router, prefix = "/v1", tags = ["openai-compat"])
app.include_router(providers_router, prefix = "/api/providers", tags = ["providers"])
app.include_router(datasets_router, prefix = "/api/datasets", tags = ["datasets"])
app.include_router(data_recipe_router, prefix = "/api/data-recipe", tags = ["data-recipe"])
app.include_router(export_router, prefix = "/api/export", tags = ["export"])
app.include_router(
    training_history_router, prefix = "/api/train", tags = ["training-history"]
)


# ============ Health and System Endpoints ============


@app.get("/api/health")
async def health_check(request: Request):
    """Liveness only; full diagnostic dict gated on a valid bearer."""
    minimal = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return minimal
    try:
        from auth.authentication import get_current_subject as _gcs
        from fastapi.security import HTTPAuthorizationCredentials

        creds = HTTPAuthorizationCredentials(
            scheme = "Bearer", credentials = auth.split(" ", 1)[1]
        )
        # Must await: a bare coroutine is truthy and would skip the auth check.
        subject = await _gcs(creds)
    except HTTPException:
        return minimal
    except Exception:
        return minimal
    if not subject:
        return minimal

    platform_map = {"darwin": "mac", "win32": "windows", "linux": "linux"}
    device_type = platform_map.get(sys.platform, sys.platform)
    return {
        **minimal,
        "service": "Unsloth UI Backend",
        "version": UNSLOTH_VERSION,
        "studio_version": STUDIO_VERSION,
        "device_type": device_type,
        "chat_only": _hw_module.CHAT_ONLY,
        "desktop_protocol_version": 1,
        "desktop_manageability_version": 1,
        "supports_desktop_auth": True,
        "supports_desktop_backend_ownership": True,
        # Hex digest of the install path; launchers reject sibling Studios on the same port.
        "studio_root_id": _studio_root_id(),
        "native_path_leases_supported": native_path_leases_supported(),
        **({"desktop_owner": owner} if (owner := _desktop_owner()) else {}),
    }


@app.get("/api/studio/install-source")
def studio_install_source(_current_subject: str = Depends(get_current_subject)):
    """Return source-aware install metadata without remote update checks."""
    return get_studio_install_source_status(UNSLOTH_VERSION)


@app.get("/api/studio/update-status")
def studio_update_status(_current_subject: str = Depends(get_current_subject)):
    """Return source-aware manual update status for browser-served Studio."""
    return get_studio_update_status(UNSLOTH_VERSION)


@app.post("/api/shutdown")
async def shutdown_server(
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """Gracefully shut down the Unsloth Studio server.

    Called by the frontend quit dialog so users can stop the server from the UI
    without needing to use the CLI or kill the process manually.
    """
    import asyncio

    async def _delayed_shutdown():
        await asyncio.sleep(0.2)  # Let the HTTP response return first
        trigger = getattr(request.app.state, "trigger_shutdown", None)
        if trigger is not None:
            trigger()
        else:
            # Fallback when not launched via run_server() (e.g. direct uvicorn)
            import signal
            import os

            os.kill(os.getpid(), signal.SIGTERM)

    request.app.state._shutdown_task = asyncio.create_task(_delayed_shutdown())
    return {"status": "shutting_down"}


@app.get("/api/system")
async def get_system_info(
    current_subject: str = Depends(get_current_subject),
):
    """Get system information.

    Gated behind auth: the response includes platform, Python version,
    GPU name, memory total, and ML package set -- enough to fingerprint
    a host. Studio's chat-only-mode design assumes only the local user
    reaches /api/system; in -H 0.0.0.0 / Colab / Tauri-relayed setups
    that assumption breaks unless we require a bearer.
    """
    import platform
    import psutil
    from utils.hardware import get_device
    from utils.hardware.hardware import _backend_label

    visibility_info = get_backend_visible_gpu_info()
    gpu_info = {
        "available": visibility_info["available"],
        "devices": visibility_info["devices"],
    }

    # CPU & Memory
    memory = psutil.virtual_memory()

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        # Use the centralized _backend_label helper so the /api/system
        # endpoint reports "rocm" on AMD hosts instead of "cuda", matching
        # the /api/hardware and /api/gpu-visibility endpoints.
        "device_backend": _backend_label(get_device()),
        "cpu_count": psutil.cpu_count(),
        "memory": {
            "total_gb": round(memory.total / 1e9, 2),
            "available_gb": round(memory.available / 1e9, 2),
            "percent_used": memory.percent,
        },
        "gpu": gpu_info,
    }


@app.get("/api/system/gpu-visibility")
async def get_gpu_visibility(
    current_subject: str = Depends(get_current_subject),
):
    return get_backend_visible_gpu_info()


@app.get("/api/system/hardware")
async def get_hardware_info(
    current_subject: str = Depends(get_current_subject),
):
    """Return GPU name, total VRAM, and key ML package versions.

    Gated behind auth alongside /api/system -- same fingerprinting
    concern. /api/system/gpu-visibility is also auth-gated already.
    """
    from utils.hardware import get_gpu_summary, get_package_versions

    return {
        "gpu": get_gpu_summary(),
        "versions": get_package_versions(),
    }


# ============ Serve Frontend (Optional) ============


def _strip_crossorigin(html_bytes: bytes) -> bytes:
    """Remove ``crossorigin`` attributes from script/link tags.

    Vite adds ``crossorigin`` by default which forces CORS mode on font
    subresource loads.  When Studio is served over plain HTTP, Firefox
    HTTPS-Only Mode does not exempt CORS font requests -- causing all
    @font-face downloads to fail silently.  Stripping the attribute
    makes them regular same-origin fetches that work on any protocol.
    """
    import re as _re

    html = html_bytes.decode("utf-8")
    html = _re.sub(r'\s+crossorigin(?:="[^"]*")?', "", html)
    return html.encode("utf-8")


def _inject_bootstrap(html_bytes: bytes, app: FastAPI):
    """Inject bootstrap credentials when password change is pending.

    Returns ``(html_bytes, script_nonce_or_None)``. Callers must forward
    the nonce via ``_CSP_SCRIPT_NONCE_HEADER`` so the inline script is
    not blocked by CSP.
    """
    import json as _json
    import secrets as _secrets

    if not storage.requires_password_change(storage.DEFAULT_ADMIN_USERNAME):
        return html_bytes, None

    bootstrap_pw = getattr(app.state, "bootstrap_password", None)
    if not bootstrap_pw:
        return html_bytes, None

    payload = _json.dumps(
        {
            "username": storage.DEFAULT_ADMIN_USERNAME,
            "password": bootstrap_pw,
        }
    )
    nonce = _secrets.token_urlsafe(16)
    tag = f'<script nonce="{nonce}">window.__UNSLOTH_BOOTSTRAP__={payload}</script>'
    html = html_bytes.decode("utf-8")
    html = html.replace("</head>", f"{tag}</head>", 1)
    return html.encode("utf-8"), nonce


def setup_frontend(app: FastAPI, build_path: Path):
    """Mount frontend static files (optional)"""
    if not build_path.exists():
        return False

    # Mount assets
    assets_dir = build_path / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory = assets_dir), name = "assets")

    def _build_index_response() -> Response:
        content = (build_path / "index.html").read_bytes()
        content = _strip_crossorigin(content)
        content, nonce = _inject_bootstrap(content, app)
        headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}
        if nonce:
            headers[_CSP_SCRIPT_NONCE_HEADER] = nonce
        return Response(
            content = content,
            media_type = "text/html",
            headers = headers,
        )

    @app.get("/")
    async def serve_root():
        return _build_index_response()

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        if full_path in {"api", "v1"} or full_path.startswith(("api/", "v1/")):
            return {"error": "API endpoint not found"}

        file_path = (build_path / full_path).resolve()

        # Block path traversal — ensure resolved path stays inside build_path
        if not file_path.is_relative_to(build_path.resolve()):
            return Response(status_code = 403)

        if file_path.is_file():
            return FileResponse(file_path)

        # Serve index.html as bytes — avoids Content-Length mismatch
        return _build_index_response()

    return True
