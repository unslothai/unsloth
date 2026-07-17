# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Main FastAPI application for Unsloth UI Backend
"""

import os
import sys
import threading
from pathlib import Path as _Path
import asyncio
from dataclasses import asdict

from typing import Any, Optional

# Suppress C-level dependency warnings globally
os.environ["PYTHONWARNINGS"] = "ignore"

# Pin GPU index ordering to PCI bus id before any torch import creates a CUDA
# context. Without this, torch/CUDA default to FASTEST_FIRST while nvidia-smi
# (and Studio's VRAM probes) use PCI-bus order, so a GPU index chosen from
# nvidia-smi data can resolve to a different physical card via
# CUDA_VISIBLE_DEVICES. setdefault so an explicit user override wins. See
# utils/hardware/hardware.py for the full rationale; set here too so the entry
# process is covered before its heavy ML imports.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# Windows terminals default to the active system code page. Reconfigure
# stdout/stderr before the startup banner so non-ASCII output cannot crash the
# backend process.
if sys.platform == "win32":
    for _win_stream in (sys.stdout, sys.stderr):
        if _win_stream is not None and hasattr(_win_stream, "reconfigure"):
            try:
                _win_stream.reconfigure(encoding = "utf-8", errors = "replace")
            except Exception:
                pass
    del _win_stream

_SYSTEM_GPU_CACHE_TTL_SECONDS = 10.0
_system_gpu_cache_lock = threading.Lock()
_system_gpu_cache: Optional[tuple[float, dict[str, Any]]] = None

# ── Windows AMD ROCm DLL injection ──────────────────────────────────────────
# Python 3.8+ ignores PATH for extension modules; register ROCm bin dirs with
# os.add_dll_directory() so amdhip64.dll etc. are found before any torch import.
if sys.platform == "win32":
    # Retained at module scope; os.add_dll_directory returns a handle that
    # removes the search-path entry when garbage collected.
    _ROCM_DLL_HANDLES: list = []

    def _add_rocm_dll_dirs() -> None:
        candidates = []
        # 1. HIP_PATH / ROCM_PATH set by the AMD HIP SDK installer
        for _var in ("HIP_PATH", "ROCM_PATH"):
            _val = os.environ.get(_var)
            if _val:
                candidates.append(os.path.join(_val, "bin"))
        # 2. AMD installer: C:\Program Files\AMD\ROCm\<ver>\bin, newest first.
        _default_root = os.path.join(
            os.environ.get("ProgramFiles", r"C:\Program Files"), "AMD", "ROCm"
        )

        def _ver_key(name: str) -> tuple:
            # Numeric tuple key so "10.0" sorts after "7.0"; non-numeric chunks fall back to string
            parts = []
            for chunk in name.split("."):
                try:
                    parts.append((0, int(chunk)))
                except ValueError:
                    parts.append((1, chunk))
            return tuple(parts)

        try:
            if os.path.isdir(_default_root):
                for _ver in sorted(os.listdir(_default_root), key = _ver_key, reverse = True):
                    _bin = os.path.join(_default_root, _ver, "bin")
                    if os.path.isdir(_bin):
                        candidates.append(_bin)
        except OSError:
            pass
        for _d in candidates:
            if os.path.isdir(_d):
                try:
                    _ROCM_DLL_HANDLES.append(os.add_dll_directory(_d))
                except (OSError, AttributeError):
                    pass

    _add_rocm_dll_dirs()
    del _add_rocm_dll_dirs

    # ── Windows AMD ROCm: make hipInfo.exe resolvable for subprocess probes ──
    # bitsandbytes' get_rocm_gpu_arch() runs `hipinfo.exe` via PATH at import
    # time; the AMD torch wheel ships it in the venv Scripts dir, which is on
    # PATH only when the venv is activated -- Studio launches python directly.
    # Without this, every bitsandbytes import logs a scary (but harmless)
    # "Could not detect ROCm GPU architecture: [WinError 2]" ERROR + WARNING.
    # Gated on the file existing: only AMD ROCm wheels ship hipInfo.exe, so
    # NVIDIA/CPU hosts are untouched. os.add_dll_directory above does not help
    # here -- subprocess PATH resolution ignores DLL search directories.
    _scripts_dir = os.path.dirname(sys.executable)
    if os.path.isfile(os.path.join(_scripts_dir, "hipInfo.exe")):
        import shutil as _shutil
        if not _shutil.which("hipinfo.exe"):
            os.environ["PATH"] = _scripts_dir + os.pathsep + os.environ.get("PATH", "")
        del _shutil
    del _scripts_dir

    # ── Windows AMD ROCm: set BNB_ROCM_VERSION before any bitsandbytes import ─
    # bitsandbytes derives the rocm<ver>.dll name from torch.version.hip, but the
    # wheel ships rocm72.dll, so the server crashes ("Configured ROCm binary not
    # found") without this. Detect the shipped DLL (mirrors worker.py); gate on
    # the rocm bnb DLL rather than torch.version.hip to avoid importing torch on
    # every Windows host.
    # Values seeded by the installer's sitecustomize.py are redetectable
    # defaults; explicit caller values remain authoritative.
    if (
        "BNB_ROCM_VERSION" not in os.environ
        or os.environ.get("UNSLOTH_BNB_ROCM_VERSION_SOURCE") == "sitecustomize"
    ):
        import glob as _glob
        import logging as _logging

        _bnb_rocm_ver = None
        _found_rocm_bnb = False
        try:
            import importlib.util as _ilu
            _bnb_spec = _ilu.find_spec("bitsandbytes")
            # submodule_search_locations (not spec.origin) handles editable installs
            if _bnb_spec and _bnb_spec.submodule_search_locations:
                import re as _re_bnb

                _all_vers_main: list[str] = []
                for _pkg_dir in _bnb_spec.submodule_search_locations:
                    for _dll in _glob.glob(os.path.join(_pkg_dir, "libbitsandbytes_rocm*.dll")):
                        _found_rocm_bnb = True
                        _km = _re_bnb.search(
                            r"libbitsandbytes_rocm(\d+)\.dll", os.path.basename(_dll)
                        )
                        if _km:
                            _all_vers_main.append(_km.group(1))
                if _all_vers_main:
                    _bnb_rocm_ver = max(_all_vers_main, key = lambda v: int(v))
        except Exception as _e:
            _logging.getLogger(__name__).warning(
                "Windows ROCm: BNB DLL detection failed (%s); leaving BNB_ROCM_VERSION as is",
                _e,
            )
        # Only when a ROCm bnb DLL actually exists: HIP_PATH/ROCM_PATH alone
        # (HIP SDK on a CUDA/CPU box) must not force a ROCm backend onto a
        # non-ROCm bitsandbytes, which raises at import. DLL unparsable -> "72".
        if _found_rocm_bnb:
            _bnb_rocm_ver_final = _bnb_rocm_ver or os.environ.get("BNB_ROCM_VERSION") or "72"
            os.environ["BNB_ROCM_VERSION"] = _bnb_rocm_ver_final
            os.environ["UNSLOTH_BNB_ROCM_VERSION_SOURCE"] = "detected"
            _logging.getLogger(__name__).info(
                "Windows ROCm: set BNB_ROCM_VERSION=%s (from installed BNB wheel)",
                _bnb_rocm_ver_final,
            )

    # Setting BNB_ROCM_VERSION makes bitsandbytes log a benign override notice on
    # import; drop only that record so real errors and mismatch warnings show.
    if os.environ.get("BNB_ROCM_VERSION"):
        import logging as _logging
        _logging.getLogger("bitsandbytes.cextension").addFilter(
            lambda _r: "environment variable detected" not in _r.getMessage()
        )

# ── WSL AMD Strix Halo (gfx1151): enable ROCDXG before any torch import ──────
# In WSL the AMD GPU is reached via the ROCDXG bridge (librocdxg.so over
# /dev/dxg), which HSA loads only when HSA_ENABLE_DXG_DETECTION=1 is set BEFORE
# torch touches the GPU. A worker launched outside a login shell (e.g.
# `wsl.exe -d Ubuntu-24.04 python ...`) misses the installer's persisted env
# and silently falls back to CPU. Set it here, gated to no-op unless BOTH
# /dev/dxg AND librocdxg.so exist -- native Linux ROCm, NVIDIA, macOS and
# Windows are unaffected.
elif sys.platform.startswith("linux") and "HSA_ENABLE_DXG_DETECTION" not in os.environ:
    try:
        if os.path.exists("/dev/dxg") and any(
            os.path.exists(os.path.join(_p, "librocdxg.so"))
            for _p in ("/opt/rocm/lib", "/opt/rocm/lib64")
        ):
            os.environ["HSA_ENABLE_DXG_DETECTION"] = "1"
            import logging as _logging
            _logging.getLogger(__name__).info(
                "WSL ROCm: set HSA_ENABLE_DXG_DETECTION=1 (librocdxg bridge present)"
            )
    except Exception:
        pass

# Put backend dir on sys.path so _platform_compat is importable when main.py
# is launched directly (e.g. `uvicorn main:app`).
_backend_dir = str(_Path(__file__).parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

# `uvicorn main:app` bypasses run.py; seed thread caps here too.
from utils.cpu_threads import configure_cpu_threads

try:
    configure_cpu_threads()
except ValueError as exc:
    _raw = os.environ.get("UNSLOTH_CPU_THREADS")
    raise SystemExit(f"Error: Invalid UNSLOTH_CPU_THREADS value {_raw!r}: {exc}") from None

# Anaconda/conda-forge Python: seed platform._sys_version_cache before any
# library import triggers attrs -> rich -> structlog -> platform crash.
# See: https://github.com/python/cpython/issues/102396
import _platform_compat  # noqa: F401

# Direct `uvicorn main:app` launches bypass run.py, so re-export here too
# (mirrors run.py). Required BEFORE the unsloth-zoo import below, whose
# LLAMA_CPP_DEFAULT_DIR binding is import-time.
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

# The studio bundles unsloth_zoo; declare unsloth present (as `import unsloth`
# does) so its lazy submodule imports (export, hardware, mlx) and the
# DiffusionGemma runner never trip the install guard on a clean install.
os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

import hashlib
import ipaddress
import mimetypes
import re as _re
import shutil
import warnings
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version as package_version
from urllib.parse import urlparse


_STUDIO_INSTALL_ID_RE = _re.compile(r"^[0-9a-f]{64}$")


def _read_studio_install_id() -> str:
    """Per-install opaque id at $STUDIO_HOME/share/studio_install_id.

    Returns "" when absent or not a 64-char lowercase-hex token; then
    /api/health emits "" and the launcher accepts any healthy backend.
    Carries no install-path info (matters when Studio runs -H 0.0.0.0)."""
    try:
        token = (_STUDIO_ROOT_RESOLVED / "share" / "studio_install_id").read_text().strip()
    except (OSError, ValueError):
        return ""
    return token if _STUDIO_INSTALL_ID_RE.fullmatch(token) else ""


_STUDIO_ROOT_ID_CACHE: str = _read_studio_install_id()


def _studio_root_id() -> str:
    """Same-install discriminator for /api/health (cached at import).

    Empty when no installer token is present; the launcher treats "" as
    "accept any healthy backend"."""
    return _STUDIO_ROOT_ID_CACHE


# Fix broken Windows registry MIME types: some installs map .js to text/plain,
# which mimetypes (hence StaticFiles) inherits and browsers reject for ES
# modules. add_type() before StaticFiles forces correct types.
if sys.platform == "win32":
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")

# Suppress dependency warnings in production
if os.getenv("ENVIRONMENT_TYPE", "production") == "production":
    warnings.filterwarnings("ignore")
    # Or be more specific:
    # warnings.filterwarnings("ignore", category=DeprecationWarning)
    # warnings.filterwarnings("ignore", module="triton.*")

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, Response
from pathlib import Path
from datetime import datetime

from routes import (
    auth_router,
    chat_history_router,
    data_recipe_router,
    datasets_router,
    export_router,
    inference_router,
    inference_studio_router,
    mcp_servers_router,
    models_router,
    providers_router,
    rag_router,
    training_history_router,
    training_router,
)
from routes.llama import router as llama_router
from routes.preview import router as preview_router
from hub.routes import (
    inventory_router as hub_inventory_router,
    datasets_router as hub_datasets_router,
)
from hub.schemas.downloads import TransportCapabilities
from hub.utils.download_registry import (
    get_download_transport_capabilities,
    reap_orphan_workers as reap_hub_orphan_workers,
    terminate_active_downloads as terminate_hub_downloads,
)
from routes.settings import router as settings_router
from routes.prompts import router as prompts_router
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
from utils.lifespan_shutdown import run_lifespan_shutdown
from utils.native_path_leases import native_path_leases_supported
from utils.update_status import (
    get_studio_install_source_status,
    get_studio_update_status,
)
from utils.studio_version import get_studio_version
from utils.api_errors import install_api_error_handlers


def get_unsloth_version() -> str:
    try:
        return package_version("unsloth")
    except PackageNotFoundError:
        pass

    version_file = _Path(__file__).resolve().parents[2] / "unsloth" / "models" / "_utils.py"
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

# The Tauri desktop app runs the backend on the owner's own machine, so local
# stdio MCP servers are safe there. setdefault lets an explicit "0" opt out.
if _DESKTOP_OWNER:
    os.environ.setdefault("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")


def _desktop_owner() -> dict[str, str] | None:
    return _DESKTOP_OWNER


def _start_helper_precache_if_enabled() -> None:
    """Start optional Helper LLM GGUF pre-cache only after explicit opt-in."""
    try:
        from utils.helper_precache_settings import should_preload_helper_on_startup
        if not should_preload_helper_on_startup():
            return
    except Exception:
        return

    import threading

    def _precache():
        try:
            from utils.datasets.llm_assist import precache_helper_gguf
            precache_helper_gguf()
        except Exception:
            pass  # non-critical

    threading.Thread(target = _precache, daemon = True, name = "helper-gguf-precache").start()


def _run_llama_cpp_startup_probes(app: FastAPI) -> None:
    """llama.cpp capability (MTP support) + freshness (release age) probes.

    Runs OFF the startup critical path (see _start_llama_cpp_probes_if_enabled).
    Both are cached and freshness has a 24h disk TTL, but on a cold/expired cache
    the freshness check makes a blocking GitHub request, and on macOS the first
    `llama-server --help` exec can stall on Gatekeeper verification -- neither must
    ever gate `Application startup complete`. Writes app.state only; nothing reads
    those values synchronously at startup (the status routes call
    check_prebuilt_freshness directly at request time), so populating them late is
    safe.
    """
    try:
        from core.inference.llama_cpp import LlamaCppBackend
        from utils.llama_cpp_freshness import (
            check_prebuilt_freshness,
            format_stale_warning,
        )

        _bin = LlamaCppBackend._find_llama_server_binary()
        _caps = LlamaCppBackend.probe_server_capabilities(_bin)
        app.state.llama_cpp_capabilities = _caps
        _freshness = check_prebuilt_freshness(_bin)
        app.state.llama_cpp_freshness = _freshness

        import structlog as _structlog

        _log = _structlog.get_logger(__name__)
        if _caps.get("found") and not _caps.get("supports_mtp"):
            _msg = (
                "llama.cpp prebuilt lacks MTP support "
                "(--spec-type mtp/draft-mtp). Run `unsloth studio update`. "
                "MTP GGUFs will load without speculative decoding."
            )
            _log.warning(_msg)
            print(f"WARNING: {_msg}", flush = True)
        if _freshness.get("stale"):
            _msg = format_stale_warning(_freshness)
            _log.warning(_msg)
            print(f"WARNING: {_msg}", flush = True)
    except Exception as _probe_exc:
        import structlog as _structlog
        _structlog.get_logger(__name__).debug("llama.cpp startup probes failed: %s", _probe_exc)


def _start_llama_cpp_probes_if_enabled(app: FastAPI) -> None:
    """Run the llama.cpp startup probes on a daemon thread, off the startup
    critical path so they never delay `Application startup complete`. Skipped
    entirely when update checks are disabled, so a fully offline boot makes no
    background network calls."""
    if os.environ.get("UNSLOTH_DISABLE_UPDATE_CHECK") == "1":
        return

    threading.Thread(
        target = _run_llama_cpp_startup_probes,
        args = (app,),
        daemon = True,
        name = "llama-cpp-startup-probe",
    ).start()


def _warm_rag_embedder() -> None:
    """Warm RAG embeddings without blocking backend readiness."""
    try:
        from storage import rag_db

        if not rag_db.RAG_AVAILABLE:
            return
        from core.rag import embeddings

        embeddings.warm()
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: detect hardware, seed default admin if needed. Shutdown: clean up compiled cache."""

    import time as _time

    _lifespan_started = _time.perf_counter()
    import structlog as _structlog

    _lifespan_log = _structlog.get_logger(__name__)
    clear_unsloth_compiled_cache()

    # Remove stale .venv_overlay from old versions; switching now uses .venv_t5/.
    overlay_dir = Path(__file__).resolve().parent.parent.parent / ".venv_overlay"
    if overlay_dir.is_dir():
        shutil.rmtree(overlay_dir, ignore_errors = True)

    # Detect hardware first — sets the DEVICE global used everywhere.
    detect_hardware()

    _lifespan_log.info(
        "lifespan hardware detection completed in %.1fms",
        (_time.perf_counter() - _lifespan_started) * 1000,
    )

    # Apple Silicon with MLX missing => Train/Export are greyed out (chat-only).
    # Reinstall mlx by name on a background thread (off the critical path) and
    # re-detect, so a reinstall/update that dropped mlx self-heals. No-op
    # elsewhere; opt out with UNSLOTH_DISABLE_MLX_AUTOREPAIR=1.
    try:
        from utils.mlx_repair import start_mlx_autorepair_if_needed
        start_mlx_autorepair_if_needed()
    except Exception as _mlx_exc:
        import structlog as _structlog
        _structlog.get_logger(__name__).debug("mlx autorepair skipped: %s", _mlx_exc)

    # Reap workers/runs orphaned by a previous crash before new work starts.
    try:
        from storage.studio_db import cleanup_orphaned_runs
        cleanup_orphaned_runs()
    except Exception as exc:
        _lifespan_log.warning("cleanup_orphaned_runs failed at startup: %s", exc)

    reap_hub_orphan_workers()

    # llama.cpp probes: capability (MTP support) + freshness (release age).
    # These used to run inline here and could block `Application startup complete`
    # for tens of seconds on macOS (cold GitHub freshness cache / slow network, and
    # Gatekeeper verifying the unsigned binary on first `--help` exec). They only
    # write app.state and nothing reads it synchronously at startup, so run them on
    # a daemon thread off the startup critical path (mirrors the helper-precache and
    # RAG-warm threads). Default to None until the thread populates them.
    app.state.llama_cpp_capabilities = None
    app.state.llama_cpp_freshness = None
    _start_llama_cpp_probes_if_enabled(app)

    try:
        from storage.rag_db import reconcile_orphaned_ingestion_jobs
        reconcile_orphaned_ingestion_jobs()
    except Exception as exc:
        _lifespan_log.warning("reconcile_orphaned_ingestion_jobs failed at startup: %s", exc)

    _start_helper_precache_if_enabled()
    threading.Thread(target = _warm_rag_embedder, daemon = True, name = "rag-embedder-warm").start()

    # Idle auto-unload loop (no-op unless the OpenAI auto-unload TTL is set).
    from core.inference.llama_keepwarm import idle_unload_loop

    app.state.idle_unload_task = asyncio.create_task(idle_unload_loop())

    # Initialize RSA key pair for API key encryption (external providers).
    from core.inference.key_exchange import init_key_pair

    init_key_pair()
    _lifespan_log.info(
        "lifespan pre-auth setup completed in %.1fms",
        (_time.perf_counter() - _lifespan_started) * 1000,
    )

    # run_server's pre-bind gate sets suppress_bootstrap_injection when a public
    # URL is about to serve with the default credential active: never (re)capture
    # the bootstrap password into app.state, or the HTML would hand it out.
    _suppress_bootstrap = getattr(app.state, "suppress_bootstrap_injection", False)
    if storage.ensure_default_admin():
        bootstrap_pw = None if _suppress_bootstrap else storage.get_bootstrap_password()
        app.state.bootstrap_password = bootstrap_pw

        bootstrap_path = storage.DB_PATH.parent / ".bootstrap_password"
        print("\n" + "=" * 60)
        print("DEFAULT ADMIN ACCOUNT CREATED")
        print(f"    username: {storage.DEFAULT_ADMIN_USERNAME}")
        print(f"    password saved to: {bootstrap_path}")
        print("    Open the Studio UI to sign in and change it.")
        print("=" * 60 + "\n")
    else:
        app.state.bootstrap_password = (
            None if _suppress_bootstrap else storage.get_bootstrap_password()
        )

    _lifespan_log.info(
        "lifespan startup completed in %.1fms",
        (_time.perf_counter() - _lifespan_started) * 1000,
    )
    yield

    _idle_task = getattr(app.state, "idle_unload_task", None)
    if _idle_task is not None:
        _idle_task.cancel()
        try:
            await _idle_task
        except asyncio.CancelledError:
            pass

    from core.inference.llama_http import aclose as _close_llama_http

    await _close_llama_http()

    await run_lifespan_shutdown(
        terminate_hub_downloads,
        clear_unsloth_compiled_cache,
        _hw_module,
    )


app = FastAPI(
    title = "Unsloth UI Backend",
    version = UNSLOTH_VERSION,
    description = "Backend API for Unsloth UI - Training and Model Management",
    lifespan = lifespan,
)

from loggers.config import LogConfig
from loggers.handlers import LoggingMiddleware

logger = LogConfig.setup_logging(
    service_name = "unsloth-studio-backend",
    env = os.getenv("ENVIRONMENT_TYPE", "production"),
)

app.add_middleware(LoggingMiddleware)


# img/media-src allow any https origin so HF model-card assets render (mirrors
# tauri.conf.json); scripts/frames/connect-src stay same-origin + HF.
from starlette.datastructures import MutableHeaders  # noqa: E402


_CSP_SCRIPT_NONCE_HEADER = "x-internal-script-nonce"
_ARTIFACT_PREVIEW_FRAME_PATH = "/api/inference/artifact-preview-frame"


# /content is Colab's working directory — more reliable than env vars, which
# aren't always set depending on Colab runtime version.
import importlib.util as _importlib_util

_IS_COLAB = os.path.isdir("/content") and (
    bool(os.environ.get("COLAB_BACKEND_URL"))
    or bool(os.environ.get("COLAB_JUPYTER_IP"))
    or _importlib_util.find_spec("google.colab") is not None
)


def _build_csp(script_nonce: "str | None" = None) -> str:
    script_src = "script-src 'self'"
    if script_nonce:
        script_src += f" 'nonce-{script_nonce}'"
    # Colab parent frames span multi-level *.prod.colab.dev subdomains (CSP
    # wildcards match one level only) and null-origin iframes; use '*' since
    # Colab is already a sandboxed single-user environment.
    frame_ancestors = "*" if _IS_COLAB else "'none'"

    # In Colab, the kernel/output scaffolding injects scripts and fetch/WS from
    # *.prod.colab.dev and *.googleusercontent.com, so widen script-src and
    # connect-src for those. Scripts still use a nonce, not 'unsafe-inline'.
    if _IS_COLAB:
        script_src += " https://*.prod.colab.dev https://*.googleusercontent.com"
        connect_src = (
            "'self' blob: data: "
            "https://huggingface.co https://datasets-server.huggingface.co "
            "https://*.prod.colab.dev wss://*.prod.colab.dev "
            "https://*.googleusercontent.com wss://*.googleusercontent.com"
        )
    else:
        connect_src = "'self' https://huggingface.co https://datasets-server.huggingface.co"

    return (
        "default-src 'self'; "
        "img-src 'self' data: blob: https:; "
        "media-src 'self' data: blob: https:; "
        f"connect-src {connect_src}; "
        "style-src 'self' 'unsafe-inline'; "
        f"{script_src}; "
        "font-src 'self' data:; "
        "frame-src 'self'; "
        f"frame-ancestors {frame_ancestors}; "
        "form-action 'self'; "
        "base-uri 'self'"
    )


class SecurityHeadersMiddleware:
    """Set baseline security headers; splice per-response inline-script nonces into CSP.

    Pure ASGI (not BaseHTTPMiddleware) so streaming responses are not wrapped in
    an anyio stream. Header logic mirrors the prior version exactly via
    MutableHeaders on the response-start message.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # ASGI headers are an iterable; coerce to a list so MutableHeaders
                # can mutate in place even if a server sends a tuple or omits it.
                raw = message.setdefault("headers", [])
                if not isinstance(raw, list):
                    raw = list(raw)
                    message["headers"] = raw
                headers = MutableHeaders(raw = raw)
                # Strip the internal nonce hand-off header so it never reaches the client
                nonce = headers.get(_CSP_SCRIPT_NONCE_HEADER)
                if nonce is not None:
                    del headers[_CSP_SCRIPT_NONCE_HEADER]
                headers.setdefault("Content-Security-Policy", _build_csp(nonce))
                # Omit X-Frame-Options in Colab: CSP frame-ancestors handles it, and
                # DENY would block serve_kernel_port_as_iframe regardless of CSP.
                if not _IS_COLAB and path != _ARTIFACT_PREVIEW_FRAME_PATH:
                    headers.setdefault("X-Frame-Options", "DENY")
                headers.setdefault("X-Content-Type-Options", "nosniff")
                headers.setdefault("Referrer-Policy", "no-referrer")
                headers.setdefault(
                    "Permissions-Policy",
                    "camera=(), microphone=(self), geolocation=()",
                )
                headers["server"] = "unsloth-studio"
            await send(message)

        await self.app(scope, receive, send_wrapper)


app.add_middleware(SecurityHeadersMiddleware)


# Cap request bodies on protected POSTs. Upload routes get explicit multipart
# headroom; non-upload routes keep the default body cap.
import json as _json_for_413  # noqa: E402
from utils.upload_limits import (  # noqa: E402
    UNSTRUCTURED_RECIPE_UPLOAD_MAX_BYTES,
    default_request_body_limit_bytes,
    upload_request_limit_bytes,
)

_BODY_PROTECTED_PREFIXES = (
    "/v1/chat/completions",
    "/v1/completions",
    "/p/",
    "/api/inference",
    "/api/data-recipe",
    "/api/datasets",
    "/api/hub",
    "/api/chat",
    "/api/settings",
    "/api/train",
    "/api/export",
)
_DATASET_UPLOAD_PASSTHROUGH_PREFIX = "/api/datasets/upload"
_DATA_RECIPE_UNSTRUCTURED_UPLOAD_PASSTHROUGH_PREFIX = (
    "/api/data-recipe/seed/upload-unstructured-file"
)
_BODY_UPLOAD_PASSTHROUGH_PREFIXES = (
    _DATASET_UPLOAD_PASSTHROUGH_PREFIX,
    _DATA_RECIPE_UNSTRUCTURED_UPLOAD_PASSTHROUGH_PREFIX,
)


def _get_upload_passthrough_request_max_bytes(path: str) -> int:
    if path.startswith(_DATA_RECIPE_UNSTRUCTURED_UPLOAD_PASSTHROUGH_PREFIX):
        return upload_request_limit_bytes(UNSTRUCTURED_RECIPE_UPLOAD_MAX_BYTES)
    if path.startswith(_DATASET_UPLOAD_PASSTHROUGH_PREFIX):
        return upload_request_limit_bytes()
    return default_request_body_limit_bytes()


async def _send_411(send) -> None:
    payload = _json_for_413.dumps(
        {"detail": "Content-Length required for upload requests."},
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 411,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload, "more_body": False})


async def _send_413(send, total_bytes: int, max_bytes: int) -> None:
    payload = _json_for_413.dumps(
        {"detail": (f"Request body too large ({total_bytes:,} bytes; max {max_bytes:,}).")},
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

    def __init__(
        self,
        app,
        max_bytes_getter,
        protected_prefixes: tuple,
        upload_passthrough_prefixes: tuple = (),
        upload_passthrough_max_bytes_getter = None,
    ):
        self.app = app
        self.max_bytes_getter = max_bytes_getter
        self.protected_prefixes = protected_prefixes
        self.upload_passthrough_prefixes = upload_passthrough_prefixes
        self.upload_passthrough_max_bytes_getter = upload_passthrough_max_bytes_getter

    def _upload_passthrough_max_bytes(self, path: str) -> int:
        if self.upload_passthrough_max_bytes_getter is None:
            return int(self.max_bytes_getter())
        try:
            return int(self.upload_passthrough_max_bytes_getter(path))
        except TypeError:
            try:
                return int(self.upload_passthrough_max_bytes_getter())
            except Exception:
                return int(self.max_bytes_getter())
        except Exception:
            return int(self.max_bytes_getter())

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

        max_bytes = int(self.max_bytes_getter())
        declared = None
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value.decode("latin-1"))
                except (ValueError, UnicodeDecodeError):
                    declared = None
                break

        if any(path.startswith(p) for p in self.upload_passthrough_prefixes):
            upload_max_bytes = self._upload_passthrough_max_bytes(path)
            if declared is None:
                await _send_411(send)
                return
            if declared > upload_max_bytes:
                await _send_413(send, declared, upload_max_bytes)
                return
            await self.app(scope, receive, send)
            return

        if declared is not None and declared > max_bytes:
            await _send_413(send, declared, max_bytes)
            return

        chunks: list = []
        total = 0
        while True:
            msg = await receive()
            mtype = msg.get("type")
            if mtype == "http.disconnect":
                return
            if mtype != "http.request":
                # Mid-stream unexpected frame: forwarding would corrupt downstream
                return
            body = msg.get("body", b"") or b""
            if body:
                total += len(body)
                if total > max_bytes:
                    await _send_413(send, total, max_bytes)
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
    max_bytes_getter = default_request_body_limit_bytes,
    protected_prefixes = _BODY_PROTECTED_PREFIXES,
    upload_passthrough_prefixes = _BODY_UPLOAD_PASSTHROUGH_PREFIXES,
    upload_passthrough_max_bytes_getter = _get_upload_passthrough_request_max_bytes,
)

# Tracks in-flight inference requests for idle auto-unload; off -> passthrough.
from core.inference.llama_keepwarm import LlamaKeepWarmMiddleware  # noqa: E402

app.add_middleware(LlamaKeepWarmMiddleware)


from starlette.responses import RedirectResponse as _RedirectResponse  # noqa: E402


@app.get("/recipes", include_in_schema = False)
@app.get("/recipes/{rest:path}", include_in_schema = False)
async def _recipes_redirect(rest: str = ""):
    target = "/data-recipes" + (("/" + rest) if rest else "")
    return _RedirectResponse(url = target, status_code = 308)


from utils.host_policy import cors_origins_for_mode  # noqa: E402

_cors_origins = cors_origins_for_mode(
    api_only = os.environ.get("UNSLOTH_API_ONLY") == "1",
    secure = os.environ.get("UNSLOTH_SECURE") == "1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = _cors_origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)


# ============ Register API Routes ============

# Register routers
app.include_router(auth_router, prefix = "/api/auth", tags = ["auth"])
app.include_router(training_router, prefix = "/api/train", tags = ["training"])
app.include_router(models_router, prefix = "/api/models", tags = ["models"])
app.include_router(chat_history_router, prefix = "/api/chat", tags = ["chat"])
app.include_router(inference_router, prefix = "/api/inference", tags = ["inference"])
# Studio-only inference endpoints (cancel, etc.) are NOT exposed on the /v1
# OpenAI-compat prefix below.
app.include_router(inference_studio_router, prefix = "/api/inference", tags = ["inference"])

# OpenAI-compatible: mount the inference router at /v1 for external tools.
app.include_router(inference_router, prefix = "/v1", tags = ["openai-compat"])
app.include_router(preview_router, prefix = "/p", tags = ["preview"])
app.include_router(providers_router, prefix = "/api/providers", tags = ["providers"])
app.include_router(settings_router, prefix = "/api/settings", tags = ["settings"])
app.include_router(mcp_servers_router, prefix = "/api/mcp/servers", tags = ["mcp"])
app.include_router(prompts_router, prefix = "/api/prompts", tags = ["prompts"])
app.include_router(datasets_router, prefix = "/api/datasets", tags = ["datasets"])
app.include_router(data_recipe_router, prefix = "/api/data-recipe", tags = ["data-recipe"])
app.include_router(llama_router, prefix = "/api/llama", tags = ["llama"])
app.include_router(export_router, prefix = "/api/export", tags = ["export"])
app.include_router(rag_router, prefix = "/api/rag", tags = ["rag"])
app.include_router(training_history_router, prefix = "/api/train", tags = ["training-history"])
app.include_router(hub_inventory_router, prefix = "/api/hub", tags = ["hub"])
app.include_router(hub_datasets_router, prefix = "/api/hub/datasets", tags = ["hub"])

# Re-wrap client-error responses on the /v1/* surface into OpenAI/Anthropic
# error envelopes; non-/v1 paths keep FastAPI's default {"detail": ...} shape.
install_api_error_handlers(app)


# ============ Health and System Endpoints ============


@app.get("/api/liveness")
async def liveness_check():
    """Cheap process liveness for desktop port validation."""
    return {
        "status": "alive",
        "service": "Unsloth UI Backend",
        "desktop_protocol_version": 1,
        "desktop_manageability_version": 1,
        "supports_desktop_auth": True,
        "supports_desktop_backend_ownership": True,
        "studio_root_id": _studio_root_id(),
        **({"desktop_owner": owner} if (owner := _desktop_owner()) else {}),
    }


@app.get("/api/health")
async def health_check(request: Request):
    """Liveness plus launcher capability bits; host fingerprint gated on a bearer.

    Unauthenticated callers get non-sensitive fields (service, studio_root_id,
    chat_only, desktop_*, native_path_leases_supported) to re-adopt a sibling
    backend and gate UI before a token exists. version / studio_version /
    device_type require a bearer since they fingerprint the host.
    """
    base = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Unsloth UI Backend",
        "chat_only": _hw_module.CHAT_ONLY,
        "desktop_protocol_version": 1,
        "desktop_manageability_version": 1,
        "supports_desktop_auth": True,
        "supports_desktop_backend_ownership": True,
        # Opaque per-install id; launchers reject sibling Studios on the same port.
        "studio_root_id": _studio_root_id(),
        "native_path_leases_supported": native_path_leases_supported(),
        **({"desktop_owner": owner} if (owner := _desktop_owner()) else {}),
    }
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return base
    try:
        from auth.authentication import get_current_subject as _gcs
        from fastapi.security import HTTPAuthorizationCredentials

        creds = HTTPAuthorizationCredentials(scheme = "Bearer", credentials = auth.split(" ", 1)[1])
        # Must await: a bare coroutine is truthy and would skip the auth check
        subject = await _gcs(creds)
    except HTTPException:
        return base
    except Exception:
        return base
    if not subject:
        return base

    platform_map = {"darwin": "mac", "win32": "windows", "linux": "linux"}
    device_type = platform_map.get(sys.platform, sys.platform)
    return {
        **base,
        # Why chat_only is set. This fingerprints the host, so keep it authed.
        "chat_only_reason": getattr(_hw_module, "CHAT_ONLY_REASON", None),
        "version": UNSLOTH_VERSION,
        "studio_version": STUDIO_VERSION,
        "device_type": device_type,
        # API-screen fields (authed-only; they fingerprint how the host is exposed).
        "cloudflare_url": getattr(request.app.state, "cloudflare_url", None),
        "server_url": getattr(request.app.state, "server_url", None),
        "secure": bool(getattr(request.app.state, "secure", False)),
    }


@app.get("/api/studio/install-source")
def studio_install_source(_current_subject: str = Depends(get_current_subject)):
    """Return source-aware install metadata without remote update checks."""
    return get_studio_install_source_status(UNSLOTH_VERSION)


@app.get("/api/studio/update-status")
def studio_update_status(_current_subject: str = Depends(get_current_subject)):
    """Return source-aware manual update status for browser-served Studio."""
    return get_studio_update_status(UNSLOTH_VERSION)


@app.get(
    "/api/studio/download-transport-capabilities",
    response_model = TransportCapabilities,
)
def studio_download_transport_capabilities(_current_subject: str = Depends(get_current_subject)):
    return asdict(get_download_transport_capabilities())


@app.post("/api/shutdown")
async def shutdown_server(request: Request, current_subject: str = Depends(get_current_subject)):
    """Gracefully shut down the Unsloth Studio server.

    Called by the frontend quit dialog so users can stop the server from the UI
    without the CLI or killing the process manually.
    """

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


def _get_cached_system_gpu_info(logger) -> dict[str, Any]:
    """Return merged GPU visibility/utilization with bounded live-probe churn."""
    import time
    from utils.hardware import get_backend_visible_gpu_info, get_visible_gpu_utilization

    global _system_gpu_cache
    now = time.monotonic()
    with _system_gpu_cache_lock:
        if _system_gpu_cache is not None:
            cached_at, cached_gpu_info = _system_gpu_cache
            if now - cached_at < _SYSTEM_GPU_CACHE_TTL_SECONDS:
                return cached_gpu_info

        try:
            visibility_info = get_backend_visible_gpu_info() or {"available": False, "devices": []}
        except Exception as e:
            logger.debug(f"Failed to get GPU visibility info: {e}")
            visibility_info = {"available": False, "devices": []}

        try:
            utilization_info = get_visible_gpu_utilization() or {"devices": []}
        except Exception as e:
            logger.debug(f"Failed to get GPU utilization info: {e}")
            utilization_info = {"devices": []}

        util_devices = {d.get("index"): d for d in utilization_info.get("devices", [])}
        enriched_devices = []

        for dev in visibility_info.get("devices", []):
            idx = dev.get("index")
            util = util_devices.get(idx, {})

            total_vram = util.get("vram_total_gb") or dev.get("memory_total_gb") or 0
            used_vram = util.get("vram_used_gb") or 0

            enriched_dev = dict(dev)
            enriched_dev["vram_used_gb"] = used_vram
            enriched_dev["vram_free_gb"] = round(total_vram - used_vram, 2) if total_vram else 0
            enriched_dev["vram_utilization_pct"] = util.get("vram_utilization_pct")
            enriched_devices.append(enriched_dev)

        gpu_info = {
            "available": visibility_info.get("available", False),
            "devices": enriched_devices,
        }
        _system_gpu_cache = (time.monotonic(), gpu_info)
        return gpu_info


@app.get("/api/system")
def get_system_info(current_subject: str = Depends(get_current_subject)):
    """Get system information.

    Auth-gated: the response (platform, Python/GPU, memory, ML packages) can
    fingerprint a host, which matters in -H 0.0.0.0 / Colab / Tauri-relayed
    setups where remote callers can reach /api/system.
    """
    import platform
    import psutil
    import os
    import time
    import logging
    from utils.hardware import get_device, export_capability
    from utils.hardware.hardware import _backend_label

    logger = logging.getLogger(__name__)

    gpu_info = _get_cached_system_gpu_info(logger)

    memory = psutil.virtual_memory()

    try:
        cpu_freq = psutil.cpu_freq()
    except Exception as e:
        logger.debug(f"Failed to get CPU frequency: {e}")
        cpu_freq = None

    try:
        disk = psutil.disk_usage(os.path.abspath(os.sep))
    except Exception as e:
        logger.debug(f"Failed to get disk usage: {e}")
        disk = None

    try:
        current_process = psutil.Process(os.getpid())
        process_used_mb = round(current_process.memory_info().rss / 1024**2)
    except Exception as e:
        logger.debug(f"Failed to get current process memory: {e}")
        process_used_mb = 0

    try:
        boot_time = psutil.boot_time()
    except Exception as e:
        logger.debug(f"Failed to get boot time: {e}")
        boot_time = None

    # Read versions from metadata so a 3s poll never imports heavy ML libs (or 500s on their import errors).
    from importlib.metadata import PackageNotFoundError, version as pkg_version

    ml_packages = {}
    for pkg in ("torch", "transformers"):
        try:
            ml_packages[pkg] = pkg_version(pkg)
        except PackageNotFoundError:
            pass
        except Exception as e:
            logger.debug(f"Failed to read {pkg} version: {e}")

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "device_backend": _backend_label(get_device()),
        "cpu_count": psutil.cpu_count(logical = True),
        "uptime_seconds": max(0, round(time.time() - boot_time)) if boot_time else None,
        "cpu": {
            "logical_count": psutil.cpu_count(logical = True),
            "physical_count": psutil.cpu_count(logical = False),
            "usage_percent": psutil.cpu_percent(interval = None),
            "frequency_mhz": round(cpu_freq.current, 2)
            if cpu_freq and cpu_freq.current is not None
            else None,
        },
        "memory": {
            "total_gb": round(memory.total / 1024**3, 2),
            "available_gb": round(memory.available / 1024**3, 2),
            "percent_used": memory.percent,
            "process_used_mb": process_used_mb,
        },
        "disk": {
            "total_gb": round(disk.total / 1e9, 2) if disk else 0,
            "free_gb": round(disk.free / 1e9, 2) if disk else 0,
            "percent_used": disk.percent if disk else 0,
        },
        "gpu": gpu_info,
        "ml_packages": ml_packages,
        # Export capability + torch-aware reason. See /api/system/hardware.
        **export_capability(),
    }


@app.get("/api/system/gpu-visibility")
async def get_gpu_visibility(current_subject: str = Depends(get_current_subject)):
    return get_backend_visible_gpu_info()


@app.get("/api/system/hardware")
def get_hardware_info(
    include_details: bool = Query(False), current_subject: str = Depends(get_current_subject)
):
    """Return GPU name, total VRAM, and key ML package versions.

    Gated behind auth alongside /api/system -- same fingerprinting concern.
    /api/system/gpu-visibility is also auth-gated.

    ``include_details`` is for About/diagnostics. The default response stays
    cheap for callers that only need the primary GPU summary, like training
    method auto-selection. Sync def (not async): hardware/detail probes can
    shell out, and FastAPI runs sync endpoints in a threadpool.
    """
    from utils.hardware import get_gpu_summary, get_package_versions, export_capability

    body = {
        "gpu": get_gpu_summary(),
        "versions": get_package_versions(),
        # Export capability + torch-aware reason; the Export UI grays out with the message.
        **export_capability(),
    }
    if include_details:
        from utils.llama_cpp_update import get_installed_llama_version

        # All backend-visible GPUs (respects CUDA_VISIBLE_DEVICES), so multi-GPU
        # hosts list every device -- get_gpu_summary alone reports only the primary.
        # Sort by visible_ordinal: the nvidia-smi path returns rows in physical order,
        # so under a reordering CUDA_VISIBLE_DEVICES (e.g. "5,3") labeling by array
        # index would otherwise disagree with the GPU 0/1 the backend actually sees.
        devices = get_backend_visible_gpu_info().get("devices", [])
        body["gpus"] = [
            {"name": d.get("name"), "vram_total_gb": d.get("memory_total_gb")}
            for d in sorted(devices, key = lambda d: d.get("visible_ordinal", 0))
        ]
        body["llama_cpp"] = get_installed_llama_version()
    return body


# ============ Serve Frontend (Optional) ============


def _strip_crossorigin(html_bytes: bytes) -> bytes:
    """Remove ``crossorigin`` attributes from script/link tags.

    Vite's default ``crossorigin`` forces CORS mode on font loads, which
    Firefox HTTPS-Only Mode breaks over plain HTTP; stripping it makes them
    same-origin fetches that work on any protocol.
    """
    html = html_bytes.decode("utf-8")
    html = _re.sub(r'\s+crossorigin(?:="[^"]*")?', "", html)
    return html.encode("utf-8")


def _inject_bootstrap(html_bytes: bytes, app: FastAPI):
    """Inject bootstrap credentials when password change is pending.
    Returns ``(html_bytes, script_nonce_or_None)``; callers forward the nonce
    via ``_CSP_SCRIPT_NONCE_HEADER`` so CSP allows the inline script.
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


_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}


def _canonical_origin(scheme: str, netloc: str) -> Optional[tuple[str, str, int]]:
    """Canonicalise an Origin to ``(scheme, host, port)`` for equality.
    Browsers strip default ports (RFC 6454 sec 6.1) and scheme/host are
    case-insensitive (RFC 3986), so a bare string compare misclassifies
    same-origin requests as cross-origin. Returns ``None`` on unparseable input
    so callers fall to the safer cross-origin default.
    """
    scheme = (scheme or "").strip().lower()
    if not scheme or not netloc:
        return None
    # Strip userinfo (RFC 3986); Origin never carries credentials.
    if "@" in netloc:
        netloc = netloc.rsplit("@", 1)[1]
    # IPv6 hosts use brackets (RFC 3986 sec 3.2.2): ``[::1]:8902``. Bare
    # ``partition(":")`` mis-parses these, breaking ``unsloth studio -H ::1``.
    if netloc.startswith("["):
        close = netloc.find("]")
        if close == -1:
            return None
        host = netloc[1:close]
        rest = netloc[close + 1 :]
        if rest.startswith(":"):
            port_str = rest[1:]
        elif rest == "":
            port_str = ""
        else:
            return None
    else:
        host, _, port_str = netloc.partition(":")
    host = host.strip().lower()
    if not host:
        return None
    if port_str:
        try:
            port = int(port_str)
        except ValueError:
            return None
    else:
        port = _DEFAULT_PORTS.get(scheme, 0)
    return (scheme, host, port)


def _is_loopback_ip(host: Optional[str]) -> bool:
    """Return whether ``host`` is a loopback IP, including IPv4-mapped IPv6."""
    if not host or "%" in host:  # a scope id (::1%eth0) is never a plain loopback
        return False
    try:
        ip = ipaddress.ip_address(host)
    except (TypeError, ValueError):
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    return ip.is_loopback or (mapped is not None and mapped.is_loopback)


# A loopback peer carrying any of these is a proxy/tunnel relaying a remote
# client, so the peer is the proxy, not the caller: cloudflared sets
# cf-connecting-ip, reverse proxies set the rest (uvicorn only consumes
# x-forwarded-for, so the others survive to here).
_PROXIED_CLIENT_HEADERS = (
    "cf-connecting-ip",
    "forwarded",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-real-ip",
)


def _host_header_is_loopback(host_header: Optional[str]) -> bool:
    """Loopback/localhost check on the raw Host header.

    Reads the header directly so a malformed or absent Host cannot fall back to
    ``request.url.hostname``'s (loopback) ASGI server address.
    """
    if not host_header:
        return False
    host = host_header.strip()
    if host.startswith("["):  # [IPv6] or [IPv6]:port
        end = host.find("]")
        if end == -1 or (host[end + 1 :] and not host[end + 1 :].startswith(":")):
            return False  # unclosed bracket or junk after ] (e.g. [::1]evil)
        host = host[1:end]
    elif host.count(":") == 1:  # host:port
        host = host.split(":", 1)[0]
    host = host.lower().rstrip(".")
    return host == "localhost" or _is_loopback_ip(host)


def _is_local_bootstrap_request(request: Request) -> bool:
    """Allow bootstrap injection only through a direct loopback authority."""
    client = request.client
    if client is None or not _is_loopback_ip(client.host):
        return False
    if any(request.headers.get(h) is not None for h in _PROXIED_CLIENT_HEADERS):
        return False
    return _host_header_is_loopback(request.headers.get("host"))


def _is_same_origin_request(request: Request) -> bool:
    """True when Origin is missing or matches request's scheme://host:port.

    Missing Origin counts as same-origin (top-level GETs omit it). Both sides
    are canonicalised via :func:`_canonical_origin`; callers must emit
    ``Vary: Origin``.
    """
    origin = request.headers.get("origin")
    if origin is None:
        # Missing header: top-level same-document GETs omit Origin.
        return True
    # Empty string is not a valid serialised origin (RFC 6454 sec 6.1).
    if not origin:
        return False
    # "null" token (sandboxed iframes, file:// pages) is never same-origin.
    if origin == "null":
        return False
    # ``urlparse`` raises ``ValueError`` on malformed IPv6 brackets; swallow
    # so a garbage Origin doesn't 500 the SPA handler.
    try:
        parsed = urlparse(origin)
    except ValueError:
        return False
    origin_canon = _canonical_origin(parsed.scheme, parsed.netloc)
    if origin_canon is None:
        return False
    try:
        self_canon = _canonical_origin(request.url.scheme, request.url.netloc)
    except ValueError:
        return False
    if self_canon is None:
        return False
    return origin_canon == self_canon


def _should_inject_bootstrap(request: Request) -> bool:
    """Whether to embed the seeded bootstrap password in index.html."""
    if not _is_same_origin_request(request):
        return False
    if _IS_COLAB:
        # Single-user notebook proxy: allow autofill, but never a public
        # shareable tunnel (a Colab Cloudflare link sets cf-connecting-ip).
        return request.headers.get("cf-connecting-ip") is None
    return _is_local_bootstrap_request(request)


def setup_frontend(app: FastAPI, build_path: Path):
    """Mount frontend static files (optional)"""
    if not build_path.exists():
        return False

    assets_dir = build_path / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory = assets_dir), name = "assets")

    def _build_index_response(request: Request) -> Response:
        content = (build_path / "index.html").read_bytes()
        content = _strip_crossorigin(content)
        # Bootstrap pw goes only to a same-origin, direct-loopback client (or
        # Colab's single-user notebook proxy): a wildcard bind must not serve it
        # in-page to a LAN or proxied peer. Vary: Origin keeps caches honest.
        if _should_inject_bootstrap(request):
            content, nonce = _inject_bootstrap(content, app)
        else:
            nonce = None
        headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Vary": "Origin",
        }
        if nonce:
            headers[_CSP_SCRIPT_NONCE_HEADER] = nonce
        return Response(
            content = content,
            media_type = "text/html",
            headers = headers,
        )

    @app.get("/")
    async def serve_root(request: Request):
        return _build_index_response(request)

    @app.get("/{full_path:path}")
    async def serve_frontend(request: Request, full_path: str):
        # Unknown API paths: raise a real 404 so the api_errors handlers can
        # render the correct envelope for /v1/* (and {"detail":...} for /api/*).
        # This handler only sees paths NOT matched by a real route. The full
        # request path is "/" + full_path.
        if full_path in {"api", "v1"} or full_path.startswith(("api/", "v1/")):
            raise HTTPException(status_code = 404, detail = "API endpoint not found")

        file_path = (build_path / full_path).resolve()

        # Block path traversal — resolved path must stay inside build_path
        if not file_path.is_relative_to(build_path.resolve()):
            return Response(status_code = 403)

        if file_path.is_file():
            return FileResponse(file_path)

        # Serve index.html as bytes — avoids Content-Length mismatch
        return _build_index_response(request)

    return True
