# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Main FastAPI application for Unsloth UI Backend
"""

import os
import sys
import threading
import time
from pathlib import Path as _Path
import asyncio
from dataclasses import asdict

from typing import Any, Optional

os.environ["PYTHONWARNINGS"] = "ignore"

# Pin GPU index ordering to PCI bus id before any torch import creates a CUDA context.
# Otherwise torch/CUDA default to FASTEST_FIRST while nvidia-smi (and Unsloth's VRAM
# probes) use PCI-bus order, so an index chosen from nvidia-smi can resolve to a different
# card. setdefault so an override wins; full rationale in utils/hardware/hardware.py.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# Windows terminals default to the active system code page. Reconfigure stdout/stderr
# before the startup banner so non-ASCII output cannot crash the backend process.
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
_system_gpu_cache: Optional[tuple[float, tuple[dict[str, Any], dict[str, Any]]]] = None

# ── Windows AMD ROCm DLL injection ──────────────────────────────────────────
# Python 3.8+ ignores PATH for extension modules; register ROCm bin dirs with
# os.add_dll_directory() so amdhip64.dll etc. are found before any torch import.
if sys.platform == "win32":
    # Module scope: the handle removes the search-path entry when garbage collected.
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
    # bitsandbytes' get_rocm_gpu_arch() runs `hipinfo.exe` via PATH at import time; the AMD
    # torch wheel ships it in the venv Scripts dir, which is on PATH only when the venv is
    # activated -- Unsloth launches python directly. Without this every bitsandbytes import
    # logs a scary (harmless) "Could not detect ROCm GPU architecture" error. Gated on the
    # file existing, so non-AMD hosts are untouched; subprocess PATH ignores DLL dirs.
    _scripts_dir = os.path.dirname(sys.executable)
    if os.path.isfile(os.path.join(_scripts_dir, "hipInfo.exe")):
        import shutil as _shutil
        if not _shutil.which("hipinfo.exe"):
            os.environ["PATH"] = _scripts_dir + os.pathsep + os.environ.get("PATH", "")
        del _shutil
    del _scripts_dir

    # ── Windows AMD ROCm: set BNB_ROCM_VERSION before any bitsandbytes import ─
    # bitsandbytes derives the rocm<ver>.dll name from torch.version.hip, but the wheel ships
    # rocm72.dll, so the server crashes ("Configured ROCm binary not found") without this.
    # Detect the shipped DLL (mirrors worker.py); gate on it rather than torch.version.hip to
    # avoid importing torch. Installer-seeded values are defaults; caller values win.
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
        # Only when a ROCm bnb DLL actually exists: HIP_PATH/ROCM_PATH alone (HIP SDK on a
        # CUDA/CPU box) must not force a ROCm backend. Unparsable DLL name -> "72".
        if _found_rocm_bnb:
            _bnb_rocm_ver_final = _bnb_rocm_ver or os.environ.get("BNB_ROCM_VERSION") or "72"
            os.environ["BNB_ROCM_VERSION"] = _bnb_rocm_ver_final
            os.environ["UNSLOTH_BNB_ROCM_VERSION_SOURCE"] = "detected"
            _logging.getLogger(__name__).info(
                "Windows ROCm: set BNB_ROCM_VERSION=%s (from installed BNB wheel)",
                _bnb_rocm_ver_final,
            )

    # Setting BNB_ROCM_VERSION makes bitsandbytes log a benign override notice; drop that record only.
    if os.environ.get("BNB_ROCM_VERSION"):
        import logging as _logging
        _logging.getLogger("bitsandbytes.cextension").addFilter(
            lambda _r: "environment variable detected" not in _r.getMessage()
        )

# ── WSL AMD Strix Halo (gfx1151): enable ROCDXG before any torch import ──────
# In WSL the AMD GPU is reached via the ROCDXG bridge (librocdxg.so over /dev/dxg), which
# HSA loads only when HSA_ENABLE_DXG_DETECTION=1 is set BEFORE torch touches the GPU. A
# worker launched outside a login shell misses the installer's persisted env and falls
# back to CPU. Gated on both /dev/dxg and librocdxg.so, so other platforms no-op.
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

# Backend dir on sys.path so _platform_compat imports under `uvicorn main:app`.
_backend_dir = str(_Path(__file__).parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

# OS trust store for TLS before anything opens a connection: behind a
# TLS-inspecting proxy certifi alone rejects every Hub request.
from utils.native_tls import activate_native_tls

activate_native_tls()

# `uvicorn main:app` bypasses run.py; seed thread caps here too.
from utils.cpu_threads import configure_cpu_threads

try:
    configure_cpu_threads()
except ValueError as exc:
    _raw = os.environ.get("UNSLOTH_CPU_THREADS")
    raise SystemExit(f"Error: Invalid UNSLOTH_CPU_THREADS value {_raw!r}: {exc}") from None

# Anaconda/conda-forge Python: seed platform._sys_version_cache before attrs -> rich ->
# structlog -> platform crashes. See https://github.com/python/cpython/issues/102396
import _platform_compat  # noqa: F401

# Direct `uvicorn main:app` bypasses run.py, so re-export here too. Required BEFORE the
# unsloth-zoo import below, whose LLAMA_CPP_DEFAULT_DIR binding is import-time.
from utils.paths.storage_roots import studio_root as _studio_root

# Same reason, same deadline: unsloth_zoo.compiler reads UNSLOTH_COMPILE_LOCATION
# at import time, and without this a direct start falls back to a CWD-relative
# unsloth_compiled_cache (on Windows that is the user profile).
from utils.paths.storage_roots import setup_cache_env as _setup_cache_env

try:
    _setup_cache_env()
except Exception:  # noqa: BLE001
    pass

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
    _MANAGED_LLAMA_CPP_PATH = _STUDIO_ROOT_RESOLVED / "llama.cpp"
    if not os.environ.get("UNSLOTH_LLAMA_CPP_PATH"):
        os.environ["UNSLOTH_LLAMA_CPP_PATH"] = str(_MANAGED_LLAMA_CPP_PATH)
    # A CLI/desktop launcher may already have exported Unsloth's own install path.
    # Classify by the canonical value so that inherited default remains editable.
    from utils.llama_cpp_path_settings import mark_managed_llama_cpp_path

    mark_managed_llama_cpp_path(_MANAGED_LLAMA_CPP_PATH)

# The studio bundles unsloth_zoo; declare unsloth present (as `import unsloth` does) so its
# lazy submodule imports and the DiffusionGemma runner don't trip the install guard.
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
    Carries no install-path info (matters when Unsloth runs -H 0.0.0.0)."""
    try:
        token = (
            (_STUDIO_ROOT_RESOLVED / "share" / "studio_install_id")
            .read_text(encoding = "utf-8")
            .strip()
        )
    except (OSError, ValueError):
        return ""
    return token if _STUDIO_INSTALL_ID_RE.fullmatch(token) else ""


_STUDIO_ROOT_ID_CACHE: str = _read_studio_install_id()


def _studio_root_id() -> str:
    """Same-install discriminator for /api/health (cached at import).

    Empty when no installer token is present; the launcher treats "" as
    "accept any healthy backend"."""
    return _STUDIO_ROOT_ID_CACHE


# Some Windows installs map .js to text/plain, which mimetypes (hence StaticFiles) inherits
# and browsers reject for ES modules. add_type() before StaticFiles forces correct types.
if sys.platform == "win32":
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")

# Suppress dependency warnings in production
if os.getenv("ENVIRONMENT_TYPE", "production") == "production":
    warnings.filterwarnings("ignore")

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, Response
from starlette.middleware.gzip import GZipMiddleware
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
    openai_codex_auth_router,
    rag_router,
    research_runs_router,
    chat_generation_runs_router,
    training_history_router,
    training_router,
    video_router,
    youtube_router,
)
from routes.llama import router as llama_router
from routes.whisper import router as whisper_router
from routes.preview import router as preview_router
from hub.routes import (
    inventory_router as hub_inventory_router,
    datasets_router as hub_datasets_router,
    token_router as hub_token_router,
)
from picker.routes import templates_router as picker_templates_router
from hub.schemas.downloads import TransportCapabilities
from hub.utils.download_registry import (
    get_download_transport_capabilities,
    reap_orphan_workers as reap_hub_orphan_workers,
    terminate_active_downloads as terminate_hub_downloads,
)
from routes.settings import router as settings_router
from routes.prompts import router as prompts_router
from routes.profile_stats import router as profile_stats_router
from auth import storage
from auth.authentication import get_current_subject
from utils.hardware import (
    start_background_detection,
    get_device,
    DeviceType,
    get_backend_visible_gpu_info,
)
import utils.hardware.hardware as _hw_module

from utils.torch_warmup import (
    DISABLE_ENV_VAR,
    join_background_warm,
    reset_background_warm,
    start_background_warm,
    warm_status,
)
from utils.cache_cleanup import (
    clear_compiled_cache_unless_shared as _clear_compiled_cache_unless_shared,
)
from utils.lifespan_shutdown import run_lifespan_shutdown
from utils.native_path_leases import native_path_leases_supported
from utils.update_status import (
    get_studio_install_source_status,
    get_studio_update_status,
)
from utils.release_notes import get_release_notes, is_supported_version_query
from utils.studio_version import get_studio_version
from utils.api_errors import install_api_error_handlers


def get_unsloth_version() -> str:
    try:
        return package_version("unsloth")
    except PackageNotFoundError:
        pass

    # Both files: the literal moved to _version.py, and models/_utils.py now holds only a
    # re-export, which this prefix scan does not match. Trying both keeps a half-updated
    # tree reporting a real version instead of falling through to "dev".
    root = _Path(__file__).resolve().parents[2] / "unsloth"
    for version_file in (root / "_version.py", root / "models" / "_utils.py"):
        try:
            for line in version_file.read_text(encoding = "utf-8").splitlines():
                if line.startswith("__version__ = "):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except (OSError, UnicodeDecodeError):
            continue
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

# The Tauri desktop app runs the backend locally, so stdio MCP servers are safe ("0" opts
# out). Tracked as an automatic loopback default so publishing a runtime tunnel can suspend
# it without overriding an explicit operator choice.
if _DESKTOP_OWNER:
    from utils.host_policy import apply_stdio_mcp_loopback_default as _apply_desktop_stdio_default
    _apply_desktop_stdio_default("127.0.0.1")
    del _apply_desktop_stdio_default


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
        if (
            _caps.get("found")
            and not _caps.get("supports_mtp")
            and not _caps.get("mtp_probe_inconclusive")
        ):
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


_post_warm_thread: Optional[threading.Thread] = None
_post_warm_lock = threading.Lock()
# Bumped by every start and stop. A worker captures the value it started with and stops once
# it no longer matches, so one parked in join_background_warm() cannot act after shutdown.
_post_warm_generation = 0


def _post_warm_current_generation() -> int:
    with _post_warm_lock:
        return _post_warm_generation


def _start_post_warm_thread() -> bool:
    """Put up a post-warm worker for this lifespan. True iff one was started.

    Starts one even while a previous worker is parked in the warm join. Declining there
    left a restart with no worker at all: the old one was alive so this returned early,
    then read the shutdown and exited. Generations make the overlap safe -- the stale
    worker drops out by itself and a parked thread is free.
    """
    global _post_warm_thread, _post_warm_generation
    with _post_warm_lock:
        _post_warm_generation += 1
        mine = _post_warm_generation
        thread = threading.Thread(
            target = _post_warm_background_work,
            args = (mine,),
            daemon = True,
            name = f"post-warm-{mine}",
        )
        _post_warm_thread = thread
    thread.start()
    return True


def _stop_post_warm_thread() -> None:
    """Retire whatever worker is current; never wait for it.

    Joining would hold shutdown for the rest of the ML stack import, the stall this path
    exists to avoid. Bumping the generation suffices: the worker re-reads it after its join.
    """
    global _post_warm_generation
    with _post_warm_lock:
        _post_warm_generation += 1


def _post_warm_retired(generation: Optional[int]) -> bool:
    """True when this post-warm worker's lifespan has ended. Logs once when it has.

    A mismatch means the application that wanted this work has stopped. The remaining
    work imports optional platform or RAG scheduling modules, so none of it may start for
    a stopped lifespan.
    """
    if generation is None or _post_warm_current_generation() == generation:
        return False
    import structlog as _structlog

    _structlog.get_logger(__name__).info(
        "post-warm work %s stood down: its lifespan ended while the ML stack was still loading",
        generation,
    )
    return True


def _start_linked_folder_auto_sync(generation: Optional[int]) -> None:
    # A real lifespan worker carries a generation; direct calls without one are tests.
    if generation is None:
        return
    try:
        from core.rag.folder_sync import start_auto_sync
        from storage.studio_db import get_chat_project
        start_auto_sync(
            admission_lock = _post_warm_lock,
            admit = lambda: _post_warm_generation == generation,
            project_exists = lambda project_id: get_chat_project(project_id) is not None,
        )
    except Exception as exc:
        import structlog as _structlog
        _structlog.get_logger(__name__).warning(
            "linked-folder auto-sync failed at startup: %s", exc
        )


def _post_warm_background_work(generation: Optional[int] = None) -> None:
    """Platform repair and linked-folder lifecycle work after the coordinated warm.

    MLX repair used to probe the runtime before the socket bound. Joining first keeps that
    optional probe out of the login-screen critical path. Linked-folder startup only loads
    embeddings when a queued sync has real ingestion work; an idle scheduler stays cold.
    """
    # No-op when the warm never started, so this is safe under the kill switch.
    join_background_warm()

    # Shutdown routinely lands while parked in the join above, and everything below imports or
    # loads part of the stack. Rechecked before every action; generation is None only in tests.
    if _post_warm_retired(generation):
        return

    # Apple Silicon with MLX missing => chat-only; reinstall mlx and re-detect so a dropped mlx
    # self-heals. Opt out with UNSLOTH_DISABLE_MLX_AUTOREPAIR=1; after the warm, the probe imports MLX.
    try:
        from utils.mlx_repair import start_mlx_autorepair_if_needed
        if _post_warm_retired(generation):
            return
        start_mlx_autorepair_if_needed()
    except Exception as _mlx_exc:
        import structlog as _structlog

        # Warning, not debug: this decides the MLX verdict on every healthy Apple Silicon
        # boot, so a half-applied update (new mlx_repair.py over older hardware.py) arrives
        # here and would silently leave Train/Export greyed out for the session.
        _structlog.get_logger(__name__).warning("mlx autorepair skipped: %s", _mlx_exc)

    if _post_warm_retired(generation):
        return
    _start_linked_folder_auto_sync(generation)


def clear_compiled_cache_unless_shared(app: FastAPI) -> None:
    """Clear the compiled cache unless a sibling backend of this install is live.

    The decision lives in cache_cleanup, next to the paths it clears and the lock
    that serializes it against a sibling's startup; run_server puts the probe on
    app.state because main.py must not import run.py back.
    """
    _clear_compiled_cache_unless_shared(getattr(app.state, "live_sibling_backend", None))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: detect hardware, seed default admin if needed. Shutdown: clean up compiled cache."""

    import time as _time

    _lifespan_started = _time.perf_counter()
    import structlog as _structlog

    _lifespan_log = _structlog.get_logger(__name__)
    clear_compiled_cache_unless_shared(app)

    # Move the legacy sandbox up here rather than from the first request: the
    # copy can be minutes when the studio home is on another filesystem.
    try:
        from core.inference.tools import (
            migrate_legacy_sandbox_in_background,
            start_sandbox_recovery,
        )
        migrate_legacy_sandbox_in_background()
        # A tree renamed for deletion by a run that was killed, and the
        # workspace deletes it left pending: both waited for the next Python or
        # terminal call, which ordinary chat never makes.
        start_sandbox_recovery()
    except Exception:  # noqa: BLE001
        pass

    # Remove stale .venv_overlay from old versions; switching now uses .venv_t5/.
    overlay_dir = Path(__file__).resolve().parent.parent.parent / ".venv_overlay"
    if overlay_dir.is_dir():
        shutil.rmtree(overlay_dir, ignore_errors = True)

    # Hardware detection and MLX autorepair moved out of this lifespan: both import heavy
    # runtimes and uvicorn binds only once this returns, so they held the login screen.

    # Reap workers/runs orphaned by a previous crash before new work starts.
    try:
        from storage.studio_db import cleanup_orphaned_runs
        cleanup_orphaned_runs()
    except Exception as exc:
        _lifespan_log.warning("cleanup_orphaned_runs failed at startup: %s", exc)

    try:
        from storage.chat_generation_runs_db import reconcile_orphaned_runs
        reconciled_chat_runs = reconcile_orphaned_runs()
        if reconciled_chat_runs:
            _lifespan_log.warning(
                "Marked %s interrupted chat generation run(s) failed after restart.",
                reconciled_chat_runs,
            )
    except Exception as exc:
        _lifespan_log.warning("chat generation orphan reconciliation failed: %s", exc)

    reap_hub_orphan_workers()
    try:
        from hub.utils.download_manifest import migrate_ordinary_v2_manifests_for_downgrade
        migrated_manifests = migrate_ordinary_v2_manifests_for_downgrade()
        if migrated_manifests:
            _lifespan_log.info(
                "Migrated %s Hub download manifest(s) for downgrade compatibility.",
                migrated_manifests,
            )
    except Exception as exc:
        _lifespan_log.warning("Hub manifest compatibility migration failed: %s", exc)

    # llama.cpp probes: capability (MTP support) + freshness (release age). Inline they could
    # block `Application startup complete` for tens of seconds on macOS (cold GitHub cache,
    # Gatekeeper verifying the unsigned binary). Nothing reads them synchronously at startup,
    # so run them on a daemon thread; app.state stays None until it populates them.
    app.state.llama_cpp_capabilities = None
    app.state.llama_cpp_freshness = None
    _start_llama_cpp_probes_if_enabled(app)

    try:
        from storage.rag_db import reconcile_orphaned_ingestion_jobs
        reconcile_orphaned_ingestion_jobs()
    except Exception as exc:
        _lifespan_log.warning("reconcile_orphaned_ingestion_jobs failed at startup: %s", exc)

    # Embeddings stay cold until ingestion or retrieval actually requests vectors.
    _start_helper_precache_if_enabled()

    from core.research_runs import ResearchSupervisor

    app.state.research_supervisor = ResearchSupervisor(app)
    app.state.research_supervisor.start()

    from core.inference.chat_generation_runs import ChatGenerationSupervisor

    app.state.chat_generation_supervisor = ChatGenerationSupervisor(app)

    # Idle auto-unload loop (no-op unless the OpenAI auto-unload TTL is set).
    from core.inference.llama_keepwarm import idle_unload_loop, sweep_slot_save_dir

    sweep_slot_save_dir()
    app.state.idle_unload_task = asyncio.create_task(idle_unload_loop())

    # Initialize RSA key pair for API key encryption (external providers).
    from core.inference.key_exchange import init_key_pair

    init_key_pair()
    _lifespan_log.info(
        "lifespan pre-auth setup completed in %.1fms",
        (_time.perf_counter() - _lifespan_started) * 1000,
    )

    # run_server's pre-bind gate sets suppress_bootstrap_injection when a public URL is about
    # to serve with the default credential: never capture the bootstrap password into app.state.
    _suppress_bootstrap = getattr(app.state, "suppress_bootstrap_injection", False)
    if storage.ensure_default_admin():
        bootstrap_pw = None if _suppress_bootstrap else storage.get_bootstrap_password()
        app.state.bootstrap_password = bootstrap_pw

        bootstrap_path = storage.DB_PATH.parent / ".bootstrap_password"
        print("\n" + "=" * 60)
        print("DEFAULT ADMIN ACCOUNT CREATED")
        print(f"    username: {storage.DEFAULT_ADMIN_USERNAME}")
        print(f"    password saved to: {bootstrap_path}")
        print("    Open the Unsloth UI to sign in and change it.")
        print("=" * 60 + "\n")
    else:
        app.state.bootstrap_password = (
            None if _suppress_bootstrap else storage.get_bootstrap_password()
        )

    # Last, so it never contends for the GIL: the socket binds as soon as this returns, so the
    # login screen is up while torch/transformers/datasets load.
    start_background_warm()
    _start_post_warm_thread()

    _lifespan_log.info(
        "lifespan startup completed in %.1fms",
        (_time.perf_counter() - _lifespan_started) * 1000,
    )
    # Persist only terminal scalar usage from authenticated third-party API
    # requests. The monitor itself stays storage-agnostic until the production
    # lifespan is ready, which keeps imports and unit tests deterministic.
    from core.inference.api_monitor import api_monitor as _api_monitor
    from storage.api_usage_db import (
        acquire_api_usage_writer as _acquire_api_usage_writer,
        enqueue_api_usage as _enqueue_api_usage,
        release_api_usage_writer as _release_api_usage_writer,
    )

    _api_usage_writer_lease = _acquire_api_usage_writer()
    _api_usage_callback_lease = _api_monitor.acquire_terminal_callback(_enqueue_api_usage)
    yield

    # Remove only this lifespan's callback. A concurrently live sibling keeps
    # both the monitor sink and the shared serialized writer. The final owner
    # drains accepted receipts off the event loop before stopping the worker.
    _api_monitor.release_terminal_callback(_api_usage_callback_lease)

    # Before any shutdown await: a warm finishing during one would still read the lifespan as current.
    _stop_post_warm_thread()

    # Retire the coordinated warm at shutdown entry too. run_lifespan_shutdown() repeats
    # this after cleanup, but its awaits would otherwise let startup imports continue for
    # a lifespan that has already stopped.
    _invalidate_detection = getattr(_hw_module, "invalidate_detection", None)
    if _invalidate_detection is not None:
        _invalidate_detection()

    await asyncio.to_thread(_release_api_usage_writer, _api_usage_writer_lease)

    from core.inference.openai_codex_auth import shutdown_flows

    await shutdown_flows()
    try:
        from core.rag.folder_sync import stop_auto_sync
        stop_auto_sync()
    except Exception as exc:
        _lifespan_log.warning("linked-folder auto-sync failed at shutdown: %s", exc)

    _idle_task = getattr(app.state, "idle_unload_task", None)
    if _idle_task is not None:
        _idle_task.cancel()
        try:
            await _idle_task
        except asyncio.CancelledError:
            pass

    _research_supervisor = getattr(app.state, "research_supervisor", None)
    if _research_supervisor is not None:
        await _research_supervisor.stop()

    _chat_generation_supervisor = getattr(app.state, "chat_generation_supervisor", None)
    if _chat_generation_supervisor is not None:
        await _chat_generation_supervisor.stop()

    from core.inference.llama_http import aclose as _close_llama_http

    await _close_llama_http()

    await run_lifespan_shutdown(
        terminate_hub_downloads,
        lambda: clear_compiled_cache_unless_shared(app),
        _hw_module,
    )
    # Shutdown cleared the state this warm produced, so release the one-per-process latch.
    reset_background_warm()


app = FastAPI(
    title = "Unsloth UI Backend",
    version = UNSLOTH_VERSION,
    description = "Backend API for Unsloth UI - Training and Model Management",
    lifespan = lifespan,
    # Swagger UI and ReDoc are re-registered below on these same paths, against vendored
    # assets instead of a CDN. FastAPI's built-ins point at cdn.jsdelivr.net, and this origin
    # holds the auth tokens, so nothing third-party may execute here.
    docs_url = None,
    redoc_url = None,
    swagger_ui_oauth2_redirect_url = None,
)
app.state.secure = os.environ.get("UNSLOTH_SECURE") == "1"

# The MCP surface is opt-in: it can start GPU jobs and write model artifacts.
if os.environ.get("UNSLOTH_STUDIO_ENABLE_MCP") == "1":
    from fastmcp.utilities.lifespan import combine_lifespans

    from mcp_server import BearerTokenMiddleware, create_studio_mcp

    _studio_mcp_app = create_studio_mcp().http_app(path = "/")
    _studio_mcp_lifespan = _studio_mcp_app.lifespan
    _mcp_token = os.environ.get("UNSLOTH_STUDIO_MCP_TOKEN")
    if not _mcp_token:
        raise RuntimeError("UNSLOTH_STUDIO_MCP_TOKEN is required when MCP is enabled")
    _studio_mcp_app = BearerTokenMiddleware(_studio_mcp_app, _mcp_token)
    app.router.lifespan_context = combine_lifespans(lifespan, _studio_mcp_lifespan)
    app.mount("/mcp", _studio_mcp_app)

from loggers.config import LogConfig
from loggers.handlers import LoggingMiddleware

logger = LogConfig.setup_logging(
    service_name = "unsloth-studio-backend",
    env = os.getenv("ENVIRONMENT_TYPE", "production"),
)

app.add_middleware(LoggingMiddleware)


class ResearchPortMiddleware:
    """Capture the bound port without replacing the ASGI receive channel."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request_app = scope.get("app")
            supervisor = getattr(getattr(request_app, "state", None), "research_supervisor", None)
            if supervisor is not None:
                supervisor.note_server_port(scope.get("server"))
        await self.app(scope, receive, send)


app.add_middleware(ResearchPortMiddleware)


# img/media-src allow any https origin so HF model-card assets render (mirrors
# tauri.conf.json); scripts/frames/connect-src stay same-origin + HF.
from starlette.datastructures import MutableHeaders  # noqa: E402


_CSP_SCRIPT_NONCE_HEADER = "x-internal-script-nonce"
_ARTIFACT_PREVIEW_FRAME_PATH = "/api/inference/artifact-preview-frame"
_DOCS_FONT_CSS = "https://fonts.googleapis.com"
_DOCS_FONT_FILES = "https://fonts.gstatic.com"
_DOCS_PATHS = frozenset({"/docs", "/docs/oauth2-redirect", "/redoc"})
_DOCS_ASSETS_URL = "/docs-assets"
_DOCS_ASSETS_DIR = Path(__file__).parent / "assets" / "docs_ui"


# /content is Colab's working directory -- more reliable than env vars.
import importlib.util as _importlib_util

_IS_COLAB = os.path.isdir("/content") and (
    bool(os.environ.get("COLAB_BACKEND_URL"))
    or bool(os.environ.get("COLAB_JUPYTER_IP"))
    or _importlib_util.find_spec("google.colab") is not None
)


def _build_csp(script_nonce: "str | None" = None, *, docs: bool = False) -> str:
    script_src = "script-src 'self'"
    style_src = "style-src 'self' 'unsafe-inline'"
    worker_src = "worker-src 'self'"
    font_src = "font-src 'self' data:"
    if docs:
        # script-src is deliberately untouched: the docs bundles are served from this origin
        # and their inline init runs off the nonce. What is left cannot execute script, only
        # style and lay out the page. ReDoc's Google Fonts sheet pulls faces from gstatic, and
        # its search index runs in a worker it builds from a blob.
        style_src += f" {_DOCS_FONT_CSS}"
        font_src += f" {_DOCS_FONT_FILES}"
        worker_src += " blob:"
    if script_nonce:
        script_src += f" 'nonce-{script_nonce}'"
    # Colab parent frames span multi-level *.prod.colab.dev subdomains (CSP wildcards match
    # one level) and null-origin iframes; '*' is safe as Colab is a sandboxed single user.
    frame_ancestors = "*" if _IS_COLAB else "'none'"

    # In Colab the kernel scaffolding injects scripts and fetch/WS from *.prod.colab.dev and
    # *.googleusercontent.com, so widen script-src/connect-src. Scripts still use a nonce.
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
        f"{style_src}; "
        f"{script_src}; "
        f"{worker_src}; "
        f"{font_src}; "
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
                # ASGI headers are an iterable; coerce to a list so MutableHeaders can mutate in place.
                raw = message.setdefault("headers", [])
                if not isinstance(raw, list):
                    raw = list(raw)
                    message["headers"] = raw
                headers = MutableHeaders(raw = raw)
                # Strip the internal nonce hand-off header so it never reaches the client
                nonce = headers.get(_CSP_SCRIPT_NONCE_HEADER)
                if nonce is not None:
                    del headers[_CSP_SCRIPT_NONCE_HEADER]
                headers.setdefault(
                    "Content-Security-Policy",
                    _build_csp(nonce, docs = path in _DOCS_PATHS),
                )
                # Omit X-Frame-Options in Colab: DENY would block serve_kernel_port_as_iframe regardless of CSP.
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


# Swagger UI and ReDoc, on FastAPI's own paths but served entirely from this origin.
# FastAPI's built-in pages load ~2.3 MB of JavaScript from cdn.jsdelivr.net and start it with
# an inline script. localStorage is origin-scoped, not path-scoped, so anything running on
# /docs can read the Unsloth tokens session.ts keeps there and call the API as that user. The
# bundles are vendored under assets/docs_ui (pinned + digest-checked by
# tests/test_docs_ui_assets.py) and the inline init runs off the same per-response nonce the
# bootstrap script uses, so script-src stays 'self' and works offline as a bonus.
import secrets as _secrets_for_docs  # noqa: E402
from fastapi.openapi.docs import (  # noqa: E402
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)

# fastapi is unpinned, so match the opening tag by what follows it rather than by the
# surrounding whitespace and comment: a reflowed template must not 500 the page.
_SWAGGER_INIT_TAG = _re.compile(r"<script>(?=\s*const ui = SwaggerUIBundle)")
_OAUTH2_REDIRECT_TAG = _re.compile(r"<script>")


def _nonced_docs_response(html: str, *, tag: "_re.Pattern[str]") -> HTMLResponse:
    """Hand the page's own inline script a nonce; injected script never gets one."""
    nonce = _secrets_for_docs.token_urlsafe(16)
    nonced, replaced = tag.subn(f'<script nonce="{nonce}">', html, count = 1)
    if not replaced:
        # Upstream retemplated the page: fail loudly rather than serve a blank one.
        raise RuntimeError(f"docs template changed, inline script tag not found: {tag.pattern!r}")
    return HTMLResponse(nonced, headers = {_CSP_SCRIPT_NONCE_HEADER: nonce})


if _DOCS_ASSETS_DIR.is_dir():
    app.mount(
        _DOCS_ASSETS_URL,
        StaticFiles(directory = _DOCS_ASSETS_DIR),
        name = "docs-assets",
    )

    def _docs_url(request: Request, path: str) -> str:
        """Prefix with the mount point, as FastAPI's own docs routes do.

        Behind a path-stripping proxy (or `uvicorn --root-path`) the browser sees the prefix
        the server never does, so an unprefixed URL escapes the mapping and 404s.
        """
        return f"{request.scope.get('root_path', '').rstrip('/')}{path}"

    @app.get("/docs", include_in_schema = False)
    async def swagger_ui_html(request: Request):
        assets = _docs_url(request, _DOCS_ASSETS_URL)
        html = get_swagger_ui_html(
            openapi_url = _docs_url(request, app.openapi_url),
            title = f"{app.title} - Swagger UI",
            oauth2_redirect_url = _docs_url(request, "/docs/oauth2-redirect"),
            swagger_js_url = f"{assets}/swagger-ui-bundle.js",
            swagger_css_url = f"{assets}/swagger-ui.css",
            swagger_favicon_url = f"{assets}/favicon-32x32.png",
        ).body.decode()
        return _nonced_docs_response(html, tag = _SWAGGER_INIT_TAG)

    @app.get("/docs/oauth2-redirect", include_in_schema = False)
    async def swagger_ui_redirect():
        # This page is nothing but an inline script, so it needs the nonce too.
        html = get_swagger_ui_oauth2_redirect_html().body.decode()
        return _nonced_docs_response(html, tag = _OAUTH2_REDIRECT_TAG)

    @app.get("/redoc", include_in_schema = False)
    async def redoc_html(request: Request):
        assets = _docs_url(request, _DOCS_ASSETS_URL)
        # ReDoc's bundle carries no inline init, so this one needs no nonce.
        return HTMLResponse(
            get_redoc_html(
                openapi_url = _docs_url(request, app.openapi_url),
                title = f"{app.title} - ReDoc",
                redoc_js_url = f"{assets}/redoc.standalone.js",
                redoc_favicon_url = f"{assets}/favicon-32x32.png",
            ).body.decode()
        )


# Cap request bodies on protected POSTs; upload routes get explicit multipart headroom.
import json as _json_for_413  # noqa: E402
from utils.upload_limits import (  # noqa: E402
    STT_AUDIO_JSON_MAX_BYTES,
    STT_AUDIO_RAW_MAX_BYTES,
    UNSTRUCTURED_RECIPE_UPLOAD_MAX_BYTES,
    default_request_body_limit_bytes,
    upload_request_limit_bytes,
)

# Public auth routes (/api/auth/login, /refresh, /link-exchange, ...) are
# unauthenticated and take only small JSON bodies, so cap them well below the
# default upload-sized limit: /api/auth/link-exchange in particular accepts an
# attacker-controlled token that FastAPI buffers and exchange_link_token_with_secret
# then scans/decodes/HMACs, so bound the buffered body here before it is read.
AUTH_REQUEST_BODY_MAX_BYTES = 64 * 1024

_BODY_PROTECTED_PREFIXES = (
    # Blanket-protect the whole /v1 surface, like /api/inference: every /v1 POST buffers a JSON
    # body and none is a multipart passthrough, so one prefix caps them all.
    "/v1",
    "/p/",
    "/api/auth",
    "/api/inference",
    "/api/picker",
    "/api/data-recipe",
    "/api/datasets",
    "/api/hub",
    "/api/chat",
    "/api/settings",
    "/api/train",
    "/api/export",
    "/mcp",
)
_DATASET_UPLOAD_PASSTHROUGH_PREFIXES = (
    "/api/datasets/upload",
    "/api/hub/datasets/upload",
)
_DATA_RECIPE_UNSTRUCTURED_UPLOAD_PASSTHROUGH_PREFIX = (
    "/api/data-recipe/seed/upload-unstructured-file"
)
# The diffusion dataset upload (POST /api/train/diffusion/dataset) is a multipart upload
# under /api/train; like /api/datasets/upload it enforces its own cap. EXACT path.
_DIFFUSION_DATASET_UPLOAD_PATH = "/api/train/diffusion/dataset"
_STT_MULTIPART_UPLOAD_PATHS = (
    "/v1/audio/transcriptions",
    "/api/inference/audio/transcriptions",
)
_BODY_UPLOAD_PASSTHROUGH_PREFIXES = (
    *_DATASET_UPLOAD_PASSTHROUGH_PREFIXES,
    _DATA_RECIPE_UNSTRUCTURED_UPLOAD_PASSTHROUGH_PREFIX,
)
# Matched by EXACT path (multipart uploads only), so sibling JSON sub-routes keep the normal cap.
_BODY_UPLOAD_PASSTHROUGH_EXACT_PATHS = (
    _DIFFUSION_DATASET_UPLOAD_PATH,
    *_STT_MULTIPART_UPLOAD_PATHS,
)


def _get_upload_passthrough_request_max_bytes(path: str) -> int:
    if path.startswith(_DATA_RECIPE_UNSTRUCTURED_UPLOAD_PASSTHROUGH_PREFIX):
        return upload_request_limit_bytes(UNSTRUCTURED_RECIPE_UPLOAD_MAX_BYTES)
    if path.rstrip("/") in _STT_MULTIPART_UPLOAD_PATHS:
        return upload_request_limit_bytes(STT_AUDIO_RAW_MAX_BYTES)
    # The trailing-slash variant reaches this middleware BEFORE the router's redirect_slashes
    # 307, so it must resolve to the same cap. JSON sub-routes keep extra path components.
    if (
        path.startswith(_DATASET_UPLOAD_PASSTHROUGH_PREFIXES)
        or path.rstrip("/") == _DIFFUSION_DATASET_UPLOAD_PATH
    ):
        return upload_request_limit_bytes()
    return default_request_body_limit_bytes()


def _get_request_body_max_bytes(path: str) -> int:
    if path.startswith("/api/auth"):
        return AUTH_REQUEST_BODY_MAX_BYTES
    if path.startswith("/api/inference/audio/transcribe/raw"):
        return STT_AUDIO_RAW_MAX_BYTES
    if path.startswith("/api/inference/audio/transcribe"):
        return STT_AUDIO_JSON_MAX_BYTES
    # multipart headroom over the raw stt cap for the openai transcription route on both mounts
    if path.rstrip("/") in _STT_MULTIPART_UPLOAD_PATHS:
        return upload_request_limit_bytes(STT_AUDIO_RAW_MAX_BYTES)
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
        request_max_bytes_getter = None,
        upload_passthrough_prefixes: tuple = (),
        upload_passthrough_max_bytes_getter = None,
        upload_passthrough_exact_paths: tuple = (),
    ):
        self.app = app
        self.max_bytes_getter = max_bytes_getter
        self.protected_prefixes = protected_prefixes
        self.request_max_bytes_getter = request_max_bytes_getter
        self.upload_passthrough_prefixes = upload_passthrough_prefixes
        self.upload_passthrough_max_bytes_getter = upload_passthrough_max_bytes_getter
        # Exact path, not prefix: sibling JSON sub-routes must keep the normal (small) body cap.
        self.upload_passthrough_exact_paths = upload_passthrough_exact_paths

    def _is_upload_passthrough(self, path: str) -> bool:
        # Exact paths also match their trailing-slash variant (this runs before redirect_slashes).
        return path.rstrip("/") in self.upload_passthrough_exact_paths or any(
            path.startswith(p) for p in self.upload_passthrough_prefixes
        )

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

    def _request_max_bytes(self, path: str) -> int:
        if self.request_max_bytes_getter is None:
            return int(self.max_bytes_getter())
        try:
            return int(self.request_max_bytes_getter(path))
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

        max_bytes = self._request_max_bytes(path)
        declared = None
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value.decode("latin-1"))
                except (ValueError, UnicodeDecodeError):
                    declared = None
                break

        if self._is_upload_passthrough(path):
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
    request_max_bytes_getter = _get_request_body_max_bytes,
    upload_passthrough_prefixes = _BODY_UPLOAD_PASSTHROUGH_PREFIXES,
    upload_passthrough_max_bytes_getter = _get_upload_passthrough_request_max_bytes,
    upload_passthrough_exact_paths = _BODY_UPLOAD_PASSTHROUGH_EXACT_PATHS,
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


class RemoteAccessCORSMiddleware(CORSMiddleware):
    """Allow remote browser origins only while a Cloudflare URL is published."""

    def __init__(self, cors_app, *, remote_access_state, **kwargs):
        self.remote_access_state = remote_access_state
        super().__init__(cors_app, **kwargs)

    def is_allowed_origin(self, origin: str) -> bool:
        return bool(
            getattr(self.remote_access_state, "cloudflare_url", None)
        ) or super().is_allowed_origin(origin)


_cors_origins = cors_origins_for_mode(
    api_only = os.environ.get("UNSLOTH_API_ONLY") == "1",
    secure = os.environ.get("UNSLOTH_SECURE") == "1",
)

app.add_middleware(
    RemoteAccessCORSMiddleware,
    remote_access_state = app.state,
    allow_origins = _cors_origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
    # is_allowed_origin closes the moment the tunnel URL clears, but a preflight
    # already cached by the browser does not. Measured in WebKit: with Starlette's
    # 600s default, a state-changing request still REACHED the server after remote
    # access was stopped (Chromium/Firefox/Edge re-preflighted). Keep the stale
    # window short so revocation is nearly as immediate as every other trust
    # signal here.
    max_age = 60,
)

from utils.keyless_api_access import KeylessToolPolicyMiddleware  # noqa: E402

app.add_middleware(KeylessToolPolicyMiddleware)

from utils.remote_access_settings import RemoteAccessStopResponseMiddleware  # noqa: E402

app.add_middleware(RemoteAccessStopResponseMiddleware)


# ============ Register API Routes ============

app.include_router(auth_router, prefix = "/api/auth", tags = ["auth"])
app.include_router(training_router, prefix = "/api/train", tags = ["training"])
app.include_router(models_router, prefix = "/api/models", tags = ["models"])
app.include_router(chat_history_router, prefix = "/api/chat", tags = ["chat"])
app.include_router(research_runs_router, prefix = "/api/chat/research-runs", tags = ["research-runs"])
app.include_router(
    chat_generation_runs_router,
    prefix = "/api/inference/chat-runs",
    tags = ["inference"],
)
app.include_router(inference_router, prefix = "/api/inference", tags = ["inference"])
# Unsloth-only inference endpoints (cancel, etc.) are not on the /v1 OpenAI-compat prefix.
app.include_router(inference_studio_router, prefix = "/api/inference", tags = ["inference"])

# Unsloth-only text-to-video endpoints; not exposed on the /v1 OpenAI-compat prefix.
app.include_router(video_router, prefix = "/api/inference", tags = ["inference"])

# OpenAI-compatible: mount the inference router at /v1 for external tools.
app.include_router(inference_router, prefix = "/v1", tags = ["openai-compat"])
app.include_router(preview_router, prefix = "/p", tags = ["preview"])
app.include_router(providers_router, prefix = "/api/providers", tags = ["providers"])

app.include_router(openai_codex_auth_router, prefix = "/api/providers", tags = ["providers"])

app.include_router(settings_router, prefix = "/api/settings", tags = ["settings"])
app.include_router(mcp_servers_router, prefix = "/api/mcp/servers", tags = ["mcp"])
app.include_router(prompts_router, prefix = "/api/prompts", tags = ["prompts"])
app.include_router(profile_stats_router, prefix = "/api/profile", tags = ["profile"])
app.include_router(datasets_router, prefix = "/api/datasets", tags = ["datasets"])
app.include_router(data_recipe_router, prefix = "/api/data-recipe", tags = ["data-recipe"])
app.include_router(llama_router, prefix = "/api/llama", tags = ["llama"])
app.include_router(whisper_router, prefix = "/api/whisper", tags = ["whisper"])
app.include_router(export_router, prefix = "/api/export", tags = ["export"])
app.include_router(rag_router, prefix = "/api/rag", tags = ["rag"])
app.include_router(training_history_router, prefix = "/api/train", tags = ["training-history"])
app.include_router(hub_inventory_router, prefix = "/api/hub", tags = ["hub"])
app.include_router(hub_datasets_router, prefix = "/api/hub/datasets", tags = ["hub"])
app.include_router(picker_templates_router, prefix = "/api/picker", tags = ["picker"])
app.include_router(hub_token_router, prefix = "/api/hub", tags = ["hub"])
app.include_router(youtube_router, prefix = "/api/youtube", tags = ["youtube"])

# Re-wrap /v1/* client errors into OpenAI/Anthropic envelopes; non-/v1 keeps {"detail": ...}.
install_api_error_handlers(app)


# ============ Health and System Endpoints ============

# /api/health has a hard deadline: preflight/backend.rs probes it with a 2s timeout right
# after TAURI_PORT is emitted, and a timeout is not retried -- it falls through to
# "desktop_owned_backend_starting", a dead end the user must clear by hand.
# A target, not a guarantee: the wait polls on the event loop and a C-extension import can
# hold the GIL past it. 1.5s measured a 1.742s worst case (0.26s margin); 1.0s buys one
# extra provisional reply, and only the web UI reads chat_only anyway.
_HEALTH_DETECT_BUDGET_S = 1.0


async def _await_hardware_detection(budget: float) -> bool:
    """Wait up to ``budget`` seconds for DEVICE to be set. True iff it is.

    Polls on the event loop instead of awaiting ensure_hardware_detected() in a thread:
    asyncio.wait_for cannot cancel a to_thread, so a timed-out call holds the executor slot
    for the rest of the import and a polled endpoint would drain the pool. Detection runs on
    the warm thread, or the one start_background_detection() puts up.

    Returns False without kicking anything when the warm is switched off. Health is probed
    automatically (desktop preflight, the frontend's first fetch), so kicking detection here
    would import torch on every such host and the switch would buy nothing. The provisional
    answer ships instead and the first hardware-dependent operation detects.
    """
    if os.environ.get(DISABLE_ENV_VAR) == "1":
        return _hw_module.DETECTION_COMPLETE.is_set() and _hw_module.DEVICE is not None
    # The event AND DEVICE: branches assign DEVICE and keep probing, and shutdown clears
    # DEVICE then the event, so event-set-with-DEVICE-None would serve a torn-down verdict.
    if _hw_module.DETECTION_COMPLETE.is_set() and _hw_module.DEVICE is not None:
        return True
    start_background_detection()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget
    while not (_hw_module.DETECTION_COMPLETE.is_set() and _hw_module.DEVICE is not None):
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(0.02)
    return True


def _hardware_snapshot() -> Optional[tuple[bool, Optional[str], Optional[str]]]:
    """``(chat_only, chat_only_reason, chat_only_detail)`` if detection is settled, else ``None``.

    A seqlock read rather than ``_DETECT_LOCK``: that lock would park the endpoint for the
    whole torch import, the stall this startup path removes. A forced re-detect clears the
    event on the way in and bumps the generation before setting it again, so a read bracketed
    by both lands wholly before or after one pass, never mid-pass where CHAT_ONLY is back to
    True and the reason to None.

    That middle must not be published: config/env.ts caches the first reply carrying
    `device_type` as authoritative, and the sidebar's recovery poll runs only while it reads
    `chat_only_reason == "mlx_unavailable"`, so one such reply hides Train for the session.
    """
    for _ in range(3):
        if not _hw_module.DETECTION_COMPLETE.is_set():
            return None
        generation = _hw_module.DETECTION_GENERATION
        device = _hw_module.DEVICE
        chat_only = bool(_hw_module.CHAT_ONLY)
        reason = getattr(_hw_module, "CHAT_ONLY_REASON", None)
        # Inside the guarded read, with the reason it belongs to. Read after it, a forced
        # re-detect starting in between would pair this reply's reason with a detail from
        # a different pass, or with none at all.
        detail = getattr(_hw_module, "CHAT_ONLY_DETAIL", None)
        if (
            device is not None
            and _hw_module.DETECTION_COMPLETE.is_set()
            and _hw_module.DETECTION_GENERATION == generation
        ):
            return chat_only, reason, detail
    return None


# How long a self-heal that has not started yet may keep holding a verdict back once the
# warm that schedules it is over. start_mlx_autorepair_if_needed() runs in
# _post_warm_background_work, immediately after join_background_warm(), so the handoff is
# the gap this covers; the warm itself is covered by _torch_warm_in_progress(), which is
# minutes on a cold Mac and cannot be replaced by any fixed number.
_MLX_PRESTART_GRACE_AFTER_WARM_S = 30.0
# Absolute backstop, measured from the first hold. _torch_warm_in_progress() goes false when
# the warm thread dies for any reason, but a warm parked forever inside an import never
# does, and "the scheduler is still coming" would then be a permanent answer: Train and
# Video would spin for the whole session instead of settling into the greyed state a broken
# MLX stack has genuinely earned. Well above the warm's own worst case, since firing this on
# a healthy boot would reintroduce the bug the hold exists to fix.
_MLX_PRESTART_CEILING_S = 900.0

_MLX_PRESTART_LOCK = threading.Lock()
# (detection generation, first hold, first tick the warm was seen STOPPED, None while it
# runs). Keyed by generation because detection is not once-per-process: a re-detect that
# republishes mlx_unavailable is a new verdict and gets its own window rather than
# inheriting a spent one. Guarded rather than atomic only because the three move together.
#
# The third field is when the warm was first seen stopped, not when it was last seen
# running, because nothing guarantees a health request lands near the end of the warm. The
# final stages are C-extension imports that hold the GIL for seconds at a time, so requests
# queue behind them and the next one served can be the first in a minute. Measuring the
# grace from the last observed poll would then start it in the past and expire it before
# the handoff it exists to cover, publishing the mlx_unavailable verdict the frontend
# stores as final -- the exact bug this hold prevents.
_mlx_prestart_hold: Optional[tuple[int, float, Optional[float]]] = None

# Indirected so tests can drive the windows without sleeping through them.
_mlx_prestart_clock = time.monotonic


def _mlx_prestart_hold_ok(generation: int) -> bool:
    """True while a self-heal that has not started yet may still hold a verdict back."""
    global _mlx_prestart_hold
    now = _mlx_prestart_clock()
    warming = _torch_warm_in_progress()
    with _MLX_PRESTART_LOCK:
        held = _mlx_prestart_hold
        if held is None or held[0] != generation:
            _mlx_prestart_hold = (generation, now, None if warming else now)
            return True
        _, first, stopped_seen = held
        if now - first >= _MLX_PRESTART_CEILING_S:
            return False
        if warming:
            # Still (or again) running, so the handoff has not happened yet and any earlier
            # stopped reading was a lull, not the end.
            _mlx_prestart_hold = (generation, first, None)
            return True
        if stopped_seen is None:
            # First time this pass has seen it stopped: the grace starts here, whenever the
            # warm actually ended, so a gap in polling cannot spend it before it opens.
            stopped_seen = now
            _mlx_prestart_hold = (generation, first, stopped_seen)
        return now - stopped_seen < _MLX_PRESTART_GRACE_AFTER_WARM_S


def _superseded_by_mlx_repair(snapshot: Optional[tuple[bool, Optional[str]]]) -> bool:
    """True when the MLX self-heal is about to replace this settled verdict.

    Scoped to /api/health rather than folded into ``_hardware_snapshot()``: the launcher's
    watchdog reads /api/liveness and holds its startup grace open while hardware_detecting
    is set, so a 15-minute reinstall must not stretch that grace. Only the UI reads
    chat_only, and only the UI has a row to grey out on it.

    Bounded, never open-ended. A live worker holds the verdict for as long as its install
    takes, capped by mlx_repair._WORKER_BUDGET_S: the repair's own subprocess timeout plus
    an allowance for the post-install imports that verify it, which are not themselves
    timed, so a worker parked in one cannot hold the verdict for the rest of the process.
    A repair that has not started yet is only a promise, and this is where that promise
    expires: the scheduler runs after the warm, so the hold lasts while the warm does and
    a short handoff beyond it, under an absolute ceiling for the warm that never ends.
    Past that the verdict settles exactly as it did before any of this existed.
    """
    if snapshot is None:
        return False
    if not _hw_module.verdict_pending_mlx_repair(snapshot[0], snapshot[1]):
        return False
    try:
        from utils.mlx_repair import mlx_repair_started

        # Read after the predicate, so a repair that claims the latch between the two calls
        # resolves the safe way: still held, and now on the worker rather than on a clock.
        if mlx_repair_started():
            return True
    except Exception as exc:
        logger.debug("MLX repair start check failed, holding on the pre-start window: %s", exc)
    return _mlx_prestart_hold_ok(_hw_module.DETECTION_GENERATION)


def _torch_warm_in_progress() -> bool:
    """True while the coordinated warm thread is still working through its stages.

    A separate field from ``hardware_detecting`` on purpose, rather than widening that one.
    Hardware detection is only ``_STAGES[0]``; inference_backend, transformers, and datasets
    run after it, and those C-extension imports can hold the GIL
    for seconds at a time. A launcher ending its startup grace on ``hardware_detecting``
    alone ends it with the expensive half of the warm still ahead of it, which is the window
    the grace exists for. But that marker also means "this hardware verdict is provisional,
    re-read it", and config/hardware-verdict.ts keeps the UI provisional and polling while it
    is set, so keeping it lit through datasets would hide Train for the whole warm over a
    verdict that settled seconds in. Two meanings, two fields.

    A snapshot read of module state, no lock and no wait, so /api/liveness stays cheap.

    False whenever no warm thread is running, which is what keeps the deferred case working:
    with UNSLOTH_STUDIO_DISABLE_TORCH_WARM=1 the warm never starts, and one retired mid-stage
    by a shutdown never finishes. Neither will ever set ``finished``, so deriving this from
    "not finished" would report warming forever and hold the launcher's startup grace open
    until it expired on its own. Absence therefore covers both "warm is over" and "no warm is
    coming", and the field needs no deferred companion of its own.
    """
    status = warm_status()
    return bool(status["started"] and not status["finished"] and status["alive"])


# Modules that expose generation_in_flight(). The three engines answer for the render itself;
# routes.inference answers for the image-persist tail, which outlives the engine's own marker.
_MEDIA_BACKEND_MODULES = (
    "core.inference.video",
    "core.inference.diffusion",
    "core.inference.sd_cpp_backend",
    "routes.inference",
)


def _media_generation_active() -> bool:
    """Check imported media backends without importing, constructing, or locking them."""
    for module_name in _MEDIA_BACKEND_MODULES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        try:
            if module.generation_in_flight():
                return True
        except Exception:
            continue
    return False


def _inference_active() -> bool:
    """True while at least one generation is in flight.

    Published so the desktop health watchdog can tell a backend that is busy serving from
    one that has died: a saturated host can stall the event loop past a probe budget, and
    killing there ends a response the user is still waiting on.

    A len() under a threading.Lock held only for that read, plus a bool off each resident
    media backend, so the route stays cheap. Failures report "not busy", the same answer as
    before this field existed.
    """
    try:
        from state import active_generations
        if active_generations.count() > 0:
            return True
    except Exception:
        pass
    return _media_generation_active()


@app.get("/api/liveness")
async def liveness_check():
    """Cheap process liveness for desktop port validation."""
    alive = {
        "status": "alive",
        "service": "Unsloth UI Backend",
        "desktop_protocol_version": 1,
        # Lockstep with DESKTOP_MANAGEABILITY_VERSION in
        # studio/src-tauri/src/preflight/version.rs and `desktop-capabilities`.
        "desktop_manageability_version": 2,
        "supports_desktop_auth": True,
        "supports_desktop_backend_ownership": True,
        "studio_root_id": _studio_root_id(),
        **({"desktop_owner": owner} if (owner := _desktop_owner()) else {}),
    }
    # Same unsettled markers /api/health publishes, and for the desktop health watchdog they
    # are the point of the route: it probes liveness every 15s and holds its startup grace
    # period open until a reply says the warm-up is over, because the warm thread's
    # `import torch` holds the GIL and can stall the next probes on a healthy process.
    # The watchdog reads torch_warm_in_progress for that, not hardware_detecting: later
    # transformers and datasets stages can also hold the GIL after detection settles.
    # Both are non-blocking reads of module-level state, so unlike health this neither starts
    # detection nor waits on it and the route stays cheap.
    if _torch_warm_in_progress():
        alive["torch_warm_in_progress"] = True
    # Startup is not the only window where a healthy backend can miss probes: an
    # oversubscribed host generating on every slot stalls this loop the same way, long
    # after the warm is over. The watchdog widens its failure budget on this marker
    # rather than ending a stream that is still producing tokens.
    if _inference_active():
        alive["inference_active"] = True
    if _hardware_snapshot() is None:
        alive["hardware_detecting"] = True
        if os.environ.get(DISABLE_ENV_VAR) == "1":
            # Nothing is detecting while the warm is switched off, so the verdict will not
            # settle on its own. Say so, or the watchdog holds its grace open for nothing.
            alive["hardware_detection_deferred"] = True
    return alive


@app.get("/api/health")
async def health_check(request: Request):
    """Liveness plus launcher capability bits; host fingerprint gated on a bearer.

    Unauthenticated callers get non-sensitive fields (service, studio_root_id,
    chat_only, desktop_*, native_path_leases_supported) to re-adopt a sibling
    backend and gate UI before a token exists. version / studio_version /
    device_type require a bearer since they fingerprint the host.
    """
    # Wait for detection rather than grey out Train/Export on a GPU host, but only up to a
    # budget. Called for the wait, not the answer: _hardware_snapshot() below decides the reply.
    await _await_hardware_detection(_HEALTH_DETECT_BUDGET_S)
    # Snapshot, not a bare global read: a forced re-detect can start at any moment.
    snapshot = _hardware_snapshot()
    # A chat-only verdict the MLX self-heal is about to overturn is not an answer yet. Hold it
    # back and keep replying provisionally, or the Mac gets Train greyed out under a tooltip the
    # reinstall makes wrong minutes later. Video does not wait on this: it runs on Metal without
    # MLX, so it reads /api/system/hardware instead.
    mlx_repairing = _superseded_by_mlx_repair(snapshot)
    if mlx_repairing:
        snapshot = None
    base = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Unsloth UI Backend",
        # Literal True with no snapshot, not a CHAT_ONLY read: a pass in flight sets the flag False
        # before a probe that can still fall back to CPU.
        "chat_only": snapshot[0] if snapshot is not None else True,
        "desktop_protocol_version": 1,
        # Lockstep: see the note in /api/liveness above.
        "desktop_manageability_version": 2,
        "supports_desktop_auth": True,
        "supports_desktop_backend_ownership": True,
        # Opaque per-install id; launchers reject sibling Unsloth instances on the same port.
        "studio_root_id": _studio_root_id(),
        "native_path_leases_supported": native_path_leases_supported(),
        **({"desktop_owner": owner} if (owner := _desktop_owner()) else {}),
    }
    # Lockstep with /api/liveness: the launcher falls back to this route on a backend too old
    # to have liveness, so the warm marker has to reach it by the same path.
    if _torch_warm_in_progress():
        base["torch_warm_in_progress"] = True
    # Lockstep with /api/liveness for the same reason: the fallback route has to carry the
    # busy marker too, or an older backend loses the widened budget.
    if _inference_active():
        base["inference_active"] = True
    if snapshot is None:
        # chat_only above is the pre-detection default, not a measurement; clients should re-read.
        base["hardware_detecting"] = True
        # Not for a held-back verdict: the repair settles it on its own, and "deferred" means
        # nothing ever will, which env.ts answers by storing the conservative chat_only.
        if os.environ.get(DISABLE_ENV_VAR) == "1" and not mlx_repairing:
            # Nothing is detecting until a hardware-dependent operation runs; say so instead of making clients poll.
            base["hardware_detection_deferred"] = True
    auth = request.headers.get("authorization", "")
    bearer = auth.split(" ", 1)[1] if auth.lower().startswith("bearer ") else None
    try:
        from auth.authentication import credentials_for_token
        from auth.authentication import get_current_subject as _gcs

        # resolved rather than built, so a scope covering this route answers it in full
        creds = await credentials_for_token(request, bearer)
        if creds is None:
            return base
        # Must await: a bare coroutine is truthy and would skip the auth check
        subject = await _gcs(creds)
    except HTTPException:
        return base
    except Exception:
        return base
    if not subject:
        return base

    # Re-read: the bearer check awaits, so a forced re-detect can land in between.
    snapshot = _hardware_snapshot()
    if _superseded_by_mlx_repair(snapshot):
        mlx_repairing = True
        snapshot = None

    platform_map = {"darwin": "mac", "win32": "windows", "linux": "linux"}
    device_type = platform_map.get(sys.platform, sys.platform)
    # Alongside device_type, not folded into it: "mac" is every Darwin host, and an Intel
    # Mac with a discrete GPU spills to system RAM like a PC while Apple Silicon has one
    # pool and nowhere to spill. The UI words its memory warnings from this. Same gate the
    # Metal context budget uses, and a pure platform check, so a health poll pays nothing.
    from utils.hardware import is_apple_silicon

    authed = {
        **base,
        "version": UNSLOTH_VERSION,
        "studio_version": STUDIO_VERSION,
        # API-screen fields (authed-only; they fingerprint how the host is exposed).
        "cloudflare_url": getattr(request.app.state, "cloudflare_url", None),
        "server_url": getattr(request.app.state, "server_url", None),
        "secure": bool(getattr(request.app.state, "secure", False)),
    }
    if snapshot is not None:
        # Why chat_only is set; fingerprints the host, so keep it authed. One snapshot for all three.
        authed["chat_only"] = snapshot[0]
        authed["chat_only_reason"] = snapshot[1]
        # What specifically blocked that reason, when detection recorded one. Only the MLX
        # gate does today, and only because it is all-or-nothing: without it the greyed-out
        # Train row can only say "run `unsloth studio update`", which is no help to someone
        # whose update has already run and left one package behind. From the snapshot, so it
        # cannot come from a different detection pass than the reason beside it.
        authed["chat_only_detail"] = snapshot[2]
        authed["device_type"] = device_type
        authed["apple_silicon"] = is_apple_silicon()
        # base predates the bearer await; never ship "detecting" beside a measurement.
        authed.pop("hardware_detecting", None)
        # Same for the deferred marker: the client reads it first and would keep the old reason.
        authed.pop("hardware_detection_deferred", None)
        # torch_warm_in_progress deliberately survives. It does not qualify the verdict below;
        # a settled verdict is exactly the state where the warm has finished stage one and is
        # off importing transformers, and dropping it here would hand the watchdog the same
        # too-early "startup is over" this field exists to replace.
    else:
        # A re-detect started during the bearer await and base carries no chat_only_reason, so a
        # client reading this as measured would store reason null and stop the sidebar's recovery
        # poll. Mark provisional and omit device_type: env.ts treats it as authoritative.
        authed["hardware_detecting"] = True
        if mlx_repairing:
            # base was built before the repair was noticed, so drop a marker that now
            # contradicts it: the repair will settle this verdict, deferred means nothing will.
            authed.pop("hardware_detection_deferred", None)
    return authed


@app.get("/api/studio/install-source")
def studio_install_source(_current_subject: str = Depends(get_current_subject)):
    """Return source-aware install metadata without remote update checks."""
    return get_studio_install_source_status(UNSLOTH_VERSION)


@app.get("/api/studio/update-status")
def studio_update_status(_current_subject: str = Depends(get_current_subject)):
    """Return source-aware manual update status for browser-served Unsloth."""
    return get_studio_update_status(UNSLOTH_VERSION)


@app.get("/api/studio/release-notes")
def studio_release_notes(
    version: str = Query(..., max_length = 64),
    refresh: bool = Query(False),
    _current_subject: str = Depends(get_current_subject),
):
    """Return the newest release's notes. `version` is echoed, not looked up."""
    if not is_supported_version_query(version):
        raise HTTPException(status_code = 422, detail = "Invalid version.")
    return get_release_notes(version, refresh = refresh)


@app.get(
    "/api/studio/download-transport-capabilities",
    response_model = TransportCapabilities,
)
def studio_download_transport_capabilities(
    probe: bool = False, _current_subject: str = Depends(get_current_subject)
):
    # Sync def, so FastAPI runs this in the threadpool and an opted-in probe cannot block the loop.
    return asdict(get_download_transport_capabilities(probe = probe))


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


def _get_cached_system_gpu_info(logger) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return training and inference GPU info with bounded live-probe churn."""
    import time
    from utils.hardware import (
        get_backend_visible_gpu_info,
        get_visible_gpu_utilization,
        get_vulkan_inference_gpu_info,
    )

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

        # Device indices are backend-specific. Never overlay CUDA/ROCm metrics
        # onto compact Vulkan ordinals merely because both happen to start at 0.
        visibility_backend = visibility_info.get("backend")
        utilization_backend = utilization_info.get("backend")
        metrics_match = (
            not visibility_backend
            or not utilization_backend
            or visibility_backend == utilization_backend
        )
        util_devices = (
            {d.get("index"): d for d in utilization_info.get("devices", [])}
            if metrics_match
            else {}
        )
        enriched_devices = []

        for dev in visibility_info.get("devices", []):
            idx = dev.get("index")
            util = util_devices.get(idx, {})

            total_vram = util.get("vram_total_gb") or dev.get("memory_total_gb") or 0
            # Keep None (usage unknown, e.g. Windows ROCm perf counter) so the UI shows unknown, not 0.
            used_vram = util.get("vram_used_gb", dev.get("vram_used_gb"))
            reported_free_vram = util.get("vram_free_gb", dev.get("vram_free_gb"))

            enriched_dev = dict(dev)
            enriched_dev["vram_used_gb"] = used_vram
            enriched_dev["vram_free_gb"] = (
                round(total_vram - used_vram, 2)
                if total_vram and used_vram is not None
                else reported_free_vram
            )
            enriched_dev["vram_utilization_pct"] = util.get(
                "vram_utilization_pct", dev.get("vram_utilization_pct")
            )
            enriched_devices.append(enriched_dev)

        # The tile divides the aggregate by the SUMMED per-device totals, so both must
        # describe the same cards. The two probes enumerate independently: visibility
        # drops a device whose mem_get_info raises, the aggregate side reads torch
        # properties only and keeps it. A device in one and not the other inflates the
        # percentage and floors free at 0, so identical index sets only (#7452).
        aggregate_basis_matches = metrics_match and {
            d.get("index") for d in utilization_info.get("devices", [])
        } == {d.get("index") for d in enriched_devices}

        try:
            from core.inference.llama_cpp import LlamaCppBackend
            from utils.hardware import DeviceType, get_device

            llama_uses_vulkan = LlamaCppBackend._is_vulkan_backend()
            if llama_uses_vulkan:
                # The separate inference inventory owns Vulkan ordinals. Keep this false so a failed
                # Vulkan probe cannot expose torch indices that llama.cpp reads in another namespace.
                gpu_ids_supported = False
            else:
                # XPU indices cannot yet be applied safely across Level Zero's FLAT and COMPOSITE modes.
                # A proven CPU-only llama.cpp build cannot apply a CUDA pin either.
                gpu_ids_supported = (
                    get_device() != DeviceType.XPU and not LlamaCppBackend._backend_lacks_gpu_lib()
                )
        except Exception as e:
            logger.debug(f"Could not resolve gpu_ids support: {e}")
            llama_uses_vulkan = False
            gpu_ids_supported = True
        # Preserve backend/index metadata from the visibility probe: a CPU training host can expose
        # a Vulkan inference GPU, and the UI must label it Vulkan, not the top-level CPU backend.
        gpu_info = {
            **visibility_info,
            "available": visibility_info.get("available", False),
            "devices": enriched_devices,
            "backend": visibility_info.get("backend"),
            "gguf_gpu_ids_supported": gpu_ids_supported,
            # Host-level used VRAM, for when no counter is attributable to one card
            # (#7452). Only the Windows ROCm path sets it; None everywhere else.
            "vram_used_gb_aggregate": utilization_info.get("vram_used_gb_aggregate")
            if aggregate_basis_matches
            else None,
        }

        # Keep inference placement separate on train-capable hosts where a forced Vulkan llama.cpp
        # bundle can enumerate a different device set. If Vulkan is installed but its probe fails,
        # retain the unavailable Vulkan shape instead of budgeting GPUs llama.cpp cannot use.
        if visibility_info.get("backend") == "vulkan":
            gpu_info["gguf_gpu_ids_supported"] = bool(enriched_devices)
            inference_gpu_info = gpu_info
        else:
            vulkan_info = get_vulkan_inference_gpu_info()
            inference_gpu_info = (
                {
                    **vulkan_info,
                    # Pinnable only once the probe enumerated devices: without ordinals there is nothing to offer.
                    "gguf_gpu_ids_supported": bool(vulkan_info.get("devices")),
                }
                if vulkan_info is not None
                else gpu_info
            )

        combined_info = (gpu_info, inference_gpu_info)
        _system_gpu_cache = (time.monotonic(), combined_info)
        return combined_info


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
    from utils.hardware import (
        get_device,
        export_capability,
        video_capability,
        cpu_frequency_mhz,
    )
    from utils.hardware.hardware import _backend_label

    logger = logging.getLogger(__name__)

    gpu_info, inference_gpu_info = _get_cached_system_gpu_info(logger)

    memory = psutil.virtual_memory()

    # Corrects psutil's 1000x-too-small Apple Silicon M4+ reading (issue #8519).
    cpu_freq_mhz = cpu_frequency_mhz()

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
            "frequency_mhz": cpu_freq_mhz,
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
        "inference_gpu": inference_gpu_info,
        "ml_packages": ml_packages,
        # Export capability + torch-aware reason. See /api/system/hardware.
        **export_capability(),
        # Video capability + reason, same shape. Additive: older clients ignore the extra keys.
        **video_capability(),
    }


@app.get("/api/system/gpu-visibility")
async def get_gpu_visibility(current_subject: str = Depends(get_current_subject)):
    # Off-loop: get_device() blocks on detection while the warm is still importing torch.
    return await asyncio.to_thread(get_backend_visible_gpu_info)


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
    from utils.hardware import (
        get_gpu_summary,
        get_package_versions,
        export_capability,
        video_capability,
    )

    body = {
        "gpu": get_gpu_summary(),
        "versions": get_package_versions(),
        # Export capability + torch-aware reason; the Export UI grays out with the message.
        **export_capability(),
        # Video capability + reason; the Video page shows the message in place of the generator.
        **video_capability(),
    }
    if include_details:
        from utils.llama_cpp_update import get_installed_llama_version

        # All backend-visible GPUs (respects CUDA_VISIBLE_DEVICES); get_gpu_summary reports only
        # the primary. Sort by visible_ordinal: the nvidia-smi path returns physical order, so a
        # reordering CUDA_VISIBLE_DEVICES (e.g. "5,3") would mislabel by array index.
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
    # IPv6 hosts use brackets (RFC 3986 3.2.2): bare partition(":") breaks `-H ::1`.
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


# A loopback peer carrying any of these is a proxy/tunnel relaying a remote client, so the
# peer is the proxy, not the caller: cloudflared sets cf-connecting-ip, reverse proxies set
# the rest (uvicorn only consumes x-forwarded-for, so the others survive to here).
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
    # urlparse raises ValueError on malformed IPv6 brackets; swallow so it doesn't 500.
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
        # Single-user notebook proxy: allow autofill, but never a public tunnel (sets cf-connecting-ip).
        return request.headers.get("cf-connecting-ip") is None
    return _is_local_bootstrap_request(request)


_IMMUTABLE_ASSET_CACHE_CONTROL = "public, max-age=31536000, immutable"


class ImmutableStaticFiles(StaticFiles):
    """Serve Vite's content-hashed assets without browser revalidation."""

    def file_response(
        self,
        full_path,
        stat_result,
        scope,
        status_code = 200,
    ):
        response = super().file_response(full_path, stat_result, scope, status_code)
        response.headers["Cache-Control"] = _IMMUTABLE_ASSET_CACHE_CONTROL
        return response


class _AssetGZipMiddleware(GZipMiddleware):
    """Serve range requests uncompressed; gzip + 206 mislabels Content-Range."""

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and any(key == b"range" for key, _ in scope["headers"]):
            await self.app(scope, receive, send)
            return
        await super().__call__(scope, receive, send)


def _is_live_cloudflare_frontend_request(scope, app_state) -> bool:
    cloudflare_url = getattr(app_state, "cloudflare_url", None)
    headers = dict(scope.get("headers", ()))
    if not cloudflare_url or not headers.get(b"cf-connecting-ip"):
        return False
    try:
        expected_host = urlparse(cloudflare_url).hostname
        request_host = urlparse(f"//{headers.get(b'host', b'').decode('latin-1')}").hostname
    except (UnicodeDecodeError, ValueError):
        return False
    return bool(expected_host) and request_host == expected_host


def _is_remote_frontend_request(scope, app_state) -> bool:
    """True for a request the desktop backend may answer with its packaged web UI.

    Two ways in, both identified by the connection itself rather than a client
    header the caller controls: Cloudflare's own edge, or one of the sockets the
    runtime LAN listener bound (Settings > LAN access).
    """
    from lan_access import request_on_lan_listener
    return _is_live_cloudflare_frontend_request(scope, app_state) or request_on_lan_listener(scope)


class _TunnelOnlyFrontend:
    def __init__(self, frontend_app, app_state):
        self.frontend_app = frontend_app
        self.app_state = app_state

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or _is_remote_frontend_request(scope, self.app_state):
            await self.frontend_app(scope, receive, send)
            return
        await Response(status_code = 404)(scope, receive, send)


def setup_frontend(
    app: FastAPI,
    build_path: Path,
    *,
    tunnel_only: bool = False,
):
    """Mount frontend static files (optional).

    ``tunnel_only`` restricts the mount to remote callers: the Cloudflare edge, or
    a socket the runtime LAN listener bound. See :func:`_is_remote_frontend_request`.
    """
    if not build_path.exists():
        return False

    assets_dir = build_path / "assets"
    if assets_dir.exists():
        assets_app = _AssetGZipMiddleware(
            ImmutableStaticFiles(directory = assets_dir),
            minimum_size = 1024,
            compresslevel = 6,
        )
        if tunnel_only:
            assets_app = _TunnelOnlyFrontend(assets_app, app.state)
        app.mount("/assets", assets_app, name = "assets")

    def _frontend_request_allowed(request: Request) -> bool:
        return not tunnel_only or _is_remote_frontend_request(request.scope, app.state)

    def _build_index_response(request: Request) -> Response:
        content = (build_path / "index.html").read_bytes()
        content = _strip_crossorigin(content)
        # Bootstrap pw goes only to a same-origin, direct-loopback client (or Colab's single-user
        # proxy): a wildcard bind must not serve it to a LAN or proxied peer. Vary: Origin.
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
        if not _frontend_request_allowed(request):
            return Response(status_code = 404)
        return _build_index_response(request)

    @app.get("/{full_path:path}")
    async def serve_frontend(request: Request, full_path: str):
        # Unknown API paths: raise a real 404 so the api_errors handlers render the right envelope
        # for /v1/* ({"detail": ...} for /api/*). The request path is "/" + full_path.
        if full_path in {"api", "v1"} or full_path.startswith(("api/", "v1/")):
            raise HTTPException(status_code = 404, detail = "API endpoint not found")
        if not _frontend_request_allowed(request):
            return Response(status_code = 404)

        file_path = (build_path / full_path).resolve()

        # Block path traversal — resolved path must stay inside build_path
        if not file_path.is_relative_to(build_path.resolve()):
            return Response(status_code = 403)

        if file_path.is_file():
            return FileResponse(file_path)

        # Serve index.html as bytes — avoids Content-Length mismatch
        return _build_index_response(request)

    return True
