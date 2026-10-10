# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import sys
import threading
import time
from pathlib import Path as _Path
import asyncio
from dataclasses import asdict

from typing import Any, Optional

os.environ["PYTHONWARNINGS"] = "ignore"

# PCI_BUS_ID before any CUDA context so torch indices match nvidia-smi; setdefault lets override win.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# Same ROCm AOTriton opt-in as unsloth/__init__.py, for a backend that defers importing torch;
# spawned workers inherit it. `setdefault` preserves an explicit override, including "0".
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

# The desktop app hands this process a GUI environment, and a GUI environment has
# no ~/.bashrc in it. `shell_path::fix_path()` in src-tauri spawns the
# login shell and then takes PATH out of it and nothing else, so an AMD host's
# HSA_OVERRIDE_GFX_VERSION / ROCM_PATH / USE_CK are dropped on the desktop path
# and kept on the `unsloth studio` one. #9926 is that difference: identical model
# and machine, SIGSEGV from the app and a clean run from a terminal.
#
# Here because HSA reads HSA_OVERRIDE_GFX_VERSION and USE_CK when torch first
# touches the GPU, which is far below this. A no-op unless the desktop app owns
# this process AND the host has an AMD GPU AND the variable is absent, so a
# terminal launch and every non-AMD host are unchanged.
_backend_dir = str(_Path(__file__).parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
try:
    from utils.desktop_shell_env import import_rocm_env_from_login_shell as _import_rocm_env
    _import_rocm_env()
except Exception:
    pass

# normalize allocator booleans before torch parses them; spawned workers inherit the environment.
from utils.allocator_conf import normalize_allocator_conf as _normalize_allocator_conf

for _name, _old, _new in _normalize_allocator_conf():
    print(
        f"Unsloth: {_name}={_old!r} has noncanonical boolean casing, which PyTorch rejects; "
        f"using {_new!r}.",
        file = sys.stderr,
    )

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

# Python 3.8+ ignores PATH for extension DLLs, so register ROCm bin dirs before torch import.
if sys.platform == "win32":
    # Module scope: the handle removes the search-path entry when garbage collected.
    _ROCM_DLL_HANDLES: list = []

    def _add_rocm_dll_dirs() -> None:
        candidates = []
        for _var in ("HIP_PATH", "ROCM_PATH"):
            _val = os.environ.get(_var)
            if _val:
                candidates.append(os.path.join(_val, "bin"))
        _default_root = os.path.join(
            os.environ.get("ProgramFiles", r"C:\Program Files"), "AMD", "ROCm"
        )

        def _ver_key(name: str) -> tuple:
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

    # bitsandbytes runs hipinfo.exe via PATH at import; add the venv Scripts dir (AMD wheel ships it).
    _scripts_dir = os.path.dirname(sys.executable)
    if os.path.isfile(os.path.join(_scripts_dir, "hipInfo.exe")):
        import shutil as _shutil
        if not _shutil.which("hipinfo.exe"):
            os.environ["PATH"] = _scripts_dir + os.pathsep + os.environ.get("PATH", "")
        del _shutil
    del _scripts_dir

    # bitsandbytes derives the DLL name from torch.version.hip but the wheel ships rocm72.dll.
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
        # Only when a ROCm bnb DLL exists: a HIP SDK on CUDA/CPU must not force ROCm.
        if _found_rocm_bnb:
            _bnb_rocm_ver_final = _bnb_rocm_ver or os.environ.get("BNB_ROCM_VERSION") or "72"
            os.environ["BNB_ROCM_VERSION"] = _bnb_rocm_ver_final
            os.environ["UNSLOTH_BNB_ROCM_VERSION_SOURCE"] = "detected"
            _logging.getLogger(__name__).info(
                "Windows ROCm: set BNB_ROCM_VERSION=%s (from installed BNB wheel)",
                _bnb_rocm_ver_final,
            )

    if os.environ.get("BNB_ROCM_VERSION"):
        import logging as _logging
        _logging.getLogger("bitsandbytes.cextension").addFilter(
            lambda _r: "environment variable detected" not in _r.getMessage()
        )

# WSL gfx1151 needs HSA_ENABLE_DXG_DETECTION=1 before torch touches the GPU.
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

_backend_dir = str(_Path(__file__).parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

# OS trust store before any connection: TLS-inspecting proxies break certifi.
from utils.native_tls import activate_native_tls
from utils.happy_eyeballs import activate_happy_eyeballs

activate_native_tls()
activate_happy_eyeballs()

from utils.cpu_threads import configure_cpu_threads

try:
    configure_cpu_threads()
except ValueError as exc:
    _raw = os.environ.get("UNSLOTH_CPU_THREADS")
    raise SystemExit(f"Error: Invalid UNSLOTH_CPU_THREADS value {_raw!r}: {exc}") from None

# Conda Python: seed platform._sys_version_cache (cpython#102396).
import _platform_compat  # noqa: F401

# uvicorn main:app bypasses run.py; must precede unsloth-zoo import (import-time binding).
from utils.paths.storage_roots import studio_root as _studio_root

# unsloth_zoo.compiler reads UNSLOTH_COMPILE_LOCATION at import time.
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
from utils.paths.storage_roots import unsloth_home as _unsloth_home

_MASTER_ROOT = _unsloth_home()
# A master root at the legacy path still owns runtimes, so equality alone must not skip the export.
if _STUDIO_ROOT_RESOLVED != _LEGACY_STUDIO_ROOT or _MASTER_ROOT is not None:
    if not os.environ.get("UNSLOTH_STUDIO_HOME"):
        os.environ["UNSLOTH_STUDIO_HOME"] = str(_STUDIO_ROOT_RESOLVED)
    _MANAGED_ROOT = _MASTER_ROOT or _STUDIO_ROOT_RESOLVED
    _MANAGED_LLAMA_CPP_PATH = _MANAGED_ROOT / "llama.cpp"
    if not os.environ.get("UNSLOTH_LLAMA_CPP_PATH"):
        os.environ["UNSLOTH_LLAMA_CPP_PATH"] = str(_MANAGED_LLAMA_CPP_PATH)
    from utils.llama_cpp_path_settings import mark_managed_llama_cpp_path

    mark_managed_llama_cpp_path(_MANAGED_LLAMA_CPP_PATH)

# huggingface_hub reads HF_ENDPOINT itself, at import, unnormalised and unvalidated.
from utils.hub_settings import apply_hub_settings as _apply_hub_settings

_apply_hub_settings()
del _apply_hub_settings

# The studio bundles unsloth_zoo; declare unsloth present (as `import unsloth` does) so its
# lazy submodule imports and the DiffusionGemma runner don't trip the install guard.
os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

# Must not import unsloth here (pulls torch/GPU); before transformers reads sentencepiece.
from utils.sentencepiece_guard import disable_sentencepiece_on_windows as _no_sentencepiece

_no_sentencepiece()
del _no_sentencepiece

import hashlib
import mimetypes
import re as _re
import shutil
import warnings
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version as package_version
from urllib.parse import urlparse


_STUDIO_INSTALL_ID_RE = _re.compile(r"^[0-9a-f]{64}$")


def _read_studio_install_id() -> str:
    """Per-install opaque id at $STUDIO_HOME/share/studio_install_id. Returns "" when absent or not a 64-char
    lowercase-hex token; a launcher with a baked id rejects "", so it restores a missing id before starting
    Studio. Carries no install-path info (matters when Unsloth runs -H 0.0.0.0)."""
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
    """Same-install discriminator for /api/health (cached at import). Empty when no installer token is
    present."""
    return _STUDIO_ROOT_ID_CACHE


# Some Windows installs map .js to text/plain; fix before StaticFiles.
if sys.platform == "win32":
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")

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
    external_import_router,
    inference_router,
    inference_studio_router,
    mcp_servers_router,
    skills_router,
    models_router,
    providers_router,
    openai_codex_auth_router,
    rag_router,
    research_runs_router,
    chat_generation_runs_router,
    training_history_router,
    training_router,
    video_router,
    video_openai_router,
    youtube_router,
)
import routes.browser as _browser_routes
from routes.llama import router as llama_router
from routes.engines import router as engines_router
from routes.llama_compat import is_engine_probe_path, router as llama_compat_router
from routes.whisper import router as whisper_router
from routes.npu import router as npu_router
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
from routes.sandbox_capability import router as sandbox_capability_router
from routes.systemone import MCP_PATH as DECISIONS_MCP_PATH, RequireStudioAuth, decisions_mcp
from routes.systemone import router as systemone_router
from routes.prompts import router as prompts_router
from routes.library import router as library_router
from routes.profile_stats import router as profile_stats_router
from auth import policy as auth_policy, storage
from auth.authentication import authenticated_via_api_key, get_current_subject
from hub.utils.host_paths import redact_inventory_host_paths
from utils.hardware import (
    start_background_detection,
    get_device,
    DeviceType,
    get_backend_visible_gpu_info,
)
import utils.hardware.hardware as _hw_module

from utils.torch_warmup import (
    DISABLE_ENV_VAR,
    background_media_import,
    join_background_warm,
    prewarm_diffusers_if_image_models_exist,
    reset_background_warm,
    start_background_warm,
    warm_status,
)
from utils.cache_cleanup import (
    clear_compiled_cache_unless_shared as _clear_compiled_cache_unless_shared,
)
from utils.lifespan_shutdown import run_lifespan_shutdown
from utils.native_path_leases import native_path_leases_supported

from utils.client_ip import client_ip
from utils.hf_endpoint import (
    DEFAULTS_BY_HEALTH_KEY as _HF_ENDPOINT_DEFAULTS,
    endpoint_is_reachable_by as _endpoint_is_reachable_by,
    browser_hf_endpoint,
    csp_asset_sources,
    csp_connect_sources,
    get_hf_datasets_server,
)
from hub import endpoint_proxy as _hub_endpoint_proxy
from hub.modelscope.router import (
    BROWSER_PREFIX as _MODELSCOPE_BROWSER_PREFIX,
    build_router as _build_modelscope_router,
)
from utils.hub_settings import active_source as _active_hub_source
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

    # Version literal moved to _version.py; scan both for a half-updated tree.
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

# Desktop runs the backend locally, so stdio MCP is safe; a tunnel can suspend this default.
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
            pass

    threading.Thread(target = _precache, daemon = True, name = "helper-gguf-precache").start()


def _run_llama_cpp_startup_probes(app: FastAPI) -> None:
    """llama.cpp capability (MTP support) + freshness (release age) probes, run OFF the startup critical path.
    Both are cached and freshness has a 24h disk TTL, but on a cold/expired cache the freshness check makes
    a blocking GitHub request, and on macOS the first `llama-server --help` exec can stall on Gatekeeper
    verification, and neither must gate `Application startup complete`. Writes app.state only; nothing reads
    those values synchronously at startup."""
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
    """Run the llama.cpp startup probes on a daemon thread, off the startup critical path. Skipped entirely
    when update checks are disabled, so a fully offline boot makes no background network calls."""
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
# Bumped every start/stop so a worker parked in join_background_warm() cannot act after shutdown.
_post_warm_generation = 0


def _post_warm_current_generation() -> int:
    with _post_warm_lock:
        return _post_warm_generation


def _start_post_warm_thread() -> bool:
    """Put up a post-warm worker for this lifespan. True iff one was started. Starts one even while a previous
    worker is parked in the warm join: declining there left a restart with no worker at all, since the old
    one was alive so this returned early, then read the shutdown and exited. Generations make the overlap
    safe."""
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
    """Retire whatever worker is current; never wait for it. Joining would hold shutdown for the rest of the ML
    stack import, the stall this path exists to avoid; bumping the generation suffices, since the worker
    re-reads it after its join."""
    global _post_warm_generation
    with _post_warm_lock:
        _post_warm_generation += 1


def _post_warm_retired(generation: Optional[int]) -> bool:
    """True when this post-warm worker's lifespan has ended (logs once when it has). The remaining work imports
    optional platform or RAG scheduling modules, so none of it may start for a stopped lifespan."""
    if generation is None or _post_warm_current_generation() == generation:
        return False
    import structlog as _structlog

    _structlog.get_logger(__name__).info(
        "post-warm work %s stood down: its lifespan ended while the ML stack was still loading",
        generation,
    )
    return True


def _start_linked_folder_auto_sync(generation: Optional[int]) -> None:
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
    """Platform repair and linked-folder lifecycle work after the coordinated warm. MLX repair used to probe the
    runtime before the socket bound; joining first keeps that optional probe out of the login-screen
    critical path. Linked-folder startup only loads embeddings when a queued sync has real ingestion
    work."""
    join_background_warm()

    # Shutdown can land during the join above; recheck before every action.
    if _post_warm_retired(generation):
        return

    # Apple Silicon without MLX: reinstall and re-detect (UNSLOTH_DISABLE_MLX_AUTOREPAIR=1 opts out).
    try:
        from utils.mlx_repair import start_mlx_autorepair_if_needed
        if _post_warm_retired(generation):
            return
        start_mlx_autorepair_if_needed()
    except Exception as _mlx_exc:
        import structlog as _structlog

        # Warning, not debug: a half-applied update here silently greys out Train/Export.
        _structlog.get_logger(__name__).warning("mlx autorepair skipped: %s", _mlx_exc)

    if _post_warm_retired(generation):
        return
    # Off the polled path: /api/system must not import torchao itself.
    if "torch" in sys.modules:
        with background_media_import() as _window_open:
            if _window_open:
                try:
                    _refresh_dense_quant_capability()
                except Exception as _dq_exc:  # noqa: BLE001 -- a picker label must never break the warm
                    import structlog as _structlog
                    _structlog.get_logger(__name__).debug(
                        "dense quant capability skipped: %s", _dq_exc
                    )
                try:
                    _refresh_quantised_streaming_capability()
                except Exception as _qs_exc:  # noqa: BLE001 -- a picker tier must never break the warm
                    import structlog as _structlog
                    _structlog.get_logger(__name__).debug(
                        "quantised streaming capability skipped: %s", _qs_exc
                    )

    if _post_warm_retired(generation):
        return
    _start_linked_folder_auto_sync(generation)

    try:
        from core import chat_originals
        from core.training.account_jobs import startup_reconciliation_accounts
        from utils.account_context import run_as

        for account in startup_reconciliation_accounts():
            if _post_warm_retired(generation):
                return
            run_as(account, chat_originals.sweep, True)
    except Exception:  # noqa: BLE001
        pass

    # Last: pure latency work (~5s diffusers import), only with image/video models.
    if _post_warm_retired(generation):
        return
    try:
        prewarm_diffusers_if_image_models_exist()
    except Exception as _prewarm_exc:  # noqa: BLE001 -- latency work must never end the worker
        import structlog as _structlog
        _structlog.get_logger(__name__).debug("diffusers prewarm skipped: %s", _prewarm_exc)


def clear_compiled_cache_unless_shared(app: FastAPI) -> None:
    """Clear the compiled cache unless a sibling backend of this install is live. The decision lives in
    cache_cleanup, next to the paths it clears and the lock that serializes it against a sibling's startup;
    run_server puts the probe on app.state because main.py must not import run.py back."""
    _clear_compiled_cache_unless_shared(getattr(app.state, "live_sibling_backend", None))


def banner_autofill_available(app_state, environ) -> bool:
    """Whether _inject_bootstrap will hand the login page the credential.

    Read from the launch, not the environment: run_server sets UNSLOTH_API_ONLY and never
    clears it, and an embedded host may call run_server() again in the same process with
    different flags, so the variable outlives the launch that set it. The environment is
    only the fallback for a direct uvicorn launch that never went through run_server.
    """
    if getattr(app_state, "suppress_bootstrap_injection", False):
        return False
    api_only = getattr(app_state, "api_only", None)
    if api_only is None:
        api_only = environ.get("UNSLOTH_API_ONLY") == "1"
    return not api_only or _desktop_owner() is not None


def bootstrap_banner_lines(
    username: str, bootstrap_path, password: Optional[str], *, autofill_available: bool
) -> "list[str]":
    """The first-boot banner for a freshly created admin account.

    Printing the password is the exception, not the rule: _inject_bootstrap fills the
    login form in, so a launch that gets the injection must keep the credential out of
    a log that ends up in a bug report.
    """
    lines = ["=" * 60, "DEFAULT ADMIN ACCOUNT CREATED", f"    username: {username}"]
    if autofill_available or not password:
        lines.append(f"    password saved to: {bootstrap_path}")
    else:
        lines.append(f"    password: {password}")
        lines.append(f"    also saved to: {bootstrap_path}")
    lines.append("    Open the Unsloth UI to sign in and change it.")
    lines.append("=" * 60)
    return lines


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: detect hardware, seed default admin if needed. Shutdown: clean up compiled cache."""

    import time as _time

    _lifespan_started = _time.perf_counter()
    import structlog as _structlog

    _lifespan_log = _structlog.get_logger(__name__)
    clear_compiled_cache_unless_shared(app)

    # Without a catch-all the engine paths answer 405, read as 'exists'.
    if not getattr(app.state, "frontend_mounted", False):
        try:
            from routes.llama_compat import add_get_denials
            add_get_denials(app)
        except Exception:  # noqa: BLE001 -- never block startup over a discovery route
            _lifespan_log.warning("could not install API-only probe denials", exc_info = True)

    try:
        from core.inference.tools import (
            migrate_legacy_sandbox_in_background,
            start_sandbox_recovery,
        )
        migrate_legacy_sandbox_in_background()
        start_sandbox_recovery()
    except Exception:  # noqa: BLE001
        pass

    try:
        from core.inference.os_sandbox import start_tool_isolation_warmup
        start_tool_isolation_warmup()
    except Exception:  # noqa: BLE001 -- the first tool call probes instead
        _lifespan_log.warning("could not start the sandbox warm-up", exc_info = True)

    try:
        from hub.services.models.account_access import adopt_unnamed_public_proofs
        from utils.hub_settings import operator_hf_endpoint
        adopt_unnamed_public_proofs(operator_hf_endpoint())
    except Exception:  # noqa: BLE001 -- unnamed proofs are then only ignored
        _lifespan_log.warning("could not name recorded public-repo proofs", exc_info = True)

    overlay_dir = Path(__file__).resolve().parent.parent.parent / ".venv_overlay"
    if overlay_dir.is_dir():
        shutil.rmtree(overlay_dir, ignore_errors = True)

    # Before the first writer, so startup's own connections stop checkpointing too.
    try:
        from storage.studio_db import open_wal_keeper
        open_wal_keeper()
    except Exception as exc:
        _lifespan_log.warning("studio.db WAL keeper failed at startup: %s", exc)

    from utils.account_context import OWNER as _owner_account, run_as as _run_as

    try:
        from core.training.account_jobs import startup_reconciliation_accounts
        _reconcile_accounts = startup_reconciliation_accounts()
    except Exception as exc:
        _lifespan_log.warning("could not enumerate accounts to reconcile: %s", exc)
        _reconcile_accounts = [_owner_account]

    for _account in _reconcile_accounts:
        try:
            from storage.studio_db import cleanup_orphaned_runs
            _run_as(_account, cleanup_orphaned_runs)
        except Exception as exc:
            _lifespan_log.warning("cleanup_orphaned_runs failed at startup: %s", exc)

        try:
            from storage.chat_generation_runs_db import reconcile_orphaned_runs
            reconciled_chat_runs = _run_as(_account, reconcile_orphaned_runs)
            if reconciled_chat_runs:
                _lifespan_log.warning(
                    "Marked %s interrupted chat generation run(s) failed after restart.",
                    reconciled_chat_runs,
                )
        except Exception as exc:
            _lifespan_log.warning("chat generation orphan reconciliation failed: %s", exc)

        try:
            from storage.rag_db import reconcile_orphaned_ingestion_jobs
            _run_as(_account, reconcile_orphaned_ingestion_jobs)
        except Exception as exc:
            _lifespan_log.warning("reconcile_orphaned_ingestion_jobs failed at startup: %s", exc)

    try:
        from core.inference.chat_generation_runs import start_lease_sweeper
        start_lease_sweeper(app)
    except Exception as exc:
        _lifespan_log.warning("chat generation lease sweeper failed to start: %s", exc)

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

    app.state.llama_cpp_capabilities = None
    app.state.llama_cpp_freshness = None
    _start_llama_cpp_probes_if_enabled(app)

    _start_helper_precache_if_enabled()

    from core.inference.audio_inputs import start_sweeper as _start_audio_input_sweeper

    _start_audio_input_sweeper()

    from core.research_runs import ResearchSupervisor

    app.state.research_supervisor = ResearchSupervisor(app)
    app.state.research_supervisor.start()

    from core.inference.chat_generation_runs import ChatGenerationSupervisor

    app.state.chat_generation_supervisor = ChatGenerationSupervisor(app)

    from core.inference.llama_keepwarm import idle_unload_loop, sweep_slot_save_dir

    sweep_slot_save_dir()
    app.state.idle_unload_task = asyncio.create_task(idle_unload_loop())

    # Initialize RSA key pair for API key encryption (external providers).
    from core.inference.key_exchange import init_key_pair

    init_key_pair()

    from utils.stall_watchdog import stand_down_for_the_warm, start_stall_watchdog

    start_stall_watchdog(asyncio.get_running_loop(), suppress = stand_down_for_the_warm)

    _lifespan_log.info(
        "lifespan pre-auth setup completed in %.1fms",
        (_time.perf_counter() - _lifespan_started) * 1000,
    )

    # Never capture the bootstrap password when a public URL serves the default credential.
    _suppress_bootstrap = getattr(app.state, "suppress_bootstrap_injection", False)
    _created = storage.ensure_default_admin()
    app.state.bootstrap_password = None if _suppress_bootstrap else storage.get_bootstrap_password()
    # The tunnel pre-bind gate may have seeded the account already, so _created alone is not enough.
    if (_created or storage.admin_created_this_process()) and storage.requires_password_change(
        storage.DEFAULT_ADMIN_USERNAME
    ):
        bootstrap_path = storage.DB_PATH.parent / ".bootstrap_password"
        _autofill = banner_autofill_available(app.state, os.environ)
        print(
            "\n"
            + "\n".join(
                bootstrap_banner_lines(
                    storage.DEFAULT_ADMIN_USERNAME,
                    bootstrap_path,
                    storage.get_bootstrap_password(),
                    autofill_available = _autofill,
                )
            )
            + "\n"
        )

    # Last, so it never contends for the GIL before the socket binds.
    start_background_warm()
    _start_post_warm_thread()

    _lifespan_log.info(
        "lifespan startup completed in %.1fms",
        (_time.perf_counter() - _lifespan_started) * 1000,
    )
    from core.inference.api_monitor import api_monitor as _api_monitor
    from storage.api_usage_db import (
        acquire_api_usage_writer as _acquire_api_usage_writer,
        enqueue_api_usage as _enqueue_api_usage,
        release_api_usage_writer as _release_api_usage_writer,
    )

    _api_usage_writer_lease = _acquire_api_usage_writer()
    _api_usage_callback_lease = _api_monitor.acquire_terminal_callback(_enqueue_api_usage)
    yield

    _api_monitor.release_terminal_callback(_api_usage_callback_lease)

    # Before any shutdown await: a warm finishing during one would still read the lifespan as current.
    _stop_post_warm_thread()

    # Before teardown blocks the loop, or shutdown dumps as a stall.
    from utils.stall_watchdog import stop_stall_watchdog

    stop_stall_watchdog()

    # Retire the warm at shutdown entry too, or startup imports continue during shutdown awaits.
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

    from core.systemone.laya_runtime import shutdown as shutdown_decisions

    await asyncio.to_thread(shutdown_decisions)

    await run_lifespan_shutdown(
        terminate_hub_downloads,
        lambda: clear_compiled_cache_unless_shared(app),
        _hw_module,
    )
    reset_background_warm()

    # Last, so every other shutdown step has had its final write first.
    from storage.studio_db import close_wal_keeper

    close_wal_keeper()


app = FastAPI(
    title = "Unsloth UI Backend",
    version = UNSLOTH_VERSION,
    description = "Backend API for Unsloth UI - Training and Model Management",
    lifespan = lifespan,
    # Docs UIs re-registered on vendored assets: the CDN copies would run on the token origin.
    docs_url = None,
    redoc_url = None,
    swagger_ui_oauth2_redirect_url = None,
)
app.state.secure = os.environ.get("UNSLOTH_SECURE") == "1"

from fastmcp.utilities.lifespan import combine_lifespans  # noqa: E402

# Mounted ahead of /mcp, which would otherwise swallow this path.
_decisions_mcp_app = decisions_mcp.http_app(path = "/", stateless_http = True, json_response = True)
app.router.lifespan_context = combine_lifespans(lifespan, _decisions_mcp_app.lifespan)
app.mount(DECISIONS_MCP_PATH, RequireStudioAuth(_decisions_mcp_app))

# The MCP surface is opt-in: it can start GPU jobs and write model artifacts.
if os.environ.get("UNSLOTH_STUDIO_ENABLE_MCP") == "1":
    from mcp_server import BearerTokenMiddleware, create_studio_mcp

    _studio_mcp_app = create_studio_mcp().http_app(path = "/")
    _studio_mcp_lifespan = _studio_mcp_app.lifespan
    _mcp_token = os.environ.get("UNSLOTH_STUDIO_MCP_TOKEN")
    if not _mcp_token:
        raise RuntimeError("UNSLOTH_STUDIO_MCP_TOKEN is required when MCP is enabled")
    _studio_mcp_app = BearerTokenMiddleware(_studio_mcp_app, _mcp_token)
    app.router.lifespan_context = combine_lifespans(
        app.router.lifespan_context, _studio_mcp_lifespan
    )
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
                supervisor.note_server_address(scope.get("server"))
        await self.app(scope, receive, send)


app.add_middleware(ResearchPortMiddleware)


from starlette.datastructures import MutableHeaders  # noqa: E402


_CSP_SCRIPT_NONCE_HEADER = "x-internal-script-nonce"
_ARTIFACT_PREVIEW_FRAME_PATH = "/api/inference/artifact-preview-frame"
# Framed shells: their own CSP frame-ancestors governs embedding, so no X-Frame-Options DENY.
_FRAME_SHELL_PATHS = frozenset(
    {
        _ARTIFACT_PREVIEW_FRAME_PATH,
        "/api/inference/mcp-app-frame",
        _browser_routes.BROWSER_FRAME_PATH,
        _browser_routes.BROWSER_PRINT_PATH,
    }
)
_DOCS_FONT_CSS = "https://fonts.googleapis.com"
_DOCS_FONT_FILES = "https://fonts.gstatic.com"
_DOCS_PATHS = frozenset({"/docs", "/docs/oauth2-redirect", "/redoc"})
_DOCS_ASSETS_URL = "/docs-assets"
_DOCS_ASSETS_DIR = Path(__file__).parent / "assets" / "docs_ui"


import importlib.util as _importlib_util

_IS_COLAB = os.path.isdir("/content") and (
    bool(os.environ.get("COLAB_BACKEND_URL"))
    or bool(os.environ.get("COLAB_JUPYTER_IP"))
    or _importlib_util.find_spec("google.colab") is not None
)


def _reportable_hf_endpoints(request) -> dict:
    """The endpoints to hand the browser, which are not always the ones we use.

    A loopback endpoint names a proxy on the MACHINE THE BACKEND RUNS ON, and a
    private-network one an address on the backend's LAN. Handing either to a
    browser elsewhere makes it fetch its OWN localhost or its OWN 10.0.0.5: the
    calls either fail, or hit an unrelated service that, if it answers the CORS
    preflight, is handed the user's Hub bearer token. endpoint_is_reachable_by
    holds the rule; the backend keeps using its own value either way.

    The client comes from client_ip(), not the socket peer: through the managed
    Cloudflare tunnel the peer IS loopback, being the local cloudflared process
    rather than the visitor, and an address it cannot determine reads as remote.
    """
    from utils.hub_settings import saved_only_endpoints

    # A saved endpoint stays behind the owner-only settings route.
    hidden = saved_only_endpoints()
    reported = {}
    for key, value in (
        ("hf_endpoint", browser_hf_endpoint()),
        ("hf_datasets_server", get_hf_datasets_server()),
    ):
        if value not in hidden and _endpoint_is_reachable_by(value, client_ip(request)):
            reported[key] = value
        else:
            reported[key] = _HF_ENDPOINT_DEFAULTS[key]
    return reported


def _build_csp(script_nonce: "str | None" = None, *, docs: bool = False) -> str:
    script_src = "script-src 'self'"
    style_src = "style-src 'self' 'unsafe-inline'"
    worker_src = "worker-src 'self'"
    font_src = "font-src 'self' data:"
    if docs:
        # script-src deliberately untouched: docs bundles are same-origin and init runs off the nonce.
        style_src += f" {_DOCS_FONT_CSS}"
        font_src += f" {_DOCS_FONT_FILES}"
        worker_src += " blob:"
    if script_nonce:
        script_src += f" 'nonce-{script_nonce}'"
    # Colab frames span multi-level subdomains and null origins; sandboxed single user.
    frame_ancestors = "*" if _IS_COLAB else "'none'"

    # Mirror must be in connect-src; origins only since a path source matches exactly.
    hf_connect_src = " ".join(
        dict.fromkeys(
            (
                "https://huggingface.co",
                "https://datasets-server.huggingface.co",
                *csp_connect_sources(),
            )
        )
    )
    asset_sources = csp_asset_sources()
    hf_asset_src = (" " + " ".join(asset_sources)) if asset_sources else ""

    if _IS_COLAB:
        script_src += " https://*.prod.colab.dev https://*.googleusercontent.com"
        connect_src = (
            f"'self' blob: data: {hf_connect_src} "
            "https://*.prod.colab.dev wss://*.prod.colab.dev "
            "https://*.googleusercontent.com wss://*.googleusercontent.com"
        )
    else:
        connect_src = f"'self' {hf_connect_src}"

    return (
        "default-src 'self'; "
        f"img-src 'self' data: blob: https:{hf_asset_src}; "
        f"media-src 'self' data: blob: https:{hf_asset_src}; "
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


_CACHE_POLICY_HEADERS = ("cache-control", "expires", "etag", "last-modified")


class SecurityHeadersMiddleware:
    """Set baseline security headers; splice per-response inline-script nonces into CSP. Pure ASGI (not
    BaseHTTPMiddleware) so streaming responses are not wrapped in an anyio stream."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                raw = message.setdefault("headers", [])
                if not isinstance(raw, list):
                    raw = list(raw)
                    message["headers"] = raw
                headers = MutableHeaders(raw = raw)
                # Strip the internal nonce hand-off header so it never reaches the client.
                nonce = headers.get(_CSP_SCRIPT_NONCE_HEADER)
                if nonce is not None:
                    del headers[_CSP_SCRIPT_NONCE_HEADER]
                headers.setdefault(
                    "Content-Security-Policy",
                    _build_csp(nonce, docs = path in _DOCS_PATHS),
                )
                # Omit X-Frame-Options in Colab: DENY would block serve_kernel_port_as_iframe regardless of CSP.
                if not _IS_COLAB and path not in _FRAME_SHELL_PATHS:
                    headers.setdefault("X-Frame-Options", "DENY")
                headers.setdefault("X-Content-Type-Options", "nosniff")
                headers.setdefault("Referrer-Policy", "no-referrer")
                headers.setdefault(
                    "Permissions-Policy",
                    "camera=(), microphone=(self), geolocation=()",
                )
                # Without no-store Chromium/WebView2 disk-cache every poll (~200 KB per 2 s idle).
                if path.startswith("/api/") and not any(
                    name in headers for name in _CACHE_POLICY_HEADERS
                ):
                    headers["Cache-Control"] = "no-store"
                headers["server"] = "unsloth-studio"
            await send(message)

        await self.app(scope, receive, send_wrapper)


app.add_middleware(SecurityHeadersMiddleware)


# Docs served from this origin: CDN scripts could read tokens in localStorage.
import secrets as _secrets_for_docs  # noqa: E402
from fastapi.openapi.docs import (  # noqa: E402
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)

# fastapi is unpinned: match the tag robustly so a reflowed template does not 500.
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
        """Prefix with the mount point, as FastAPI's own docs routes do. Behind a path-stripping proxy (or
        `uvicorn --root-path`) the browser sees the prefix the server never does, so an unprefixed URL
        escapes the mapping and 404s."""
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
        html = get_swagger_ui_oauth2_redirect_html().body.decode()
        return _nonced_docs_response(html, tag = _OAUTH2_REDIRECT_TAG)

    @app.get("/redoc", include_in_schema = False)
    async def redoc_html(request: Request):
        assets = _docs_url(request, _DOCS_ASSETS_URL)
        return HTMLResponse(
            get_redoc_html(
                openapi_url = _docs_url(request, app.openapi_url),
                title = f"{app.title} - ReDoc",
                redoc_js_url = f"{assets}/redoc.standalone.js",
                redoc_favicon_url = f"{assets}/favicon-32x32.png",
            ).body.decode()
        )


import json as _json_for_413  # noqa: E402
from utils.upload_limits import (  # noqa: E402
    AUDIO_INPUT_MAX_BYTES,
    STT_AUDIO_JSON_MAX_BYTES,
    STT_AUDIO_RAW_MAX_BYTES,
    LIBRARY_UPLOAD_MAX_BYTES,
    UNSTRUCTURED_RECIPE_UPLOAD_MAX_BYTES,
    VIDEO_INPUT_REFERENCE_JSON_MAX_BYTES,
    VIDEO_INPUT_REFERENCE_MAX_BYTES,
    default_request_body_limit_bytes,
    upload_request_limit_bytes,
)

_BODY_PROTECTED_PREFIXES = (
    "/v1",
    "/p/",
    "/api/inference",
    "/api/picker",
    "/api/data-recipe",
    "/api/datasets",
    "/api/hub",
    "/api/chat",
    "/api/settings",
    "/api/train",
    "/api/export",
    "/api/library",
    "/api/browser",
    # Unauthenticated (login, refresh): every route takes a few hundred bytes of JSON.
    "/api/auth",
    "/mcp",
    # Everything else under /api. FastAPI reads a body before the route's auth dependency runs, so an unlisted
    # prefix let an unauthenticated client stream an unbounded body into memory.
    "/api/",
)
_DATASET_UPLOAD_PASSTHROUGH_PREFIXES = (
    "/api/datasets/upload",
    "/api/hub/datasets/upload",
)
_DATA_RECIPE_UNSTRUCTURED_UPLOAD_PASSTHROUGH_PREFIX = (
    "/api/data-recipe/seed/upload-unstructured-file"
)
_DIFFUSION_DATASET_UPLOAD_PATH = "/api/train/diffusion/dataset"
_STT_MULTIPART_UPLOAD_PATHS = (
    "/v1/audio/transcriptions",
    "/api/inference/audio/transcriptions",
    "/v1/audio/translations",
    "/api/inference/audio/translations",
)
_VIDEO_MULTIPART_UPLOAD_PATHS = (
    "/v1/videos",
    "/api/inference/videos",
)
_LIBRARY_UPLOAD_PATH = "/api/library/uploads"
# RAG document uploads (knowledge base, thread, project): multipart, capped by RAG_MAX_UPLOAD_BYTES in the route and
# spooled by FastAPI, so they pass through on Content-Length instead of being held in memory here.
_RAG_DOCUMENT_UPLOAD_RE = _re.compile(
    r"^/api/rag/(?:knowledge-bases|threads|projects)/[^/]+/documents/?$"
)
# Streamed to disk and capped by the route itself; buffering here would hold 200 MiB in memory.
_AUDIO_INPUT_UPLOAD_PATHS = ("/api/inference/audio/inputs", "/v1/audio/inputs")
_BODY_UPLOAD_PASSTHROUGH_PREFIXES = (
    *_DATASET_UPLOAD_PASSTHROUGH_PREFIXES,
    _DATA_RECIPE_UNSTRUCTURED_UPLOAD_PASSTHROUGH_PREFIX,
)
_BODY_UPLOAD_PASSTHROUGH_EXACT_PATHS = (
    _DIFFUSION_DATASET_UPLOAD_PATH,
    *_AUDIO_INPUT_UPLOAD_PATHS,
    *_STT_MULTIPART_UPLOAD_PATHS,
    *_VIDEO_MULTIPART_UPLOAD_PATHS,
    _LIBRARY_UPLOAD_PATH,
)
# Runs before auth, so only small bounded uploads may omit Content-Length; others keep 411.
_CHUNKED_UPLOAD_EXACT_PATHS = _VIDEO_MULTIPART_UPLOAD_PATHS


def _get_upload_passthrough_request_max_bytes(path: str) -> int:
    if path.startswith(_DATA_RECIPE_UNSTRUCTURED_UPLOAD_PASSTHROUGH_PREFIX):
        return upload_request_limit_bytes(UNSTRUCTURED_RECIPE_UPLOAD_MAX_BYTES)
    if path.rstrip("/") in _STT_MULTIPART_UPLOAD_PATHS:
        return upload_request_limit_bytes(STT_AUDIO_RAW_MAX_BYTES)
    if path.rstrip("/") in _VIDEO_MULTIPART_UPLOAD_PATHS:
        return max(
            upload_request_limit_bytes(VIDEO_INPUT_REFERENCE_MAX_BYTES),
            VIDEO_INPUT_REFERENCE_JSON_MAX_BYTES,
        )
    if path.rstrip("/") == _LIBRARY_UPLOAD_PATH:
        return upload_request_limit_bytes(LIBRARY_UPLOAD_MAX_BYTES)
    if path.rstrip("/") in _AUDIO_INPUT_UPLOAD_PATHS:
        return AUDIO_INPUT_MAX_BYTES
    if _RAG_DOCUMENT_UPLOAD_RE.match(path):
        from core.rag import config as _rag_config

        # RAG_MAX_UPLOAD_BYTES=0 means no cap, as the route treats it.
        if _rag_config.MAX_UPLOAD_BYTES <= 0:
            return sys.maxsize
        return upload_request_limit_bytes(_rag_config.MAX_UPLOAD_BYTES)
    # The trailing-slash variant reaches this middleware BEFORE the router's redirect_slashes
    # 307, so it must resolve to the same cap. JSON sub-routes keep extra path components.
    if (
        path.startswith(_DATASET_UPLOAD_PASSTHROUGH_PREFIXES)
        or path.rstrip("/") == _DIFFUSION_DATASET_UPLOAD_PATH
    ):
        return upload_request_limit_bytes()
    return default_request_body_limit_bytes()


def _get_request_body_max_bytes(path: str) -> int:
    if path.startswith("/api/inference/audio/transcribe/raw"):
        return STT_AUDIO_RAW_MAX_BYTES
    if path.startswith("/api/inference/audio/transcribe"):
        return STT_AUDIO_JSON_MAX_BYTES
    # multipart headroom over the raw stt cap for the openai transcription/translation routes
    if path.rstrip("/") in _STT_MULTIPART_UPLOAD_PATHS:
        return upload_request_limit_bytes(STT_AUDIO_RAW_MAX_BYTES)
    if path.rstrip("/") in _VIDEO_MULTIPART_UPLOAD_PATHS:
        return max(
            upload_request_limit_bytes(VIDEO_INPUT_REFERENCE_MAX_BYTES),
            VIDEO_INPUT_REFERENCE_JSON_MAX_BYTES,
        )
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
        chunked_upload_exact_paths: tuple = (),
        upload_passthrough_pattern = None,
    ):
        self.app = app
        self.max_bytes_getter = max_bytes_getter
        self.protected_prefixes = protected_prefixes
        self.request_max_bytes_getter = request_max_bytes_getter
        self.upload_passthrough_prefixes = upload_passthrough_prefixes
        self.upload_passthrough_max_bytes_getter = upload_passthrough_max_bytes_getter
        self.upload_passthrough_exact_paths = upload_passthrough_exact_paths
        self.chunked_upload_exact_paths = chunked_upload_exact_paths
        # Uploads whose path carries an id (RAG documents), matched by a compiled pattern.
        self.upload_passthrough_pattern = upload_passthrough_pattern

    def _is_upload_passthrough(self, path: str) -> bool:
        # Exact paths also match their trailing-slash variant (this runs before redirect_slashes).
        return (
            path.rstrip("/") in self.upload_passthrough_exact_paths
            or any(path.startswith(p) for p in self.upload_passthrough_prefixes)
            or (
                self.upload_passthrough_pattern is not None
                and self.upload_passthrough_pattern.match(path) is not None
            )
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
        # Under `--root-path /x`, uvicorn keeps the prefix in `path`; match on the route path the router sees.
        root_path = scope.get("root_path") or ""
        if root_path and path.startswith(root_path):
            path = path[len(root_path) :] or "/"
        # DELETE too: several routes take a JSON body on DELETE (delete-cached, bulk thread delete).
        if method not in ("POST", "PUT", "PATCH", "DELETE") or not any(
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

        if method != "DELETE" and self._is_upload_passthrough(path):
            upload_max_bytes = self._upload_passthrough_max_bytes(path)
            if declared is not None:
                if declared > upload_max_bytes:
                    await _send_413(send, declared, upload_max_bytes)
                    return
                await self.app(scope, receive, send)
                return
            if path.rstrip("/") not in self.chunked_upload_exact_paths:
                await _send_411(send)
                return
            max_bytes = upload_max_bytes

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
    chunked_upload_exact_paths = _CHUNKED_UPLOAD_EXACT_PATHS,
    upload_passthrough_pattern = _RAG_DOCUMENT_UPLOAD_RE,
)

from core.inference.llama_keepwarm import LlamaKeepWarmMiddleware  # noqa: E402

app.add_middleware(LlamaKeepWarmMiddleware)


from starlette.responses import RedirectResponse as _RedirectResponse  # noqa: E402


@app.get("/recipes", include_in_schema = False)
@app.get("/recipes/{rest:path}", include_in_schema = False)
async def _recipes_redirect(rest: str = ""):
    target = "/data-recipes" + (("/" + rest) if rest else "")
    return _RedirectResponse(url = target, status_code = 308)


from utils.host_policy import (
    cors_origin_regex_for_mode,
    cors_origins_for_mode,
)  # noqa: E402


class RemoteAccessCORSMiddleware(CORSMiddleware):
    """Admit the published Cloudflare origin, on top of the startup allowlist."""

    def __init__(self, cors_app, *, remote_access_state, **kwargs):
        self.remote_access_state = remote_access_state
        super().__init__(cors_app, **kwargs)

    def is_allowed_origin(self, origin: str) -> bool:
        # The tunnel admits ONE origin; api-only stays locked to the Tauri app.
        published = getattr(self.remote_access_state, "cloudflare_url", None)
        if published:
            tunnel_origin = _origin_of(published)
            if tunnel_origin is not None and tunnel_origin == _origin_of(origin):
                return True
        return super().is_allowed_origin(origin)


_cors_origins = cors_origins_for_mode(
    api_only = os.environ.get("UNSLOTH_API_ONLY") == "1",
    secure = os.environ.get("UNSLOTH_SECURE") == "1",
)
_cors_origin_regex = cors_origin_regex_for_mode(
    api_only = os.environ.get("UNSLOTH_API_ONLY") == "1",
    secure = os.environ.get("UNSLOTH_SECURE") == "1",
)

app.add_middleware(
    RemoteAccessCORSMiddleware,
    remote_access_state = app.state,
    allow_origins = _cors_origins,
    allow_origin_regex = _cors_origin_regex,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
    # allow_headers is the REQUEST side; a response header is unreadable to JS unless
    # exposed, and Studio is cross-origin from tauri://localhost and tunnels.
    expose_headers = [
        "X-Unsloth-Conflict-Kind",
        "X-Unsloth-Refusal",
        "x-typesafe-request-id",
        "X-Unsloth-Monitor-ID",
        *_hub_endpoint_proxy.EXPOSED_HEADERS,
        *_browser_routes.EXPOSED_HEADERS,
    ],
    # Short preflight cache: a cached preflight outlives a stopped tunnel.
    max_age = 60,
)

from utils.keyless_api_access import KeylessToolPolicyMiddleware  # noqa: E402

app.add_middleware(KeylessToolPolicyMiddleware)

from utils.remote_access_settings import RemoteAccessStopResponseMiddleware  # noqa: E402

app.add_middleware(RemoteAccessStopResponseMiddleware)

app.include_router(auth_router, prefix = "/api/auth", tags = ["auth"])
app.include_router(
    __import__("routes.accounts", fromlist = ["router"]).router,
    prefix = "/api/accounts",
    tags = ["accounts"],
)
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
app.include_router(inference_studio_router, prefix = "/api/inference", tags = ["inference"])

app.include_router(video_router, prefix = "/api/inference", tags = ["inference"])
app.include_router(video_openai_router, prefix = "/api/inference", tags = ["inference"])
app.include_router(video_openai_router, prefix = "/v1", tags = ["openai-compat"])

app.include_router(inference_router, prefix = "/v1", tags = ["openai-compat"])
app.include_router(systemone_router, prefix = "/v1", tags = ["systemone"])
# Must register before the SPA catch-all or /props and /version return index.html.
app.include_router(llama_compat_router, tags = ["openai-compat"])
app.include_router(preview_router, prefix = "/p", tags = ["preview"])
app.include_router(providers_router, prefix = "/api/providers", tags = ["providers"])

app.include_router(openai_codex_auth_router, prefix = "/api/providers", tags = ["providers"])

app.include_router(settings_router, prefix = "/api/settings", tags = ["settings"])
app.include_router(sandbox_capability_router, prefix = "/api/sandbox", tags = ["sandbox"])
app.include_router(mcp_servers_router, prefix = "/api/mcp/servers", tags = ["mcp"])
app.include_router(skills_router, prefix = "/api/skills", tags = ["skills"])
app.include_router(prompts_router, prefix = "/api/prompts", tags = ["prompts"])
app.include_router(library_router, prefix = "/api/library", tags = ["library"])
app.include_router(profile_stats_router, prefix = "/api/profile", tags = ["profile"])
app.include_router(datasets_router, prefix = "/api/datasets", tags = ["datasets"])
app.include_router(data_recipe_router, prefix = "/api/data-recipe", tags = ["data-recipe"])
app.include_router(llama_router, prefix = "/api/llama", tags = ["llama"])
app.include_router(engines_router, prefix = "/api/engines", tags = ["engines"])
app.include_router(whisper_router, prefix = "/api/whisper", tags = ["whisper"])
app.include_router(npu_router, prefix = "/api/npu", tags = ["npu"])
app.include_router(export_router, prefix = "/api/export", tags = ["export"])
app.include_router(external_import_router, prefix = "/api/import", tags = ["import"])
app.include_router(rag_router, prefix = "/api/rag", tags = ["rag"])
app.include_router(training_history_router, prefix = "/api/train", tags = ["training-history"])
app.include_router(hub_inventory_router, prefix = "/api/hub", tags = ["hub"])
app.include_router(hub_datasets_router, prefix = "/api/hub/datasets", tags = ["hub"])
app.include_router(picker_templates_router, prefix = "/api/picker", tags = ["picker"])
app.include_router(hub_token_router, prefix = "/api/hub", tags = ["hub"])
app.include_router(
    _build_modelscope_router(browser = True),
    prefix = _MODELSCOPE_BROWSER_PREFIX,
    include_in_schema = False,
)
for _prefix, _upstream, _pages in (
    (_hub_endpoint_proxy.HUB_PREFIX, browser_hf_endpoint, True),
    (_hub_endpoint_proxy.DATASETS_SERVER_PREFIX, get_hf_datasets_server, False),
):
    app.include_router(
        _hub_endpoint_proxy.build_router(_prefix, _upstream, anonymous_pages = _pages),
        prefix = _prefix,
        tags = ["hub"],
    )
app.include_router(youtube_router, prefix = "/api/youtube", tags = ["youtube"])
app.include_router(_browser_routes.router, prefix = "/api/browser", tags = ["browser"])

install_api_error_handlers(app)

# preflight/backend.rs probes with a non-retried 2s timeout; 1.5s measured a 1.742s worst case.
_HEALTH_DETECT_BUDGET_S = 1.0


async def _await_hardware_detection(budget: float) -> bool:
    """Wait up to ``budget`` seconds for DEVICE to be set. True iff it is. Polls on the event loop
    instead of awaiting ensure_hardware_detected() in a thread: asyncio.wait_for cannot cancel a
    to_thread, so a timed-out call holds the executor slot for the rest of the import and a polled
    endpoint would drain the pool. Returns False without kicking anything when the warm is switched
    off. Health is probed automatically, so kicking detection here would import torch on every such
    host and the switch would buy nothing."""
    if os.environ.get(DISABLE_ENV_VAR) == "1":
        return _hw_module.DETECTION_COMPLETE.is_set() and _hw_module.DEVICE is not None
    # Check event AND DEVICE: shutdown clears DEVICE before the event.
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
    """``(chat_only, chat_only_reason, chat_only_detail)`` if detection is settled, else ``None``. A
    seqlock read rather than ``_DETECT_LOCK``: that lock would park the endpoint for the whole torch
    import. A forced re-detect clears the event on the way in and bumps the generation before
    setting it again, so a read bracketed by both lands wholly before or after one pass, never
    mid-pass where CHAT_ONLY is back to True and the reason to None. That middle must not be
    published: config/env.ts caches the first reply carrying `device_type` as authoritative, and the
    sidebar's recovery poll runs only while it reads `chat_only_reason == "mlx_unavailable"`."""
    for _ in range(3):
        if not _hw_module.DETECTION_COMPLETE.is_set():
            return None
        generation = _hw_module.DETECTION_GENERATION
        device = _hw_module.DEVICE
        chat_only = bool(_hw_module.CHAT_ONLY)
        # Refreshed verdict, reason and detail from one pass (they can change after startup).
        try:
            reason, detail = _hw_module.current_chat_only_verdict()
        except Exception:
            reason = getattr(_hw_module, "CHAT_ONLY_REASON", None)
            detail = getattr(_hw_module, "CHAT_ONLY_DETAIL", None)
        if (
            device is not None
            and _hw_module.DETECTION_COMPLETE.is_set()
            and _hw_module.DETECTION_GENERATION == generation
        ):
            return chat_only, reason, detail
    return None


_MLX_PRESTART_GRACE_AFTER_WARM_S = 30.0
# Absolute backstop: a warm parked forever inside an import never reports stopped.
_MLX_PRESTART_CEILING_S = 900.0

_MLX_PRESTART_LOCK = threading.Lock()
# (generation, first hold, first-seen-stopped); grace starts when stop is first seen.
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
            _mlx_prestart_hold = (generation, first, None)
            return True
        if stopped_seen is None:
            stopped_seen = now
            _mlx_prestart_hold = (generation, first, stopped_seen)
        return now - stopped_seen < _MLX_PRESTART_GRACE_AFTER_WARM_S


def _superseded_by_mlx_repair(snapshot: Optional[tuple[bool, Optional[str]]]) -> bool:
    """True when the MLX self-heal is about to replace this settled verdict. Scoped to /api/health
    rather than folded into ``_hardware_snapshot()``: the launcher's watchdog reads /api/liveness
    and holds its startup grace open while hardware_detecting is set, so a 15-minute reinstall must
    not stretch that grace. Bounded, never open-ended. A live worker holds the verdict for as long
    as its install takes, capped by mlx_repair._WORKER_BUDGET_S. A repair that has not started yet
    is only a promise, and this is where that promise expires: the hold lasts while the warm does
    and a short handoff beyond it, under an absolute ceiling for the warm that never ends."""
    if snapshot is None:
        return False
    if not _hw_module.verdict_pending_mlx_repair(snapshot[0], snapshot[1]):
        return False
    try:
        from utils.mlx_repair import mlx_repair_started

        # Read after the predicate, so a repair claiming the latch in between stays held.
        if mlx_repair_started():
            return True
    except Exception as exc:
        logger.debug("MLX repair start check failed, holding on the pre-start window: %s", exc)
    return _mlx_prestart_hold_ok(_hw_module.DETECTION_GENERATION)


def _torch_warm_in_progress() -> bool:
    """True while the coordinated warm thread is still working through its stages. A separate field
    from ``hardware_detecting`` on purpose. Hardware detection is only ``_STAGES[0]``;
    inference_backend, transformers and datasets run after it, and those C-extension imports can
    hold the GIL for seconds at a time, so a launcher ending its startup grace on
    ``hardware_detecting`` alone ends it with the expensive half of the warm still ahead. But that
    marker also means "this hardware verdict is provisional, re-read it", and
    config/hardware-verdict.ts keeps the UI provisional and polling while it is set, so keeping it
    lit through datasets would hide Train for the whole warm. Two meanings, two fields. False
    whenever no warm thread is running, which keeps the deferred case working: with
    UNSLOTH_STUDIO_DISABLE_TORCH_WARM=1 the warm never starts, and one retired mid-stage by a
    shutdown never finishes, so deriving this from "not finished" would report warming forever."""
    status = warm_status()
    return bool(status["started"] and not status["finished"] and status["alive"])


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
    """True while at least one generation is in flight, published so the desktop health watchdog can tell a
    backend that is busy serving from one that has died: a saturated host can stall the event loop past a
    probe budget, and killing there ends a response the user is still waiting on. Failures report "not
    busy"."""
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
        # Lockstep with DESKTOP_MANAGEABILITY_VERSION in src-tauri/src/preflight/version.rs.
        "desktop_manageability_version": 2,
        "supports_desktop_auth": True,
        "supports_desktop_backend_ownership": True,
        "studio_root_id": _studio_root_id(),
        **({"desktop_owner": owner} if (owner := _desktop_owner()) else {}),
    }
    # The desktop watchdog holds its startup grace on torch_warm_in_progress (GIL-holding imports).
    if _torch_warm_in_progress():
        alive["torch_warm_in_progress"] = True
    # Watchdog widens its failure budget on this marker under heavy generation load.
    if _inference_active():
        alive["inference_active"] = True
    if _hardware_snapshot() is None:
        alive["hardware_detecting"] = True
        if os.environ.get(DISABLE_ENV_VAR) == "1":
            alive["hardware_detection_deferred"] = True
    return alive


async def _desktop_shell_subject(request: Request) -> Optional[str]:
    """Owner subject for a request carrying the desktop secret, None without one: on a multi-account install desktop-login mints no session, so the shell proves ownership with the secret itself."""
    secret = request.headers.get("x-desktop-secret")
    if secret is None:
        return None
    from starlette.concurrency import run_in_threadpool

    if await run_in_threadpool(storage.validate_desktop_secret, secret) is None:
        raise HTTPException(status_code = 401, detail = "Desktop authentication failed")
    return storage.DEFAULT_ADMIN_USERNAME


@app.get("/api/health")
async def health_check(request: Request):
    """Liveness plus launcher capability bits; host fingerprint gated on a bearer.

    Unauthenticated callers get non-sensitive fields (service, studio_root_id,
    chat_only, desktop_*, native_path_leases_supported) to re-adopt a sibling
    backend and gate UI before a token exists. version / studio_version /
    device_type require a bearer since they fingerprint the host.
    """
    await _await_hardware_detection(_HEALTH_DETECT_BUDGET_S)
    snapshot = _hardware_snapshot()
    # Hold a chat-only verdict the MLX self-heal is about to overturn.
    mlx_repairing = _superseded_by_mlx_repair(snapshot)
    if mlx_repairing:
        snapshot = None
    base = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Unsloth UI Backend",
        # Literal True, not a CHAT_ONLY read: a pass in flight sets it False before a CPU fallback.
        "chat_only": snapshot[0] if snapshot is not None else True,
        "desktop_protocol_version": 1,
        # Lockstep: see the note in /api/liveness above.
        "desktop_manageability_version": 2,
        "supports_desktop_auth": True,
        "supports_desktop_backend_ownership": True,
        "studio_root_id": _studio_root_id(),
        "native_path_leases_supported": native_path_leases_supported(),
        # Unauthenticated on purpose: an endpoint URL is not a host fingerprint.
        **_reportable_hf_endpoints(request),
        "hub_source": _active_hub_source(),
        "hub_proxy": _hub_endpoint_proxy.relay_path(
            _hub_endpoint_proxy.HUB_PREFIX,
            browser_hf_endpoint(),
            _HF_ENDPOINT_DEFAULTS["hf_endpoint"],
        ),
        "datasets_server_proxy": _hub_endpoint_proxy.relay_path(
            _hub_endpoint_proxy.DATASETS_SERVER_PREFIX,
            get_hf_datasets_server(),
            _HF_ENDPOINT_DEFAULTS["hf_datasets_server"],
        ),
        **({"desktop_owner": owner} if (owner := _desktop_owner()) else {}),
    }
    # Lockstep with /api/liveness: older launchers fall back to this route.
    if _torch_warm_in_progress():
        base["torch_warm_in_progress"] = True
    if _inference_active():
        base["inference_active"] = True
    if snapshot is None:
        base["hardware_detecting"] = True
        if os.environ.get(DISABLE_ENV_VAR) == "1" and not mlx_repairing:
            base["hardware_detection_deferred"] = True
    subject = await _desktop_shell_subject(request)
    auth = request.headers.get("authorization", "")
    bearer = auth.split(" ", 1)[1] if auth.lower().startswith("bearer ") else None
    if subject is None:
        try:
            from auth.authentication import credentials_for_token
            from auth.authentication import get_current_subject as _gcs

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
    # Separate from device_type: Intel Macs spill to system RAM; Apple Silicon has one pool.
    from utils.hardware import is_apple_silicon

    authed = {
        **base,
        "version": UNSLOTH_VERSION,
        "studio_version": STUDIO_VERSION,
        # Authed-only: these fingerprint how the host is exposed.
        "cloudflare_url": getattr(request.app.state, "cloudflare_url", None),
        "server_url": getattr(request.app.state, "server_url", None),
        "secure": bool(getattr(request.app.state, "secure", False)),
    }
    if snapshot is not None:
        authed["chat_only"] = snapshot[0]
        authed["chat_only_reason"] = snapshot[1]
        authed["chat_only_detail"] = snapshot[2]
        authed["device_type"] = device_type
        authed["apple_silicon"] = is_apple_silicon()
        from utils.paths.file_manager import file_manager_kind

        authed["file_manager"] = file_manager_kind()
        authed.pop("hardware_detecting", None)
        authed.pop("hardware_detection_deferred", None)
        # torch_warm_in_progress deliberately survives: the desktop watchdog needs it.
    else:
        # Re-detect during the bearer await: mark provisional and omit device_type.
        authed["hardware_detecting"] = True
        if mlx_repairing:
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


@app.post(
    "/api/shutdown",
    dependencies = [Depends(get_current_subject), Depends(auth_policy.require_owner)],
)
async def shutdown_server(request: Request, current_subject: str = Depends(get_current_subject)):
    """Gracefully shut down the Unsloth Studio server.

    Called by the frontend quit dialog so users can stop the server from the UI
    without the CLI or killing the process manually.
    """

    return _schedule_shutdown(request)


@app.post("/api/desktop/shutdown")
async def desktop_shutdown_server(request: Request):
    """The desktop shell's quit path, authenticated by its secret."""
    if await _desktop_shell_subject(request) is None:
        raise HTTPException(status_code = 401, detail = "Desktop authentication failed")
    return _schedule_shutdown(request)


def _schedule_shutdown(request: Request) -> dict:
    async def _delayed_shutdown():
        await asyncio.sleep(0.2)  # Let the HTTP response return first
        trigger = getattr(request.app.state, "trigger_shutdown", None)
        if trigger is not None:
            trigger()
        else:
            import signal
            import os
            os.kill(os.getpid(), signal.SIGTERM)

    request.app.state._shutdown_task = asyncio.create_task(_delayed_shutdown())
    return {"status": "shutting_down"}


def _get_cached_system_gpu_info(
    logger, *, refresh_memory: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return training and inference GPU info with bounded live-probe churn."""
    import time
    from utils.hardware import (
        get_backend_visible_gpu_info,
        get_cross_vendor_inference_gpu_info,
        get_visible_gpu_utilization,
        get_vulkan_inference_gpu_info,
    )

    global _system_gpu_cache
    now = time.monotonic()
    with _system_gpu_cache_lock:
        if not refresh_memory and _system_gpu_cache is not None:
            cached_at, cached_gpu_info = _system_gpu_cache
            if now - cached_at < _SYSTEM_GPU_CACHE_TTL_SECONDS:
                return cached_gpu_info

        try:
            visibility_info = get_backend_visible_gpu_info() or {"available": False, "devices": []}
        except Exception as e:
            logger.debug(f"Failed to get GPU visibility info: {e}")
            visibility_info = {"available": False, "devices": []}

        try:
            import contextlib

            from utils.hardware import gpu_query
            with (
                contextlib.nullcontext() if refresh_memory else gpu_query.display_reads(max_stale = 0)
            ):
                utilization_info = get_visible_gpu_utilization() or {"devices": []}
        except Exception as e:
            logger.debug(f"Failed to get GPU utilization info: {e}")
            utilization_info = {"devices": []}

        # Device indices are backend-specific: never overlay CUDA/ROCm onto Vulkan ordinals.
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
            # Apple unified memory: free is not total - used.
            enriched_dev["vram_free_gb"] = (
                reported_free_vram
                if reported_free_vram is not None
                else round(total_vram - used_vram, 2)
                if total_vram and used_vram is not None
                else None
            )
            enriched_dev["vram_utilization_pct"] = util.get(
                "vram_utilization_pct", dev.get("vram_utilization_pct")
            )
            enriched_devices.append(enriched_dev)

        # Both sides must describe the same cards or the percentage inflates (#7452).
        aggregate_basis_matches = metrics_match and {
            d.get("index") for d in utilization_info.get("devices", [])
        } == {d.get("index") for d in enriched_devices}

        try:
            from core.inference.llama_cpp import LlamaCppBackend
            from utils.hardware import DeviceType, get_device

            llama_uses_vulkan = LlamaCppBackend._is_vulkan_backend()
            if llama_uses_vulkan:
                # Vulkan ordinals live in another namespace; a failed probe must not expose torch indices.
                gpu_ids_supported = False
            else:
                # XPU pins are unsafe across Level Zero FLAT/COMPOSITE; CPU-only llama.cpp cannot pin.
                gpu_ids_supported = (
                    get_device() != DeviceType.XPU and not LlamaCppBackend._backend_lacks_gpu_lib()
                )
        except Exception as e:
            logger.debug(f"Could not resolve gpu_ids support: {e}")
            llama_uses_vulkan = False
            gpu_ids_supported = True
        gpu_info = {
            **visibility_info,
            "available": visibility_info.get("available", False),
            "devices": enriched_devices,
            "backend": visibility_info.get("backend"),
            "gguf_gpu_ids_supported": gpu_ids_supported,
            "vram_used_gb_aggregate": utilization_info.get("vram_used_gb_aggregate")
            if aggregate_basis_matches
            else None,
        }

        if visibility_info.get("backend") == "vulkan":
            gpu_info["gguf_gpu_ids_supported"] = bool(enriched_devices)
            inference_gpu_info = gpu_info
        else:
            vulkan_info = get_vulkan_inference_gpu_info()
            cross_vendor_info = (
                get_cross_vendor_inference_gpu_info() if vulkan_info is None else None
            )
            if vulkan_info is not None:
                inference_gpu_info = {
                    **vulkan_info,
                    "gguf_gpu_ids_supported": bool(vulkan_info.get("devices")),
                }
            elif cross_vendor_info is not None:
                inference_gpu_info = {**cross_vendor_info, "gguf_gpu_ids_supported": False}
            else:
                inference_gpu_info = gpu_info

        combined_info = (gpu_info, inference_gpu_info)
        _system_gpu_cache = (time.monotonic(), combined_info)
        return combined_info


def _probe_dense_quant_supported() -> bool:
    """Whether an ``auto`` request could engage a dense quant on EVERY visible card.

    The picker cannot see which card a load lands on, so a mixed host answers for the least capable.

    IMPORTS the ML stack, so only ``_refresh_dense_quant_capability`` calls it, and only from the
    post-warm worker or a request that already has both modules. Never memoised: the answer
    sharpens, since an unprobed scheme counts as usable and a later load can record a kernel
    failure in ``_SMOKE_CACHE``."""
    try:
        from core.inference.diffusion_device import resolve_diffusion_device_target
        from core.inference.diffusion_transformer_quant import dense_quant_host_capable

        import torch

        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if count <= 1:
            return bool(dense_quant_host_capable(resolve_diffusion_device_target()))
        # No device scope: cudaSetDevice pins a primary context on every card (CUDA 12).
        for ordinal in range(count):
            if not dense_quant_host_capable(resolve_diffusion_device_target(ordinal = ordinal)):
                return False
        return True
    except Exception:  # noqa: BLE001 -- a capability probe must never fail a status request
        return False


def _probe_dense_quant_schemes() -> list[str]:
    """The auto ladder's schemes for this host, best first, on the same ladder and deny list the
    loader's ``auto_scheme_candidates`` reads; a mixed host answers with the INTERSECTION. IMPORTS
    the ML stack.

    The CACHED variant, since a request holding torch reaches this from the polled ``/api/system``:
    the load-time helper runs ``_scheme_supported``, which spawns the smoke probe or allocates in
    this process. Like the capability bit, it sharpens as loads record verdicts in ``_SMOKE_CACHE``."""
    try:
        from core.inference.diffusion_device import resolve_diffusion_device_target
        from core.inference.diffusion_transformer_quant import auto_scheme_candidates_cached

        import torch

        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if count <= 1:
            return list(auto_scheme_candidates_cached(resolve_diffusion_device_target()))
        common: Optional[list[str]] = None
        for ordinal in range(count):
            schemes = list(
                auto_scheme_candidates_cached(resolve_diffusion_device_target(ordinal = ordinal))
            )
            common = schemes if common is None else [s for s in common if s in schemes]
        return common or []
    except Exception:  # noqa: BLE001 -- a capability probe must never fail a status request
        return []


_dense_quant_capability: Optional[bool] = None
_dense_quant_scheme_ladder: list[str] = []


def _refresh_dense_quant_capability() -> bool:
    """Resolve the dense-quant bit and cache it. Imports torch and torchao; never call from a route
    that has not already got them."""
    global _dense_quant_capability, _dense_quant_scheme_ladder
    _dense_quant_capability = _probe_dense_quant_supported()
    _dense_quant_scheme_ladder = _probe_dense_quant_schemes() if _dense_quant_capability else []
    return _dense_quant_capability


def _dense_quant_supported() -> bool:
    """The dense-quant bit for ``/api/system``, from already-loaded state only.

    This route is polled throughout startup, and ``import torch`` and ``import torchao.quantization``
    cost ~0.8s each and hold the GIL, the stall ``_await_hardware_detection`` already keeps off this
    path. So never import here. The post-warm worker resolves it once the warm has the stack up, and
    a request finding both modules loaded refreshes it, so a load's kernel verdict reaches the picker
    on the next poll. False before that is honest: the picker renders no fast label, as it does for
    an unknown VRAM budget."""
    if "torch" in sys.modules and "torchao" in sys.modules:
        return _refresh_dense_quant_capability()
    return bool(_dense_quant_capability)


_quantised_streaming_capability: Optional[bool] = None


def _refresh_quantised_streaming_capability() -> bool:
    """Resolve and cache the streaming bit. Imports diffusers; never call from the polled route."""
    global _quantised_streaming_capability
    from core.inference.video import h3_streamed_int8_supported

    _quantised_streaming_capability = _probe_quantised_streaming(h3_streamed_int8_supported)
    return _quantised_streaming_capability


def _probe_quantised_streaming(supported: Any) -> bool:
    """Every visible CUDA card must qualify: a load may be pinned to any of them, and one that
    resolves to float16 keeps bf16 (the same intersection as ``_probe_dense_quant_schemes``)."""
    try:
        import torch

        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if count <= 1:
            return bool(supported())
        from core.inference.diffusion_device import resolve_diffusion_device_target

        return all(
            bool(supported(resolve_diffusion_device_target(ordinal = ordinal)))
            for ordinal in range(count)
        )
    except Exception:  # noqa: BLE001 -- an unanswerable probe hides the tier
        return False


def _quantised_streaming() -> bool:
    """The streaming bit for ``/api/system``. Resolved here only once a load has already loaded every
    module it reads, since a cold warm (UNSLOTH_STUDIO_DISABLE_TORCH_WARM=1) never resolves it.
    Needs each module initialised (probing mid-load races the load), not core.inference.video (image loads skip it)."""
    if _quantised_streaming_capability is None and all(
        (module := sys.modules.get(name)) is not None
        and not getattr(getattr(module, "__spec__", None), "_initializing", False)
        for name in (
            "torch",
            "torchao",
            "diffusers",
            "diffusers.hooks",
            "diffusers.hooks.group_offloading",
        )
    ):
        try:
            return _refresh_quantised_streaming_capability()
        except Exception:  # noqa: BLE001 -- the picker then keeps the tier hidden
            return False
    return bool(_quantised_streaming_capability)


def _diffusers_offload_tiers() -> dict:
    """Extra picker fit tiers per curated Diffusers repo (lower-cased id), in the picker's GiB units.
    Torch-free; the picker unions them with the catalog's own tiers, so they can only widen."""
    try:
        from core.inference.video_minimax_h3 import h3_diffusers_fit_tiers
        tiers = h3_diffusers_fit_tiers()
    except Exception:  # noqa: BLE001 -- a picker hint must never break the polled route
        return {}
    return {"minimaxai/minimax-h3": tiers} if tiers else {}


def _nvfp4_diffusion_enabled() -> bool:
    """Whether image and video generation may offer NVFP4 (``UNSLOTH_NVFP4_DIFFUSION``)."""
    try:
        from core.inference.diffusion_nvfp4_flag import nvfp4_diffusion_enabled
        return nvfp4_diffusion_enabled()
    except Exception:  # noqa: BLE001 -- a capability read must never fail a status request
        return False


def _dense_quant_schemes() -> list[str]:
    """The scheme ladder for ``/api/system``, a pure read of already-resolved state: the polled route
    must never import torch, and the entry beside it refreshed both in one pass."""
    return list(_dense_quant_scheme_ladder)


@app.get("/api/system")
def get_system_info(
    current_subject: str = Depends(get_current_subject), refresh_memory: bool = False
):
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

    gpu_info, inference_gpu_info = _get_cached_system_gpu_info(
        logger, refresh_memory = refresh_memory
    )

    memory = psutil.virtual_memory()
    memory_total = memory.total
    memory_available = memory.available
    memory_percent = memory.percent
    # The picker's RAM tiers compare against available_gb: publish the cgroup-capped view.
    try:
        from utils import host_memory

        _budgets = host_memory.cgroup_memory_budgets()
        _headroom_mib = host_memory.cgroup_headroom_mib(_budgets)
        _limit_mib = host_memory.cgroup_limit_mib(_budgets)
        if _limit_mib is not None:
            memory_total = min(memory_total, _limit_mib * 1024**2)
        if _headroom_mib is not None:
            memory_available = min(memory_available, _headroom_mib * 1024**2)
        if _limit_mib is not None or _headroom_mib is not None:
            memory_available = min(memory_available, memory_total)
            memory_percent = (
                round((memory_total - memory_available) / memory_total * 100, 1)
                if memory_total
                else memory_percent
            )
    except Exception as e:
        logger.debug(f"Failed to read the cgroup memory limit: {e}")

    # Corrects psutil's 1000x-too-small Apple Silicon M4+ reading (issue #8519).
    cpu_freq_mhz = cpu_frequency_mhz()

    try:
        disk = psutil.disk_usage(os.path.abspath(os.sep))
    except Exception as e:
        logger.debug(f"Failed to get disk usage: {e}")
        disk = None

    from utils.system_disk import cached_models_disk_usage

    models_disk = cached_models_disk_usage()

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
        "memory_refreshed": refresh_memory,
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
            "total_gb": round(memory_total / 1024**3, 2),
            "available_gb": round(memory_available / 1024**3, 2),
            "percent_used": memory_percent,
            "process_used_mb": process_used_mb,
        },
        "disk": {
            "total_gb": round(disk.total / 1e9, 2) if disk else 0,
            "free_gb": round(disk.free / 1e9, 2) if disk else 0,
            "percent_used": disk.percent if disk else 0,
        },
        "models_disk": models_disk,
        "gpu": gpu_info,
        "inference_gpu": inference_gpu_info,
        "ml_packages": ml_packages,
        **export_capability(),
        **video_capability(),
        "dense_quant_supported": _dense_quant_supported(),
        "dense_quant_schemes": _dense_quant_schemes(),
        "quantised_streaming": _quantised_streaming(),
        "diffusers_offload_tiers": _diffusers_offload_tiers(),
        "nvfp4_diffusion": _nvfp4_diffusion_enabled(),
    }


@app.get("/api/system/disk")
def get_disk_space(
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    """Free space where downloads land. One syscall, and nothing else.

    Separate from /api/system because that route enumerates GPUs, reads package metadata and
    samples CPU: fine for a screen the user is looking at, far too much to run on the chance
    that a disk is filling. The low-disk notice asks for THIS instead, and only when a download
    is about to start.

    shutil.disk_usage is statvfs on Linux and macOS and GetDiskFreeSpaceExW on Windows, so this
    is microseconds on every platform Studio runs on and needs no directory walk. Note that on
    Windows it reports the quota available to the CALLING user, which is the number that decides
    whether the download fits, so that difference is the correct one.

    Measured at the models root rather than the filesystem root: those are different volumes
    whenever HF_HUB_CACHE, or the Studio root, sits on another disk, and the free space that
    matters is the one the bytes are going to.

    Both roots that receive bytes are measured, not just the hub one. HF_XET_CACHE is resolved
    independently of HF_HUB_CACHE and holds the Xet chunks every download now streams through,
    so the two can sit on different volumes and the wrong one has ample room. The TIGHTEST
    reading wins, because the volume that runs out first is the one that stops the download.
    Deduplicated by device, so the ordinary install where both live on one disk still costs a
    single syscall.
    """
    from utils.paths.storage_roots import hf_default_cache_dir, studio_root

    # ACTIVE caches, not defaults: hf_default_cache_dir() ignores HF_HUB_CACHE and Models Folder.
    roots = []
    try:
        from utils.hf_cache_settings import get_hf_cache_paths
        paths = get_hf_cache_paths()
        roots.extend((paths.hub_cache, paths.xet_cache))
    except Exception as exc:  # noqa: BLE001 - a settings read must not cost the reading
        logger.debug(f"Could not resolve the active caches for the disk reading: {exc}")

    def _locate(probe):
        """(first existing ancestor, its device), or None when this root is unreadable.

        Two failures that look alike and must not be treated alike. A MISSING directory is
        ordinary, since the cache legitimately does not exist yet on a fresh install, and the
        volume it would live on is its nearest existing parent. A permission error, an I/O
        error or a network mount that is not answering is not missing: climbing past it would
        report the parent filesystem's free space for a disk nothing could read, which is the
        confidently wrong answer this route exists to avoid. Those leave the root unreadable.
        """
        try:
            chain = [probe, *probe.parents]
        except (OSError, ValueError, RuntimeError):
            return None
        for candidate in chain:
            try:
                return candidate, os.stat(candidate).st_dev
            except (FileNotFoundError, NotADirectoryError):
                continue
            except OSError as exc:
                logger.debug(f"Cache root {candidate} could not be read: {exc}")
                return None
        return None

    def _read(candidate):
        """The reading for one already-located directory, or None if it cannot be taken."""
        try:
            usage = shutil.disk_usage(candidate)
        except (OSError, ValueError):
            return None
        return {
            "path": str(candidate),
            "total_gb": round(usage.total / 1e9, 2),
            "free_gb": round(usage.free / 1e9, 2),
            "percent_used": (
                round((usage.total - usage.free) / usage.total * 100, 1) if usage.total else 0
            ),
        }

    located = []
    seen = set()
    unreadable = False
    for root in roots:
        found = _locate(root)
        if found is None:
            # Unreadable ACTIVE destination makes the whole answer unknown (Hub and Xet may differ).
            unreadable = True
            continue
        candidate, device = found
        if device in seen:
            continue
        seen.add(device)
        located.append(candidate)

    readings = [] if unreadable else [r for r in map(_read, located) if r is not None]
    if not unreadable and len(readings) != len(located):
        unreadable = True
        readings = []

    if unreadable:
        # Nulls, not zeros, and not a fallback volume either: the caller reads this as
        # "could not tell", which neither warns nor blocks a download.
        return {"path": None, "total_gb": None, "free_gb": None, "percent_used": None}

    if not readings:
        for probe in (hf_default_cache_dir(), studio_root(), Path(os.path.abspath(os.sep))):
            found = _locate(probe)
            reading = None if found is None else _read(found[0])
            if reading is not None:
                readings.append(reading)
                break

    if readings:
        tightest = min(readings, key = lambda r: r["free_gb"])
        answer = dict(tightest)
        # Redact the path for non-owners (API keys, managed accounts); use the INVENTORY redactor,
        # since redact_host_paths leaves a field named 'path' alone.
        from utils.account_context import is_owner_context

        return redact_inventory_host_paths(
            answer, via_api_key = via_api_key or not is_owner_context()
        )
    # Nulls, not zeros: diskPressure() reads 0 total as failure and 0 free as full.
    return {"path": None, "total_gb": None, "free_gb": None, "percent_used": None}


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
        **export_capability(),
        **video_capability(),
    }
    if include_details:
        from utils.llama_cpp_update import get_installed_llama_version

        # Sort by visible_ordinal: nvidia-smi returns physical order.
        devices = get_backend_visible_gpu_info().get("devices", [])
        body["gpus"] = [
            {"name": d.get("name"), "vram_total_gb": d.get("memory_total_gb")}
            for d in sorted(devices, key = lambda d: d.get("visible_ordinal", 0))
        ]
        body["llama_cpp"] = get_installed_llama_version()
    return body


def _strip_crossorigin(html_bytes: bytes) -> bytes:
    """Remove ``crossorigin`` attributes from script/link tags. Vite's default ``crossorigin`` forces
    CORS mode on font loads, which Firefox HTTPS-Only Mode breaks over plain HTTP."""
    html = html_bytes.decode("utf-8")
    html = _re.sub(r'\s+crossorigin(?:="[^"]*")?', "", html)
    return html.encode("utf-8")


def _inject_bootstrap(html_bytes: bytes, app: FastAPI):
    """Inject bootstrap credentials when password change is pending. Returns
    ``(html_bytes, script_nonce_or_None)``; callers forward the nonce via
    ``_CSP_SCRIPT_NONCE_HEADER`` so CSP allows the inline script."""
    import json as _json
    import secrets as _secrets

    if not storage.requires_password_change(storage.DEFAULT_ADMIN_USERNAME):
        return html_bytes, None

    from auth.policy import installation_has_managed_accounts

    # A local browser may belong to any account, including a deactivated one.
    if installation_has_managed_accounts():
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
    """Canonicalise an Origin to ``(scheme, host, port)`` for equality. Browsers strip default ports (RFC 6454
    sec 6.1) and scheme/host are case-insensitive (RFC 3986), so a bare string compare misclassifies
    same-origin requests as cross-origin. Returns ``None`` on unparseable input so callers fall to the safer
    cross-origin default."""
    scheme = (scheme or "").strip().lower()
    if not scheme or not netloc:
        return None
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


def _origin_of(url: Optional[str]) -> Optional[tuple[str, str, int]]:
    """Canonical origin of a URL or of an Origin header value, or ``None`` when it is neither."""
    if not url:
        return None
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    return _canonical_origin(parsed.scheme, parsed.netloc)


from utils.client_ip import (  # noqa: E402
    _PROXIED_CLIENT_HEADERS,
    _is_loopback_ip,
    is_direct_local_request as _is_local_bootstrap_request,
)


def _is_same_origin_request(request: Request) -> bool:
    """True when Origin is missing or matches request's scheme://host:port. Missing Origin counts as same-origin
    (top-level GETs omit it); both sides are canonicalised via :func:`_canonical_origin`, and callers must
    emit ``Vary: Origin``."""
    origin = request.headers.get("origin")
    if origin is None:
        # Top-level same-document GETs omit Origin.
        return True
    if not origin:
        return False
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


# Colab proxy identified by its own authority; other relays must not read as local.
_COLAB_PROXY_HOST_SUFFIXES = ("colab.googleusercontent.com", ".googleusercontent.com", ".colab.dev")
# The proxy relays only the notebook owner and sets x-forwarded-for; other forwarded headers are foreign.
_COLAB_TOLERATED_PROXY_HEADERS = frozenset({"x-forwarded-for"})


def _host_header_is_colab_proxy(host_header: Optional[str]) -> bool:
    """Whether the Host authority belongs to Colab's own port proxy."""
    if not host_header:
        return False
    host = host_header.strip()
    if host.startswith("["):
        return False
    if host.count(":") == 1:
        host = host.split(":", 1)[0]
    host = host.lower().rstrip(".")
    return host.endswith(_COLAB_PROXY_HOST_SUFFIXES)


def _is_colab_notebook_request(request: Request) -> bool:
    """Allow bootstrap injection through Colab's single-user notebook proxy, and nothing else."""
    for header in _PROXIED_CLIENT_HEADERS:
        if header in _COLAB_TOLERATED_PROXY_HEADERS:
            continue
        if request.headers.get(header) is not None:
            return False
    return _host_header_is_colab_proxy(request.headers.get("host"))


def _should_inject_bootstrap(request: Request) -> bool:
    """Whether to embed the seeded bootstrap password in index.html."""
    if not _is_same_origin_request(request):
        return False
    if _IS_COLAB and _is_colab_notebook_request(request):
        return True
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


def _is_direct_loopback_frontend_request(scope) -> bool:
    server = scope.get("server")
    if not server or not _is_loopback_ip(server[0]):
        return False
    return _is_local_bootstrap_request(Request(scope))


def _is_remote_frontend_request(scope, app_state) -> bool:
    """True for a request the desktop backend may answer with its packaged web UI: Cloudflare's own edge, one
    of the sockets the runtime LAN listener bound (both keyed on the connection, not a client header), or a
    direct unproxied browser on the loopback listener."""
    from lan_access import request_on_lan_listener
    return (
        _is_live_cloudflare_frontend_request(scope, app_state)
        or request_on_lan_listener(scope)
        or _is_direct_loopback_frontend_request(scope)
    )


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
    """Mount frontend static files (optional). ``tunnel_only`` restricts the mount to the callers
    `_is_remote_frontend_request` admits."""
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
        # Bootstrap pw only to same-origin direct-loopback (or Colab proxy). Vary: Origin.
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
        if full_path in {"api", "v1"} or full_path.startswith(("api/", "v1/")):
            raise HTTPException(status_code = 404, detail = "API endpoint not found")
        if not _frontend_request_allowed(request):
            return Response(status_code = 404)

        file_path = (build_path / full_path).resolve()

        if not file_path.is_relative_to(build_path.resolve()):
            return Response(status_code = 403)

        if file_path.is_file():
            return FileResponse(file_path)

        # Last, so a real asset wins; unserved engine endpoints must 404, not render the shell.
        if is_engine_probe_path(full_path):
            raise HTTPException(status_code = 404, detail = "API endpoint not found")

        # Serve index.html as bytes - avoids Content-Length mismatch
        return _build_index_response(request)

    # The lifespan reads this to decide whether engine paths need their own GET denial.
    app.state.frontend_mounted = True
    return True
