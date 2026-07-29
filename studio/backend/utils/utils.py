# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared backend utilities."""

import os
import structlog
import threading
import time
from loggers import get_logger
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
import shutil
import tempfile


logger = get_logger(__name__)


# ── Offline / HF-cache helpers ──────────────────────────────────
# An offline load must never touch the network (a DNS-dead session hangs on hub retries);
# these read the local HF cache the load itself uses.

_HF_OFFLINE_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def hf_env_offline() -> bool:
    """True when HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE requests offline mode.

    Also honors TRANSFORMERS_OFFLINE (hub honors only HF_HUB_OFFLINE) since users set it
    to keep transformers loads local.
    """
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(var, "").strip().lower() in _HF_OFFLINE_TRUE_VALUES:
            return True
    return False


def hf_endpoint_url() -> str:
    """Configured hub endpoint, scheme-normalised. Mirror users point this elsewhere."""
    endpoint = (os.environ.get("HF_ENDPOINT") or "").strip() or "https://huggingface.co"
    return endpoint if "://" in endpoint else "https://" + endpoint


def hf_endpoint_host() -> str:
    """Host of the configured endpoint; probing huggingface.co would misjudge a mirror."""
    try:
        from urllib.parse import urlparse
        return urlparse(hf_endpoint_url()).hostname or "huggingface.co"
    except Exception:
        return "huggingface.co"


def hf_proxy_configured() -> bool:
    """True when egress to the endpoint goes through a proxy.

    The proxy resolves the hub host, so local DNS proves nothing about reachability and
    must not be used to declare the hub offline.
    """
    try:
        import urllib.request
        from urllib.parse import urlparse

        url = hf_endpoint_url()
        proxies = urllib.request.getproxies()
        scheme = urlparse(url).scheme or "https"
        if scheme not in proxies and "all" not in proxies:
            return False
        host = urlparse(url).hostname or ""
        try:
            if host and urllib.request.proxy_bypass(host):
                return False
        except Exception:
            pass
        return True
    except Exception:
        return False


def dns_host_dead(host: str, timeout: float = 2.0) -> bool:
    """True when host does not resolve. Runs on a daemon thread so a wedged resolver
    cannot block past the deadline and so socket.setdefaulttimeout is left alone."""
    result: list = [None]

    def _probe() -> None:
        import socket as _socket
        try:
            _socket.gethostbyname(host)
            result[0] = False
        except Exception:
            result[0] = True

    t = threading.Thread(target = _probe, daemon = True)
    t.start()
    t.join(timeout)
    # Still running -> resolver wedged -> treat as dead.
    return True if result[0] is None else result[0]


def hf_dns_dead(timeout: float = 2.0) -> bool:
    """Fast offline shortcut: the endpoint's host does not resolve and no proxy is in play.

    Returns False whenever a proxy is configured, so proxy-only setups fall through to the
    real reachability probe instead of being wrongly declared offline.
    """
    if hf_proxy_configured():
        return False
    return dns_host_dead(hf_endpoint_host(), timeout)


# One load makes many hub calls, so the verdict is shared briefly to avoid re-probing on
# each. Kept short in BOTH directions: a stale "reachable" misses the plug being pulled
# (the case this whole path exists for), and a stale "unreachable" sends a load to the
# cache after the user reconnected, failing it if the model is not cached.
_HF_REACHABILITY_TTL_S = 5.0
_hf_reachability: Optional[tuple] = None
_hf_reachability_lock = threading.Lock()


def _reachability_fresh(entry) -> bool:
    """True while a cached (timestamp, unreachable) verdict may still be reused."""
    return entry is not None and (time.monotonic() - entry[0]) < _HF_REACHABILITY_TTL_S


def hf_probe_disabled() -> bool:
    """True when UNSLOTH_OFFLINE_PROBE opts out of the reachability probe."""
    return os.environ.get("UNSLOTH_OFFLINE_PROBE", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }


def reset_hf_reachability_cache() -> None:
    """Drop the memoised verdict so the next call re-probes (tests, network changes)."""
    global _hf_reachability
    with _hf_reachability_lock:
        _hf_reachability = None


def hf_unreachable(timeout: int = 3) -> bool:
    """True when the HF endpoint is unreachable, memoised for _HF_REACHABILITY_TTL_S.

    Resolving DNS does not mean the Hub is reachable: a router that is up with the WAN
    down, a captive portal or a stale DNS cache all answer lookups while every request
    then burns huggingface_hub's retry backoff. This is the bounded, proxy-aware probe
    the export path already uses; disable it with UNSLOTH_OFFLINE_PROBE=0.

    Fails open: an unavailable probe reports reachable, so the load decides as it does today.
    """
    if hf_probe_disabled():
        return False

    global _hf_reachability
    cached = _hf_reachability
    if _reachability_fresh(cached):
        return cached[1]

    with _hf_reachability_lock:
        cached = _hf_reachability
        if _reachability_fresh(cached):
            return cached[1]
        try:
            from utils.transformers_version import hf_endpoint_unreachable
            unreachable = hf_endpoint_unreachable(timeout)
        except Exception:
            unreachable = False
        _hf_reachability = (time.monotonic(), unreachable)
        return unreachable


def _reset_hf_sessions() -> None:
    """Drop cached hub sessions so they remount with the current offline adapter."""
    try:
        from huggingface_hub.utils import _http

        for name in ("_get_session_from_cache", "get_session"):
            cache_clear = getattr(getattr(_http, name, None), "cache_clear", None)
            if cache_clear is not None:
                cache_clear()
        reset = getattr(_http, "reset_sessions", None)
        if reset is not None:
            reset()
    except Exception:
        pass


# Process-global, so nested/concurrent loads refcount rather than restore out from under
# each other.
_force_offline_depth = 0
_force_offline_saved: list = []
_force_offline_saved_env: dict = {}
_force_offline_lock = threading.Lock()

_OFFLINE_ENV_KEYS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
_OFFLINE_CONSTANTS = (
    ("huggingface_hub.constants", ("HF_HUB_OFFLINE",)),
    ("transformers.utils.hub", ("_is_offline_mode", "OFFLINE")),
)


@contextmanager
def force_hf_offline():
    """Force HF offline for this block, in-process.

    Setting the env vars is not enough once the process is running: huggingface_hub and
    transformers read their offline constants at import, and hub sessions cache a
    non-offline adapter. Flip the constants too and rebuild the sessions, so hub calls
    fail fast instead of retrying. Everything is restored on exit.
    """
    global _force_offline_depth, _force_offline_saved, _force_offline_saved_env
    import importlib

    with _force_offline_lock:
        if _force_offline_depth == 0:
            saved: list = []
            saved_env: dict = {}
            # Snapshot constants BEFORE forcing the env, else a module first imported
            # inside the window reads the "1" and we would restore it as offline.
            for mod_name, attrs in _OFFLINE_CONSTANTS:
                try:
                    mod = importlib.import_module(mod_name)
                except Exception:
                    continue
                for attr in attrs:
                    if hasattr(mod, attr):
                        saved.append((mod, attr, getattr(mod, attr)))
            for key in _OFFLINE_ENV_KEYS:
                saved_env[key] = os.environ.get(key)
                os.environ[key] = "1"
            for mod, attr, _ in saved:
                try:
                    setattr(mod, attr, True)
                except Exception:
                    pass
            _force_offline_saved = saved
            _force_offline_saved_env = saved_env
            _reset_hf_sessions()
        _force_offline_depth += 1
    try:
        yield
    finally:
        with _force_offline_lock:
            _force_offline_depth -= 1
            if _force_offline_depth == 0:
                for mod, attr, val in _force_offline_saved:
                    try:
                        setattr(mod, attr, val)
                    except Exception:
                        pass
                _force_offline_saved = []
                for key, val in _force_offline_saved_env.items():
                    if val is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = val
                _force_offline_saved_env = {}
                _reset_hf_sessions()


def st_repo_id_candidates(model_name: str) -> list:
    """Repo ids a Sentence-Transformers load may resolve model_name to; a slashless name
    also resolves under the sentence-transformers/ namespace, so both are candidates."""
    name = (model_name or "").strip().strip("/")
    if not name:
        return []
    candidates = [name]
    if "/" not in name:
        candidates.append(f"sentence-transformers/{name}")
    return candidates


def _expand_path(raw: str) -> Path:
    """Expand ~ and $VARS as huggingface_hub does, so the gate resolves the loader's dir."""
    return Path(os.path.expandvars(os.path.expanduser(raw)))


def _hf_cache_roots() -> list:
    """Cache roots to search for a model's local snapshot, most-authoritative first.

    The app's selected hub cache (set via /settings) is searched first: after a
    no-restart cache switch the process env is stale, yet the loader reads the
    selected cache via ``cache_folder=active_hf_hub_cache()``, so the snapshot
    and offline security lookups must match where it actually loads. The env
    precedence (SENTENCE_TRANSFORMERS_HOME, HF_HUB_CACHE, HF_HOME/hub,
    ~/.cache/huggingface/hub) follows so a copy still in a previous cache resolves."""
    roots: list = []
    seen: set = set()

    def _add(path) -> None:
        if path is None:
            return
        expanded = _expand_path(str(path))
        key = str(expanded)
        if key not in seen:
            seen.add(key)
            roots.append(expanded)

    try:
        from utils.hf_cache_settings import get_hf_cache_paths
        _add(get_hf_cache_paths().hub_cache)
    except Exception:
        pass

    if st_home := os.environ.get("SENTENCE_TRANSFORMERS_HOME"):
        _add(st_home)
    if hub := (os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")):
        _add(hub)
    if hf_home := os.environ.get("HF_HOME"):
        _add(_expand_path(hf_home) / "hub")
    if not roots:
        _add(Path.home() / ".cache" / "huggingface" / "hub")
    return roots


def hf_cache_snapshot_dir(model_name: str) -> Optional[Path]:
    """Active local snapshot dir for model_name's main revision, or None if not cached.
    Reads refs/main then snapshots/<commit>; no network. Tries the ST alias for slashless names."""
    try:
        from huggingface_hub.file_download import repo_folder_name
    except Exception:
        repo_folder_name = None
    for cache_root in _hf_cache_roots():
        for repo_id in st_repo_id_candidates(model_name):
            try:
                if repo_folder_name is not None:
                    folder = repo_folder_name(repo_id = repo_id, repo_type = "model")
                else:
                    folder = "models--" + repo_id.replace("/", "--")
                repo_dir = cache_root / folder
                ref = repo_dir / "refs" / "main"
                if not ref.is_file():
                    continue
                commit = ref.read_text(encoding = "utf-8").strip()
                if not commit:
                    continue
                snapshot = repo_dir / "snapshots" / commit
                if snapshot.is_dir():
                    return snapshot
            except (OSError, UnicodeDecodeError):
                continue
    return None


# A weight file plus a config distinguishes a real cached model from a metadata-only
# partial cache that resolves refs/main but would fail at load time.
_LOADABLE_WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".gguf", ".pt", ".pth", ".ckpt"})


def hf_cache_snapshot_is_loadable(model_name: str) -> bool:
    """True when model_name's snapshot is cached and loadable: a config (config.json or
    modules.json) plus at least one weight file, not a metadata-only partial cache. No network."""
    snapshot = hf_cache_snapshot_dir(model_name)
    if snapshot is None:
        return False
    try:
        has_config = (snapshot / "config.json").is_file() or (snapshot / "modules.json").is_file()
        if not has_config:
            return False
        for path in snapshot.rglob("*"):
            if path.suffix.lower() in _LOADABLE_WEIGHT_SUFFIXES and path.is_file():
                return True
    except OSError:
        return False
    return False


# ── Client-safe error helpers ───────────────────────────────────
# Never return raw exception text to clients; log server-side, return generic.


def safe_error_detail(error: Exception, fallback: str = "An internal error occurred") -> str:
    """Map an exception to a generic, client-safe message (never raw
    ``str(error)``, which can leak paths). Log the real exception server-side.
    """
    text = str(error).lower()
    if (
        isinstance(error, (ConnectionError, TimeoutError))
        or "connection" in text
        or "timed out" in text
        or "timeout" in text
    ):
        return "Could not reach an upstream service. Please try again."
    if "out of memory" in text or "cuda error" in text:
        return "Ran out of memory. Try a smaller model or shorter input."
    return fallback


def safe_curated_detail(error: Exception, fallback: str = "An internal error occurred") -> str:
    """Client-safe text for curated domain/validation exceptions.

    Keeps the message (paths stripped) instead of a generic fallback; for known
    exception types only (use ``safe_error_detail`` for generic ``Exception``).
    """
    from utils.native_path_leases import redact_native_paths

    msg = redact_native_paths(str(error)).strip()
    return msg or fallback


def log_and_http_error(
    error: Exception,
    status_code: int,
    public_message: str,
    *,
    event: str = "request_failed",
    log = None,
):
    """Log ``error`` in full server-side and return an ``HTTPException`` whose
    ``detail`` is only ``public_message`` -- never the raw exception text.

    Usage:  raise log_and_http_error(e, 500, "Failed to start training")
    """
    from fastapi import HTTPException

    # exc_info=error works for both structlog and stdlib loggers.
    (log or logger).error(f"{event}: {error}", exc_info = error)
    return HTTPException(status_code = status_code, detail = public_message)


@contextmanager
def without_hf_auth():
    """
    Temporarily disable HuggingFace authentication.

    Usage:
        with without_hf_auth():
            # Code that should run without cached tokens
            model_info(model_name, token=None)
    """
    saved_env = {}
    env_vars = ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HOME"]
    for var in env_vars:
        if var in os.environ:
            saved_env[var] = os.environ[var]
            del os.environ[var]

    saved_disable = os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

    # Move token files aside temporarily
    token_files = []
    token_locations = [
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ]

    for token_loc in token_locations:
        if token_loc.exists():
            temp = tempfile.NamedTemporaryFile(delete = False)
            temp.close()
            shutil.move(str(token_loc), temp.name)
            token_files.append((token_loc, temp.name))

    try:
        yield
    finally:
        # Restore tokens
        for original, temp in token_files:
            try:
                original.parent.mkdir(parents = True, exist_ok = True)
                shutil.move(temp, str(original))
            except Exception as e:
                logger.error(f"Failed to restore token {original}: {e}")

        # Restore env
        for var, value in saved_env.items():
            os.environ[var] = value

        if saved_disable is not None:
            os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = saved_disable
        else:
            os.environ.pop("HF_HUB_DISABLE_IMPLICIT_TOKEN", None)


def is_hf_authentication_error(error: Exception) -> bool:
    """Return whether an exception chain contains a definitive HF auth failure."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        try:
            if status is not None and int(status) == 401:
                return True
        except (TypeError, ValueError):
            pass
        message = str(current).lower()
        if "invalid user token" in message or "invalid hf token" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def format_error_message(error: Exception, model_name: str) -> str:
    """
    Format a user-friendly error message for common load issues.

    Args:
        error: The exception that occurred
        model_name: Name of the model being loaded
    """
    error_str = str(error).lower()
    model_short = model_name.split("/")[-1] if "/" in model_name else model_name

    if "repository not found" in error_str or "404" in error_str:
        return f"Model '{model_short}' not found. Check the model name."

    if "401" in error_str or "unauthorized" in error_str:
        return f"Authentication failed for '{model_short}'. Please provide a valid HF token."

    if "gated" in error_str or "access to model" in error_str:
        return f"Model '{model_short}' requires authentication. Please provide a valid HF token."

    if "invalid user token" in error_str:
        return "Invalid HF token. Please check your token and try again."

    if (
        "out of memory" in error_str
        or "out of device memory" in error_str
        or "out_of_device_memory" in error_str  # ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
        or "out_of_host_memory" in error_str  # ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
        or "not enough memory" in error_str
        or "cannot allocate memory" in error_str
        or "memory allocation failed" in error_str
        or "cublas_status_alloc_failed" in error_str  # cuBLAS workspace OOM
        or ("cuda error" in error_str and "alloc" in error_str)
        or ("xpu" in error_str and ("alloc" in error_str or "memory" in error_str))
        or isinstance(error, MemoryError)
        or ("mlx" in error_str and ("memory" in error_str or "allocate" in error_str))
    ):
        # Resolve get_device() at call time (not import time) so tests that
        # monkey-patch utils.hardware.get_device after this module is loaded
        # still see the patched backend.
        from utils.hardware import get_device

        device = get_device()
        device_label = {
            "cuda": "GPU",
            "xpu": "Intel GPU",
            "mlx": "Apple Silicon GPU",
            "cpu": "system",
        }.get(device.value, "GPU")
        return f"Not enough {device_label} memory to load '{model_short}'. Try a smaller model or free memory."

    return str(error)
