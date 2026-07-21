# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared backend utilities."""

import os
import structlog
from loggers import get_logger
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
import shutil
import tempfile


logger = get_logger(__name__)


# ── Offline / HF-cache helpers ──────────────────────────────────
# An offline model load must never touch the network: a DNS-dead session hangs on
# huggingface_hub download retries. These read the local HF cache the load itself uses.

_HF_OFFLINE_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def hf_env_offline() -> bool:
    """True when ``HF_HUB_OFFLINE`` or ``TRANSFORMERS_OFFLINE`` requests offline mode.

    Broader than huggingface_hub, which honors only ``HF_HUB_OFFLINE``; the studio also
    honors ``TRANSFORMERS_OFFLINE`` because users set it to keep transformers loads local.
    """
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(var, "").strip().lower() in _HF_OFFLINE_TRUE_VALUES:
            return True
    return False


def st_repo_id_candidates(model_name: str) -> list:
    """Repo ids a Sentence-Transformers load may resolve ``model_name`` to. A slashless
    name (``all-MiniLM-L6-v2``) is loaded as ``sentence-transformers/all-MiniLM-L6-v2``,
    so both are candidate cache repos.
    """
    name = (model_name or "").strip().strip("/")
    if not name:
        return []
    candidates = [name]
    if "/" not in name:
        candidates.append(f"sentence-transformers/{name}")
    return candidates


def _expand_path(raw: str) -> Path:
    """Expand ``~`` and ``$VARS`` the way huggingface_hub does for its cache paths, so the gate
    resolves the same directory the loader will."""
    return Path(os.path.expandvars(os.path.expanduser(raw)))


def _hf_cache_roots() -> list:
    """The single local cache root a Sentence-Transformers / huggingface_hub load resolves to,
    by the loader's own precedence (it selects ONE ``cache_folder`` and does not fall through):
    an explicit ``SENTENCE_TRANSFORMERS_HOME`` (ST's cache_folder), else ``HF_HUB_CACHE``, else
    ``HF_HOME/hub``, else ``~/.cache/huggingface/hub`` (mirroring huggingface_hub). Expanded
    (``~`` / ``$VARS``), read from the env at call time. Returned as a one-element list.
    """
    st_home = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
    if st_home:
        return [_expand_path(st_home)]
    hub = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub:
        return [_expand_path(hub)]
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return [_expand_path(hf_home) / "hub"]
    return [Path.home() / ".cache" / "huggingface" / "hub"]


def hf_cache_snapshot_dir(model_name: str) -> Optional[Path]:
    """Active local snapshot dir for ``model_name``'s ``main`` revision, or None when the repo
    is not cached in any known cache root (including ``SENTENCE_TRANSFORMERS_HOME``). Reads
    ``refs/main`` then ``snapshots/<commit>``; never touches the network. Tries the
    ``sentence-transformers/`` alias for a slashless name.
    """
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
                commit = ref.read_text().strip()
                if not commit:
                    continue
                snapshot = repo_dir / "snapshots" / commit
                if snapshot.is_dir():
                    return snapshot
            except OSError:
                continue
    return None


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
        "memory" in error_str
        or "cuda" in error_str
        or "mlx" in error_str
        or "out of memory" in error_str
    ):
        from utils.hardware import get_device

        device = get_device()
        device_label = {"cuda": "GPU", "mlx": "Apple Silicon GPU", "cpu": "system"}.get(
            device.value, "GPU"
        )
        return f"Not enough {device_label} memory to load '{model_short}'. Try a smaller model or free memory."

    return str(error)
