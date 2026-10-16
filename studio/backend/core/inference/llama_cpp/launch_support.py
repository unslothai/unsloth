# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Token counting abort, API keys, TLS and process timing helpers."""

import contextlib
import functools
import os
import ssl
import sys
import time

from pathlib import Path
from typing import (
    Iterable,
    Mapping,
    Optional,
)


class CountAborted(Exception):
    """A token count stood down mid-flight because its answer stopped mattering."""

    __slots__ = ()


# Module level so tests can point the orphan sweep at a fixture tree.
_PROC_ROOT = "/proc"


def _metal_capable_host() -> bool:
    """Apple Silicon, where an empty CUDA/HIP probe still means GPU offload. Not
    ``darwin``: auto_detect_backend resolves an Intel Mac to CPU, which wants the line."""
    try:
        from utils.hardware import is_apple_silicon
        return bool(is_apple_silicon())
    except Exception:  # noqa: BLE001 -- a diagnostic must not break a load
        return sys.platform == "darwin"


# Shared with the settings reader so the search, the settings UI and the process
# allowlist expand one value the same way, and none of them raises on a named user
# the password database cannot answer for. Path.expanduser() does raise, which is
# what took runtime discovery down on a stale ~deleted-user pin.
def _expanded_user_path(value) -> Path:
    try:
        from utils.llama_cpp_path_settings import expanded_user_path
        return expanded_user_path(value)
    except Exception:
        return Path(os.path.expanduser(str(value)))


def _llama_server_api_key_enabled() -> bool:
    """UNSLOTH_LLAMA_SERVER_API_KEY=0 opts out, unless UNSLOTH_DIRECT_STREAM=1 needs the key."""
    return (
        os.getenv("UNSLOTH_DIRECT_STREAM", "0") == "1"
        or os.getenv("UNSLOTH_LLAMA_SERVER_API_KEY", "1") != "0"
    )


def _write_direct_stream_key(key: str, previous: "Optional[Path]" = None) -> "Path":
    """0600 key file, one per backend so concurrent launches never swap keys; ``previous`` is rewritten on relaunch."""
    import secrets as _secrets

    from utils.paths.storage_roots import auth_root

    directory = auth_root()
    directory.mkdir(parents = True, exist_ok = True)
    if previous is None:
        # Left by a killed Studio; llama-server reads its key only at startup.
        stale_before = time.time() - 600
        for stale in directory.glob("llama_api_key_*"):
            with contextlib.suppress(OSError):
                if stale.stat().st_mtime < stale_before:
                    stale.unlink()
    path = (
        previous if previous is not None else directory / f"llama_api_key_{_secrets.token_hex(8)}"
    )
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding = "utf-8") as handle:
        handle.write(key)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def _llama_server_key_via_env() -> bool:
    """True where llama-server cannot open a key file under Studio's auth directory.

    On Windows llama-server re-encodes argv to UTF-8 and opens --api-key-file with a narrow std::ifstream,
    which reads the path in the ANSI code page (common/arg.cpp), so a non-ASCII profile path never opens (#13123).
    """
    from utils.paths.storage_roots import auth_root
    return sys.platform == "win32" and not str(auth_root()).isascii()


def _llama_server_key_launch(
    key: str, previous: "Optional[Path]" = None
) -> "tuple[list[str], dict[str, str], Optional[Path]]":
    """argv and child env that hand ``key`` to llama-server, plus the key file when one was written."""
    if _llama_server_key_via_env():
        # Like the 0600 file, a process environment is readable only by this user, and it stays off the command line.
        return [], {"LLAMA_API_KEY": key}, None
    path = _write_direct_stream_key(key, previous)
    return ["--api-key-file", str(path)], {}, path


def _extra_args_have_tensor_split(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]]
) -> bool:
    """PRESENCE of a pass-through split, parseable or not, in argv or the CHILD env.

    _extra_args_tensor_split returns None for a value it cannot parse, and the repointer
    does no numeric validation and takes the LAST occurrence, so a parse-check veto lets
    the repointer permute a user share it exists to protect. Reads the child env rather
    than os.environ because LLAMA_ARG_TENSOR_SPLIT is scrubbed before this point, and a
    value the child never receives must not suppress the reorder.
    """
    for token in extra_args or ():
        if str(token).split("=", 1)[0] in ("--tensor-split", "-ts", "--tensor_split"):
            return True
    return bool(str((env or {}).get("LLAMA_ARG_TENSOR_SPLIT", "")).strip())


@functools.lru_cache(maxsize = 1)
def _local_ssl_context() -> ssl.SSLContext:
    # httpx loads CA certificates even for loopback HTTP. Share that setup, not streaming sockets.
    return ssl.create_default_context()


# Freeing a large model's GPU memory on a busy driver took over 5 s after SIGKILL; this only bounds a stuck process.
_SIGKILL_REAP_TIMEOUT_S = 30
_QUIT_SIGNAL_EXIT_GRACE_S = 2
