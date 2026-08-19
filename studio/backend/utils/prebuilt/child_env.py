# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Child-process environment hygiene for the managed ggml servers.

Secret-env scrubbing and the WSL2 ROCm library-dir probe, shared by the STT
sidecar (and any future launcher of a downloaded binary). Kept in sync with
install_llama_prebuilt.py's scrub_env / _wsl_system_rocm_lib_dirs; the backend
cannot import the studio/ installer scripts, so this copy stays importable with
only the backend root on sys.path.
"""

from __future__ import annotations

import os
import re
from typing import Mapping

SECRET_ENV_EXACT = frozenset(
    {
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "WANDB_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "AZURE_CLIENT_SECRET",
        "KUBECONFIG",
        "SSH_AUTH_SOCK",
    }
)
# Case-insensitive substring markers for names we do not enumerate (no bare "KEY").
SECRET_ENV_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "PASSPHRASE",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "API_KEY",
)
# Proxy / index URLs embed creds in their value; the offline server never needs them.
SECRET_ENV_URL_NAMES = frozenset(
    {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "FTP_PROXY",
        "RSYNC_PROXY",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "UV_INDEX_URL",
        "UV_DEFAULT_INDEX",
        "UV_EXTRA_INDEX_URL",
    }
)
# Also drop values with URL userinfo creds (scheme://user:secret@host).
URL_USERINFO_RE = re.compile(r"://[^/@\s]+@")


def is_secret_env_name(name: str) -> bool:
    upper = name.upper()
    return (
        upper in SECRET_ENV_EXACT
        or upper in SECRET_ENV_URL_NAMES
        or any(marker in upper for marker in SECRET_ENV_MARKERS)
    )


def scrub_env(env: Mapping[str, str]) -> dict[str, str]:
    """Copy of ``env`` without secret-bearing names or URL-userinfo values."""
    return {
        k: v
        for k, v in env.items()
        if not is_secret_env_name(k) and not URL_USERINFO_RE.search(v or "")
    }


# Filesystem pointers a downloaded binary could follow to on-disk credential
# stores (token caches under $HF_HOME, ~/.netrc, XDG config). Dropped, not
# repointed; the offline inference server needs none. Mirrors the cred-location
# list of the tools bypass env (core/inference/tools.py).
CRED_LOCATION_ENV_NAMES = frozenset(
    {
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_XET_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
        "NETRC",
        "BASH_ENV",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "GIT_ASKPASS",
        "SSH_ASKPASS",
        "HOMEDRIVE",
        "HOMEPATH",
    }
)
# Home dirs are repointed (not dropped): loaders and SDKs expect them present,
# but they must not resolve to the user's real profile with its token caches.
HOME_ENV_NAMES = ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA")


def isolate_home(env: dict[str, str], scratch_dir: str) -> dict[str, str]:
    """Repoint home/profile vars at ``scratch_dir`` and drop credential-store
    pointers so a compromised downloaded server cannot read token caches or cred
    files through the environment. Mutates and returns ``env``."""
    os.makedirs(scratch_dir, exist_ok = True)
    for name in HOME_ENV_NAMES:
        if name in env:
            env[name] = scratch_dir
    for name in CRED_LOCATION_ENV_NAMES:
        env.pop(name, None)
    return env


def wsl_system_rocm_lib_dirs() -> list[str]:
    """System ROCm lib dir(s) to load before a bundle's HIP on WSL2. Strict no-op
    off WSL (needs /dev/dxg, a "microsoft" /proc/version, and a librocdxg)."""
    try:
        if not os.path.exists("/dev/dxg"):
            return []
        with open("/proc/version", encoding = "utf-8", errors = "replace") as fh:
            if "microsoft" not in fh.read().lower():
                return []
    except OSError:
        return []
    dirs: list[str] = []
    for d in ("/opt/rocm/lib", "/opt/rocm/lib64"):
        if os.path.exists(os.path.join(d, "librocdxg.so")) or os.path.exists(
            os.path.join(d, "librocdxg.so.1")
        ):
            dirs.append(d)
    return dirs
