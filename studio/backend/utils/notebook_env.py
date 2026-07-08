# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hosted notebook environment detection shared by Studio entrypoints."""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Mapping
from pathlib import Path


_KAGGLE_RUNTIME_ENV_VARS = (
    "KAGGLE_KERNEL_RUN_TYPE",
    "KAGGLE_URL_BASE",
    "KAGGLE_CONTAINER_NAME",
)


def is_colab_environment(env: Mapping[str, str] | None = None) -> bool:
    """True when running inside a Colab runtime."""
    env = os.environ if env is None else env
    if not Path("/content").is_dir():
        return False
    if env.get("COLAB_BACKEND_URL") or env.get("COLAB_JUPYTER_IP"):
        return True
    return importlib.util.find_spec("google.colab") is not None


def is_kaggle_environment(env: Mapping[str, str] | None = None) -> bool:
    """True for Kaggle notebook runtimes, but not for local Kaggle credentials."""
    env = os.environ if env is None else env
    has_runtime_env = any(bool(env.get(key)) for key in _KAGGLE_RUNTIME_ENV_VARS)
    if not has_runtime_env:
        return False
    return Path("/kaggle/working").is_dir() or bool(
        env.get("KAGGLE_KERNEL_RUN_TYPE") or env.get("KAGGLE_URL_BASE")
    )


def is_hosted_notebook_environment(env: Mapping[str, str] | None = None) -> bool:
    """True when Studio is running behind a notebook iframe/proxy surface."""
    env = os.environ if env is None else env
    return (
        is_colab_environment(env)
        or is_kaggle_environment(env)
        or env.get("UNSLOTH_STUDIO_HOSTED_NOTEBOOK") == "1"
    )
