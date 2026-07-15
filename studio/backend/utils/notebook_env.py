# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Notebook runtime detection shared by Studio entrypoints."""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Mapping
from pathlib import Path


def is_colab_environment(env: Mapping[str, str] | None = None) -> bool:
    """True when running inside a Colab runtime."""
    env = os.environ if env is None else env
    return Path("/content").is_dir() and bool(
        env.get("COLAB_BACKEND_URL")
        or env.get("COLAB_JUPYTER_IP")
        or importlib.util.find_spec("google.colab") is not None
    )


def is_kaggle_environment(env: Mapping[str, str] | None = None) -> bool:
    """True for Kaggle notebook runtimes, but not for local Kaggle credentials."""
    env = os.environ if env is None else env
    return bool(
        env.get("KAGGLE_KERNEL_RUN_TYPE")
        or env.get("KAGGLE_URL_BASE")
        or env.get("KAGGLE_CONTAINER_NAME")
        and Path("/kaggle/working").is_dir()
    )
