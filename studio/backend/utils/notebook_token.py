# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resolve hosted-notebook secrets for the Studio backend process."""

import importlib
import importlib.util
import os
from pathlib import Path


def _read_colab_secret(name: str) -> str | None:
    if importlib.util.find_spec("google.colab") is None:
        return None
    userdata = importlib.import_module("google.colab.userdata")
    return userdata.get(name)


def _read_kaggle_secret(name: str) -> str | None:
    if importlib.util.find_spec("kaggle_secrets") is None:
        return None
    secrets = importlib.import_module("kaggle_secrets")
    return secrets.UserSecretsClient().get_secret(name)


def resolve_notebook_hf_token(secret_name: str = "HF_TOKEN") -> tuple[str | None, str | None]:
    """Return ``(token, source)`` and install a notebook secret into the environment."""
    is_colab = bool(
        os.environ.get("COLAB_BACKEND_URL")
        or os.environ.get("COLAB_JUPYTER_IP")
        or Path("/content").is_dir()
    )
    is_kaggle = bool(
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        or os.environ.get("KAGGLE_URL_BASE")
        or Path("/kaggle/working").is_dir()
    )
    existing = os.environ.get("HF_TOKEN")
    if existing:
        # This endpoint is allowed to return credentials only in a hosted notebook.
        # Desktop/server environment variables must never be copied into a browser.
        source = "colab" if is_colab else "kaggle" if is_kaggle else None
        return (existing, source) if source else (None, None)

    readers = []
    if is_colab:
        readers.append(("colab", _read_colab_secret))
    if is_kaggle:
        readers.append(("kaggle", _read_kaggle_secret))

    for source, reader in readers:
        try:
            token = reader(secret_name)
        except Exception:
            continue
        if isinstance(token, str) and token.strip():
            os.environ["HF_TOKEN"] = token.strip()
            return os.environ["HF_TOKEN"], source
    return None, None
