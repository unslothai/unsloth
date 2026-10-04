# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared sensitive path-component policy."""

from __future__ import annotations

import os


SENSITIVE_PATH_COMPONENTS = {
    ".aws",
    ".azure",
    ".config",
    ".docker",
    ".gcloud",
    ".gnupg",
    ".huggingface",
    ".kaggle",
    ".kube",
    ".modelscope",
    ".ngc",
    ".local",
    ".mozilla",
    ".pki",
    ".thunderbird",
    ".ssh",
    ".1password",
    ".bitwarden",
    ".password-store",
    "1password",
    "bitwarden",
    "keychains",
    "keyrings",
    "mozilla",
    "thunderbird",
}


def is_sensitive_path_component(name: str) -> bool:
    return name.lower() in SENSITIVE_PATH_COMPONENTS


def contains_sensitive_path_component(path: str) -> bool:
    parts = os.path.normpath(path).split(os.sep)
    return any(is_sensitive_path_component(part) for part in parts)
