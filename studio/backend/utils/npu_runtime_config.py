# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local NPU runtime settings shared by Studio and its standalone installer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

LEMONADE_ARCHIVE_ENV = "UNSLOTH_LEMONADE_ARCHIVE"
FLM_BINARY_ENV = "UNSLOTH_FLM_NPU_BIN"


def local_lemonade_archive() -> Optional[Path]:
    value = os.environ.get(LEMONADE_ARCHIVE_ENV, "").strip()
    return Path(value) if value else None


def local_flm_binary() -> Optional[Path]:
    value = os.environ.get(FLM_BINARY_ENV, "").strip()
    return Path(value) if value else None


def custom_runtime_configured() -> bool:
    return local_lemonade_archive() is not None or local_flm_binary() is not None
