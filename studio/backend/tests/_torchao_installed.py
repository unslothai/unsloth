# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Skip for tests whose claim needs a real torchao install.

Read from the installed distribution, not ``import torchao``: other tests put fake torchao modules in
``sys.modules``, so an import can succeed where torchao is absent (macOS, which installs MLX instead).
"""

from __future__ import annotations

import importlib.metadata

import pytest


def torchao_installed() -> bool:
    try:
        importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


needs_torchao = pytest.mark.skipif(
    not torchao_installed(),
    reason = "needs torchao installed: without it no pre-quant checkpoint can load, so none is planned",
)
