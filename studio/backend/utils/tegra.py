# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NVIDIA Jetson (Tegra) detection, cheap enough for loader-path builders."""

import functools
import os
import re

_TEGRA_RELEASE = "/etc/nv_tegra_release"
_DEVICE_TREE_COMPATIBLE = "/proc/device-tree/compatible"
TEGRA_LIB_DIR = "/usr/lib/aarch64-linux-gnu/tegra"


@functools.lru_cache(maxsize = 1)
def is_tegra() -> bool:
    """True on a Jetson: JetPack's release file, or a Tegra SoC in the device tree. Never raises."""
    try:
        if os.path.exists(_TEGRA_RELEASE):
            return True
        with open(_DEVICE_TREE_COMPATIBLE, "rb") as fh:
            return b"nvidia,tegra" in fh.read()
    except OSError:
        return False


@functools.lru_cache(maxsize = 1)
def tegra_before_cuda13() -> bool:
    """A Jetson on the installers' CPU route (JetPack 5/6, before CUDA 13). JetPack 7 is L4T R38 and Thor is
    tegra264; an unreadable release counts as before, as the installers' unknown-CUDA rule does. Never raises."""
    if not is_tegra():
        return False
    try:
        with open(_TEGRA_RELEASE, encoding = "utf-8", errors = "replace") as fh:
            match = re.match(r"#\s*R(\d+)", fh.readline())
        if match:
            return int(match.group(1)) < 38
    except OSError:
        pass
    try:
        with open(_DEVICE_TREE_COMPATIBLE, "rb") as fh:
            return b"nvidia,tegra264" not in fh.read()
    except OSError:
        return True
