# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NVIDIA Jetson (Tegra) detection, cheap enough for loader-path builders."""

import functools
import os

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
