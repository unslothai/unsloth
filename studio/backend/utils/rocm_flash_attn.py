# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""flash-attn on ROCm through its Triton AMD backend (pure Python, no CK compile).

Studio's diffusion attention (core/inference/diffusion_attention.py) already routes to flash_attn on
ROCm when it imports and wins a timing probe; this module is the install and import side. Stdlib
only and no torch import: main.py imports it before torch, and the installer imports it as
``backend.utils.rocm_flash_attn``.

A Triton-only build has no ``flash_attn_2_cuda`` extension and imports only with
``FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`` (flash_attn_interface reads it at import), so without the
variable every importer (diffusers, xformers, unsloth) sees a broken install.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from typing import Iterable, MutableMapping, Optional

TRITON_AMD_ENV = "FLASH_ATTENTION_TRITON_AMD_ENABLE"
# Set beside TRITON_AMD_ENV only when Studio supplied the default: unsloth then leaves LLM attention
# on SDPA (as before this build was installed); a user who set TRITON_AMD_ENV themselves opted in.
DIFFUSION_ONLY_ENV = "UNSLOTH_ROCM_FLASH_ATTN_DIFFUSION_ONLY"
SKIP_INSTALL_ENV = "UNSLOTH_STUDIO_SKIP_ROCM_FLASHATTN_INSTALL"

# Dao-AILab/flash-attention main (2.8.4 dev), the commit the gfx1151 CI runner builds. The GitHub
# archive has no submodules, which the Triton backend does not need (setup.py skips the CK build).
FLASH_ATTN_TRITON_AMD_SHA = "bbe25bacff9f337568b66e69950e549df9839068"
FLASH_ATTN_TRITON_AMD_URL = (
    f"https://github.com/Dao-AILab/flash-attention/archive/{FLASH_ATTN_TRITON_AMD_SHA}.tar.gz"
)
FLASH_ATTN_TRITON_AMD_SPEC = f"flash-attn @ {FLASH_ATTN_TRITON_AMD_URL}"
# Imported by setup.py under --no-build-isolation (setuptools, wheel, packaging), named in its
# setup_requires (psutil, ninja; setuptools would try to fetch them), and its one runtime dep (einops).
FLASH_ATTN_TRITON_AMD_BUILD_DEPS = ("setuptools", "wheel", "packaging", "psutil", "ninja", "einops")
# FORCE_BUILD skips setup.py's guessed prebuilt-wheel download, which could fetch a CK build.
FLASH_ATTN_TRITON_AMD_BUILD_ENV = {TRITON_AMD_ENV: "TRUE", "FLASH_ATTENTION_FORCE_BUILD": "TRUE"}

# RDNA3 / RDNA3.5 / RDNA4 (gfx11xx, gfx12xx), where the Triton AMD backend runs and no CK flash-attn
# wheel exists. CDNA (gfx90a / gfx942 / gfx950) has CK and aiter routes and is left alone.
_SUPPORTED_GFX = re.compile(r"^gfx1[12][0-9a-f]{2}$")

# Child-process import probe that sees what a Studio process would: a Triton-only build on ROCm
# torch gets the backend switch first. CUDA builds (flash_attn_2_cuda present) are probed as before.
IMPORT_PROBE = (
    "import importlib.util, os\n"
    "if importlib.util.find_spec('flash_attn') is not None and "
    "importlib.util.find_spec('flash_attn_2_cuda') is None:\n"
    "    import torch\n"
    "    if getattr(torch.version, 'hip', None):\n"
    f"        os.environ.setdefault({TRITON_AMD_ENV!r}, 'TRUE')\n"
    "import flash_attn\n"
)


def supported_gfx(gfx_codes: Iterable[str]) -> list[str]:
    return [code for code in gfx_codes if _SUPPORTED_GFX.match(str(code).strip().lower())]


def install_skip_reason(
    *, platform: str, hip_version: Optional[str], gfx_codes: Iterable[str]
) -> Optional[str]:
    """Why the Triton AMD flash-attn is not installed here, or None to install it."""
    if os.environ.get(SKIP_INSTALL_ENV) == "1":
        return f"{SKIP_INSTALL_ENV}=1"
    if not platform.startswith("linux"):
        return "Linux only"
    if not hip_version:
        return "needs a ROCm build of torch"
    codes = [str(code) for code in gfx_codes]
    if not codes:
        return "no AMD GPU architecture detected"
    if not supported_gfx(codes):
        return f"no RDNA3/RDNA4 GPU ({', '.join(sorted(set(codes)))})"
    return None


def _find_spec(name: str):
    try:
        return importlib.util.find_spec(name)
    except Exception:  # noqa: BLE001 - a broken finder means "not determinable"
        return None


def _torch_is_rocm() -> bool:
    """Whether the resident torch is a ROCm build, without importing torch (main.py defers it)."""
    torch = sys.modules.get("torch")
    if torch is not None:
        return bool(getattr(getattr(torch, "version", None), "hip", None))
    spec = _find_spec("torch")
    locations = list(getattr(spec, "submodule_search_locations", None) or [])
    for location in locations:
        try:
            with open(os.path.join(location, "version.py"), encoding = "utf-8") as fh:
                text = fh.read()
        except OSError:
            continue
        match = re.search(r"^hip\s*(?::[^=\n]*)?=\s*(.+)$", text, re.MULTILINE)
        return bool(match) and match.group(1).strip() not in ("None", "''", '""')
    return False


def triton_only_flash_attn_installed() -> bool:
    """flash_attn is installed without a compiled extension (it can only run on Triton)."""
    return _find_spec("flash_attn") is not None and _find_spec("flash_attn_2_cuda") is None


def default_triton_amd_env(environ: Optional[MutableMapping[str, str]] = None) -> bool:
    """setdefault the Triton AMD backend switch for a Triton-only flash_attn on ROCm torch, before
    anything imports flash_attn. Spawned workers inherit it. An explicit value (even FALSE) wins.
    True when the default was applied."""
    env = os.environ if environ is None else environ
    if not sys.platform.startswith("linux"):
        return False
    if TRITON_AMD_ENV in env or "flash_attn" in sys.modules:
        return False
    if not triton_only_flash_attn_installed() or not _torch_is_rocm():
        return False
    env[TRITON_AMD_ENV] = "TRUE"
    env.setdefault(DIFFUSION_ONLY_ENV, "1")
    return True
