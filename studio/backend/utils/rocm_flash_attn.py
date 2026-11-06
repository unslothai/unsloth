# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""flash-attn on ROCm via its Triton AMD backend. Stdlib only: main.py imports it before torch.

A Triton-only build imports only with FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from typing import Iterable, MutableMapping, Optional

TRITON_AMD_ENV = "FLASH_ATTENTION_TRITON_AMD_ENABLE"
# Set only when Studio supplied the TRITON_AMD_ENV default: unsloth keeps LLM attention on SDPA.
DIFFUSION_ONLY_ENV = "UNSLOTH_ROCM_FLASH_ATTN_DIFFUSION_ONLY"
SKIP_INSTALL_ENV = "UNSLOTH_STUDIO_SKIP_ROCM_FLASHATTN_INSTALL"

# flash-attention main (2.8.4 dev), as built by the gfx1151 CI runner.
FLASH_ATTN_TRITON_AMD_SHA = "bbe25bacff9f337568b66e69950e549df9839068"
FLASH_ATTN_TRITON_AMD_URL = (
    f"https://github.com/Dao-AILab/flash-attention/archive/{FLASH_ATTN_TRITON_AMD_SHA}.tar.gz"
)
FLASH_ATTN_TRITON_AMD_SPEC = f"flash-attn @ {FLASH_ATTN_TRITON_AMD_URL}"
# setup.py imports / setup_requires under --no-build-isolation, plus runtime einops.
FLASH_ATTN_TRITON_AMD_BUILD_DEPS = ("setuptools", "wheel", "packaging", "psutil", "ninja", "einops")
# FORCE_BUILD skips setup.py's guessed prebuilt-wheel download, which could fetch a CK build.
FLASH_ATTN_TRITON_AMD_BUILD_ENV = {TRITON_AMD_ENV: "TRUE", "FLASH_ATTENTION_FORCE_BUILD": "TRUE"}

# RDNA3/3.5/4 only; CDNA has CK and aiter routes.
_SUPPORTED_GFX = re.compile(r"^gfx1[12][0-9a-f]{2}$")

# Import probe mirroring Studio: a Triton-only build on ROCm gets the backend switch first.
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
    except Exception:  # noqa: BLE001
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
    return _find_spec("flash_attn") is not None and _find_spec("flash_attn_2_cuda") is None


def default_triton_amd_env(environ: Optional[MutableMapping[str, str]] = None) -> bool:
    """Default the Triton AMD switch on; an explicit value (even FALSE) wins. True if applied."""
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
