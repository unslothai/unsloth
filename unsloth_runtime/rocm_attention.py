# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Conservative process-local enablement for PyTorch ROCm AOTriton SDPA."""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Any, MutableMapping

AOTRITON_ENV = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"

# This is intentionally an exact allowlist. The experimental gate selects bundled GPU kernels, so
# neither a newer PyTorch version nor a nearby GPU architecture implies that the same kernels are
# safe. Expand only after a forward and backward numerical probe on that exact runner image.
_VALIDATED_TORCH_BUILD = "2.11.0+rocm7.13.0"
_VALIDATED_ARCH = "gfx1151"


def _torch_build(torch_module: Any | None) -> str:
    if torch_module is not None:
        return str(getattr(torch_module, "__version__", "") or "")
    try:
        return package_version("torch")
    except (PackageNotFoundError, ValueError):
        return ""


def _gpu_arch(properties: Any) -> str:
    for name in ("gcnArchName", "gcn_arch_name", "arch_name", "gfx_arch_name"):
        value = str(getattr(properties, name, "") or "").strip().lower()
        if value:
            return value.split(":", 1)[0]
    return ""


def enable_rocm_aotriton_attention(
    env: MutableMapping[str, str] | None = None,
    *,
    torch_module: Any | None = None,
    platform_name: str | None = None,
    dxg_present: bool | None = None,
) -> bool:
    """Enable the gate only on the exact native-Linux gfx1151 stack validated in CI.

    Existing values always win. Unknown builds, missing device metadata, mixed architectures,
    Windows, WSL, and non-ROCm runtimes fail closed without changing the environment.
    """

    target = os.environ if env is None else env
    if AOTRITON_ENV in target:
        return False

    if (platform_name or sys.platform) != "linux":
        return False
    if _torch_build(torch_module) != _VALIDATED_TORCH_BUILD:
        return False
    if dxg_present is None:
        dxg_present = os.path.exists("/dev/dxg")
    if dxg_present:
        return False

    if torch_module is None:
        try:
            import torch as torch_module
        except Exception:
            return False

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return False
    try:
        if not cuda.is_available():
            return False
        count = int(cuda.device_count())
        arches = tuple(_gpu_arch(cuda.get_device_properties(index)) for index in range(count))
    except Exception:
        return False

    # The variable is process-wide, so every visible device must be on the validated path. This
    # prevents a mixed gfx1151/gfx1150 process from enabling the known-risk gfx1150 kernels too.
    if not arches or any(arch != _VALIDATED_ARCH for arch in arches):
        return False

    target[AOTRITON_ENV] = "1"
    return True
