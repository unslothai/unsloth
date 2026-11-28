# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""No native library: every handle exists but raises when called and has no ctypes `restype`,
which is how callers (Unsloth's native_kernels_ready, bitsandbytes' own error mock) tell a
missing native library apart from a loaded one."""

import torch

HIP_ENVIRONMENT = torch.version.hip is not None
ROCM_WARP_SIZE_64 = False
BNB_BACKEND = "Unsloth"


class _MissingNativeFunction:
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"Unsloth: `{self.name}` is a bitsandbytes native function, and this is Unsloth's "
            "bitsandbytes stand-in (bitsandbytes is not installed or its native library could "
            "not load). Install bitsandbytes for this code path."
        )


class _MissingNativeLibrary:
    compiled_with_cuda = False

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _MissingNativeFunction(name)


lib = _MissingNativeLibrary()
