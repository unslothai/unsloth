# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""ROCm build detection with no imports of its own, so loaders that avoid importing torch can use it."""

__all__ = ["is_rocm_torch"]


def is_rocm_torch(torch_mod = None):
    """ROCm torch build: torch.version.hip, or a +rocm tag in torch.__version__ as a fallback for a build that
    leaves hip unset (the AMD SDK gfx1151 wheels checked so far set it). Read live, not cached: tests and callers
    patch torch.version.hip."""
    if torch_mod is None:
        try:
            import torch as torch_mod
        except Exception:
            return False
    if getattr(getattr(torch_mod, "version", None), "hip", None):
        return True
    # Never str() a TorchVersion (torch.__version__): Dynamo cannot trace its C str() under fullgraph. A plain str
    # there (set by a caller or test) wins, else the plain-str torch.version.__version__.
    version = getattr(torch_mod, "__version__", None)
    if type(version) is not str:
        version = getattr(getattr(torch_mod, "version", None), "__version__", None)
    if type(version) is not str:
        version = str(version or "")
    return "rocm" in version.lower()
