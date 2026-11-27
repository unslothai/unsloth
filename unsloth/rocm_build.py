# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""ROCm build detection with no imports of its own, so loaders that avoid importing torch can use it."""

__all__ = ["is_rocm_torch"]


def is_rocm_torch(torch_mod = None):
    """ROCm torch build. AMD SDK / Radeon wheels leave torch.version.hip unset and carry the ROCm tag only in
    torch.__version__, so check both. Read live (not cached): tests and callers patch torch.version.hip."""
    if torch_mod is None:
        try:
            import torch as torch_mod
        except Exception:
            return False
    if getattr(getattr(torch_mod, "version", None), "hip", None):
        return True
    return "rocm" in str(getattr(torch_mod, "__version__", "") or "").lower()
