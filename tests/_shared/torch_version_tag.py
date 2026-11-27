# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests that simulate a CUDA build by setting torch.version.hip = None must also drop a +rocm version tag:
is_rocm_torch falls back to that tag, so on a real ROCm host the simulated build would still read as ROCm."""


def drop_rocm_tag(monkeypatch, torch_mod):
    version = str(torch_mod.__version__)
    if "rocm" in version.lower():
        monkeypatch.setattr(
            torch_mod, "__version__", torch_mod.torch_version.TorchVersion(version.split("+")[0])
        )
    plain = getattr(torch_mod.version, "__version__", None)
    if isinstance(plain, str) and "rocm" in plain.lower():
        monkeypatch.setattr(torch_mod.version, "__version__", plain.split("+")[0])
