#!/usr/bin/env python3
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""
RMS layernorm grad check without `import unsloth` or real `transformers`.

On Windows, full `import unsloth` pulls unsloth_zoo → transformers → torchao →
torch._inductor, which expects Triton symbols (e.g. AttrsDescriptor) that may not
match `triton-windows`. This script only needs torch + triton + the RMS kernel file.

Usage (NVIDIA CUDA required):
  python scripts/verify_rms_layernorm_grads_standalone.py
"""
from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import nullcontext
from importlib.machinery import ModuleSpec
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _install_minimal_transformers_stub_for_llama_rms() -> None:
    """So unsloth.kernels.rms_layernorm can load without the real transformers stack."""
    import torch
    from torch import nn

    modeling_llama = types.ModuleType("transformers.models.llama.modeling_llama")

    class LlamaRMSNorm(nn.Module):
        """Matches common HF LlamaRMSNorm numerics used by Unsloth Triton forward."""

        def __init__(self, hidden_size, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(
                variance + self.variance_epsilon
            )
            # Match transformers LlamaRMSNorm (e.g. 5.x): W * normed.to(input_dtype)
            return self.weight * hidden_states.to(input_dtype)

    modeling_llama.LlamaRMSNorm = LlamaRMSNorm

    llama = types.ModuleType("transformers.models.llama")
    llama.__path__ = []
    models = types.ModuleType("transformers.models")
    models.__path__ = []
    root = types.ModuleType("transformers")
    root.__path__ = []

    sys.modules["transformers.models.llama.modeling_llama"] = modeling_llama
    sys.modules["transformers.models.llama"] = llama
    sys.modules["transformers.models"] = models
    sys.modules["transformers"] = root


def _install_kernels_utils_stub() -> None:
    """Avoid importing real kernels/utils.py (bitsandbytes, device_type, unsloth_zoo)."""
    import triton
    import torch

    utils_stub = types.ModuleType("unsloth.kernels.utils")
    MAX_FUSED_SIZE = 65536

    def calculate_settings(n: int) -> tuple[int, int]:
        block_size = int(triton.next_power_of_2(n))
        if block_size > MAX_FUSED_SIZE:
            raise RuntimeError(
                f"Cannot launch Triton kernel since n = {n} exceeds "
                f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}."
            )
        num_warps = 4
        if block_size >= 32768:
            num_warps = 32
        elif block_size >= 8192:
            num_warps = 16
        elif block_size >= 2048:
            num_warps = 8
        return block_size, num_warps

    def torch_gpu_device(device):
        if torch.cuda.device_count() > 1:
            return torch.cuda.device(device)
        return nullcontext()

    utils_stub.calculate_settings = calculate_settings
    utils_stub.torch_gpu_device = torch_gpu_device
    sys.modules["unsloth.kernels.utils"] = utils_stub


def _load_rms_layernorm_module():
    repo = _repo_root()
    kernels_dir = repo / "unsloth" / "kernels"

    unsloth_pkg = types.ModuleType("unsloth")
    unsloth_pkg.__path__ = [str(repo / "unsloth")]
    _us = ModuleSpec("unsloth", None, is_package=True)
    _us.submodule_search_locations = [str(repo / "unsloth")]
    unsloth_pkg.__spec__ = _us
    sys.modules.setdefault("unsloth", unsloth_pkg)

    kernels_pkg = types.ModuleType("unsloth.kernels")
    kernels_pkg.__path__ = [str(kernels_dir)]
    _ks = ModuleSpec("unsloth.kernels", None, is_package=True)
    _ks.submodule_search_locations = [str(kernels_dir)]
    kernels_pkg.__spec__ = _ks
    sys.modules.setdefault("unsloth.kernels", kernels_pkg)

    _install_kernels_utils_stub()

    rms_spec = importlib.util.spec_from_file_location(
        "unsloth.kernels.rms_layernorm",
        kernels_dir / "rms_layernorm.py",
    )
    rms_mod = importlib.util.module_from_spec(rms_spec)
    rms_mod.__package__ = "unsloth.kernels"
    sys.modules["unsloth.kernels.rms_layernorm"] = rms_mod
    assert rms_spec.loader is not None
    rms_spec.loader.exec_module(rms_mod)
    return rms_mod


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        print("CUDA required for this check.", file=sys.stderr)
        sys.exit(1)

    _install_minimal_transformers_stub_for_llama_rms()
    rms = _load_rms_layernorm_module()

    rms.test_rms_layernorm(
        dim=128,
        eps=1e-5,
        dtype=torch.bfloat16,
        bsz=2,
        seqlen=64,
        random_state=0,
    )
    print("verify_rms_layernorm_grads_standalone: OK")


if __name__ == "__main__":
    main()
