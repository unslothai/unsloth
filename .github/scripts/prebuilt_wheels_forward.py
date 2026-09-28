#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Run one real kernel from a freshly installed prebuilt wheel, on a real GPU.

The import gate in prebuilt-cuda-wheels.yml runs on a CPU runner and catches the failure this
whole release exists to avoid: a wheel built against the wrong torch resolves no symbols at
dlopen time and dies with `undefined symbol`. This catches the next one along, which an import
cannot: a wheel that loads fine but carries no cubin the card can run, and fails at the first
launch with `no kernel image is available for execution on the device`.

So the assertions here are about the kernel executing and producing finite numbers, not about
numerical quality -- that is upstream's test suite's job, and it is not what a wheel build can
regress. flash-attn is additionally compared against SDPA, because it is the one of the three
with a cheap reference implementation already in torch.

Usage: prebuilt_wheels_forward.py <flash-attn|causal-conv1d|mamba-ssm>
"""

from __future__ import annotations

import sys

import torch


def _device() -> str:
    if not torch.cuda.is_available():
        raise SystemExit("::error::no CUDA device visible; this script must run on a GPU runner")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(
        f"torch {torch.__version__}, CUDA {torch.version.cuda}, {name} sm_{capability[0]}{capability[1]}"
    )
    return "cuda"


def check_flash_attn(device: str) -> None:
    import flash_attn
    from flash_attn import flash_attn_func

    print("flash_attn", flash_attn.__version__)
    batch, seq, heads, dim = 2, 256, 8, 64
    shape = (batch, seq, heads, dim)
    query, key, value = (
        torch.randn(shape, device = device, dtype = torch.bfloat16, requires_grad = True)
        for _ in range(3)
    )
    out = flash_attn_func(query, key, value, causal = True)
    assert out.shape == shape, out.shape
    assert torch.isfinite(out).all(), "flash_attn forward produced non-finite values"

    out.float().pow(2).mean().backward()
    for name, tensor in (("q", query), ("k", key), ("v", value)):
        assert tensor.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(tensor.grad).all(), f"non-finite gradient for {name}"

    reference = torch.nn.functional.scaled_dot_product_attention(
        *(tensor.detach().transpose(1, 2).float() for tensor in (query, key, value)),
        is_causal = True,
    ).transpose(1, 2)
    error = (out.float() - reference).abs().max().item()
    print(f"  max abs error against SDPA: {error:.4f}")
    # bf16 accumulation, not a tolerance anyone should tighten: the point is that the kernel
    # computed attention rather than returning plausible-looking garbage.
    assert error < 0.1, error


def check_causal_conv1d(device: str) -> None:
    import causal_conv1d
    from causal_conv1d import causal_conv1d_fn

    print("causal_conv1d", getattr(causal_conv1d, "__version__", "unknown"))
    batch, channels, length, width = 2, 64, 128, 4
    x = torch.randn(batch, channels, length, device = device, dtype = torch.bfloat16)
    weight = torch.randn(channels, width, device = device, dtype = torch.bfloat16)
    bias = torch.randn(channels, device = device, dtype = torch.bfloat16)
    out = causal_conv1d_fn(x, weight, bias, activation = "silu")
    assert out.shape == (batch, channels, length), out.shape
    assert torch.isfinite(out).all(), "causal_conv1d produced non-finite values"
    print(f"  out {tuple(out.shape)} mean {out.float().mean().item():.6f}")


def check_mamba_ssm(device: str) -> None:
    import mamba_ssm
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    # Imported for its own sake: this is the extension torch 2.13 stopped loading, and the one
    # the C++20 patch exists for. Importing mamba_ssm alone does not prove it is present.
    import selective_scan_cuda

    print("mamba_ssm", mamba_ssm.__version__, selective_scan_cuda.__file__)
    batch, dim, length, state = 2, 64, 128, 16
    u = torch.randn(batch, dim, length, device = device, requires_grad = True)
    delta = torch.rand(batch, dim, length, device = device, requires_grad = True)
    a = (-torch.rand(dim, state, device = device)).detach().clone().requires_grad_(True)
    b = torch.randn(batch, 1, state, length, device = device, requires_grad = True)
    c = torch.randn(batch, 1, state, length, device = device, requires_grad = True)
    d = torch.randn(dim, device = device, requires_grad = True)

    out = selective_scan_fn(u, delta, a, b, c, d, delta_softplus = True)
    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == (batch, dim, length), out.shape
    assert torch.isfinite(out).all(), "selective_scan produced non-finite values"

    out.float().pow(2).mean().backward()
    assert u.grad is not None and torch.isfinite(u.grad).all(), "non-finite gradient for u"
    print(f"  out {tuple(out.shape)} grad_u_norm {u.grad.float().norm().item():.6f}")


CHECKS = {
    "flash-attn": check_flash_attn,
    "causal-conv1d": check_causal_conv1d,
    "mamba-ssm": check_mamba_ssm,
}


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in CHECKS:
        print(f"usage: prebuilt_wheels_forward.py <{'|'.join(CHECKS)}>", file = sys.stderr)
        return 2
    package = argv[1]
    torch.manual_seed(0)
    CHECKS[package](_device())
    print(f"{package}: forward pass ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
