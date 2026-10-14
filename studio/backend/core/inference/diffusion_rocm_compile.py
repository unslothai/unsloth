# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whether the regional block compile may run on THIS ROCm install.

ROCm used to be refused outright: inductor's pooled buffers (``memory_planning``) come back from
``inductor::_alloc_from_pool`` as views AOTAutograd rejects at the first call. Studio now compiles with
``memory_planning`` off and never captures graphs on ROCm, which is what the regional compile needs. Whether the
toolchain (torch, Triton, the HIP compiler) can then build and run a transformer block is a property of the install,
so it is probed once per process: one small block (LayerNorm, Linear, SDPA, GELU MLP) compiled exactly as the
default tier compiles (``dynamic=True``, ``fullgraph=True``, no cudagraphs), run at two sequence lengths and checked
against eager. Any failure keeps the denoiser eager, as before.

``UNSLOTH_DIFFUSION_ROCM_COMPILE``: ``0`` never compiles on ROCm, ``1`` skips the probe, unset / ``auto`` probes.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

ENV = "UNSLOTH_DIFFUSION_ROCM_COMPILE"

_lock = threading.Lock()
_result: dict[Any, tuple[bool, str]] = {}


def rocm_compile_setting() -> str:
    raw = (os.environ.get(ENV) or "auto").strip().lower()
    if raw in ("0", "off", "false", "no"):
        return "off"
    if raw in ("1", "on", "true", "yes", "force"):
        return "on"
    return "auto"


def rocm_compile_allowed() -> bool:
    """The device-level answer: False only when the kill switch says so. The probe runs later, at compile time."""
    return rocm_compile_setting() != "off"


def rocm_block_compile_works(device: Any = None) -> bool:
    """Whether a regional block compile builds and runs correctly here (cached per device for the process)."""
    return rocm_block_compile_probe(device)[0]


def rocm_block_compile_probe(device: Any = None) -> tuple[bool, str]:
    setting = rocm_compile_setting()
    if setting == "off":
        return False, f"{ENV}=0"
    if setting == "on":
        return True, f"{ENV}=1 (probe skipped)"
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return False, f"torch unavailable: {exc}"
    if not getattr(getattr(torch, "version", None), "hip", None):
        return True, "not ROCm"
    try:
        index = torch.cuda.current_device() if device is None else torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
    except Exception:  # noqa: BLE001
        index = 0
    with _lock:
        if index not in _result:
            _result[index] = _probe(torch, index)
            ok, why = _result[index]
            (logger.info if ok else logger.warning)(
                "diffusion.speed: ROCm regional compile probe on cuda:%s: %s (%s)",
                index,
                "ok" if ok else "failed, staying eager",
                why,
            )
        return _result[index]


def _probe(torch: Any, index: int) -> tuple[bool, str]:
    start = time.perf_counter()
    try:
        import torch.nn.functional as F
        from torch import nn

        class _Block(nn.Module):
            def __init__(self, dim: int = 128, heads: int = 4) -> None:
                super().__init__()
                self.heads = heads
                self.norm1 = nn.LayerNorm(dim, elementwise_affine = False)
                self.qkv = nn.Linear(dim, 3 * dim)
                self.out = nn.Linear(dim, dim)
                self.norm2 = nn.LayerNorm(dim)
                self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate = "tanh"), nn.Linear(4 * dim, dim))

            def forward(self, x: Any, scale: Any) -> Any:
                b, n, d = x.shape
                h = self.norm1(x) * (1 + scale)
                q, k, v = self.qkv(h).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
                a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
                x = x + self.out(a)
                return x + self.mlp(self.norm2(x))

        dev = torch.device("cuda", index)
        dtype = torch.bfloat16
        gen = torch.Generator(device = "cpu").manual_seed(0)
        block = _Block().to(device = dev, dtype = dtype).eval()
        inductor_cfg = torch._inductor.config
        patches = {"memory_planning": False} if hasattr(inductor_cfg, "memory_planning") else {}
        compiled = torch.compile(block, dynamic = True, fullgraph = True)
        worst = 0.0
        with torch.no_grad(), inductor_cfg.patch(patches), torch._dynamo.config.patch(suppress_errors = False):
            for n in (77, 160):
                x = torch.randn(1, n, 128, generator = gen).to(device = dev, dtype = dtype)
                s = torch.randn(1, 1, 128, generator = gen).to(device = dev, dtype = dtype) * 0.1
                ref = block(x, s).float()
                got = compiled(x, s).float()
                if not bool(torch.isfinite(got).all()):
                    return False, "non-finite compiled output"
                worst = max(worst, float((got - ref).abs().max() / ref.abs().max().clamp_min(1e-6)))
        torch.cuda.synchronize(dev)
        if worst > 0.05:
            return False, f"compiled block disagrees with eager (rel max err {worst:.3g})"
        return True, f"rel max err {worst:.3g}, {time.perf_counter() - start:.1f}s"
    except Exception as exc:  # noqa: BLE001 - any failure is the answer
        return False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"


def _reset_for_tests() -> None:
    with _lock:
        _result.clear()


def probe_result(index: int = 0) -> Optional[tuple[bool, str]]:
    return _result.get(index)
