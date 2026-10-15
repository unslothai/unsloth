# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-process probe: can this ROCm install regionally compile a transformer block?

ROCm was refused because inductor's pooled buffers (``memory_planning``) are views AOTAutograd rejects; Studio now
compiles with it off and no cudagraphs on ROCm. The probe compiles one small block as the default tier does and checks
it against eager; any failure stays eager.

``UNSLOTH_DIFFUSION_ROCM_COMPILE``: ``0`` never compiles on ROCm, ``1`` skips the probe, unset / ``auto`` probes.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import logging
import os
import threading
import time
from typing import Any, Callable, Iterator, Optional

def _module_logger() -> Any:
    # Studio's structured logger; stdlib logging alone is not routed at info level.
    try:
        from loggers import get_logger
        return get_logger(__name__)
    except Exception:  # noqa: BLE001
        return logging.getLogger(__name__)


logger = _module_logger()

ENV = "UNSLOTH_DIFFUSION_ROCM_COMPILE"

_lock = threading.Lock()
# Process-wide answers. Never an OOM: that says the card was busy, not that the compiler is broken.
_result: dict[Any, tuple[bool, str]] = {}
# The running load's own OOM answers, so one load never re-probes a busy card; another load starts empty.
_load_oom: contextvars.ContextVar[Optional[dict[Any, tuple[bool, str]]]] = contextvars.ContextVar(
    "rocm_probe_load_oom", default = None
)


@contextlib.contextmanager
def load_scope() -> Iterator[None]:
    """One load: an OOM probe inside it is answered from memo for the rest of the load and nowhere else."""
    token = _load_oom.set({})
    try:
        yield
    finally:
        _load_oom.reset(token)


def load_scoped(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with load_scope():
            return fn(*args, **kwargs)

    return wrapper


def rocm_compile_setting() -> str:
    raw = (os.environ.get(ENV) or "auto").strip().lower()
    if raw in ("0", "off", "false", "no"):
        return "off"
    if raw in ("1", "on", "true", "yes", "force"):
        return "on"
    return "auto"


def rocm_compile_allowed() -> bool:
    return rocm_compile_setting() != "off"


def rocm_block_compile_works(device: Any = None) -> bool:
    return rocm_block_compile_probe(device)[0]


def rocm_block_compile_known(device: Any = None) -> Optional[bool]:
    """The probe's answer without running it (None = unprobed): preflight and status paths must not allocate."""
    setting = rocm_compile_setting()
    if setting != "auto":
        return setting == "on"
    # Lock-free: _lock is held for the whole multi-second probe, and a status read must not wait on it.
    hit = _result.get(_index(device))
    return None if hit is None else hit[0]


def _index(device: Any) -> Any:
    # A bare ordinal: torch.device(int) is not accepted by every supported torch.
    if isinstance(device, int) and not isinstance(device, bool):
        return device
    try:
        import torch
        index = torch.cuda.current_device() if device is None else torch.device(device).index
        return torch.cuda.current_device() if index is None else index
    except Exception:  # noqa: BLE001
        return 0


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
    from .rocm_bf16 import is_rocm_torch

    # Not torch.version.hip alone: AMD SDK / Radeon wheels leave it unset and carry the tag in __version__.
    if not is_rocm_torch(torch):
        return True, "not ROCm"
    index = _index(device)
    memo = _load_oom.get()
    with _lock:
        # The load's own OOM first: a concurrent load's later success must not send this load into a compile.
        if memo is not None and index in memo:
            return memo[index]
        if index in _result:
            return _result[index]
        ok, why = _probe(torch, index)
        (logger.info if ok else logger.warning)(
            f"diffusion.speed: ROCm regional compile probe on cuda:{index}: "
            f"{'ok' if ok else 'failed, staying eager'} ({why})"
        )
        if not ok and "OutOfMemoryError" in why:
            if memo is not None:
                memo[index] = (ok, why)
        else:
            _result[index] = (ok, why)
        return ok, why


def _probe(torch: Any, index: int) -> tuple[bool, str]:
    start = time.perf_counter()
    try:
        import torch.nn.functional as F
        from torch import nn

        class _Block(nn.Module):
            def __init__(
                self,
                dim: int = 128,
                heads: int = 4,
            ) -> None:
                super().__init__()
                self.heads = heads
                self.norm1 = nn.LayerNorm(dim, elementwise_affine = False)
                self.qkv = nn.Linear(dim, 3 * dim)
                self.out = nn.Linear(dim, dim)
                self.norm2 = nn.LayerNorm(dim)
                self.mlp = nn.Sequential(
                    nn.Linear(dim, 4 * dim), nn.GELU(approximate = "tanh"), nn.Linear(4 * dim, dim)
                )

            def forward(self, x: Any, scale: Any) -> Any:
                b, n, d = x.shape
                h = self.norm1(x) * (1 + scale)
                q, k, v = (
                    self.qkv(h).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
                )
                a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
                x = x + self.out(a)
                return x + self.mlp(self.norm2(x))

        dev = torch.device("cuda", index)
        dtype = torch.bfloat16
        gen = torch.Generator(device = "cpu").manual_seed(0)
        # Forked: the parameter init must not advance the process CPU RNG a caller may have seeded.
        with torch.random.fork_rng(devices = []):
            block = _Block()
        block = block.to(device = dev, dtype = dtype).eval()
        inductor_cfg = torch._inductor.config
        patches = {"memory_planning": False} if hasattr(inductor_cfg, "memory_planning") else {}
        compiled = torch.compile(block, dynamic = True, fullgraph = True)
        worst = 0.0
        with (
            torch.no_grad(),
            inductor_cfg.patch(patches),
            torch._dynamo.config.patch(suppress_errors = False),
        ):
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
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        try:
            from .diffusion_batched import is_oom_error

            # Classified before flattening: a compiler error wrapping an OOM, or a bare "HIP out of memory".
            if is_oom_error(exc) and name != "OutOfMemoryError":
                name = f"OutOfMemoryError ({name})"
        except Exception:  # noqa: BLE001, S110
            pass
        return False, f"{name}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"


def _reset_for_tests() -> None:
    with _lock:
        _result.clear()


def probe_result(index: int = 0) -> Optional[tuple[bool, str]]:
    return _result.get(index)
