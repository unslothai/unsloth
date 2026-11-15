# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-row AdaLN modulation and gated residual for MiniMax-H3 blocks via Triton, reading AdaLN rows through
``idx`` instead of ``index_select`` gathers. Bit-identical to eager: each op is fp32 rounded to the activation dtype
with explicit-rounding PTX (no FMA contraction). Engages per (device, dtype, hidden) only after a bit-exact check and
a timed win; otherwise (and on a first call during CUDA graph capture) the stock math runs. ``UNSLOTH_H3_ROW_MOD=0``
disables it.
"""

from __future__ import annotations

import inspect
import os
import threading
import types
from functools import lru_cache
from typing import Any, Optional

from .rocm_bf16 import is_rocm_torch

ROW_MOD_ENV = "UNSLOTH_H3_ROW_MOD"
_FORWARD_MARK = "_unsloth_h3_row_mod"
_BLOCK = 1024
_LOCK = threading.Lock()
_VERDICTS: dict = {}
STATS = {"fused": 0, "stock": 0}


def row_mod_enabled() -> bool:
    return (os.environ.get(ROW_MOD_ENV) or "1").strip().lower() not in ("0", "off", "false", "no")


def reference_modulate(x: Any, scale: Any, shift: Any, idx: Any) -> Any:
    return x * (1.0 + scale.index_select(0, idx)) + shift.index_select(0, idx)


def reference_gated_residual(residual: Any, gate: Any, out: Any, idx: Any) -> Any:
    return residual + gate.index_select(0, idx) * out


@lru_cache(maxsize = None)
def _kernels() -> Any:
    import triton
    import triton.language as tl

    @triton.jit
    def _mul_rn(a, b):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype = tl.float32, is_pure = True, pack = 1
        )

    @triton.jit
    def _add_rn(a, b):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype = tl.float32, is_pure = True, pack = 1
        )

    @triton.jit(do_not_specialize = ["n_cols", "x_stride", "t_stride"])
    def _modulate_kernel(
        x_ptr, scale_ptr, shift_ptr, idx_ptr, y_ptr, n_cols, x_stride, t_stride, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < n_cols
        t = tl.load(idx_ptr + row).to(tl.int64)
        dt = y_ptr.dtype.element_ty
        x = tl.load(x_ptr + row * x_stride + cols, mask = mask, other = 0.0).to(tl.float32)
        s = tl.load(scale_ptr + t * t_stride + cols, mask = mask, other = 0.0).to(tl.float32)
        sh = tl.load(shift_ptr + t * t_stride + cols, mask = mask, other = 0.0).to(tl.float32)
        one = tl.full([BLOCK], 1.0, tl.float32)
        s1 = _add_rn(s, one).to(dt).to(tl.float32)
        p = _mul_rn(x, s1).to(dt).to(tl.float32)
        tl.store(y_ptr + row * n_cols + cols, _add_rn(p, sh).to(dt), mask = mask)

    @triton.jit(do_not_specialize = ["n_cols", "r_stride", "o_stride", "t_stride"])
    def _gated_residual_kernel(
        r_ptr,
        gate_ptr,
        o_ptr,
        idx_ptr,
        y_ptr,
        n_cols,
        r_stride,
        o_stride,
        t_stride,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < n_cols
        t = tl.load(idx_ptr + row).to(tl.int64)
        dt = y_ptr.dtype.element_ty
        r = tl.load(r_ptr + row * r_stride + cols, mask = mask, other = 0.0).to(tl.float32)
        g = tl.load(gate_ptr + t * t_stride + cols, mask = mask, other = 0.0).to(tl.float32)
        o = tl.load(o_ptr + row * o_stride + cols, mask = mask, other = 0.0).to(tl.float32)
        p = _mul_rn(g, o).to(dt).to(tl.float32)
        tl.store(y_ptr + row * n_cols + cols, _add_rn(r, p).to(dt), mask = mask)

    return _modulate_kernel, _gated_residual_kernel


def _rows(t: Any) -> Optional[tuple]:
    if t.dim() == 3:
        if t.shape[0] != 1:
            return None
        if t.stride(-1) != 1:
            return None
        return int(t.shape[1]), int(t.stride(1))
    if t.dim() == 2 and t.stride(-1) == 1:
        return int(t.shape[0]), int(t.stride(0))
    return None


def _table_ok(table: Any, x: Any) -> bool:
    return (
        table.dim() == 2
        and table.dtype == x.dtype
        and table.device == x.device
        and table.stride(-1) == 1
        and table.shape[-1] == x.shape[-1]
    )


def _eligible(x: Any, idx: Any, *tables: Any) -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    if not (getattr(x, "is_cuda", False) and x.dtype in (torch.bfloat16, torch.float16)):
        return False
    if is_rocm_torch(torch) or _rows(x) is None:
        return False
    if not (idx.dim() == 1 and idx.device == x.device and idx.dtype in (torch.int32, torch.int64)):
        return False
    if idx.shape[0] != _rows(x)[0]:
        return False
    return all(_table_ok(t, x) for t in tables)


def _launch_modulate(x: Any, scale: Any, shift: Any, idx: Any) -> Any:
    import torch
    import triton

    n_rows, stride = _rows(x)
    n_cols = int(x.shape[-1])
    y = torch.empty(x.shape, dtype = x.dtype, device = x.device)
    if n_rows:
        _kernels()[0][(n_rows, triton.cdiv(n_cols, _BLOCK))](
            x, scale, shift, idx, y, n_cols, stride, scale.stride(0), BLOCK = _BLOCK, num_warps = 4
        )
    return y


def _launch_gated(residual: Any, gate: Any, out: Any, idx: Any) -> Any:
    import torch
    import triton

    n_rows, r_stride = _rows(residual)
    _, o_stride = _rows(out)
    n_cols = int(residual.shape[-1])
    y = torch.empty(residual.shape, dtype = residual.dtype, device = residual.device)
    if n_rows:
        _kernels()[1][(n_rows, triton.cdiv(n_cols, _BLOCK))](
            residual,
            gate,
            out,
            idx,
            y,
            n_cols,
            r_stride,
            o_stride,
            gate.stride(0),
            BLOCK = _BLOCK,
            num_warps = 4,
        )
    return y


def _probe(
    device: Any,
    dtype: Any,
    dim: int,
    seed: int,
    rows: int = 389,
    table_rows: int = 6,
) -> tuple:
    import torch

    g = torch.Generator(device = "cpu").manual_seed(seed)
    x = torch.randn(1, rows, dim, generator = g) * torch.exp(torch.randn(1, rows, 1, generator = g) * 3)
    x[0, :3] = 0.0
    x[0, 3] = -0.0
    x[0, 4] *= 1e-30
    full = torch.randn(table_rows, 6 * dim, generator = g) * torch.exp(
        torch.randn(table_rows, 1, generator = g) * 2
    )
    full[0, :dim] = -1.0  # 1 + scale == 0
    idx = torch.randint(0, table_rows, (rows,), generator = g)
    out = torch.randn(1, rows, dim, generator = g) * 4
    x, full, out = (t.to(dtype).to(device) for t in (x, full, out))
    chunks = full.chunk(6, dim = -1)
    return x, chunks, idx.to(device), out


def _race_ms(fn: Any, reps: int = 7) -> float:
    import torch

    for _ in range(2):
        fn()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return sorted(times)[len(times) // 2]


def _check(device: Any, dtype: Any, dim: int) -> dict:
    import torch

    exact = True
    try:
        for seed in (21, 22, 23):
            x, (s0, s1, s2, s3, s4, s5), idx, out = _probe(device, dtype, dim, seed)
            if not torch.equal(
                _launch_modulate(x, s1, s0, idx), reference_modulate(x, s1, s0, idx)
            ):
                exact = False
            if not torch.equal(
                _launch_gated(x, s2, out, idx), reference_gated_residual(x, s2, out, idx)
            ):
                exact = False
            if not exact:
                break
    except torch.OutOfMemoryError:
        raise
    except Exception:  # noqa: BLE001 - a JIT / launch failure keeps the stock math
        exact = False
    ms: dict = {}
    if exact:
        x, (s0, s1, s2, *_rest), idx, out = _probe(device, dtype, dim, 29, rows = 4096, table_rows = 3)
        ms["stock"] = _race_ms(
            lambda: (reference_modulate(x, s1, s0, idx), reference_gated_residual(x, s2, out, idx))
        )
        ms["fused"] = _race_ms(
            lambda: (_launch_modulate(x, s1, s0, idx), _launch_gated(x, s2, out, idx))
        )
    use = bool(exact and ms.get("fused", float("inf")) < 0.98 * ms.get("stock", 0.0))
    return {"exact": exact, "use": use, "ms": {k: round(v, 5) for k, v in ms.items()}}


def verdict(x: Any) -> Optional[dict]:
    """``_check`` cached per (device, dtype, hidden size); None while capturing a CUDA graph before it ran."""
    import torch

    index = x.device.index if x.device.index is not None else torch.cuda.current_device()
    key = (index, x.dtype, int(x.shape[-1]))
    found = _VERDICTS.get(key)
    if found is not None:
        return found
    if torch.cuda.is_current_stream_capturing():
        return None
    with _LOCK:
        if key not in _VERDICTS:
            with torch.cuda.device(x.device):
                _VERDICTS[key] = _check(x.device, x.dtype, int(x.shape[-1]))
    return _VERDICTS[key]


def _use(x: Any, idx: Any, *tables: Any) -> bool:
    if not (row_mod_enabled() and _eligible(x, idx, *tables)):
        return False
    v = verdict(x)
    return bool(v and v["use"])


def modulate(x: Any, scale: Any, shift: Any, idx: Any) -> Any:
    if _use(x, idx, scale, shift):
        STATS["fused"] += 1
        return _launch_modulate(x, scale, shift, idx)
    STATS["stock"] += 1
    return reference_modulate(x, scale, shift, idx)


def gated_residual(residual: Any, gate: Any, out: Any, idx: Any) -> Any:
    if (
        out.dtype == residual.dtype
        and out.shape == residual.shape
        and _rows(out) is not None
        and _use(residual, idx, gate)
    ):
        STATS["fused"] += 1
        return _launch_gated(residual, gate, out, idx)
    STATS["stock"] += 1
    return reference_gated_residual(residual, gate, out, idx)


def _row_mod_block_forward(
    self: Any,
    hidden_states: Any,
    temb: Any,
    adaln_indices: Any,
    rotary_emb: Any,
    attention_mask: Any = None,
) -> Any:
    """The stock MiniMax-H3 block forward with the two row ops routed through ``modulate`` / ``gated_residual``."""
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)

    residual = hidden_states
    norm_hidden_states = modulate(self.norm1(hidden_states), scale_msa, shift_msa, adaln_indices)
    attn_output = self.attn(norm_hidden_states, rotary_emb, attention_mask)
    hidden_states = gated_residual(residual, gate_msa, attn_output, adaln_indices)

    residual = hidden_states
    norm_hidden_states = modulate(self.norm2(hidden_states), scale_mlp, shift_mlp, adaln_indices)
    ff_output = self.ff(norm_hidden_states)
    hidden_states = gated_residual(residual, gate_mlp, ff_output, adaln_indices)

    return hidden_states


# The stock block forward this replaces, op for op (whitespace-normalised source of its body).
_STOCK_BODY = (
    "shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)",
    "residual = hidden_states",
    "norm_hidden_states = self.norm1(hidden_states)",
    "norm_hidden_states = norm_hidden_states * ( 1.0 + scale_msa.index_select(0, adaln_indices) ) + "
    "shift_msa.index_select(0, adaln_indices)",
    "attn_output = self.attn(norm_hidden_states, rotary_emb, attention_mask)",
    "hidden_states = residual + gate_msa.index_select(0, adaln_indices) * attn_output",
    "residual = hidden_states",
    "norm_hidden_states = self.norm2(hidden_states)",
    "norm_hidden_states = norm_hidden_states * ( 1.0 + scale_mlp.index_select(0, adaln_indices) ) + "
    "shift_mlp.index_select(0, adaln_indices)",
    "ff_output = self.ff(norm_hidden_states)",
    "hidden_states = residual + gate_mlp.index_select(0, adaln_indices) * ff_output",
    "return hidden_states",
)


def _body_statements(fn: Any) -> Optional[tuple]:
    import ast
    import textwrap

    try:
        src = textwrap.dedent(inspect.getsource(fn))
        node = ast.parse(src).body[0]
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(node, ast.FunctionDef):
        return None
    params = [a.arg for a in node.args.args]
    if params != ["self", "hidden_states", "temb", "adaln_indices", "rotary_emb", "attention_mask"]:
        return None
    out = []
    lines = src.splitlines()
    for stmt in node.body:
        if isinstance(stmt, ast.Expr) and isinstance(getattr(stmt, "value", None), ast.Constant):
            continue
        text = " ".join(" ".join(lines[stmt.lineno - 1 : stmt.end_lineno]).split())
        out.append(text)
    return tuple(out)


def _norm(s: str) -> str:
    return " ".join(s.replace("(", " ( ").replace(")", " ) ").split())


def matches_stock_block(block: Any) -> bool:
    """True when the block's class forward is the stock MiniMax-H3 block forward, statement for statement."""
    for name in ("adaln_proj", "norm1", "norm2", "attn", "ff"):
        if not hasattr(block, name):
            return False
    body = _body_statements(type(block).forward)
    if body is None:
        return False
    return tuple(_norm(s) for s in body) == tuple(_norm(s) for s in _STOCK_BODY)


def install(transformer: Any, logger: Any = None) -> int:
    """Swap every matching block's forward for the row-op form; returns the count. Never raises."""
    if not row_mod_enabled():
        return 0
    try:
        import torch
        if is_rocm_torch(torch) or not torch.cuda.is_available():
            return 0
        import triton  # noqa: F401
    except Exception:  # noqa: BLE001
        return 0
    n = 0
    for block in getattr(transformer, "transformer_blocks", ()) or ():
        if block.__dict__.get(_FORWARD_MARK) or "forward" in block.__dict__:
            continue  # an instance forward someone else installed: leave it
        if not matches_stock_block(block):
            continue
        block.forward = types.MethodType(_row_mod_block_forward, block)
        block.__dict__[_FORWARD_MARK] = True
        n += 1
    if n and logger is not None:
        logger.info(
            "video.h3_rowmod: per-row AdaLN modulation + gated residual without gathers on %d blocks",
            n,
        )
    return n


def uninstall(transformer: Any) -> int:
    n = 0
    for block in getattr(transformer, "transformer_blocks", ()) or ():
        if block.__dict__.pop(_FORWARD_MARK, None):
            block.__dict__.pop("forward", None)
            n += 1
    return n
