# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Prebuilt CUTLASS int8 GEMM (C ABI, ctypes) with torchao's epilogue fused: sm75 fp16 out, sm8x bf16 out.

Any load failure or tile failing the bit-exact probe keeps today's path; a tile replaces it only when it wins
the race by ``_MARGIN``. ``UNSLOTH_DIFFUSION_INT8_CUTLASS=0`` disables, ``=1`` skips the race."""

from __future__ import annotations

import ctypes
import os
import threading
from functools import lru_cache
from typing import Any, Callable, Optional

CUTLASS_ENV = "UNSLOTH_DIFFUSION_INT8_CUTLASS"
CUTLASS_LIB_ENV = "UNSLOTH_DIFFUSION_INT8_CUTLASS_LIB"
WHEEL_MODULE = "unsloth_studio_i8gemm"
ABI_VERSION = 1
LIB_NAMES = ("libunsloth_i8gemm.so", "unsloth_i8gemm.dll")

_STATUS = {
    0: "ok",
    1: "bad arguments",
    2: "no kernel for this tile / dtype",
    3: "cannot implement",
    4: "initialize failed",
    5: "launch failed",
}
# ties keep today's path
_MARGIN = 0.02
_RACE_M = (512, 4096)
_LOCK = threading.Lock()
_LOAD_ERROR: list = [None]
_CALLS = [0]


def mode() -> str:
    raw = (os.environ.get(CUTLASS_ENV) or "").strip().lower()
    if raw in ("0", "off", "false", "no"):
        return "off"
    if raw in ("1", "on", "true", "yes", "force"):
        return "force"
    return "auto"


def family(capability: Optional[tuple]) -> Optional[int]:
    if capability is None:
        return None
    cap = (int(capability[0]), int(capability[1]))
    if cap == (7, 5):
        return 75
    if cap[0] == 8:
        return 80
    return None


def out_dtype_for(fam: int) -> Any:
    import torch
    return torch.float16 if fam == 75 else torch.bfloat16


def _candidate_paths() -> list:
    paths = []
    explicit = (os.environ.get(CUTLASS_LIB_ENV) or "").strip()
    if explicit:
        paths.append(explicit)
        return paths
    try:
        import importlib.util
        spec = importlib.util.find_spec(WHEEL_MODULE)
        roots = list(spec.submodule_search_locations or ()) if spec is not None else []
    except Exception:  # noqa: BLE001 - no wheel
        roots = []
    for root in roots:
        for name in LIB_NAMES:
            for sub in ("", "lib"):
                p = os.path.join(root, sub, name)
                if os.path.isfile(p):
                    paths.append(p)
    return paths


def _bind(lib: Any) -> Any:
    c = ctypes
    lib.ui8_abi_version.restype = c.c_int
    lib.ui8_abi_version.argtypes = []
    lib.ui8_num_tiles.restype = c.c_int
    lib.ui8_num_tiles.argtypes = [c.c_int]
    lib.ui8_tile_info.restype = c.c_int
    lib.ui8_tile_info.argtypes = [c.c_int, c.c_int, c.POINTER(c.c_int)]
    lib.ui8_has_kernel.restype = c.c_int
    lib.ui8_has_kernel.argtypes = [c.c_int] * 7
    lib.ui8_gemm.restype = c.c_int
    lib.ui8_gemm.argtypes = [c.c_int] * 6 + [c.c_void_p] * 6 + [c.c_int64] * 6 + [c.c_void_p]
    return lib


@lru_cache(maxsize = 1)
def library() -> Optional[Any]:
    if mode() == "off":
        return None
    try:
        import torch

        from .rocm_bf16 import is_rocm_torch
        if not torch.cuda.is_available() or is_rocm_torch(torch):
            return None
    except Exception:  # noqa: BLE001
        return None
    for path in _candidate_paths():
        try:
            lib = _bind(ctypes.CDLL(path))
            if lib.ui8_abi_version() != ABI_VERSION:
                _LOAD_ERROR[0] = f"{path}: ABI {lib.ui8_abi_version()} != {ABI_VERSION}"
                continue
            return lib
        except Exception as exc:  # noqa: BLE001 - a broken binary keeps today's path
            _LOAD_ERROR[0] = f"{path}: {exc}"
    return None


def load_error() -> Optional[str]:
    return _LOAD_ERROR[0]


def library_path() -> Optional[str]:
    lib = library()
    return getattr(lib, "_name", None) if lib is not None else None


@lru_cache(maxsize = 4)
def tiles(fam: int) -> tuple:
    lib = library()
    if lib is None:
        return ()
    out = []
    buf = (ctypes.c_int * 8)()
    for i in range(lib.ui8_num_tiles(fam)):
        if lib.ui8_tile_info(fam, i, buf) == 0:
            out.append(tuple(buf))
    return tuple(out)


def _kind(dtype: Any) -> Optional[int]:
    import torch
    return {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}.get(dtype)


EPI_TORCHAO = 0
EPI_NATIVE = 1


def has_kernel(
    fam: int,
    tile: int,
    out_dtype: Any,
    xs_dtype: Any,
    ws_dtype: Any,
    bias: bool,
    epilogue: int = EPI_TORCHAO,
) -> bool:
    lib = library()
    kinds = (_kind(out_dtype), _kind(xs_dtype), _kind(ws_dtype))
    if lib is None or None in kinds:
        return False
    return bool(
        lib.ui8_has_kernel(fam, tile, kinds[0], kinds[1], kinds[2], int(bool(bias)), int(epilogue))
    )


def eligible(a: Any, w: Any, xs: Any, ws: Any, bias: Any, out_dtype: Any) -> bool:
    import torch
    try:
        if a.dtype != torch.int8 or w.dtype != torch.int8 or a.dim() != 2 or w.dim() != 2:
            return False
        if a.device != w.device or not a.is_cuda:
            return False
        m, k = a.shape
        n = w.shape[0]
        if w.shape[1] != k or k % 16 or n % 8 or m < 1:
            return False
        if a.stride(1) != 1 or w.stride(1) != 1 or a.stride(0) % 16 or w.stride(0) % 16:
            return False
        if a.data_ptr() % 16 or w.data_ptr() % 16:
            return False
        for vec, size in ((xs, m), (ws, n)):
            if vec.device != a.device or vec.numel() != size or (vec.dim() and vec.stride(-1) != 1):
                return False
            if vec.data_ptr() % 16 or _kind(vec.dtype) is None:
                return False
        if bias is not None:
            if (
                bias.dtype != out_dtype
                or bias.numel() != n
                or bias.data_ptr() % 16
                or bias.device != a.device
            ):
                return False
            if bias.dim() and bias.stride(-1) != 1:
                return False
        return True
    except Exception:  # noqa: BLE001
        return False


def launch(
    a: Any,
    w: Any,
    xs: Any,
    ws: Any,
    bias: Any,
    fam: int,
    tile: int,
    out_dtype: Any,
    epilogue: int = EPI_TORCHAO,
) -> Any:
    import torch

    lib = library()
    if lib is None:
        raise RuntimeError("int8 CUTLASS library not loaded")
    m, k = a.shape
    n = w.shape[0]
    out = torch.empty((m, n), device = a.device, dtype = out_dtype)
    with torch.cuda.device(a.device):
        stream = torch.cuda.current_stream(a.device).cuda_stream
        status = lib.ui8_gemm(
            fam,
            tile,
            _kind(out_dtype),
            _kind(xs.dtype),
            _kind(ws.dtype),
            int(epilogue),
            a.data_ptr(),
            w.data_ptr(),
            xs.data_ptr(),
            ws.data_ptr(),
            bias.data_ptr() if bias is not None else None,
            out.data_ptr(),
            m,
            n,
            k,
            a.stride(0),
            w.stride(0),
            out.stride(0),
            stream,
        )
    if status != 0:
        raise RuntimeError(f"ui8_gemm tile {tile}: {_STATUS.get(status, status)}")
    _CALLS[0] += 1
    return out


def reference(a: Any, w: Any, xs: Any, ws: Any, bias: Any, out_dtype: Any) -> Any:
    """torchao's int8 dynamic Linear epilogue op for op (fp16 x_scale computes in fp32)."""
    import torch

    m = a.shape[0]
    if m < 17:  # _int_mm row floor
        a = torch.cat([a, a.new_zeros((17 - m, a.shape[1]))])
    c = torch._int_mm(a, w.t())[:m]
    inter = torch.float32 if xs.dtype == torch.float16 else xs.dtype
    y = (c * xs.reshape(-1, 1).to(inter)).to(out_dtype)
    y = y * ws.reshape(-1)
    if bias is not None:
        y = y + bias
    return y.to(out_dtype)


def reference_native(a: Any, w: Any, xs: Any, ws: Any, out_dtype: Any) -> Any:
    """``NativeWeightOnlyLinear._int_mm_rows``: fp32 epilogue, one cast."""
    import torch

    m = a.shape[0]
    if m < 17:
        a = torch.cat([a, a.new_zeros((17 - m, a.shape[1]))])
    c = torch._int_mm(a, w.t())[:m]
    out = torch.mul(c, xs.reshape(-1, 1).float())
    return out.mul_(ws.reshape(-1).float()).to(out_dtype)


# (M, N, K, bias): ragged tails, K below / above one tile
_PROBE_SHAPES = (
    (257, 384, 512, False),
    (33, 208, 144, True),
    (300, 528, 1040, True),
    (1, 64, 64, False),
)


def _scale_dtypes(fam: int) -> tuple:
    import torch
    if fam == 75:
        return tuple(
            (x, w)
            for x in (torch.float16, torch.float32)
            for w in (torch.float16, torch.bfloat16, torch.float32)
        )
    return tuple(
        (x, w) for x in (torch.bfloat16, torch.float32) for w in (torch.bfloat16, torch.float32)
    )


def _probe_tile(index: int, fam: int, tile: int, pairs: tuple) -> bool:
    import torch

    from .diffusion_int8_gemm import tie_operands

    dev = torch.device("cuda", index)
    out_dtype = out_dtype_for(fam)
    g = torch.Generator(device = "cpu").manual_seed(0)
    try:
        for xs_dt, ws_dt in pairs:
            for m, n, k, bias in _PROBE_SHAPES:
                if not has_kernel(fam, tile, out_dtype, xs_dt, ws_dt, bias):
                    return False
                a = torch.randint(-127, 128, (m, k), generator = g, dtype = torch.int8).to(dev)
                w = torch.randint(-127, 128, (n, k), generator = g, dtype = torch.int8).to(dev)
                xs = (torch.rand(m, generator = g) * 0.02 + 1e-4).to(xs_dt).to(dev)
                ws = (torch.rand(n, generator = g) * 0.002 + 1e-5).to(ws_dt).to(dev)
                b = (torch.randn(n, generator = g) * 0.1).to(out_dtype).to(dev) if bias else None
                if not torch.equal(
                    launch(a, w, xs, ws, b, fam, tile, out_dtype),
                    reference(a, w, xs, ws, b, out_dtype),
                ):
                    return False
            # near-midpoint products: the double rounding must match
            a, w = tie_operands(dev)
            xs = torch.ones(a.shape[0], device = dev, dtype = xs_dt)
            ws = torch.ones(w.shape[0], device = dev, dtype = ws_dt)
            if fam == 75:
                # keep fp16 out of inf
                xs = torch.full((a.shape[0],), 2.0**-12, device = dev, dtype = xs_dt)
            if not torch.equal(
                launch(a, w, xs, ws, None, fam, tile, out_dtype),
                reference(a, w, xs, ws, None, out_dtype),
            ):
                return False
        if has_kernel(fam, tile, out_dtype, torch.float32, torch.float32, False, EPI_NATIVE):
            for m, n, k, _bias in _PROBE_SHAPES:
                a = torch.randint(-127, 128, (m, k), generator = g, dtype = torch.int8).to(dev)
                w = torch.randint(-127, 128, (n, k), generator = g, dtype = torch.int8).to(dev)
                xs = (torch.rand(m, generator = g) * 0.02 + 1e-4).to(dev)
                ws = (torch.rand(n, generator = g) * 0.002 + 1e-5).to(dev)
                got = launch(a, w, xs, ws, None, fam, tile, out_dtype, EPI_NATIVE)
                if not torch.equal(got, reference_native(a, w, xs, ws, out_dtype)):
                    return False
            a, w = tie_operands(dev)
            xs = torch.full((a.shape[0],), 2.0**-12, device = dev, dtype = torch.float32)
            ws = torch.ones(w.shape[0], device = dev, dtype = torch.float32)
            got = launch(a, w, xs, ws, None, fam, tile, out_dtype, EPI_NATIVE)
            if not torch.equal(got, reference_native(a, w, xs, ws, out_dtype)):
                return False
        torch.cuda.synchronize(dev)
        return True
    except Exception:  # noqa: BLE001
        return False


_PROBED: dict = {}


def probed_tiles(index: int) -> tuple:
    if index in _PROBED:
        return _PROBED[index]
    ok: tuple = ()
    try:
        import torch
        fam = family(torch.cuda.get_device_capability(index))
        if fam is not None and library() is not None:
            pairs = _scale_dtypes(fam)
            ok = tuple(t for t in range(len(tiles(fam))) if _probe_tile(index, fam, t, pairs))
    except Exception:  # noqa: BLE001
        ok = ()
    _PROBED[index] = ok
    return ok


def available(index: int) -> bool:
    return bool(probed_tiles(index))


def time_us(
    fns: dict,
    rounds: int = 3,
    reps: int = 3,
    warm: int = 1,
) -> dict:
    """{name: us}, min over interleaved rounds."""
    import torch

    start, end = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
    for fn in fns.values():
        for _ in range(warm):
            fn()
    torch.cuda.synchronize()
    best = {name: float("inf") for name in fns}
    for _ in range(rounds):
        for name, fn in fns.items():
            start.record()
            for _ in range(reps):
                fn()
            end.record()
            end.synchronize()
            best[name] = min(best[name], start.elapsed_time(end) * 1000.0 / reps)
    return best


def race(
    index: int,
    w: Any,
    ws: Any,
    bias: Any,
    xs_dtype: Any,
    today: Optional[Callable],
    m: int,
    epilogue: int = EPI_TORCHAO,
) -> dict:
    """Best probed tile vs ``today`` (None = cuBLAS ``_int_mm``); tile None keeps today's path."""
    import torch

    fam = family(torch.cuda.get_device_capability(index))
    tiles_ok = probed_tiles(index)
    if fam is None or not tiles_ok:
        return {"tile": None, "us": {}}
    out_dtype = out_dtype_for(fam)
    dev = w.device
    k = w.shape[1]
    g = torch.Generator(device = "cpu").manual_seed(7)
    a = torch.randint(-127, 128, (m, k), generator = g, dtype = torch.int8).to(dev)
    xs = (torch.rand(m, generator = g) * 0.02 + 1e-4).to(xs_dtype).to(dev)
    if not eligible(a, w, xs, ws, bias, out_dtype):
        return {"tile": None, "us": {}}
    cands = [
        t
        for t in tiles_ok
        if has_kernel(fam, t, out_dtype, xs_dtype, ws.dtype, bias is not None, epilogue)
    ]
    if not cands:
        return {"tile": None, "us": {}}

    def cut(t):
        return lambda: launch(a, w, xs, ws, bias, fam, t, out_dtype, epilogue)

    if today is None:
        base = lambda: torch._int_mm(a, w.t()).amax()  # noqa: E731
        base_name = "cublas"
    else:
        base = lambda: today(a, w, xs, ws, bias)  # noqa: E731
        base_name = "today"
    with torch.cuda.device(dev):
        screen = time_us({t: cut(t) for t in cands}, rounds = 1, reps = 2, warm = 1)
        top = sorted(cands, key = lambda t: screen[t])[:3]
        fns = {base_name: base}
        fns.update({t: cut(t) for t in top})
        timing = time_us(fns, rounds = 3, reps = 3, warm = 1)
    best = min(top, key = lambda t: timing[t])
    keep = timing[best] < timing[base_name] * (1.0 - _MARGIN)
    us = {str(k_): round(v, 1) for k_, v in timing.items()}
    return {"tile": best if keep else None, "us": us, "m": m}


# (card, N, K) -> {race M: tile or None}, sm75 NativeWeightOnlyLinear
_NATIVE_PLAN: dict = {}
_NATIVE_RACES: dict = {}
_NATIVE_LARGE_M = 1024


def native_prepare(weight_q: Any, x_dtype: Any, has_bias: bool) -> None:
    """Once per (card, N, K), outside any capture: race native-epilogue tiles (sm75, fp16, no bias)."""
    import torch
    try:
        if mode() == "off" or has_bias or x_dtype != torch.float16 or not weight_q.is_cuda:
            return
        index = weight_q.device.index
        key = (index, int(weight_q.shape[0]), int(weight_q.shape[1]))
        if key in _NATIVE_PLAN:
            return
        if torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling():
            return
        _NATIVE_PLAN[key] = {}
        if family(torch.cuda.get_device_capability(index)) != 75 or not probed_tiles(index):
            return
        g = torch.Generator(device = "cpu").manual_seed(13)
        ws = (torch.rand(weight_q.shape[0], generator = g) * 1e-3 + 1e-5).to(weight_q.device)
        today = lambda a, w_, x_, s_, b_: reference_native(a, w_, x_, s_, torch.float16)  # noqa: E731
        entry = {}
        for m in _RACE_M:
            r = race(index, weight_q, ws, None, torch.float32, today, m, epilogue = EPI_NATIVE)
            if mode() == "force" and r.get("us"):
                digits = [t for t in r["us"] if t.isdigit()]
                r["tile"] = int(min(digits, key = lambda t: r["us"][t])) if digits else None
            entry[m] = r["tile"]
            _NATIVE_RACES.setdefault(key, {})[m] = r
        _NATIVE_PLAN[key] = entry
    except Exception:  # noqa: BLE001
        return


def native_tile(xq: Any, weight_q: Any, x_scale: Any, w_scale: Any) -> Optional[int]:
    import torch

    if torch.compiler.is_compiling():
        return None
    entry = _NATIVE_PLAN.get(
        (weight_q.device.index, int(weight_q.shape[0]), int(weight_q.shape[1]))
    )
    if not entry:
        return None
    tile = entry.get(max(_RACE_M) if xq.shape[0] >= _NATIVE_LARGE_M else min(_RACE_M))
    if tile is None:
        return None
    if x_scale.dtype != torch.float32 or w_scale.dtype != torch.float32:
        return None
    return (
        tile
        if eligible(xq, weight_q, x_scale.reshape(-1), w_scale.reshape(-1), None, torch.float16)
        else None
    )


def native_gemm(xq: Any, weight_q: Any, x_scale: Any, w_scale: Any, tile: int) -> Any:
    import torch
    return launch(
        xq,
        weight_q,
        x_scale.reshape(-1),
        w_scale.reshape(-1),
        None,
        75,
        tile,
        torch.float16,
        EPI_NATIVE,
    )
