# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Grouped-scale packed integer weights (W4A8 / W6A8): 4- or 6-bit codes, fp8 group steps, an fp32 channel scale.

Storage of one Linear ``L`` (``N`` outputs, ``K`` inputs, ``b`` = 4 or 6 bits, group ``g``):

- ``L.weight`` int8 ``[N, K * b / 8]``: a ``K / 2``-byte plane of nibbles (column ``2j`` in the low nibble of byte
  ``j``, column ``2j + 1`` in the high nibble), followed at 6 bits by a ``K / 4``-byte plane holding each code's top two
  bits (column ``c`` in byte ``c // 4`` at bit ``2 * (c % 4)``). The bit depth is implied by the row width.
- ``L.weight_s_rel`` float8_e4m3fn (or its raw bytes) ``[N, K / g]``: each group's step relative to the channel scale.
- ``L.weight_s_channel`` float32 ``[N]``.
- ``L.weight_codebook`` float32 ``[16]``, 4-bit only and optional: the level of each code. Without it a 4-bit code
  ``c`` means ``c - 8`` and a 6-bit code ``c - 32``.
- ``L.comfy_quant``: JSON ``{"format": "asym_w4a8_int8" | "w6a8_int8", "group_size": g, "convrot": true,
  "convrot_groupsize": 256}`` (defaults 16 / 256; the rotation is always on).

Decode: ``q8 = clamp(rint(level(code) * s_rel), -127, 127)`` int8, rounded in float32 (half to even); a layer is then
exactly an int8 ConvRot layer with a per-row scale, so every int8 runtime and the dense fallback serve it unchanged.
"""

from __future__ import annotations

import json
from typing import Any, Optional

ASYM_W4A8 = "asym_w4a8_int8"
W6A8 = "w6a8_int8"
GROUPED_BITS = {ASYM_W4A8: 4, W6A8: 6}
DEFAULT_GROUP = 16
DEFAULT_ROTATION = 256
GROUPED_SUFFIXES = (".weight_s_rel", ".weight_s_channel", ".weight_codebook")
_E4M3_MAX = 448.0
_DECODE_CHUNK_ELEMS = 1 << 22
_DECODE_SCRATCH_BYTES = 24


def decode_host_bytes(rows: int, k: int) -> int:
    """Host bytes ``decode_int8_torch`` allocates for a ``rows x k`` layer: the int8 grid plus one chunk's scratch."""
    step = max(1, _DECODE_CHUNK_ELEMS // max(int(k), 1))
    return int(rows) * int(k) + min(int(rows), step) * int(k) * _DECODE_SCRATCH_BYTES


def grouped_conf(raw: Any) -> tuple:
    """``(bits, group_size, rotation_group)`` of a declaration, the file format's defaults filled in."""
    params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
    fmt = raw.get("format")
    group = raw.get("group_size", params.get("group_size", DEFAULT_GROUP))
    rotation = raw.get("convrot_groupsize", params.get("convrot_groupsize", DEFAULT_ROTATION))
    return GROUPED_BITS.get(fmt), group, rotation


def _numel(entry: Optional[dict]) -> int:
    n = 1
    for d in (entry or {}).get("shape") or ():
        n *= int(d)
    return n


def grouped_layer_problem(
    name: str, raw: Any, header: dict, other_suffixes: tuple
) -> Optional[str]:
    """Why a W4A8 / W6A8 layer cannot be decoded, from its header entries alone, or None. Torch-free."""
    fmt = raw.get("format")
    bits, group, rotation = grouped_conf(raw)
    weight = header.get(name + ".weight")
    if weight is None:
        return f"{fmt} declared but there is no {name}.weight"
    shape = weight.get("shape") or ()
    if len(shape) != 2 or weight.get("dtype") != "I8":
        return f"{fmt} weight {weight.get('dtype')} {list(shape)} (expected packed 2-D int8)"
    if not isinstance(group, int) or isinstance(group, bool) or group < 4:
        return f"{fmt} group_size {group!r}"
    if not isinstance(rotation, int) or isinstance(rotation, bool):
        return f"{fmt} convrot_groupsize {rotation!r}"
    from .diffusion_convrot import is_power_of_four

    if not is_power_of_four(rotation):
        return f"ConvRot group {rotation!r} is not a power of 4"
    rows, cols = (int(d) for d in shape)
    if (cols * 8) % bits:
        return f"{fmt} weight with {cols} packed columns"
    k = cols * 8 // bits
    if k % 16 or k % group or k % rotation or (16 % group and group % 16):
        return f"{fmt} in_features {k} with group_size {group} and ConvRot group {rotation}"
    if bits == 6 and (k % 32 or group % 16):
        return f"{fmt} needs in_features divisible by 32 and a group_size multiple of 16 (K={k}, group {group})"
    s_rel = header.get(name + ".weight_s_rel")
    if s_rel is None or tuple(int(d) for d in s_rel.get("shape") or ()) != (rows, k // group):
        return f"{fmt} needs weight_s_rel of shape {[rows, k // group]}"
    if s_rel.get("dtype") not in ("F8_E4M3", "U8"):
        return f"{fmt} group scales stored as {s_rel.get('dtype')}"
    s_channel = header.get(name + ".weight_s_channel")
    if (
        s_channel is None
        or _numel(s_channel) != rows
        or s_channel.get("dtype") not in ("F32", "F16", "BF16")
    ):
        return f"{fmt} needs a floating weight_s_channel with one entry per row ({rows})"
    codebook = header.get(name + ".weight_codebook")
    if codebook is not None:
        if bits != 4:
            return f"{fmt} carries a codebook (6-bit codes are uniform)"
        if _numel(codebook) != 16 or codebook.get("dtype") not in ("F32", "F16", "BF16"):
            return f"{fmt} codebook of shape {codebook.get('shape')} (expected 16 levels)"
    leftover = [s for s in other_suffixes if name + s in header and s not in GROUPED_SUFFIXES]
    if leftover:
        return f"{fmt} layer carries {', '.join(leftover)}, which this format does not use"
    return None


def grouped_in_features(fmt: str, packed_cols: int) -> int:
    return int(packed_cols) * 8 // GROUPED_BITS[fmt]


def unpack_codes(qdata: Any, k: int, bits: int) -> Any:
    """Packed rows ``[N, K * bits / 8]`` (int8 or uint8) -> unsigned codes ``[N, K]`` as int32."""
    import torch

    packed = qdata.view(torch.uint8).to(torch.int32)
    rows = packed.shape[0]
    low = packed[:, : k // 2]
    codes = torch.stack((low & 0xF, low >> 4), dim = -1).reshape(rows, k)
    if bits == 6:
        hi = packed[:, k // 2 : k // 2 + k // 4]
        top = torch.stack([(hi >> (2 * j)) & 0x3 for j in range(4)], dim = -1).reshape(rows, k)
        codes = codes | (top << 4)
    return codes


def pack_codes(codes: Any, bits: int) -> Any:
    """Unsigned codes ``[N, K]`` -> the packed int8 rows ``unpack_codes`` reads."""
    import torch

    c = codes.to(torch.int32)
    low = ((c[:, 0::2] & 0xF) | ((c[:, 1::2] & 0xF) << 4)).to(torch.uint8)
    if bits == 4:
        return low.view(torch.int8).contiguous()
    top = (c >> 4) & 0x3
    hi = top[:, 0::4] | (top[:, 1::4] << 2) | (top[:, 2::4] << 4) | (top[:, 3::4] << 6)
    return torch.cat((low, hi.to(torch.uint8)), dim = 1).view(torch.int8).contiguous()


def level_table(
    bits: int,
    codebook: Any = None,
    device: Any = "cpu",
) -> Any:
    """float32 level of every code: the codebook (4-bit), else ``code - 2 ** (bits - 1)``."""
    import torch

    if codebook is not None:
        if bits != 4:
            raise ValueError("only 4-bit codes take a codebook")
        return codebook.reshape(-1).to(device = device, dtype = torch.float32)
    return torch.arange(1 << bits, dtype = torch.float32, device = device) - float(1 << (bits - 1))


def s_rel_float32(s_rel: Any) -> Any:
    """Exact float32 of fp8 e4m3 group steps (raw bytes or float8), by table where float8 does not cast (MPS)."""
    import torch

    if s_rel.dtype == torch.float32:
        return s_rel
    if s_rel.dtype not in (torch.uint8, torch.float8_e4m3fn):
        return s_rel.to(torch.float32)
    raw = s_rel.view(torch.uint8)
    if raw.device.type in ("cpu", "cuda"):
        return raw.view(torch.float8_e4m3fn).to(torch.float32)
    from .diffusion_native_quant import fp8_lookup_table

    table = fp8_lookup_table("float8_e4m3fn", str(raw.device))
    return table.index_select(0, raw.reshape(-1).int()).view(raw.shape)


def decode_int8_torch(qdata: Any, s_rel: Any, codebook: Any, group_size: int, bits: int) -> Any:
    """The int8 grid ``[N, K]`` the GEMM consumes, row-chunked; bit-exact on every device (float32 maths only)."""
    import torch

    rows = int(qdata.shape[0])
    groups = int(s_rel.shape[1])
    k = groups * int(group_size)
    if int(qdata.shape[1]) * 8 != k * bits:
        raise ValueError(f"packed width {qdata.shape[1]} does not match K={k} at {bits} bits")
    table = level_table(bits, codebook, qdata.device)
    out = torch.empty((rows, k), dtype = torch.int8, device = qdata.device)
    step = max(1, _DECODE_CHUNK_ELEMS // max(k, 1))
    for r in range(0, rows, step):
        codes = unpack_codes(qdata[r : r + step], k, bits)
        values = table[codes].view(codes.shape[0], groups, group_size)
        steps = s_rel_float32(s_rel[r : r + step]).unsqueeze(-1)
        q = (values * steps).round_().clamp_(-127, 127)
        out[r : r + step] = q.view(codes.shape[0], k).to(torch.int8)
    return out


def decode_int8(qdata: Any, s_rel: Any, codebook: Any, group_size: int, bits: int) -> Any:
    """``decode_int8_torch``, by the Triton kernel on a CUDA / ROCm tensor when it builds (bit-identical)."""
    if getattr(qdata, "is_cuda", False):
        from .diffusion_grouped_int8_triton import decode_int8_triton
        out = decode_int8_triton(qdata, s_rel, codebook, group_size, bits)
        if out is not None:
            return out
    return decode_int8_torch(qdata, s_rel, codebook, group_size, bits)


def dequant_grouped(
    qdata: Any,
    s_rel: Any,
    s_channel: Any,
    codebook: Any,
    group_size: int,
    bits: int,
    rotation: int,
    dtype: Any,
) -> Any:
    """The dense weight in the original basis: ``q8 * s_channel`` in float32, the rotation undone, cast to ``dtype``."""
    import torch

    from .diffusion_convrot import build_convrot_hadamard

    q8 = decode_int8(qdata, s_rel, codebook, group_size, bits)
    rows, k = q8.shape
    w = q8.to(torch.float32) * s_channel.reshape(-1, 1).to(torch.float32)
    h = build_convrot_hadamard(rotation, device = w.device, dtype = torch.float32)
    return (w.reshape(rows, k // rotation, rotation) @ h).reshape(rows, k).to(dtype)


def _nearest_grid(target: Any, lo_level: int, hi_level: int, table: Any, s_rel32: Any) -> Any:
    """Per weight, the code whose decoded int8 value is nearest ``target`` (``W_rot / s_channel``), for a uniform
    level table (``table[c] = c - offset``): round, then check both neighbours on the decoded grid."""
    import torch

    offset = -float(table[0].item())
    steps = s_rel32.unsqueeze(-1)
    base = torch.where(steps > 0, target / steps.clamp(min = 1e-30), torch.zeros_like(target))
    base = base.round_().clamp_(lo_level, hi_level)
    best = None
    best_err = None
    for delta in (0.0, -1.0, 1.0):
        level = (base + delta).clamp_(lo_level, hi_level)
        decoded = (level * steps).round_().clamp_(-127, 127)
        err = (decoded - target).abs_()
        if best is None:
            best, best_err = level, err
        else:
            better = err < best_err
            best = torch.where(better, level, best)
            best_err = torch.where(better, err, best_err)
    return (best + offset).to(torch.int32)


def _nearest_codebook(target: Any, table: Any, s_rel32: Any) -> Any:
    import torch
    decoded = (
        (table.view(1, 1, 1, -1) * s_rel32.unsqueeze(-1).unsqueeze(-1)).round_().clamp_(-127, 127)
    )
    return (decoded - target.unsqueeze(-1)).abs_().argmin(dim = -1).to(torch.int32)


def check_grouped_config(
    fmt: str,
    group_size: Any,
    rotation: Any,
    codebook: Any = None,
) -> None:
    """Raise unless the reader (``grouped_layer_problem``) accepts this group / rotation / codebook for ``fmt``."""
    from .diffusion_convrot import is_power_of_four

    bits = GROUPED_BITS.get(fmt)
    if bits is None:
        raise ValueError(f"unknown packed format {fmt!r}")
    if (
        not isinstance(group_size, int)
        or isinstance(group_size, bool)
        or group_size < 4
        or (16 % group_size and group_size % 16)
        or (bits == 6 and group_size % 16)
    ):
        raise ValueError(
            f"{fmt}: group_size {group_size!r} (4-bit: a divisor or multiple of 16; 6-bit: a multiple of 16)"
        )
    if (
        not isinstance(rotation, int)
        or isinstance(rotation, bool)
        or not is_power_of_four(rotation)
    ):
        raise ValueError(f"{fmt}: ConvRot group {rotation!r} is not a power of 4")
    if codebook is not None and (bits != 4 or int(codebook.numel()) != 16):
        raise ValueError(f"{fmt}: a codebook is 16 levels and 4-bit only")


def quantize_grouped(
    weight: Any,
    fmt: str,
    *,
    group_size: int = DEFAULT_GROUP,
    rotation: int = DEFAULT_ROTATION,
    codebook: Any = None,
) -> dict:
    """Round-to-nearest W4A8 / W6A8 storage of a floating ``[N, K]`` weight, each code the nearest level on the
    decoded int8 grid. ``codebook`` (4-bit only): 16 sorted levels in [-1, 1]. Returns tensors by suffix."""
    import torch

    from .diffusion_convrot import build_convrot_hadamard

    bits = GROUPED_BITS[fmt]
    check_grouped_config(fmt, group_size, rotation, codebook)
    if weight.dim() != 2:
        raise ValueError(f"{fmt} needs a 2-D weight, got {tuple(weight.shape)}")
    rows, k = (int(d) for d in weight.shape)
    if k % rotation or k % group_size or k % 16 or (bits == 6 and (k % 32 or group_size % 16)):
        raise ValueError(
            f"{fmt}: in_features {k} with group {group_size} and ConvRot group {rotation}"
        )
    if codebook is not None and bits != 4:
        raise ValueError("only 4-bit codes take a codebook")
    groups = k // group_size
    h = build_convrot_hadamard(rotation, device = weight.device, dtype = torch.float32)
    w = (weight.to(torch.float32).reshape(rows, k // rotation, rotation) @ h).reshape(
        rows, groups, group_size
    )
    amax = w.abs().amax(dim = -1)
    if codebook is not None:
        table = codebook.reshape(-1).to(device = w.device, dtype = torch.float32)
        step = amax
        reach = table.abs().max()
    else:
        table = level_table(bits, None, w.device)
        reach = float((1 << (bits - 1)) - 1)  # 31 / 7: the symmetric half of the code range
        step = amax / reach
    recon_max = (step * reach).amax(dim = -1)
    s_channel = (recon_max / 127.0).clamp(min = 1e-12)
    s_rel = (step / s_channel.unsqueeze(-1)).clamp(max = _E4M3_MAX).to(torch.float8_e4m3fn)
    s_rel32 = s_rel.to(torch.float32)
    target = w / s_channel.view(-1, 1, 1)
    if codebook is not None:
        codes = _nearest_codebook(target, table, s_rel32)
    else:
        half = 1 << (bits - 1)
        # 4-bit uses -8..7, 6-bit the symmetric -31..31 (code 0 is never written)
        codes = _nearest_grid(target, -half if bits == 4 else 1 - half, half - 1, table, s_rel32)
    out = {
        "": pack_codes(codes.reshape(rows, k), bits),
        ".weight_s_rel": s_rel.contiguous(),
        ".weight_s_channel": s_channel.to(torch.float32).contiguous(),
    }
    if codebook is not None:
        out[".weight_codebook"] = table.clone()  # own storage: a shared one fails serialization
    return out


def grouped_declaration(
    fmt: str,
    group_size: int = DEFAULT_GROUP,
    rotation: int = DEFAULT_ROTATION,
) -> dict:
    return {
        "format": fmt,
        "group_size": int(group_size),
        "convrot": True,
        "convrot_groupsize": int(rotation),
    }


def declaration_tensor(conf: dict) -> Any:
    import torch
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype = torch.uint8)


import os  # noqa: E402
import threading  # noqa: E402

# Keep codes packed on device: unset = auto (``grouped_packed_choice``), ``1`` = any device, ``0`` = decode to int8 at load.
GROUPED_PACKED_ENV = "UNSLOTH_DIFFUSION_GROUPED_PACKED"
# unset = race int8 vs dense widen per shape, ``1`` = int8 GEMM, ``0`` = dense widen (kill switch).
GROUPED_GEMM_ENV = "UNSLOTH_DIFFUSION_GROUPED_INT8_GEMM"
GROUPED_PACKED_ATTR = "_unsloth_grouped_packed"
GROUPED_FORM_ATTR = "_unsloth_grouped_form"
_OFF = ("0", "off", "false", "no")
_ON = ("1", "on", "true", "yes", "force")
_MIN_ROWS = 17
_RACE_ITERS = 3
_RACE_LOCK = threading.Lock()
# (device, N, K, bits, rows bucket) -> "int8" | "dense"
_RACE: dict = {}
_CALLS = {"int8": 0, "int8r": 0, "fused": 0, "dense": 0}
_TUNED: dict = {}
_ROTQ: dict = {}
# per-card probes, decided once in eager (a compiled forward never probes)
_FUSED: dict = {}
# A route the race adds on top of the incumbent (int8 / dense) wins only by more than this fraction.
_RACE_MARGIN = 0.02


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip().lower()


def grouped_packed_choice(
    target: Any = None,
    int8_backend: Optional[str] = None,
    *,
    lora: bool = False,
    image: bool = False,
    tight: bool = False,
) -> tuple:
    """``(keep packed, reason)`` for the W4A8 / W6A8 layers of a load (``GROUPED_PACKED_ENV``). Never under a LoRA
    (dense Linears). An image denoiser stays packed only when ``tight`` (streamed / offloaded) on a card with Studio's
    tuned int8 GEMM; a video denoiser keeps packed codes (measured: bf16 speed at half the memory)."""
    raw = _env(GROUPED_PACKED_ENV)
    if lora:
        return False, "lora"
    if raw in _OFF:
        return False, "env_off"
    if raw in _ON:
        return True, "env_on"
    device = getattr(target, "torch_device", None) or getattr(target, "device", None)
    kind = getattr(device, "type", str(device or "").split(":")[0])
    if kind != "cuda":
        return False, "not_cuda"
    if int8_backend is None:
        return False, "no_int8_runtime"
    if not image:
        return True, "video"
    if tight:
        if not _tuned_arch(device):
            # L4 measured: streamed packed moves 18% fewer bytes but runs 9% slower than streamed int8
            return False, "memory_tight_untuned_arch"
        return True, "memory_tight"
    return False, "image_resident"


def _tuned_arch(device: Any) -> bool:
    """The card ``device`` names has a Studio tuned int8 GEMM tile (``diffusion_int8_gemm.arch_config``): only there
    does a memory-tight image load gain from packed codes. Eager, host-side only (no probe launch)."""
    try:
        import torch

        from .diffusion_int8_gemm import arch_config

        dev = torch.device(device) if not isinstance(device, torch.device) else device
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        return arch_config(torch.cuda.get_device_capability(index)) is not None
    except Exception:  # noqa: BLE001 -- unknown card: decode at load (the INT8 path)
        return False


def grouped_packed_wanted(
    target: Any = None,
    int8_backend: Optional[str] = None,
    *,
    lora: bool = False,
    image: bool = False,
    tight: bool = False,
) -> bool:
    """Whether a load keeps W4A8 / W6A8 layers packed (``grouped_packed_choice``)."""
    return grouped_packed_choice(target, int8_backend, lora = lora, image = image, tight = tight)[0]


def grouped_form_status(model: Any) -> Optional[dict]:
    """``{"form": "packed" | "decoded_at_load" | "dequantized", "formats", "layers", "reason"}`` of a denoiser loaded from a W4A8 /
    W6A8 file (the status field), None for any other load."""
    form = getattr(model, GROUPED_FORM_ATTR, None)
    return dict(form) if isinstance(form, dict) else None


def grouped_form_note(model: Any) -> str:
    """One status-line clause for ``grouped_form_status`` (empty without one)."""
    form = grouped_form_status(model)
    if not form:
        return ""
    names = (
        "/".join("W4A8" if f == ASYM_W4A8 else "W6A8" for f in form.get("formats") or ())
        or "W4A8/W6A8"
    )
    how = {"packed": "kept packed", "decoded_at_load": "decoded to int8 at load"}.get(
        form.get("form"), "dequantized at load"
    )
    return f"; {names} {how} ({form.get('layers', 0)} layers, {form.get('reason')})"


def grouped_gemm_mode() -> str:
    raw = _env(GROUPED_GEMM_ENV)
    if raw in _OFF:
        return "dense"
    if raw in _ON:
        return "int8"
    return "auto"


def grouped_packed_layers(model: Any) -> int:
    try:
        return int(getattr(model, GROUPED_PACKED_ATTR, 0) or 0)
    except (TypeError, ValueError):
        return 0


def grouped_call_counts() -> dict:
    """Engagement census of packed forwards by route (never read inside a traced region)."""
    return dict(_CALLS)


def _tuned_on(index: Any) -> bool:
    """Eager, once per card: Studio's int8 GEMM op is registered and passed its bit-exact probe there."""
    hit = _TUNED.get(index)
    if hit is None:
        try:
            from . import diffusion_int8_gemm as g
            hit = (
                g.int8_gemm_mode() != "off"
                and g._op() is not None
                and g.device_config(index) is not None
            )
        except Exception:  # noqa: BLE001 -- the native W8A8 maths
            hit = False
        _TUNED[index] = bool(hit)
    return bool(hit)


def _rotq_on(index: Any) -> bool:
    """Eager, once per card: arm the fused rotation + act quant kernel (its bit-exact probe; the verdict persists per
    card) so packed layers quantize their activations through it even when no int8 Linear swap ran on the load."""
    hit = _ROTQ.get(index)
    if hit is None:
        try:
            from .diffusion_convrot_quant import rotquant_device_config, rotquant_enabled
            hit = rotquant_enabled() and rotquant_device_config(int(index)) is not None
        except Exception:  # noqa: BLE001 -- the stock rotation + quant
            hit = False
        _ROTQ[index] = bool(hit)
    return bool(hit)


def _fused_on(index: Any) -> bool:
    """Eager, once per card: the fused unpack-in-mainloop GEMM passed its probe there (``diffusion_grouped_int8_gemm``)."""
    if index not in _FUSED:
        try:
            from .diffusion_grouped_int8_gemm import fused_config
            _FUSED[index] = fused_config(int(index))
        except Exception:  # noqa: BLE001 -- decode + the int8 GEMM
            _FUSED[index] = None
    return _FUSED[index] is not None


def race_winner(times: dict) -> str:
    """The race verdict: the faster incumbent (``int8`` / ``dense``) unless an added route (``fused``, ``int8r``) beats
    it by more than ``_RACE_MARGIN``; with no incumbent timed, the fastest route."""
    base = {r: t for r, t in times.items() if r in ("int8", "dense")}
    if not base:
        return min(times, key = times.get)
    best = min(base, key = base.get)
    added = {r: t for r, t in times.items() if r not in base}
    if added:
        new = min(added, key = added.get)
        if added[new] < base[best] * (1.0 - _RACE_MARGIN):
            return new
    return best


def decide_on_card(layer: Any, index: Any) -> None:
    """Eager, at load or placement: card ``index``'s tuned GEMM, fused rotation + act quant and fused unpack GEMM
    decisions for a packed layer (a compiled forward only reads them; an offloaded / streamed load may never move the
    layer with ``.to()``)."""
    if getattr(layer, "_tuned", None) is None:
        layer._tuned = _tuned_on(index)
    if getattr(layer, "_rotq", None) is None:
        layer._rotq = _rotq_on(index)
    if getattr(layer, "_fused_ok", None) is None:
        layer._fused_ok = _fused_on(index)


def _rows_bucket(rows: int) -> int:
    return 0 if rows < 256 else 1 if rows < 2048 else 2


_CLASS = None


def grouped_linear_class():
    """``GroupedInt8Linear``, defined on first use so this module imports torch-free."""
    global _CLASS
    if _CLASS is not None:
        return _CLASS
    import torch
    from torch import nn
    from torch.nn import functional as F

    from .diffusion_convrot import build_convrot_hadamard

    class GroupedInt8Linear(nn.Module):
        """A Linear whose W4A8 / W6A8 codes stay packed; all state is integer buffers (a module-wide ``.to(dtype)``
        never casts it) except the rotation, which follows the compute dtype like the activations."""

        def __init__(
            self,
            in_features: int,
            out_features: int,
            *,
            qdata: Any,
            s_rel: Any,
            s_channel: Any,
            codebook: Any = None,
            bits: int,
            group_size: int,
            rotation: int,
            bias: Any = None,
            compute_dtype: Any = None,
        ):
            super().__init__()
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.bits = int(bits)
            self.group_size = int(group_size)
            self.rot_group = int(rotation)
            if (
                self.in_features % self.rot_group
                or int(s_rel.shape[1]) * self.group_size != self.in_features
            ):
                raise ValueError(
                    f"packed layer K={self.in_features} does not match its group scales"
                )
            if (
                int(qdata.shape[1]) * 8 != self.in_features * self.bits
                or int(qdata.shape[0]) != self.out_features
            ):
                raise ValueError(
                    f"packed layer {tuple(qdata.shape)} is not {self.out_features} x {self.in_features}"
                )
            self.compute_dtype = compute_dtype or torch.bfloat16
            self.register_buffer("weight_packed", qdata.contiguous().view(torch.int8))
            self.register_buffer("weight_s_rel", s_rel.contiguous().view(torch.uint8))
            self.register_buffer(
                "weight_s_channel",
                s_channel.to(torch.float32).reshape(-1).contiguous().view(torch.int32),
            )
            self.register_buffer(
                "weight_codebook",
                None
                if codebook is None
                else codebook.to(torch.float32).reshape(-1).contiguous().view(torch.int32),
            )
            self.register_buffer("bias", bias)
            h = build_convrot_hadamard(self.rot_group, device = qdata.device, dtype = torch.float32)
            self.register_buffer("rot_h", h.to(self.compute_dtype, copy = True), persistent = False)
            self._route = None
            try:  # registered here, in eager, so a compiled forward only reads the handle
                from .diffusion_grouped_int8_triton import decode_int8_op
                self._decode_op = decode_int8_op()
            except Exception:  # noqa: BLE001 -- no custom ops on this torch: a trace runs the torch decode
                self._decode_op = None
            self._mode = grouped_gemm_mode()
            self._tuned = None
            self._rotq = None
            self._fused_ok = None

        def _apply(self, fn, *args, **kwargs):
            out = super()._apply(fn, *args, **kwargs)
            if self.weight_packed.is_cuda and None in (self._tuned, self._rotq, self._fused_ok):
                # placed on a card, in eager: decide now, since a compiled forward never probes
                decide_on_card(self, self.weight_packed.device.index)
            probe = fn(torch.empty(0, dtype = self.compute_dtype, device = self.weight_packed.device))
            if isinstance(probe, torch.Tensor) and probe.dtype in (
                torch.bfloat16,
                torch.float16,
                torch.float32,
            ):
                self.compute_dtype = probe.dtype
            return out

        def _codebook(self) -> Any:
            return (
                None if self.weight_codebook is None else self.weight_codebook.view(torch.float32)
            )

        def decoded_int8(self) -> Any:
            """The int8 grid ``[N, K]`` (rotated basis); a compiled graph sees one opaque op."""
            if torch.compiler.is_compiling():
                op = self._decode_op
                if op is not None:
                    return op(
                        self.weight_packed,
                        self.weight_s_rel,
                        self._codebook(),
                        self.group_size,
                        self.bits,
                    )
                return decode_int8_torch(
                    self.weight_packed,
                    self.weight_s_rel,
                    self._codebook(),
                    self.group_size,
                    self.bits,
                )
            return decode_int8(
                self.weight_packed, self.weight_s_rel, self._codebook(), self.group_size, self.bits
            )

        def dequantized_weight(self, dtype: Any = None) -> Any:
            """The dense weight in the original basis (exact float32 maths, then ``dtype``)."""
            dtype = dtype or self.compute_dtype
            w = (
                self.decoded_int8().to(torch.float32)
                * self.weight_s_channel.view(torch.float32)[:, None]
            )
            h = self.rot_h.to(torch.float32)
            g = self.rot_group
            return (
                (w.reshape(self.out_features, -1, g) @ h).reshape(self.out_features, -1).to(dtype)
            )

        @property
        def weight(self) -> Any:
            return self.dequantized_weight()

        def _rotate(self, x: Any) -> Any:
            g = self.rot_group
            grouped = x.reshape(*x.shape[:-1], self.in_features // g, g)
            return (grouped @ self.rot_h.to(x.dtype)).reshape(x.shape)

        def _dense(self, x: Any) -> Any:
            w = (
                self.decoded_int8().to(torch.float32)
                * self.weight_s_channel.view(torch.float32)[:, None]
            ).to(x.dtype)
            return F.linear(
                self._rotate(x), w, None if self.bias is None else self.bias.to(x.dtype)
            )

        def _tuned_int8(self, x2: Any, q8: Any) -> Any:
            """bf16: per-row int8 activations of ``rotate(x2)`` (Studio's fused rotate + quant kernel where it covers
            the shape) into Studio's int8 GEMM with the fused dequant epilogue (its probed tile for this card)."""
            from .diffusion_convrot_quant import convrot_act_quant

            q = convrot_act_quant(x2, self.rot_group, True, True)
            if q is not None:
                xq, xs = q
                xs = xs.reshape(-1)
            else:
                xr = self._rotate(x2)
                xs = torch.linalg.vector_norm(xr, ord = float("inf"), dim = 1, dtype = torch.float32)
                xs = xs.clamp(min = 1e-12) / 127.0
                xq = torch.div(xr, xs[:, None]).round_().clamp_(-127, 127).to(torch.int8)
            ws = self.weight_s_channel.view(torch.float32)
            bias = None if self.bias is None else self.bias.to(torch.bfloat16)
            return torch.ops.unsloth_studio.int8_mm_dequant(xq, q8, xs.to(torch.float32), ws, bias)

        def _tuned_ready(self, x: Any) -> bool:
            return _tuned_on(x.device.index)

        def _placed(self, x: Any) -> None:
            """Eager: the per-card decisions a layer built off the card (or before a move) has not made yet."""
            if self._tuned is None:
                self._tuned = self._tuned_ready(x)
            decide_on_card(self, x.device.index)

        def _fused(self, x: Any) -> Any:
            """bf16: the fused rotation + act quant codes into the unpack-in-mainloop GEMM (the packed codes are decoded
            in registers, never written as an int8 grid); bit-identical to decode + ``_tuned_int8`` / ``_int8r``."""
            from .diffusion_convrot_quant import convrot_act_quant
            from .diffusion_grouped_int8_gemm import grouped_mm_dequant

            x2 = x.reshape(-1, self.in_features)
            q = convrot_act_quant(x2, self.rot_group, True, True)
            if q is None:
                return self._int8(x)
            xq, xs = q
            out = grouped_mm_dequant(
                xq,
                self.weight_packed,
                self.weight_s_rel,
                self._codebook(),
                xs.reshape(-1).to(torch.float32),
                self.weight_s_channel.view(torch.float32),
                None if self.bias is None else self.bias.to(torch.bfloat16),
                self.bits,
                self.group_size,
            )
            if out is None:  # not covered here: decode + the int8 GEMM
                return self._int8(x)
            return out.reshape(*x.shape[:-1], self.out_features)

        def _int8r(self, x: Any) -> Any:
            """bf16 where the tuned GEMM is off (sm_8x on Triton >= 3.8): the fused rotation + act quant kernel, then
            ``torch._int_mm`` and torchao's epilogue (``diffusion_int8_gemm.reference``), rows chunked like ``_int8``;
            the same output as the tuned route."""
            from .diffusion_convrot_quant import convrot_act_quant
            from .diffusion_grouped_int8_triton import int8_epilogue_triton
            from .diffusion_int8_gemm import reference
            from .diffusion_native_quant import _INT_MM_CHUNK_ELEMS

            x2 = x.reshape(-1, self.in_features)
            q8 = self.decoded_int8()
            ws = self.weight_s_channel.view(torch.float32)
            bias = None if self.bias is None else self.bias.to(torch.bfloat16)
            rows = x2.shape[0]
            if (
                torch.compiler.is_compiling()
            ):  # one graph: the compiler bounds the temporaries, no row loop
                q = convrot_act_quant(x2, self.rot_group, True)
                if q is None:
                    return self._int8(x)
                xq, xs = q
                out = reference(xq, q8, xs.reshape(-1).to(torch.float32), ws, bias)
                return out.reshape(*x.shape[:-1], self.out_features)
            chunk = max(_MIN_ROWS, _INT_MM_CHUNK_ELEMS // max(self.in_features, self.out_features))
            out = x2.new_empty((rows, self.out_features))
            for start in range(0, rows, chunk):
                part = x2[
                    start : start + chunk
                ]  # a tail under _int_mm's row floor is padded by ``reference``
                q = convrot_act_quant(part, self.rot_group, True, True)
                if q is None:
                    return self._int8(x)
                xq, xs = q
                xs = xs.reshape(-1).to(torch.float32)
                got = None
                if part.shape[0] >= _MIN_ROWS:
                    # torch._int_mm, then the epilogue in one Triton pass (bit-identical to the eager reference)
                    got = int8_epilogue_triton(torch._int_mm(xq, q8.t()), xs, ws, bias)
                if got is None:
                    got = reference(xq, q8, xs, ws, bias)
                out[start : start + part.shape[0]] = got
            return out.reshape(*x.shape[:-1], self.out_features)

        def _fused_covers(self, x: Any) -> bool:
            """The fused unpack GEMM has a measured tile for this row count / bit depth on this card (it decodes each
            weight once per M tile, so it pays only at small batches; larger ones keep decode + the int8 GEMM)."""
            try:
                from .diffusion_grouped_int8_gemm import tile_for
                rows = x.numel() // max(1, self.in_features)
                return tile_for(_FUSED.get(x.device.index), rows, self.bits) is not None
            except Exception:  # noqa: BLE001
                return False

        def _routes(self, x: Any) -> tuple:
            """``(name, forward)`` candidates the race times here: the incumbents first, then the added routes."""
            routes = [("int8", self._int8), ("dense", self._dense)]
            if x.dtype == torch.bfloat16:
                if self._fused_ok and self._rotq and self._fused_covers(x):
                    routes.append(("fused", self._fused))
                if not self._tuned and self._rotq:
                    routes.append(("int8r", self._int8r))
            return tuple(routes)

        def _int8(self, x: Any) -> Any:
            from .diffusion_native_quant import (
                _INT_MM_CHUNK_ELEMS,
                _fused_w8a8,
                _fused_w8a8_failed,
                _w8a8_rows,
            )

            x2 = x.reshape(-1, self.in_features)
            q8 = self.decoded_int8()
            if x.dtype == torch.bfloat16 and self._tuned:
                return self._tuned_int8(x2, q8).reshape(*x.shape[:-1], self.out_features)
            bias = self.bias
            fn = None
            if not torch.compiler.is_compiling() and not torch.cuda.is_current_stream_capturing():
                from .diffusion_bg_compile import eager_forced
                fn = None if eager_forced() else _fused_w8a8(x.device)

            def run(part: Any) -> Any:
                nonlocal fn
                if fn is not None:
                    try:
                        return fn(part.detach(), self.rot_h, q8, self.weight_s_channel, bias)
                    except torch.cuda.OutOfMemoryError:
                        raise
                    except Exception as exc:  # noqa: BLE001 -- the eager maths below are the same function
                        _fused_w8a8_failed(exc, x.device)
                        fn = None
                return _w8a8_rows(part, self.rot_h, q8, self.weight_s_channel, bias)

            rows = x2.shape[0]
            # the int32 accumulator and float32 epilogue are [rows, N]: chunk rows as the native int8 Linear does
            chunk = _INT_MM_CHUNK_ELEMS // max(self.in_features, self.out_features)
            if torch.compiler.is_compiling() or rows <= chunk or chunk < 2 * _MIN_ROWS:
                out = run(x2)
            else:
                out = None
                start = 0
                for part in torch.tensor_split(x2, -(-rows // chunk)):
                    got = run(part)
                    if out is None:
                        out = got.new_empty((rows, self.out_features))
                    out[start : start + part.shape[0]].copy_(got)
                    start += part.shape[0]
                    del got
            return out.reshape(*x.shape[:-1], self.out_features)

        def _int8_ok(self, x: Any) -> bool:
            return (
                x.is_cuda
                and x.dtype in (torch.bfloat16, torch.float16)
                and x.numel() // max(1, self.in_features) >= _MIN_ROWS
                and self.in_features % 8 == 0
                and self.out_features % 8 == 0
                and not (torch.is_grad_enabled() and x.requires_grad)
            )

        def _race(self, x: Any) -> str:
            rows = x.numel() // max(1, self.in_features)
            key = (
                str(x.device),
                self.out_features,
                self.in_features,
                self.bits,
                _rows_bucket(rows),
                x.dtype,
            )
            hit = _RACE.get(key)
            if hit is not None:
                return hit
            if torch.cuda.is_current_stream_capturing():
                return "int8"
            with _RACE_LOCK:
                hit = _RACE.get(key)
                if hit is not None:
                    return hit
                times = {}
                self._placed(x)
                errors: dict = {}
                for route, run in self._routes(x):
                    try:
                        run(x)  # warm-up (and the compiled linear's first build)
                        start = torch.cuda.Event(enable_timing = True)
                        end = torch.cuda.Event(enable_timing = True)
                        start.record()
                        for _ in range(_RACE_ITERS):
                            run(x)
                        end.record()
                        end.synchronize()
                        times[route] = start.elapsed_time(end)
                    except Exception as exc:  # noqa: BLE001
                        errors[route] = exc
                if not times:
                    raise errors["dense"]
                hit = race_winner(times)
                _RACE[key] = hit
                try:
                    from loggers import get_logger
                    log = get_logger(__name__)
                except Exception:  # noqa: BLE001
                    import logging
                    log = logging.getLogger(__name__)
                log.info(
                    "diffusion.grouped_int8: %s x %s (%d-bit, rows bucket %d) on %s: %s -> %s",
                    self.out_features,
                    self.in_features,
                    self.bits,
                    key[4],
                    key[0],
                    ", ".join(f"{r} {t / _RACE_ITERS:.3f} ms" for r, t in times.items()),
                    hit,
                )
                return hit

        def forward(self, x: Any) -> Any:
            mode = self._mode
            if mode != "dense" and self._int8_ok(x):
                if mode == "int8":
                    route = "int8"
                elif torch.compiler.is_compiling():
                    # no eager race before a trace: rotation kernel codes into torch._int_mm
                    route = self._route or (
                        "int8r"
                        if not self._tuned and self._rotq and x.dtype == torch.bfloat16
                        else "int8"
                    )
                else:
                    route = self._race(x)
                    self._route = route
                if route != "dense":
                    eager = not torch.compiler.is_compiling()
                    if eager:
                        self._placed(x)
                    bf16 = x.dtype == torch.bfloat16
                    if route == "fused" and not (
                        bf16 and self._fused_ok and self._rotq and eager and self._fused_covers(x)
                    ):
                        # raced in eager only (a trace keeps the int8 route)
                        route = "int8"
                    if route == "int8r" and not (bf16 and self._rotq):
                        route = "int8"
                    if eager:
                        _CALLS[route] = _CALLS.get(route, 0) + 1
                    if route == "fused":
                        return self._fused(x)
                    if route == "int8r":
                        return self._int8r(x)
                    return self._int8(x)
            if not torch.compiler.is_compiling():
                _CALLS["dense"] += 1
            return self._dense(x)

        def extra_repr(self) -> str:  # pragma: no cover - debug aid
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, {self.bits}-bit packed, "
                f"group={self.group_size}, rotation={self.rot_group}, bias={self.bias is not None}"
            )

    _CLASS = GroupedInt8Linear
    return _CLASS


def is_grouped_linear(module: Any) -> bool:
    return type(module).__name__ == "GroupedInt8Linear" and hasattr(module, "weight_packed")


def _declared_layers(path: str) -> list:
    """Names of the quantized layers a reference file declares (its keep-list), header only."""
    from .diffusion_comfy_quant import scan_comfy_quant

    scan = scan_comfy_quant(path)
    if scan is None:
        raise ValueError(f"{path} declares no quantized layers")
    return sorted(scan.layers)


def export_grouped_file(
    src: str,
    out: str,
    fmt: str,
    *,
    layers: Any = None,
    like: Optional[str] = None,
    group_size: int = DEFAULT_GROUP,
    rotation: int = DEFAULT_ROTATION,
    codebook: Any = None,
    device: Any = "cpu",
    record: Optional[dict] = None,
    logger: Any = None,
) -> dict:
    """Write ``src`` (a floating safetensors single file, original key names) to ``out`` with the chosen Linears in
    ``fmt``: ``like`` takes the layer set a reference quantized file declares, ``layers`` an explicit list of layer
    names (no ``.weight``) or a predicate on them. Other tensors and unfit layers are copied unchanged.
    Returns ``{"quantized": n, "kept": [...], "bytes": size}``."""
    import torch
    from safetensors import safe_open

    check_grouped_config(fmt, group_size, rotation, codebook)
    bits = GROUPED_BITS[fmt]
    if like is not None:
        chosen = set(_declared_layers(like))
        pick = chosen.__contains__
    elif callable(layers):
        pick = layers
    elif layers is not None:
        pick = set(layers).__contains__
    else:
        raise ValueError("choose the layers: like=<reference file> or layers=<names or predicate>")
    conf = grouped_declaration(fmt, group_size, rotation)
    tensors: dict = {}
    skipped: list = []
    count = 0
    with safe_open(str(src), framework = "pt", device = "cpu") as handle:
        source_metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            value = handle.get_tensor(key)
            name = key[: -len(".weight")] if key.endswith(".weight") else None
            if name is None or not pick(name):
                tensors[key] = value
                continue
            k = int(value.shape[-1]) if value.dim() == 2 else 0
            if (
                value.dim() != 2
                or not value.is_floating_point()
                or k % rotation
                or k % group_size
                or (bits == 6 and k % 32)
            ):
                skipped.append(key)
                tensors[key] = value
                continue
            parts = quantize_grouped(
                value.to(device), fmt, group_size = group_size, rotation = rotation, codebook = codebook
            )
            for suffix, part in parts.items():
                tensors[name + (suffix or ".weight")] = part.cpu()
            tensors[name + ".comfy_quant"] = declaration_tensor(conf)
            count += 1
    if like is not None:
        missing = sorted(n for n in chosen if n + ".weight" not in tensors)
        if missing:
            raise ValueError(
                f"{src} lacks {len(missing)} layer(s) the reference declares (e.g. {missing[0]})"
            )
    metadata = dict(source_metadata)
    metadata.setdefault("format", "pt")
    if record:
        from .diffusion_comfy_quant import CONVERSION_RECORD_KEY
        metadata[CONVERSION_RECORD_KEY] = json.dumps(record, sort_keys = True)
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in tensors.items()}, str(out), metadata = metadata)
    size = os.path.getsize(str(out))
    if logger is not None:
        logger.info(
            "diffusion.grouped_int8: wrote %s (%d %s layers, %d left as stored, %.2f GiB)",
            out,
            count,
            fmt,
            len(skipped),
            size / 2**30,
        )
    return {"quantized": count, "kept": skipped, "bytes": size}
