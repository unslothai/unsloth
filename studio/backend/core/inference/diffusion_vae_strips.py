# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Row-strip decode for the FLUX.2 / FLUX.3 VAE (AutoencoderKLFlux2) and every AutoencoderKL-structured VAE (FLUX.1,
Kontext, Z-Image, HiDream, Lumina-2, Chroma, SDXL) on top of the fused GroupNorm kernels.

GroupNorm stats are taken once over the full map; norm, SiLU, padding and the 3x3 conv then run on bounded row
strips with a one-row halo, so no normalised activation exists at full size. Residuals are written in place (halving
blocks into the front half of their input). The nearest-2x upsample writes its rows straight into the padded strip
and its conv bias is carried into the next block. Same math as the fused decode. NVIDIA CUDA only; eager only.

Every decode on cards measured no slower (``strip_caps``: T4, A100, L4, RTX PRO 6000); elsewhere FLUX.2 takes only
the decodes the fused path would not fit, and the AutoencoderKL family none (the tile planner's whole-map strips). ``UNSLOTH_FLUX_VAE_STRIPS``: ``0`` off, ``1`` every decode on any NVIDIA card;
``UNSLOTH_VAE_KL_STRIPS`` the same for the AutoencoderKL family alone.

``full_map_decode`` runs the same strip decode where the VAE would otherwise tile (speed mode off, no Triton, ROCm,
MPS, CPU, a failed fused install, low free memory): the three Triton primitives fall back to torch ops, so every device
decodes with whole-map GroupNorm stats. Tiles cut from a large latent take their own stats and shift colour (FLUX.2
2048 px: 25.6 dB vs the untiled decode). It takes every KL VAE whose structure it reproduces (``full_map_ok``), not a
list of classes. ``UNSLOTH_FLUX_VAE_FULL_MAP=0`` keeps the tiles.
"""

from __future__ import annotations

import os
import threading
import weakref
from functools import lru_cache
from typing import Any, Optional

from . import diffusion_vae_fused as fused

STRIPS_ENV = "UNSLOTH_FLUX_VAE_STRIPS"
FULL_MAP_ENV = "UNSLOTH_FLUX_VAE_FULL_MAP"
# Upsample stage: Triton writes the nearest-2x rows into the halo-padded strip, cuDNN writes the conv straight into the
# output, and the conv bias is carried into the next block's GroupNorm and residual (no bias pass, no strip copy).
# ``0`` keeps the torch copy + conv-with-bias + copy.
STRIP_UPSAMPLE_ENV = "UNSLOTH_VAE_STRIP_UPSAMPLE"
# Strip decode for AutoencoderKL-structured VAEs (FLUX.1, Kontext, Z-Image, HiDream, Lumina-2, Chroma, SDXL) on the
# default path, matched by structure (``kl_strip_candidate``). ``1`` every decode, ``0`` off; unset: the measured cards.
KL_STRIPS_ENV = "UNSLOTH_VAE_KL_STRIPS"
STRIP_VAES = frozenset({"AutoencoderKLFlux2"})
# Elements of the larger of a strip's input and output (None: by card). sm80+ (L4 / A100) decode fastest at 2^24
# (32 MiB of bf16); Turing's implicit-GEMM convs lose 5-10% on strips that short, so sm75 takes 2^26.
STRIP_ELEMS: Optional[int] = None
_LOCAL = threading.local()


@lru_cache(maxsize = 16)
def _card_strip_elems(index: int) -> int:
    major, _ = fused._torch().cuda.get_device_capability(index)
    return 1 << (26 if major < 8 else 24)


def strip_elems(device: Any) -> int:
    if STRIP_ELEMS:
        return int(STRIP_ELEMS)
    torch = fused._torch()
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _card_strip_elems(index)


def strips_disabled() -> bool:
    return os.environ.get(STRIPS_ENV, "").strip().lower() in fused._FALSE


def full_map_disabled() -> bool:
    return os.environ.get(FULL_MAP_ENV, "").strip().lower() in fused._FALSE


def _portable() -> bool:
    return bool(getattr(_LOCAL, "portable", False))


def strips_forced() -> bool:
    return os.environ.get(STRIPS_ENV, "").strip().lower() in ("1", "true", "yes", "on")


# (major, minor) compute capabilities where every strip decode measured within A/A noise of the fused one or faster:
# with the upsample written into the strip and its bias carried (``CARRY_CAPS``: FLUX.1 / SDXL / FLUX.2 decoders at
# 1024 and 2048 px, 3.5-7.4% faster than the fused decode on T4, A100, L4 and RTX PRO 6000), else the older upsample
# (``ALWAYS_CAPS``: A100 3% slower). Other cards (H100, B200) take strips only for decodes the fused path would not fit.
ALWAYS_CAPS = frozenset({(7, 5), (8, 9)})
CARRY_CAPS = frozenset({(7, 5), (8, 0), (8, 9), (12, 0)})


def strip_caps() -> frozenset:
    return ALWAYS_CAPS if strip_upsample_disabled() else CARRY_CAPS


@lru_cache(maxsize = 16)
def _card_cap(index: int) -> tuple:
    return tuple(fused._torch().cuda.get_device_capability(index))


def _card_in_caps(device: Any) -> bool:
    torch = fused._torch()
    return (
        _card_cap(device.index if device.index is not None else torch.cuda.current_device())
        in strip_caps()
    )


def always_on(device: Any) -> bool:
    """Every decode on this card takes the strip path (else only the ones the fused decode would not fit)."""
    return strips_forced() or _card_in_caps(device)


def _kl_setting() -> Optional[bool]:
    v = os.environ.get(KL_STRIPS_ENV, "").strip().lower()
    if v in fused._FALSE:
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return None


def kl_always_on(device: Any) -> bool:
    """Every AutoencoderKL decode on this card takes the strip path (else none: a decode the fused path would not fit
    reaches the whole-map strips through the tile planner): ``UNSLOTH_VAE_KL_STRIPS``, else
    ``UNSLOTH_FLUX_VAE_STRIPS=1``, else the measured cards."""
    setting = _kl_setting()
    if setting is not None:
        return setting
    return always_on(device)


def kl_strip_candidate(vae: Any) -> bool:
    """A KL VAE outside ``STRIP_VAES`` whose decode is AutoencoderKL's (post_quant_conv, then ``decoder``) and whose
    strips are not switched off: the decoder itself is checked by ``decoder_ok``."""
    return (
        vae is not None
        and type(vae).__name__ not in STRIP_VAES
        and _kl_setting() is not False
        and _plain_decode_class(type(vae))
        and _post_quant_ok(vae)
    )


def fused_would_not_fit(vae: Any, z: Any) -> bool:
    """The tile planner's own rule: the fused decode of ``z`` needs more than its share of the free VRAM."""
    try:
        from .diffusion_vae_tiling import FREE_FRACTION, _free_mib, _geometry, decode_mib_per_latent

        free = _free_mib(vae, z)
        if free is None:
            return False
        mib, scale = free
        per_latent = decode_mib_per_latent(_geometry(vae)[0], True, type(vae).__name__) * scale
        return per_latent * int(z.shape[-2]) * int(z.shape[-1]) > FREE_FRACTION * mib
    except Exception:  # noqa: BLE001
        return False


@lru_cache(maxsize = 1)
def _residual_kernel() -> Any:
    if fused._kernels() is None:
        return None
    import triton
    import triton.language as tl

    # Triton 3.3 resolves a kernel's names in its module globals only (a closure ``tl`` fails to lower), as in fused._kernels.
    globals().update(triton = triton, tl = tl)

    @triton.jit(do_not_specialize = ("P", "W"))
    def _rows_residual(
        o_ptr, r_ptr, y_ptr, b_ptr, P, C, W, osh, osw, osc, rsh, rsw, rsc, ysh, ysw, ysc, inv_scale,
        HAS_B: tl.constexpr, SCALE: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr  # HAS_B bias present, SCALE output scale != 1, BLOCK_P pixels per program, BLOCK_C channels per program
    ):  # fmt: skip
        p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
        c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        cm = c < C
        m = (p < P)[:, None] & cm[None, :]
        hh = p // W
        ww = p - hh * W
        hh = hh.to(tl.int64)[:, None]
        ww = ww.to(tl.int64)[:, None]
        c64 = c.to(tl.int64)[None, :]
        o_off = hh * osh + ww * osw + c64 * osc
        v = tl.load(r_ptr + hh * rsh + ww * rsw + c64 * rsc, mask = m, other = 0.0).to(tl.float32)
        v = v + tl.load(y_ptr + hh * ysh + ww * ysw + c64 * ysc, mask = m, other = 0.0).to(tl.float32)
        if HAS_B:
            v = v + tl.load(b_ptr + c, mask = cm, other = 0.0).to(tl.float32)[None, :]
        if SCALE:
            v = v * inv_scale
        tl.store(o_ptr + o_off, v.to(o_ptr.dtype.element_ty), mask = m)

    return _rows_residual


def _conv3x3_ok(conv: Any) -> bool:
    torch = fused._torch()
    return (
        isinstance(conv, torch.nn.Conv2d)
        and tuple(conv.kernel_size) == (3, 3)
        and tuple(conv.stride) == (1, 1)
        and tuple(conv.padding) == (1, 1)
        and tuple(conv.dilation) == (1, 1)
        and conv.groups == 1
        and getattr(conv, "padding_mode", "zeros") == "zeros"
    )


def _norm_ok(norm: Any, kernels: bool = True) -> bool:
    """An affine GroupNorm; ``kernels`` adds the Triton primitives' limits (power-of-two channels and groups, at most
    1024 channels). The torch fallback of ``full_map_decode`` takes any group count that divides the channels (MOSS-Image's
    96 / 192 / 384-channel decoder)."""
    torch = fused._torch()
    if not isinstance(norm, torch.nn.GroupNorm) or not norm.affine or norm.weight is None:
        return False
    c, g = int(norm.num_channels), int(norm.num_groups)
    if not (0 < g and c % g == 0):
        return False
    return not kernels or (c & (c - 1) == 0 and g & (g - 1) == 0 and c <= 1024)


def _resnet_ok(block: Any, kernels: bool = True) -> bool:
    torch = fused._torch()
    if not (
        type(block).__name__ == "ResnetBlock2D"
        and fused._resnet2d_fusable(block)
        and _norm_ok(block.norm1, kernels)
        and _norm_ok(block.norm2, kernels)
        and _conv3x3_ok(block.conv1)
        and _conv3x3_ok(block.conv2)
        and block.conv1.out_channels == block.conv2.in_channels
        and float(getattr(getattr(block, "dropout", None), "p", 0.0) or 0.0) == 0.0
    ):
        return False
    sc = block.conv_shortcut
    if sc is None:
        return block.conv1.in_channels == block.conv2.out_channels
    return (
        isinstance(sc, torch.nn.Conv2d)
        and tuple(sc.kernel_size) == (1, 1)
        and tuple(sc.stride) == (1, 1)
        and tuple(sc.padding) == (0, 0)
        and sc.groups == 1
    )


def _upsampler_ok(up: Any) -> bool:
    return (
        type(up).__name__ == "Upsample2D"
        and getattr(up, "use_conv", False)
        and not getattr(up, "use_conv_transpose", False)
        and getattr(up, "interpolate", True)
        and getattr(up, "norm", None) is None
        and getattr(up, "name", "conv") == "conv"
        and _conv3x3_ok(getattr(up, "conv", None))
    )


def decoder_ok(decoder: Any, kernels: bool = True) -> bool:
    """Whether every stage after the mid block is a shape the strip decode reproduces exactly (``kernels``: on the Triton
    primitives; False: on their torch fallbacks)."""
    torch = fused._torch()
    ups = getattr(decoder, "up_blocks", None)
    if (
        ups is None
        or not len(ups)
        or not isinstance(getattr(decoder, "conv_in", None), torch.nn.Conv2d)
        or not _norm_ok(getattr(decoder, "conv_norm_out", None), kernels)
        or not isinstance(getattr(decoder, "conv_act", None), torch.nn.SiLU)
        or not _conv3x3_ok(getattr(decoder, "conv_out", None))
    ):
        return False
    for up in ups:
        if type(up).__name__ != "UpDecoderBlock2D" or not all(
            _resnet_ok(r, kernels) for r in up.resnets
        ):
            return False
        ups_ = getattr(up, "upsamplers", None)
        if ups_ is not None and (len(ups_) != 1 or not _upsampler_ok(ups_[0])):
            return False
    return not any(hasattr(m, "_hf_hook") for m in decoder.modules())


def _steps(rows: int, elems: int) -> int:
    strip = STRIP_ELEMS or getattr(_LOCAL, "strip", None) or 1 << 24
    pieces = max(1, -(-int(elems) // strip))
    return max(1, -(-int(rows) // pieces))


def _row_cl(x: Any) -> bool:
    """B=1, channels innermost, pixels of a row adjacent: every row strip is a dense (rows, W, C) block."""
    _, c, _, w = x.shape
    return x.shape[0] == 1 and x.stride(1) == 1 and x.stride(3) == c and x.stride(2) >= w * c


def _owned_cl(x: Any) -> Any:
    torch = fused._torch()
    return x if _row_cl(x) else x.contiguous(memory_format = torch.channels_last)


def _empty_cl(c: int, h: int, w: int, like: Any) -> Any:
    torch = fused._torch()
    return torch.empty((1, h, w, c), dtype = like.dtype, device = like.device).permute(0, 3, 1, 2)


def _stats(
    x: Any,
    norm: Any,
    in_bias: Any = None,
) -> tuple:
    if _portable():
        return _stats_torch(x, norm, in_bias)
    return fused.group_norm_stats(x.unsqueeze(2), norm, in_bias)


def _group_rows(x: Any, a: int, b: int, g: int, in_bias: Any) -> Any:
    """fp32 rows [a, b) of ``x + in_bias`` as (G, C/G, rows, W)."""
    seg = x.narrow(2, a, b - a).float()
    if in_bias is not None:
        seg = seg + in_bias.float().view(1, -1, 1, 1)
    _, c, n, w = seg.shape
    return seg.reshape(g, c // g, n, w)


def _stats_torch(
    x: Any,
    norm: Any,
    in_bias: Any = None,
) -> tuple:
    """fp32 per-group ``(mean, rstd)`` over the whole map in row strips; fp32 on device (no fp64 on XPU/MPS, no host sync)."""
    torch = fused._torch()
    _, c, h, w = x.shape
    g = int(norm.num_groups)
    count = float(c // g) * h * w
    total = torch.zeros(g, dtype = torch.float32, device = x.device)
    for a, b in _strips(h, x.numel()):
        total += _group_rows(x, a, b, g, in_bias).sum(dim = (1, 2, 3))
    mean = total / count
    sq = torch.zeros_like(total)
    for a, b in _strips(h, x.numel()):
        d = _group_rows(x, a, b, g, in_bias) - mean.view(g, 1, 1, 1)
        sq += (d * d).sum(dim = (1, 2, 3))
    rstd = torch.rsqrt(sq / count + float(norm.eps))
    return mean, rstd


def _padded_rows(c: int, rows: int, top: int, bottom: int, w: int, like: Any) -> tuple:
    """(1, rows + top + bottom, W, C) buffer with zeroed edge rows, and its interior."""
    torch = fused._torch()
    buf = torch.empty((1, rows + top + bottom, w, c), dtype = like.dtype, device = like.device)
    if top:
        buf[:, 0].zero_()
    if bottom:
        buf[:, -1].zero_()
    return buf, buf[:, top : top + rows]


def _norm_rows(
    x: Any,
    a: int,
    b: int,
    norm: Any,
    stats: tuple,
    in_bias: Any = None,
) -> Any:
    """silu(group_norm(x + in_bias)) of rows [a, b) plus a one-row halo (zeros past the image edge), with the
    global ``stats``: what a padding-1 3x3 conv reads for output rows [a, b)."""
    if _portable():
        return _norm_rows_torch(x, a, b, norm, stats, in_bias)
    k = fused._kernels()
    _, c, h, w = x.shape
    lo, hi = max(0, a - 1), min(h, b + 1)
    top, bottom = int(a == 0), int(b == h)
    buf, inner = _padded_rows(c, hi - lo, top, bottom, w, x)
    x5 = x.narrow(2, lo, hi - lo).unsqueeze(2)
    sb, sc, st, sh, sw = x5.stride()
    hw = (hi - lo) * w
    g = int(norm.num_groups)
    block_c = min(128, fused._next_pow2(c))
    block_p = max(1, 4096 // block_c)
    grid = ((hw + block_p - 1) // block_p, 1, (c + block_c - 1) // block_c)
    with fused._on_card(x):
        k.gn_apply[grid](
            x5, inner, stats[0], stats[1], norm.weight, norm.bias, in_bias if in_bias is not None else x5,
            c, 1, hw, w, g, sb, sc, st, sh, sw,
            CPG = c // g, ACT = True, HAS_IN_BIAS = in_bias is not None, BLOCK_P = block_p, BLOCK_C = block_c,
            num_warps = 4,
        )  # fmt: skip
    return buf.permute(0, 3, 1, 2)


def _norm_rows_torch(
    x: Any,
    a: int,
    b: int,
    norm: Any,
    stats: tuple,
    in_bias: Any = None,
) -> Any:
    """``_norm_rows`` in torch ops: fp32 normalise, affine and SiLU, one rounding to the map's dtype."""
    import torch.nn.functional as F

    _, c, h, w = x.shape
    lo, hi = max(0, a - 1), min(h, b + 1)
    g = int(norm.num_groups)
    y = (_group_rows(x, lo, hi, g, in_bias) - stats[0].view(g, 1, 1, 1)) * stats[1].view(g, 1, 1, 1)
    y = y.reshape(1, c, hi - lo, w) * norm.weight.float().view(
        1, -1, 1, 1
    ) + norm.bias.float().view(1, -1, 1, 1)
    y = F.silu(y).to(x.dtype)
    return F.pad(y, (0, 0, int(a == 0), int(b == h)))


def _conv_rows(padded: Any, conv: Any, bias: bool) -> Any:
    """The 3x3 conv of a row-padded strip: rows are already padded, columns pad here."""
    import torch.nn.functional as F
    return F.conv2d(padded, conv.weight, conv.bias if bias else None, 1, (0, 1))


def _dense_cl(t: Any) -> bool:
    torch = fused._torch()
    return t.shape[0] == 1 and t.is_contiguous(memory_format = torch.channels_last)


def _conv_to(padded: Any, conv: Any, dst: Any) -> None:
    """dst = conv_nobias(padded): cuDNN writes a dense channels-last ``dst`` directly (no strip copy)."""
    torch = fused._torch()
    cudnn = torch.backends.cudnn
    if _CUDNN_OUT and _dense_cl(dst) and _dense_cl(padded) and cudnn.enabled and padded.is_cuda:
        try:
            torch.ops.aten.cudnn_convolution.out(
                padded, conv.weight, [0, 1], [1, 1], [1, 1], 1,
                cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32, out = dst,
            )  # fmt: skip
            return
        except (RuntimeError, AttributeError, TypeError, NotImplementedError):
            pass
    dst.copy_(_conv_rows(padded, conv, False))


_CUDNN_OUT = True


def _add_rows(
    dst: Any,
    y: Any,
    bias: Any,
    scale: float,
    res: Any = None,
) -> None:
    """dst = (res + y + bias) / scale (``res`` defaults to ``dst``), fp32 sum, one rounding; row strips."""
    if _portable():
        v = (dst if res is None else res).float() + y.float()
        if bias is not None:
            v = v + bias.float().view(1, -1, 1, 1)
        if float(scale) != 1.0:
            v = v * (1.0 / float(scale))
        dst.copy_(v)
        return
    k = _residual_kernel()
    res = dst if res is None else res
    _, c, n, w = dst.shape
    pixels = n * w
    block_c = min(128, fused._next_pow2(c))
    block_p = max(1, 4096 // block_c)
    grid = ((pixels + block_p - 1) // block_p, (c + block_c - 1) // block_c)
    with fused._on_card(dst):
        k[grid](
            dst, res, y, bias if bias is not None else dst, pixels, c, w,
            dst.stride(2), dst.stride(3), dst.stride(1), res.stride(2), res.stride(3), res.stride(1),
            y.stride(2), y.stride(3), y.stride(1), 1.0 / float(scale),
            HAS_B = bias is not None, SCALE = float(scale) != 1.0, BLOCK_P = block_p, BLOCK_C = block_c, num_warps = 4,
        )  # fmt: skip


def _strips(h: int, elems: int):
    step = _steps(h, elems)
    for a in range(0, h, step):
        yield a, min(h, a + step)


def _conv_into(
    x: Any,
    norm: Any,
    conv: Any,
    dst: Any,
    in_bias: Any = None,
) -> None:
    """dst = conv_nobias(silu(group_norm(x + in_bias))) strip by strip."""
    stats = _stats(x, norm, in_bias)
    h = x.shape[2]
    for a, b in _strips(h, max(x.numel(), dst.numel())):
        _conv_to(_norm_rows(x, a, b, norm, stats, in_bias), conv, dst.narrow(2, a, b - a))


def _conv_add(
    x: Any,
    norm: Any,
    conv: Any,
    dst: Any,
    scale: float,
    in_bias: Any = None,
    shortcut: Any = None,
    res_bias: Any = None,
) -> None:
    """dst = (res + res_bias + conv(silu(group_norm(x + in_bias)))) / scale strip by strip (conv bias included);
    ``res`` is ``dst`` itself, or ``shortcut`` (a 1x1 conv) of that strip's rows of its input. ``res_bias`` (fp32) is a
    bias the residual still owes (an upsample conv's, deferred); with a shortcut it is already the shortcut's total
    bias, so the 1x1 runs bias-free."""
    import torch.nn.functional as F

    stats = _stats(x, norm, in_bias)
    h = x.shape[2]
    bias = conv.bias if res_bias is None else _fold_bias(conv.bias, res_bias)
    for a, b in _strips(h, max(x.numel(), dst.numel())):
        y = _conv_rows(_norm_rows(x, a, b, norm, stats, in_bias), conv, False)
        res = None
        if shortcut is not None:
            sc, src = shortcut
            res = F.conv2d(
                src.narrow(2, a, b - a), sc.weight, sc.bias if res_bias is None else None
            )
        _add_rows(dst.narrow(2, a, b - a), y, bias, scale, res)


def _halve(
    block: Any,
    x: Any,
    hidden: Optional[Any],
    pending: Any = None,
) -> tuple:
    """A 2C -> C block whose output takes the front half of ``x``'s storage. Output rows [a, b) overwrite input rows
    [a / 2, b / 2), which only strips before b read (the shortcut is 1x1 and conv1 has finished)."""
    c = block.conv1.out_channels
    _, _, h, w = x.shape
    if hidden is None or tuple(hidden.shape) != (1, c, h, w) or not _dense_cl(hidden):
        hidden = _empty_cl(c, h, w, x)
    _conv_into(x, block.norm1, block.conv1, hidden, pending)
    out = x.as_strided((1, c, h, w), (h * w * c, 1, w * c, c), x.storage_offset())
    _conv_add(
        hidden, block.norm2, block.conv2, out, block.output_scale_factor, block.conv1.bias,
        shortcut = (block.conv_shortcut, x),
        res_bias = None if pending is None else _shortcut_bias(block.conv_shortcut, pending),
    )  # fmt: skip
    return out, hidden


def _resnet(
    block: Any,
    x: Any,
    hidden: Optional[Any],
    pending: Any = None,
) -> tuple:
    """(output, hidden buffer for the level's next block). ``pending``: a bias ``x`` still owes (the upsample conv's),
    folded into norm1's input and the residual."""
    import torch.nn.functional as F

    cin, cout = block.conv1.in_channels, block.conv1.out_channels
    _, _, h, w = x.shape
    if hidden is None or tuple(hidden.shape) != (1, cout, h, w) or not _dense_cl(hidden):
        hidden = _empty_cl(cout, h, w, x)
    if cin == cout and block.conv_shortcut is None:
        x = _owned_cl(x)
        _conv_into(x, block.norm1, block.conv1, hidden, pending)
        _conv_add(
            hidden, block.norm2, block.conv2, x, block.output_scale_factor, block.conv1.bias,
            res_bias = None if pending is None else pending.float(),
        )  # fmt: skip
        return x, hidden
    if cin == 2 * cout and _dense_cl(x):
        return _halve(block, x, hidden, pending)
    sc = block.conv_shortcut
    sc_bias = sc.bias if pending is None else _shortcut_bias(sc, pending).to(sc.weight.dtype)
    out = _owned_cl(F.conv2d(x, sc.weight, sc_bias))
    _conv_into(x, block.norm1, block.conv1, hidden, pending)
    _conv_add(hidden, block.norm2, block.conv2, out, block.output_scale_factor, block.conv1.bias)
    return out, hidden


def strip_upsample_disabled() -> bool:
    return os.environ.get(STRIP_UPSAMPLE_ENV, "").strip().lower() in fused._FALSE


def _up_rows(
    x: Any,
    lo: int,
    n: int,
    inner: Any,
    kernel: bool = True,
) -> None:
    """``inner`` (dense (1, 2n, 2W, C)) = nearest_2x(rows [lo, lo + n) of ``x``)."""
    if not kernel or _portable() or not x.is_cuda or fused._kernels() is None:
        _, c, _, w = x.shape
        src = x.narrow(2, lo, n).permute(0, 2, 3, 1)
        inner.view(1, n, 2, w, 2, c).copy_(src[:, :, None, :, None, :].expand(1, n, 2, w, 2, c))
        return
    k = fused._kernels()
    _, c, _, w = x.shape
    x5 = x.narrow(2, lo, n).unsqueeze(2)
    block_c = min(128, fused._next_pow2(c))
    block_p = max(1, 4096 // block_c)
    grid = ((4 * n * w + block_p - 1) // block_p, 1, (c + block_c - 1) // block_c)
    with fused._on_card(x):
        k.up_nearest2x[grid](
            x5, inner, x5, c, 1, n, w, *x5.stride(), INTERLEAVE = False, BLOCK_P = block_p, BLOCK_C = block_c,
            HAS_BIAS = False, num_warps = 4,
        )  # fmt: skip


# Nearest 2x + conv3x3(p1) as one ConvTranspose2d(k4, s2, p1) whose taps are the 3x3 taps summed in fp32 (rounded once
# to the conv dtype): 2.25x fewer MACs and the 2x map is never written. Rounding-level only (1024 px decode vs the 2x
# path: LPIPS 7e-5 bf16, 1e-6 fp16 T4; PSNR vs fp32 -0.04 to -0.28 dB). Measured on the FLUX.1 / Z-Image / FLUX.2 VAEs
# (strip decode, cuDNN 9.24), 1024 / 2048 px: T4 fp16 1.11-1.14x, A100 1.06-1.12x, L4 1.05-1.11x, RTX PRO 6000
# 1.07-1.13x per decode. ``UNSLOTH_VAE_UPSAMPLE_FOLD``: ``0`` off, ``1`` on any device (CUDA, ROCm, MPS, CPU); unset:
# the measured CUDA cards only (MIOpen / MPS transposed convs unmeasured).
UPSAMPLE_FOLD_ENV = "UNSLOTH_VAE_UPSAMPLE_FOLD"
_FOLD_CAPS = frozenset({(7, 5), (8, 0), (8, 9), (12, 0)})
# Output elements up to which one transposed conv covers the whole map (the measured shape: FLUX.1 1024 px's last
# upsample is 2^28); larger maps fold strip by strip (bounded workspace, one copy into the output).
FOLD_FULL_MAX_ELEMS = 1 << 28
_FOLD_TAPS = {0: (2,), 1: (1, 2), 2: (0, 1), 3: (0,)}
# conv -> ((data_ptr, dtype, device, shape, version), folded): dies with the conv, and a ``.data`` swap (Studio's fp32
# retry, module.to) changes the key, which weight identity + version alone miss.
_FOLD_CACHE: Any = weakref.WeakKeyDictionary()
_FOLD_CALLS = [0]


def _fold_mode() -> str:
    raw = os.environ.get(UPSAMPLE_FOLD_ENV, "").strip().lower()
    if raw in fused._FALSE:
        return "off"
    return "force" if raw in ("1", "on", "true", "yes", "force") else "auto"


def _fold_on(x: Any) -> bool:
    mode = _fold_mode()
    if mode == "off":
        return False
    if mode == "force":
        return True
    torch = fused._torch()
    if not x.is_cuda or getattr(torch.version, "hip", None):
        return False
    try:
        return tuple(torch.cuda.get_device_capability(x.device)) in _FOLD_CAPS
    except Exception:  # noqa: BLE001
        return False


def folded_upsample_weight(conv: Any) -> Any:
    """The ConvTranspose2d(k4, s2, p1) kernel equal to nearest_2x then ``conv`` (3x3, p1): taps summed in fp32, one
    rounding to the conv dtype, channels-last like the conv. Cached per conv while its weight tensor and version hold."""
    torch = fused._torch()
    w = conv.weight
    key = (w.data_ptr(), w.dtype, w.device, tuple(w.shape), w._version)
    hit = _FOLD_CACHE.get(conv)
    if hit is not None and hit[0] == key:
        return hit[1]
    wf = w.detach().float()
    cout, cin = wf.shape[:2]
    wt = wf.new_zeros(cin, cout, 4, 4)
    for a in range(4):
        for b in range(4):
            acc = wf.new_zeros(cout, cin)
            for i in _FOLD_TAPS[a]:
                for j in _FOLD_TAPS[b]:
                    acc += wf[:, :, i, j]
            wt[:, :, a, b] = acc.t()
    wt = wt.to(w.dtype).contiguous(memory_format = torch.channels_last)
    _FOLD_CACHE[conv] = (key, wt)
    return wt


def _fold_ok(conv: Any, x: Any) -> bool:
    return (
        tuple(conv.kernel_size) == (3, 3)
        and tuple(conv.stride) == (1, 1)
        and tuple(conv.padding) == (1, 1)
        and tuple(conv.dilation) == (1, 1)
        and conv.groups == 1
        and conv.padding_mode == "zeros"
        and x.dim() == 4
        and x.dtype == conv.weight.dtype
    )


def _fold_upsample(
    conv: Any,
    x: Any,
    bias: Any,
    full_max: Optional[int] = None,
) -> Any:
    """conv(nearest_2x(x)) (+ ``bias``) via the folded transposed conv: one call over the map up to ``full_max`` output
    elements, else per strip of input rows with a one-row halo (rows [a - 1, b + 1) give output rows [2a, 2b))."""
    F = fused._torch().nn.functional
    wt = folded_upsample_weight(conv)
    _FOLD_CALLS[0] += 1
    _, _, h, w = x.shape
    elems = conv.out_channels * 4 * h * w
    if elems <= (FOLD_FULL_MAX_ELEMS if full_max is None else full_max):
        return _owned_cl(F.conv_transpose2d(x, wt, bias, stride = 2, padding = 1))
    out = _empty_cl(conv.out_channels, 2 * h, 2 * w, x)
    for a, b in _strips(h, elems):
        lo, hi = max(0, a - 1), min(h, b + 1)
        # padding 0 on rows: strip output row o' is full output row 2 lo + o' - 1
        ys = F.conv_transpose2d(x.narrow(2, lo, hi - lo), wt, bias, stride = 2, padding = (0, 1))
        first = 2 * (a - lo) + 1
        out.narrow(2, 2 * a, 2 * (b - a)).copy_(ys.narrow(2, first, 2 * (b - a)))
    return out


def _upsample(
    up: Any,
    x: Any,
    defer_bias: bool = False,
) -> tuple:
    """(conv(nearest_2x(x)) in strips of input rows, the conv bias still to add). The 2x rows are written straight
    into the padded strip; ``defer_bias``: cuDNN writes the bias-free conv into the output and the bias is returned
    for the next stage to fold in, else the bias is added here. On the fold's cards one transposed conv replaces both
    (``UPSAMPLE_FOLD_ENV``)."""
    conv = up.conv
    if _fold_on(x) and _fold_ok(conv, x):
        out = _fold_upsample(conv, x, None if defer_bias else conv.bias)
        return out, (conv.bias if defer_bias else None)
    _, c, h, w = x.shape
    out = _empty_cl(conv.out_channels, 2 * h, 2 * w, x)
    for a, b in _strips(h, out.numel()):
        lo, hi = max(0, a - 1), min(h, b + 1)
        n = hi - lo
        top, bottom = int(a == 0), int(b == h)
        buf, inner = _padded_rows(c, 2 * n, top, bottom, 2 * w, x)
        _up_rows(x, lo, n, inner, defer_bias)
        # output rows [2a, 2b) read 2x rows [2a - 1, 2b + 1)
        padded = buf.permute(0, 3, 1, 2).narrow(2, 2 * (a - lo) + top - 1, 2 * (b - a) + 2)
        dst = out.narrow(2, 2 * a, 2 * (b - a))
        if defer_bias:
            _conv_to(padded, conv, dst)
        else:
            dst.copy_(_conv_rows(padded, conv, True))
    return out, (conv.bias if defer_bias else None)


def _fold_bias(*parts: Any) -> Any:
    out = None
    for p in parts:
        if p is not None:
            out = p.float() if out is None else out + p.float()
    return out


def _shortcut_bias(sc: Any, pending: Any) -> Any:
    """Bias of the 1x1 ``sc`` applied to ``x + pending``: sc.bias + W @ pending, fp32."""
    fold = (
        (sc.weight.reshape(sc.out_channels, -1).float() @ pending.float())
        if pending is not None
        else None
    )
    return _fold_bias(sc.bias, fold)


def _decode_one(decoder: Any, z: Any) -> Any:
    _LOCAL.strip = strip_elems(z.device) if z.is_cuda else (STRIP_ELEMS or 1 << 24)
    h = decoder.conv_in(z)
    if decoder.mid_block is not None:
        h = decoder.mid_block(h, None)
    h = h.to(decoder.up_blocks[0].resnets[0].conv1.weight.dtype)
    defer = not strip_upsample_disabled()
    pending = None  # a bias ``h`` still owes: the last upsample conv's
    for up in decoder.up_blocks:
        h, hidden = _owned_cl(h), None
        for block in up.resnets:
            h, hidden = _resnet(block, h, hidden, pending)
            pending = None
        del hidden
        if up.upsamplers is not None:
            if pending is not None:
                h = _owned_cl(h + pending.to(h.dtype).view(1, -1, 1, 1))
                pending = None
            h, pending = _upsample(up.upsamplers[0], h, defer)
    conv = decoder.conv_out
    _, _, hh, ww = h.shape
    out = _empty_cl(conv.out_channels, hh, ww, h)
    stats = _stats(h, decoder.conv_norm_out, pending)
    for a, b in _strips(hh, h.numel()):
        out.narrow(2, a, b - a).copy_(
            _conv_rows(_norm_rows(h, a, b, decoder.conv_norm_out, stats, pending), conv, True)
        )
    return out


def _kernels_ok(decoder: Any, z: Any) -> bool:
    return (
        bool(z.is_cuda)
        and decoder_ok(decoder)
        and fused.runtime_ok()
        and _residual_kernel() is not None
        and _call_ok(decoder, z)
    )


def _decode_ast(fn: Any) -> Optional[str]:
    import ast
    import inspect
    import textwrap

    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except Exception:  # noqa: BLE001 - no source: not comparable
        return None
    node = tree.body[0] if tree.body else None
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    body = node.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
    ):
        body = body[1:]  # the docstring
    return ast.dump(ast.Module(body = body, type_ignores = []), annotate_fields = False)


@lru_cache(maxsize = 32)
def _plain_decode_class(cls: type) -> bool:
    """Whether ``cls._decode`` is AutoencoderKL's, statement for statement (the tiling branch, the optional
    post_quant_conv, then ``self.decoder``): diffusers copies that body into every KL VAE built on its Decoder."""
    try:
        from diffusers import AutoencoderKL
    except Exception:  # noqa: BLE001
        return False
    mine = getattr(cls, "_decode", None)
    if mine is None:
        return False
    if mine is AutoencoderKL._decode:
        return True
    reference = _decode_ast(AutoencoderKL._decode)
    return reference is not None and _decode_ast(mine) == reference


def _post_quant_ok(vae: Any) -> bool:
    torch = fused._torch()
    conv = getattr(vae, "post_quant_conv", None)
    return conv is None or (
        isinstance(conv, torch.nn.Conv2d) and tuple(conv.kernel_size) == (1, 1) and conv.groups == 1
    )


def full_map_ok(vae: Any) -> bool:
    """Whether ``full_map_decode`` reproduces this VAE's untiled decode: read off its structure, not its class, so
    every KL VAE built on diffusers' Decoder qualifies (FLUX.1, FLUX.2 / FLUX.3, Z-Image, HiDream, Lumina-2, Chroma,
    SDXL, Ideogram-4, MOSS-Image, and a new family on the same pieces): a decode that is post_quant_conv then the
    decoder, and a decoder whose every stage the strips reproduce (``decoder_ok``)."""
    decoder = getattr(vae, "decoder", None)
    return (
        vae is not None
        and decoder is not None
        and not full_map_disabled()
        and _plain_decode_class(type(vae))
        and _post_quant_ok(vae)
        and decoder_ok(decoder, kernels = False)
    )


def _full_map_run(decoder: Any, z: Any, portable: bool) -> Any:
    torch = fused._torch()
    prev = getattr(_LOCAL, "portable", False)
    _LOCAL.portable = portable
    try:
        # The mid block attends over every latent: where no fused SDPA takes head dim 512 (ROCm, MPS), Studio's
        # chunked mode bounds the score matrix, else the math path holds all of it (65,536 tokens at 2048 px: 8 GiB per bf16 copy).
        with torch.no_grad(), _attention_bound():
            out = None
            for i in range(z.shape[0]):
                one = _decode_one(decoder, z[i : i + 1])
                if z.shape[0] == 1:
                    return one
                if out is None:
                    out = torch.empty(
                        (z.shape[0], *one.shape[1:]), dtype = one.dtype, device = one.device
                    )
                out[i : i + 1].copy_(one)
                del one
            return out
    finally:
        _LOCAL.portable = prev


def full_map_decode(vae: Any, z: Any) -> Any:
    """Untiled decode of a (B, C, H, W) latent with whole-map GroupNorm, in bounded row strips."""
    decoder = vae.decoder
    dtype = decoder.conv_in.weight.dtype
    if getattr(vae, "post_quant_conv", None) is not None:
        z = vae.post_quant_conv(z.to(vae.post_quant_conv.weight.dtype))
    z = z.to(dtype)
    # A strip decode that already failed on this decoder (same kernels) or fails here: the torch ops, not the tiles.
    kernels = _kernels_ok(decoder, z) and not decoder.__dict__.get(
        "_unsloth_vae_strips_failed", False
    )
    try:
        out = _full_map_run(decoder, z, portable = not kernels)
    except Exception as exc:  # noqa: BLE001
        if not kernels or fused._is_oom(exc):
            raise
        decoder._unsloth_vae_strips_failed = True
        out = None
    if out is None:
        out = _full_map_run(decoder, z, portable = True)
    vae._unsloth_vae_full_map_calls = vae.__dict__.get("_unsloth_vae_full_map_calls", 0) + 1
    return out


def _attention_bound() -> Any:
    """Studio's chunked VAE attention mode (exact; engages only on SDPA calls no fused kernel takes)."""
    import contextlib

    try:
        from . import diffusion_vae_attn_chunked as chunked
    except Exception:  # noqa: BLE001
        return contextlib.nullcontext()
    if chunked.chunking_disabled():
        return contextlib.nullcontext()
    return chunked.chunked_attention_mode(chunked.score_budget_bytes(_score_budget_mb()))


def _score_budget_mb() -> int:
    try:
        from . import diffusion_vae_attn_chunked as chunked
        return chunked.DEFAULT_SCORE_BUDGET_MB
    except Exception:  # noqa: BLE001
        return 1024


def full_map_decode_mib(vae: Any, z: Any) -> float:
    """Peak MiB of ``full_map_decode``: strip slope over the frame + fp32 strip temporaries + chunked attention scores."""
    slope = DECODE_STRIPS_MIB_PER_LATENT
    elem = vae.decoder.conv_in.weight.element_size()
    strip = (strip_elems(z.device) if z.is_cuda else (STRIP_ELEMS or 1 << 24)) * 4
    try:
        from . import diffusion_vae_attn_chunked as chunked
        scores = chunked.score_budget_bytes(_score_budget_mb()) / 2**20
    except Exception:  # noqa: BLE001
        scores = 1024.0
    # the slope was measured on FLUX.2's decoder; another decoder scales with its largest feature map per latent
    slope *= largest_pass_elems(vae.decoder, 1, 1) / _REF_PASS_ELEMS_PER_LATENT
    return slope * (elem / 2) * int(z.shape[-2]) * int(z.shape[-1]) + 8 * strip / 2**20 + scores


# 16-bit peak MiB per latent of the strip decode (diffusion_vae_tiling.DECODE_STRIPS_MIB_PER_LATENT_BY_CLASS).
DECODE_STRIPS_MIB_PER_LATENT = 0.048
# largest_pass_elems(decoder, 1, 1) of the decoder that slope was measured on (FLUX.2: 128 / 256 / 512 / 512 channels)
_REF_PASS_ELEMS_PER_LATENT = 16384


def largest_pass_elems(decoder: Any, height: int, width: int) -> int:
    """Elements of the largest feature map a decode of a (height, width) latent passes through."""
    area, best = int(height) * int(width), 0
    for i, up in enumerate(decoder.up_blocks):
        best = max(best, up.resnets[0].conv1.in_channels * area * 4**i)
        if up.upsamplers is not None:
            best = max(best, up.upsamplers[0].conv.out_channels * area * 4 ** (i + 1))
    return best


def worth_strips(decoder: Any, z: Any) -> bool:
    """A decode whose largest pass fits in one strip would hold every strip buffer at full size (more than the
    fused peak) and only pay launches: small tiles keep the fused path."""
    return largest_pass_elems(decoder, z.shape[-2], z.shape[-1]) >= 2 * strip_elems(z.device)


def _call_ok(decoder: Any, z: Any) -> bool:
    torch = fused._torch()
    return (
        isinstance(z, torch.Tensor)
        and z.dim() == 4
        and z.is_cuda
        and z.numel() > 0
        and z.dtype == decoder.conv_in.weight.dtype
        and z.dtype
        == decoder.up_blocks[0].resnets[0].conv1.weight.dtype
        == decoder.conv_out.weight.dtype
        and z.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and decoder.conv_out.weight.device == z.device
        and not decoder.training
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
    )


def install(vae: Any, logger: Any = None) -> bool:
    """Route ``vae.decoder``'s forward through the strip decode. The fused per-module paths stay as the fallback."""
    decoder = getattr(vae, "decoder", None)
    kl = vae is not None and type(vae).__name__ not in STRIP_VAES
    if (
        vae is None
        or (kl and not kl_strip_candidate(vae))
        or decoder is None
        or strips_disabled()
        or not fused.runtime_ok()
        or _residual_kernel() is None
        or getattr(decoder.__dict__.get("forward"), "_unsloth_vae_fused", False)
        or not decoder_ok(decoder)
    ):
        return False
    prev = decoder.forward

    def forward(
        sample: Any,
        latent_embeds: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if (
            args
            or kwargs
            or latent_embeds is not None
            or decoder.__dict__.get("_unsloth_vae_strips_failed", False)
            or strips_disabled()
            or not _call_ok(decoder, sample)
            or not worth_strips(decoder, sample)
            # the KL family only on its cards: the free-memory reading costs ~2 ms a decode (A100), and a decode the
            # fused path would not fit reaches the whole-map strips through the tile planner (full_map_decode)
            or not (
                kl_always_on(sample.device)
                if kl
                else (always_on(sample.device) or fused_would_not_fit(vae, sample))
            )
        ):
            return prev(sample, latent_embeds, *args, **kwargs)
        try:
            if sample.shape[0] == 1:
                out = _decode_one(decoder, sample)
            else:
                out = None
                for i in range(sample.shape[0]):
                    one = _decode_one(decoder, sample[i : i + 1])
                    if out is None:
                        out = fused._torch().empty(
                            (sample.shape[0], *one.shape[1:]), dtype = one.dtype, device = one.device
                        )
                    out[i : i + 1].copy_(one)
                    del one
            decoder._unsloth_vae_strips_calls = (
                decoder.__dict__.get("_unsloth_vae_strips_calls", 0) + 1
            )
            return out
        except Exception as exc:  # noqa: BLE001
            if fused._is_oom(exc):
                raise  # the OOM retry above re-decodes in smaller tiles
            decoder._unsloth_vae_strips_failed = True
            if logger is not None:
                logger.warning(
                    "diffusion.vae_strips: strip decode failed, using the fused path: %s", exc
                )
        return prev(
            sample, latent_embeds
        )  # outside the handler, so the failed call's tensors are freed first

    forward._unsloth_vae_fused = True
    forward._unsloth_vae_strips = True
    decoder.forward = forward
    vae._unsloth_vae_strips = True
    if logger is not None:
        logger.info("diffusion.vae_strips: row-strip decode on %s", type(vae).__name__)
    return True


def _kl_takes_strips(vae: Any, decoder: Any) -> bool:
    """A KL-family decoder takes the strips only where ``kl_always_on``; FLUX.2's also when the fused would not fit."""
    if type(vae).__name__ in STRIP_VAES:
        return True
    if _kl_setting() is not None:
        return bool(_kl_setting())
    try:
        device = decoder.conv_out.weight.device
        return device.type == "cuda" and kl_always_on(device)
    except Exception:  # noqa: BLE001
        return False


def engaged(vae: Any) -> bool:
    """Whether the next decode of ``vae`` takes the strip path (what the tile planner prices)."""
    decoder = getattr(vae, "decoder", None)
    return (
        bool(getattr(vae, "_unsloth_vae_strips", False))
        and decoder is not None
        and getattr(decoder.__dict__.get("forward"), "_unsloth_vae_strips", False)
        and not decoder.__dict__.get("_unsloth_vae_strips_failed", False)
        and not strips_disabled()
        and _kl_takes_strips(vae, decoder)
    )
