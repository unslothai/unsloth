# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Row-strip decode for the FLUX.2 / FLUX.3 VAE (AutoencoderKLFlux2) on top of the fused GroupNorm kernels.

GroupNorm stats are taken once over the full map; norm, SiLU, padding and the 3x3 conv then run on bounded row
strips with a one-row halo, so no normalised activation exists at full size. Residuals are written in place (halving
blocks into the front half of their input). Same math as the fused decode. NVIDIA CUDA only; eager only.

Every decode on cards measured no slower (T4, L4); elsewhere (A100 4% slower) only decodes the fused path would not
fit. ``UNSLOTH_FLUX_VAE_STRIPS``: ``0`` off, ``1`` every decode on any NVIDIA card.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Any, Optional

from . import diffusion_vae_fused as fused

STRIPS_ENV = "UNSLOTH_FLUX_VAE_STRIPS"
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


def strips_forced() -> bool:
    return os.environ.get(STRIPS_ENV, "").strip().lower() in ("1", "true", "yes", "on")


# (major, minor) compute capabilities where every strip decode measured within A/A noise of the fused one or faster.
ALWAYS_CAPS = frozenset({(7, 5), (8, 9)})


@lru_cache(maxsize = 16)
def _card_always(index: int) -> bool:
    return tuple(fused._torch().cuda.get_device_capability(index)) in ALWAYS_CAPS


def always_on(device: Any) -> bool:
    """Every decode on this card takes the strip path (else only the ones the fused decode would not fit)."""
    if strips_forced():
        return True
    torch = fused._torch()
    return _card_always(device.index if device.index is not None else torch.cuda.current_device())


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


def _norm_ok(norm: Any) -> bool:
    torch = fused._torch()
    if not isinstance(norm, torch.nn.GroupNorm) or not norm.affine or norm.weight is None:
        return False
    c, g = int(norm.num_channels), int(norm.num_groups)
    return 0 < g and c % g == 0 and c & (c - 1) == 0 and g & (g - 1) == 0 and c <= 1024


def _resnet_ok(block: Any) -> bool:
    torch = fused._torch()
    if not (
        type(block).__name__ == "ResnetBlock2D"
        and fused._resnet2d_fusable(block)
        and _norm_ok(block.norm1)
        and _norm_ok(block.norm2)
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


def decoder_ok(decoder: Any) -> bool:
    """Whether every stage after the mid block is a shape the strip decode reproduces exactly."""
    torch = fused._torch()
    ups = getattr(decoder, "up_blocks", None)
    if (
        ups is None
        or not len(ups)
        or not isinstance(getattr(decoder, "conv_in", None), torch.nn.Conv2d)
        or not _norm_ok(getattr(decoder, "conv_norm_out", None))
        or not isinstance(getattr(decoder, "conv_act", None), torch.nn.SiLU)
        or not _conv3x3_ok(getattr(decoder, "conv_out", None))
    ):
        return False
    for up in ups:
        if type(up).__name__ != "UpDecoderBlock2D" or not all(_resnet_ok(r) for r in up.resnets):
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
    return fused.group_norm_stats(x.unsqueeze(2), norm, in_bias)


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
) -> None:
    """dst = (res + conv(silu(group_norm(x + in_bias)))) / scale strip by strip (conv bias included); ``res`` is
    ``dst`` itself, or ``shortcut`` (a 1x1 conv) of that strip's rows of its input."""
    import torch.nn.functional as F

    stats = _stats(x, norm, in_bias)
    h = x.shape[2]
    for a, b in _strips(h, max(x.numel(), dst.numel())):
        y = _conv_rows(_norm_rows(x, a, b, norm, stats, in_bias), conv, False)
        res = None
        if shortcut is not None:
            sc, src = shortcut
            res = F.conv2d(src.narrow(2, a, b - a), sc.weight, sc.bias)
        _add_rows(dst.narrow(2, a, b - a), y, conv.bias, scale, res)


def _halve(block: Any, x: Any, hidden: Optional[Any]) -> tuple:
    """A 2C -> C block whose output takes the front half of ``x``'s storage. Output rows [a, b) overwrite input rows
    [a / 2, b / 2), which only strips before b read (the shortcut is 1x1 and conv1 has finished)."""
    c = block.conv1.out_channels
    _, _, h, w = x.shape
    if hidden is None or tuple(hidden.shape) != (1, c, h, w) or not _dense_cl(hidden):
        hidden = _empty_cl(c, h, w, x)
    _conv_into(x, block.norm1, block.conv1, hidden)
    out = x.as_strided((1, c, h, w), (h * w * c, 1, w * c, c), x.storage_offset())
    _conv_add(
        hidden, block.norm2, block.conv2, out, block.output_scale_factor, block.conv1.bias,
        shortcut = (block.conv_shortcut, x),
    )  # fmt: skip
    return out, hidden


def _resnet(block: Any, x: Any, hidden: Optional[Any]) -> tuple:
    """(output, hidden buffer for the level's next block)."""
    import torch.nn.functional as F

    cin, cout = block.conv1.in_channels, block.conv1.out_channels
    _, _, h, w = x.shape
    if hidden is None or tuple(hidden.shape) != (1, cout, h, w) or not _dense_cl(hidden):
        hidden = _empty_cl(cout, h, w, x)
    if cin == cout and block.conv_shortcut is None:
        x = _owned_cl(x)
        _conv_into(x, block.norm1, block.conv1, hidden)
        _conv_add(hidden, block.norm2, block.conv2, x, block.output_scale_factor, block.conv1.bias)
        return x, hidden
    if cin == 2 * cout and _dense_cl(x):
        return _halve(block, x, hidden)
    sc = block.conv_shortcut
    out = _owned_cl(F.conv2d(x, sc.weight, sc.bias))
    _conv_into(x, block.norm1, block.conv1, hidden)
    _conv_add(hidden, block.norm2, block.conv2, out, block.output_scale_factor, block.conv1.bias)
    return out, hidden


def _upsample(up: Any, x: Any) -> Any:
    """conv(nearest_2x(x)) in strips of input rows; the 2x rows are written straight into the padded strip."""
    conv = up.conv
    _, c, h, w = x.shape
    out = _empty_cl(conv.out_channels, 2 * h, 2 * w, x)
    for a, b in _strips(h, out.numel()):
        lo, hi = max(0, a - 1), min(h, b + 1)
        n = hi - lo
        top, bottom = int(a == 0), int(b == h)
        buf, inner = _padded_rows(c, 2 * n, top, bottom, 2 * w, x)
        src = x.narrow(2, lo, n).permute(0, 2, 3, 1)
        inner.view(1, n, 2, w, 2, c).copy_(src[:, :, None, :, None, :].expand(1, n, 2, w, 2, c))
        # output rows [2a, 2b) read 2x rows [2a - 1, 2b + 1)
        padded = buf.permute(0, 3, 1, 2).narrow(2, 2 * (a - lo) + top - 1, 2 * (b - a) + 2)
        out.narrow(2, 2 * a, 2 * (b - a)).copy_(_conv_rows(padded, conv, True))
    return out


def _decode_one(decoder: Any, z: Any) -> Any:
    _LOCAL.strip = strip_elems(z.device) if z.is_cuda else (STRIP_ELEMS or 1 << 24)
    h = decoder.conv_in(z)
    if decoder.mid_block is not None:
        h = decoder.mid_block(h, None)
    h = h.to(decoder.up_blocks[0].resnets[0].conv1.weight.dtype)
    for up in decoder.up_blocks:
        h, hidden = _owned_cl(h), None
        for block in up.resnets:
            h, hidden = _resnet(block, h, hidden)
        del hidden
        if up.upsamplers is not None:
            h = _upsample(up.upsamplers[0], h)
    conv = decoder.conv_out
    _, _, hh, ww = h.shape
    out = _empty_cl(conv.out_channels, hh, ww, h)
    stats = _stats(h, decoder.conv_norm_out)
    for a, b in _strips(hh, h.numel()):
        out.narrow(2, a, b - a).copy_(
            _conv_rows(_norm_rows(h, a, b, decoder.conv_norm_out, stats), conv, True)
        )
    return out


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
    if (
        vae is None
        or type(vae).__name__ not in STRIP_VAES
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
            or not (always_on(sample.device) or fused_would_not_fit(vae, sample))
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


def engaged(vae: Any) -> bool:
    """Whether the next decode of ``vae`` takes the strip path (what the tile planner prices)."""
    decoder = getattr(vae, "decoder", None)
    return (
        bool(getattr(vae, "_unsloth_vae_strips", False))
        and decoder is not None
        and getattr(decoder.__dict__.get("forward"), "_unsloth_vae_strips", False)
        and not decoder.__dict__.get("_unsloth_vae_strips_failed", False)
        and not strips_disabled()
    )
