# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The decoder geometry (windows, shifts, global tokens, rope ids, frame trim) follows NVIDIA's Apache-2.0 reference code
# in Efficient-Large-Model/Sol-H3-VAE (code/sol_h3_vae/model.py, decoder.py, template.py).

"""Opt-in Sol-H3-VAE video decoder for MiniMax-H3 (UNSLOTH_H3_SOL_VAE=1).

Sol-H3-VAE is a 24-layer non-causal decoder distilled from H3's 36-layer causal one. It decodes a whole clip at once:
16x16-token spatial windows over 5-latent temporal windows, both shifted by half a window on odd layers (time by 2),
plus 5 global tokens per native 5-latent block that see their whole block and are seen by every window overlapping
it. Rope ids are window-local. Frames = 4t - 3*ceil(t/5), the same lattice as the stock decoder.

Here it is block-streamed: only the float32 residual stream spans the clip. Q/K/V live in a ring of block slots, the
attention and FFN run one temporal window at a time, and the pixel projection one block at a time. Attention is SDPA
only. The checkpoint's float32 weights are cast to the pipeline dtype at load (the small embed / norm / head layers
stay float32, as in the reference's autocast). The decoder replaces the H3 video VAE's ``_decode`` only; the audio VAE
and the encoder are untouched, and any failure falls back to the stock decoder with the reason logged."""

from __future__ import annotations

import math
import os
import time
from typing import Any, Optional

SOL_VAE_ENV = "UNSLOTH_H3_SOL_VAE"
SOL_LEAN_WINDOWS_ENV = "UNSLOTH_H3_SOL_VAE_LEAN_WINDOWS"
SOL_VAE_REPO = "Efficient-Large-Model/Sol-H3-VAE"
SOL_VAE_FILE = "sol_h3_vae.safetensors"
SOL_LAYER = "h3_sol_vae"
SOL_SPEED_OPTIM = "h3_sol_vae"

SOL_BLOCK = 5  # latents per native H3 block (17 frames)
SOL_WINDOW = 16  # spatial window in tokens (256 px)
SOL_T_SHIFT = 2  # temporal shift of the odd layers
SOL_FRAMES_TO_TRIM = 3  # the first latent of every block keeps 1 of its 4 frames
PATCH, PATCH_T, OUT_CH = 16, 4, 3

# Released checkpoint: 1.62B parameters, ~3.24 GB with the GEMM weights in a 2-byte dtype.
SOL_WEIGHTS_GB = 3.24
_DIM, _FFN = 2048, 8192
# FFN rows per GEMM chunk: bounds the 2 x 8192-wide hidden activation to ~0.4 GB in bf16.
_FFN_ROWS = 8192
# Device headroom kept beyond the weights and one decode's peak before the weights stay resident on the card.
_RESIDENT_MARGIN_GB = 4.0

_TRUE = ("1", "true", "yes", "on")


def h3_sol_vae_requested() -> bool:
    return os.environ.get(SOL_VAE_ENV, "").strip().lower() in _TRUE


def _lean_windows() -> bool:
    return os.environ.get(SOL_LEAN_WINDOWS_ENV, "").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def sol_num_frames(latents: int) -> int:
    """Frames Sol (and the stock H3 decoder) returns for ``latents`` latent frames: 4t - 3*ceil(t/5)."""
    t = int(latents)
    if t < 1:
        raise ValueError(f"latent frames must be positive, got {t}")
    return 4 * t - SOL_FRAMES_TO_TRIM * math.ceil(t / SOL_BLOCK)


def sol_latents_for_frames(frames: int) -> Optional[int]:
    """Inverse of ``sol_num_frames``; None when no latent count decodes to exactly ``frames``."""
    f = int(frames)
    if f < 1:
        return None
    blocks, rest = divmod(f, 4 * SOL_BLOCK - SOL_FRAMES_TO_TRIM)
    if rest == 0:
        return blocks * SOL_BLOCK
    # a partial block of m latents gives 4m - 3 frames
    if (rest + SOL_FRAMES_TO_TRIM) % 4:
        return None
    m = (rest + SOL_FRAMES_TO_TRIM) // 4
    return blocks * SOL_BLOCK + m if 1 <= m < SOL_BLOCK else None


def sol_decode_activation_gb(width: int, height: int, frames: int) -> float:
    """Peak device GB of one streamed Sol decode beyond its weights: the float32 residual stream and pixel output (clip
    wide), plus one temporal window's ring slots, padded window buffers, FFN chunk and block projection."""
    h = max(1, math.ceil(int(height) / PATCH))
    w = max(1, math.ceil(int(width) / PATCH))
    t = sol_latents_for_frames(frames) or max(1, math.ceil(int(frames) * SOL_BLOCK / 17) + 1)
    f = sol_num_frames(t)
    tokens = t * h * w
    block = SOL_BLOCK * h * w
    hp = math.ceil((h + SOL_WINDOW // 2) / SOL_WINDOW) * SOL_WINDOW
    wp = math.ceil((w + SOL_WINDOW // 2) / SOL_WINDOW) * SOL_WINDOW
    padded = SOL_BLOCK * hp * wp
    resid = tokens * _DIM * 4
    pixels = f * h * PATCH * w * PATCH * OUT_CH * 4
    ring = 2 * block * 3 * _DIM * 2
    window = 4 * padded * _DIM * 2 + 2 * block * _DIM * 4
    ffn = min(block, _FFN_ROWS) * (3 * _FFN + _DIM) * 2
    head = 2 * block * OUT_CH * PATCH_T * PATCH * PATCH * 4
    return (resid + pixels + ring + window + ffn + head) * 1.15 / 1e9


def sol_decode_need_gb(width: int, height: int, frames: int) -> float:
    """Decode-phase device GB with Sol requested (weights + activation peak), 0.0 when the switch is off."""
    if not h3_sol_vae_requested():
        return 0.0
    try:
        return SOL_WEIGHTS_GB + sol_decode_activation_gb(width, height, frames)
    except Exception:  # noqa: BLE001 - a planner term must never fail a request
        return 0.0


def _is_oom(exc: BaseException) -> bool:
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:  # noqa: BLE001
        pass
    return "out of memory" in str(exc).lower()


def _kernels_module():
    from . import video_minimax_h3_vae as fused
    return fused


def _build_module_classes():
    import torch
    from torch import nn

    class _Block(nn.Module):
        def __init__(self, dim: int, ffn: int, eps: float, dtype: Any) -> None:
            super().__init__()
            self.norm1 = nn.RMSNorm(dim, eps = eps)
            self.norm2 = nn.RMSNorm(dim, eps = eps)
            self.scale1 = nn.Parameter(torch.zeros(dim))
            self.scale2 = nn.Parameter(torch.zeros(dim))
            self.to_q = nn.Linear(dim, dim, dtype = dtype)
            self.to_k = nn.Linear(dim, dim, dtype = dtype)
            self.to_v = nn.Linear(dim, dim, dtype = dtype)
            self.to_out = nn.Linear(dim, dim, dtype = dtype)
            # rows [up | gate]: the fused SwiGLU computes first-half * silu(second-half)
            self.w1 = nn.Linear(dim, 2 * ffn, dtype = dtype)
            self.w2 = nn.Linear(ffn, dim, dtype = dtype)

    class SolH3Decoder(nn.Module):
        """Sol-H3-VAE decoder over raw (denormalized) H3 latents (B, C, T, H, W) -> (B, 3, F, 16H, 16W) float32 in the
        H3 VAE's ImageNet-normalized pixel space, the same contract as ``AutoencoderKLMiniMaxH3._decode``."""

        def __init__(
            self,
            *,
            dim: int = _DIM,
            heads: int = 32,
            layers: int = 24,
            ffn: int = _FFN,
            registers: int = 4,
            rope_dim: int = 48,
            rope_theta: float = 100.0,
            eps: float = 1e-5,
            latent_channels: int = 24,
            window: int = SOL_WINDOW,
            block: int = SOL_BLOCK,
            t_shift: int = SOL_T_SHIFT,
            time_norm: float = float(SOL_BLOCK),
            dtype: Any = None,
        ) -> None:
            super().__init__()
            dtype = dtype or torch.float32
            if dim % heads:
                raise ValueError(f"dim {dim} is not divisible by {heads} heads")
            if rope_dim % 6:
                raise ValueError(f"rope_dim {rope_dim} must be divisible by 6")
            self.dim, self.heads, self.dh = dim, heads, dim // heads
            self.rope_dim, self.rope_theta, self.eps = rope_dim, float(rope_theta), float(eps)
            self.window, self.block, self.t_shift, self.time_norm = (
                window,
                block,
                t_shift,
                float(time_norm),
            )
            self.compute_dtype = dtype
            self.post_quant_conv = nn.Linear(latent_channels, latent_channels)
            self.x_embedder = nn.Linear(latent_channels, dim)
            self.register_tokens = nn.Parameter(torch.zeros(1, registers, dim))
            self.blocks = nn.ModuleList([_Block(dim, ffn, eps, dtype) for _ in range(layers)])
            self.norm_out = nn.LayerNorm(dim, eps = eps)
            self.proj_out = nn.Linear(dim, OUT_CH * PATCH_T * PATCH * PATCH)

        # ------------------------------------------------------------------------------------------------ geometry
        def _shifts(self, li: int) -> tuple[int, int]:
            odd = li % 2 == 1
            return (self.window // 2 if odd else 0), (self.t_shift if odd else 0)

        def _rope(self, t0: int, n: int, H: int, W: int, s: int, st: int, device: Any) -> tuple:
            """cos / sin (n*H*W, rope_dim) for latents [t0, t0 + n) in (t, y, x) order, window-local ids."""
            lt = ((torch.arange(t0, t0 + n, device = device) + st) % self.block).float()
            ly = ((torch.arange(H, device = device) + s) % self.window).float()
            lx = ((torch.arange(W, device = device) + s) % self.window).float()
            t = (lt + 0.5) / self.time_norm * 2 - 1
            hy = (ly + 0.5) / self.window * 2 - 1
            wx = (lx + 0.5) / self.window * 2 - 1
            grid = torch.stack(torch.meshgrid(t, hy, wx, indexing = "ij"), -1).reshape(-1, 3)
            inv_freq = 1.0 / self.rope_theta ** torch.arange(
                0, 1, 6 / self.rope_dim, dtype = torch.float32, device = device
            )
            angles = 2.0 * math.pi * grid[:, :, None] * inv_freq[None, None, :]
            angles = angles.flatten(1, 2).tile(2)
            return angles.cos(), angles.sin()

        def _embed(self, z: Any) -> Any:
            """z (C, T, H, W) raw latents -> float32 tokens (T, H, W, D)."""
            raw = z.float().permute(1, 2, 3, 0)
            raw = torch.nn.functional.linear(
                raw, self.post_quant_conv.weight.float(), self.post_quant_conv.bias.float()
            )
            return torch.nn.functional.linear(
                raw, self.x_embedder.weight.float(), self.x_embedder.bias.float()
            ).contiguous()

        def _globals(self, nb: int, device: Any) -> Any:
            reg = self.register_tokens[0].float()
            g = torch.cat([reg, reg.new_zeros(1, reg.shape[-1])], 0)
            return g.unsqueeze(0).expand(nb, -1, -1).contiguous().to(device)

        def _head_norm(self, x: Any) -> Any:
            shape = x.shape
            y = x.reshape(*shape[:-1], self.heads, self.dh)
            y = torch.nn.functional.rms_norm(y.float(), (self.dh,), None, self.eps).to(x.dtype)
            return y.reshape(shape)

        # ------------------------------------------------------------------------------------------------ streamed
        def _slot(self, blk: Any, x: Any, b: int, s: int, st: int) -> tuple:
            """Q / K / V (roped, head-normed) of native block ``b`` from the layer's input residual."""
            fused = _kernels_module()
            T, H, W, D = x.shape
            t0, t1 = b * self.block, min(T, (b + 1) * self.block)
            cdt = self.compute_dtype
            h = fused._add_rmsnorm(x[t0:t1], None, None, blk.norm1, cdt).reshape(-1, D)
            q, k, v = blk.to_q(h), blk.to_k(h), blk.to_v(h)
            cos, sin = self._slot_rope(t0, t1 - t0, H, W, s, st, x.device)
            fused._qk_norm_rope_(q, cos, sin, self.heads, self.eps)
            fused._qk_norm_rope_(k, cos, sin, self.heads, self.eps)
            n = t1 - t0
            return q.view(n, H, W, D), k.view(n, H, W, D), v.view(n, H, W, D)

        def _slot_rope(
            self, t0: int, n: int, H: int, W: int, s: int, st: int, device: Any
        ) -> tuple:
            """``_rope`` in the compute dtype, cached per decode: with t0 a multiple of block it does not depend on t0."""
            cache = getattr(self, "_rope_tables", None)
            key = (n, H, W, s, st, str(device))
            if cache is not None and key in cache:
                return cache[key]
            cos, sin = self._rope(t0, n, H, W, s, st, device)
            hit = (cos.to(self.compute_dtype).contiguous(), sin.to(self.compute_dtype).contiguous())
            if cache is not None:
                cache[key] = hit
            return hit

        def _global_attn(self, qg: Any, kg: Any, vg: Any, k: Any, v: Any) -> Any:
            """A block's S global tokens over its own tokens + its global tokens: (S, D)."""
            S, D = qg.shape
            hd, dh = self.heads, self.dh
            keys = torch.cat([k.reshape(-1, D), kg], 0).view(-1, hd, dh).transpose(0, 1)
            vals = torch.cat([v.reshape(-1, D), vg], 0).view(-1, hd, dh).transpose(0, 1)
            out = torch.nn.functional.scaled_dot_product_attention(
                qg.view(S, hd, dh).transpose(0, 1)[None], keys[None], vals[None]
            )
            return out[0].transpose(0, 1).reshape(S, D)

        def _window_attn(
            self, q: Any, k: Any, v: Any, kgw: Any, vgw: Any, gvalid: Any, k0: int, s: int
        ) -> Any:
            """Attention over every spatial window of one temporal window. q/k/v (n, H, W, D) start ``k0`` latents into
            the padded window; kgw / vgw (G, D) are the global tokens of the blocks it overlaps, gvalid (G,) bool or
            None when all are valid."""
            n, H, W, D = q.shape
            ws, tb, hd, dh = self.window, self.block, self.heads, self.dh
            Hp = -(-(H + s) // ws) * ws
            Wp = -(-(W + s) // ws) * ws
            nh, nw = Hp // ws, Wp // ws
            nW, L = nh * nw, tb * ws * ws

            def part(t: Any) -> Any:
                buf = t.new_zeros(tb, Hp, Wp, D)
                buf[k0 : k0 + n, s : s + H, s : s + W] = t
                buf = buf.view(tb, nh, ws, nw, ws, hd, dh).permute(1, 3, 5, 0, 2, 4, 6)
                return buf.reshape(nW, hd, L, dh)

            G = kgw.shape[0]
            kg = kgw.view(G, hd, dh).transpose(0, 1)[None].expand(nW, hd, G, dh)
            vg = vgw.view(G, hd, dh).transpose(0, 1)[None].expand(nW, hd, G, dh)
            keys = torch.cat([part(k), kg], 2)
            vals = torch.cat([part(v), vg], 2)
            mask = None
            # every padded slot holds a token iff the window starts at 0, is full, unshifted and unpadded
            lean = gvalid is None and _lean_windows()
            full = k0 == 0 and n == tb and s == 0 and Hp == H and Wp == W
            if not (lean and full):
                valid = torch.zeros(tb, Hp, Wp, dtype = torch.bool, device = q.device)
                valid[k0 : k0 + n, s : s + H, s : s + W] = True
                valid = valid.view(tb, nh, ws, nw, ws).permute(1, 3, 0, 2, 4).reshape(nW, L)
                if gvalid is None:
                    gvalid = torch.ones(G, dtype = torch.bool, device = q.device)
                kmask = torch.cat([valid, gvalid[None].expand(nW, G)], 1)
                if lean or not bool(kmask.all()):
                    mask = torch.zeros(kmask.shape, dtype = q.dtype, device = q.device)
                    mask.masked_fill_(~kmask, float("-inf"))
                    mask = mask[:, None, None, :]
            out = torch.nn.functional.scaled_dot_product_attention(
                part(q), keys, vals, attn_mask = mask
            )
            out = (
                out.view(nh, nw, hd, tb, ws, ws, dh)
                .permute(3, 0, 4, 1, 5, 2, 6)
                .reshape(tb, Hp, Wp, D)
            )
            return out[k0 : k0 + n, s : s + H, s : s + W].contiguous()

        def _ffn_(self, blk: Any, h: Any, normed: Any) -> None:
            """h (rows, D) float32 residual += w2(swiglu(w1(normed))) * scale2, in row chunks."""
            fused = _kernels_module()
            for r0 in range(0, h.shape[0], _FFN_ROWS):
                r1 = min(h.shape[0], r0 + _FFN_ROWS)
                out = blk.w2(fused._swiglu(blk.w1(normed[r0:r1])))
                h[r0:r1] += out * blk.scale2

        def _layer_(self, li: int, blk: Any, x: Any, g: Any) -> None:
            """One layer, in place on the residual x (T, H, W, D) and the global tokens g (nb, S, D)."""
            fused = _kernels_module()
            T, H, W, D = x.shape
            nb, S = g.shape[:2]
            tb = self.block
            cdt = self.compute_dtype
            s, st = self._shifts(li)
            hg = fused._add_rmsnorm(g, None, None, blk.norm1, cdt).reshape(-1, D)
            qg = self._head_norm(blk.to_q(hg)).view(nb, S, D)
            kg = self._head_norm(blk.to_k(hg)).view(nb, S, D)
            vg = blk.to_v(hg).view(nb, S, D)
            og = torch.empty(nb, S, D, dtype = cdt, device = x.device)
            slots: dict = {}
            nt = -(-(T + st) // tb)
            for j in range(nt):
                lo = max(0, j * tb - st)
                hi = min(T - 1, (j + 1) * tb - st - 1)
                b0, b1 = lo // tb, hi // tb
                for b in range(b0, b1 + 1):
                    if b not in slots:
                        slots[b] = self._slot(blk, x, b, s, st)
                        og[b] = self._global_attn(qg[b], kg[b], vg[b], slots[b][1], slots[b][2])
                for b in [b for b in slots if b < b0]:
                    del slots[b]  # the ring: at most the two blocks a window can overlap stay live
                parts = []
                for b in range(b0, b1 + 1):
                    a0, a1 = max(lo, b * tb) - b * tb, min(hi + 1, (b + 1) * tb) - b * tb
                    parts.append([t[a0:a1] for t in slots[b]])
                q, k, v = (
                    torch.cat([p[i] for p in parts]) if len(parts) > 1 else parts[0][i]
                    for i in range(3)
                )
                if b1 != b0:
                    kgw, vgw = kg[[b0, b1]].reshape(-1, D), vg[[b0, b1]].reshape(-1, D)
                else:
                    # the reference pads the second block slot and masks it; leaving it out is the same softmax
                    kgw, vgw = kg[b0], vg[b0]
                o = self._window_attn(q, k, v, kgw, vgw, None, lo - (j * tb - st), s)
                del q, k, v, parts
                xw = x[lo : hi + 1].view(-1, D)
                normed = fused._add_rmsnorm(
                    xw, blk.to_out(o.view(-1, D)), blk.scale1, blk.norm2, cdt
                )
                del o
                self._ffn_(blk, xw, normed)
                del normed
            slots.clear()
            gf = g.view(-1, D)
            normed = fused._add_rmsnorm(gf, blk.to_out(og.view(-1, D)), blk.scale1, blk.norm2, cdt)
            self._ffn_(blk, gf, normed)

        def _project_(self, x: Any, out: Any) -> None:
            """Per block: norm_out + proj_out (float32) -> pixels, the first latent of each block keeping its last frame."""
            T, H, W, D = x.shape
            tb = self.block
            f0 = 0
            for b in range(-(-T // tb)):
                t0, t1 = b * tb, min(T, (b + 1) * tb)
                n = t1 - t0
                h = self.norm_out(x[t0:t1].float())
                h = torch.nn.functional.linear(
                    h, self.proj_out.weight.float(), self.proj_out.bias.float()
                )
                h = h.view(n, H, W, OUT_CH, PATCH_T, PATCH, PATCH).permute(3, 0, 4, 1, 5, 2, 6)
                h = h.reshape(OUT_CH, n * PATCH_T, H * PATCH, W * PATCH)[:, SOL_FRAMES_TO_TRIM:]
                out[:, f0 : f0 + h.shape[1]] = h
                f0 += h.shape[1]

        def _decode_one(self, z: Any) -> Any:
            """z (C, T, H, W) raw latents -> (3, F, 16H, 16W) float32."""
            C, T, H, W = z.shape
            if T == 1:
                # a still is decoded as a block of copies, keeping the first frame (the released decoder's behaviour)
                return self._decode_one(z.expand(C, self.block, H, W))[:, :1]
            x = self._embed(z)
            g = self._globals(-(-T // self.block), x.device)
            self._rope_tables = {} if _lean_windows() else None
            try:
                for li, blk in enumerate(self.blocks):
                    self._layer_(li, blk, x, g)
            finally:
                self._rope_tables = None
            out = torch.empty(
                OUT_CH,
                sol_num_frames(T),
                H * PATCH,
                W * PATCH,
                dtype = torch.float32,
                device = x.device,
            )
            self._project_(x, out)
            return out

        @torch.no_grad()
        def decode_raw(self, z: Any) -> Any:
            """Raw (denormalized) latents (B, C, T, H, W) -> pixels (B, 3, F, 16H, 16W), float32."""
            if z.dim() != 5:
                raise ValueError(f"expected (B, C, T, H, W) latents, got {tuple(z.shape)}")
            device = self.x_embedder.weight.device
            ctx = torch.autocast(device_type = device.type, enabled = False)
            with ctx:
                outs = [self._decode_one(z[i].to(device)) for i in range(z.shape[0])]
            return torch.stack(outs).to(z.device)

        # ------------------------------------------------------------------------------------------------ reference
        @torch.no_grad()
        def reference_decode_raw(self, z: Any) -> Any:
            """Unstreamed decode in plain torch, laid out as the vendor's reference engine: every layer's Q/K/V over the
            whole clip, the clip partitioned into padded windows with masks. The streamed path must match it."""
            outs = []
            for i in range(z.shape[0]):
                zi = z[i]
                C, T, H, W = zi.shape
                still = T == 1
                if still:
                    zi = zi.expand(C, self.block, H, W)
                    T = self.block
                outs.append(self._reference_one(zi)[:, :1] if still else self._reference_one(zi))
            return torch.stack(outs)

        def _reference_one(self, z: Any) -> Any:
            C, T, H, W = z.shape
            dev = z.device
            tb, ws, hd, dh = self.block, self.window, self.heads, self.dh
            nb = -(-T // tb)
            cdt = self.compute_dtype
            x = self._embed(z)
            D = x.shape[-1]
            g = self._globals(nb, dev)
            S = g.shape[1]

            def rms(t: Any, norm: Any) -> Any:
                return torch.nn.functional.rms_norm(
                    t.float(), (D,), norm.weight.float(), norm.eps
                ).to(cdt)

            def rope(t: Any, cos: Any, sin: Any) -> Any:
                shape = t.shape
                y = t.reshape(-1, hd, dh)
                c, sn = cos[:, None].to(t.dtype), sin[:, None].to(t.dtype)
                rot = c.shape[-1]
                a, tail = y[..., :rot], y[..., rot:]
                first, second = a.chunk(2, -1)
                rotated = torch.cat([-second, first], -1)
                return torch.cat([a * c + rotated * sn, tail], -1).reshape(shape)

            def swiglu(t: Any) -> Any:
                up, gate = t.chunk(2, -1)
                return up * torch.nn.functional.silu(gate)

            for li, blk in enumerate(self.blocks):
                s, st = self._shifts(li)
                h = rms(x, blk.norm1).reshape(-1, D)
                hg = rms(g, blk.norm1).reshape(-1, D)
                cos, sin = self._rope(0, T, H, W, s, st, dev)
                q = rope(self._head_norm(blk.to_q(h)), cos, sin).view(T, H, W, hd, dh)
                k = rope(self._head_norm(blk.to_k(h)), cos, sin).view(T, H, W, hd, dh)
                v = blk.to_v(h).view(T, H, W, hd, dh)
                qg = self._head_norm(blk.to_q(hg)).view(nb, S, hd, dh)
                kg = self._head_norm(blk.to_k(hg)).view(nb, S, hd, dh)
                vg = blk.to_v(hg).view(nb, S, hd, dh)
                Tp = -(-(T + st) // tb) * tb
                Hp = -(-(H + s) // ws) * ws
                Wp = -(-(W + s) // ws) * ws
                nt, nh, nw = Tp // tb, Hp // ws, Wp // ws
                nW = nt * nh * nw

                def part(t: Any) -> Any:
                    buf = t.new_zeros(Tp, Hp, Wp, hd, dh)
                    buf[st : st + T, s : s + H, s : s + W] = t
                    buf = buf.view(nt, tb, nh, ws, nw, ws, hd, dh).permute(0, 2, 4, 1, 3, 5, 6, 7)
                    return buf.reshape(nW, tb * ws * ws, hd, dh)

                valid = torch.zeros(Tp, Hp, Wp, dtype = torch.bool, device = dev)
                valid[st : st + T, s : s + H, s : s + W] = True
                vmask = valid.view(nt, tb, nh, ws, nw, ws).permute(0, 2, 4, 1, 3, 5).reshape(nW, -1)
                tw = torch.arange(nt, device = dev)
                b0 = (tw * tb - st).clamp(min = 0) // tb
                b1 = ((tw + 1) * tb - st - 1).clamp(max = T - 1) // tb
                gidx = torch.stack([b0, b1], 1)
                gval = torch.stack([torch.ones_like(b0, dtype = torch.bool), b1 != b0], 1)
                kgw = (
                    kg[gidx]
                    .reshape(nt, 2 * S, hd, dh)[:, None, None]
                    .expand(nt, nh, nw, 2 * S, hd, dh)
                )
                vgw = (
                    vg[gidx]
                    .reshape(nt, 2 * S, hd, dh)[:, None, None]
                    .expand(nt, nh, nw, 2 * S, hd, dh)
                )
                gmask = gval[:, :, None].expand(nt, 2, S).reshape(nt, 2 * S)
                gmask = gmask[:, None, None].expand(nt, nh, nw, 2 * S).reshape(nW, 2 * S)
                K = torch.cat([part(k), kgw.reshape(nW, 2 * S, hd, dh)], 1).transpose(1, 2)
                V = torch.cat([part(v), vgw.reshape(nW, 2 * S, hd, dh)], 1).transpose(1, 2)
                kmask = torch.cat([vmask, gmask], 1)[:, None, None, :]
                ow = torch.nn.functional.scaled_dot_product_attention(
                    part(q).transpose(1, 2), K, V, attn_mask = kmask
                ).transpose(1, 2)
                o = ow.reshape(nt, nh, nw, tb, ws, ws, hd, dh).permute(0, 3, 1, 4, 2, 5, 6, 7)
                o = o.reshape(Tp, Hp, Wp, D)[st : st + T, s : s + H, s : s + W]
                kb = k.new_zeros(nb * tb, H, W, hd, dh)
                kb[:T] = k
                vb = v.new_zeros(nb * tb, H, W, hd, dh)
                vb[:T] = v
                bmask = (
                    (torch.arange(nb * tb, device = dev) < T).view(nb, tb, 1).expand(nb, tb, H * W)
                )
                Kg = torch.cat([kb.view(nb, -1, hd, dh), kg], 1).transpose(1, 2)
                Vg = torch.cat([vb.view(nb, -1, hd, dh), vg], 1).transpose(1, 2)
                gm = torch.cat(
                    [bmask.reshape(nb, -1), torch.ones(nb, S, dtype = torch.bool, device = dev)], 1
                )
                og = (
                    torch.nn.functional.scaled_dot_product_attention(
                        qg.transpose(1, 2), Kg, Vg, attn_mask = gm[:, None, None, :]
                    )
                    .transpose(1, 2)
                    .reshape(nb, S, D)
                )
                x = x + blk.to_out(o.reshape(-1, D)).view(T, H, W, D) * blk.scale1
                g = g + blk.to_out(og.reshape(-1, D)).view(nb, S, D) * blk.scale1
                x = x + blk.w2(swiglu(blk.w1(rms(x, blk.norm2)))) * blk.scale2
                g = g + blk.w2(swiglu(blk.w1(rms(g, blk.norm2)))) * blk.scale2
            out = torch.empty(
                OUT_CH, sol_num_frames(T), H * PATCH, W * PATCH, dtype = torch.float32, device = dev
            )
            self._project_(x, out)
            return out

    return SolH3Decoder


_CLASS: Any = None


def sol_decoder_class():
    global _CLASS
    if _CLASS is None:
        _CLASS = _build_module_classes()
    return _CLASS


# ---------------------------------------------------------------------------------------------------- checkpoint
def _is_gemm_key(key: str) -> bool:
    return any(f".{n}." in key for n in ("to_q", "to_k", "to_v", "to_out", "w1", "w2"))


def convert_sol_state_dict(sd: dict, *, heads: int, dtype: Any) -> dict:
    """Vendor checkpoint keys -> this module's: the fused per-head [q | k | v] projection split into to_q / to_k / to_v,
    SwiGLU w1 rows [gate | up] swapped to [up | gate], post_quant_conv's 1x1x1 kernel flattened; GEMM weights and biases
    in ``dtype``, everything else float32."""
    import torch

    out: dict = {}
    for key, value in sd.items():
        if key.endswith("attn.to_qkv.weight") or key.endswith("attn.to_qkv.bias"):
            prefix = key.split("attn.to_qkv.")[0]
            leaf = key.rsplit(".", 1)[1]
            dim3 = value.shape[0]
            dh = dim3 // (3 * heads)
            per = value.reshape(heads, 3, dh, *value.shape[1:])
            for i, name in enumerate(("to_q", "to_k", "to_v")):
                out[f"{prefix}{name}.{leaf}"] = per[:, i].reshape(heads * dh, *value.shape[1:])
            continue
        if ".ff.w1." in key:
            half = value.shape[0] // 2
            value = torch.cat([value[half:], value[:half]], 0)
            key = key.replace(".ff.w1.", ".w1.")
        elif ".ff.w2." in key:
            key = key.replace(".ff.w2.", ".w2.")
        elif ".attn.to_out." in key:
            key = key.replace(".attn.to_out.", ".to_out.")
        elif key == "post_quant_conv.weight":
            value = value.reshape(value.shape[0], value.shape[1])
        out[key] = value
    for key in list(out):
        out[key] = out[key].to(dtype if _is_gemm_key(key) else torch.float32).contiguous()
    return out


def sol_config_from_state_dict(sd: dict, vae_config: Any = None) -> dict:
    """Module sizes from the checkpoint shapes; heads / rope / eps from the H3 VAE config (released defaults)."""

    def cfg(name: str, default: Any) -> Any:
        if vae_config is None:
            return default
        value = (
            vae_config.get(name, default)
            if isinstance(vae_config, dict)
            else getattr(vae_config, name, default)
        )
        return default if value is None else value

    dim = int(sd["x_embedder.weight"].shape[0])
    heads = int(cfg("decoder_num_attention_heads", 32))
    dh = dim // heads
    layers = len(
        {k.split(".")[1] for k in sd if k.startswith("blocks.") and k.endswith("norm1.weight")}
    )
    w2 = next(
        v
        for k, v in sd.items()
        if k.startswith("blocks.0.") and (".ff.w2.weight" in k or k.endswith(".w2.weight"))
    )
    return dict(
        dim = dim,
        heads = heads,
        layers = layers,
        ffn = int(w2.shape[1]),
        registers = int(sd["register_tokens"].shape[1]),
        rope_dim = int(dh * float(cfg("decoder_rope_dim_ratio", 0.75))),
        rope_theta = float(cfg("decoder_rope_theta", 100.0)),
        eps = float(cfg("decoder_norm_eps", 1e-5)),
        latent_channels = int(sd["x_embedder.weight"].shape[1]),
    )


def build_sol_decoder(
    sd: dict,
    *,
    vae_config: Any = None,
    dtype: Any = None,
    device: Any = "cpu",
    **overrides: Any,
) -> Any:
    """A SolH3Decoder from a vendor-layout state dict (tensors may be float32 on any device); ``overrides`` replace
    module sizes (tests shrink the window)."""
    import torch

    dtype = dtype or torch.float32
    config = {**sol_config_from_state_dict(sd, vae_config), **overrides}
    converted = convert_sol_state_dict(sd, heads = config["heads"], dtype = dtype)
    cls = sol_decoder_class()
    with torch.device("meta"):
        model = cls(**config, dtype = dtype)
    model.load_state_dict({k: v.to(device) for k, v in converted.items()}, strict = True, assign = True)
    return model.eval().requires_grad_(False)


def fetch_sol_weights(*, hf_token: Optional[str] = None, local_files_only: bool = False) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        SOL_VAE_REPO, SOL_VAE_FILE, token = hf_token or None, local_files_only = local_files_only
    )


def load_sol_decoder(
    path: str,
    *,
    vae_config: Any = None,
    dtype: Any = None,
    device: Any = "cpu",
) -> Any:
    """Read the float32 safetensors one tensor at a time, casting each straight onto ``device`` (host RAM stays small)."""
    from safetensors import safe_open

    sd: dict = {}
    target = str(device)
    with safe_open(path, framework = "pt", device = "cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            sd[key] = (
                tensor.to(target, dtype = dtype)
                if _is_vendor_gemm_key(key) and dtype is not None
                else tensor.to(target)
            )
    return build_sol_decoder(sd, vae_config = vae_config, dtype = dtype, device = target)


def _is_vendor_gemm_key(key: str) -> bool:
    return any(f".{n}." in key for n in ("to_qkv", "to_out", "w1", "w2"))


# ---------------------------------------------------------------------------------------------------- install
def _free_device_gb(device: Any) -> Optional[float]:
    try:
        import torch

        dev = torch.device(device)
        if dev.type != "cuda":
            return None
        free, total = torch.cuda.mem_get_info(dev)
        try:
            from .diffusion_memory import user_budget_reserve_bytes
            free = max(0, int(free) - user_budget_reserve_bytes(dev, total))
        except Exception:  # noqa: BLE001
            pass
        return free / 1e9
    except Exception:  # noqa: BLE001
        return None


def _on_host(model: Any) -> bool:
    return model.x_embedder.weight.device.type == "cpu"


def _sol_dtype(dtype: Any) -> Any:
    import torch
    return dtype if dtype in (torch.bfloat16, torch.float16) else torch.float32


def install_h3_sol_vae(
    vae: Any,
    *,
    device: Any,
    dtype: Any,
    hf_token: Optional[str] = None,
    local_files_only: bool = False,
    logger: Any = None,
    weights_path: Optional[str] = None,
) -> bool:
    """Route the H3 video VAE's ``_decode`` through Sol when UNSLOTH_H3_SOL_VAE=1. Never raises: a missing download, a
    load error or no room keeps the stock decoder with the reason logged. True when engaged."""
    if vae is None or not h3_sol_vae_requested():
        return False
    try:
        from . import decode_layers

        if decode_layers.find(vae, "_decode", SOL_LAYER) is not None:
            return True
        if type(vae).__name__ != "AutoencoderKLMiniMaxH3" or not callable(
            getattr(vae, "_decode", None)
        ):
            raise RuntimeError(f"{type(vae).__name__} is not the MiniMax-H3 video VAE")
        import torch

        cdt = _sol_dtype(dtype)
        t0 = time.perf_counter()
        path = weights_path or fetch_sol_weights(
            hf_token = hf_token, local_files_only = local_files_only
        )
        free_gb = _free_device_gb(device)
        # resident on the card when it has room for the weights plus a long clip's decode; else parked on the host and
        # moved on for each decode (a streamed / low-VRAM placement)
        resident = free_gb is None or free_gb >= SOL_WEIGHTS_GB + _RESIDENT_MARGIN_GB + 4.0
        home = torch.device(device) if resident else torch.device("cpu")
        model = load_sol_decoder(
            path, vae_config = getattr(vae, "config", None), dtype = cdt, device = home
        )
    except Exception as exc:  # noqa: BLE001 - optional decoder, the stock one stays
        if logger is not None:
            logger.warning("video.h3_sol_vae: not engaged, keeping the stock decoder: %s", exc)
        return False

    stats = {"calls": 0, "fallbacks": 0, "last_s": None, "last_reason": None}
    exec_device = torch.device(device)

    def make(inner: Any, layer: Any) -> Any:
        def _decode(z: Any, *args: Any, **kwargs: Any) -> Any:
            if vae.__dict__.get("_unsloth_h3_sol_vae_failed") or args or kwargs:
                return inner(z, *args, **kwargs)
            reason = None
            parked = _on_host(model) and exec_device.type != "cpu"
            try:
                t = time.perf_counter()
                if parked:
                    model.to(exec_device)
                out = model.decode_raw(z)
                if z.is_cuda:
                    torch.cuda.synchronize(z.device)
                stats["calls"] += 1
                stats["last_s"] = round(time.perf_counter() - t, 3)
                if logger is not None:
                    logger.info(
                        "video.h3_sol_vae: decoded %d latent frames -> %d frames at %dx%d in %.2f s",
                        int(z.shape[2]),
                        int(out.shape[2]),
                        int(out.shape[-1]),
                        int(out.shape[-2]),
                        stats["last_s"],
                    )
                return out
            except Exception as exc:  # noqa: BLE001
                reason = f"{type(exc).__name__}: {exc}"[:400]
                if not _is_oom(exc):
                    vae.__dict__["_unsloth_h3_sol_vae_failed"] = True
            finally:
                if parked:
                    model.to("cpu")
            # outside the handler: the failed decode's frames (and its activations) are gone before the retry
            if exec_device.type != "cpu" and not _on_host(model):
                # a resident decoder leaves the card before the stock fallback; after an OOM it stays parked on the
                # host, so later decodes move it on and back off
                try:
                    model.to("cpu")
                except Exception:  # noqa: BLE001
                    pass
            stats["fallbacks"] += 1
            stats["last_reason"] = reason
            try:
                if z.is_cuda:
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            if logger is not None:
                logger.warning(
                    "video.h3_sol_vae: decode failed (%s); using the stock decoder", reason
                )
            return inner(z, *args, **kwargs)

        _decode._unsloth_h3_sol_vae = True
        return _decode

    decode_layers.wrap(vae, "_decode", SOL_LAYER, make)
    vae.__dict__["_unsloth_h3_sol_vae"] = model
    vae.__dict__["_unsloth_h3_sol_vae_stats"] = stats
    if logger is not None:
        logger.info(
            "video.h3_sol_vae: engaged %s (%d layers, %s, %s, loaded in %.1f s); the stock decoder stays as fallback",
            SOL_VAE_REPO,
            len(model.blocks),
            str(cdt).replace("torch.", ""),
            f"resident on {home}" if resident else "parked on the host between decodes",
            time.perf_counter() - t0,
        )
    return True


def h3_sol_vae_failed(vae: Any) -> bool:
    return bool(getattr(vae, "__dict__", {}).get("_unsloth_h3_sol_vae_failed"))
