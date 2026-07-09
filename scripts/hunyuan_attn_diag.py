#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Diagnose HunyuanVideo-1.5 joint-attention cost: capture the REAL joint sequence length
and per-text-stream padding, then time SDPA three ways at those exact shapes --
  (a) dense [B,1,N,N] bool mask   (current default)
  (b) attn_mask=None              (flash path; only valid if no padding remains)
  (c) dense mask at trimmed N     (text padding removed, mask still built)
so we know whether the win is the N-reduction (trim) or the mask-elimination (null).

Run: CUDA_VISIBLE_DEVICES=3 python scripts/hunyuan_attn_diag.py [--repo ...] [--frames 121]
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

import torch
import torch.nn.functional as F


def _import_diffusers():
    # diffusers eagerly imports bitsandbytes through its quantizers; the bnb build here is
    # mismatched (cuda130), so disable the availability flag before importing (same trick as
    # scripts/video_speedmem_bench.py).
    import diffusers.utils.import_utils as iu

    iu._bitsandbytes_available = False
    import diffusers

    return diffusers


CAP: dict = {}


class _StopCapture(Exception):
    pass


def _block_pre_hook(module, args, kwargs):
    # HunyuanVideo15TransformerBlock.forward(hidden_states, encoder_hidden_states, temb,
    #   attention_mask, image_rotary_emb) -- positional per transformer:786-792.
    def _get(i, name):
        if name in kwargs:
            return kwargs[name]
        return args[i] if i < len(args) else None

    hs = _get(0, "hidden_states")
    ehs = _get(1, "encoder_hidden_states")
    amask = _get(3, "attention_mask")
    if hs is None or ehs is None:
        return None
    CAP["n_video"] = int(hs.shape[1])
    CAP["n_text"] = int(ehs.shape[1])
    CAP["heads"] = int(getattr(module.attn, "heads", 0))
    CAP["dim_head"] = int(hs.shape[-1] // max(CAP["heads"], 1))
    CAP["batch"] = int(hs.shape[0])
    CAP["dtype"] = hs.dtype
    if amask is not None:
        m = amask.bool()
        CAP["text_valid_per_batch"] = m.sum(dim = 1).tolist()
        CAP["text_cols_valid_any"] = int(m.any(dim = 0).sum())  # what our global-trim would keep
    raise _StopCapture


def _model_pre_hook(module, args, kwargs):
    # capture the raw per-stream padding breakdown before the reorder
    def g(name):
        return kwargs.get(name)

    for key, mkey in (
        ("encoder_hidden_states", "encoder_attention_mask"),
        ("encoder_hidden_states_2", "encoder_attention_mask_2"),
    ):
        s = g(key)
        m = g(mkey)
        if s is not None:
            CAP.setdefault("streams", {})[key] = {
                "len": int(s.shape[1]),
                "valid": (m.bool().sum(dim = 1).tolist() if m is not None else None),
            }
    ie = g("image_embeds")
    if ie is not None:
        CAP["image_embeds_len"] = int(ie.shape[1])
        CAP["image_is_t2v"] = bool(torch.all(ie == 0).item())
    return None


def _time_sdpa(
    q,
    k,
    v,
    mask,
    iters = 30,
):
    # q,k,v: [B, H, N, D]
    torch.cuda.synchronize()
    for _ in range(3):  # warmup
        F.scaled_dot_product_attention(q, k, v, attn_mask = mask)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        F.scaled_dot_product_attention(q, k, v, attn_mask = mask)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v")
    ap.add_argument("--frames", type = int, default = 121)
    ap.add_argument("--width", type = int, default = 832)
    ap.add_argument("--height", type = int, default = 480)
    args = ap.parse_args()

    diffusers = _import_diffusers()
    dev = "cuda:0"
    print(f"loading {args.repo} ...", flush = True)
    pipe = diffusers.DiffusionPipeline.from_pretrained(args.repo, torch_dtype = torch.bfloat16)
    pipe = pipe.to(dev)

    pipe.transformer.register_forward_pre_hook(_model_pre_hook, with_kwargs = True)
    pipe.transformer.transformer_blocks[0].register_forward_pre_hook(
        _block_pre_hook, with_kwargs = True
    )

    print("running 1 capture step ...", flush = True)
    try:
        pipe(
            prompt = "a cat playing piano",
            num_frames = args.frames,
            width = args.width,
            height = args.height,
            num_inference_steps = 1,
        )
    except _StopCapture:
        pass
    except Exception as exc:  # noqa: BLE001
        # the StopCapture may surface wrapped; if we captured, continue
        if "n_video" not in CAP:
            raise
        print(f"(generation aborted after capture: {type(exc).__name__})", flush = True)

    print("\n===== CAPTURED SHAPES =====", flush = True)
    for kk in (
        "batch",
        "n_video",
        "n_text",
        "heads",
        "dim_head",
        "dtype",
        "text_valid_per_batch",
        "text_cols_valid_any",
        "image_embeds_len",
        "image_is_t2v",
        "streams",
    ):
        if kk in CAP:
            print(f"  {kk}: {CAP[kk]}", flush = True)

    B = CAP["batch"]
    H = CAP["heads"]
    D = CAP["dim_head"]
    n_video = CAP["n_video"]
    n_text = CAP["n_text"]
    N = n_video + n_text
    # trimmed joint length if we drop globally-invalid text columns
    keep_text = CAP.get("text_cols_valid_any", n_text)
    N_trim = n_video + keep_text
    dtype = CAP["dtype"]
    print(
        f"\n  joint N = {N} (video {n_video} + text {n_text}); "
        f"trimmed N = {N_trim} (text kept {keep_text})",
        flush = True,
    )

    def mk(n):
        return torch.randn(B, H, n, D, device = dev, dtype = dtype)

    # (a) dense mask over full N (current). Build [B,1,N,N] bool (mostly True).
    print("\n===== SDPA TIMING (ms/call, real shapes) =====", flush = True)
    q, k, v = mk(N), mk(N), mk(N)
    dense = torch.ones(B, 1, N, N, dtype = torch.bool, device = dev)
    # emulate text padding: last (n_text - keep_text) columns invalid
    if n_text - keep_text > 0:
        dense[:, :, :, n_video + keep_text :] = False
        dense[:, :, n_video + keep_text :, :] = False
    t_dense = _time_sdpa(q, k, v, dense)
    mask_gb = dense.numel() / 1e9
    print(
        f"  (a) dense [B,1,N,N] mask   N={N:>6}  : {t_dense:7.3f} ms   (mask {mask_gb:.2f} GB)",
        flush = True,
    )

    # (b) no mask over full N (upper bound of flash path if all valid)
    t_none = _time_sdpa(q, k, v, None)
    print(
        f"  (b) attn_mask=None         N={N:>6}  : {t_none:7.3f} ms   ({t_dense/t_none:.2f}x vs a)",
        flush = True,
    )

    # (c) trimmed N, dense all-True mask (text padding removed but mask still built)
    qt, kt, vt = mk(N_trim), mk(N_trim), mk(N_trim)
    dense_t = torch.ones(B, 1, N_trim, N_trim, dtype = torch.bool, device = dev)
    t_dense_trim = _time_sdpa(qt, kt, vt, dense_t)
    print(
        f"  (c) dense mask @trimmed    N={N_trim:>6}  : {t_dense_trim:7.3f} ms   ({t_dense/t_dense_trim:.2f}x vs a)",
        flush = True,
    )

    # (d) trimmed N, no mask (trim + null: the full proposed fast path)
    t_none_trim = _time_sdpa(qt, kt, vt, None)
    print(
        f"  (d) no mask   @trimmed     N={N_trim:>6}  : {t_none_trim:7.3f} ms   ({t_dense/t_none_trim:.2f}x vs a)",
        flush = True,
    )

    print("\n  Interpretation:", flush = True)
    print(
        f"    trim-only ceiling (a->c): {(1-t_dense_trim/t_dense)*100:5.1f}% attn saving",
        flush = True,
    )
    print(f"    null-only ceiling (a->b): {(1-t_none/t_dense)*100:5.1f}% attn saving", flush = True)
    print(
        f"    trim+null   (a->d):       {(1-t_none_trim/t_dense)*100:5.1f}% attn saving", flush = True
    )


if __name__ == "__main__":
    main()
