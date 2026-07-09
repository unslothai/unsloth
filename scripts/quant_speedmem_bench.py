# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Speed + memory benchmark for the text-encoder auto-quant and VAE quant casters.

The companion accuracy sweep (``quant_accuracy_sweep.py``) proved the two new quantisers do not
hurt quality; this measures the other half: the memory they save and their latency impact. It
drives the REAL casters -- ``quantize_text_encoders`` (``core.inference.diffusion_precision``) and
``quantize_vae`` (``core.inference.diffusion_vae_quant``) -- exactly as the loader does, but on the
components in isolation so the numbers are not drowned by the dominant DiT (a separate, already
shipped quant feature that is held constant here).

Three modes:
  --mode te    load ONLY the text_encoder subfolder(s), measure steady weight memory + an encode
               forward latency, dense vs ``quantize_text_encoders(mode="auto")``.
  --mode vae   load ONLY the vae subfolder, measure weight memory + a decode latency, dense vs
               ``quantize_vae(mode="auto")`` (layerwise fp8); for flux.2 also explicit fp8_dynamic.
  --mode e2e   load the full pipeline (DiT dense in both runs) and measure load-peak / gen-peak
               resident memory + generation latency, dense TE+VAE vs auto TE+VAE.

Memory idiom is lifted from scripts/diffusion_bench.py + scripts/quant_probe.py:
``reset_peak_memory_stats`` -> load -> ``memory_allocated`` (steady weights) / ``max_memory_allocated``
(peak) with ``synchronize`` + ``perf_counter`` around timed sections. torch / torchao / diffusers are
imported lazily so ``--help`` works without them.

Example:
    CUDA_VISIBLE_DEVICES=4 python scripts/quant_speedmem_bench.py --family qwen-image --mode te \\
        --out outputs/quant_speedmem
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_ROOT = _REPO_ROOT / "studio" / "backend"
for _p in (str(_BACKEND_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PROMPT = "A photograph of an astronaut riding a horse on the surface of the moon, detailed, 8k"

# base diffusers repos (all cached). flux.2 uses the cached FLUX.2-dev repo whose vae/text_encoder
# subfolders are the same architecture as klein; it is included to exercise the explicit VAE
# fp8_dynamic opt-in path (the only image family where fp8_dynamic is measured in-bar).
_FAMILIES: dict[str, dict[str, Any]] = {
    "qwen-image": {"repo": "Qwen/Qwen-Image"},
    "flux.1": {"repo": "black-forest-labs/FLUX.1-dev"},
    "sdxl": {"repo": "stabilityai/stable-diffusion-xl-base-1.0"},
    "flux.2": {"repo": "black-forest-labs/FLUX.2-dev"},
    # video families -- Conv3d VAEs, which is where the VAE actually holds real memory.
    "ltx-2": {"repo": "Lightricks/LTX-2"},
    "hunyuanvideo-1.5": {"repo": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"},
    "wan2.2-ti2v-5b": {"repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "vae_force_fp32": True},
}

_TE_ATTRS = ("text_encoder", "text_encoder_2", "text_encoder_3")


# ── cuda memory / timing helpers (lifted from diffusion_bench.py) ──────────────
def _sync() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _alloc_gb() -> float:
    import torch
    return torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def _peak_gb() -> float:
    import torch
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def _empty() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _median(xs: list[float]) -> float:
    return sorted(xs)[len(xs) // 2] if xs else 0.0


_LP: dict = {}


def _lpips_alex(ref_arr, arr):
    """LPIPS(AlexNet) between two HxWx3 uint8 images (net kept on CPU). None if lpips missing."""
    try:
        import lpips
        import torch

        fn = _LP.get("fn")
        if fn is None:
            fn = lpips.LPIPS(net = "alex", verbose = False).eval()
            _LP["fn"] = fn

        def _t(a):
            import torch as _torch
            return _torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0

        with torch.no_grad():
            return round(float(fn(_t(ref_arr), _t(arr)).item()), 4)
    except Exception:
        return None


def _timed_generate(pipe, *, steps, res, seed):
    """One generation returning (PIL image, total seconds, [per-step ms]). Per-step wall-clock
    via callback_on_step_end (each step synchronised)."""
    import time as _time

    import torch

    g = torch.Generator(device = "cuda").manual_seed(seed)
    step_ts: list[float] = []
    last = [0.0]

    def _cb(pp, i, t, kw):
        torch.cuda.synchronize()
        now = _time.perf_counter()
        if last[0]:
            step_ts.append((now - last[0]) * 1000.0)
        last[0] = now
        return kw

    _sync()
    t0 = _time.perf_counter()
    img = pipe(
        prompt = PROMPT,
        width = res,
        height = res,
        num_inference_steps = steps,
        generator = g,
        callback_on_step_end = _cb,
    ).images[0]
    _sync()
    return img, (_time.perf_counter() - t0), step_ts


def _import_diffusers():
    """diffusers with the bnb quantiser disabled (we quant via torchao / layerwise only)."""
    import torch  # noqa: F401  (torch/torchao first so their extensions register)
    import torchao  # noqa: F401
    import diffusers.utils.import_utils as iu

    iu._bitsandbytes_available = False
    import diffusers

    return diffusers


def _target():
    """The object the casters read (.device == "cuda" string, .dtype is torch.bfloat16)."""
    import types

    import torch

    return types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)


# ── model_index component class resolution ────────────────────────────────────
def _model_index(repo: str) -> dict:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo, "model_index.json")
    with open(path, "r", encoding = "utf-8") as fh:
        return json.load(fh)


def _load_named(repo: str, name: str):
    """Load one pipeline sub-component (text_encoder[_2/_3]) with the exact class the pipeline
    uses, resolved from model_index.json -> [library, class]. Same modules the real loader casts."""
    import importlib

    import torch

    idx = _model_index(repo)
    spec = idx.get(name)
    if not spec or not isinstance(spec, list) or len(spec) != 2 or spec[0] is None:
        return None
    lib, cls_name = spec
    module = importlib.import_module(lib)
    klass = getattr(module, cls_name)
    return klass.from_pretrained(repo, subfolder = name, torch_dtype = torch.bfloat16)


def _load_text_encoders(repo: str, device: str):
    """A bag exposing .text_encoder[_2/_3] (the attrs quantize_text_encoders iterates)."""
    import types

    bag = types.SimpleNamespace()
    n = 0
    for attr in _TE_ATTRS:
        mod = _load_named(repo, attr)
        if mod is not None:
            mod = mod.to(device).eval()
            n += 1
        setattr(bag, attr, mod)
    if n == 0:
        raise RuntimeError(f"{repo}: no text encoders found in model_index.json")
    return bag


_PROMPT_SUITE = (
    "A photograph of an astronaut riding a horse on the moon",
    "a serene mountain lake at sunrise, mist over the water, ultra detailed",
    "portrait of an old fisherman with a weathered face, dramatic lighting",
    "a bustling futuristic city street at night, neon signs, rain reflections",
    "a bowl of ripe strawberries on a wooden table, macro, soft light",
    "an oil painting of a stormy sea with a lighthouse",
)


def _load_tokenizers(repo: str):
    """One tokenizer per present text-encoder slot (tokenizer / tokenizer_2 / tokenizer_3)."""
    from transformers import AutoTokenizer

    toks = {}
    for te_attr, tk_sub in (
        ("text_encoder", "tokenizer"),
        ("text_encoder_2", "tokenizer_2"),
        ("text_encoder_3", "tokenizer_3"),
    ):
        try:
            toks[te_attr] = AutoTokenizer.from_pretrained(repo, subfolder = tk_sub)
        except Exception:
            toks[te_attr] = None
    return toks


def _encoder_hidden(te, ids, mask):
    """Mean-pooled last hidden state (float vector) -- the encoder output that feeds the DiT."""
    import torch

    with torch.inference_mode():
        try:
            out = te(input_ids = ids, attention_mask = mask, output_hidden_states = True)
        except TypeError:
            out = te(ids, output_hidden_states = True)
    hs = getattr(out, "last_hidden_state", None)
    if hs is None:
        hidden = getattr(out, "hidden_states", None)
        hs = hidden[-1] if hidden else (out[0] if isinstance(out, (tuple, list)) else out)
    m = mask.unsqueeze(-1).to(hs.dtype)
    v = (hs * m).sum(1) / m.sum(1).clamp(min = 1)
    return v.float().flatten()


def _te_hidden_refs(bag, toks, device):
    """{te_attr: [pooled hidden vector per prompt]} across the present encoders."""
    import torch

    refs: dict[str, list] = {}
    for attr in _TE_ATTRS:
        te = getattr(bag, attr, None)
        tok = toks.get(attr)
        if te is None or tok is None:
            continue
        vecs = []
        for p in _PROMPT_SUITE:
            enc = tok(p, return_tensors = "pt", padding = "max_length", truncation = True, max_length = 64)
            ids = enc["input_ids"].to(device)
            mask = enc.get("attention_mask")
            mask = mask.to(device) if mask is not None else torch.ones_like(ids)
            vecs.append(_encoder_hidden(te, ids, mask))
        refs[attr] = vecs
    return refs


def measure_te_accuracy(
    family: str,
    *,
    schemes = ("fp8", "fp8_dynamic"),
    logger = None,
) -> list[dict]:
    """Hidden-state cosine / relL2 of each TE scheme vs the dense bf16 encoder (per encoder),
    on real prompts. Bar (PR#150): cosine >= 0.99 and min_cosine >= 0.98."""
    import torch
    import torch.nn.functional as F

    from core.inference.diffusion_precision import quantize_text_encoders

    repo = _FAMILIES[family]["repo"]
    device = "cuda"
    toks = _load_tokenizers(repo)

    _empty()
    bag = _load_text_encoders(repo, device)
    refs = _te_hidden_refs(bag, toks, device)
    del bag
    _empty()

    rows: list[dict] = []
    for scheme in schemes:
        bag = _load_text_encoders(repo, device)
        engaged = quantize_text_encoders(bag, _target(), mode = scheme, family = family, logger = logger)
        cur = _te_hidden_refs(bag, toks, device)
        for attr, ref_vecs in refs.items():
            q_vecs = cur.get(attr, [])
            if not q_vecs:
                continue
            cosines = [F.cosine_similarity(r, q, dim = 0).item() for r, q in zip(ref_vecs, q_vecs)]
            rell2 = [
                ((q - r).norm() / r.norm().clamp(min = 1e-8)).item() for r, q in zip(ref_vecs, q_vecs)
            ]
            mean_cos = sum(cosines) / len(cosines)
            min_cos = min(cosines)
            rows.append(
                {
                    "family": family,
                    "encoder": attr,
                    "scheme": engaged or scheme,
                    "cosine": round(mean_cos, 5),
                    "min_cosine": round(min_cos, 5),
                    "relL2": round(sum(rell2) / len(rell2), 4),
                    "pass": bool(mean_cos >= 0.99 and min_cos >= 0.98),
                }
            )
        del bag
        _empty()
    return rows


def _encode_once(bag) -> None:
    """One forward through every present text encoder on a fixed short token batch. Uses a length
    within each encoder's max positions (CLIP caps at 77) so position embeddings never overflow."""
    import torch
    with torch.inference_mode():
        for attr in _TE_ATTRS:
            te = getattr(bag, attr, None)
            if te is None:
                continue
            cfg = getattr(te, "config", None)
            vocab = int(getattr(cfg, "vocab_size", 30000) or 30000)
            maxpos = int(getattr(cfg, "max_position_embeddings", 64) or 64)
            length = max(8, min(64, maxpos))
            ids = torch.randint(1, min(vocab, 30000), (1, length), device = "cuda")
            mask = torch.ones_like(ids)
            try:
                te(input_ids = ids, attention_mask = mask)
            except TypeError:
                te(ids)


# ── VAE loading + latent shape (reused from quant_accuracy_sweep) ──────────────
def _load_vae(repo: str, device: str):
    import torch

    diffusers = _import_diffusers()
    vae = diffusers.AutoModel.from_pretrained(repo, subfolder = "vae", torch_dtype = torch.bfloat16)
    return vae.to(device).eval()


def _latent_spec(vae) -> tuple[int, bool]:
    from torch import nn

    conv = None
    dec = getattr(vae, "decoder", None)
    for mod in (dec, vae):
        if mod is None:
            continue
        for m in mod.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                conv = m
                break
        if conv is not None:
            break
    is_3d = isinstance(conv, nn.Conv3d)
    channels = None
    for key in ("latent_channels", "z_dim", "in_channels"):
        v = getattr(getattr(vae, "config", object()), key, None)
        if isinstance(v, int):
            channels = v
            break
    if conv is not None:
        channels = conv.in_channels
    return int(channels), bool(is_3d)


def _decode_once(vae, z) -> None:
    import torch
    with torch.inference_mode():
        try:
            out = vae.decode(z)
        except TypeError:
            out = vae.decode(z, return_dict = True)
        _ = out.sample if hasattr(out, "sample") else out[0]


def _make_latent(vae, device: str):
    """A fixed modest latent at the VAE's native shape (spatial 64 -> ~512px; 3D uses a few frames)."""
    import torch

    channels, is_3d = _latent_spec(vae)
    g = torch.Generator().manual_seed(1234)
    shape = (1, channels, 3, 32, 32) if is_3d else (1, channels, 64, 64)
    z = torch.randn(shape, generator = g, dtype = torch.float32)
    return z.to(device = device, dtype = torch.bfloat16)


# ── measurement primitives ────────────────────────────────────────────────────
def _time_median(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    dts = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        dts.append((time.perf_counter() - t0) * 1000.0)  # ms
    return _median(dts)


# ── mode: te ──────────────────────────────────────────────────────────────────
def measure_te(
    family: str,
    *,
    warmup: int,
    iters: int,
    scheme: str = "auto",
    logger = None,
) -> list[dict]:
    from core.inference.diffusion_precision import quantize_text_encoders

    repo = _FAMILIES[family]["repo"]
    device = "cuda"
    rows: list[dict] = []

    # dense
    _empty()
    _reset_peak()
    bag = _load_text_encoders(repo, device)
    _sync()
    mem_dense = _alloc_gb()
    _reset_peak()
    lat_dense = _time_median(lambda: _encode_once(bag), warmup = warmup, iters = iters)
    peak_dense = _peak_gb()

    # quant in place (scheme="auto" resolves the ladder; else force an explicit scheme to compare)
    engaged = quantize_text_encoders(bag, _target(), mode = scheme, family = family, logger = logger)
    _empty()
    _sync()
    mem_quant = _alloc_gb()
    _reset_peak()
    lat_quant = _time_median(lambda: _encode_once(bag), warmup = warmup, iters = iters)
    peak_quant = _peak_gb()

    del bag
    _empty()
    rows.append(
        {
            "family": family,
            "component": "text_encoder",
            "scheme": engaged or "dense(none-engaged)",
            "mem_dense_gb": round(mem_dense, 3),
            "mem_quant_gb": round(mem_quant, 3),
            "saved_gb": round(mem_dense - mem_quant, 3),
            "mem_ratio": round(mem_quant / mem_dense, 3) if mem_dense else None,
            "peak_dense_gb": round(peak_dense, 3),
            "peak_quant_gb": round(peak_quant, 3),
            "lat_dense_ms": round(lat_dense, 2),
            "lat_quant_ms": round(lat_quant, 2),
            "lat_delta_pct": round((lat_quant - lat_dense) / lat_dense * 100.0, 1)
            if lat_dense
            else None,
        }
    )
    return rows


# ── mode: vae ───────────────────────────────────────────────────────────────
def _measure_vae_scheme(
    family: str,
    repo: str,
    mode: str,
    *,
    warmup: int,
    iters: int,
    logger = None,
) -> dict:
    import types

    from core.inference.diffusion_vae_quant import quantize_vae

    device = "cuda"
    _empty()
    _reset_peak()
    vae = _load_vae(repo, device)
    z = _make_latent(vae, device)
    _sync()
    mem_dense = _alloc_gb()
    _reset_peak()
    lat_dense = _time_median(lambda: _decode_once(vae, z), warmup = warmup, iters = iters)
    peak_dense = _peak_gb()

    # quantize_vae reads pipe.vae, so hand it a bag exposing .vae (it mutates that module in place).
    # Pass the family's force_fp32 (Wan) so the real dense-only behaviour is reflected.
    force_fp32 = bool(_FAMILIES.get(family, {}).get("vae_force_fp32", False))
    engaged = quantize_vae(
        types.SimpleNamespace(vae = vae),
        _target(),
        mode = mode,
        family = family,
        force_fp32 = force_fp32,
        logger = logger,
    )
    _empty()
    _sync()
    mem_quant = _alloc_gb()
    _reset_peak()
    lat_quant = _time_median(lambda: _decode_once(vae, z), warmup = warmup, iters = iters)
    peak_quant = _peak_gb()

    del vae, z
    _empty()
    return {
        "family": family,
        "component": "vae",
        "requested": mode,
        "scheme": engaged or "dense(none-engaged)",
        "mem_dense_gb": round(mem_dense, 3),
        "mem_quant_gb": round(mem_quant, 3),
        "saved_gb": round(mem_dense - mem_quant, 3),
        "mem_ratio": round(mem_quant / mem_dense, 3) if mem_dense else None,
        "peak_dense_gb": round(peak_dense, 3),
        "peak_quant_gb": round(peak_quant, 3),
        "lat_dense_ms": round(lat_dense, 2),
        "lat_quant_ms": round(lat_quant, 2),
        "lat_delta_pct": round((lat_quant - lat_dense) / lat_dense * 100.0, 1)
        if lat_dense
        else None,
    }


def measure_vae(
    family: str,
    *,
    warmup: int,
    iters: int,
    logger = None,
) -> list[dict]:
    repo = _FAMILIES[family]["repo"]
    rows = [_measure_vae_scheme(family, repo, "auto", warmup = warmup, iters = iters, logger = logger)]
    if family == "flux.2":
        # the one image family where the explicit fp8_dynamic conv opt-in is measured in-bar.
        rows.append(
            _measure_vae_scheme(
                family, repo, "fp8_dynamic", warmup = warmup, iters = iters, logger = logger
            )
        )
    return rows


# ── mode: e2e (qwen-image) ────────────────────────────────────────────────────
def _e2e_run(
    repo: str,
    *,
    quant: bool,
    steps: int,
    res: int,
    seed: int,
    iters: int,
    family: str,
    logger = None,
):
    import torch

    from core.inference.diffusion_precision import quantize_text_encoders
    from core.inference.diffusion_vae_quant import quantize_vae

    diffusers = _import_diffusers()
    _empty()
    _reset_peak()
    pipe = diffusers.DiffusionPipeline.from_pretrained(repo, torch_dtype = torch.bfloat16)
    pipe = pipe.to("cuda")
    load_peak = _peak_gb()
    te_scheme = vae_scheme = None
    if quant:
        te_scheme = quantize_text_encoders(
            pipe, _target(), mode = "auto", family = family, logger = logger
        )
        vae_scheme = quantize_vae(pipe, _target(), mode = "auto", family = family, logger = logger)
        _empty()
    weights_gb = _alloc_gb()

    def _gen():
        g = torch.Generator(device = "cuda").manual_seed(seed)
        step_ts: list[float] = []
        last = [0.0]

        def _cb(pp, i, t, kw):
            torch.cuda.synchronize()
            now = time.perf_counter()
            if last[0]:
                step_ts.append((now - last[0]) * 1000.0)
            last[0] = now
            return kw

        _sync()
        t0 = time.perf_counter()
        img = pipe(
            prompt = PROMPT,
            width = res,
            height = res,
            num_inference_steps = steps,
            generator = g,
            callback_on_step_end = _cb,
        ).images[0]
        _sync()
        return img, (time.perf_counter() - t0), step_ts

    img, _, _ = _gen()  # warmup
    _reset_peak()
    dts, steps_ms, last_img = [], [], img
    for _ in range(iters):
        last_img, dt, st = _gen()
        dts.append(dt)
        steps_ms.append(_median(st) if st else 0.0)
    gen_peak = _peak_gb()
    del pipe
    _empty()
    return {
        "variant": "auto" if quant else "dense",
        "te_scheme": te_scheme or ("auto" if quant else "dense"),
        "vae_scheme": vae_scheme or ("auto" if quant else "dense"),
        "load_peak_gb": round(load_peak, 2),
        "weights_gb": round(weights_gb, 2),
        "gen_peak_gb": round(gen_peak, 2),
        "gen_latency_s": round(_median(dts), 3),
        "per_step_ms": round(_median(steps_ms), 1),
    }, last_img


def measure_e2e(
    family: str,
    *,
    steps: int,
    res: int,
    seed: int,
    iters: int,
    out: Path,
    variant: str = "both",
    logger = None,
) -> list[dict]:
    repo = _FAMILIES[family]["repo"]
    # "both" runs dense then auto in one process (fast, but the 2nd run is on a hotter GPU / a
    # fragmented allocator). "dense"/"auto" run a single variant in a fresh process so an
    # interleaved dense/auto/dense sequence separates a real quant effect from GPU thermal drift.
    variants = {"both": (False, True), "dense": (False,), "auto": (True,)}[variant]
    rows = []
    for quant in variants:
        row, img = _e2e_run(
            repo,
            quant = quant,
            steps = steps,
            res = res,
            seed = seed,
            iters = iters,
            family = family,
            logger = logger,
        )
        try:
            img.save(out / f"e2e_{family}_{row['variant']}.png")
        except Exception:
            pass
        rows.append(row)
        print(f"  e2e {row['variant']:5s}: {json.dumps(row)}", flush = True)
    return rows


# ── mode: dit (transformer quant, the real speed lever) ───────────────────────
def _compile_blocks(transformer) -> bool:
    """Regional block compile (the real feature path); torchao dynamic quant is ~30x slower eager,
    so both dense and quant variants are compiled for a fair speedup comparison."""
    fn = getattr(transformer, "compile_repeated_blocks", None)
    if not callable(fn):
        return False
    for kw in ({"dynamic": True}, {}):
        try:
            fn(**kw)
            return True
        except Exception:
            continue
    return False


def _dit_run(
    repo: str,
    family: str,
    *,
    dit_quant: str,
    steps: int,
    res: int,
    seed: int,
    iters: int,
    compile_blocks: bool = True,
    logger = None,
):
    """Load the full pipeline dense, quantise ONLY the transformer (TE + VAE stay dense to isolate
    the DiT), regional-compile it (the real feature path), then measure per-step + total latency and
    peak resident memory. Returns row + image."""
    import torch

    from core.inference.diffusion_transformer_quant import quantize_transformer

    diffusers = _import_diffusers()
    _empty()
    _reset_peak()
    pipe = diffusers.DiffusionPipeline.from_pretrained(repo, torch_dtype = torch.bfloat16).to("cuda")
    load_peak = _peak_gb()
    engaged = None
    if dit_quant and dit_quant != "none":
        engaged = quantize_transformer(
            pipe, _target(), mode = dit_quant, family = family, logger = logger
        )
        _empty()
    weights_gb = _alloc_gb()
    compiled = _compile_blocks(getattr(pipe, "transformer", None)) if compile_blocks else False

    img, _, _ = _timed_generate(pipe, steps = steps, res = res, seed = seed)  # warmup (triggers compile)
    _reset_peak()
    dts, steps_ms, last_img = [], [], img
    for _ in range(iters):
        last_img, dt, st = _timed_generate(pipe, steps = steps, res = res, seed = seed)
        dts.append(dt)
        steps_ms.append(_median(st) if st else 0.0)
    gen_peak = _peak_gb()
    del pipe
    _empty()
    return {
        "family": family,
        "dit_quant": dit_quant,
        "dit_scheme": engaged or "dense",
        "compiled": compiled,
        "load_peak_gb": round(load_peak, 2),
        "weights_gb": round(weights_gb, 2),
        "gen_peak_gb": round(gen_peak, 2),
        "gen_latency_s": round(_median(dts), 3),
        "per_step_ms": round(_median(steps_ms), 1),
    }, last_img


def measure_dit(
    family: str,
    *,
    schemes,
    steps: int,
    res: int,
    seed: int,
    iters: int,
    out: Path,
    logger = None,
):
    """Dense reference + each DiT scheme (auto/fp8/int8/mxfp8), reporting speedup, peak-memory drop,
    and LPIPS(AlexNet) vs the dense render (the whole-image accuracy metric)."""
    import numpy as np

    repo = _FAMILIES[family]["repo"]
    dense_row, dense_img = _dit_run(
        repo, family, dit_quant = "none", steps = steps, res = res, seed = seed, iters = iters, logger = logger
    )
    try:
        dense_img.save(out / f"dit_{family}_dense.png")
    except Exception:
        pass
    ref_arr = np.array(dense_img)
    dense_row["lpips_vs_dense"] = 0.0
    dense_row["speedup_vs_dense"] = 1.0
    rows = [dense_row]
    print(f"  dit dense: {json.dumps(dense_row)}", flush = True)
    base_lat = dense_row["gen_latency_s"] or 1.0
    for scheme in schemes:
        row, img = _dit_run(
            repo,
            family,
            dit_quant = scheme,
            steps = steps,
            res = res,
            seed = seed,
            iters = iters,
            logger = logger,
        )
        row["lpips_vs_dense"] = _lpips_alex(ref_arr, np.array(img))
        row["speedup_vs_dense"] = (
            round(base_lat / row["gen_latency_s"], 3) if row["gen_latency_s"] else None
        )
        try:
            img.save(out / f"dit_{family}_{scheme}.png")
        except Exception:
            pass
        rows.append(row)
        print(f"  dit {scheme:5s}: {json.dumps(row)}", flush = True)
    return rows


# ── main ──────────────────────────────────────────────────────────────────────
def main(argv = None) -> int:
    ap = argparse.ArgumentParser(description = __doc__)
    ap.add_argument("--family", required = True, choices = sorted(_FAMILIES))
    ap.add_argument("--mode", required = True, choices = ("te", "vae", "e2e", "teacc", "dit"))
    ap.add_argument(
        "--dit-schemes", default = "auto", help = "dit mode: comma list e.g. auto,fp8,int8,mxfp8"
    )
    ap.add_argument("--warmup", type = int, default = 2)
    ap.add_argument("--iters", type = int, default = 5)
    ap.add_argument("--steps", type = int, default = 20, help = "e2e denoise steps")
    ap.add_argument("--res", type = int, default = 1024, help = "e2e image size")
    ap.add_argument("--seed", type = int, default = 42)
    ap.add_argument("--e2e-iters", type = int, default = 3)
    ap.add_argument(
        "--variant", choices = ("both", "dense", "auto"), default = "both", help = "e2e variant(s)"
    )
    ap.add_argument("--te-scheme", default = "auto", help = "te mode: auto | fp8_dynamic | fp8 | int8")
    ap.add_argument("--out", default = "outputs/quant_speedmem")
    args = ap.parse_args(argv)

    import logging

    logging.basicConfig(level = logging.INFO, format = "%(message)s")
    logger = logging.getLogger("speedmem")

    out = Path(args.out)
    out.mkdir(parents = True, exist_ok = True)

    print(f"== speed+mem bench: family={args.family} mode={args.mode} ==", flush = True)
    if args.mode == "dit":
        schemes = [s.strip() for s in args.dit_schemes.split(",") if s.strip()]
        rows = measure_dit(
            args.family,
            schemes = schemes,
            steps = args.steps,
            res = args.res,
            seed = args.seed,
            iters = args.e2e_iters,
            out = out,
            logger = logger,
        )
    elif args.mode == "teacc":
        rows = measure_te_accuracy(args.family, logger = logger)
    elif args.mode == "te":
        rows = measure_te(
            args.family, warmup = args.warmup, iters = args.iters, scheme = args.te_scheme, logger = logger
        )
    elif args.mode == "vae":
        rows = measure_vae(args.family, warmup = args.warmup, iters = args.iters, logger = logger)
    else:
        rows = measure_e2e(
            args.family,
            steps = args.steps,
            res = args.res,
            seed = args.seed,
            iters = args.e2e_iters,
            out = out,
            variant = args.variant,
            logger = logger,
        )

    for r in rows:
        print("  " + json.dumps(r), flush = True)
    suffix = ""
    if args.mode == "e2e" and args.variant != "both":
        suffix = f"_{args.variant}"
    elif args.mode == "te" and args.te_scheme != "auto":
        suffix = f"_{args.te_scheme}"
    dest = out / f"{args.mode}_{args.family}{suffix}.json"
    with open(dest, "w", encoding = "utf-8") as fh:
        json.dump(rows, fh, indent = 2)
    print(f"wrote {dest}", flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
