# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Offline GPTQ for the hosted NVFP4 DiT builds (``scripts/build_prequant_checkpoint.py --gptq-prompts N``).

Calibration renders a fixed prompt list through the dense pipeline and accumulates ``H = E[x x^T]`` per quantised
Linear; ``gptq_nvfp4`` then solves each weight. The result is swapped into the torchao ``NVFP4Tensor`` that
``quantize_`` built for that layer, so the file format, flags and swizzle are exactly the runtime ones. Linears the
build ConvRot-rotates are solved in the rotated basis (``R^T H R`` on the rotated weight)."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Optional

from .diffusion_nvfp4_weight_quant import NVFP4Weight, gptq_nvfp4, replace_payload, rotate_hessian

# Fixed and disjoint from Studio's LPIPS eval prompts; the build is deterministic for a given list prefix.
CALIB_PROMPTS = (
    "Macro photo of a dragonfly on a blade of grass covered in dew drops",
    "A medieval castle on a hill surrounded by autumn forest, aerial view, sunny afternoon",
    "A cyberpunk girl with short silver hair and a glowing visor, city skyline background, digital art",
    "A golden retriever puppy playing with a blue ball on a lawn, shallow depth of field",
    "An isometric illustration of a tiny coffee shop with people inside, pastel colors",
    "A vintage red sports car parked in front of a pastel pink building in Havana",
    "Snow-capped mountains reflected in a calm alpine lake, early morning mist",
    "A chef tossing vegetables in a flaming wok in a restaurant kitchen, motion blur",
    "A handwritten letter and a fountain pen on a wooden desk, warm window light, close up",
    "A crowded subway car at rush hour, people reading phones, fluorescent light, documentary photo",
    "A low-poly 3D render of a fox in a forest clearing, soft shadows",
    "An oil painting of a harbor at sunset with fishing boats and gulls",
    "A product photo of a white sneaker on a seamless orange background, studio lighting",
    "A charcoal sketch of an old man's hands holding a cup of tea",
    'A neon sign reading "NOODLES" in a rainy alley at night, reflections on wet ground',
    "A family having a picnic in a sunflower field, wide angle, bright summer day",
)


def _rotated_copy(weight: Any, group: int) -> Any:
    """The weight exactly as ``rotate_convrot_weight_`` will store it (same ops, same dtype round trip)."""
    from .diffusion_convrot import rotate_convrot_weight_

    holder = SimpleNamespace(weight = SimpleNamespace(data = weight.detach().clone()))
    rotate_convrot_weight_(holder, group)
    return holder.weight.data


def calibrate_and_solve(
    transformer: Any,
    fqns: Iterable[str],
    render_all: Callable[[], None],
    *,
    method: str,
    rotated: Iterable[str] = (),
    group: int = 0,
    budget_bytes: Optional[int] = None,
    damp: float = 0.01,
    blocksize: int = 128,
    log: Callable[[str], None] = print,
) -> dict:
    """GPTQ-solve every Linear in ``fqns`` (dense, unrotated weights): ``render_all()`` runs the calibration renders;
    it is called once per Hessian chunk so at most ``budget_bytes`` of H is resident. Returns fqn -> NVFP4Weight on
    CPU (rotated basis for fqns in ``rotated``)."""
    import torch

    mods = dict(transformer.named_modules())
    fqns = [f for f in fqns if isinstance(mods.get(f), torch.nn.Linear)]
    rotated = set(rotated)
    sizes = {f: mods[f].in_features ** 2 * 4 for f in fqns}
    chunks, cur, acc = [], [], 0
    for f in fqns:
        if cur and budget_bytes is not None and acc + sizes[f] > budget_bytes:
            chunks.append(cur)
            cur, acc = [], 0
        cur.append(f)
        acc += sizes[f]
    if cur:
        chunks.append(cur)
    log(
        f"[gptq] {len(fqns)} linears, H {sum(sizes.values()) / 2 ** 30:.1f} GiB in {len(chunks)} calibration pass(es)"
    )
    hadamard = None
    if rotated:
        from .diffusion_convrot import build_convrot_hadamard
        hadamard = build_convrot_hadamard(int(group), dtype = torch.float32)
    out: dict = {}
    t_cal = t_solve = 0.0
    for ci, chunk in enumerate(chunks):
        dev = mods[chunk[0]].weight.device
        Hs = {f: torch.zeros(mods[f].in_features, mods[f].in_features, device = dev) for f in chunk}
        cnt = dict.fromkeys(chunk, 0)

        def hook_for(f):
            def hook(mod, inp):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).to(torch.float32)
                Hs[f].addmm_(x.t(), x)
                cnt[f] += x.shape[0]

            return hook

        hooks = [mods[f].register_forward_pre_hook(hook_for(f)) for f in chunk]
        t0 = time.time()
        try:
            with torch.no_grad():
                render_all()
        finally:
            for h in hooks:
                h.remove()
        t_cal += time.time() - t0
        t0 = time.time()
        with torch.no_grad():
            for f in chunk:
                if not cnt[f]:
                    raise RuntimeError(
                        f"GPTQ calibration never reached {f}; refusing to build it blind"
                    )
                H = Hs.pop(f).div_(cnt[f])
                w = mods[f].weight
                if f in rotated:
                    w = _rotated_copy(w, int(group))
                    H = rotate_hessian(H, int(group), hadamard)
                q = gptq_nvfp4(w, H, method, blocksize = blocksize, damp = damp)
                out[f] = q._replace(
                    qdata = q.qdata.cpu(),
                    scale = q.scale.cpu(),
                    per_tensor_scale = q.per_tensor_scale.cpu(),
                )
                del H, q
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_solve += time.time() - t0
        Hs.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(
            f"[gptq] pass {ci + 1}/{len(chunks)}: {len(chunk)} linears, calib {t_cal:.0f}s, solve {t_solve:.0f}s"
        )
    return out


def apply_gptq_payloads(transformer: Any, solved: dict) -> int:
    """Swap each solved payload into the NVFP4Tensor ``quantize_`` left on that Linear; raises on a layer that did not
    become NVFP4 or whose shape differs (the build would otherwise ship RTN weights labelled GPTQ)."""
    import torch

    mods = dict(transformer.named_modules())
    done = 0
    for f, q in solved.items():
        mod = mods.get(f)
        weight = getattr(mod, "weight", None)
        if type(weight).__name__ != "NVFP4Tensor":
            raise RuntimeError(
                f"GPTQ layer {f} is {type(weight).__name__} after quantize_, not NVFP4Tensor"
            )
        if tuple(weight.qdata.shape) != tuple(q.qdata.shape):
            raise RuntimeError(
                f"GPTQ payload for {f} is {tuple(q.qdata.shape)}, weight {tuple(weight.qdata.shape)}"
            )
        dev = weight.qdata.device
        q = q._replace(
            qdata = q.qdata.to(dev),
            scale = q.scale.to(dev),
            per_tensor_scale = q.per_tensor_scale.to(dev).reshape(weight.per_tensor_scale.shape),
        )
        mod.weight = torch.nn.Parameter(replace_payload(weight, q), requires_grad = False)
        done += 1
    return done


def pipeline_renderer(
    pipe: Any,
    prompts: Iterable[str],
    *,
    steps: int,
    height: int,
    width: int,
    seed: int,
    extra_kwargs: Optional[dict] = None,
) -> Callable[[], None]:
    """``render_all`` over ``prompts`` (seed + i each), latents only (no VAE decode)."""
    import torch

    prompts = list(prompts)
    extra = dict(extra_kwargs or {})

    def render_all() -> None:
        device = getattr(pipe, "_execution_device", None) or "cuda"
        for i, prompt in enumerate(prompts):
            g = torch.Generator(device = device).manual_seed(int(seed) + i)
            pipe(
                prompt = prompt,
                num_inference_steps = int(steps),
                height = int(height),
                width = int(width),
                generator = g,
                output_type = "latent",
                **extra,
            )

    return render_all
