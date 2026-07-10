# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Decoded-image accuracy sweep for the auto VAE (and end-to-end) quantisation.

The VAE is the most quality-sensitive stage of a diffusion pipeline: it turns the DiT
latent into RGB pixels, so a coarse fp8 grid on its convs can BAND the output. The new
default on fp8-GEMM silicon casts the VAE to torchao PerTensor ``fp8_dynamic`` (Conv2d /
Conv3d) or diffusers layerwise ``fp8``. This harness measures exactly what ships: it
loads a family's VAE, decodes a FIXED seeded latent set through the dense bf16 VAE
(reference) and through the same VAE quantised by the repo's own casters
(``core.inference.diffusion_vae_quant.quantize_vae``), then reports decoded-image
LPIPS(AlexNet) / PSNR / SSIM of quantised vs dense. A (family, scheme) that exceeds the
bar (LPIPS <= 0.05, SSIM >= 0.95) belongs in ``_VAE_FAMILY_SCHEME_DENY``.

``--mode e2e`` instead runs a full pipeline generate dense-bf16 vs everything-auto
(auto transformer + auto text encoder + auto VAE) and reports mean LPIPS over a prompt
set, the PyTorch-blog "nearly indistinguishable" (~0.1) composed-defaults check.

The LPIPS AlexNet net is kept on CPU (or a separate --lpips-device) so it never holds
memory on the measured GPU. torch / torchao / diffusers / lpips are imported lazily so
``--help`` works on a host without them.

Examples:
    python scripts/quant_accuracy_sweep.py --family sdxl flux.1 qwen-image
    python scripts/quant_accuracy_sweep.py --family ltx-2 --latent-t 3 --latent-hw 32
    python scripts/quant_accuracy_sweep.py --mode e2e --family flux.1 --e2e-model \\
        black-forest-labs/FLUX.1-schnell
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

# ── env: the ancient bitsandbytes in this venv cannot build for CUDA 13 and hard-raises when
# diffusers lazily imports its bnb quantiser. We never use bnb here (quant is torchao / layerwise),
# so tell diffusers bnb is unavailable BEFORE any VAE class import, and silence the bnb welcome.
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_ROOT = _REPO_ROOT / "studio" / "backend"
for _p in (str(_BACKEND_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── VAE-only accuracy bars (decoder is the most sensitive stage) ──────────────
LPIPS_BAR = 0.05
SSIM_BAR = 0.95
# End-to-end composed-defaults bar (PyTorch blog "nearly indistinguishable").
E2E_LPIPS_BAR = 0.10

# family -> the diffusers base repo whose ``vae`` subfolder we decode with. Only the VAE
# subfolder is fetched; the class is resolved from its config by diffusers AutoModel.
_VAE_FAMILIES: dict[str, dict[str, Any]] = {
    "sdxl": {"repo": "stabilityai/stable-diffusion-xl-base-1.0"},
    "flux.1": {"repo": "black-forest-labs/FLUX.1-schnell"},
    "qwen-image": {"repo": "Qwen/Qwen-Image"},
    "flux.2-klein": {"repo": "black-forest-labs/FLUX.2-klein-4B"},
    "ltx-2": {"repo": "Lightricks/LTX-2"},
    "hunyuanvideo-1.5": {"repo": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"},
}


def _check_deps() -> None:
    import importlib.util as ilu
    missing = [
        m
        for m in ("torch", "torchao", "diffusers", "lpips", "numpy", "PIL")
        if not ilu.find_spec(m)
    ]
    if missing:
        print(
            "missing deps: " + ", ".join(missing) + "\n"
            "  uv pip install torch torchao diffusers lpips numpy pillow",
            file = sys.stderr,
            flush = True,
        )
        raise SystemExit(2)


def _import_diffusers():
    """Import diffusers with the bnb quantiser disabled (see env note at top)."""
    import torch  # noqa: F401  (torch/torchao first so their extensions register)
    import torchao  # noqa: F401
    import diffusers.utils.import_utils as iu

    iu._bitsandbytes_available = False
    import diffusers

    return diffusers


# ── VAE loading + latent shape introspection ─────────────────────────────────


def _load_vae(repo: str, subfolder: str, device: str):
    import torch

    diffusers = _import_diffusers()
    vae = diffusers.AutoModel.from_pretrained(repo, subfolder = subfolder, torch_dtype = torch.bfloat16)
    vae = vae.to(device).eval()
    return vae


def _first_decoder_conv(vae: Any):
    """Return the decoder's input conv (its in_channels == latent channels, its ndim tells
    2D vs 3D). Falls back to the first conv anywhere."""
    from torch import nn

    dec = getattr(vae, "decoder", None)
    for mod in (dec, vae):
        if mod is None:
            continue
        for m in mod.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                return m
    return None


def _latent_spec(vae: Any) -> tuple[int, bool]:
    """(latent_channels, is_3d) for a VAE, from its decoder input conv (robust across
    AutoencoderKL / QwenImage / Flux2 / LTX2 / HunyuanVideo naming)."""
    from torch import nn

    conv = _first_decoder_conv(vae)
    is_3d = isinstance(conv, nn.Conv3d)
    channels = None
    for key in ("latent_channels", "z_dim", "in_channels"):
        v = getattr(getattr(vae, "config", object()), key, None)
        if isinstance(v, int):
            channels = v
            break
    if conv is not None:
        channels = conv.in_channels  # authoritative: what decode actually consumes
    return int(channels), bool(is_3d)


def _ref_images(args: argparse.Namespace, size: int) -> list:
    """Natural reference photos (resized to size x size) for the encode round-trip."""
    import glob

    import numpy as np
    from PIL import Image

    ref_dir = Path(args.ref_image_dir)
    files = sorted(glob.glob(str(ref_dir / "*.jpg")) + glob.glob(str(ref_dir / "*.png")))[
        : args.num_samples
    ]
    imgs = []
    for f in files:
        im = Image.open(f).convert("RGB").resize((size, size), Image.BICUBIC)
        imgs.append(np.asarray(im, dtype = np.uint8))
    return imgs


def _encode_latent(vae: Any, x: Any):
    """Encode a preprocessed pixel tensor to a deterministic latent (mode of the posterior
    when the VAE exposes one), robust across AutoencoderKL / QwenImage / Flux2 / LTX2 / HV15."""
    import torch

    with torch.no_grad():
        enc = vae.encode(x)
    dist = getattr(enc, "latent_dist", None)
    if dist is not None:
        return dist.mode() if hasattr(dist, "mode") else dist.sample()
    if hasattr(enc, "latent"):
        return enc.latent
    if isinstance(enc, (tuple, list)):
        first = enc[0]
        return first.mode() if hasattr(first, "mode") else first
    return enc


def _make_latents(vae: Any, args: argparse.Namespace, device: str):
    """A fixed latent batch at the family's latent shape (one per sample). Default: encode
    natural photos through the dense VAE (in-distribution, natural decoded content -- the
    regime the LPIPS/SSIM bars are calibrated for). ``--latents random`` uses seeded N(0,1)."""
    import torch

    channels, is_3d = _latent_spec(vae)
    lat = []
    if args.latents == "encode":
        size = args.enc_hw_3d if is_3d else args.enc_hw
        imgs = _ref_images(args, size)
        for arr in imgs:
            x = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0).div(127.5).sub(1.0)
            if is_3d:
                x = x.unsqueeze(2).repeat(1, 1, args.enc_frames, 1, 1)  # static clip [1,3,T,H,W]
            x = x.to(device = device, dtype = torch.bfloat16)
            lat.append(_encode_latent(vae, x))
        if lat:
            return lat, is_3d
        print("  (no ref images found; falling back to random latents)", flush = True)
    for seed in range(args.num_samples):
        g = torch.Generator().manual_seed(1000 + seed)
        shape = (
            (1, channels, args.latent_t, args.latent_hw_3d, args.latent_hw_3d)
            if is_3d
            else (1, channels, args.latent_hw, args.latent_hw)
        )
        z = torch.randn(shape, generator = g, dtype = torch.float32)
        lat.append(z.to(device = device, dtype = torch.bfloat16))
    return lat, is_3d


def _decode(vae: Any, z: Any):
    """Decode one latent, returning a list of HxWx3 uint8 numpy frames (>1 for a video VAE)."""
    import numpy as np
    import torch

    with torch.no_grad():
        try:
            out = vae.decode(z)
        except TypeError:
            out = vae.decode(z, return_dict = True)
        sample = out.sample if hasattr(out, "sample") else out[0]
    sample = sample.float().clamp(-1, 1)
    # [B,C,H,W] (image) or [B,C,T,H,W] (video). Emit one frame per temporal slot.
    frames = []
    if sample.dim() == 5:
        b, c, t, h, w = sample.shape
        for ti in range(t):
            frames.append(sample[0, :, ti])
    else:
        frames.append(sample[0])
    imgs = []
    for f in frames:
        arr = (
            ((f.permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)
        )
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis = 2)
        imgs.append(arr)
    return imgs  # list of HxWx3 uint8


# ── metrics ──────────────────────────────────────────────────────────────────


class _Lpips:
    """AlexNet LPIPS kept off the measured GPU. Inputs are HxWx3 uint8 arrays mapped to [-1,1]."""

    def __init__(self, device: str = "cpu") -> None:
        import lpips
        import torch

        self.torch = torch
        self.device = device
        self.fn = lpips.LPIPS(net = "alex", verbose = False).to(device).eval()

    def __call__(self, a: Any, b: Any) -> float:
        t = self.torch

        def to_t(x):
            return (
                t.from_numpy(x)
                .float()
                .permute(2, 0, 1)
                .unsqueeze(0)
                .div(127.5)
                .sub(1.0)
                .to(self.device)
            )

        with t.no_grad():
            return float(self.fn(to_t(a), to_t(b)).item())


def _metrics(ref_frames: list, q_frames: list, lp: "_Lpips") -> dict[str, float]:
    from diffusion_quality import psnr, ssim  # pure-numpy PSNR/SSIM
    from PIL import Image

    ls, ps, ss = [], [], []
    for a, b in zip(ref_frames, q_frames):
        ls.append(lp(a, b))
        ps.append(psnr(Image.fromarray(a), Image.fromarray(b)))
        ss.append(ssim(Image.fromarray(a), Image.fromarray(b)))

    def _m(xs):
        fin = [x for x in xs if x != float("inf")]
        base = fin if fin else xs
        return round(sum(base) / len(base), 4) if base else None

    return {"lpips": _m(ls), "psnr": _m(ps), "ssim": _m(ss)}


# ── VAE isolation sweep ──────────────────────────────────────────────────────


def _apply_fp8_dynamic_no1x1(vae_q: Any) -> None:
    """Diagnostic caster: the shipped PerTensor fp8_dynamic config, but the conv filter
    ALSO excludes pointwise (1x1 / 1x1x1) convs. torchao 0.17's f8f8bf16_conv kernel
    rejects pointwise convs ("Activation and filter channels must match"), so the shipped
    fp8_dynamic caster crashes at decode on any VAE that has a 1x1 conv with %16 channels.
    This variant isolates the fp8_dynamic MATH accuracy on the convs that DO run, to show
    whether excluding 1x1 (a recommended caster fix) keeps fp8_dynamic in-bar."""
    from torch import nn
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerTensor, quantize_

    from core.inference.diffusion_vae_quant import _VAE_KEEP_DENSE_TOKENS

    def filter_fn(module: Any, fqn: str = "") -> bool:
        if not isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            return False
        w = getattr(module, "weight", None)
        if w is None or w.dim() < 2 or w.shape[0] % 16 or w.shape[1] % 16:
            return False
        ks = getattr(module, "kernel_size", None)
        if isinstance(ks, tuple) and all(
            k == 1 for k in ks
        ):  # pointwise conv -> torchao kernel fails
            return False
        name = fqn.lower() if fqn else ""
        return not any(tok in name for tok in _VAE_KEEP_DENSE_TOKENS)

    quantize_(
        vae_q,
        Float8DynamicActivationFloat8WeightConfig(granularity = PerTensor()),
        filter_fn = filter_fn,
    )


def _sweep_vae(args: argparse.Namespace, lp: "_Lpips", out_dir: Path) -> list[dict]:
    import copy

    import numpy as np
    from PIL import Image

    from core.inference import diffusion_vae_quant as vq

    class _Target:
        def __init__(self):
            import torch
            self.device = "cuda"
            self.dtype = torch.bfloat16

    target = _Target()
    rows: list[dict] = []
    for family in args.family:
        repo = _VAE_FAMILIES[family]["repo"]
        print(f"\n=== VAE {family} ({repo}) ===", flush = True)
        t0 = time.time()
        try:
            vae = _load_vae(repo, "vae", "cuda")
        except Exception as exc:  # noqa: BLE001
            print(f"  load FAILED: {type(exc).__name__}: {str(exc)[:200]}", flush = True)
            rows.append({"family": family, "scheme": "-", "error": f"load: {exc}"})
            continue
        ch, is_3d = _latent_spec(vae)
        cls = type(vae).__name__
        print(
            f"  {cls} latent_ch={ch} {'3D' if is_3d else '2D'} loaded {time.time()-t0:.0f}s",
            flush = True,
        )

        latents, _ = _make_latents(vae, args, "cuda")
        ref_by_sample = [_decode(vae, z) for z in latents]

        fam_dir = out_dir / family
        fam_dir.mkdir(parents = True, exist_ok = True)
        Image.fromarray(ref_by_sample[0][0]).save(fam_dir / "dense_s0.png")

        for scheme in args.scheme:
            try:
                vae_q = copy.deepcopy(vae)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{scheme}] deepcopy FAILED: {exc}", flush = True)
                continue
            pipe = type("P", (), {"vae": vae_q})()
            # "fp8_dynamic_no1x1" is a diagnostic that bypasses quantize_vae to apply the
            # PerTensor fp8 config with pointwise convs excluded (torchao 0.17 kernel gap).
            if scheme == "fp8_dynamic_no1x1":
                try:
                    _apply_fp8_dynamic_no1x1(vae_q)
                    engaged = scheme
                except Exception as exc:  # noqa: BLE001
                    print(f"  [{scheme}] apply FAILED: {str(exc)[:120]}", flush = True)
                    del vae_q
                    _empty_cache()
                    continue
            else:
                engaged = vq.quantize_vae(
                    pipe, target, mode = scheme, family = family, offload_active = False, force_fp32 = False
                )
            if engaged != scheme:
                print(f"  [{scheme}] NOT engaged (returned {engaged}); skipping", flush = True)
                rows.append(
                    {"family": family, "vae_class": cls, "scheme": scheme, "verdict": "NOT_ENGAGED"}
                )
                del vae_q
                _empty_cache()
                continue
            try:
                q_by_sample = [_decode(vae_q, z) for z in latents]
            except Exception as exc:  # noqa: BLE001 — the shipped caster produced a VAE that crashes at decode
                emsg = f"{type(exc).__name__}: {str(exc)[:100]}"
                print(f"  [{scheme}] DECODE CRASH: {emsg}", flush = True)
                rows.append(
                    {
                        "family": family,
                        "vae_class": cls,
                        "scheme": scheme,
                        "verdict": "CRASH",
                        "error": emsg,
                    }
                )
                del vae_q
                _empty_cache()
                continue
            all_ref = [f for frames in ref_by_sample for f in frames]
            all_q = [f for frames in q_by_sample for f in frames]
            m = _metrics(all_ref, all_q, lp)
            Image.fromarray(q_by_sample[0][0]).save(fam_dir / f"{scheme}_s0.png")
            lp_pass = m["lpips"] is not None and m["lpips"] <= LPIPS_BAR
            ss_pass = m["ssim"] is not None and m["ssim"] >= SSIM_BAR
            verdict = "PASS" if (lp_pass and ss_pass) else "FAIL"
            row = {
                "family": family,
                "vae_class": cls,
                "scheme": scheme,
                "n_frames": len(all_ref),
                **m,
                "verdict": verdict,
            }
            rows.append(row)
            print(
                f"  [{scheme}] LPIPS={m['lpips']} PSNR={m['psnr']} SSIM={m['ssim']}  -> {verdict}",
                flush = True,
            )
            del vae_q
            _empty_cache()
        del vae
        _empty_cache()
    return rows


def _empty_cache() -> None:
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


# ── end-to-end (dense bf16 vs everything-auto) ───────────────────────────────

_E2E_PROMPTS = [
    "A cozy reading nook by a rain-streaked window, warm lamplight, a cat asleep on a stack of books",
    "A lone lighthouse on a rocky cliff at sunset, dramatic clouds, crashing waves, highly detailed",
    "A bustling night market street in the rain, neon signs reflected in puddles, cinematic",
    "A close-up portrait of an elderly fisherman, weathered skin, soft window light, film grain",
    "A red fox trotting through a snowy pine forest at dawn, volumetric light",
    "A steaming bowl of ramen on a wooden table, chopsticks, shallow depth of field",
]


def _apply_auto(pipe: Any, family: str, components: list[str]) -> dict[str, Optional[str]]:
    """Apply the shipped auto stack in place for the selected components (subset of
    {transformer, text_encoder, vae}), so the VAE's end-to-end contribution can be isolated."""
    import torch

    from core.inference import diffusion_precision as dp
    from core.inference import diffusion_transformer_quant as tq
    from core.inference import diffusion_vae_quant as vq

    class _Target:
        device = "cuda"
        dtype = torch.bfloat16

    tgt = _Target()
    engaged: dict[str, Optional[str]] = {}
    if "transformer" in components:
        engaged["transformer"] = tq.quantize_transformer(pipe, tgt, mode = "auto", family = family)
    if "text_encoder" in components:
        try:
            engaged["text_encoder"] = dp.quantize_text_encoders(
                pipe, tgt, mode = "auto", family = family
            )
        except Exception as exc:  # noqa: BLE001
            engaged["text_encoder"] = f"err:{type(exc).__name__}"
    if "vae" in components:
        engaged["vae"] = vq.quantize_vae(pipe, tgt, mode = "auto", family = family)
    return engaged


def _sweep_e2e(args: argparse.Namespace, lp: "_Lpips", out_dir: Path) -> list[dict]:
    import torch

    diffusers = _import_diffusers()
    rows: list[dict] = []
    for family in args.family:
        model = args.e2e_model or _VAE_FAMILIES.get(family, {}).get("repo")
        print(f"\n=== E2E {family} ({model}) ===", flush = True)
        prompts = args.prompts or _E2E_PROMPTS
        seeds = args.seeds

        def _gen(pipe):
            imgs = []
            for pi, prompt in enumerate(prompts):
                for seed in seeds:
                    g = torch.Generator(device = "cuda").manual_seed(seed)
                    kw = dict(
                        prompt = prompt,
                        num_inference_steps = args.steps,
                        generator = g,
                        height = args.height,
                        width = args.width,
                    )
                    if args.guidance is not None:
                        kw["guidance_scale"] = args.guidance
                    out = pipe(**kw)
                    imgs.append((pi, seed, out.images[0]))
            return imgs

        try:
            pipe = diffusers.AutoPipelineForText2Image.from_pretrained(
                model, torch_dtype = torch.bfloat16
            ).to("cuda")
        except Exception as exc:  # noqa: BLE001
            print(f"  pipe load FAILED: {type(exc).__name__}: {str(exc)[:200]}", flush = True)
            rows.append({"family": family, "error": f"load: {exc}"})
            continue
        ref = _gen(pipe)
        del pipe
        _empty_cache()

        pipe2 = diffusers.AutoPipelineForText2Image.from_pretrained(
            model, torch_dtype = torch.bfloat16
        ).to("cuda")
        engaged = _apply_auto(pipe2, family, args.e2e_components)
        print(f"  engaged: {engaged}", flush = True)
        q = _gen(pipe2)
        del pipe2
        _empty_cache()

        import numpy as np

        ls = []
        fam_dir = out_dir / f"e2e_{family}"
        fam_dir.mkdir(parents = True, exist_ok = True)
        for (pi, seed, a), (_, _, b) in zip(ref, q):
            aa, bb = np.asarray(a.convert("RGB")), np.asarray(b.convert("RGB"))
            ls.append(lp(aa, bb))
            a.save(fam_dir / f"dense_p{pi}_s{seed}.png")
            b.save(fam_dir / f"auto_p{pi}_s{seed}.png")
        mean_l = round(sum(ls) / len(ls), 4) if ls else None
        verdict = "PASS" if (mean_l is not None and mean_l <= E2E_LPIPS_BAR) else "FAIL"
        rows.append(
            {"family": family, "engaged": engaged, "mean_lpips": mean_l, "verdict": verdict}
        )
        print(f"  mean LPIPS={mean_l}  -> {verdict}", flush = True)
    return rows


# ── output ───────────────────────────────────────────────────────────────────


def _write(out_dir: Path, mode: str, rows: list[dict]) -> None:
    (out_dir / f"{mode}_results.json").write_text(json.dumps(rows, indent = 2))
    print(f"\nwrote {out_dir / f'{mode}_results.json'}", flush = True)
    print(f"\n=== {mode.upper()} RESULTS ===", flush = True)
    if mode == "vae":
        print(f"  bars: LPIPS <= {LPIPS_BAR}, SSIM >= {SSIM_BAR}", flush = True)
        print(
            f"  {'family':<20}{'scheme':<14}{'LPIPS':>9}{'PSNR':>9}{'SSIM':>9}  verdict", flush = True
        )
        for r in rows:
            if "error" in r:
                print(f"  {r['family']:<20}{'(error)':<14} {r['error'][:60]}", flush = True)
                continue
            print(
                f"  {r['family']:<20}{r['scheme']:<14}{_f(r.get('lpips')):>9}"
                f"{_f(r.get('psnr')):>9}{_f(r.get('ssim')):>9}  {r.get('verdict')}",
                flush = True,
            )
    else:
        print(f"  bar: mean LPIPS <= {E2E_LPIPS_BAR}", flush = True)
        for r in rows:
            if "error" in r:
                print(f"  {r['family']}: (error) {r['error'][:80]}", flush = True)
                continue
            print(
                f"  {r['family']:<20} mean_lpips={r.get('mean_lpips')}  {r.get('verdict')}  {r.get('engaged')}",
                flush = True,
            )


def _f(v: Any) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "-"


# ── cli ──────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description = "Decoded-image accuracy sweep for the auto VAE / end-to-end quantisation.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices = ["vae", "e2e"], default = "vae")
    p.add_argument("--family", nargs = "+", default = list(_VAE_FAMILIES.keys()))
    p.add_argument("--scheme", nargs = "+", default = ["fp8_dynamic", "fp8_dynamic_no1x1", "fp8"])
    p.add_argument("--num-samples", type = int, default = 5, help = "latents to average over")
    p.add_argument(
        "--latents",
        choices = ["encode", "random"],
        default = "encode",
        help = "encode natural photos (in-distribution) or seeded N(0,1) latents",
    )
    p.add_argument(
        "--ref-image-dir",
        default = "outputs/quant_accuracy/_refs",
        help = "natural photos to encode for the round-trip",
    )
    p.add_argument("--enc-hw", type = int, default = 512, help = "2D encode pixel H=W")
    p.add_argument("--enc-hw-3d", type = int, default = 256, help = "3D encode pixel H=W")
    p.add_argument("--enc-frames", type = int, default = 9, help = "3D encode pixel frame count")
    p.add_argument("--latent-hw", type = int, default = 64, help = "2D random-latent H=W (x8 -> 512px)")
    p.add_argument("--latent-hw-3d", type = int, default = 32, help = "3D random-latent H=W")
    p.add_argument("--latent-t", type = int, default = 3, help = "3D random-latent temporal length")
    p.add_argument(
        "--lpips-device", default = "cpu", help = "device for the LPIPS net (keep off the measured GPU)"
    )
    p.add_argument("--out-dir", default = "outputs/quant_accuracy")
    # e2e-only
    p.add_argument("--e2e-model", default = None, help = "full model repo for --mode e2e")
    p.add_argument(
        "--e2e-components",
        nargs = "+",
        default = ["transformer", "text_encoder", "vae"],
        choices = ["transformer", "text_encoder", "vae"],
        help = "which components to auto-quantise for the e2e (isolate the VAE with: --e2e-components vae)",
    )
    p.add_argument("--prompts", nargs = "*", default = None)
    p.add_argument("--seeds", nargs = "*", type = int, default = [12345])
    p.add_argument("--steps", type = int, default = 8)
    p.add_argument("--guidance", type = float, default = None)
    p.add_argument("--height", type = int, default = 1024)
    p.add_argument("--width", type = int, default = 1024)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    _check_deps()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents = True, exist_ok = True)
    lp = _Lpips(args.lpips_device)
    if args.mode == "vae":
        rows = _sweep_vae(args, lp, out_dir)
    else:
        rows = _sweep_e2e(args, lp, out_dir)
    _write(out_dir, args.mode, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
