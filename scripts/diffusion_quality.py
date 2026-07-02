# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image quality-vs-quant harness for the Studio diffusion backend.

The accuracy analogue of the KLD workflow: hold the prompt + seed fixed, render a
grid with a high-fidelity reference quant (default BF16), then render the same grid
with each candidate quant and measure how far the output drifts from the reference.
For every quant it records mean PSNR / SSIM (pixel + structural fidelity vs the
reference image), optional CLIP scores (perceptual: prompt alignment + similarity to
the reference), plus file size, generation latency, and peak VRAM. It then prints a
quality-vs-cost table and recommends the smallest quant that stays within a quality
budget, so "retain accuracy" becomes a number you can set defaults from.

Lean by design: PSNR + SSIM are pure numpy (no skimage/scipy); CLIP is optional and
gated on ``--clip`` (uses transformers, downloads a small CLIP once). torch /
diffusers / the backend are imported lazily so ``--help`` and ``--selftest`` work on
a host without them. Not part of CPU CI for the GPU path; ``--selftest`` is CPU-only.

Examples:
    # CPU metric sanity check (no GPU, no model):
    python scripts/diffusion_quality.py --selftest

    # GPU sweep of a few quants against the BF16 reference:
    python scripts/diffusion_quality.py --model unsloth/Z-Image-Turbo-GGUF \\
        --reference-quant z-image-turbo-BF16.gguf \\
        --quants z-image-turbo-Q8_0.gguf z-image-turbo-Q4_K_M.gguf z-image-turbo-Q2_K.gguf \\
        --clip --out-dir outputs/diffusion_quality/zimage
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

_BACKEND_ROOT = Path(__file__).resolve().parent.parent / "studio" / "backend"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

DEFAULT_PROMPTS = [
    "A cozy reading nook by a rain-streaked window, warm lamplight, a cat asleep on a stack of books",
    "A lone lighthouse on a rocky cliff at sunset, dramatic clouds, crashing waves, highly detailed",
    "A bustling night market street in the rain, neon signs reflected in puddles, cinematic",
]


# ── image metrics (pure numpy) ───────────────────────────────────────────────


def _to_gray(img: Any) -> Any:
    import numpy as np
    return np.asarray(img.convert("L"), dtype = np.float64)


def _to_rgb(path_or_img: Any) -> Any:
    import numpy as np
    from PIL import Image

    img = path_or_img if hasattr(path_or_img, "convert") else Image.open(path_or_img)
    return np.asarray(img.convert("RGB"), dtype = np.float64)


# Finite PSNR (dB) a perfect (inf) sample is capped to when averaged with imperfect ones,
# so a lossless render counts as excellent without hiding diverged samples. Well above the
# ~37 dB compile and ~21 dB quant noise floors this harness reports.
_PERFECT_MATCH_PSNR = 100.0


def psnr(a_img: Any, b_img: Any) -> float:
    """PSNR (dB) between two images; inf when identical, 0 when shapes differ."""
    a, b = _to_rgb(a_img), _to_rgb(b_img)
    if a.shape != b.shape:
        return 0.0
    mse = float(((a - b) ** 2).mean())
    if mse == 0.0:
        return math.inf
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


def _box_mean(x: Any, w: int) -> Any:
    """Uniform (w x w) box mean over a 2D array via an integral image; edge-padded
    so the output keeps the input shape. Vectorised, no python loop."""
    import numpy as np

    r = w // 2
    xp = np.pad(x, r, mode = "edge")
    ii = np.cumsum(np.cumsum(xp, axis = 0), axis = 1)
    ii = np.pad(ii, ((1, 0), (1, 0)), mode = "constant")
    h, wd = x.shape
    total = ii[w : h + w, w : wd + w] - ii[0:h, w : wd + w] - ii[w : h + w, 0:wd] + ii[0:h, 0:wd]
    return total / float(w * w)


def ssim(
    a_img: Any,
    b_img: Any,
    window: int = 7,
) -> float:
    """Mean structural similarity (luminance) over a uniform window; 1.0 when
    identical. Pure numpy box-window SSIM (Wang et al. constants), no skimage."""
    a, b = _to_gray(a_img), _to_gray(b_img)
    if a.shape != b.shape:
        return 0.0
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = _box_mean(a, window), _box_mean(b, window)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = _box_mean(a * a, window) - mu_a2
    var_b = _box_mean(b * b, window) - mu_b2
    cov_ab = _box_mean(a * b, window) - mu_ab
    ssim_map = ((2 * mu_ab + c1) * (2 * cov_ab + c2)) / (
        (mu_a2 + mu_b2 + c1) * (var_a + var_b + c2)
    )
    return float(ssim_map.mean())


# ── optional CLIP (perceptual) ───────────────────────────────────────────────


class _Clip:
    """Lazy CLIP scorer: prompt-image alignment + image-image cosine similarity."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32") -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)

    def _image_embed(self, img: Any) -> Any:
        inputs = self.proc(images = img.convert("RGB"), return_tensors = "pt").to(self.device)
        with self.torch.no_grad():
            emb = self.model.get_image_features(**inputs)
        return emb / emb.norm(dim = -1, keepdim = True)

    def _text_embed(self, text: str) -> Any:
        inputs = self.proc(text = [text], return_tensors = "pt", padding = True, truncation = True).to(
            self.device
        )
        with self.torch.no_grad():
            emb = self.model.get_text_features(**inputs)
        return emb / emb.norm(dim = -1, keepdim = True)

    def text_score(self, img: Any, prompt: str) -> float:
        return float((self._image_embed(img) * self._text_embed(prompt)).sum().item())

    def image_similarity(self, img: Any, ref_img: Any) -> float:
        return float((self._image_embed(img) * self._image_embed(ref_img)).sum().item())


# ── GPU measurement helpers (mirrors diffusion_bench) ─────────────────────────


def _cuda(call: str) -> Optional[int]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return int(getattr(torch.cuda, call)())
    except Exception:
        return None


def _cuda_reset_peak() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    except Exception:
        pass


def _wait_for_load(backend: Any, timeout_s: int = 3600) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        p = backend.load_progress()
        if p.get("phase") == "ready":
            return
        if p.get("phase") == "error":
            raise RuntimeError(f"load error: {p.get('error')}")
        time.sleep(2)
    raise TimeoutError("model load did not reach ready")


def _hf_file_size_mib(repo: str, filename: str) -> Optional[int]:
    # A local model dir / file: stat it directly. The Hub lookup below returns None for
    # a local path, which would drop every candidate from _recommend (file_size_mib None).
    try:
        local = Path(repo).expanduser()
        if local.is_dir():
            f = local / filename
            if f.is_file():
                return int(f.stat().st_size // (1024 * 1024))
        elif local.is_file():
            return int(local.stat().st_size // (1024 * 1024))
    except Exception:
        pass
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, files_metadata = True, token = os.environ.get("HF_TOKEN"))
        for s in info.siblings:
            if s.rfilename == filename and s.size:
                return int(s.size // (1024 * 1024))
    except Exception:
        return None
    return None


# ── one quant: load, render the grid, measure ────────────────────────────────


def _render_grid(
    backend: Any, args: argparse.Namespace, gguf: str, out_dir: Path
) -> dict[str, Any]:
    """Load ``gguf`` and render one image per (prompt, seed); return images keyed by
    (prompt_index, seed) plus latency / VRAM metrics."""
    _cuda_reset_peak()
    backend.begin_load(
        args.model,
        gguf_filename = gguf,
        base_repo = args.base_repo,
        family_override = args.family_override,
        hf_token = os.environ.get("HF_TOKEN"),
        memory_mode = args.memory_mode,
    )
    _wait_for_load(backend)
    status = backend.status()

    images: dict[tuple, Any] = {}
    latencies: list[float] = []
    _cuda_reset_peak()
    quant_dir = out_dir / gguf.replace("/", "_")
    quant_dir.mkdir(parents = True, exist_ok = True)
    for pi, prompt in enumerate(args.prompts):
        for seed in args.seeds:
            t0 = time.time()
            result = backend.generate(
                prompt = prompt,
                width = args.width,
                height = args.height,
                steps = args.steps,
                guidance = args.guidance,
                seed = seed,
                batch_size = 1,
            )
            latencies.append(time.time() - t0)
            img = result["images"][0]
            images[(pi, seed)] = img
            img.save(quant_dir / f"p{pi}_s{seed}.png")
    try:
        backend.unload()
    except Exception:
        pass

    latencies.sort()
    return {
        "images": images,
        "status": status,
        "median_latency_s": round(latencies[len(latencies) // 2], 4) if latencies else None,
        "peak_vram_bytes": _cuda("max_memory_allocated"),
        "file_size_mib": _hf_file_size_mib(args.model, gguf),
    }


def _compare(
    grid: dict, ref_grid: dict, clip: Optional[_Clip], prompts: list[str]
) -> dict[str, Any]:
    psnrs, ssims, clip_txt, clip_sim = [], [], [], []
    for key, img in grid["images"].items():
        ref = ref_grid["images"].get(key)
        if ref is None:
            continue
        psnrs.append(psnr(img, ref))
        ssims.append(ssim(img, ref))
        if clip is not None:
            clip_txt.append(clip.text_score(img, prompts[key[0]]))
            clip_sim.append(clip.image_similarity(img, ref))

    def _mean(xs: list[float]) -> Optional[float]:
        # +inf marks an identical render (reference vs itself, or a lossless quant/offload)
        # scoring PSNR=inf -- the case this harness verifies. Report inf ONLY when every
        # sample is inf; a mix of inf and finite means some renders diverged, so a bare inf
        # would mask those bad samples. Cap the perfect ones to a high finite PSNR and
        # average so the drift still shows. (Only PSNR is ever inf; SSIM/CLIP stay finite.)
        if not xs:
            return None
        if all(x == math.inf for x in xs):
            return math.inf
        vals = [
            _PERFECT_MATCH_PSNR if x == math.inf else x for x in xs if math.isfinite(x) or x == math.inf
        ]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "mean_psnr": _mean(psnrs),
        "mean_ssim": _mean(ssims),
        "mean_clip_text": _mean(clip_txt) if clip is not None else None,
        "mean_clip_sim": _mean(clip_sim) if clip is not None else None,
    }


# ── sweep ─────────────────────────────────────────────────────────────────────


def _sweep(args: argparse.Namespace) -> int:
    from core.inference.diffusion import get_diffusion_backend

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents = True, exist_ok = True)
    backend = get_diffusion_backend()
    clip = _Clip() if args.clip else None

    print(f"=== reference: {args.reference_quant} ===", flush = True)
    ref_grid = _render_grid(backend, args, args.reference_quant, out_dir)

    quants = [args.reference_quant] + [q for q in args.quants if q != args.reference_quant]
    rows: list[dict[str, Any]] = []
    for gguf in quants:
        print(f"=== quant: {gguf} ===", flush = True)
        grid = (
            gguf == args.reference_quant and ref_grid or _render_grid(backend, args, gguf, out_dir)
        )
        metrics = _compare(grid, ref_grid, clip, args.prompts)
        rows.append(
            {
                "quant": gguf,
                "file_size_mib": grid["file_size_mib"],
                "is_reference": gguf == args.reference_quant,
                "median_latency_s": grid["median_latency_s"],
                "peak_vram_mib": (grid["peak_vram_bytes"] or 0) // (1024 * 1024) or None,
                **metrics,
            }
        )
        print(f"  {metrics}", flush = True)

    _write_outputs(args, out_dir, rows)
    _print_table(rows)
    _recommend(args, rows)
    return 0


def _write_outputs(args: argparse.Namespace, out_dir: Path, rows: list[dict]) -> None:
    (out_dir / "quality.json").write_text(
        json.dumps(
            {
                "config": {
                    "model": args.model,
                    "reference_quant": args.reference_quant,
                    "prompts": args.prompts,
                    "seeds": args.seeds,
                    "steps": args.steps,
                    "width": args.width,
                    "height": args.height,
                    "guidance": args.guidance,
                    "memory_mode": args.memory_mode,
                    "clip": args.clip,
                },
                "rows": rows,
            },
            indent = 2,
        )
    )
    fields = [
        "quant",
        "file_size_mib",
        "peak_vram_mib",
        "median_latency_s",
        "mean_psnr",
        "mean_ssim",
        "mean_clip_text",
        "mean_clip_sim",
    ]
    with (out_dir / "quality.csv").open("w", newline = "") as fh:
        writer = csv.DictWriter(fh, fieldnames = fields, extrasaction = "ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\n  wrote {out_dir / 'quality.csv'} and quality.json", flush = True)


def _print_table(rows: list[dict]) -> None:
    print(
        "\n=== QUALITY vs QUANT (lower size/latency/VRAM better; higher PSNR/SSIM/CLIP better) ===",
        flush = True,
    )
    hdr = f"  {'quant':<28}{'size_MB':>9}{'vram_MB':>9}{'lat_s':>8}{'PSNR':>8}{'SSIM':>8}{'CLIPt':>8}{'CLIPs':>8}"
    print(hdr, flush = True)
    for r in rows:

        def _f(v, fmt):
            return format(v, fmt) if isinstance(v, (int, float)) else "-"

        psnr_str = "inf" if r.get("mean_psnr") == math.inf else _f(r.get("mean_psnr"), ".2f")
        print(
            f"  {r['quant']:<28}{_f(r.get('file_size_mib'), '>9'):>9}"
            f"{_f(r.get('peak_vram_mib'), '>9'):>9}{_f(r.get('median_latency_s'), '>8.2f'):>8}"
            f"{psnr_str:>8}{_f(r.get('mean_ssim'), '>8.4f'):>8}"
            f"{_f(r.get('mean_clip_text'), '>8.4f'):>8}{_f(r.get('mean_clip_sim'), '>8.4f'):>8}",
            flush = True,
        )


def _recommend(args: argparse.Namespace, rows: list[dict]) -> None:
    # The smallest-on-disk non-reference quant that stays within the quality budget.
    passing = [
        r
        for r in rows
        if not r["is_reference"]
        and r.get("mean_ssim") is not None
        and r["mean_ssim"] >= args.ssim_threshold
        and (r.get("mean_psnr") is None or r["mean_psnr"] >= args.psnr_threshold)
        and r.get("file_size_mib") is not None
    ]
    print("\n=== RECOMMENDATION ===", flush = True)
    print(f"  budget: SSIM >= {args.ssim_threshold}, PSNR >= {args.psnr_threshold} dB", flush = True)
    if not passing:
        print("  no candidate quant met the quality budget; keep the reference quant.", flush = True)
        return
    best = min(passing, key = lambda r: r["file_size_mib"])
    print(
        f"  smallest quant within budget: {best['quant']} "
        f"({best['file_size_mib']} MB, SSIM {best['mean_ssim']}, PSNR "
        f"{'inf' if best['mean_psnr'] == math.inf else best['mean_psnr']})",
        flush = True,
    )


# ── self-test (CPU, no GPU/model) ─────────────────────────────────────────────


def _selftest() -> int:
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (128, 128, 3), dtype = np.uint8)
    a = Image.fromarray(base)
    b = Image.fromarray(base)  # identical
    noisy = Image.fromarray(
        np.clip(base.astype(int) + rng.integers(-40, 40, base.shape), 0, 255).astype(np.uint8)
    )

    checks = []
    checks.append(("identical PSNR is inf", psnr(a, b) == math.inf))
    checks.append(("identical SSIM ~ 1.0", abs(ssim(a, b) - 1.0) < 1e-9))
    checks.append(
        ("noisy PSNR is finite + lower", math.isfinite(psnr(a, noisy)) and psnr(a, noisy) < 60)
    )
    checks.append(("noisy SSIM < identical", ssim(a, noisy) < ssim(a, b)))
    checks.append(("shape mismatch -> 0", psnr(a, Image.fromarray(base[:64])) == 0.0))
    # box mean of a constant field equals the constant
    const = np.full((32, 32), 7.0)
    checks.append(
        ("box mean of constant is constant", abs(_box_mean(const, 7).mean() - 7.0) < 1e-9)
    )

    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}", flush = True)
        ok = ok and passed
    print("SELFTEST OK" if ok else "SELFTEST FAILED", flush = True)
    return 0 if ok else 1


# ── cli ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description = "Image quality-vs-quant harness for the Studio diffusion backend.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model", default = "unsloth/Z-Image-Turbo-GGUF", help = "GGUF repo id or local path"
    )
    p.add_argument(
        "--reference-quant",
        default = "z-image-turbo-BF16.gguf",
        help = "high-fidelity reference GGUF filename",
    )
    p.add_argument(
        "--quants",
        nargs = "*",
        default = [
            "z-image-turbo-Q8_0.gguf",
            "z-image-turbo-Q4_K_M.gguf",
            "z-image-turbo-Q2_K.gguf",
        ],
        help = "candidate GGUF filenames to score against the reference",
    )
    p.add_argument("--base-repo", default = None)
    p.add_argument("--family-override", default = None)
    p.add_argument("--prompts", nargs = "*", default = DEFAULT_PROMPTS)
    p.add_argument("--seeds", nargs = "*", type = int, default = [12345])
    p.add_argument("--width", type = int, default = 1024)
    p.add_argument("--height", type = int, default = 1024)
    p.add_argument("--steps", type = int, default = 9)
    p.add_argument("--guidance", type = float, default = 0.0)
    p.add_argument("--memory-mode", default = None, choices = ["auto", "fast", "balanced", "low_vram"])
    p.add_argument("--clip", action = "store_true", help = "also compute CLIP text + image scores")
    p.add_argument(
        "--psnr-threshold",
        type = float,
        default = 30.0,
        help = "min mean PSNR (dB) vs reference for the recommendation",
    )
    p.add_argument(
        "--ssim-threshold",
        type = float,
        default = 0.92,
        help = "min mean SSIM vs reference for the recommendation",
    )
    p.add_argument("--out-dir", default = "outputs/diffusion_quality")
    p.add_argument("--selftest", action = "store_true", help = "CPU metric sanity check; no GPU/model")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.selftest:
        return _selftest()
    return _sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
