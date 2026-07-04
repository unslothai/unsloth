# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Video quality-vs-cost harness for the Studio video backend.

The video analogue of scripts/diffusion_quality.py: hold the prompt + seed +
shape fixed, render one clip with a high-fidelity reference configuration
(default the family's BF16 artifact), then render the same clip with each
candidate configuration (a GGUF quant, a dense torchao quant, a speed profile,
a step cache) and measure how far the output drifts from the reference.

Per candidate it reports:
- mean PSNR / SSIM over evenly sampled frames (pixel + structural fidelity),
- a temporal-consistency deviation: the relative error between the reference's
  and the candidate's frame-to-frame motion-energy series, which catches
  flicker/juddering that per-frame SSIM alone can miss,
- black-frame and NaN collapse checks (the failure mode quant bugs actually
  produce, per the image backend's qwen fp8 incident),
- an audio check for families that generate sound (LTX-2): RMS ratio vs the
  reference and a silence trip-wire,
- wall time per generate and peak VRAM.

Verdict bands map the standing accuracy budget: a candidate that keeps mean
SSIM at or above 0.75 PASSes (a ~25 percent structural drift is acceptable for
a large speed/memory win), 0.50-0.75 WARNs, and below 0.50 or any black/NaN/
silence collapse FAILs regardless of how fast it is.

Runtime-budgeted: ONE short clip per candidate (default 33 frames at 480p-class
sizes) so a full family sweep stays in minutes, not hours. Metrics are pure
numpy; torch / diffusers / the backend load lazily so --help and --selftest run
on a host without them.

Examples:
    # CPU metric sanity check (no GPU, no model):
    python scripts/video_quality.py --selftest

    # LTX-2.3 GGUF quants against the BF16 GGUF reference:
    CUDA_VISIBLE_DEVICES=1 python scripts/video_quality.py \\
        --model unsloth/LTX-2.3-GGUF --model-kind gguf \\
        --reference "gguf_filename=distilled-1.1/ltx-2.3-22b-distilled-1.1-BF16.gguf" \\
        --candidates "gguf_filename=distilled-1.1/ltx-2.3-22b-distilled-1.1-Q8_0.gguf" \\
                     "gguf_filename=distilled-1.1/ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf" \\
        --steps 8 --guidance 1.0 --out-dir outputs/video_quality/ltx23

    # Wan2.2-5B dense int8 + speed profiles against plain bf16:
    CUDA_VISIBLE_DEVICES=1 python scripts/video_quality.py \\
        --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \\
        --reference "" \\
        --candidates "transformer_quant=int8" "speed_mode=max" \\
        --steps 20 --out-dir outputs/video_quality/wan5b
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

_BACKEND_ROOT = Path(__file__).resolve().parent.parent / "studio" / "backend"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

DEFAULT_PROMPT = (
    "a golden retriever puppy runs through shallow ocean waves at sunset, "
    "splashing water, cinematic, camera tracking sideways"
)

# Finite PSNR (dB) an identical clip is capped to when averaged, matching
# scripts/diffusion_quality.py.
_PERFECT_MATCH_PSNR = 100.0


# ── frame metrics (pure numpy; frames are uint8 HxWx3 arrays) ────────────────


def _gray(frame: Any) -> Any:
    import numpy as np
    f = np.asarray(frame, dtype = np.float64)
    return f @ np.array([0.299, 0.587, 0.114])


def frame_psnr(a: Any, b: Any) -> float:
    import numpy as np

    a64 = np.asarray(a, dtype = np.float64)
    b64 = np.asarray(b, dtype = np.float64)
    if a64.shape != b64.shape:
        return 0.0
    mse = float(((a64 - b64) ** 2).mean())
    if mse == 0.0:
        return math.inf
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


def _box_mean(x: Any, w: int) -> Any:
    import numpy as np

    r = w // 2
    xp = np.pad(x, r, mode = "edge")
    ii = np.cumsum(np.cumsum(xp, axis = 0), axis = 1)
    ii = np.pad(ii, ((1, 0), (1, 0)), mode = "constant")
    h, wd = x.shape
    total = ii[w : h + w, w : wd + w] - ii[0:h, w : wd + w] - ii[w : h + w, 0:wd] + ii[0:h, 0:wd]
    return total / float(w * w)


def frame_ssim(
    a: Any,
    b: Any,
    window: int = 7,
) -> float:
    """Pure numpy box-window SSIM on luminance (Wang et al. constants); identical
    math to scripts/diffusion_quality.py so image and video budgets compare."""
    ga, gb = _gray(a), _gray(b)
    if ga.shape != gb.shape:
        return 0.0
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = _box_mean(ga, window), _box_mean(gb, window)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = _box_mean(ga * ga, window) - mu_a2
    var_b = _box_mean(gb * gb, window) - mu_b2
    cov_ab = _box_mean(ga * gb, window) - mu_ab
    ssim_map = ((2 * mu_ab + c1) * (2 * cov_ab + c2)) / (
        (mu_a2 + mu_b2 + c1) * (var_a + var_b + c2)
    )
    return float(ssim_map.mean())


def motion_energy(frames: Any) -> list[float]:
    """Mean absolute frame-to-frame luminance difference, one value per frame
    transition. The temporal signature of the clip: flicker inflates it, frozen
    or smeared motion deflates it."""
    import numpy as np

    grays = [_gray(f) for f in frames]
    return [float(np.abs(grays[i + 1] - grays[i]).mean()) for i in range(len(grays) - 1)]


def temporal_deviation(ref_frames: Any, cand_frames: Any) -> float:
    """Relative L1 error between the two motion-energy series (0 = identical
    temporal behaviour). Series lengths must match (same frame count)."""
    ref_series = motion_energy(ref_frames)
    cand_series = motion_energy(cand_frames)
    if len(ref_series) != len(cand_series) or not ref_series:
        return math.inf
    denom = sum(abs(v) for v in ref_series) + 1e-6
    return sum(abs(r - c) for r, c in zip(ref_series, cand_series)) / denom


def clip_metrics(
    ref_frames: Any,
    cand_frames: Any,
    sample_count: int = 5,
) -> dict[str, Any]:
    """All frame metrics for one candidate clip vs the reference clip."""
    import numpy as np

    n = min(len(ref_frames), len(cand_frames))
    idx = sorted({int(round(i * (n - 1) / max(1, sample_count - 1))) for i in range(sample_count)})
    psnrs = [min(frame_psnr(ref_frames[i], cand_frames[i]), _PERFECT_MATCH_PSNR) for i in idx]
    ssims = [frame_ssim(ref_frames[i], cand_frames[i]) for i in idx]
    lumas = [float(_gray(cand_frames[i]).mean() / 255.0) for i in idx]
    has_nan = any(
        bool(np.isnan(np.asarray(f, dtype = np.float64)).any()) for f in (cand_frames[i] for i in idx)
    )
    return {
        "frames_compared": len(idx),
        "psnr_mean": sum(psnrs) / len(psnrs),
        "ssim_mean": sum(ssims) / len(ssims),
        "temporal_deviation": temporal_deviation(ref_frames[:n], cand_frames[:n]),
        "min_luma": min(lumas),
        "has_nan": has_nan,
    }


def audio_metrics(ref_audio: Optional[Any], cand_audio: Optional[Any]) -> dict[str, Any]:
    """RMS comparison for families with sound. None audio on both sides is fine;
    losing the track (or emitting silence) when the reference has one is not."""
    import numpy as np

    def _rms(a: Any) -> Optional[float]:
        if a is None:
            return None
        arr = np.asarray(a, dtype = np.float64)
        return float(np.sqrt((arr**2).mean())) if arr.size else 0.0

    ref_rms, cand_rms = _rms(ref_audio), _rms(cand_audio)
    silent_collapse = (
        ref_rms is not None and ref_rms >= 1e-3 and (cand_rms is None or cand_rms < 1e-4)
    )
    return {"ref_rms": ref_rms, "cand_rms": cand_rms, "silent_collapse": silent_collapse}


def verdict(metrics: dict[str, Any], audio: dict[str, Any]) -> str:
    """PASS / WARN / FAIL per the standing accuracy budget (~25 percent structural
    drift acceptable, 50 percent or a collapse never)."""
    if metrics["has_nan"] or metrics["min_luma"] < 0.02 or audio.get("silent_collapse"):
        return "FAIL"
    if metrics["ssim_mean"] < 0.50 or metrics["temporal_deviation"] > 1.0:
        return "FAIL"
    if metrics["ssim_mean"] < 0.75 or metrics["temporal_deviation"] > 0.5:
        return "WARN"
    return "PASS"


# ── mp4 decode (PyAV, same dependency the backend encodes with) ─────────────


def decode_mp4(mp4_bytes: bytes, workdir: Path, name: str) -> tuple[list[Any], Optional[Any]]:
    """Frames (uint8 arrays) + mono audio samples (float array or None) from bytes."""
    import av
    import numpy as np

    path = workdir / f"{name}.mp4"
    path.write_bytes(mp4_bytes)
    container = av.open(str(path))
    frames = [f.to_ndarray(format = "rgb24") for f in container.decode(container.streams.video[0])]
    audio = None
    if container.streams.audio:
        container.close()
        container = av.open(str(path))
        chunks = [c.to_ndarray() for c in container.decode(container.streams.audio[0])]
        if chunks:
            audio = np.concatenate([c.reshape(c.shape[0], -1).mean(axis = 0) for c in chunks])
    container.close()
    return frames, audio


# ── configuration plumbing ───────────────────────────────────────────────────


def parse_spec(spec: str) -> dict[str, str]:
    """'k=v;k=v' (or space-free 'k=v,k=v') -> dict; empty string -> {} (pure base)."""
    out: dict[str, str] = {}
    for part in spec.replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad candidate spec fragment '{part}' (expected key=value)")
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def spec_label(spec: dict[str, str]) -> str:
    if not spec:
        return "base"
    return ",".join(
        f"{k}={Path(v).name if k == 'gguf_filename' else v}" for k, v in sorted(spec.items())
    )


def run_config(
    backend: Any, args: Any, spec: dict[str, str], workdir: Path, name: str
) -> dict[str, Any]:
    """Load per spec, generate the fixed clip, unload. Returns frames/audio/cost."""
    import torch

    load_kwargs: dict[str, Any] = {
        "gguf_filename": spec.get("gguf_filename"),
        "model_kind": spec.get("model_kind", args.model_kind),
        "memory_mode": spec.get("memory_mode"),
        "speed_mode": spec.get("speed_mode"),
        "attention_backend": spec.get("attention_backend"),
        "transformer_cache": spec.get("transformer_cache"),
        "transformer_quant": spec.get("transformer_quant"),
    }
    t0 = time.monotonic()
    status = backend.load_pipeline(args.model, **load_kwargs)
    load_s = time.monotonic() - t0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    result = backend.generate(
        prompt = args.prompt,
        width = args.width,
        height = args.height,
        num_frames = args.frames,
        fps = args.fps,
        steps = args.steps,
        guidance = args.guidance,
        seed = args.seed,
    )
    generate_s = time.monotonic() - t0
    peak_gib = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    backend.unload()
    frames, audio = decode_mp4(result["mp4_bytes"], workdir, name)
    return {
        "frames": frames,
        "audio": audio,
        "load_s": round(load_s, 1),
        "generate_s": round(generate_s, 1),
        "peak_vram_gib": round(peak_gib, 2),
        "resolved": {
            k: v
            for k, v in status.items()
            if k
            in (
                "speed_mode",
                "attention_backend",
                "transformer_cache",
                "transformer_quant",
                "offload_policy",
                "model_kind",
            )
        },
    }


def run_gate(args: Any) -> int:
    from core.inference.video import get_video_backend

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)
    backend = get_video_backend()

    print(f"reference: {args.reference or 'base'}", flush = True)
    ref = run_config(backend, args, parse_spec(args.reference), out_dir, "reference")
    print(
        f"  load {ref['load_s']}s, generate {ref['generate_s']}s, "
        f"peak {ref['peak_vram_gib']} GiB",
        flush = True,
    )

    rows = []
    for spec_str in args.candidates:
        spec = parse_spec(spec_str)
        label = spec_label(spec)
        print(f"candidate: {label}", flush = True)
        cand = run_config(backend, args, spec, out_dir, label.replace("/", "_").replace("=", "-"))
        metrics = clip_metrics(ref["frames"], cand["frames"], sample_count = args.sample_frames)
        audio = audio_metrics(ref["audio"], cand["audio"])
        row = {
            "candidate": label,
            **{
                k: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else v)
                for k, v in metrics.items()
            },
            **{f"audio_{k}": v for k, v in audio.items()},
            "load_s": cand["load_s"],
            "generate_s": cand["generate_s"],
            "ref_generate_s": ref["generate_s"],
            "peak_vram_gib": cand["peak_vram_gib"],
            "resolved": cand["resolved"],
            "verdict": verdict(metrics, audio),
        }
        rows.append(row)
        print(
            f"  ssim {row['ssim_mean']:.3f} | psnr {row['psnr_mean']:.1f} dB | "
            f"temporal {row['temporal_deviation']:.3f} | luma>={row['min_luma']:.3f} | "
            f"gen {row['generate_s']}s (ref {ref['generate_s']}s) | "
            f"vram {row['peak_vram_gib']} GiB | {row['verdict']}",
            flush = True,
        )

    report = {
        "model": args.model,
        "reference": args.reference or "base",
        "prompt": args.prompt,
        "shape": [args.width, args.height, args.frames, args.fps],
        "steps": args.steps,
        "guidance": args.guidance,
        "seed": args.seed,
        "reference_cost": {k: ref[k] for k in ("load_s", "generate_s", "peak_vram_gib")},
        "candidates": rows,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent = 1))
    print(f"report: {out_dir / 'report.json'}", flush = True)
    return 0 if all(r["verdict"] != "FAIL" for r in rows) else 1


# ── selftest (CPU-only, synthetic clips, no torch/model) ────────────────────


def selftest() -> int:
    import numpy as np

    rng = np.random.default_rng(0)
    h, w, n = 64, 96, 12

    def make_clip(
        offset = 0.0,
        noise = 0.0,
        black = False,
    ):
        frames = []
        for t in range(n):
            x = np.linspace(0, 1, w)[None, :] + t * 0.05 + offset
            base = (np.sin(x * 6.283) * 0.5 + 0.5) * 255.0
            frame = np.repeat(base[..., None], 3, axis = 2) * np.ones((h, 1, 1))
            if noise:
                frame = frame + rng.normal(0, noise, frame.shape)
            if black:
                frame = frame * 0.0
            frames.append(np.clip(frame, 0, 255).astype(np.uint8))
        return frames

    ref = make_clip()
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS: " if cond else "FAIL: ") + msg)
        ok = ok and cond

    same = clip_metrics(ref, make_clip())
    check(
        same["ssim_mean"] > 0.99 and same["temporal_deviation"] < 0.01,
        f"identical clip scores ~1 (ssim {same['ssim_mean']:.3f})",
    )
    check(verdict(same, {"silent_collapse": False}) == "PASS", "identical clip verdict PASS")

    noisy = clip_metrics(ref, make_clip(noise = 12.0))
    check(0.3 < noisy["ssim_mean"] < 0.99, f"noisy clip degrades ssim ({noisy['ssim_mean']:.3f})")

    black = clip_metrics(ref, make_clip(black = True))
    check(
        verdict(black, {"silent_collapse": False}) == "FAIL",
        f"black clip verdict FAIL (min_luma {black['min_luma']:.3f})",
    )

    shifted = clip_metrics(ref, make_clip(offset = 0.5))
    check(shifted["ssim_mean"] < same["ssim_mean"], "content shift lowers ssim")

    audio = audio_metrics(np.sin(np.linspace(0, 100, 16000)), np.zeros(16000))
    check(audio["silent_collapse"] is True, "silent audio collapse detected")
    audio_ok = audio_metrics(
        np.sin(np.linspace(0, 100, 16000)), np.sin(np.linspace(0, 100, 16000)) * 0.8
    )
    check(audio_ok["silent_collapse"] is False, "attenuated audio is not a collapse")

    print("VIDEO-QUALITY-SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description = __doc__.split("\n")[0])
    parser.add_argument("--selftest", action = "store_true", help = "CPU metric sanity check")
    parser.add_argument("--model", help = "Repo id handed to the video backend")
    parser.add_argument("--model-kind", default = None, help = "pipeline | gguf | single_file")
    parser.add_argument(
        "--reference", default = "", help = "Reference spec 'k=v;k=v' ('' = plain base load)"
    )
    parser.add_argument("--candidates", nargs = "+", default = [], help = "Candidate specs 'k=v;k=v'")
    parser.add_argument("--prompt", default = DEFAULT_PROMPT)
    parser.add_argument("--width", type = int, default = 768)
    parser.add_argument("--height", type = int, default = 512)
    parser.add_argument("--frames", type = int, default = 33)
    parser.add_argument("--fps", type = int, default = 24)
    parser.add_argument("--steps", type = int, default = None)
    parser.add_argument("--guidance", type = float, default = None)
    parser.add_argument("--seed", type = int, default = 7)
    parser.add_argument("--sample-frames", type = int, default = 5)
    parser.add_argument("--out-dir", default = "outputs/video_quality")
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    if not args.model or not args.candidates:
        parser.error("--model and --candidates are required (or use --selftest)")
    return run_gate(args)


if __name__ == "__main__":
    sys.exit(main())
