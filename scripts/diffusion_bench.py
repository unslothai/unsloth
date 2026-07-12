# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Standalone GPU benchmark + regression harness for the Studio diffusion backend.

Drives ``DiffusionBackend`` directly (no HTTP server) to measure load time, peak
VRAM, and generation latency for a single GGUF image model, plus an accuracy
guard: a fixed-seed image is rendered and compared (PSNR) against a stored
reference so a precision/dtype/guard regression that silently changes output is
caught, not just speed/memory.

Two modes:

  --write-baseline PATH   run once, save metrics JSON + reference.png next to it.
  --compare PATH          run again, diff against the baseline, exit nonzero if a
                          latency / VRAM / PSNR threshold is exceeded.

torch / diffusers are imported lazily (only after argument parsing and only
inside functions) so ``--help`` works on a host without them. Not part of CPU CI;
this needs a real GPU and a downloadable model.

Example:
    python scripts/diffusion_bench.py --write-baseline outputs/diffusion_bench/baseline.json \\
        --model unsloth/Z-Image-Turbo-GGUF --gguf z-image-turbo-Q4_K_M.gguf
    # ... make changes ...
    python scripts/diffusion_bench.py --compare outputs/diffusion_bench/baseline.json \\
        --model unsloth/Z-Image-Turbo-GGUF --gguf z-image-turbo-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# The backend package lives at unsloth/studio/backend; this file is at
# unsloth/scripts/diffusion_bench.py. Put the backend root on sys.path so
# ``core.inference.diffusion`` imports as the server does. (The backend import is deferred
# into main() so --help never triggers torch.)
_BACKEND_ROOT = Path(__file__).resolve().parent.parent / "studio" / "backend"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


# ── small helpers ──────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd = str(Path(__file__).resolve().parent),
            capture_output = True,
            text = True,
            timeout = 10,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile over a small sample (no numpy)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = int(math.ceil(pct / 100.0 * len(ordered))) - 1
    rank = max(0, min(rank, len(ordered) - 1))
    return ordered[rank]


def _is_cuda(device: Optional[str]) -> bool:
    return bool(device) and device.split(":", 1)[0] == "cuda"


def _cuda_reset_peak() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _cuda_sync() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cuda_peak_alloc() -> Optional[int]:
    import torch
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None


def _cuda_peak_reserved() -> Optional[int]:
    import torch
    return int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None


def _cuda_alloc() -> Optional[int]:
    import torch
    return int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else None


def _gpu_name() -> Optional[str]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def _versions() -> dict[str, Optional[str]]:
    out: dict[str, Optional[str]] = {"torch": None, "diffusers": None}
    try:
        import torch
        out["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import diffusers
        out["diffusers"] = diffusers.__version__
    except Exception:
        pass
    return out


def _psnr(ref_png: Path, cand_png: Path) -> float:
    """PSNR (dB) between two PNGs; inf when identical."""
    import numpy as np
    from PIL import Image

    with Image.open(ref_png) as im_a:
        a = np.asarray(im_a.convert("RGB"), dtype = np.float64)
    with Image.open(cand_png) as im_b:
        b = np.asarray(im_b.convert("RGB"), dtype = np.float64)
    if a.shape != b.shape:
        # Different geometry means the comparison is meaningless; report worst case.
        return 0.0
    mse = float(((a - b) ** 2).mean())
    if mse == 0.0:
        return math.inf
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


# ── load + generate ────────────────────────────────────────────────────────


def _wait_for_load(backend: Any, timeout_s: int = 2400) -> None:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        p = backend.load_progress()
        phase = p.get("phase")
        if phase != last:
            last = phase
            frac = p.get("fraction") or 0.0
            bt = (p.get("bytes_total") or 0) / 1e9
            print(f"  load phase={phase} frac={frac:.3f} total={bt:.2f}GB", flush = True)
        if phase == "ready":
            return
        if phase == "error":
            raise RuntimeError(f"load error: {p.get('error')}")
        time.sleep(2)
    raise TimeoutError(f"model load did not reach ready within {timeout_s}s")


def _generate_once(backend: Any, args: argparse.Namespace) -> Any:
    """One generation at the fixed seed; returns the first PIL image."""
    result = backend.generate(
        prompt = args.prompt,
        width = args.width,
        height = args.height,
        steps = args.steps,
        guidance = args.guidance,
        seed = args.seed,
        batch_size = args.batch_size,
    )
    images = result["images"]
    return images[0]


def _run(args: argparse.Namespace) -> dict[str, Any]:
    """Load the model, measure load + generation, render the fixed-seed image.

    Returns the metrics dict; writes the rendered image to ``args._image_out``.
    """
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status: dict[str, Any] = {}
    load_metrics: dict[str, Any] = {}
    gen_metrics: dict[str, Any] = {}

    try:
        # ── load ──
        _cuda_reset_peak()
        t0 = time.time()
        backend.begin_load(
            args.model,
            gguf_filename = args.gguf,
            base_repo = args.base_repo,
            family_override = args.family_override,
            hf_token = os.environ.get("HF_TOKEN"),
            cpu_offload = args.cpu_offload,
            memory_mode = args.memory_mode,
            speed_mode = args.speed_mode,
            text_encoder_quant = args.text_encoder_quant,
            transformer_quant = args.transformer_quant,
            transformer_quant_fast_accum = {"auto": None, "on": True, "off": False}[
                args.fp8_fast_accum
            ],
        )
        _wait_for_load(backend)
        _cuda_sync()
        load_metrics = {
            "wall_seconds": round(time.time() - t0, 2),
            "peak_vram_bytes": _cuda_peak_alloc(),
            "peak_reserved_bytes": _cuda_peak_reserved(),
            "final_vram_bytes": _cuda_alloc(),
        }
        status = backend.status()
        print(f"  loaded: {status}", flush = True)

        # ── warmup (discarded) ──
        for _ in range(max(0, args.warmup)):
            _generate_once(backend, args)

        # ── measured generations (fixed seed -> deterministic) ──
        _cuda_reset_peak()
        latencies: list[float] = []
        first_image = None
        for i in range(max(1, args.iters)):
            _cuda_sync()
            g0 = time.time()
            image = _generate_once(backend, args)
            _cuda_sync()
            latencies.append(time.time() - g0)
            if first_image is None:
                first_image = image
            print(f"  gen[{i}] {latencies[-1]:.3f}s", flush = True)

        total = sum(latencies)
        gen_metrics = {
            "iters": len(latencies),
            "warmup": max(0, args.warmup),
            "latencies_s": [round(x, 4) for x in latencies],
            "median_latency_s": round(_percentile(latencies, 50), 4),
            "p90_latency_s": round(_percentile(latencies, 90), 4),
            "images_per_sec": round((args.batch_size * len(latencies)) / total, 4)
            if total > 0
            else None,
            "peak_vram_bytes": _cuda_peak_alloc(),
        }

        # The fixed-seed image is the accuracy anchor.
        args._image_out.parent.mkdir(parents = True, exist_ok = True)
        first_image.save(args._image_out)
        print(f"  saved image -> {args._image_out}", flush = True)
    finally:
        try:
            backend.unload()
        except Exception as exc:  # noqa: BLE001 — best-effort cleanup
            print(f"  warn: unload failed: {exc}", flush = True)

    return {
        "env": {
            "timestamp": _now_iso(),
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": _versions(),
            "gpu_name": _gpu_name(),
            "status": status,
        },
        "load": load_metrics,
        "generate": gen_metrics,
        "config": {
            "model": args.model,
            "gguf": args.gguf,
            "base_repo": args.base_repo,
            "family_override": args.family_override,
            "prompt": args.prompt,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "guidance": args.guidance,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "memory_mode": args.memory_mode,
            "speed_mode": args.speed_mode,
            "cpu_offload": args.cpu_offload,
            "text_encoder_quant": args.text_encoder_quant,
            "transformer_quant": args.transformer_quant,
            "fp8_fast_accum": args.fp8_fast_accum,
        },
    }


# ── modes ──────────────────────────────────────────────────────────────────


def _write_baseline(args: argparse.Namespace) -> int:
    baseline_path = Path(args.write_baseline).resolve()
    ref_png = baseline_path.parent / "reference.png"
    args._image_out = ref_png

    metrics = _run(args)
    metrics["accuracy"] = {
        "reference_png": str(ref_png),
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "seed": args.seed,
        "dtype": (metrics["env"]["status"] or {}).get("dtype"),
    }

    baseline_path.parent.mkdir(parents = True, exist_ok = True)
    baseline_path.write_text(json.dumps(metrics, indent = 2))
    print("\n=== BASELINE WRITTEN ===", flush = True)
    print(f"  json:      {baseline_path}", flush = True)
    print(f"  reference: {ref_png}", flush = True)
    print(f"  load: {metrics['load']}", flush = True)
    print(
        f"  generate: median={metrics['generate'].get('median_latency_s')}s "
        f"p90={metrics['generate'].get('p90_latency_s')}s "
        f"img/s={metrics['generate'].get('images_per_sec')} "
        f"peak_vram={metrics['generate'].get('peak_vram_bytes')}",
        flush = True,
    )
    return 0


def _compare(args: argparse.Namespace) -> int:
    baseline_path = Path(args.compare).resolve()
    baseline = json.loads(baseline_path.read_text())
    out_dir = Path(args.out_dir).resolve()
    args._image_out = out_dir / "compare.png"

    # Refuse a noisy cross-hardware / cross-dtype comparison unless forced.
    base_env = baseline.get("env", {})
    base_status = base_env.get("status") or {}
    cur_gpu = _gpu_name()
    base_gpu = base_env.get("gpu_name")
    metrics = _run(args)
    cur_status = metrics["env"]["status"] or {}

    mismatch = []
    if base_gpu != cur_gpu:
        mismatch.append(f"gpu {base_gpu!r} -> {cur_gpu!r}")
    if base_status.get("device") != cur_status.get("device"):
        mismatch.append(f"device {base_status.get('device')!r} -> {cur_status.get('device')!r}")
    if base_status.get("dtype") != cur_status.get("dtype"):
        mismatch.append(f"dtype {base_status.get('dtype')!r} -> {cur_status.get('dtype')!r}")
    if mismatch:
        print("\n!! environment mismatch vs baseline: " + "; ".join(mismatch), flush = True)
        if not args.force_compare:
            print("   refusing noisy comparison (pass --force-compare to override).", flush = True)
            return 2

    # PSNR vs the stored reference image. The baseline stores an absolute reference_png, which
    # breaks if the baseline dir was copied/moved, so fall back to reference.png next to the
    # baseline JSON. A still-missing reference is a failure below, not a silent pass -- else the
    # benchmark reports PASS having done no image comparison.
    ref_png = Path(baseline.get("accuracy", {}).get("reference_png", ""))
    if not ref_png.is_file():
        ref_png = baseline_path.parent / "reference.png"
    psnr = _psnr(ref_png, args._image_out) if ref_png.is_file() else float("nan")

    base_gen = baseline.get("generate", {})
    cur_gen = metrics["generate"]
    base_median = base_gen.get("median_latency_s") or 0.0
    cur_median = cur_gen.get("median_latency_s") or 0.0
    latency_reg = (cur_median - base_median) / base_median if base_median > 0 else 0.0

    base_peak = base_gen.get("peak_vram_bytes")
    cur_peak = cur_gen.get("peak_vram_bytes")
    vram_reg = ((cur_peak - base_peak) / base_peak) if (base_peak and cur_peak) else 0.0

    print("\n=== REGRESSION REPORT ===", flush = True)
    print(f"  {'metric':<22}{'baseline':>16}{'current':>16}{'delta':>12}", flush = True)
    print(
        f"  {'median_latency_s':<22}{base_median:>16.4f}{cur_median:>16.4f}{latency_reg * 100:>11.1f}%",
        flush = True,
    )
    if base_peak and cur_peak:
        print(
            f"  {'peak_vram_MB':<22}{base_peak / 1e6:>16.1f}{cur_peak / 1e6:>16.1f}{vram_reg * 100:>11.1f}%",
            flush = True,
        )
    print(f"  {'psnr_dB(vs ref)':<22}{'-':>16}{psnr:>16.2f}{'':>12}", flush = True)

    failures = []
    if latency_reg > args.max_latency_regression:
        failures.append(
            f"latency +{latency_reg * 100:.1f}% > {args.max_latency_regression * 100:.0f}%"
        )
    if base_peak and cur_peak and vram_reg > args.max_vram_regression:
        failures.append(f"peak VRAM +{vram_reg * 100:.1f}% > {args.max_vram_regression * 100:.0f}%")
    if math.isnan(psnr):
        failures.append("PSNR reference image missing; cannot verify output quality")
    elif psnr < args.min_psnr:
        failures.append(f"PSNR {psnr:.2f}dB < {args.min_psnr:.1f}dB (output changed)")

    if failures:
        print("\n  FAIL: " + "; ".join(failures), flush = True)
        return 1
    print("\n  PASS: no regression beyond thresholds.", flush = True)
    return 0


# ── cli ────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description = "Benchmark + regression guard for the Studio diffusion backend.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model", default = "unsloth/Z-Image-Turbo-GGUF", help = "GGUF repo id or local path"
    )
    p.add_argument(
        "--gguf",
        default = "z-image-turbo-Q4_K_M.gguf",
        help = "transformer GGUF filename inside --model",
    )
    p.add_argument("--base-repo", default = None, help = "override the diffusers base repo")
    p.add_argument("--family-override", default = None, help = "force a diffusion family")
    p.add_argument(
        "--prompt",
        default = "A cozy reading nook by a rain-streaked window, warm lamplight, "
        "a cat asleep on a stack of books, highly detailed",
    )
    p.add_argument("--width", type = int, default = 1024)
    p.add_argument("--height", type = int, default = 1024)
    p.add_argument("--steps", type = int, default = 9)
    p.add_argument("--guidance", type = float, default = 0.0)
    p.add_argument("--seed", type = int, default = 12345, help = "fixed seed -> deterministic image")
    p.add_argument("--batch-size", type = int, default = 1)
    p.add_argument("--warmup", type = int, default = 1, help = "discarded warmup generations")
    p.add_argument("--iters", type = int, default = 3, help = "measured generations")
    p.add_argument(
        "--memory-mode",
        default = None,
        choices = ["auto", "fast", "balanced", "low_vram"],
        help = "memory policy (default: backend auto)",
    )
    p.add_argument(
        "--speed-mode",
        default = None,
        choices = ["off", "default", "max"],
        help = "speed profile: off is bit-identical; default adds compile + "
        "cudnn.benchmark (near-lossless); max also adds TF32 + fused QKV",
    )
    p.add_argument(
        "--text-encoder-quant",
        default = None,
        choices = ["fp8", "nvfp4"],
        help = "quantise the companion text encoder (fp8 or nvfp4)",
    )
    p.add_argument(
        "--transformer-quant",
        default = None,
        choices = ["auto", "int8", "fp8", "nvfp4", "mxfp8"],
        help = "opt-in fast transformer: load the DENSE bf16 transformer and torchao-"
        "quantise it onto the low-precision tensor cores (faster than GGUF, higher "
        "VRAM). auto picks per GPU; falls back to GGUF if unsupported / no VRAM",
    )
    p.add_argument(
        "--fp8-fast-accum",
        default = "auto",
        choices = ["auto", "on", "off"],
        help = "fp8 accumulate: auto picks by GPU class (fast on consumer, precise on "
        "data-center); on/off force it",
    )
    p.add_argument(
        "--cpu-offload", action = "store_true", help = "legacy: force whole-module CPU offload"
    )
    p.add_argument(
        "--write-baseline",
        metavar = "PATH",
        default = None,
        help = "run once and save metrics JSON + reference.png",
    )
    p.add_argument(
        "--compare", metavar = "PATH", default = None, help = "run again and diff against a baseline JSON"
    )
    p.add_argument(
        "--max-latency-regression",
        type = float,
        default = 0.10,
        help = "fail if median latency rises by more than this fraction",
    )
    p.add_argument(
        "--max-vram-regression",
        type = float,
        default = 0.10,
        help = "fail if peak generation VRAM rises by more than this fraction",
    )
    p.add_argument(
        "--min-psnr",
        type = float,
        default = 35.0,
        help = "fail if the fixed-seed image PSNR vs reference drops below this",
    )
    p.add_argument(
        "--force-compare",
        action = "store_true",
        help = "compare even when GPU/device/dtype differ from the baseline",
    )
    p.add_argument(
        "--out-dir", default = "outputs/diffusion_bench", help = "where compare.png is written"
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if bool(args.write_baseline) == bool(args.compare):
        print("error: pass exactly one of --write-baseline / --compare", file = sys.stderr)
        return 2
    if args.write_baseline:
        return _write_baseline(args)
    return _compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
