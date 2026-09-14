#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Attention backend A/B on the real shipped Studio image path.

One process, one load, all backends resident, paired by seed in a round robin, because the enemy
here is drift: two backends measured in two processes differ by whatever else the box was doing.

The graph makes the rotation expensive. A captured CUDA graph BAKED the attention
kernel that was live at capture time, and the compiled block guards on
``processor._attention_backend``, so switching backends means
``set_attention_backend`` on every denoiser, ``cuda_graph.reset_all`` and one warm render to
re-capture, before the timed render of that rotation. That warm render is paid on every switch and
is never timed.

Numerics: the final pre-VAE latent is captured at the same seed for every backend and compared
against the cuDNN one (max-abs, mean-abs, equality), plus LPIPS on the decoded image.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The budget profiler owns the bucket table, the phase hooks and the contention detector, and pulls
# in nothing heavier than argparse at module scope.
import nvfp4_budget_profile as PG  # noqa: E402

BACKENDS = {
    "cudnn": "_native_cudnn",
    "flash": "_native_flash",
    "efficient": "_native_efficient",
    "math": "_native_math",
}


def observed_backends(modules) -> list[str]:
    """The backend actually installed on the attention processors of ``modules``.

    ``set_attention_backend`` writes ``processor._attention_backend`` on every attention submodule
    (diffusers modeling_utils.py:642-649); the denoiser itself carries no such attribute, so reading
    it off the module would report ``None`` for a switch that worked. Empty means nothing exposed a
    backend, which is unverifiable rather than wrong."""
    seen = set()
    for module in modules:
        for sub in module.modules():
            processor = getattr(sub, "processor", None)
            if processor is None:
                continue
            value = getattr(processor, "_attention_backend", None)
            if value is not None:
                seen.add(str(getattr(value, "value", value)))
    return sorted(seen)


def switch_failure(info: dict) -> str | None:
    """Why this arm must not be timed, or None if the requested backend is really installed.

    A rejected ``set_attention_backend`` leaves the PREVIOUS backend live, and the render that
    follows then succeeds on it: timing that arm would publish the old kernel's number under the new
    kernel's label. Both the recorded exception and an observed mismatch are refusals."""
    if info.get("errors"):
        return f"set_attention_backend refused: {info['errors'][0]}"
    requested = info.get("requested")
    observed = info.get("observed") or []
    if observed and observed != [requested]:
        return f"backend observed as {observed}, not {requested}"
    return None


def main(argv = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required = True)
    ap.add_argument("--model", required = True)
    ap.add_argument("--gguf-filename", default = None)
    ap.add_argument("--base-repo", default = None)
    ap.add_argument("--arm", default = "nvfp4", choices = ("bf16", "fp8", "nvfp4"))
    ap.add_argument("--nvfp4-backend", default = "flashinfer")
    ap.add_argument("--prequant-path", default = None)
    ap.add_argument("--transformer-cache", default = None)
    ap.add_argument("--resolution", type = int, default = 1024)
    ap.add_argument("--steps", type = int, required = True)
    ap.add_argument("--guidance", type = float, default = None)
    ap.add_argument("--graphs", default = "on", choices = ("on", "off"))
    ap.add_argument("--backends", default = "cudnn,flash,efficient")
    ap.add_argument("--rotations", type = int, default = 7)
    ap.add_argument("--seed-base", type = int, default = 1234)
    ap.add_argument("--warmups", type = int, default = 3)
    ap.add_argument(
        "--prompt", default = "a red sailboat on a calm blue lake at sunrise, crisp detail"
    )
    ap.add_argument("--gpu-uuid", default = None)
    ap.add_argument("--backend-root", default = str(PG.DEFAULT_BACKEND_ROOT))
    ap.add_argument("--out", required = True)
    args = ap.parse_args(argv)

    import os

    sys.path.insert(0, args.backend_root)
    if args.arm == "nvfp4":
        os.environ["UNSLOTH_NVFP4_BACKEND"] = args.nvfp4_backend
    os.environ.pop("UNSLOTH_DISABLE_CUDA_GRAPH", None)
    if args.graphs == "off":
        os.environ["UNSLOTH_DISABLE_CUDA_GRAPH"] = "1"

    labels = [b.strip() for b in args.backends.split(",") if b.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents = True, exist_ok = True)
    trace_dir = out_path.parent / "traces"
    trace_dir.mkdir(parents = True, exist_ok = True)

    pre = PG.contention_check("pre", str(out_path.parent))
    record: dict = {
        "tag": f"attn_ab_{args.family}_{args.resolution}_{args.arm}_graphs{args.graphs}",
        "family": args.family,
        "model": args.model,
        "arm": args.arm,
        "resolution": args.resolution,
        "steps": args.steps,
        "graphs": args.graphs,
        "backends": labels,
        "rotations": args.rotations,
        "seed_base": args.seed_base,
        "backend_root": args.backend_root,
        "gpu_uuid": args.gpu_uuid,
        "contention": {"pre": pre, "post": None},
        "clean": None,
        "argv": sys.argv,
        "run_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    def flush(exc = None):
        if exc is not None:
            import traceback
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()[-4000:]
        out_path.write_text(json.dumps(record, indent = 2, default = str) + "\n")

    backend = None
    try:
        import numpy as np
        import torch

        record["bitsandbytes_neutralised"] = PG.neutralise_bitsandbytes()
        record["hub_compat"] = PG.patch_hub_compat()
        from core.inference.diffusion import get_diffusion_backend

        backend = get_diffusion_backend()
        load_kwargs: dict = {
            "hf_token": os.environ.get("HF_TOKEN"),
            "speed_mode": "default",
            "transformer_quant": None if args.arm == "bf16" else args.arm,
        }
        for key, value in (
            ("gguf_filename", args.gguf_filename),
            ("base_repo", args.base_repo),
            ("transformer_cache", args.transformer_cache),
        ):
            if value:
                load_kwargs[key] = value
        if args.arm == "nvfp4" and args.prequant_path:
            load_kwargs["transformer_prequant_path"] = args.prequant_path
        t0 = time.time()
        backend.begin_load(args.model, **load_kwargs)
        while True:
            progress = backend.load_progress()
            if progress.get("phase") == "ready":
                break
            if progress.get("phase") == "error":
                raise RuntimeError(progress.get("error"))
            time.sleep(2)
        record["load_s"] = round(time.time() - t0, 2)
        status = backend.status()
        record["speed_optims"] = status.get("speed_optims")
        state = backend._state
        pipe = state.pipe
        from core.inference import diffusion_cuda_graph as cg
        from core.inference.diffusion_speed import _denoiser_dits

        denoisers = _denoiser_dits(pipe)
        record["n_denoisers"] = len(denoisers)
        record["cuda_graph_reason"] = getattr(pipe, "_unsloth_cuda_graph_reason", None)

        guidance = args.guidance
        if guidance is None:
            from core.inference.diffusion_families import (
                default_generation_params,
                detect_family,
            )
            fam = detect_family(args.family) or detect_family(args.model)
            _s, guidance = default_generation_params(args.model, getattr(fam, "name", None))
        record["guidance"] = guidance

        latent_box: dict = {"arm": False, "tensor": None}
        vae = pipe.vae
        vae_decode_orig = vae.decode

        def vae_decode_capture(*a, **k):
            if latent_box["arm"]:
                cand = a[0] if a else k.get("z", k.get("latents"))
                if torch.is_tensor(cand):
                    latent_box["tensor"] = cand.detach().float().clone()
                    latent_box["arm"] = False
            return vae_decode_orig(*a, **k)

        vae.decode = vae_decode_capture

        images: dict = {}

        def render(seed: int, keep_image: str | None = None) -> float:
            torch.cuda.synchronize()
            t = time.perf_counter()
            result = backend.generate(
                prompt = args.prompt,
                width = args.resolution,
                height = args.resolution,
                steps = args.steps,
                guidance = guidance,
                seed = seed,
                batch_size = 1,
            )
            torch.cuda.synchronize()
            secs = time.perf_counter() - t
            if keep_image and result.get("images"):
                images[keep_image] = np.asarray(result["images"][0].convert("RGB")).astype("uint8")
            return secs

        def switch(label: str) -> dict:
            """Force one SDPA backend on every denoiser and make the graph forget the old one."""
            name = BACKENDS[label]
            errs = []
            for m in denoisers:
                try:
                    m.set_attention_backend(name)
                except Exception as exc:  # noqa: BLE001 - a refusal is the result
                    errs.append(f"{type(m).__name__}: {type(exc).__name__}: {exc}"[:300])
            cg.reset_all(getattr(state, "cuda_graphs", ()) or ())
            return {
                "requested": name,
                "observed": observed_backends(denoisers),
                "module_attrs": [getattr(m, "_attention_backend", None) for m in denoisers],
                "errors": errs,
            }

        # Warm the load once on the shipped backend before any switching.
        for _ in range(args.warmups):
            render(args.seed_base - 1)

        per_backend: dict = {label: {"times": [], "switch": None} for label in labels}
        dropped = set()
        for label in labels:
            info = switch(label)
            per_backend[label]["switch"] = info
            failure = switch_failure(info)
            if failure is not None:
                per_backend[label]["switch_error"] = failure
                dropped.add(label)
                print(f"[ab] DROPPED {label}: {failure}", flush = True)
                continue
            try:
                render(args.seed_base - 1)  # re-capture the graph on this kernel
            except Exception as exc:  # noqa: BLE001
                per_backend[label]["warm_error"] = f"{type(exc).__name__}: {exc}"[:600]
                dropped.add(label)
                print(f"[ab] DROPPED {label}: {per_backend[label]['warm_error']}", flush = True)
        live = [b for b in labels if b not in dropped]
        record["dropped"] = sorted(dropped)
        record["dropped_reasons"] = {
            label: per_backend[label].get("switch_error") or per_backend[label].get("warm_error")
            for label in sorted(dropped)
        }

        for rot in range(args.rotations):
            seed = args.seed_base + rot
            for label in live:
                info = switch(label)
                failure = switch_failure(info)
                if failure is not None:
                    # A backend that validated above and refuses now would silently hand the rest of
                    # the rotation to whatever kernel stayed live.
                    raise RuntimeError(f"{label}: {failure}")
                render(args.seed_base - 1)  # re-capture after the switch, never timed
                want_numerics = rot == 0
                latent_box["arm"] = want_numerics
                secs = render(seed, keep_image = label if want_numerics else None)
                per_backend[label]["times"].append(secs)
                if want_numerics and latent_box["tensor"] is not None:
                    per_backend[label]["latent"] = latent_box["tensor"].cpu()
            print(
                f"[ab] rot {rot}: "
                + "  ".join(f"{b}={per_backend[b]['times'][-1]:.4f}" for b in live),
                flush = True,
            )
            flush()

        # One profiled render per backend, for the attention bucket and the kernel names.
        for label in live:
            info = switch(label)
            failure = switch_failure(info)
            if failure is not None:
                raise RuntimeError(f"{label}: {failure}")
            render(args.seed_base - 1)
            hooks = PG.PhaseHooks(torch, pipe, denoisers).install()
            from torch.profiler import ProfilerActivity, profile

            trace_path = trace_dir / f"attn_ab_{args.family}_{args.resolution}_{label}.json"
            with profile(
                activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes = False,
                with_stack = False,
            ) as prof:
                render(args.seed_base)
            hooks.remove()
            prof.export_chrome_trace(str(trace_path))
            table = PG.bucket_table(trace_path, 1, args.steps)
            per_backend[label]["buckets"] = table["buckets"]
            per_backend[label]["attention_kernel_names"] = table["attention_kernel_names"]
            per_backend[label]["attention_by_kernel"] = table["attention_by_kernel"]
            per_backend[label]["gpu_busy_union_s"] = table["gpu_busy_union_per_render_s"]
            try:
                trace_path.unlink()
            except Exception:  # noqa: BLE001
                pass

        ref = per_backend.get("cudnn", {}).get("latent")
        lpips_scorer = PG.LpipsScorer("cuda")

        summary = {}
        for label in live:
            d = per_backend[label]
            times = d["times"]
            row = {
                "n": len(times),
                "p50_s": statistics.median(times),
                "min_s": min(times),
                "mean_s": statistics.fmean(times),
                "sd_s": statistics.stdev(times) if len(times) > 1 else 0.0,
                "gpu_busy_union_s": d.get("gpu_busy_union_s"),
                "attention_ms_per_render": sum(
                    v["ms_per_render"]
                    for k, v in (d.get("buckets") or {}).items()
                    if k.startswith("attention")
                ),
                "attention_kernel_names": d.get("attention_kernel_names"),
                "attention_by_kernel": d.get("attention_by_kernel"),
                "switch": d.get("switch"),
                "times": [round(x, 5) for x in times],
            }
            if "cudnn" in per_backend and per_backend["cudnn"].get("times"):
                row["paired_vs_cudnn"] = PG.paired_times(per_backend["cudnn"]["times"], times)
            lat = d.get("latent")
            if ref is not None and lat is not None:
                diff = lat - ref
                row["latent_vs_cudnn"] = {
                    "equal": bool(lat.shape == ref.shape and (diff == 0).all().item()),
                    "max_abs": float(diff.abs().max()),
                    "mean_abs": float(diff.abs().mean()),
                    "ref_abs_mean": float(ref.abs().mean()),
                }
            if lpips_scorer is not None and "cudnn" in images and label in images:
                row["lpips_vs_cudnn"] = lpips_scorer.score(images["cudnn"], images[label])
            summary[label] = row
        record["per_backend"] = summary
        for label, row in summary.items():
            print(
                f"[ab] {label:10s} p50 {row['p50_s']:.4f}s  busy "
                f"{(row['gpu_busy_union_s'] or 0):.4f}s  attn "
                f"{row['attention_ms_per_render']:.2f} ms  lpips "
                f"{row.get('lpips_vs_cudnn')}",
                flush = True,
            )
        backend.unload()
        backend = None
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        record["contention"]["post"] = PG.contention_check("post", str(out_path.parent))
        flush(exc)
        try:
            if backend is not None:
                backend.unload()
        except Exception:  # noqa: BLE001
            pass
        return 1
    post = PG.contention_check("post", str(out_path.parent))
    record["contention"]["post"] = post
    record["clean"] = pre.get("verdict") == "clean" and post.get("verdict") == "clean"
    flush()
    print(f"[ab] wrote {out_path} clean={record['clean']}", flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
