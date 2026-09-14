#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-render GPU time budget for the SHIPPED Studio diffusion path.

It drives the real Studio ``DiffusionBackend``, so the thing measured is the shipped loader, the
shipped regional compile, the shipped attention pin and the shipped manual CUDA graph, not a
re-implementation.

What it adds over the speed driver is attribution. On torch 2.12.1 CUPTI records graph-replayed
kernels, but a replayed kernel carries no correlation id back to a host launch, so ``key_averages``
cannot attribute it to an ATen op. Everything here is therefore bucketed by TWO facts that survive
graph replay: the kernel NAME, and the GPU-timeline WINDOW it lands in. The windows come from
``record_function`` annotations opened in forward hooks that ``torch.cuda.synchronize()`` first, so
a kernel's device timestamp really does fall inside the annotation that produced it.

Protocol: contention bookend, load, 3 warm-ups (graph capture + flashinfer autotune finish here, so
the profiler window never contains a capture), N unprofiled timed renders (the reference wall), one
hooked-but-unprofiled render (the cost of the three syncs), 2 profiled renders with
``record_shapes`` and ``with_stack`` OFF, contention bookend. ``clean`` is true only when both
contention verdicts are clean.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BACKEND_ROOT = Path(
    os.environ.get("NVFP4_BACKEND_ROOT") or (REPO_ROOT / "studio" / "backend")
)


def _union_busy_us(trace_path, window = None) -> dict:
    """GPU busy time as the UNION of device-side intervals, in microseconds.

    Summing durations over-counts whenever two kernels overlap (different streams, or a memcpy
    concurrent with compute), and a diffusion step does exactly that. Merging the intervals is the
    only way to get a number that can be honestly compared against wall time."""
    trace = json.loads(Path(trace_path).read_text())
    intervals = []
    per_cat: dict = defaultdict(float)
    for e in trace.get("traceEvents", []):
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset", "Kernel", "Memcpy", "Memset"):
            continue
        ts, dur = float(e["ts"]), float(e.get("dur", 0.0))
        if window and (ts + dur < window[0] or ts > window[1]):
            continue
        intervals.append((ts, ts + dur))
        per_cat[cat.lower()] += dur
    if not intervals:
        return {"busy_us": 0.0, "span_us": 0.0, "n_intervals": 0, "sum_dur_us": 0.0}
    intervals.sort()
    busy = 0.0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start > cur_end:
            busy += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    busy += cur_end - cur_start
    return {
        "busy_us": busy,
        "span_us": intervals[-1][1] - intervals[0][0],
        "n_intervals": len(intervals),
        "sum_dur_us": sum(e - s for s, e in intervals),
        "per_category_sum_us": dict(per_cat),
    }


def paired_times(a: list, b: list) -> dict:
    """``b`` relative to ``a``, paired by position. speedup > 1 means ``b`` is faster."""
    deltas = [x - y for x, y in zip(a, b)]
    n = len(deltas)
    mean = statistics.fmean(deltas) if deltas else 0.0
    sd = statistics.stdev(deltas) if n > 1 else 0.0
    return {
        "n": n,
        "mean_delta_s": mean,
        "sd_delta_s": sd,
        "t": (mean / (sd / n**0.5)) if sd else (float("inf") if mean else 0.0),
        "wins": sum(1 for x in deltas if x > 0),
        "speedup_min": min(a) / min(b),
        "speedup_p50": statistics.median(a) / statistics.median(b),
    }


def measure_contention(
    iters_launch = 5000,
    iters_sync = 500,
    iters_rt = 500,
) -> dict:
    """Is this GPU free, or is another CUDA context stealing time slices from it?

    Whole-card ``nvidia-smi`` utilisation cannot answer that: a card sampled at 0 percent between
    somebody else's bursts is indistinguishable from an idle one, and a foreign context costs us
    even when it computes little, because the driver time-slices contexts and a sampler that drains
    its stream every step pays a context switch on every round trip.

      launch_us    enqueue cost of one tiny kernel with the stream kept busy; insensitive to
                   sharing, because the GPU is never given up.
      sync_us      ``torch.cuda.synchronize()`` on an already idle stream; pure API cost.
      rt_us        tiny kernel THEN synchronize, repeatedly, so every iteration re-acquires the SMs.
      slice_ratio  ``rt_us / (launch_us + sync_us)``. 1 to 3 healthy, >> 10 means time-sliced.

    Sustained matmul TFLOPS is reported too because it stays HIGH under exactly this contention (a
    run of back-to-back big GEMMs rarely yields), which is why a throughput benchmark alone will
    call a contended card fine."""
    import torch

    x = torch.zeros(1, device = "cuda")
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters_launch):
        x.zero_()
    torch.cuda.synchronize()
    launch_us = (time.perf_counter() - t0) / iters_launch * 1e6

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters_sync):
        torch.cuda.synchronize()
    sync_us = (time.perf_counter() - t0) / iters_sync * 1e6

    samples = []
    for _ in range(iters_rt):
        t0 = time.perf_counter()
        x.zero_()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6)
    samples.sort()
    rt_us = samples[len(samples) // 2]

    n = 8192
    a = torch.randn(n, n, device = "cuda", dtype = torch.bfloat16)
    b = torch.randn(n, n, device = "cuda", dtype = torch.bfloat16)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 30
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    tflops = 2 * n**3 / ((time.perf_counter() - t0) / iters) / 1e12

    ratio = rt_us / (launch_us + sync_us)
    return {
        "launch_us": launch_us,
        "sync_us": sync_us,
        "rt_us_p50": rt_us,
        "rt_us_min": samples[0],
        "rt_us_p90": samples[int(len(samples) * 0.9)],
        "slice_ratio": ratio,
        "sustained_bf16_tflops": tflops,
        "verdict": "CONTENDED" if ratio > 10 else "clean",
    }


def contention_check(tag: str, out_dir: str) -> dict:
    """Run the contention detector and append the verdict to ``contention.jsonl`` in ``out_dir``."""
    record = measure_contention()
    record["tag"] = tag
    Path(out_dir).mkdir(parents = True, exist_ok = True)
    with open(Path(out_dir) / "contention.jsonl", "a") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


class LpipsScorer:
    """Lazily-built LPIPS(vgg) scorer, reused across pairs (the net is heavy to build).

    ``score`` takes two HxWx3 uint8 arrays and returns None when ``lpips`` is not installed, so a
    speed cell never fails for want of an optional quality metric."""

    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._net = None

    def _net_or_none(self):
        if self._net is None:
            try:
                import lpips
                import torch

                net = lpips.LPIPS(net = "vgg")
                self._net = net.to(
                    device = self._device if torch.cuda.is_available() else "cpu"
                ).eval()
            except Exception as exc:  # noqa: BLE001 - the metric is optional, the timing is not
                print(f"[lpips] unavailable: {type(exc).__name__}: {exc}", flush = True)
                self._net = False
        return self._net or None

    def score(self, ref, cand):
        net = self._net_or_none()
        if net is None:
            return None
        try:
            import numpy as np
            import torch

            device = next(net.parameters()).device

            def to_tensor(a):
                # lpips expects NCHW in [-1, 1].
                t = torch.from_numpy(a.astype(np.float32) / 255.0)
                return (t.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0).to(device)

            with torch.no_grad():
                return float(net(to_tensor(ref), to_tensor(cand)).item())
        except Exception as exc:  # noqa: BLE001
            print(f"[lpips] scoring failed: {type(exc).__name__}: {exc}", flush = True)
            return None


def neutralise_bitsandbytes() -> dict:
    """Compatibility shim. Make diffusers/transformers/peft see NO bitsandbytes when the installed
    one cannot import.

    A bitsandbytes built for another CUDA minor (no matching ``libbitsandbytes_cuda*.so``) raises at
    import, and ``diffusers.quantizers.bitsandbytes.utils`` imports it unconditionally under
    ``is_bitsandbytes_available()``, so a broken-but-present bnb takes the whole diffusers model
    stack down and nothing loads. This is a no-op when bnb is absent or imports cleanly; otherwise
    it flips the availability flags only, which is the bnb-ABSENT state, and installs, writes and
    touches nothing. Reported in the JSON as ``bitsandbytes_neutralised``.
    """
    info = {"present": False, "importable": None, "neutralised": False, "patched": []}
    import importlib.util

    if importlib.util.find_spec("bitsandbytes") is None:
        return info
    info["present"] = True
    try:
        import bitsandbytes  # noqa: F401
        info["importable"] = True
        return info
    except Exception as exc:  # noqa: BLE001 - a broken bnb is the case this exists for
        info["importable"] = False
        info["import_error"] = f"{type(exc).__name__}: {str(exc).strip()[:200]}"
    for mod_name, attrs in (
        ("diffusers.utils.import_utils", ("_bitsandbytes_available",)),
        ("transformers.utils.import_utils", ("_bitsandbytes_available",)),
    ):
        try:
            mod = __import__(mod_name, fromlist = ["*"])
        except Exception:  # noqa: BLE001
            continue
        for attr in attrs:
            if hasattr(mod, attr):
                setattr(mod, attr, False)
                info["patched"].append(f"{mod_name}.{attr}")
    try:
        import peft.import_utils as pu
        for fn in ("is_bnb_available", "is_bnb_4bit_available"):
            if hasattr(pu, fn):
                setattr(pu, fn, lambda: False)
                info["patched"].append(f"peft.import_utils.{fn}")
    except Exception:  # noqa: BLE001
        pass
    info["neutralised"] = bool(info["patched"])
    return info


def patch_hub_compat() -> dict:
    """Compatibility shim. Give ``huggingface_hub`` the two hub-1.x names a recent diffusers imports.

    ``diffusers.pipelines.pipeline_utils`` does a top-level
    ``from huggingface_hub import get_cached_repo_tree`` plus
    ``from huggingface_hub.errors import CachedRepoTreeNotFoundError``. On huggingface_hub < 1.0
    neither exists, so every pipeline import dies before any model is touched. Both names are only
    reached on the OFFLINE listing path, so a shim that reports "nothing cached" is
    behaviour-identical for a harness run off local or already-cached picks, and nothing is
    installed or written. A no-op when the installed hub already has them. Reported in the JSON as
    ``hub_compat``.
    """
    info = {"patched": [], "hub_version": None}
    try:
        import huggingface_hub as hub
        from huggingface_hub import errors as hub_errors
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
        return info
    info["hub_version"] = getattr(hub, "__version__", None)
    if not hasattr(hub_errors, "CachedRepoTreeNotFoundError"):
        base = getattr(hub_errors, "EntryNotFoundError", FileNotFoundError)

        class CachedRepoTreeNotFoundError(base):  # type: ignore[misc, valid-type]
            """No cached file listing for this repo (hub 1.x name, absent in 0.36)."""

        hub_errors.CachedRepoTreeNotFoundError = CachedRepoTreeNotFoundError
        hub.CachedRepoTreeNotFoundError = CachedRepoTreeNotFoundError
        info["patched"].append("huggingface_hub.errors.CachedRepoTreeNotFoundError")
    if not hasattr(hub, "get_cached_repo_tree"):

        def get_cached_repo_tree(*_args, **_kwargs):
            raise hub_errors.CachedRepoTreeNotFoundError(
                "this huggingface_hub has no cached repo tree; harness compatibility shim"
            )

        hub.get_cached_repo_tree = get_cached_repo_tree
        info["patched"].append("huggingface_hub.get_cached_repo_tree")
    return info


# --------------------------------------------------------------------------------- bucket table
# FIRST MATCH WINS and the order is the design (the g845 lesson: testing ``flash`` before
# ``flashinfer`` files the NVFP4 GEMM under attention and inverts the conclusion). The window
# overrides come first of all: a GEMM inside the text encoder is text-encoder time, not denoise
# time, whatever kernel implements it.
_WINDOW_BUCKET = {"phase:te": "text_encoder", "phase:vae": "vae_decode"}

_KERNEL_CATS = ("kernel", "gpu_memcpy", "gpu_memset")


def classify(name: str, window: str | None) -> str:
    """Bucket one device-side kernel by name plus the phase window it ran in."""
    if window in _WINDOW_BUCKET:
        return _WINDOW_BUCKET[window]
    low = name.lower()
    # NVFP4, most specific first.
    if "quantize_with_block_size" in low or "blockscalequantization" in low:
        return "fp4_quantize"
    if "devicegemmfp4" in low or "gemmfp4" in low or "fp4gemm" in low:
        return "fp4_gemm"
    if "fillfunctor" in low:
        return "barrier_fill"
    # fp8: torchao/cublas scaled_mm. ``nvjet_`` is cublasLt's SM100 GEMM family; inside the denoise
    # window on a quantised arm it is the scaled_mm, and outside it is somebody else's GEMM.
    if "enable_3x_kernel_for_sm10" in low or "scaled_mm" in low:
        return "fp8_scaled_mm"
    if "nvjet" in low:
        return "fp8_scaled_mm" if window == "phase:denoise" else "gemm_other"
    if "int8" in low or "i8gemm" in low or "s8s8" in low:
        return "int8"
    # An inductor kernel is an inductor kernel whatever it was fused from, so this outranks the
    # fuzzy attention substrings (``triton_poi_fused__scaled_dot_product_cudnn_attention_...``).
    if low.startswith("triton"):
        return "inductor_triton"
    if "sm100_flash_fwd" in low or "cudnn_generated_fort_native_sdpa" in low or "cudnn" in low:
        return "attention_cudnn"
    if "flash_fwd" in low or "pytorch_flash" in low or "flash_fprop" in low:
        return "attention_flash"
    if "fmha_cutlassf" in low or "mem_eff" in low:
        return "attention_mem_efficient"
    if "fmha" in low or "attention" in low or "mha_" in low or "sdpa" in low:
        return "attention_other"
    if "memcpy dtod" in low or "memcpy" in low or "memset" in low:
        return "memcpy_d2d"
    if "layer_norm" in low or "rms_norm" in low or "welford" in low or "moments" in low:
        return "norm_moments"
    if (
        "elementwise_kernel" in low
        or "at::native" in low
        or "at::cuda" in low
        or "vectorized_" in low
        or "unrolled_" in low
    ):
        return "elementwise_eager"
    if "cutlass" in low or "gemm" in low or "cublas" in low or "gemv" in low:
        return "gemm_other"
    return "other"


ATTENTION_BUCKETS = (
    "attention_cudnn",
    "attention_flash",
    "attention_mem_efficient",
    "attention_other",
)


# ------------------------------------------------------------------------------- trace analysis
def _phase_windows(trace: dict) -> list:
    """``[(start_us, end_us, name)]`` for every ``phase:*`` annotation, sorted by start.

    Only the CPU-side ``user_annotation`` events are used. The profiler also emits a projected
    ``gpu_user_annotation`` copy, and counting both would double every window."""
    out = []
    for e in trace.get("traceEvents", []):
        if e.get("ph") != "X":
            continue
        name = e.get("name") or ""
        if not name.startswith("phase:"):
            continue
        if (e.get("cat") or "").lower() != "user_annotation":
            continue
        ts = float(e["ts"])
        out.append((ts, ts + float(e.get("dur", 0.0)), name))
    out.sort()
    return out


def _window_of(windows: list, ts: float) -> str | None:
    """The innermost phase window containing ``ts``, or None. Linear over a handful of windows per
    render; the trace has at most (steps + 3) of them."""
    hit = None
    for start, end, name in windows:
        if start <= ts <= end:
            hit = name
    return hit


def bucket_table(trace_path: Path, n_renders: int, steps: int) -> dict:
    trace = json.loads(Path(trace_path).read_text())
    windows = _phase_windows(trace)
    per_bucket: dict = defaultdict(lambda: [0.0, 0])
    per_name: dict = defaultdict(lambda: [0.0, 0, ""])
    per_window: dict = defaultdict(lambda: [0.0, 0])
    memcpy_d2d = {"calls": 0, "us": 0.0}
    host = defaultdict(int)
    for e in trace.get("traceEvents", []):
        if e.get("ph") != "X":
            continue
        cat = (e.get("cat") or "").lower()
        name = e.get("name") or ""
        if cat in ("cuda_runtime", "cuda_driver"):
            if name in (
                "cudaLaunchKernel",
                "cuLaunchKernel",
                "cudaLaunchKernelExC",
                "cudaLaunchKernelEx",
                "cudaGraphLaunch",
                "cuGraphLaunch",
                "cudaMemcpyAsync",
            ):
                host[name] += 1
            continue
        if cat not in _KERNEL_CATS:
            continue
        dur = float(e.get("dur", 0.0))
        ts = float(e["ts"])
        win = _window_of(windows, ts + dur / 2.0)
        bucket = classify(name, win)
        b = per_bucket[bucket]
        b[0] += dur
        b[1] += 1
        k = per_name[name]
        k[0] += dur
        k[1] += 1
        k[2] = bucket
        w = per_window[win or "scheduler_other"]
        w[0] += dur
        w[1] += 1
        if cat == "gpu_memcpy" and "dtod" in name.lower():
            memcpy_d2d["calls"] += 1
            memcpy_d2d["us"] += dur
    # Kernels outside every phase window are the sampler's own work (scheduler step, latent maths).
    total_us = sum(v[0] for v in per_bucket.values())
    busy = _union_busy_us(Path(trace_path))
    busy_us = busy["busy_us"]

    def rows(d):
        return {
            k: {
                "ms_per_render": v[0] / n_renders / 1e3,
                "calls_per_render": v[1] / n_renders,
                "calls_per_step": v[1] / n_renders / steps if steps else None,
                "us_per_call": v[0] / v[1] if v[1] else 0.0,
                "pct_busy": 100.0 * v[0] / busy_us if busy_us else 0.0,
                "pct_kernel_sum": 100.0 * v[0] / total_us if total_us else 0.0,
            }
            for k, v in sorted(d.items(), key = lambda kv: -kv[1][0])
        }

    top = sorted(
        (
            {
                "name": n,
                "bucket": v[2],
                "ms_per_render": v[0] / n_renders / 1e3,
                "calls_per_render": v[1] / n_renders,
                "us_per_call": v[0] / v[1],
            }
            for n, v in per_name.items()
        ),
        key = lambda r: -r["ms_per_render"],
    )
    unmatched = [r for r in top if r["bucket"] == "other"][:30]
    attn_names = sorted({n for n, v in per_name.items() if v[2] in ATTENTION_BUCKETS})
    attn_rows = [r for r in top if r["bucket"] in ATTENTION_BUCKETS]
    return {
        "buckets": rows(per_bucket),
        "by_window": rows(per_window),
        "top_kernels": top[:30],
        "unmatched_top": unmatched,
        "attention_kernel_names": attn_names,
        "attention_by_kernel": attn_rows,
        "memcpy_d2d": {
            "calls_per_render": memcpy_d2d["calls"] / n_renders,
            "calls_per_step": memcpy_d2d["calls"] / n_renders / steps if steps else None,
            "ms_per_render": memcpy_d2d["us"] / n_renders / 1e3,
        },
        "phase_window_events": len(windows),
        "distinct_kernels": len(per_name),
        "device_kernel_launches_per_render": sum(v[1] for v in per_bucket.values()) / n_renders,
        "host_launch_api_calls_per_render": {k: v / n_renders for k, v in host.items()},
        "gpu_busy_union_raw": busy,
        "gpu_busy_union_per_render_s": busy_us / n_renders / 1e6,
        "kernel_sum_per_render_s": total_us / n_renders / 1e6,
    }


# ---------------------------------------------------------------------------------- phase hooks
class PhaseHooks:
    """``record_function`` windows around the text encoders, the VAE decode and every denoiser.

    Each boundary synchronizes first, which is what makes a GPU timestamp attributable to the
    window: without it a kernel launched inside the annotation can execute after it closed. The
    three syncs per render are a real cost and are measured (``phase_sync_overhead_s``) rather than
    assumed negligible."""

    def __init__(self, torch_mod, pipe, denoisers):
        self.torch = torch_mod
        self.pipe = pipe
        self.denoisers = list(denoisers)
        self.handles = []
        self.stack: dict = {}
        self.counts: dict = defaultdict(int)
        self._vae_orig = None

    def _open(self, key, label):
        self.torch.cuda.synchronize()
        ctx = self.torch.profiler.record_function(label)
        ctx.__enter__()
        self.stack[key] = ctx
        self.counts[label] += 1

    def _close(self, key):
        ctx = self.stack.pop(key, None)
        if ctx is not None:
            self.torch.cuda.synchronize()
            ctx.__exit__(None, None, None)

    def install(self):
        for attr in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            m = getattr(self.pipe, attr, None)
            if m is None or not hasattr(m, "register_forward_pre_hook"):
                continue
            key = f"te:{attr}"
            self.handles.append(
                m.register_forward_pre_hook(
                    lambda _m, _a, _k = key: self._open(_k, "phase:te"), with_kwargs = False
                )
            )
            self.handles.append(m.register_forward_hook(lambda _m, _a, _o, _k = key: self._close(_k)))
        for i, m in enumerate(self.denoisers):
            key = f"dit:{i}"
            self.handles.append(
                m.register_forward_pre_hook(lambda _m, _a, _k = key: self._open(_k, "phase:denoise"))
            )
            self.handles.append(m.register_forward_hook(lambda _m, _a, _o, _k = key: self._close(_k)))
        vae = getattr(self.pipe, "vae", None)
        if vae is not None and hasattr(vae, "decode"):
            self._vae_orig = vae.decode

            def wrapped(*a, **k):
                self._open("vae", "phase:vae")
                try:
                    return self._vae_orig(*a, **k)
                finally:
                    self._close("vae")

            vae.decode = wrapped
        return self

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass
        self.handles = []
        vae = getattr(self.pipe, "vae", None)
        if self._vae_orig is not None and vae is not None:
            vae.decode = self._vae_orig
            self._vae_orig = None


# ------------------------------------------------------------------------------------------ run
def _install_whole_compile(torch_mod, denoisers, state) -> dict:
    """``torch.compile(fullgraph=True, dynamic=False)`` over the WHOLE denoiser forward.

    ``module.compile()`` sets ``_compiled_call_impl``, which ``_wrapped_call_impl`` takes BEFORE the
    instance-slot ``forward``, so it would silently bypass ``GraphedForward`` and the graph arm
    would measure something else. The bound forward is compiled instead and re-installed: under
    graphs it replaces ``GraphedForward.orig`` (what the capture records), and without graphs it
    goes straight into the instance slot.

    NOTE: Studio's REGIONAL compile has already run by this point and there is no shipped env knob
    to turn it off (`UNSLOTH_COMPILE_SCOPE` is a proposal in the plan, not code), and turning it off
    would mean editing the worktree. So this arm is whole-compile ON TOP OF regional compile, and
    says so.
    """
    out = {"modules": [], "nested_on_regional_compile": True}
    torch_mod._dynamo.config.recompile_limit = 64
    torch_mod._dynamo.config.cache_size_limit = 64
    for module in denoisers:
        name = type(module).__name__
        slot = module.__dict__.get("forward", None)
        try:
            bound = type(module).forward.__get__(module)
            compiled = torch_mod.compile(bound, fullgraph = True, dynamic = False)
            if slot is not None and hasattr(slot, "orig"):
                slot.orig = compiled
                where = "GraphedForward.orig"
            else:
                module.__dict__["forward"] = compiled
                where = "instance slot"
            out["modules"].append({"module": name, "installed": where})
        except Exception as exc:  # noqa: BLE001 - a refusal is the result
            out["modules"].append({"module": name, "error": f"{type(exc).__name__}: {exc}"[:600]})
    return out


def _resolve_guidance(args, model):
    if args.guidance is not None:
        return args.guidance
    from core.inference.diffusion_families import default_generation_params, detect_family

    fam = detect_family(args.family) or detect_family(model)
    _steps, guidance = default_generation_params(model, getattr(fam, "name", None))
    return guidance


def main(argv = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default = "image", choices = ("image", "video"))
    ap.add_argument("--family", required = True)
    ap.add_argument("--model", required = True, help = "repo id or local pick dir")
    ap.add_argument("--arm", required = True, choices = ("bf16", "fp8", "nvfp4"))
    ap.add_argument("--nvfp4-backend", default = "flashinfer", choices = ("torchao", "flashinfer"))
    ap.add_argument("--prequant-path", default = None)
    ap.add_argument("--gguf-filename", default = None)
    ap.add_argument("--base-repo", default = None)
    ap.add_argument("--model-kind", default = None)
    ap.add_argument("--family-override", default = None)
    ap.add_argument("--transformer-cache", default = None)
    ap.add_argument("--text-encoder-quant", default = None)
    ap.add_argument("--resolution", default = "1024", help = "square edge for image, WxH for video")
    ap.add_argument("--frames", type = int, default = None)
    ap.add_argument("--steps", type = int, required = True)
    ap.add_argument("--guidance", type = float, default = None)
    ap.add_argument("--seed", type = int, default = 20260909)
    ap.add_argument(
        "--prompt", default = "a red sailboat on a calm blue lake at sunrise, crisp detail"
    )
    ap.add_argument("--graphs", default = "on", choices = ("on", "off", "both"))
    ap.add_argument(
        "--attention", default = "shipped", choices = ("shipped", "cudnn", "flash", "efficient", "math")
    )
    ap.add_argument(
        "--compile", dest = "compile_mode", default = "regional", choices = ("regional", "whole", "off")
    )
    ap.add_argument("--speed-mode", default = "default")
    ap.add_argument("--warmups", type = int, default = 3)
    ap.add_argument("--timed", type = int, default = 7)
    ap.add_argument("--profiled", type = int, default = 2)
    ap.add_argument(
        "--hooked",
        type = int,
        default = 2,
        help = "unprofiled renders WITH the phase hooks, for the sync overhead",
    )
    ap.add_argument(
        "--save-latent",
        default = None,
        help = "write the final pre-VAE latent of the first timed render here",
    )
    ap.add_argument("--negative-prompt", default = None)
    ap.add_argument("--gpu-uuid", default = None, help = "recorded; set CUDA_VISIBLE_DEVICES too")
    ap.add_argument("--backend-root", default = str(DEFAULT_BACKEND_ROOT))
    ap.add_argument("--tag", default = None)
    ap.add_argument(
        "--trace-dir", default = None, help = "chrome traces; defaults to a `traces` dir beside --out"
    )
    ap.add_argument("--keep-trace", action = "store_true")
    ap.add_argument("--out", required = True, help = "path; may contain {graphs} when --graphs both")
    args = ap.parse_args(argv)

    root = Path(args.backend_root)
    sys.path.insert(0, str(root))
    if args.arm == "nvfp4":
        os.environ["UNSLOTH_NVFP4_BACKEND"] = args.nvfp4_backend

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents = True, exist_ok = True)
    args.trace_dir = args.trace_dir or str(out_dir / "traces")
    Path(args.trace_dir).mkdir(parents = True, exist_ok = True)

    graph_arms = ["on", "off"] if args.graphs == "both" else [args.graphs]
    if args.graphs == "both" and "{graphs}" not in args.out:
        print("--graphs both needs a {graphs} placeholder in --out", flush = True)
        return 2

    pre = contention_check("pre", str(out_dir))
    print(f"[contention pre] {pre['verdict']} slice_ratio={pre['slice_ratio']:.2f}", flush = True)

    rc = 0
    for graphs in graph_arms:
        rc |= run_one(args, graphs, pre, str(root))
    return rc


def run_one(args, graphs: str, pre: dict, root: str) -> int:
    import torch

    os.environ.pop("UNSLOTH_DISABLE_CUDA_GRAPH", None)
    if graphs == "off":
        os.environ["UNSLOTH_DISABLE_CUDA_GRAPH"] = "1"

    tag = args.tag or (
        f"{args.family}_{args.resolution}_{args.arm}_graphs{graphs}" f"_{args.attention}"
    )
    out_path = Path(args.out.format(graphs = graphs) if "{graphs}" in args.out else args.out)
    trace_path = Path(args.trace_dir) / f"{tag}.json"
    record: dict = {
        "tag": tag,
        "family": args.family,
        "model": args.model,
        "arm": args.arm,
        "nvfp4_backend_requested": args.nvfp4_backend if args.arm == "nvfp4" else None,
        "prequant_path": args.prequant_path,
        "resolution": args.resolution,
        "frames": args.frames,
        "steps": args.steps,
        "graphs": graphs,
        "attention_requested": args.attention,
        "compile_mode": args.compile_mode,
        "backend_root": root,
        "backend_kind": args.backend,
        "warmups": args.warmups,
        "timed": args.timed,
        "profiled": args.profiled,
        "gpu_uuid": args.gpu_uuid,
        "seed": args.seed,
        "contention": {"pre": pre, "post": None},
        "clean": None,
        "argv": sys.argv,
        "env": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("UNSLOTH_", "TORCHDYNAMO", "TORCHINDUCTOR", "CUDA_VISIBLE"))
        },
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

        record["bitsandbytes_neutralised"] = neutralise_bitsandbytes()
        record["hub_compat"] = patch_hub_compat()
        if args.backend == "video":
            from core.inference.video import get_video_backend as get_backend
        else:
            from core.inference.diffusion import get_diffusion_backend as get_backend

        record["versions"] = {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
        }
        for mod in ("torchao", "diffusers", "flashinfer", "triton"):
            try:
                record["versions"][mod] = __import__(mod).__version__
            except Exception:  # noqa: BLE001
                record["versions"][mod] = None

        backend = get_backend()
        load_kwargs: dict = {
            "hf_token": os.environ.get("HF_TOKEN"),
            "speed_mode": args.speed_mode,
            "transformer_quant": None if args.arm == "bf16" else args.arm,
        }
        for key, value in (
            ("text_encoder_quant", args.text_encoder_quant),
            ("model_kind", args.model_kind),
            ("gguf_filename", args.gguf_filename),
            ("family_override", args.family_override),
            ("base_repo", args.base_repo),
            ("transformer_cache", args.transformer_cache),
        ):
            if value:
                load_kwargs[key] = value
        if args.arm == "nvfp4" and args.prequant_path and args.backend != "video":
            # The video loader has no local-path override: it resolves the family's hosted
            # prequant repo itself (video_families.prequant_repos), which is the shipped path.
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
        record["resolved"] = status.get("resolved")
        state = backend._state
        pipe = state.pipe
        record["cuda_graph_reason"] = getattr(pipe, "_unsloth_cuda_graph_reason", None)

        def _graph_io_counts() -> list:
            """Static input / output tensor counts per captured graph.

            The D2D memcpy count per step is expected to be ``2 * n_inputs + n_outputs`` (copy in,
            the capture's own copies, clone out), so the bucket is checkable rather than merely
            reported."""
            rows = []
            for handle in getattr(state, "cuda_graphs", ()) or ():
                for entry in getattr(handle, "cache", {}).values():
                    try:
                        import torch as _t

                        static = getattr(entry, "static", None) or ()
                        n_in = sum(
                            1 for x in _t.utils._pytree.tree_leaves(static) if _t.is_tensor(x)
                        )
                        outs = getattr(entry, "out_tensors", None) or ()
                        n_out = sum(
                            1 for x in _t.utils._pytree.tree_leaves(outs) if _t.is_tensor(x)
                        )
                        rows.append(
                            {
                                "n_inputs": n_in,
                                "n_outputs": n_out,
                                "expected_memcpy_per_step": 2 * n_in + n_out,
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        rows.append({"error": str(exc)})
            return rows

        flush()

        from core.inference.diffusion_speed import _denoiser_dits

        denoisers = _denoiser_dits(pipe)
        record["n_denoisers"] = len(denoisers)
        try:
            from core.inference.diffusion_nvfp4_linear import is_nvfp4_flashinfer_linear

            observed = None
            for module in denoisers:
                for _n, sub in module.named_modules():
                    if is_nvfp4_flashinfer_linear(sub):
                        observed = "flashinfer"
                        break
                if observed:
                    break
            if observed is None and args.arm == "nvfp4":
                observed = "torchao"
            record["backend_observed"] = observed
        except Exception as exc:  # noqa: BLE001 - provenance, not a gate
            record["backend_observed"] = f"unknown: {exc}"

        # Attention: ``shipped`` touches nothing, which is the point of this pass. The others go
        # straight at the diffusers dispatcher, bypassing Studio's alias table.
        from core.inference import diffusion_attention as attn_mod

        if args.attention != "shipped":
            want = {
                "cudnn": "_native_cudnn",
                "flash": "_native_flash",
                "efficient": "_native_efficient",
                "math": "_native_math",
            }[args.attention]
            for m in denoisers:
                m.set_attention_backend(want)
            from core.inference import diffusion_cuda_graph as _cg

            _cg.reset_all(getattr(state, "cuda_graphs", ()) or ())
        record["attention_per_denoiser"] = [
            getattr(m, "_attention_backend", None) for m in denoisers
        ]
        try:
            record["attention_engaged"] = attn_mod._active_attention_backend()
        except Exception as exc:  # noqa: BLE001
            record["attention_engaged"] = f"unknown: {exc}"

        if args.compile_mode == "whole":
            record["whole_compile"] = _install_whole_compile(torch, denoisers, state)
            print(f"[whole-compile] {record['whole_compile']}", flush = True)

        guidance = _resolve_guidance(args, args.model)
        record["guidance"] = guidance
        if args.backend == "video":
            width, height = (
                (int(x) for x in str(args.resolution).split("x"))
                if "x" in str(args.resolution)
                else (args.resolution, args.resolution)
            )
            gen_kwargs = dict(
                prompt = args.prompt,
                width = width,
                height = height,
                steps = args.steps,
                guidance = guidance,
                seed = args.seed,
            )
            if args.frames:
                gen_kwargs["num_frames"] = args.frames
            if args.negative_prompt:
                gen_kwargs["negative_prompt"] = args.negative_prompt
        else:
            gen_kwargs = dict(
                prompt = args.prompt,
                width = int(args.resolution),
                height = int(args.resolution),
                steps = args.steps,
                guidance = guidance,
                seed = args.seed,
                batch_size = 1,
            )

        # Final pre-VAE latent, for the bit-identity / max-abs comparisons between arms. Cloned
        # only while ``latent_box["arm"]`` is set, so the steady timing renders pay nothing.
        latent_box: dict = {"arm": False, "tensor": None}
        _vae = getattr(pipe, "vae", None)
        if _vae is not None and hasattr(_vae, "decode"):
            _vae_decode_orig = _vae.decode

            def _vae_decode_capture(*a, **k):
                if latent_box["arm"]:
                    cand = a[0] if a else k.get("z", k.get("latents"))
                    if torch.is_tensor(cand):
                        latent_box["tensor"] = cand.detach().float().clone()
                        latent_box["arm"] = False
                return _vae_decode_orig(*a, **k)

            _vae.decode = _vae_decode_capture

        def render() -> float:
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = backend.generate(**gen_kwargs)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            images = (result.get("images") or []) if isinstance(result, dict) else []
            if images:
                arr = np.asarray(images[0].convert("RGB")).astype("float32") / 255.0
                render.last_luma = float(arr.mean())  # type: ignore[attr-defined]
            return seconds

        t0 = time.time()
        warmup_walls = [render() for _ in range(args.warmups)]
        record["warmup_s"] = round(time.time() - t0, 2)
        # The FIRST warm-up is where a cold compile is paid, so it is reported separately rather
        # than averaged into the others.
        record["warmup_walls_s"] = [round(x, 4) for x in warmup_walls]
        record["cold_first_render_s"] = warmup_walls[0] if warmup_walls else None
        try:
            from core.inference import diffusion_cuda_graph as _cg
            record["cuda_graph_stats"] = _cg.stats(getattr(state, "cuda_graphs", ()) or ())
        except Exception as exc:  # noqa: BLE001
            record["cuda_graph_stats"] = str(exc)
        try:
            record["cuda_graph_io"] = _graph_io_counts()
        except Exception as exc:  # noqa: BLE001
            record["cuda_graph_io"] = str(exc)
        flush()

        torch.cuda.reset_peak_memory_stats()
        latent_box["arm"] = True
        walls = [render() for _ in range(args.timed)]
        lat = latent_box["tensor"]
        if lat is not None:
            record["latent"] = {
                "shape": list(lat.shape),
                "dtype": str(lat.dtype),
                "abs_mean": float(lat.abs().mean()),
                "abs_max": float(lat.abs().max()),
                "sum": float(lat.double().sum()),
            }
            if args.save_latent:
                Path(args.save_latent).parent.mkdir(parents = True, exist_ok = True)
                torch.save(lat.cpu(), args.save_latent)
                record["latent"]["path"] = args.save_latent
        record["unprofiled_s"] = [round(x, 5) for x in walls]
        record["p50_s"] = statistics.median(walls)
        record["min_s"] = min(walls)
        record["mean_luma"] = getattr(render, "last_luma", None)
        record["steady_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        record["steady_reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 3)
        print(
            f"[{tag}] unprofiled p50 {record['p50_s']:.4f}s min {record['min_s']:.4f}s", flush = True
        )
        flush()

        # The phase hooks cost three synchronizes per render. Measure that WITHOUT the profiler so
        # the profiler's own overhead is not folded into it.
        hooks = PhaseHooks(torch, pipe, denoisers).install()
        hooked = [render() for _ in range(args.hooked)] or [record["p50_s"]]
        record["hooked_unprofiled_s"] = [round(x, 5) for x in hooked]
        record["phase_sync_overhead_s"] = statistics.median(hooked) - record["p50_s"]
        record["phase_calls"] = dict(hooks.counts)

        from torch.profiler import ProfilerActivity, profile

        prof_walls = []
        with profile(
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes = False,
            with_stack = False,
            with_flops = False,
        ) as prof:
            for _ in range(args.profiled):
                prof_walls.append(render())
        hooks.remove()
        prof.export_chrome_trace(str(trace_path))
        record["profiled_walls_s"] = [round(x, 5) for x in prof_walls]
        record["profiler_overhead_ratio"] = statistics.median(prof_walls) / record["p50_s"]
        record["paired_profiled_vs_unprofiled"] = paired_times(walls[: len(prof_walls)], prof_walls)
        record["trace"] = str(trace_path)

        table = bucket_table(trace_path, args.profiled, args.steps)
        record.update(
            {
                k: table[k]
                for k in (
                    "buckets",
                    "by_window",
                    "top_kernels",
                    "unmatched_top",
                    "attention_kernel_names",
                    "attention_by_kernel",
                    "memcpy_d2d",
                    "phase_window_events",
                    "distinct_kernels",
                    "device_kernel_launches_per_render",
                    "host_launch_api_calls_per_render",
                )
            }
        )
        record["gpu_busy_union_s"] = table["gpu_busy_union_per_render_s"]
        record["gpu_busy_union_raw"] = table["gpu_busy_union_raw"]
        record["kernel_sum_per_render_s"] = table["kernel_sum_per_render_s"]
        record["host_idle_s"] = record["p50_s"] - record["gpu_busy_union_s"]
        record["gpu_busy_fraction_of_wall"] = record["gpu_busy_union_s"] / record["p50_s"]
        flush()

        print(
            f"[{tag}] busy {record['gpu_busy_union_s']:.4f}s  idle "
            f"{record['host_idle_s']:.4f}s",
            flush = True,
        )
        for name, row in list(record["buckets"].items())[:14]:
            print(
                f"    {row['ms_per_render']:9.2f} ms  x{row['calls_per_render']:8.1f}  "
                f"{row['pct_busy']:5.1f}%  {name}",
                flush = True,
            )
        if record["unmatched_top"]:
            print(f"[{tag}] unmatched kernels:", flush = True)
            for r in record["unmatched_top"][:30]:
                print(f"    {r['ms_per_render']:9.3f} ms  {r['name'][:110]}", flush = True)

        backend.unload()
        backend = None
        if not args.keep_trace:
            try:
                trace_path.unlink()
                record["trace"] = f"{trace_path} (deleted)"
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001 - a refusal IS the result and is recorded
        import traceback

        traceback.print_exc()
        record["contention"]["post"] = contention_check("post", str(out_path.parent))
        record["clean"] = (
            pre.get("verdict") == "clean" and record["contention"]["post"].get("verdict") == "clean"
        )
        flush(exc)
        try:
            if backend is not None:
                backend.unload()
        except Exception:  # noqa: BLE001
            pass
        return 1

    post = contention_check("post", str(out_path.parent))
    record["contention"]["post"] = post
    record["clean"] = pre.get("verdict") == "clean" and post.get("verdict") == "clean"
    print(
        f"[contention post] {post['verdict']} slice_ratio={post['slice_ratio']:.2f} "
        f"clean={record['clean']}",
        flush = True,
    )
    flush()
    print(f"[{tag}] wrote {out_path}", flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
