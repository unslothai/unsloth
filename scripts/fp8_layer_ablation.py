# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-layer fp8 ablation probe for video DiTs (Wan / Hunyuan): find which layers -- if any --
break under production per-row fp8 (torch._scaled_mm), or prove the failure is systemic.

Motivation: on Blackwell the auto ladder leads with fp8, but on Wan2.2 and HunyuanVideo-1.5 that
renders every frame BLACK (int8 is clean), so the shipped fix denies fp8 -> int8. Is the black
frame a small nameable set of outlier layers (keep fp8 on the rest), or systemic to _scaled_mm
(so per-layer exclusion can't help and int8 stays right)?

Method (single-forward proxy): run one bf16 generation, capture a REAL mid-schedule DiT input
via a forward_pre_hook, then compare the dense output to the output after quantising layer
SUBSETS with the production fp8 config. Reuses ``_make_quant_config`` / ``make_filter_fn``, so
the GEMM path is byte-identical to production.

Phase 0 (default) is the gate:
  0a proxy-validity : fp8-ALL must reproduce the black signal in one forward. If it's CLEAN, the
                      black frame is multi-step / cache compounding -> single-forward bisection
                      can't localise it -> STOP, keep int8.
  0b mechanism      : (i) per-Linear finiteness hooks locate the first inf/NaN; (ii) flip
                      use_fast_accum -- if it flips black->clean the fix is a one-line accumulate
                      flag.
Also records per-Linear input outlier stats on the bf16 forward for the Phase 2 ranking.

Example:
    CUDA_VISIBLE_DEVICES=1 python scripts/fp8_layer_ablation.py --family wan2.2-ti2v-5b --phase 0
"""

from __future__ import annotations

import argparse
import copy
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

# Reuse the benchmark's real pipeline-build + family plumbing so the load is identical.
from video_speedmem_bench import _FAMILIES, PROMPT, _build_pipe, _import_diffusers  # noqa: E402
from core.inference.diffusion_transformer_quant import (  # noqa: E402
    DEFAULT_MIN_LINEAR_FEATURES,
    TQ_FP8,
    _REQUIRE_BF16_SCHEMES,
    _make_quant_config,
    make_filter_fn,
)


# ── cuda helpers ────────────────────────────────────────────────────────────────
def _empty() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _detach_to_cpu(obj: Any) -> Any:
    """Recursively detach + move tensors in a nested args/kwargs structure to CPU."""
    import torch

    if torch.is_tensor(obj):
        return obj.detach().to("cpu")
    if isinstance(obj, tuple):
        return tuple(_detach_to_cpu(o) for o in obj)
    if isinstance(obj, list):
        return [_detach_to_cpu(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _detach_to_cpu(v) for k, v in obj.items()}
    return obj


def _to_device(obj: Any, device: str) -> Any:
    """Recursively move tensors to ``device`` (dtypes preserved -- timestep stays int)."""
    import torch

    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_to_device(o, device) for o in obj)
    if isinstance(obj, list):
        return [_to_device(o, device) for o in obj]
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


def _extract_tensor(out: Any):
    import torch

    if torch.is_tensor(out):
        return out
    s = getattr(out, "sample", None)
    if torch.is_tensor(s):
        return s
    if isinstance(out, (list, tuple)) and out and torch.is_tensor(out[0]):
        return out[0]
    return out


# ── forward-tuple capture ────────────────────────────────────────────────────────
class _Stop(Exception):
    pass


def _capture_forward_tuple(
    pipe, *, steps, width, height, num_frames, guidance, seed, capture_call, gvg
):
    """Run a bf16 generation and grab the ``capture_call``-th transformer forward's
    (args, kwargs) to CPU, then abort. This is a REAL mid-schedule DiT input."""
    import torch

    holder: dict[str, Any] = {}
    calls = [0]

    def _pre(mod, args, kwargs):
        calls[0] += 1
        if calls[0] == capture_call:
            holder["args"] = _detach_to_cpu(args)
            holder["kwargs"] = _detach_to_cpu(kwargs)
            raise _Stop()
        return None

    handle = pipe.transformer.register_forward_pre_hook(_pre, with_kwargs = True)
    g = torch.Generator(device = "cuda").manual_seed(seed)
    kwargs = dict(
        prompt = PROMPT,
        width = width,
        height = height,
        num_frames = num_frames,
        num_inference_steps = steps,
        generator = g,
    )
    if gvg:
        guider = getattr(pipe, "guider", None)
        if guider is not None and hasattr(guider, "guidance_scale"):
            try:
                guider.guidance_scale = guidance
            except Exception:
                pass
    else:
        kwargs["guidance_scale"] = guidance
    try:
        pipe(**kwargs)
    except _Stop:
        pass
    finally:
        handle.remove()
    if "kwargs" not in holder:
        raise RuntimeError(
            f"transformer forward never reached call #{capture_call} "
            f"(saw {calls[0]}); lower --capture-call"
        )
    return holder


def _forward_out(module, tup: dict, device: str):
    import torch

    args = _to_device(tup.get("args", ()), device)
    kwargs = _to_device(tup.get("kwargs", {}), device)
    with torch.no_grad():
        out = module(*args, **kwargs)
    return _extract_tensor(out)


# ── scoring ──────────────────────────────────────────────────────────────────────
def _score(ref, cand) -> dict:
    """Compare a candidate DiT output to the bf16 reference. Non-finite entries are the
    strongest black-frame signal; cosine / relL2 / norm-ratio are computed on finite entries."""
    import torch

    reff = ref.detach().flatten().float()
    canf = cand.detach().flatten().float()
    fin_c = torch.isfinite(canf)
    frac_nonfinite = float(1.0 - fin_c.float().mean().item())
    mask = fin_c & torch.isfinite(reff)
    if mask.any():
        a = canf[mask]
        b = reff[mask]
        cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
        rel = float(
            (torch.linalg.vector_norm(a - b) / (torch.linalg.vector_norm(b) + 1e-12)).item()
        )
        norm_ratio = float(
            (torch.linalg.vector_norm(a) / (torch.linalg.vector_norm(b) + 1e-12)).item()
        )
    else:
        cos, rel, norm_ratio = 0.0, float("inf"), 0.0
    return {
        "cosine": round(cos, 5),
        "relL2": round(rel, 5),
        "frac_nonfinite": round(frac_nonfinite, 6),
        "norm_ratio": round(norm_ratio, 5),
        "cand_amax": round(
            float(canf[fin_c].abs().max().item()) if fin_c.any() else float("inf"), 3
        ),
    }


def _verdict(sc: dict) -> str:
    """Black-frame proxy verdict from a score dict."""
    if sc["frac_nonfinite"] > 0:
        return "BROKEN(non-finite)"
    if sc["cosine"] < 0.5 or sc["norm_ratio"] < 0.3 or sc["norm_ratio"] > 3.0:
        return "BROKEN(collapse)"
    if sc["cosine"] < 0.98:
        return "DEGRADED"
    return "CLEAN"


# ── quant helpers ──────────────────────────────────────────────────────────────
def _fp8_filter(
    min_features: int,
    exclude_tokens = (),
    only_tokens = (),
):
    """A production fp8 filter (require_bf16) optionally narrowed:
    - exclude_tokens: skip linears whose fqn contains any token (keep them bf16)
    - only_tokens   : quantise ONLY linears whose fqn contains any token."""
    base = make_filter_fn(
        min_features,
        exclude_name_tokens = exclude_tokens,
        require_bf16 = (TQ_FP8 in _REQUIRE_BF16_SCHEMES),
    )

    def filt(module, fqn: str = "") -> bool:
        if not base(module, fqn):
            return False
        name = (fqn or "").lower()
        if exclude_tokens and any(t in name for t in exclude_tokens):
            return False
        if only_tokens and not any(t in name for t in only_tokens):
            return False
        return True

    return filt


def _ablate(
    dense_cpu,
    tup,
    *,
    min_features,
    fast_accum = None,
    exclude_tokens = (),
    only_tokens = (),
    instrument = False,
):
    """Deepcopy the dense DiT -> GPU -> fp8-quantise the selected subset -> one forward.
    Returns (cpu_output_tensor, finiteness_records|None)."""
    import torch
    from torchao.quantization import quantize_

    m = copy.deepcopy(dense_cpu).to("cuda")
    filt = _fp8_filter(min_features, exclude_tokens = exclude_tokens, only_tokens = only_tokens)
    quantize_(m, _make_quant_config(TQ_FP8, fast_accum = fast_accum), filter_fn = filt)

    records = None
    hooks = []
    if instrument:
        records = []

        def mk(nm):
            def hook(mod, inp, out):
                o = (
                    out
                    if torch.is_tensor(out)
                    else (out[0] if isinstance(out, (list, tuple)) and out else None)
                )
                if o is None:
                    return
                of = o.detach().float()
                in_amax = None
                in_frac_zero_rows = None
                if inp and torch.is_tensor(inp[0]):
                    xin = inp[0].detach().float()
                    in_amax = float(xin.abs().max().item())
                    if xin.dim() >= 2:
                        ra = xin.reshape(-1, xin.shape[-1]).abs().amax(dim = -1)
                        in_frac_zero_rows = round(float((ra < 1e-6).float().mean().item()), 4)
                records.append(
                    {
                        "name": nm,
                        "out_finite": bool(torch.isfinite(of).all().item()),
                        "out_amax": float(of.abs().max().item())
                        if torch.isfinite(of).any()
                        else float("inf"),
                        "in_amax": in_amax,
                        "in_frac_zero_rows": in_frac_zero_rows,
                    }
                )

            return hook

        for nm, mod in m.named_modules():
            if isinstance(mod, torch.nn.Linear):
                hooks.append(mod.register_forward_hook(mk(nm)))

    out = _forward_out(m, tup, "cuda")
    out = out.detach().to("cpu")
    for h in hooks:
        h.remove()
    del m
    _empty()
    return out, records


# ── bf16 reference + input outlier stats (for Phase 2 ranking) ───────────────────
# ── Phase 1 buckets (fqn substring tokens), keyed by DiT kind ────────────────────
# Grounded in Phase 0: on Wan the inf originates in layers whose ROW dimension is the padded
# text sequence (condition_embedder.text_embedder + cross-attn K/V); the video attn1 + attn2.to_q
# + FFN stay finite. "textpath" is the hypothesised minimal exclude set.
_BUCKETS: dict[str, dict[str, tuple[str, ...]]] = {
    "wan": {
        "textpath": ("text_embedder", "attn2.to_k", "attn2.to_v", "attn2.add_k", "attn2.add_v"),
        "condition_embedder": ("condition_embedder",),
        "attn2_all": ("attn2.",),
        "attn2_kv": ("attn2.to_k", "attn2.to_v", "attn2.add_k", "attn2.add_v"),
        "attn1": ("attn1.",),
        "ffn": ("ffn.",),
    },
    "hunyuan": {
        # candidate fix: the input embedders consuming zero-padded conditioning (all-zero T2V
        # image_embeds, all-zero ByT5). "context_embedder" also matches "context_embedder_2".
        "embedders": ("context_embedder", "image_embedder"),
        "context_embedder": ("context_embedder",),  # text refiner + context_embedder_2
        "image_embedder": ("image_embedder",),
        "context_embedder_2": ("context_embedder_2",),
        "proj_out": ("proj_out",),
        "main_blocks": ("transformer_blocks.",),  # ONLY main blocks -> should be CLEAN
    },
}


def _dit_kind(family: str) -> str:
    return "hunyuan" if "hunyuan" in family.lower() else "wan"


def _zero_row_diag(tup: dict) -> dict:
    """Test the divide-by-zero hypothesis: fraction of near-zero (padding) rows. A per-row fp8
    scale = row_amax / 448, so row_amax 0 -> scale 0 -> x/0 = inf. Padded text is the suspect."""
    import torch

    diag = {}
    for key, val in tup.get("kwargs", {}).items():
        if torch.is_tensor(val) and val.dim() >= 2 and val.is_floating_point():
            x = val.float().reshape(-1, val.shape[-1])
            row_amax = x.abs().amax(dim = -1)
            n = int(row_amax.numel())
            n_zero = int((row_amax < 1e-6).sum().item())
            diag[key] = {
                "rows": n,
                "zero_amax_rows": n_zero,
                "frac_zero_rows": round(n_zero / n, 4) if n else 0.0,
                "min_row_amax": round(float(row_amax.min().item()), 8),
            }
    return diag


def _bf16_reference_and_stats(dense_gpu, tup, *, min_features):
    """Reference output + per-Linear input outlier stats on the bf16 forward. ``spread`` = mean
    over tokens of (channel amax / channel median-abs). Only fp8-quantised linears are recorded."""
    import torch

    keep = _fp8_filter(min_features)  # which linears fp8 would touch
    stats: list[dict] = []
    hooks = []

    def mk(nm):
        def hook(mod, inp, out):
            if not inp or not torch.is_tensor(inp[0]):
                return
            x = inp[0].detach().float()
            x2 = x.reshape(-1, x.shape[-1]).abs()  # [tokens, channels]
            amax_tok = x2.amax(dim = -1)
            med_tok = x2.median(dim = -1).values
            spread = float((amax_tok / (med_tok + 1e-9)).mean().item())
            stats.append(
                {
                    "name": nm,
                    "in_amax": float(x2.max().item()),
                    "spread": round(spread, 2),
                    "in_features": int(mod.in_features),
                    "out_features": int(mod.out_features),
                }
            )

        return hook

    for nm, mod in dense_gpu.named_modules():
        if isinstance(mod, torch.nn.Linear) and keep(mod, nm):
            hooks.append(mod.register_forward_hook(mk(nm)))

    ref = _forward_out(dense_gpu, tup, "cuda")
    ref = ref.detach().to("cpu")
    for h in hooks:
        h.remove()
    stats.sort(key = lambda s: s["spread"], reverse = True)
    return ref, stats


def main(argv = None) -> int:
    ap = argparse.ArgumentParser(description = __doc__)
    ap.add_argument("--family", default = "wan2.2-ti2v-5b", choices = sorted(_FAMILIES))
    ap.add_argument("--phase", type = int, default = 0, choices = [0, 1, 2])
    ap.add_argument("--only", default = "", help = "phase 2: comma tokens -- fp8 ONLY these linears")
    ap.add_argument("--exclude", default = "", help = "phase 2: comma tokens -- fp8 all EXCEPT these")
    ap.add_argument("--steps", type = int, default = 20)
    ap.add_argument("--num-frames", type = int, default = 25)
    ap.add_argument("--width", type = int, default = 512)
    ap.add_argument("--height", type = int, default = 320)
    ap.add_argument("--seed", type = int, default = 42)
    ap.add_argument(
        "--capture-call",
        type = int,
        default = 8,
        help = "which transformer forward call to capture (~mid schedule)",
    )
    ap.add_argument("--min-features", type = int, default = DEFAULT_MIN_LINEAR_FEATURES)
    ap.add_argument("--out", default = "outputs/fp8_ablation")
    args = ap.parse_args(argv)

    import torch

    out = Path(args.out)
    out.mkdir(parents = True, exist_ok = True)

    spec = _FAMILIES[args.family]
    repo = spec["repo"]
    force_fp32 = spec.get("vae_force_fp32", False)
    guidance = spec.get("guidance", 5.0)

    from core.inference.video_families import detect_video_family

    fam_obj = detect_video_family(repo)
    gvg = bool(getattr(fam_obj, "guidance_via_guider", False))

    print(
        f"== fp8 layer ablation: family={args.family} repo={repo} phase={args.phase} ==", flush = True
    )
    t0 = time.perf_counter()
    pipe = _build_pipe(repo, force_fp32)
    print(f"[load] pipe built in {time.perf_counter()-t0:.1f}s", flush = True)

    tup = _capture_forward_tuple(
        pipe,
        steps = args.steps,
        width = args.width,
        height = args.height,
        num_frames = args.num_frames,
        guidance = guidance,
        seed = args.seed,
        capture_call = args.capture_call,
        gvg = gvg,
    )
    ks = {
        k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__)
        for k, v in tup.get("kwargs", {}).items()
    }
    print(f"[capture] call #{args.capture_call} kwargs={ks}", flush = True)

    # Keep the dense bf16 DiT (still on GPU) as reference source; copy to CPU as the ablation seed.
    dense_gpu = pipe.transformer
    ref, stats = _bf16_reference_and_stats(dense_gpu, tup, min_features = args.min_features)
    print(
        f"[bf16 ref] out shape={tuple(ref.shape)} amax={float(ref.float().abs().max()):.3f} "
        f"n_quantized_linears={len(stats)}",
        flush = True,
    )

    dense_cpu = copy.deepcopy(dense_gpu).to("cpu")
    # Free the pipeline (VAE/text encoders) to leave the GPU for one transformer at a time.
    del pipe, dense_gpu
    _empty()

    report: dict[str, Any] = {
        "family": args.family,
        "repo": repo,
        "capture_call": args.capture_call,
        "input_shapes": ks,
        "ref_amax": round(float(ref.float().abs().max()), 3),
        "zero_row_diag": _zero_row_diag(tup),
    }
    print(f"[zero-row diag] {report['zero_row_diag']}", flush = True)

    if args.phase == 0:
        # ── Phase 0a: proxy validity -- fp8-ALL (production accum = None auto-detect) ──
        out_all, _ = _ablate(dense_cpu, tup, min_features = args.min_features, fast_accum = None)
        sc_all = _score(ref, out_all)
        report["fp8_all_autoaccum"] = {**sc_all, "verdict": _verdict(sc_all)}
        print(f"[0a fp8-ALL auto-accum] {sc_all} -> {_verdict(sc_all)}", flush = True)

        # ── Phase 0b-ii: accumulate flip ──
        for fa in (False, True):
            o, _ = _ablate(dense_cpu, tup, min_features = args.min_features, fast_accum = fa)
            sc = _score(ref, o)
            report[f"fp8_all_fast_accum_{fa}"] = {**sc, "verdict": _verdict(sc)}
            print(f"[0b accum={fa}] {sc} -> {_verdict(sc)}", flush = True)

        # ── Phase 0b-i: per-Linear finiteness (instrumented fp8-ALL forward) ──
        _, records = _ablate(
            dense_cpu, tup, min_features = args.min_features, fast_accum = None, instrument = True
        )
        if records:
            nonfinite = [r for r in records if not r["out_finite"]]
            report["first_nonfinite_linears"] = nonfinite[:10]
            report["n_nonfinite_linears"] = len(nonfinite)
            finite_sorted = sorted(
                [r for r in records if r["out_finite"]], key = lambda r: r["out_amax"], reverse = True
            )
            report["top_out_amax_linears"] = finite_sorted[:10]
            print(
                f"[0b finiteness] {len(nonfinite)}/{len(records)} linears non-finite; "
                f"first={[r['name'] for r in nonfinite[:5]]}",
                flush = True,
            )
            print(
                f"[0b top out_amax] {[(r['name'], round(r['out_amax'],1)) for r in finite_sorted[:5]]}",
                flush = True,
            )

        report["top_spread_linears"] = stats[:20]
        print(
            f"[stats] top-spread linears: " f"{[(s['name'], s['spread']) for s in stats[:8]]}",
            flush = True,
        )

        va = report["fp8_all_autoaccum"]["verdict"]
        if va.startswith("BROKEN"):
            print(
                "\n[GATE] fp8-ALL reproduces the black signal in one forward -> proxy VALID; "
                "proceed to Phase 1 bucket ablation.",
                flush = True,
            )
        elif va == "CLEAN":
            print(
                "\n[GATE] fp8-ALL is CLEAN in one forward but full gen is black -> failure is "
                "multi-step/cache; single-forward bisection cannot localise -> STOP, keep int8.",
                flush = True,
            )
        else:
            print(f"\n[GATE] fp8-ALL verdict={va}: borderline; inspect scores.", flush = True)

    elif args.phase == 1:  # ── Phase 1: bucket ablation (necessity + sufficiency) ──
        kind = _dit_kind(args.family)
        buckets = _BUCKETS[kind]
        results: dict[str, Any] = {}

        # baseline: fp8-ALL (should be BROKEN, matching Phase 0)
        o, _ = _ablate(dense_cpu, tup, min_features = args.min_features)
        results["fp8_all"] = {**_score(ref, o)}
        results["fp8_all"]["verdict"] = _verdict(results["fp8_all"])
        print(
            f"[1 baseline fp8-ALL] {results['fp8_all']['verdict']} {results['fp8_all']}", flush = True
        )

        for bname, toks in buckets.items():
            # NECESSITY: fp8 everything EXCEPT this bucket -- if CLEAN, this bucket is the culprit.
            o, _ = _ablate(dense_cpu, tup, min_features = args.min_features, exclude_tokens = toks)
            sc_ex = _score(ref, o)
            # SUFFICIENCY: fp8 ONLY this bucket -- if BROKEN, this bucket alone reproduces damage.
            o2, _ = _ablate(dense_cpu, tup, min_features = args.min_features, only_tokens = toks)
            sc_only = _score(ref, o2)
            results[bname] = {
                "exclude": {**sc_ex, "verdict": _verdict(sc_ex)},
                "only": {**sc_only, "verdict": _verdict(sc_only)},
            }
            print(
                f"[1 {bname}] EXCLUDE->{_verdict(sc_ex)} (cos {sc_ex['cosine']}, "
                f"nf {sc_ex['frac_nonfinite']}) | ONLY->{_verdict(sc_only)} "
                f"(cos {sc_only['cosine']}, nf {sc_only['frac_nonfinite']})",
                flush = True,
            )

        report["phase1"] = results

    else:  # ── Phase 2: instrument a specific only/exclude set, find first overflow ──
        only = tuple(t.strip() for t in args.only.split(",") if t.strip())
        exclude = tuple(t.strip() for t in args.exclude.split(",") if t.strip())
        o, records = _ablate(
            dense_cpu,
            tup,
            min_features = args.min_features,
            only_tokens = only,
            exclude_tokens = exclude,
            instrument = True,
        )
        sc = _score(ref, o)
        report["phase2"] = {"only": only, "exclude": exclude, **sc, "verdict": _verdict(sc)}
        print(f"[2 only={only} exclude={exclude}] {sc} -> {_verdict(sc)}", flush = True)
        if records:
            # first non-finite linears in execution order: an in_amax ~0 at the first-broken
            # layer confirms a zero-amax padding row -> scale 0 -> inf.
            nonfinite = [r for r in records if not r["out_finite"]]
            report["phase2_first_nonfinite"] = nonfinite[:15]
            report["phase2_n_nonfinite"] = len(nonfinite)
            print(
                f"[2 finiteness] {len(nonfinite)}/{len(records)} quantised-path linears non-finite",
                flush = True,
            )
            for r in nonfinite[:8]:
                print(
                    f"    first-inf {r['name']}  in_amax={r['in_amax']}  "
                    f"in_frac_zero_rows={r['in_frac_zero_rows']}  out_amax={r['out_amax']}",
                    flush = True,
                )

    dest = out / f"phase{args.phase}_{args.family}.json"
    with open(dest, "w", encoding = "utf-8") as fh:
        json.dump(report, fh, indent = 2)
    print(f"wrote {dest}", flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
