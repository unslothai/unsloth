# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Build a pre-quantized transformer checkpoint for the Unsloth diffusion fast path.

Quantise a model's dense bf16 DiT transformer ONCE and save the quantized state dict, so
the backend can load the already-quantized weights at runtime (meta-init +
load_state_dict(assign=True)) instead of materialising the dense bf16 on the GPU. That
drops the transformer GPU load peak ~2x and the download ~2x for fp8 (measured on Z-Image:
12.9 -> 6.3 GB peak, 12 -> 6.28 GB on disk), with bit-identical output -- it is the exact
same torchao config + min_features filter the runtime path uses, applied ahead of time.

Run on one CUDA (Blackwell / Ada / Hopper) GPU. fp8 works on torch 2.9+; the FP4/MX schemes
need the newer kernels (see scripts/nvfp4_t211_probe.py).

  python scripts/build_prequant_checkpoint.py \
      --base Tongyi-MAI/Z-Image-Turbo --family z-image --scheme fp8 \
      --out outputs/quant_research/prequant_fp8/transformer_fp8.pt [--upload-repo ORG/REPO]

Image and video families are both served: the image registry is asked first and the video one
is the fallback (--modality forces either). A family with more than one denoiser builds one
artifact per --component (Wan2.2 A14B's two experts).

An image family with a gated per-layer NVFP4 policy builds the MIXED artifact instead of a
whole-model one: --policy auto (the default) applies the policy core.inference.diffusion_nvfp4_policy
resolves for (family, base) when the scheme is nvfp4, --policy off forces the whole-model build, and
--policy <policy_id> pins one and refuses if that is not what resolves. A policy build runs two
quantize_ passes, stamps the layer assignment into the metadata and writes the v3 format tag.

A calibrated build takes --gptq-dir: every admitted linear whose GPTQ correction is MEASURED to
lower that layer's output error on held-out activations gets the corrected bf16 weight before
quantize_, the rest stay round-to-nearest, and the metadata records which was which. Under a policy
the corrections apply to the NVFP4 layers alone, which is the rule the campaign measured: a
correction that also becomes the source of an fp8 replica raised the error 46 percent.

A build can also run its OWN calibration instead of replaying one: --gptq-prompts N accumulates
per-layer Hessians on the layers this build quantises to 4 bits, corrects them onto the NVFP4 grid
and keeps each correction only where it lowers that layer's output error through torchao's own
quantiser, and --bake-activation-scales measures each of those layers' activation amax over the
same prompts and stores a_gsf = 6 * 448 / amax per layer. The two are independent: the bake alone
is what an artifact needs to run on the flashinfer NVFP4 backend, which refuses to calibrate a
scale at run time. Both run on the DENSE pipeline, before quantize_, in that order.

  python scripts/build_prequant_checkpoint.py --base ... --family z-image --scheme nvfp4 \
      --policy auto --bake-activation-scales --gptq-prompts 32 --out a.pt

A VIDEO family calibrates the same way, through its own pipeline and its own grid: --calib-resolution
reads as WxHxFRAMES there (default 832x480x25) and --calib-steps defaults to 20 rather than the
family's shipped schedule, because an activation amax converges long before a render does and a
calibration at the shipped 1280x704x121x50 costs hours per artifact for the same scales. A MoE
family calibrates ONE expert per build (--component): the hooks sit on that expert alone and the
pipeline's own boundary switch decides which steps reach it, so each expert's scales are measured on
exactly the steps it runs.

  python scripts/build_prequant_checkpoint.py --base <local Wan2.2-T2V-A14B> --modality video \
      --family wan2.2-t2v-a14b --scheme nvfp4 --component transformer_2 \
      --base-id Wan-AI/Wan2.2-T2V-A14B-Diffusers --gptq-dir outputs/... \
      --bake-activation-scales --out b.pt

Publishing is gated on a SECOND build: every checkpoint records an md5 fingerprint of each
quantized weight's packed payload, --verify-against diffs this build against another one, and
--upload-repo is refused unless that diff ran and matched. A build is hours of GPU time and a
multi-gigabyte upload that then renders for everyone, and a silently corrupted tensor in it
looks exactly like a good one until the pixels are wrong.

  python scripts/build_prequant_checkpoint.py --base Wan-AI/Wan2.2-T2V-A14B-Diffusers \
      --family wan2.2-t2v-a14b --scheme nvfp4 --component transformer_2 \
      --out b.pt --verify-against a.pt --upload-repo unsloth/Wan2.2-T2V-A14B-NVFP4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

BACKEND = Path(__file__).resolve().parent.parent / "studio" / "backend"

# Mirrors core.inference.diffusion_prequant.DEFAULT_PREQUANT_COMPONENT, which cannot be imported this early: the backend only joins sys.path inside main().
DEFAULT_COMPONENT = "transformer"


def convrot_refusal(
    group: int, rotatable: Sequence[str], not_divisible: Sequence[str]
) -> Optional[str]:
    """Why a ConvRot build must not be quantised and saved, or None when it is fine.

    An empty rotatable set means the group divides no quantized input axis (a group larger than
    every Linear, say). The build would still stamp the v2 tag and an empty fqn list, which
    ``rotation_metadata_error`` refuses at load time, so the only thing it produces is a
    multi-gigabyte artifact nothing can ever open. Refuse before the hours, not after."""
    if rotatable:
        return None
    return (
        f"ConvRot group {group} divides the in_features of none of the {len(not_divisible)} "
        "quantized linears, so the checkpoint would record an empty rotation and be refused at "
        "load time. Pick a smaller power-of-4 group, or drop --convrot-groupsize."
    )


def resolve_build_family(
    base: str,
    override: Optional[str] = None,
    modality: str = "auto",
) -> Optional[Any]:
    """The registry row this build quantises against: a ``DiffusionFamily`` or a ``VideoFamily``.
    
    Asked image-first, and ``modality`` pins one registry when the answer must not drift: the two
    name spaces are disjoint today, and a later family named in both would otherwise build against
    whichever is asked first."""
    from core.inference.diffusion_families import detect_family
    from core.inference.video_families import detect_video_family

    if modality != "video":
        fam = detect_family(base, override = override)
        if fam is not None or modality == "image":
            return fam
    return detect_video_family(base, override = override)


POLICY_AUTO = "auto"
POLICY_OFF = "off"


def resolve_build_policy(
    mode: Optional[str], scheme: str, family: Optional[str], base_id: Optional[str]
) -> tuple:
    """``(policy, refusal)`` for this build. A NAMED policy is a pin rather than a lookup: a moved
    table refuses instead of producing a differently-quantised artifact."""
    from core.inference.diffusion_nvfp4_policy import policy_by_id, resolve_policy
    from core.inference.diffusion_transformer_quant import TQ_NVFP4

    mode = (mode or POLICY_AUTO).strip()
    if mode == POLICY_OFF:
        return None, None
    if scheme != TQ_NVFP4:
        if mode == POLICY_AUTO:
            return None, None
        return None, (f"--policy {mode!r} describes an nvfp4 build, but --scheme is {scheme!r}")
    resolved = resolve_policy(family, base_id)
    if mode == POLICY_AUTO:
        return resolved, None
    if policy_by_id(mode) is None:
        return None, (
            f"unknown --policy {mode!r}; pass a policy id from "
            "core/inference/diffusion_nvfp4_policy.py, or 'auto' / 'off'"
        )
    if resolved is None or resolved.policy_id != mode:
        return None, (
            f"--policy {mode!r} is not the policy this build resolves for family {family!r} on "
            f"base {base_id!r} ({resolved.policy_id if resolved else 'none'}). The layer sets were "
            "solved on one checkpoint's weights, so a policy is never applied to a base it was not "
            "gated on."
        )
    return resolved, None


def upload_destination(
    fam: Any,
    scheme: str,
    *,
    rotated: bool,
    override: Optional[str] = None,
    repo_id: Optional[str] = None,
    component: str = DEFAULT_COMPONENT,
) -> str:
    """The repo-root filename this build should publish under.

    The loader asks for the family's declared ``prequant_filenames`` name first and the derived
    ``<Model>-<SCHEME>.pt`` second, so a ROTATED artifact published under the legacy
    ``transformer_<scheme>.pt`` is either never resolved at all, or resolved as the fallback by a
    build too old to honour the rotation, which then refuses the v2 tag and drops to the dense
    download. A rotated build therefore goes to the declared name or nowhere.

    A plain build publishes under the derived ``<Model>-<SCHEME>.pt``, which is the name the
    loader asks for FIRST and the layout every hosted prequant repo already uses; the legacy
    ``transformer_<scheme>.pt`` stays resolvable as the loader's fallback for the repos that only
    ever carried it. A non-default ``--component`` becomes part of the name
    (``Wan2.2-T2V-A14B-transformer_2-NVFP4.pt``): the second expert is a different set of weights
    under the same family, scheme and base, so one name per component is the only thing keeping a
    resolver from serving expert 1 where expert 2 was asked for."""
    if override:
        return override
    if not rotated:
        from core.inference.diffusion_prequant import prequant_repo_filename
        if not repo_id:
            raise ValueError(
                "a plain build's filename derives from the destination repo, so publishing "
                "needs --upload-repo (or an explicit --upload-filename)"
            )
        return prequant_repo_filename(repo_id, scheme, component = component)
    from core.inference.diffusion_families import family_prequant_filename

    preferred = family_prequant_filename(fam, scheme)
    if not preferred:
        raise ValueError(
            f"family {getattr(fam, 'name', fam)!r} declares no prequant_filenames entry for "
            f"{scheme!r}, so a rotated checkpoint has no name the loader would ask for. Add the "
            "entry to the family table, or pass --upload-filename."
        )
    return preferred



DEFAULT_GPTQ_STEPS = "0,12,25,37"

DEFAULT_CALIB_PROMPTS = str(Path(__file__).resolve().parent / "gptq_prompts.py")


DEFAULT_IMAGE_CALIB_GRID = "1024"
DEFAULT_VIDEO_CALIB_GRID = "832x480x25"

# Deliberately not the family's shipped schedule: a second moment and an amax converge over the
# trajectory rather than over its length, and the shipped one is hours per artifact.
DEFAULT_VIDEO_CALIB_STEPS = 20


def is_video_family(fam: Any) -> bool:
    """True when this build resolved a ``VideoFamily``. Asked by type: the two dataclasses share
    most of the names the builder reads."""
    from core.inference.video_families import VideoFamily
    return isinstance(fam, VideoFamily)


def parse_calib_grid(spec: Optional[str], *, video: bool) -> tuple:
    """``--calib-resolution`` -> ``(width, height, frames or None)``. An image build handed a frame
    count is refused rather than quietly rendering a still."""
    default = DEFAULT_VIDEO_CALIB_GRID if video else DEFAULT_IMAGE_CALIB_GRID
    text = str(spec if spec is not None else default)
    parts = text.strip().lower().replace("*", "x").split("x")
    if not (1 <= len(parts) <= 3) or not all(part.strip().isdigit() for part in parts):
        raise ValueError(
            f"--calib-resolution takes a square size, WxH, or WxHxFRAMES for a video family, "
            f"not {spec!r}"
        )
    values = [int(part) for part in parts]
    if any(value <= 0 for value in values):
        raise ValueError(f"--calib-resolution must be positive, not {spec!r}")
    if len(values) == 1:
        width = height = values[0]
        frames = None
    else:
        width, height = values[0], values[1]
        frames = values[2] if len(values) == 3 else None
    if frames is not None and not video:
        raise ValueError(
            f"--calib-resolution {spec!r} names a frame count, but this build's family renders "
            "images"
        )
    if video and frames is None:
        frames = int(DEFAULT_VIDEO_CALIB_GRID.split("x")[2])
    return width, height, frames


def frame_count_refusal(fam: Any, frames: Optional[int]) -> Optional[str]:
    """Why ``frames`` is off this family's ``k * frame_step + frame_offset`` lattice, or None. An
    off-lattice count is otherwise silently snapped to one the metadata does not record."""
    if frames is None:
        return None
    step = int(getattr(fam, "frame_step", 1) or 1)
    offset = int(getattr(fam, "frame_offset", 1) or 0)
    minimum = int(getattr(fam, "min_num_frames", 1) or 1)
    if frames < minimum:
        return (
            f"--calib-resolution asks for {frames} frames, below family {fam.name!r}'s minimum "
            f"of {minimum}"
        )
    if step > 1 and (frames - offset) % step:
        return (
            f"--calib-resolution asks for {frames} frames, which family {fam.name!r} cannot "
            f"render: its frame count is k * {step} + {offset}"
        )
    return None


def parse_step_spec(spec: str) -> tuple:
    """``"0,12,25,37"`` -> ``(0, 12, 25, 37)``. Raises on anything that is not a step index."""
    steps: list = []
    for piece in str(spec or "").replace(" ", "").split(","):
        if not piece:
            continue
        if not piece.isdigit():
            raise ValueError(f"--gptq-steps takes comma-separated step indices, not {piece!r}")
        steps.append(int(piece))
    if not steps:
        raise ValueError("--gptq-steps must name at least one step index")
    return tuple(sorted(set(steps)))


def load_calibration_prompts(path: Optional[str] = None) -> tuple:
    """The calibration prompts from ``path``: a .py file read for ``CALIBRATION_PROMPTS``, any
    other file one prompt per line with blanks and ``#`` comments skipped."""
    source = str(path or DEFAULT_CALIB_PROMPTS)
    if source.endswith(".py"):
        import importlib.util

        spec = importlib.util.spec_from_file_location("unsloth_calib_prompts", source)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot read the calibration prompts at {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        prompts = tuple(
            getattr(module, "CALIBRATION_PROMPTS", None) or getattr(module, "CALIB", ()) or ()
        )
    else:
        with open(source) as handle:
            prompts = tuple(
                line.strip() for line in handle if line.strip() and not line.strip().startswith("#")
            )
    if not prompts:
        raise ValueError(f"{source} defines no calibration prompts")
    if len(set(prompts)) != len(prompts):
        # A repeated prompt is counted twice in a Hessian and shifts it toward that prompt.
        raise ValueError(f"{source} repeats a calibration prompt")
    return prompts


def calibration_stage_order(gptq_prompts: int, bake: bool) -> tuple:
    """The stages between the assignment and ``quantize_``, in order. Fixed and asserted: Hessians
    on the UNCORRECTED weights, activation scales on the corrected ones the artifact ships."""
    stages: list = []
    if int(gptq_prompts) > 0:
        stages += ["hessians", "gptq"]
    if bake:
        stages.append("bake")
    return tuple(stages)


def calibration_refusal(
    *,
    scheme: str,
    nvfp4: str,
    gptq_prompts: int,
    bake: bool,
    gptq_dir: Optional[str],
    convrot_groupsize: int,
    available_prompts: int,
) -> Optional[str]:
    """Why the calibration flags cannot be honoured, or None. Decided before the dense load."""
    if not (int(gptq_prompts) > 0 or bake):
        return None
    if scheme != nvfp4:
        return (
            f"--gptq-prompts / --bake-activation-scales describe an nvfp4 build (a 4-bit grid and "
            f"a 4-bit activation scale), but --scheme is {scheme!r}"
        )
    if gptq_dir and int(gptq_prompts) > 0:
        return (
            "--gptq-prompts and --gptq-dir are two sources for the same corrected weights. Pass "
            "one: --gptq-dir replays a calibration that already ran, --gptq-prompts runs one here."
        )
    if convrot_groupsize and int(gptq_prompts) > 0:
        return (
            "--gptq-prompts and --convrot-groupsize cannot be combined: the correction is solved "
            "against unrotated activations, so rotating the corrected weight discards it."
        )
    if int(gptq_prompts) > available_prompts:
        return (
            f"--gptq-prompts {gptq_prompts} exceeds the {available_prompts} prompts the "
            "calibration file defines"
        )
    return None


def render_calibration(
    pipe: Any,
    prompts: Sequence[str],
    *,
    steps: int,
    guidance: float,
    cfg_kwarg: str = "guidance_scale",
    width: int = 1024,
    height: int = 1024,
    num_frames: Optional[int] = None,
    guidance_via_guider: bool = False,
    cfg2_kwarg: Optional[str] = None,
    guidance_2: Optional[float] = None,
    seed: int = 3407,
    device: str = "cuda",
    callback: Any = None,
    before_prompt: Any = None,
    on_prompt: Any = None,
) -> int:
    """Run ``prompts`` through ``pipe`` for their activations alone. Returns how many ran. Every
    render is seeded, since two builds must produce the same Hessians for the byte-for-byte
    reproduction gate; ``before_prompt`` covers step 0, which the pipeline callback misses."""
    import inspect as _inspect

    import torch

    # A ``**kwargs`` signature says nothing, so nothing is filtered rather than everything.
    kwargs_supported: set = set()
    try:
        parameters = _inspect.signature(pipe.__call__).parameters
        if not any(param.kind is _inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
            kwargs_supported = set(parameters)
    except Exception:  # noqa: BLE001 - a stub or a wrapped pipeline: pass what we have
        kwargs_supported = set()
    if callback is not None and kwargs_supported and "callback_on_step_end" not in kwargs_supported:
        raise ValueError(
            f"{type(pipe).__name__} takes no callback_on_step_end, so the Hessian pass cannot be "
            "gated to sampled steps; calibrate this family with --gptq-prompts 0"
        )
    if guidance_via_guider:
        # No guidance kwarg on this pipeline; the scale is an attribute of its guider.
        try:
            pipe.guider.guidance_scale = float(guidance)
        except Exception as exc:  # noqa: BLE001 - a family that declares a guider must have one
            raise ValueError(
                f"{type(pipe).__name__} declares guidance_via_guider but its guider scale could "
                f"not be set ({type(exc).__name__}: {exc})"
            ) from exc
    ran = 0
    for index, prompt in enumerate(prompts):
        if before_prompt is not None:
            before_prompt()
        call_kwargs: dict = {
            "prompt": prompt,
            "num_inference_steps": int(steps),
            "width": int(width),
            "height": int(height),
            "generator": torch.Generator(device = device).manual_seed(int(seed) + index),
            "output_type": "latent",
        }
        if num_frames is not None:
            call_kwargs["num_frames"] = int(num_frames)
        if cfg_kwarg and not guidance_via_guider:
            call_kwargs[cfg_kwarg] = float(guidance)
        if cfg2_kwarg and guidance_2 is not None:
            call_kwargs[cfg2_kwarg] = float(guidance_2)
        if callback is not None:
            call_kwargs["callback_on_step_end"] = callback
        if kwargs_supported:
            call_kwargs = {k: v for k, v in call_kwargs.items() if k in kwargs_supported}
        with torch.no_grad():
            pipe(**call_kwargs)
        ran += 1
        if on_prompt is not None:
            on_prompt(index, prompt)
    return ran


def prompt_digest(prompts: Sequence[str]) -> str:
    """A stable sha256 over the calibration set, so an artifact says which prompts made it."""
    import hashlib

    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(prompt.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def gptq_metadata_block(
    *,
    prompts: Sequence[str],
    steps_sampled: Sequence[int],
    schedule_steps: int,
    max_regressions: int,
    plan: Mapping,
    scores: Mapping,
    damps: Mapping,
    seconds: float,
) -> dict:
    """The per-layer ``gptq`` metadata block: which weights are corrected and which are RTN."""
    applied = set(plan.get("apply") or ())
    layers = {
        fqn: {
            "err_rtn": score.get("err_rtn"),
            "err_gptq": score.get("err_gptq"),
            "ratio": score.get("ratio"),
            "damp": damps.get(fqn),
            "applied": fqn in applied,
            "reason": "applied" if fqn in applied else "no_gain",
        }
        for fqn, score in sorted(scores.items())
    }
    counts = dict(plan.get("counts") or {})
    return {
        "source": "in-builder",
        "score_mode": "hessian_output_error",
        "prompts": len(prompts),
        "prompt_sha256": prompt_digest(prompts),
        "schedule_steps": int(schedule_steps),
        "steps_sampled": [int(step) for step in steps_sampled],
        "max_regressions": int(max_regressions),
        "applied": int(counts.get("applied", 0)),
        "applied_regressed": int(counts.get("applied_regressed", 0)),
        "skipped_no_gain": int(counts.get("skipped_no_gain", 0)),
        "scored": len(scores),
        "seconds": round(float(seconds), 1),
        "layers": layers,
    }


def activation_scale_metadata(
    *,
    prompts: Sequence[str],
    schedule_steps: int,
    scales: Mapping,
    layers: int,
    grid: Optional[str] = None,
) -> dict:
    """What the baked activation scales were measured on: a scale is a property of a calibration
    set and a schedule, not of the model alone."""
    values = sorted(float(value) for value in scales.values())
    return {
        "prompts": len(prompts),
        "prompt_sha256": prompt_digest(prompts),
        "schedule_steps": int(schedule_steps),
        # An activation scale is a property of a shape as much as of a prompt set.
        "grid": grid,
        # Every step: the largest-activation step is the one a sampled subset would miss.
        "steps_sampled": "all",
        "layers": int(layers),
        "scaled": len(values),
        "min_a_gsf": values[0] if values else None,
        "max_a_gsf": values[-1] if values else None,
    }


# GPTQ scoring modes. "check" decides per layer on the OUTPUT error GPTQ optimises; "meta" decides on the Frobenius
# WEIGHT error, which GPTQ raises by construction and which therefore admits nothing.
GPTQ_SCORE_MODES = ("check", "meta")


def gptq_weight_filename(fqn: str) -> str:
    """The per-layer file the GPTQ pass wrote: dots become underscores, as it saved them."""
    return fqn.replace(".", "_") + ".pt"


def gptq_sources(
    gptq_dir: str,
    component: str = DEFAULT_COMPONENT,
    *,
    meta_override: Optional[str] = None,
    score_override: Optional[str] = None,
    exists = os.path.exists,
) -> dict:
    """Where one component's GPTQ weights, Hessian meta and do-no-harm scores live.
    
    A MoE family writes per-component paths and a single-denoiser family does not; both layouts are
    probed rather than declared, so the same --gptq-dir serves either."""
    root = str(gptq_dir).rstrip("/")
    per_component = os.path.join(root, "weights", component)
    weights = per_component if exists(per_component) else os.path.join(root, "weights")
    meta = meta_override
    if not meta:
        named = os.path.join(root, f"gptq_meta_{component}.json")
        meta = named if exists(named) else os.path.join(root, "gptq_meta.json")
    score = score_override
    if not score:
        for candidate in (
            f"gptq_score_{component}.json",
            "gptq_score.json",
            f"gptq_check_{component}.json",
            "gptq_check.json",
        ):
            path = os.path.join(root, candidate)
            if exists(path):
                score = path
                break
    return {"weights": weights, "meta": meta, "score": score}


def plan_gptq(
    fqns: Sequence[str],
    meta_layers: dict,
    score_layers: dict,
    *,
    mode: str = "check",
    has_weight = lambda fqn: True,
) -> dict:
    """Which admitted linears take their GPTQ weight, and why each of the rest does not.
    
    Do no harm, per layer: a correction is applied only where it is MEASURED to help. ``missing``
    must never pass silently, or a layer whose file the pass never wrote is indistinguishable from
    a layer that was corrected."""
    if mode not in GPTQ_SCORE_MODES:
        raise ValueError(f"gptq score mode must be one of {GPTQ_SCORE_MODES}, not {mode!r}")
    layers: dict = {}
    counts = {"applied": 0, "skipped_no_gain": 0, "skipped_unscored": 0, "missing": 0}
    apply: list = []
    for fqn in fqns:
        meta = dict(meta_layers.get(fqn) or {})
        score = dict(score_layers.get(fqn) or {})
        record = {
            "err_rtn": meta.get("err_rtn"),
            "err_gptq": meta.get("err_gptq"),
            "damp": meta.get("damp"),
            "out_err_rtn": score.get("out_err_rtn"),
            "out_err_gptq": score.get("out_err_gptq"),
            "applied": False,
        }
        if not has_weight(fqn):
            record["reason"] = "missing"
            counts["missing"] += 1
            layers[fqn] = record
            continue
        if mode == "check":
            pair = (record["out_err_gptq"], record["out_err_rtn"])
        else:
            pair = (record["err_gptq"], record["err_rtn"])
        if pair[0] is None or pair[1] is None:
            record["reason"] = "unscored"
            counts["skipped_unscored"] += 1
        elif pair[0] < pair[1]:
            record["applied"] = True
            record["reason"] = "applied"
            counts["applied"] += 1
            apply.append(fqn)
        else:
            record["reason"] = "no_gain"
            counts["skipped_no_gain"] += 1
        layers[fqn] = record
    return {"apply": apply, "layers": layers, "counts": counts, "mode": mode}


def verify_gptq_idempotency(
    modules: dict,
    load_weight,
    *,
    sample: int = 0,
) -> dict:
    """Did ``quantize_`` keep the GPTQ weights it was handed, or re-round them?
    
    GPTQ writes a weight that already lies on the NVFP4 grid, so re-quantising it should reproduce
    it; anything else means the quantiser and the pass disagree about the grid. Reported, never
    asserted away."""
    import torch

    worst = {"max_abs": 0.0, "fqn": None}
    diff_elems = 0
    total_elems = 0
    layers = list(modules.items())
    if sample and sample < len(layers):
        layers = layers[:: max(1, len(layers) // sample)]
    for fqn, module in layers:
        weight = module.weight
        packed = (
            weight.dequantize(torch.float32)
            if hasattr(weight, "dequantize")
            else weight.detach().float()
        )
        want = load_weight(fqn).to(packed.device, torch.float32)
        delta = (packed - want).abs()
        max_abs = float(delta.max())
        if max_abs > worst["max_abs"]:
            worst = {"max_abs": max_abs, "fqn": fqn}
        diff_elems += int((delta != 0).sum())
        total_elems += int(delta.numel())
        del packed, want, delta
    return {
        "checked": len(layers),
        "max_abs": worst["max_abs"],
        "max_abs_fqn": worst["fqn"],
        "frac_diff": (diff_elems / total_elems) if total_elems else 0.0,
    }


def parse_key(key: str) -> tuple:
    """``'transformer_2/blocks.12.attn1.to_q.weight'`` -> ``(component, block or None, role)``.
    
    WHERE two builds differ discriminates between mechanisms that a count cannot. A bare
    state-dict fqn carries no component prefix, so that half is None."""
    component, sep, field = key.partition("/")
    if not sep:
        component, field = None, key
    parts = field.split(".")
    if parts[0] == "blocks" and len(parts) > 2 and parts[1].isdigit():
        return component, int(parts[1]), ".".join(parts[2:])
    return component, None, field


def describe_key(key: str) -> str:
    """One differing fqn with its parsed position, for the --verify-against report."""
    component, block, role = parse_key(key)
    where = f"block {block}" if block is not None else "no block"
    prefix = f"{component}, " if component else ""
    return f"{key}  ({prefix}{where}, role {role})"


def fingerprint_mismatches(mine: Any, other: Any) -> list:
    """The fqns whose packed payload differs between two builds of one artifact, sorted.
    
    A fqn present in one build and absent from the other counts as differing: a build that
    quantised a different SET of linears is not the artifact the other verified either."""
    a = (mine or {}).get("modules") or {}
    b = (other or {}).get("modules") or {}
    return sorted(key for key in set(a) | set(b) if a.get(key) != b.get(key))


def _read_metadata(path: str) -> dict:
    """The other artifact's metadata, through the loader's own restricted read."""
    from core.inference.diffusion_prequant import read_prequant_metadata
    return read_prequant_metadata(path)


def verify_against(
    other_path: str,
    fingerprint: Any,
    *,
    out: Any = print,
) -> int:
    """Diff this build's fingerprint against ``other_path``'s. 0 when identical, 3 otherwise.
    
    Two independent quantise passes over the same weights are deterministic, so any difference is
    a defect in one of them. It cannot see corruption that happens AFTER the compare, which is what
    the loader's own fingerprint check is for."""
    try:
        other = (_read_metadata(other_path) or {}).get("fingerprint") or {}
    except Exception as exc:  # noqa: BLE001 -- an unreadable comparand verifies nothing
        out(f"error: cannot read the fingerprint of {other_path}: {exc}")
        return 3
    if not other.get("modules"):
        out(
            f"error: {other_path} carries no fingerprint block, so it cannot verify this build. "
            "Rebuild it with this script."
        )
        return 3
    mine_count = int((fingerprint or {}).get("count") or 0)
    other_count = int(other.get("count") or 0)
    if mine_count != other_count:
        out(f"  fingerprint count {mine_count} != {other_count} in {other_path}")
    diffs = fingerprint_mismatches(fingerprint, other)
    if not diffs:
        out(f"  verified: {mine_count} quantized weights identical to {other_path}")
        return 0
    out(f"error: {len(diffs)} quantized weights differ from {other_path}:")
    for key in diffs[:50]:
        out(f"    {describe_key(key)}")
    if len(diffs) > 50:
        out(f"    ... and {len(diffs) - 50} more")
    return 3


def verify_target_refusal(out_path: str, verify_path: Optional[str]) -> Optional[str]:
    """Why ``--verify-against`` cannot answer the question it exists for, or None.
    
    One file compared with itself matches by construction, so a gate that accepts it reports a
    verified build and publishes it."""
    if not verify_path:
        return None
    try:
        same = os.path.realpath(out_path) == os.path.realpath(verify_path)
    except Exception:  # noqa: BLE001 -- an unresolvable path is not provably the same file
        return None
    if not same:
        return None
    return (
        f"--verify-against {verify_path} resolves to the same file as --out {out_path}. The "
        "point of the check is a SECOND build in a separate process; comparing an artifact with "
        "itself always matches."
    )


def upload_gate_refusal(upload_repo: Optional[str], verify_path: Optional[str]) -> Optional[str]:
    """Why this build may not publish, or None. No escape hatch by design.
    
    An unverified artifact is indistinguishable from a verified one once it is hosted, and it is
    then loaded by every auto pick that resolves the repo."""
    if not upload_repo:
        return None
    if not verify_path:
        return (
            "--upload-repo needs --verify-against <other.pt>: a checkpoint is published only "
            "after a second, independent build of the same artifact reproduces it byte for byte."
        )
    return None


def main(argv = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base", required = True, help = "diffusers base repo (carries the transformer subfolder)"
    )
    p.add_argument("--family", required = True, help = "diffusion family name/alias (e.g. z-image)")
    p.add_argument("--scheme", required = True, help = "quant scheme: int8 | fp8 | nvfp4 | mxfp8")
    p.add_argument("--out", required = True, help = "output .pt path for the checkpoint")
    p.add_argument(
        "--modality",
        default = "auto",
        choices = ["auto", "image", "video"],
        help = "which family registry to resolve --family in; auto asks the image one first and "
        "falls back to the video one",
    )
    p.add_argument(
        "--component",
        default = DEFAULT_COMPONENT,
        help = "denoiser subfolder inside --base to quantise (a MoE video family has a second "
        "expert in transformer_2); stamped into the metadata and the published filename",
    )
    p.add_argument("--min-features", type = int, default = 512)
    p.add_argument("--dtype", default = "bfloat16", choices = ["bfloat16"])
    p.add_argument("--hf-token", default = None)
    p.add_argument(
        "--convrot-groupsize",
        type = int,
        default = 0,
        help = "bake a ConvRot block-Hadamard activation rotation at this group size (a power of "
        "4; 0 = off). Every quantized Linear whose in_features the group divides has its "
        "weight rotated before quantize_ so the quantizer sees a flatter distribution; the "
        "exact fqn list is recorded in the checkpoint and the loader rotates the "
        "activations of that list and nothing else. Writes the v2 format tag.",
    )
    p.add_argument(
        "--policy",
        default = POLICY_AUTO,
        help = "per-layer NVFP4 policy: 'auto' applies the one diffusion_nvfp4_policy resolves for "
        "(--family, --base-id or --base) when --scheme is nvfp4 and builds the whole-model "
        "artifact otherwise, 'off' forces the whole-model build, and a policy id pins one and "
        "refuses if that is not what resolves",
    )
    p.add_argument(
        "--gptq-dir",
        default = None,
        help = "directory of GPTQ-corrected bf16 weights (as written by the calibration pass: "
        "weights/<fqn with dots as underscores>.pt, or weights/<component>/... for a MoE family). "
        "Every admitted linear whose correction is MEASURED to help takes it before quantize_",
    )
    p.add_argument(
        "--gptq-meta",
        default = None,
        help = "the calibration pass's meta json; defaults to gptq_meta_<component>.json inside "
        "--gptq-dir when that exists and gptq_meta.json otherwise",
    )
    p.add_argument(
        "--gptq-score",
        default = None,
        help = "the do-no-harm score json (per layer out_err_rtn / out_err_gptq); defaults to "
        "gptq_score_<component>.json, gptq_score.json or gptq_check.json inside --gptq-dir",
    )
    p.add_argument(
        "--gptq-prompts",
        type = int,
        default = 0,
        help = "run a GPTQ calibration IN this build over the first N prompts of --calib-prompts "
        "(0 = off). Hessians are accumulated on the layers this build quantises to 4 bits, the "
        "correction is applied only where it lowers that layer's output error through torchao's "
        "own quantiser, and the metadata records both errors per layer",
    )
    p.add_argument(
        "--gptq-steps",
        default = DEFAULT_GPTQ_STEPS,
        help = "denoise steps to accumulate the Hessians on, for a 38-step schedule; a shorter "
        "schedule is sampled at the same places (0, n/4, n/2, 3n/4) rather than the same indices",
    )
    p.add_argument(
        "--gptq-max-regressions",
        type = int,
        default = 0,
        help = "how many layers may take their correction despite scoring WORSE than "
        "round-to-nearest, least harmful first. 0 is do no harm: a correction is applied only "
        "where it is measured to help",
    )
    p.add_argument(
        "--calib-prompts",
        default = None,
        help = f"calibration prompts for --gptq-prompts and --bake-activation-scales; a .py file "
        f"is read for CALIBRATION_PROMPTS and anything else is one prompt per line. Defaults to "
        f"{DEFAULT_CALIB_PROMPTS}, which is disjoint from the accuracy gate's evaluation suite",
    )
    p.add_argument(
        "--bake-activation-scales",
        action = "store_true",
        help = "measure each 4-bit layer's activation amax over the calibration prompts and store "
        "a_gsf = 6 * 448 / amax per layer in the metadata. The flashinfer NVFP4 backend refuses an "
        "artifact without baked scales, because calibrating them at run time is neither "
        "capture-safe nor deterministic",
    )
    p.add_argument(
        "--bake-prompts",
        type = int,
        default = 8,
        help = "how many calibration prompts the activation-scale bake runs (an amax converges far "
        "faster than a Hessian); every step of each is measured",
    )
    p.add_argument(
        "--calib-steps",
        type = int,
        default = 0,
        help = "denoise steps per calibration render; 0 takes the family's default schedule for "
        f"an image family and {DEFAULT_VIDEO_CALIB_STEPS} for a video one, whose shipped schedule "
        "costs hours per artifact for the same per-layer moments",
    )
    p.add_argument(
        "--calib-resolution",
        default = None,
        help = "grid the calibration renders run at: a square size or WxH for an image family, "
        f"and WxHxFRAMES for a video one (defaults {DEFAULT_IMAGE_CALIB_GRID} and "
        f"{DEFAULT_VIDEO_CALIB_GRID})",
    )
    p.add_argument(
        "--calib-seed",
        type = int,
        default = 3407,
        help = "seed of the first calibration render (each later one adds its index). Fixed so a "
        "second build reproduces the same corrections byte for byte",
    )
    p.add_argument(
        "--gptq-score-mode",
        default = "check",
        choices = list(GPTQ_SCORE_MODES),
        help = "which error decides do-no-harm: 'check' the held-out OUTPUT error (the one GPTQ "
        "optimises), 'meta' the Frobenius WEIGHT error, which GPTQ raises by construction",
    )
    p.add_argument(
        "--verify-against",
        default = None,
        help = "another build of this same artifact to diff the packed-weight fingerprint "
        "against; exits 3 on any difference. Required before --upload-repo",
    )
    p.add_argument(
        "--upload-repo", default = None, help = "optional HF repo id to upload the checkpoint to"
    )
    p.add_argument(
        "--base-id",
        default = None,
        help = "base model id to stamp into the checkpoint when --base is a local directory; the "
        "loader compares its base against this id, so a local mirror must declare the repo it mirrors",
    )
    p.add_argument("--upload-revision", default = None)
    p.add_argument(
        "--upload-filename",
        default = None,
        help = "repo-root filename to publish under; defaults to the family's declared "
        "prequant_filenames entry for a rotated build and the legacy transformer_<scheme>.pt "
        "otherwise",
    )
    args = p.parse_args(argv)

    sys.path.insert(0, str(BACKEND))
    import torch
    import torchao
    import diffusers

    from core.inference.diffusion_prequant import packed_weight_fingerprint, prequant_format_for

    # Reuse the runtime quant factory + filter so offline == runtime (the LPIPS-0 invariant).
    from core.inference.diffusion_transformer_quant import (
        FP8_GRANULARITY,
        TQ_FP8,
        TQ_NVFP4,
        TQ_SCHEMES,
        _REQUIRE_BF16_SCHEMES,
        _make_quant_config,
        _resolve_fast_accum,
        divisible_for_scheme,
        exclude_tokens_for_scheme,
        make_filter_fn,
    )
    from torchao.quantization import quantize_

    scheme = args.scheme.strip().lower()
    if scheme not in TQ_SCHEMES:
        print(f"error: --scheme must be one of {TQ_SCHEMES} (not 'auto')", flush = True)
        return 2
    for refusal in (
        upload_gate_refusal(args.upload_repo, args.verify_against),
        verify_target_refusal(args.out, args.verify_against),
        # ConvRot rotates the weight before quantize_, which a GPTQ weight has not been corrected for: the correction was solved against the UNROTATED activation covariance.
        (
            "--gptq-dir and --convrot-groupsize cannot be combined: the correction was solved "
            "against unrotated activations, so rotating the corrected weight discards it."
            if args.gptq_dir and args.convrot_groupsize
            else None
        ),
    ):
        if refusal:
            print(f"error: {refusal}", flush = True)
            return 2
    fam = resolve_build_family(
        args.base_id or args.base, override = args.family, modality = args.modality
    )
    if fam is None:
        print(
            f"error: unknown family '{args.family}' (modality {args.modality})",
            flush = True,
        )
        return 2
    component = (args.component or DEFAULT_COMPONENT).strip() or DEFAULT_COMPONENT
    policy, policy_refusal = resolve_build_policy(
        args.policy, scheme, fam.name, args.base_id or args.base
    )
    if policy_refusal is None and policy is not None and args.convrot_groupsize:
        # Both rewrite the weights before quantize_ and both claim the one format tag slot.
        policy_refusal = (
            f"--policy {policy.policy_id!r} and --convrot-groupsize cannot be combined: a policy "
            "build quantises its layers at two precisions, and the rotation was solved for one"
        )
    if policy_refusal:
        print(f"error: {policy_refusal}", flush = True)
        return 2
    # Checked BEFORE the dense load so a typo costs a second, not a multi-gigabyte download.
    calib_prompts: tuple = ()
    calib_grid: tuple = ()
    if args.gptq_prompts > 0 or args.bake_activation_scales:
        try:
            calib_prompts = load_calibration_prompts(args.calib_prompts)
            gptq_step_spec = parse_step_spec(args.gptq_steps)
            calib_grid = parse_calib_grid(args.calib_resolution, video = is_video_family(fam))
        except ValueError as exc:
            print(f"error: {exc}", flush = True)
            return 2
        refusal = calibration_refusal(
            scheme = scheme,
            nvfp4 = TQ_NVFP4,
            gptq_prompts = args.gptq_prompts,
            bake = args.bake_activation_scales,
            gptq_dir = args.gptq_dir,
            convrot_groupsize = args.convrot_groupsize,
            available_prompts = len(calib_prompts),
        ) or frame_count_refusal(fam, calib_grid[2])
        if refusal:
            print(f"error: {refusal}", flush = True)
            return 2
    transformer_cls = getattr(diffusers, fam.transformer_class)
    # Resolved BEFORE the load, so a rotated build with nowhere resolvable to publish fails in a second rather than
    # after the quantise and the multi-gigabyte save.
    upload_dest = None
    if args.upload_repo:
        try:
            upload_dest = upload_destination(
                fam,
                scheme,
                rotated = bool(args.convrot_groupsize),
                override = args.upload_filename,
                repo_id = args.upload_repo,
                component = component,
            )
        except ValueError as exc:
            print(f"error: {exc}", flush = True)
            return 2

    policy_note = f", policy={policy.policy_id} v{policy.version}" if policy else ""
    print(
        f"== build prequant ({fam.name}/{scheme}, min_feat={args.min_features}{policy_note}) ==",
        flush = True,
    )
    print(f"  loading dense transformer from {args.base} (subfolder={component}) ...", flush = True)
    t0 = time.time()
    transformer = transformer_cls.from_pretrained(
        args.base, subfolder = component, torch_dtype = torch.bfloat16, token = args.hf_token
    ).to("cuda")
    print(f"  quantising in place ({scheme}) ...", flush = True)
    # Mirror the runtime exclusions: int8 skips the M=1 modulation projections (torch._int_mm needs M>16) plus
    # per-family ones; family=None bakes linears the runtime rejects.
    exclude_name_tokens = exclude_tokens_for_scheme(scheme, fam.name)
    # fp8 / mxfp8 need bf16 weights, so skip non-bf16 Linears; nvfp4 handles fp32. Mirrors the runtime gate.
    require_bf16 = scheme in _REQUIRE_BF16_SCHEMES
    # fp8 bakes the accumulate mode in; record it so the loader can reject a contradicting request.
    fast_accum = _resolve_fast_accum(None) if (scheme == TQ_FP8 or policy is not None) else None
    # The same GEMM tiling floor the runtime filter applies. Without it an offline build bakes the ragged linears the runtime leaves dense, and the mismatch surfaces only at the first real matmul.
    require_divisible = divisible_for_scheme(scheme)
    filter_fn = make_filter_fn(
        args.min_features,
        exclude_name_tokens = exclude_name_tokens,
        require_bf16 = require_bf16,
        require_divisible = require_divisible,
    )

    # Resolved BEFORE anything touches the weights: its count assertions turn a rename into a
    # refused build rather than a differently-quantised artifact.
    assignment: dict = {}
    if policy is not None:
        from core.inference.diffusion_nvfp4_policy import (
            PRECISION_NVFP4,
            assign_precisions,
            policy_metadata,
            quantize_with_policy,
        )

        assignment = assign_precisions(transformer, policy, min_features = args.min_features)
        counts = Counter(assignment.values())
        print(
            f"  policy {policy.policy_id} v{policy.version}: "
            + ", ".join(f"{name} {counts[name]}" for name in sorted(counts)),
            flush = True,
        )

    # REPLAYED GPTQ, before the calibration below and quantize_: only the 4-bit operand may be
    # corrected, and an activation scale has to describe the weights the artifact SHIPS.
    gptq_plan: Optional[dict] = None
    gptq_pass: dict = {}
    gptq_where: dict = {}
    gptq_applied_modules: dict = {}
    if args.gptq_dir:
        import json

        gptq_where = gptq_sources(
            args.gptq_dir,
            component,
            meta_override = args.gptq_meta,
            score_override = args.gptq_score,
        )
        try:
            with open(gptq_where["meta"]) as handle:
                gptq_pass = json.load(handle) or {}
        except Exception as exc:  # noqa: BLE001 -- an unreadable meta decides nothing
            print(f"error: cannot read the GPTQ meta {gptq_where['meta']}: {exc}", flush = True)
            return 2
        score_layers: dict = {}
        if gptq_where["score"]:
            try:
                with open(gptq_where["score"]) as handle:
                    score_layers = (json.load(handle) or {}).get("layers") or {}
            except Exception as exc:  # noqa: BLE001
                print(
                    f"error: cannot read the GPTQ scores {gptq_where['score']}: {exc}", flush = True
                )
                return 2
        elif args.gptq_score_mode == "check":
            print(
                f"error: --gptq-score-mode check needs a score file; none found in {args.gptq_dir}. "
                "Score the corrections on held-out activations first, or pass --gptq-score-mode meta.",
                flush = True,
            )
            return 2
        # Corrections go to the policy's NVFP4 set and nowhere else, not to the filter's set.
        if policy is not None:
            admitted = [
                (fqn, module)
                for fqn, module in transformer.named_modules()
                if assignment.get(fqn) == PRECISION_NVFP4
            ]
        else:
            admitted = [
                (fqn, module)
                for fqn, module in transformer.named_modules()
                if filter_fn(module, fqn)
            ]
        weights_dir = gptq_where["weights"]
        gptq_plan = plan_gptq(
            [fqn for fqn, _ in admitted],
            (gptq_pass.get("layers") or {}),
            score_layers,
            mode = args.gptq_score_mode,
            has_weight = lambda fqn: os.path.exists(
                os.path.join(weights_dir, gptq_weight_filename(fqn))
            ),
        )
        for fqn, module in admitted:
            if not gptq_plan["layers"][fqn]["applied"]:
                continue
            corrected = torch.load(
                os.path.join(weights_dir, gptq_weight_filename(fqn)), weights_only = True
            )
            if tuple(corrected.shape) != tuple(module.weight.shape):
                print(
                    f"error: GPTQ weight for {fqn} is {tuple(corrected.shape)}, module is "
                    f"{tuple(module.weight.shape)}",
                    flush = True,
                )
                return 2
            module.weight.data = corrected.to(module.weight.device, module.weight.dtype)
            gptq_applied_modules[fqn] = module
        counts = gptq_plan["counts"]
        print(
            f"  gptq ({args.gptq_score_mode}): applied {counts['applied']} of {len(admitted)} "
            f"admitted linears, {counts['skipped_no_gain']} no gain, "
            f"{counts['skipped_unscored']} unscored, {counts['missing']} missing "
            f"[{weights_dir}]",
            flush = True,
        )

    # In-builder calibration on the DENSE model. The order is fixed; see calibration_stage_order.
    inbuilder_gptq: dict = {}
    act_scales: dict = {}
    act_meta: dict = {}
    if calib_prompts:
        from core.inference.diffusion_families import default_generation_params
        from core.inference.diffusion_nvfp4_gptq import (
            ActivationAmaxAccumulator,
            HessianAccumulator,
            gptq_correct,
            plan_corrections,
            sampled_steps,
            score_correction,
        )
        from core.inference.diffusion_nvfp4_policy import PRECISION_NVFP4

        if policy is not None:
            calib_layers = {
                fqn: module
                for fqn, module in transformer.named_modules()
                if assignment.get(fqn) == PRECISION_NVFP4
            }
        else:
            calib_layers = {
                fqn: module for fqn, module in transformer.named_modules() if filter_fn(module, fqn)
            }
        if not calib_layers:
            print(
                "error: the calibration set is empty (no layer is quantised to 4 bits)", flush = True
            )
            return 2
        pipeline_cls_name = getattr(fam, "pipeline_class", None)
        if not pipeline_cls_name or not hasattr(diffusers, pipeline_cls_name):
            print(
                f"error: family {fam.name!r} names no diffusers pipeline class to calibrate "
                f"through ({pipeline_cls_name!r})",
                flush = True,
            )
            return 2
        video = is_video_family(fam)
        if video:
            default_steps = int(fam.default_steps)
            default_guidance = float(fam.default_guidance)
            calib_steps = int(args.calib_steps or DEFAULT_VIDEO_CALIB_STEPS)
        else:
            default_steps, default_guidance = default_generation_params(
                args.base_id or args.base, fam.name
            )
            calib_steps = int(args.calib_steps or default_steps)
        calib_width, calib_height, calib_frames = calib_grid
        cfg_kwarg = getattr(fam, "cfg_kwarg", "guidance_scale")
        grid_note = (
            f"{calib_width}x{calib_height}x{calib_frames}"
            if calib_frames is not None
            else f"{calib_width}x{calib_height}"
        )
        print(
            f"  calibration: {len(calib_layers)} 4-bit layers, {calib_steps} steps at "
            f"{grid_note}, stages "
            + " -> ".join(calibration_stage_order(args.gptq_prompts, args.bake_activation_scales)),
            flush = True,
        )
        # Goes in under ITS OWN name, so a dual-expert family calibrates the expert asked for.
        pipe = getattr(diffusers, pipeline_cls_name).from_pretrained(
            args.base,
            torch_dtype = torch.bfloat16,
            token = args.hf_token,
            **{component: transformer},
        )
        pipe.to("cuda")
        try:
            pipe.set_progress_bar_config(disable = True)
        except Exception:  # noqa: BLE001 - a pipeline without the knob simply prints
            pass
        render = lambda prompts, **kwargs: render_calibration(  # noqa: E731 - one call site each
            pipe,
            prompts,
            steps = calib_steps,
            guidance = default_guidance,
            cfg_kwarg = cfg_kwarg,
            width = calib_width,
            height = calib_height,
            num_frames = calib_frames,
            guidance_via_guider = bool(getattr(fam, "guidance_via_guider", False)),
            # Left unset: WanPipeline defaults the low-noise expert's guidance to the high-noise
            # one's, which IS this family's default.
            cfg2_kwarg = None,
            seed = args.calib_seed,
            **kwargs,
        )

        if args.gptq_prompts > 0:
            t_gptq = time.time()
            steps_sampled = sampled_steps(calib_steps, gptq_step_spec)
            prompts = calib_prompts[: args.gptq_prompts]
            hessians = HessianAccumulator(calib_layers)
            refusal = hessians.budget_refusal()
            if refusal:
                print(f"error: {refusal}", flush = True)
                return 2
            hessians.attach()
            print(
                f"  gptq: accumulating Hessians on steps {list(steps_sampled)} of "
                f"{len(prompts)} prompts ...",
                flush = True,
            )
            try:
                render(
                    prompts,
                    callback = hessians.step_callback(steps_sampled),
                    before_prompt = lambda: hessians.arm_first(steps_sampled),
                )
            finally:
                hessians.detach()
            unseen = hessians.unseen()
            if unseen:
                # A layer the sampled steps never reached would be "corrected" from zeros.
                print(
                    f"error: {len(unseen)} layers saw no calibration activation (first: "
                    f"{unseen[0]}); widen --gptq-steps or raise --gptq-prompts",
                    flush = True,
                )
                return 2
            moments = hessians.normalised()
            hessians.free()
            scores: dict = {}
            damps: dict = {}
            corrected_weights: dict = {}
            for index, (fqn, module) in enumerate(sorted(calib_layers.items())):
                hessian = moments.pop(fqn)
                corrected, damp = gptq_correct(module.weight.data, hessian)
                scores[fqn] = score_correction(module.weight.data, corrected, hessian)
                damps[fqn] = damp
                corrected_weights[fqn] = corrected
                del hessian
                if (index + 1) % 25 == 0:
                    print(f"    corrected {index + 1}/{len(calib_layers)} layers", flush = True)
            plan = plan_corrections(scores, max_regressions = args.gptq_max_regressions)
            for fqn in plan["apply"]:
                module = calib_layers[fqn]
                module.weight.data = corrected_weights[fqn].to(
                    module.weight.device, module.weight.dtype
                )
            corrected_weights.clear()
            torch.cuda.empty_cache()
            inbuilder_gptq = gptq_metadata_block(
                prompts = prompts,
                steps_sampled = steps_sampled,
                schedule_steps = calib_steps,
                max_regressions = args.gptq_max_regressions,
                plan = plan,
                scores = scores,
                damps = damps,
                seconds = time.time() - t_gptq,
            )
            print(
                f"  gptq: applied {inbuilder_gptq['applied']} of {len(calib_layers)} layers "
                f"({inbuilder_gptq['skipped_no_gain']} scored no better, "
                f"{inbuilder_gptq['applied_regressed']} applied against their score) in "
                f"{inbuilder_gptq['seconds']:.0f}s",
                flush = True,
            )

        if args.bake_activation_scales:
            prompts = calib_prompts[: max(1, args.bake_prompts)]
            amax = ActivationAmaxAccumulator(calib_layers).attach()
            print(f"  baking activation scales over {len(prompts)} prompts ...", flush = True)
            try:
                render(prompts)
            finally:
                amax.detach()
            unseen = amax.unseen()
            if unseen:
                print(
                    f"error: {len(unseen)} layers saw no calibration activation (first: "
                    f"{unseen[0]}), so their activation scale cannot be baked",
                    flush = True,
                )
                return 2
            act_scales = amax.global_scales()
            missing = sorted(set(calib_layers) - set(act_scales))
            if missing:
                # Refused rather than partially baked: the loader converts all or none.
                print(
                    f"error: {len(missing)} layers produced no usable activation amax (first: "
                    f"{missing[0]})",
                    flush = True,
                )
                return 2
            act_meta = activation_scale_metadata(
                prompts = prompts,
                schedule_steps = calib_steps,
                scales = act_scales,
                layers = len(calib_layers),
                grid = grid_note,
            )
            print(
                f"  baked {len(act_scales)} activation scales, a_gsf "
                f"{act_meta['min_a_gsf']:.4g} to {act_meta['max_a_gsf']:.4g}",
                flush = True,
            )

        del pipe
        torch.cuda.empty_cache()

    # ConvRot, BEFORE quantize_: rotating the weights is only worth anything if the quantizer then sees the rotated
    # distribution. The fqn list is recorded, never re-derived at load time.
    rotation: dict = {}
    if args.convrot_groupsize:
        from core.inference.diffusion_convrot import (
            rotatable_fqns,
            rotate_linears_,
            rotation_metadata,
        )

        group = int(args.convrot_groupsize)
        rotatable, not_divisible = rotatable_fqns(transformer, filter_fn, group)
        refusal = convrot_refusal(group, rotatable, not_divisible)
        if refusal:
            print(f"error: {refusal}", flush = True)
            return 2
        rotate_linears_(transformer, rotatable, group)
        rotation = rotation_metadata(group, rotatable)
        print(
            f"  rotated {len(rotatable)} linears at ConvRot group {group}; "
            f"{len(not_divisible)} quantized linears left plain (in_features not divisible)"
            + (f", e.g. {not_divisible[0]}" if not_divisible else ""),
            flush = True,
        )

    if policy is not None:
        # Two passes over disjoint fqn sets, NVFP4 first.
        quantize_with_policy(
            transformer,
            policy,
            min_features = args.min_features,
            fast_accum = fast_accum,
        )
    else:
        quantize_(transformer, _make_quant_config(scheme), filter_fn = filter_fn)

    gptq_idempotency: dict = {}
    if gptq_applied_modules:
        weights_dir = gptq_where["weights"]
        gptq_idempotency = verify_gptq_idempotency(
            gptq_applied_modules,
            lambda fqn: torch.load(
                os.path.join(weights_dir, gptq_weight_filename(fqn)), weights_only = True
            ),
        )
        print(
            f"  gptq idempotency: {gptq_idempotency['checked']} layers, max abs deviation "
            f"{gptq_idempotency['max_abs']:.3e} ({gptq_idempotency['max_abs_fqn']}), "
            f"{gptq_idempotency['frac_diff'] * 100:.3f}% of elements moved",
            flush = True,
        )

    state_dict = {
        k: (v.detach().to("cpu") if hasattr(v, "detach") else v)
        for k, v in transformer.state_dict().items()
    }
    fingerprint = packed_weight_fingerprint(state_dict)
    metadata = {
        "base_model_id": args.base_id or args.base,
        "family": fam.name,
        "scheme": scheme,
        "component": component,
        "min_features": args.min_features,
        # Let the loader reject a checkpoint that would not match the runtime path.
        "exclude_name_tokens": list(exclude_name_tokens),
        "require_bf16": require_bf16,
        "require_divisible": require_divisible,
        "fingerprint": fingerprint,
        "fast_accum": fast_accum,
        "torch_dtype": args.dtype,
        "quant_backend": "torchao",
        "transformer_class": fam.transformer_class,
        "torch_version": torch.__version__,
        "torchao_version": getattr(torchao, "__version__", "?"),
        "diffusers_version": diffusers.__version__,
    }
    # fp8 granularity: lets the loader reject a stale per-tensor checkpoint (runtime needs per-row).
    if scheme == TQ_FP8:
        metadata["fp8_granularity"] = FP8_GRANULARITY
    if policy is not None:
        metadata.update(
            policy_metadata(
                policy,
                assignment,
                gptq = bool(gptq_applied_modules or inbuilder_gptq),
                activation_scales_baked = bool(act_scales),
            )
        )
        metadata["fp8_granularity"] = FP8_GRANULARITY
    if act_scales:
        from core.inference.diffusion_nvfp4_linear import ACT_SCALES_KEY

        # The flashinfer backend converts a layer only when it finds its scale here.
        metadata[ACT_SCALES_KEY] = {fqn: float(value) for fqn, value in sorted(act_scales.items())}
        metadata["activation_calibration"] = act_meta
        if policy is None:
            metadata["activation_scales_baked"] = True
    if inbuilder_gptq:
        metadata["gptq"] = inbuilder_gptq
    if gptq_plan is not None:
        # Provenance of every corrected weight. Not part of the fingerprint's identity: it hashes the packed payloads, so a different set of corrections already reads as a different artifact there.
        metadata["gptq"] = {
            "source": os.path.abspath(args.gptq_dir),
            "meta_path": os.path.abspath(gptq_where["meta"]),
            "score_path": os.path.abspath(gptq_where["score"]) if gptq_where["score"] else None,
            "score_mode": args.gptq_score_mode,
            "prompts": gptq_pass.get("prompts"),
            "steps_sampled": gptq_pass.get("steps_sampled"),
            "base_damp": gptq_pass.get("base_damp"),
            "grid": gptq_pass.get("grid"),
            "applied": gptq_plan["counts"]["applied"],
            "skipped_no_gain": gptq_plan["counts"]["skipped_no_gain"],
            "skipped_unscored": gptq_plan["counts"]["skipped_unscored"],
            "missing": gptq_plan["counts"]["missing"],
            "idempotency": gptq_idempotency,
            "layers": gptq_plan["layers"],
        }
    metadata.update(rotation)
    ckpt = {
        # v2 when a rotation is baked in, so an Unsloth predating the online half refuses the file rather than running
        # the rotated weights against unrotated activations.
        "format": prequant_format_for(metadata),
        "metadata": metadata,
        "state_dict": state_dict,
    }

    out = Path(args.out)
    out.parent.mkdir(parents = True, exist_ok = True)
    torch.save(ckpt, out)
    size_gb = out.stat().st_size / 1e9
    print(f"  saved {out}  ({size_gb:.2f} GB) in {time.time() - t0:.0f}s", flush = True)
    shown = {
        k: v for k, v in metadata.items() if k not in ("fingerprint", "act_global_scales", "gptq")
    }
    if metadata.get("act_global_scales"):
        shown["act_global_scales"] = f"<{len(metadata['act_global_scales'])} layers>"
    if metadata.get("gptq"):
        shown["gptq"] = {k: v for k, v in metadata["gptq"].items() if k != "layers"}
    print(f"  metadata: {shown}", flush = True)
    print(
        f"  fingerprint: {fingerprint['count']} quantized weights, "
        f"{len(fingerprint['skipped'])} weights left dense ({fingerprint['algo']})",
        flush = True,
    )

    if args.verify_against:
        print(f"  verifying against {args.verify_against} ...", flush = True)
        code = verify_against(
            args.verify_against, fingerprint, out = lambda line: print(line, flush = True)
        )
        if code:
            return code

    if args.upload_repo:
        from huggingface_hub import HfApi

        dest = upload_dest
        print(f"  uploading -> {args.upload_repo}:{dest} ...", flush = True)
        api = HfApi(token = args.hf_token)
        api.create_repo(args.upload_repo, exist_ok = True)
        api.upload_file(
            path_or_fileobj = str(out),
            path_in_repo = dest,
            repo_id = args.upload_repo,
            revision = args.upload_revision,
        )
        print(f"  uploaded {dest} to {args.upload_repo}", flush = True)

    print("BUILD-PREQUANT-DONE", flush = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
