# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Quantizes the DiT once so runtime loads it with assign=True, skipping the dense bf16 GPU load peak."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

BACKEND = Path(__file__).resolve().parent.parent / "studio" / "backend"

# Mirrors core.inference.diffusion_prequant.DEFAULT_PREQUANT_COMPONENT; the backend joins sys.path only in main().
DEFAULT_COMPONENT = "transformer"


def resolve_build_family(base: str, override: Optional[str] = None) -> Optional[Any]:
    """The image family for ``base`` / ``override``, else the video family (Wan, HunyuanVideo), else None."""
    from core.inference.diffusion_families import detect_family
    from core.inference.video_families import detect_video_family

    return detect_family(base, override = override) or detect_video_family(base, override = override)


def convrot_refusal(
    group: int, rotatable: Sequence[str], not_divisible: Sequence[str]
) -> Optional[str]:
    """Refuse a ConvRot build with no rotatable Linear before the hours of quantising, not after:
    its empty fqn list is refused at load time, leaving an unopenable artifact"""
    if rotatable:
        return None
    return (
        f"ConvRot group {group} divides the in_features of none of the {len(not_divisible)} "
        "quantized linears, so the checkpoint would record an empty rotation and be refused at "
        "load time. Pick a smaller power-of-4 group, or drop --convrot-groupsize."
    )


def upload_destination(
    fam: Any,
    scheme: str,
    *,
    rotated: bool,
    safetensors: bool = False,
    override: Optional[str] = None,
    upload_repo: Optional[str] = None,
    component: str = DEFAULT_COMPONENT,
    convrot_group: Optional[int] = None,
) -> str:
    """The repo-root filename this build should publish under.

    The loader asks for the family's declared ``prequant_filenames`` name first and the derived
    ``<Model>-<SCHEME>.pt`` second, so a ROTATED artifact published under the legacy
    ``transformer_<scheme>.pt`` is either never resolved at all, or resolved as the fallback by a
    build too old to honour the rotation, which then refuses the v2 tag and drops to the dense
    download. A rotated build therefore goes to the declared name or nowhere. Plain builds keep
    the legacy name they have always used.

    A SAFETENSORS build is in the same position for a different reason: both derived names end in
    ``.pt``, so no build ever asks the Hub for a safetensors artifact unless the family names it.
    Uploading one under a derived name produces a file that is reachable by nothing and a repo that
    looks like it has a checkpoint when it does not, so it is refused here rather than discovered
    as a silent dense fallback later.

    An ``override`` skips the family table, because naming the artifact by hand is the escape hatch
    for a repo the table does not describe yet. It does NOT skip the container check: every loader
    dispatches on the extension alone, so a safetensors build published as ``.pt`` is read as a
    pickle and a pickle published as ``.safetensors`` is read from a header it does not have. Either
    way the upload succeeds and the artifact is unopenable, after the hours the quantization took.

    nvfp4 publishes safetensors only: Studio never resolves or opens an nvfp4 pickle.
    """
    if (scheme or "").strip().lower() == "nvfp4" and not safetensors:
        raise ValueError(
            "nvfp4 checkpoints are published as .safetensors only (Studio never reads an nvfp4 "
            "pickle); write --out as a .safetensors file"
        )
    if override:
        wanted = ".safetensors" if safetensors else ".pt"
        if not override.lower().endswith(wanted):
            container = "safetensors" if safetensors else "torch.save"
            raise ValueError(
                f"--upload-filename {override!r} does not end in {wanted!r}, but --out writes the "
                f"{container} container. The loader dispatches on the extension alone, so this "
                "would publish an artifact nothing can open. Rename the upload, or change --out."
            )
        return override
    if component and component != DEFAULT_COMPONENT:
        # A second denoiser resolves only its task-specific row, so other names would never be loaded.
        from core.inference.diffusion_families import family_prequant_filename

        specific = family_prequant_filename(fam, scheme, task = component)
        if specific is None or specific == family_prequant_filename(fam, scheme):
            raise ValueError(
                f"family {getattr(fam, 'name', fam)!r} declares no prequant_filenames entry for "
                f"({scheme!r}, {component!r}), so the loader would never ask for this component. "
                "Add the entry to the family table, or pass --upload-filename."
            )
        wanted = ".safetensors" if safetensors else ".pt"
        if not specific.lower().endswith(wanted):
            raise ValueError(
                f"family {getattr(fam, 'name', fam)!r} declares {specific!r} for "
                f"({scheme!r}, {component!r}), which does not end in {wanted!r} like --out. "
                "Rename --out, or pass --upload-filename."
            )
        return specific
    from core.inference.diffusion_prequant import prequant_filename

    if not rotated and not safetensors:
        return prequant_filename(scheme)
    from core.inference.diffusion_families import family_prequant_filename
    from core.inference.diffusion_transformer_quant import (
        convrot_prequant_filename,
        convrot_spec_for_scheme,
    )

    family = getattr(fam, "name", None)
    rotated_name = convrot_prequant_filename(scheme, family) if rotated else None
    if rotated_name:
        spec_group = convrot_spec_for_scheme(scheme, family)[0]
        if convrot_group is not None and int(convrot_group) != spec_group:
            raise ValueError(
                f"{rotated_name} is reserved for the ConvRot group {spec_group} build that matches "
                f"the runtime path; a group {convrot_group} build needs --upload-filename."
            )
    preferred = rotated_name or family_prequant_filename(fam, scheme)
    why = "a rotated checkpoint" if rotated else "a safetensors checkpoint"
    if not preferred:
        # Plain safetensors builds have a derived name; rotated builds still need a declared one.
        if safetensors and not rotated and upload_repo:
            from core.inference.diffusion_prequant import prequant_repo_filename
            return prequant_repo_filename(upload_repo, scheme, ".safetensors")
        raise ValueError(
            f"family {getattr(fam, 'name', fam)!r} declares no prequant_filenames entry for "
            f"{scheme!r}, so {why} has no name the loader would ask for. Add the "
            "entry to the family table, or pass --upload-filename."
        )
    # Check both directions: the declared name's extension must match the container.
    wanted = ".safetensors" if safetensors else ".pt"
    if not preferred.lower().endswith(wanted):
        reads_as = "a pickle" if safetensors else "safetensors"
        writes = "safetensors" if safetensors else "torch.save"
        raise ValueError(
            f"family {getattr(fam, 'name', fam)!r} declares {preferred!r} for {scheme!r}, which "
            f"does not end in {wanted!r}, but --out writes the {writes} container. The loader "
            f"dispatches on the extension alone, so this would be published under a name it reads "
            f"as {reads_as}. Point the prequant_filenames entry at the matching artifact, or pass "
            "--upload-filename."
        )
    return preferred


def quant_filter_settings(scheme: str, family: Optional[str]) -> dict:
    """Runtime ``quantize_`` filter inputs for ``scheme``; one dict feeds filter and metadata."""
    from core.inference.diffusion_transformer_quant import (
        _REQUIRE_BF16_SCHEMES,
        divisible_for_scheme,
        exclude_tokens_for_scheme,
    )
    return {
        "exclude_name_tokens": list(exclude_tokens_for_scheme(scheme, family)),
        "require_bf16": scheme in _REQUIRE_BF16_SCHEMES,
        "require_divisible": divisible_for_scheme(scheme),
    }


def run_gptq(
    args, fam, component, transformer, filter_fn, method, rotatable, group, pipe_kwargs, diffusers
):
    """Calibrate through the dense pipeline and GPTQ-solve every Linear the nvfp4 filter admits."""
    import gc

    import torch

    from core.inference.diffusion_nvfp4_gptq_build import (
        CALIB_PROMPTS,
        calibrate_and_solve,
        pipeline_renderer,
    )

    n = int(args.gptq_prompts)
    if n > len(CALIB_PROMPTS):
        raise SystemExit(
            f"error: --gptq-prompts {n} exceeds the {len(CALIB_PROMPTS)} built-in prompts"
        )
    prompts = CALIB_PROMPTS[:n]
    fqns = [
        f
        for f, m in transformer.named_modules()
        if isinstance(m, torch.nn.Linear) and filter_fn(m, f)
    ]
    t0 = time.time()
    pipe_cls = getattr(diffusers, fam.pipeline_class)
    print(f"  GPTQ: loading {fam.pipeline_class} for calibration ({n} prompts) ...", flush = True)
    pipe = pipe_cls.from_pretrained(
        args.base, torch_dtype = torch.bfloat16, token = args.hf_token, **{component: transformer}
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable = True)
    free = torch.cuda.mem_get_info()[0]
    budget = max(int(free - args.gptq_reserve_gib * 2**30), 1)
    render_all = pipeline_renderer(
        pipe,
        prompts,
        steps = args.gptq_steps,
        height = args.gptq_size,
        width = args.gptq_size,
        seed = args.gptq_seed,
        extra_kwargs = pipe_kwargs,
    )
    solved = calibrate_and_solve(
        transformer,
        fqns,
        render_all,
        method = method,
        rotated = rotatable,
        group = group,
        budget_bytes = budget,
        damp = args.gptq_damp,
        blocksize = args.gptq_blocksize,
        log = lambda msg: print(f"  {msg}", flush = True),
    )
    del pipe, render_all
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPTQ: {len(solved)} linears solved in {time.time() - t0:.0f}s", flush = True)
    meta = {
        "method": method,
        "prompts": n,
        "steps": int(args.gptq_steps),
        "size": int(args.gptq_size),
        "seed": int(args.gptq_seed),
        "pipe_kwargs": pipe_kwargs,
        "damp": float(args.gptq_damp),
        "blocksize": int(args.gptq_blocksize),
        "layers": len(solved),
    }
    return solved, meta


def main(argv = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base", required = True, help = "diffusers base repo (carries the transformer subfolder)"
    )
    p.add_argument("--family", required = True, help = "diffusion family name/alias (e.g. z-image)")
    p.add_argument(
        "--base-model-id",
        default = None,
        help = "the base id to RECORD in the checkpoint, when --base is a local mirror whose "
        "directory name differs from the Hub repo. Must still name this family's base model.",
    )
    p.add_argument("--scheme", required = True, help = "quant scheme: int8 | fp8 | nvfp4 | mxfp8")
    p.add_argument(
        "--component",
        default = DEFAULT_COMPONENT,
        help = "denoiser subfolder to quantise, recorded in the checkpoint (e.g. transformer_2 for "
        "Wan2.2 A14B's second expert)",
    )
    p.add_argument(
        "--out",
        required = True,
        help = "output path; a .safetensors extension writes the safetensors container, anything "
        "else writes the torch.save one",
    )
    p.add_argument("--min-features", type = int, default = 512)
    p.add_argument("--dtype", default = "bfloat16", choices = ["bfloat16"])
    p.add_argument("--hf-token", default = None)
    p.add_argument(
        "--convrot-groupsize",
        type = int,
        default = 0,
        help = "bake a ConvRot block-Hadamard activation rotation at this group size (a power of "
        "4; 0 = off). A family with an int8 ConvRot spec at this group (Qwen-Image-2.1: "
        "256) rotates exactly the Linears that spec names, so the artifact matches the opt-in "
        "runtime path (UNSLOTH_DIFFUSION_INT8_CONVROT=1); otherwise every quantized Linear whose in_features the group divides has its "
        "weight rotated before quantize_ so the quantizer sees a flatter distribution; the "
        "exact fqn list is recorded in the checkpoint and the loader rotates the "
        "activations of that list and nothing else. Writes the v2 format tag.",
    )
    p.add_argument(
        "--nvfp4-weight-quant",
        default = None,
        choices = ["max6", "4over6"],
        help = "nvfp4 block-scale choice: max6 (torchao's) or 4over6 (per 16-block max/6 vs max/4, "
        "per-tensor scale for an E4M3 top of 256). Default: UNSLOTH_NVFP4_WEIGHT_QUANT, else 4over6. "
        "With --gptq-prompts the choice runs inside the GPTQ loop, weighted by the Hessian diagonal.",
    )
    p.add_argument(
        "--gptq-prompts",
        type = int,
        default = 0,
        help = "nvfp4 only: GPTQ-calibrate on the first N built-in calibration prompts (0 = off, "
        "round-to-nearest). Same file format; one-time cost (minutes per model).",
    )
    p.add_argument(
        "--gptq-steps", type = int, default = 8, help = "denoising steps per calibration prompt"
    )
    p.add_argument(
        "--gptq-size", type = int, default = 1024, help = "calibration render height and width"
    )
    p.add_argument("--gptq-seed", type = int, default = 5000, help = "prompt i renders with seed + i")
    p.add_argument(
        "--gptq-pipe-kwargs",
        default = "{}",
        help = "JSON of extra pipeline call kwargs for calibration, e.g. '{\"guidance_scale\": 0.0}' for a "
        "distilled model, or num_frames for video",
    )
    p.add_argument("--gptq-damp", type = float, default = 0.01)
    p.add_argument("--gptq-blocksize", type = int, default = 128)
    p.add_argument(
        "--gptq-reserve-gib",
        type = float,
        default = 8.0,
        help = "GPU memory kept free while Hessians are resident (more = more calibration passes)",
    )
    p.add_argument(
        "--upload-repo", default = None, help = "optional HF repo id to upload the checkpoint to"
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
    if (args.scheme or "").strip().lower() == "nvfp4" and not str(args.out).lower().endswith(
        ".safetensors"
    ):
        print(
            "error: --scheme nvfp4 writes .safetensors only (Studio never reads an NVFP4 pickle); "
            "give --out a .safetensors path",
            flush = True,
        )
        return 2
    if (args.scheme or "").strip().lower() != "nvfp4" and (
        args.gptq_prompts or args.nvfp4_weight_quant
    ):
        print(
            "error: --gptq-prompts / --nvfp4-weight-quant apply to --scheme nvfp4 only", flush = True
        )
        return 2
    try:
        gptq_pipe_kwargs = json.loads(args.gptq_pipe_kwargs or "{}")
    except ValueError:
        gptq_pipe_kwargs = None
    if not isinstance(gptq_pipe_kwargs, dict):
        print(
            f"error: --gptq-pipe-kwargs is not a JSON object: {args.gptq_pipe_kwargs!r}", flush = True
        )
        return 2

    sys.path.insert(0, str(BACKEND))
    import torch
    import torchao
    import diffusers

    from core.inference.diffusion_prequant import prequant_format_for

    # Reuse the runtime quant factory + filter so offline == runtime (the LPIPS-0 invariant).
    from core.inference.diffusion_transformer_quant import (
        FP8_GRANULARITY,
        TQ_FP8,
        TQ_SCHEMES,
        _make_quant_config,
        _resolve_fast_accum,
        convrot_fqns,
        convrot_spec_for_scheme,
        make_filter_fn,
    )
    from torchao.quantization import quantize_

    scheme = args.scheme.strip().lower()
    if scheme not in TQ_SCHEMES:
        print(f"error: --scheme must be one of {TQ_SCHEMES} (not 'auto')", flush = True)
        return 2
    component = (args.component or "").strip() or DEFAULT_COMPONENT
    fam = resolve_build_family(args.base, override = args.family)
    if fam is None:
        print(f"error: unknown family '{args.family}'", flush = True)
        return 2
    convrot_group = int(args.convrot_groupsize)
    spec_group, spec_suffixes = convrot_spec_for_scheme(scheme, fam.name)
    convrot_suffixes: tuple = spec_suffixes if convrot_group and convrot_group == spec_group else ()
    # The recorded base may differ from the one read: a local staging dir name would fail the
    # loader's _same_base_model tail comparison, so --base-model-id (pinned to base_repo) overrides it.
    recorded_base = args.base_model_id or args.base
    if args.base_model_id:
        # Exact match, not _same_base_model, so a published file always records the canonical id.
        if recorded_base.strip() != fam.base_repo:
            print(
                f"error: --base-model-id {recorded_base!r} is not {fam.name}'s base "
                f"({fam.base_repo!r}); it would label this checkpoint as a different model",
                flush = True,
            )
            return 2
    transformer_cls = getattr(diffusers, fam.transformer_class)
    is_safetensors_out = str(args.out).lower().endswith(".safetensors")
    if is_safetensors_out:
        from core.inference.prequant_safetensors import (
            safetensors_prequant_supported,
            scheme_is_flattenable,
        )

        if not safetensors_prequant_supported():
            print(
                "error: --out names a .safetensors checkpoint but this install cannot write one "
                "(needs torchao >= 0.16 for torchao.prototype.safetensors.safetensors_support, "
                "plus the safetensors package)",
                flush = True,
            )
            return 2
        # Importable helpers do not mean this scheme flattens (int8 does not through torchao 0.17).
        # Probe on a tiny CPU Linear before the long download; None means the probe could not run, so proceed.
        from core.inference.diffusion_transformer_quant import _make_quant_config

        if scheme_is_flattenable(_make_quant_config(scheme)) is False:
            print(
                f"error: --out names a .safetensors checkpoint but this torchao quantises "
                f"{scheme!r} to a legacy tensor subclass that cannot be written to safetensors. "
                "torchao >= 0.18 produces the flattenable subclasses for every scheme Unsloth "
                "ships. Upgrade torchao, or write this build as a .pt checkpoint.",
                flush = True,
            )
            return 2
    # Resolved before the load so a bad upload target fails fast.
    upload_dest = None
    if args.upload_repo:
        try:
            upload_dest = upload_destination(
                fam,
                scheme,
                rotated = bool(convrot_group),
                convrot_group = convrot_group,
                safetensors = is_safetensors_out,
                override = args.upload_filename,
                upload_repo = args.upload_repo,
                component = component,
            )
        except ValueError as exc:
            print(f"error: {exc}", flush = True)
            return 2

    print(f"== build prequant ({fam.name}/{scheme}, min_feat={args.min_features}) ==", flush = True)
    print(f"  loading dense transformer from {args.base} (subfolder={component}) ...", flush = True)
    t0 = time.time()
    transformer = transformer_cls.from_pretrained(
        args.base, subfolder = component, torch_dtype = torch.bfloat16, token = args.hf_token
    ).to("cuda")
    print(f"  quantising in place ({scheme}) ...", flush = True)
    filter_settings = quant_filter_settings(scheme, fam.name)
    exclude_name_tokens = tuple(filter_settings["exclude_name_tokens"])
    require_bf16 = filter_settings["require_bf16"]
    # fp8 bakes the accumulate mode in; record it so the loader can reject a contradicting request.
    fast_accum = _resolve_fast_accum(None) if scheme == TQ_FP8 else None
    filter_fn = make_filter_fn(
        args.min_features,
        exclude_name_tokens = exclude_name_tokens,
        require_bf16 = require_bf16,
        require_divisible = filter_settings["require_divisible"],
    )

    # Rotate before quantize_ so the quantizer sees the rotated distribution.
    rotation: dict = {}
    rotatable: Sequence[str] = ()
    if convrot_group:
        from core.inference.diffusion_convrot import (
            rotatable_fqns,
            rotate_linears_,
            rotation_metadata,
        )

        group = int(convrot_group)
        rotatable, not_divisible = rotatable_fqns(transformer, filter_fn, group)
        if convrot_suffixes:
            rotatable = convrot_fqns(transformer, filter_fn, group, convrot_suffixes)
        refusal = convrot_refusal(group, rotatable, not_divisible)
        if refusal:
            print(f"error: {refusal}", flush = True)
            return 2

    # nvfp4 block-scale method: the quantize_ pass below reads it through the weight-quant patch.
    weight_quant = None
    gptq_solved: dict = {}
    gptq_meta = None
    if scheme == "nvfp4":
        from core.inference.diffusion_nvfp4_weight_quant import WEIGHT_QUANT_ENV, resolve_method

        weight_quant = resolve_method(args.nvfp4_weight_quant)
        os.environ[WEIGHT_QUANT_ENV] = weight_quant
        print(f"  nvfp4 weight scales: {weight_quant}", flush = True)
    if scheme == "nvfp4" and args.gptq_prompts > 0:
        gptq_solved, gptq_meta = run_gptq(
            args,
            fam,
            component,
            transformer,
            filter_fn,
            weight_quant,
            rotatable,
            int(convrot_group or 0),
            gptq_pipe_kwargs,
            diffusers,
        )

    if convrot_group:
        rotate_linears_(transformer, rotatable, group)
        rotation = rotation_metadata(group, rotatable)
        print(
            f"  rotated {len(rotatable)} linears at ConvRot group {group}; "
            f"{len(not_divisible)} quantized linears left plain (in_features not divisible)"
            + (f", e.g. {not_divisible[0]}" if not_divisible else ""),
            flush = True,
        )

    quantize_(transformer, _make_quant_config(scheme), filter_fn = filter_fn)
    if gptq_solved:
        from core.inference.diffusion_nvfp4_gptq_build import apply_gptq_payloads
        print(
            f"  GPTQ payloads swapped into {apply_gptq_payloads(transformer, gptq_solved)} nvfp4 linears",
            flush = True,
        )
        gptq_solved.clear()

    state_dict = {
        k: (v.detach().to("cpu") if hasattr(v, "detach") else v)
        for k, v in transformer.state_dict().items()
    }
    metadata = {
        "base_model_id": recorded_base,
        "family": fam.name,
        "scheme": scheme,
        "min_features": args.min_features,
        "exclude_name_tokens": list(exclude_name_tokens),
        "require_bf16": require_bf16,
        "require_divisible": filter_settings["require_divisible"],
        "fast_accum": fast_accum,
        "torch_dtype": args.dtype,
        "quant_backend": "torchao",
        "transformer_class": fam.transformer_class,
        "component": component,
        "torch_version": torch.__version__,
        "torchao_version": getattr(torchao, "__version__", "?"),
        "diffusers_version": diffusers.__version__,
    }
    if weight_quant is not None:
        metadata["nvfp4_weight_quant"] = weight_quant
        metadata["gptq"] = gptq_meta or False
    from core.inference.diffusion_prequant import canonical_weight_fingerprint

    # Version-neutral, so a file built on one torchao generation verifies on every other.
    metadata["fingerprint"] = canonical_weight_fingerprint(state_dict)
    # fp8 granularity: lets the loader reject a stale per-tensor checkpoint (runtime needs per-row).
    if scheme == TQ_FP8:
        metadata["fp8_granularity"] = FP8_GRANULARITY
    metadata.update(rotation)
    ckpt = {
        # v2 when rotated, so an older Unsloth refuses the file instead of mismatching activations.
        "format": prequant_format_for(metadata),
        "metadata": metadata,
        "state_dict": state_dict,
    }

    out = Path(args.out)
    out.parent.mkdir(parents = True, exist_ok = True)
    if is_safetensors_out:
        from core.inference.prequant_safetensors import save_prequant_safetensors
        save_prequant_safetensors(
            str(out),
            fmt = ckpt["format"],
            state_dict = state_dict,
            metadata = metadata,
        )
    else:
        torch.save(ckpt, out)
    size_gb = out.stat().st_size / 1e9
    print(f"  saved {out}  ({size_gb:.2f} GB) in {time.time() - t0:.0f}s", flush = True)
    print(f"  metadata: {ckpt['metadata']}", flush = True)

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
