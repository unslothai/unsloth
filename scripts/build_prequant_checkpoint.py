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
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

BACKEND = Path(__file__).resolve().parent.parent / "studio" / "backend"


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


def upload_destination(
    fam: Any,
    scheme: str,
    *,
    rotated: bool,
    safetensors: bool = False,
    override: Optional[str] = None,
    upload_repo: Optional[str] = None,
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
    """
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
    from core.inference.diffusion_prequant import prequant_filename

    if not rotated and not safetensors:
        return prequant_filename(scheme)
    from core.inference.diffusion_families import family_prequant_filename
    from core.inference.diffusion_transformer_quant import convrot_spec_for_scheme

    if (
        not rotated
        and safetensors
        and upload_repo
        and convrot_spec_for_scheme(scheme, getattr(fam, "name", None))[0]
    ):
        # the declared name is this family's ROTATED artifact; a plain build goes to the derived name the chain keeps
        # behind it, rather than over the rotated one
        from core.inference.diffusion_prequant import prequant_repo_filename
        return prequant_repo_filename(upload_repo, scheme, ".safetensors")
    preferred = family_prequant_filename(fam, scheme)
    why = "a rotated checkpoint" if rotated else "a safetensors checkpoint"
    if not preferred:
        # A PLAIN safetensors build now has a derived name, and only because
        # ``derived_prequant_filenames`` asks for ``<Model>-<SCHEME>.safetensors`` FIRST. The
        # reachability this refusal protects is exactly what that chain supplies, so refusing here
        # would make every family without a declared entry pass an override it could compute
        # itself. Rotation keeps needing a declared name: no derived spelling carries the marker.
        if safetensors and not rotated and upload_repo:
            from core.inference.diffusion_prequant import prequant_repo_filename
            return prequant_repo_filename(upload_repo, scheme, ".safetensors")
        raise ValueError(
            f"family {getattr(fam, 'name', fam)!r} declares no prequant_filenames entry for "
            f"{scheme!r}, so {why} has no name the loader would ask for. Add the "
            "entry to the family table, or pass --upload-filename."
        )
    # Both directions, not just one. The declared name and the container have to agree, and a
    # family that has moved its entry to a .safetensors artifact makes the REVERSE mismatch the
    # reachable one: a rotated pickle build then publishes torch.save bytes under a safetensors
    # name, every loader dispatches on the extension and hands them to safe_open, and the artifact
    # is unopenable after the hours the quantization took. Same failure as the guarded direction,
    # so it gets the same refusal.
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
        default = None,
        help = "bake a ConvRot block-Hadamard activation rotation at this group size (a power of "
        "4; 0 = off; unset = the family's runtime ConvRot spec, which rotates only the Linears it "
        "names, so the artifact matches the runtime quantize path). Every quantized Linear whose in_features the group divides has its "
        "weight rotated before quantize_ so the quantizer sees a flatter distribution; the "
        "exact fqn list is recorded in the checkpoint and the loader rotates the "
        "activations of that list and nothing else. Writes the v2 format tag.",
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

    sys.path.insert(0, str(BACKEND))
    import torch
    import torchao
    import diffusers

    from core.inference.diffusion_families import detect_family
    from core.inference.diffusion_prequant import prequant_format_for

    # Reuse the runtime quant factory + filter so offline == runtime (the LPIPS-0 invariant).
    from core.inference.diffusion_transformer_quant import (
        FP8_GRANULARITY,
        TQ_FP8,
        TQ_SCHEMES,
        _REQUIRE_BF16_SCHEMES,
        _make_quant_config,
        _resolve_fast_accum,
        convrot_fqns,
        convrot_spec_for_scheme,
        exclude_tokens_for_scheme,
        make_filter_fn,
    )
    from torchao.quantization import quantize_

    scheme = args.scheme.strip().lower()
    if scheme not in TQ_SCHEMES:
        print(f"error: --scheme must be one of {TQ_SCHEMES} (not 'auto')", flush = True)
        return 2
    fam = detect_family(args.base, override = args.family)
    if fam is None:
        print(f"error: unknown family '{args.family}'", flush = True)
        return 2
    # Unset follows the runtime path's family spec, so offline == runtime; an explicit group rotates every rotatable
    # quantized Linear, and 0 builds plain.
    convrot_suffixes: tuple = ()
    if args.convrot_groupsize is None:
        convrot_group, convrot_suffixes = convrot_spec_for_scheme(scheme, fam.name)
    else:
        convrot_group = int(args.convrot_groupsize)
    # What the artifact RECORDS as its base, which is not always what this build READ. Weights staged into a local
    # directory keep that directory's name, and the loader's ``_same_base_model`` compares final path segments: a
    # checkpoint built from ./temp/qwen_image_21 records a base whose tail is "qwen_image_21", the load asks for
    # "Qwen/Qwen-Image-2.1", the tails differ and a perfectly good artifact is refused after the dense shards were
    # already dropped. Pinned to the FAMILY's own base_repo rather than taken on trust, so the override can only ever
    # name the model this family is for, and cannot relabel one checkpoint as another.
    recorded_base = args.base_model_id or args.base
    if args.base_model_id:
        # EXACT, not the loader's ``_same_base_model``. That helper compares final path segments on
        # purpose, so a checkpoint built from ./temp/qwen_image_21 still matches Qwen/Qwen-Image-2.1;
        # borrowing it here would also accept ``other/Qwen-Image-2.1``, record the artifact under that
        # namespace, and have the loader's equally tolerant comparison wave it through as the official
        # base. What gets WRITTEN into a published file has to be the canonical id itself.
        if recorded_base.strip() != fam.base_repo:
            print(
                f"error: --base-model-id {recorded_base!r} is not {fam.name}'s base "
                f"({fam.base_repo!r}); it would label this checkpoint as a different model",
                flush = True,
            )
            return 2
    transformer_cls = getattr(diffusers, fam.transformer_class)
    # The CONTAINER is chosen by the --out extension, so one flag picks the on-disk format, the reachable upload name
    # and the writer, and they cannot be set to disagree.
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
        # The helpers importing is not the same question as this scheme producing something they can
        # flatten, and for int8 the two disagree through torchao 0.17. Probed here, on one tiny CPU
        # Linear, so the answer arrives in a second instead of after the download and the hours of
        # GPU quantization. None means the probe could not run, which is not evidence: proceed.
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
    # Resolved BEFORE the load, so a rotated build with nowhere resolvable to publish fails in a second rather than
    # after the quantise and the multi-gigabyte save.
    upload_dest = None
    if args.upload_repo:
        try:
            upload_dest = upload_destination(
                fam,
                scheme,
                rotated = bool(convrot_group),
                safetensors = is_safetensors_out,
                override = args.upload_filename,
                upload_repo = args.upload_repo,
            )
        except ValueError as exc:
            print(f"error: {exc}", flush = True)
            return 2

    print(f"== build prequant ({fam.name}/{scheme}, min_feat={args.min_features}) ==", flush = True)
    print(f"  loading dense transformer from {args.base} (subfolder=transformer) ...", flush = True)
    t0 = time.time()
    transformer = transformer_cls.from_pretrained(
        args.base, subfolder = "transformer", torch_dtype = torch.bfloat16, token = args.hf_token
    ).to("cuda")
    print(f"  quantising in place ({scheme}) ...", flush = True)
    # Mirror the runtime exclusions: int8 skips the M=1 modulation projections (torch._int_mm needs M>16) plus
    # per-family ones; family=None bakes linears the runtime rejects.
    exclude_name_tokens = exclude_tokens_for_scheme(scheme, fam.name)
    # fp8 / mxfp8 need bf16 weights, so skip non-bf16 Linears; nvfp4 handles fp32. Mirrors the runtime gate.
    require_bf16 = scheme in _REQUIRE_BF16_SCHEMES
    # fp8 bakes the accumulate mode in; record it so the loader can reject a contradicting request.
    fast_accum = _resolve_fast_accum(None) if scheme == TQ_FP8 else None
    filter_fn = make_filter_fn(
        args.min_features,
        exclude_name_tokens = exclude_name_tokens,
        require_bf16 = require_bf16,
    )

    # ConvRot, BEFORE quantize_: rotating the weights is only worth anything if the quantizer then sees the rotated
    # distribution. The fqn list is recorded, never re-derived at load time.
    rotation: dict = {}
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
        rotate_linears_(transformer, rotatable, group)
        rotation = rotation_metadata(group, rotatable)
        print(
            f"  rotated {len(rotatable)} linears at ConvRot group {group}; "
            f"{len(not_divisible)} quantized linears left plain (in_features not divisible)"
            + (f", e.g. {not_divisible[0]}" if not_divisible else ""),
            flush = True,
        )

    quantize_(transformer, _make_quant_config(scheme), filter_fn = filter_fn)

    state_dict = {
        k: (v.detach().to("cpu") if hasattr(v, "detach") else v)
        for k, v in transformer.state_dict().items()
    }
    metadata = {
        "base_model_id": recorded_base,
        "family": fam.name,
        "scheme": scheme,
        "min_features": args.min_features,
        # Let the loader reject a checkpoint that would not match the runtime path.
        "exclude_name_tokens": list(exclude_name_tokens),
        "require_bf16": require_bf16,
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
