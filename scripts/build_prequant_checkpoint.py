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
from pathlib import Path
from typing import Any, Optional, Sequence

BACKEND = Path(__file__).resolve().parent.parent / "studio" / "backend"

# The denoiser subfolder built when nothing else is asked for. Mirrors
# core.inference.diffusion_prequant.DEFAULT_PREQUANT_COMPONENT, which cannot be imported this
# early: the backend only joins sys.path inside main().
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

    Two registries, asked image-first because that is the one this script has always used and an
    image family answering keeps its behaviour byte for byte. Everything downstream reads
    ``fam.name``, ``fam.transformer_class`` and the prequant tables, which both dataclasses carry,
    so the video row needs no special case past here.

    ``modality`` pins one registry when the answer must not drift: the two name spaces are
    disjoint today, and a later family named in both would otherwise silently build against
    whichever registry is asked first."""
    from core.inference.diffusion_families import detect_family
    from core.inference.video_families import detect_video_family

    if modality != "video":
        fam = detect_family(base, override = override)
        if fam is not None or modality == "image":
            return fam
    return detect_video_family(base, override = override)


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


def parse_key(key: str) -> tuple:
    """``'transformer_2/blocks.12.attn1.to_q.weight'`` -> ``(component, block or None, role)``.

    Ported from ``scripts/g840/g840_fingerprint_diff.py``: WHERE two builds differ discriminates
    between mechanisms that a count cannot. A handful of modules concentrated in one role is a
    per-shape effect; a spread over every block index is something global; one block is
    order-dependent. A bare state-dict fqn carries no component prefix, so that half is None."""
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

    A fqn present in one build and absent from the other counts as differing, so a build that
    quantised a different SET of linears reports as loudly as one that quantised the same set to
    different bytes: neither is the artifact the other verified."""
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

    The whole publishing gate: build twice in separate processes and compare the packed bytes.
    Two independent quantise passes over the same weights are deterministic, so any difference is
    a defect in one of them (a bad DMA, a flipped bit on the way to disk, a silently truncated
    save) and not something to upload. It cannot see corruption that happens AFTER the compare,
    which is what the loader's own fingerprint check is for."""
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

    One file compared with itself matches by construction and proves nothing, so a gate that
    accepts it is worse than no gate: it reports a verified build and publishes it."""
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
    then loaded by every auto pick that resolves the repo. The cost of the gate is one more
    build; the cost of skipping it is a family rendering from corrupted weights with nothing in
    the logs."""
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
    # Both gates BEFORE the load: they are decided by the arguments alone, and finding out after
    # hours of GPU time that the build may not publish helps nobody.
    for refusal in (
        upload_gate_refusal(args.upload_repo, args.verify_against),
        verify_target_refusal(args.out, args.verify_against),
    ):
        if refusal:
            print(f"error: {refusal}", flush = True)
            return 2
    fam = resolve_build_family(args.base_id or args.base, override = args.family, modality = args.modality)
    if fam is None:
        print(
            f"error: unknown family '{args.family}' (modality {args.modality})",
            flush = True,
        )
        return 2
    component = (args.component or DEFAULT_COMPONENT).strip() or DEFAULT_COMPONENT
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

    print(f"== build prequant ({fam.name}/{scheme}, min_feat={args.min_features}) ==", flush = True)
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
    fast_accum = _resolve_fast_accum(None) if scheme == TQ_FP8 else None
    # The same GEMM tiling floor the runtime filter applies. Without it an offline fp8 / nvfp4
    # build bakes the ragged linears the runtime leaves dense, and the mismatch does not surface
    # until the first real matmul of the first render.
    require_divisible = divisible_for_scheme(scheme)
    filter_fn = make_filter_fn(
        args.min_features,
        exclude_name_tokens = exclude_name_tokens,
        require_bf16 = require_bf16,
        require_divisible = require_divisible,
    )

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

    quantize_(transformer, _make_quant_config(scheme), filter_fn = filter_fn)

    state_dict = {
        k: (v.detach().to("cpu") if hasattr(v, "detach") else v)
        for k, v in transformer.state_dict().items()
    }
    # Over the SAVED state dict, so it describes the bytes that go to disk rather than the module they came from.
    fingerprint = packed_weight_fingerprint(state_dict)
    metadata = {
        "base_model_id": args.base_id or args.base,
        "family": fam.name,
        "scheme": scheme,
        # Which denoiser this is. Both A14B experts share a family, a scheme, a base and a key set, so every other
        # check the loader makes passes on the wrong one.
        "component": component,
        "min_features": args.min_features,
        # Let the loader reject a checkpoint that would not match the runtime path.
        "exclude_name_tokens": list(exclude_name_tokens),
        "require_bf16": require_bf16,
        "require_divisible": require_divisible,
        # Per-weight md5 of the packed payload: the offline verify below and the loader's own check both read it.
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
    # The fingerprint is one line per quantized weight, so it is summarised here and printed in full only by
    # scripts/prequant_fingerprint.py.
    shown = {k: v for k, v in metadata.items() if k != "fingerprint"}
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
