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

A calibrated build takes --gptq-dir: every admitted linear whose GPTQ correction is MEASURED to
lower that layer's output error on held-out activations gets the corrected bf16 weight before
quantize_, the rest stay round-to-nearest, and the metadata records which was which.

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


def prequant_filename_task(component: str) -> Optional[str]:
    """The ``prequant_filenames`` task slot for ``component``: None for the default denoiser (the
    task-agnostic row), the component name otherwise, as ``denoiser_prequant_sources`` asks."""
    part = (component or "").strip()
    return None if not part or part == DEFAULT_COMPONENT else part


def families_sharing_prequant_repo(fam: Any, scheme: str, repo_id: str) -> tuple[str, ...]:
    """Other families publishing into ``repo_id`` for ``scheme``: they derive the SAME
    ``<Model>-<SCHEME>.pt`` name (HunyuanVideo-1.5 480p and 720p), so one would publish over the other."""
    from core.inference.diffusion_families import _FAMILIES as image_families
    from core.inference.video_families import _FAMILIES as video_families

    names: list[str] = []
    for other in tuple(image_families) + tuple(video_families):
        if getattr(other, "name", None) == getattr(fam, "name", None):
            continue
        hosted = [
            (entry[0], entry[1])
            for entry in (getattr(other, "prequant_repos", ()) or ())
            if len(entry) == 2
        ]
        hosted += [
            (entry[1], entry[2])
            for entry in (getattr(other, "prequant_variant_repos", ()) or ())
            if len(entry) == 3
        ]
        if any(
            entry_scheme == scheme and entry_repo == repo_id for entry_scheme, entry_repo in hosted
        ):
            names.append(str(getattr(other, "name", other)))
    return tuple(names)


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
    download. A rotated build therefore goes to the name declared for the COMPONENT it built, or
    nowhere: the task-agnostic row is the first expert's file.

    A plain build publishes under the derived ``<Model>-<SCHEME>.pt``, the layout every hosted
    prequant repo already uses and one the loader still resolves as its fallback; the legacy
    ``transformer_<scheme>.pt`` stays resolvable behind it for the repos that only ever carried it.
    A non-default ``--component`` becomes part of the name
    (``Wan2.2-T2V-A14B-transformer_2-NVFP4.pt``): the second expert is a different set of weights
    under the same family, scheme and base, so one name per component is the only thing keeping a
    resolver from serving expert 1 where expert 2 was asked for.
    The derived name identifies the artifact only while ONE family publishes into the repo; where
    several do, the plain build takes the declared name too, or is refused."""
    if override:
        return override
    from core.inference.diffusion_families import family_prequant_filename

    task = prequant_filename_task(component)
    preferred = family_prequant_filename(fam, scheme, task = task)
    if task and preferred == family_prequant_filename(fam, scheme):
        # An unmatched task slot falls back to the FIRST component's row: read that as undeclared.
        preferred = None
    if not rotated:
        from core.inference.diffusion_prequant import prequant_repo_filename

        if not repo_id:
            raise ValueError(
                "a plain build's filename derives from the destination repo, so publishing "
                "needs --upload-repo (or an explicit --upload-filename)"
            )
        shared = families_sharing_prequant_repo(fam, scheme, repo_id)
        if not shared:
            return prequant_repo_filename(repo_id, scheme, component = component)
        if not preferred:
            raise ValueError(
                f"{repo_id} also hosts {', '.join(shared)} for {scheme!r}, so the derived "
                f"filename names both and family {getattr(fam, 'name', fam)!r} declares no "
                f"prequant_filenames entry for ({scheme!r}, {task!r}) to publish under. Add the "
                "entry to the family table, or pass --upload-filename."
            )
        return preferred
    if not preferred:
        raise ValueError(
            f"family {getattr(fam, 'name', fam)!r} declares no prequant_filenames entry for "
            f"({scheme!r}, {task!r}), so a rotated checkpoint has no name the loader would ask "
            "for. Add the entry to the family table, or pass --upload-filename."
        )
    return preferred


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
    probed rather than declared, so the same --gptq-dir serves either. The FLAT layout is the default
    component's alone: the experts share every fqn and shape, so expert 1's correction would load
    into expert 2 and pass every check."""
    root = str(gptq_dir).rstrip("/")
    flat_ok = component == DEFAULT_COMPONENT
    per_component = os.path.join(root, "weights", component)
    weights = (
        per_component if exists(per_component) or not flat_ok else os.path.join(root, "weights")
    )
    meta = meta_override
    if not meta:
        named = os.path.join(root, f"gptq_meta_{component}.json")
        meta = named if exists(named) or not flat_ok else os.path.join(root, "gptq_meta.json")
    score = score_override
    if not score:
        candidates = (
            (f"gptq_score_{component}.json", f"gptq_check_{component}.json")
            if not flat_ok
            else (
                f"gptq_score_{component}.json",
                "gptq_score.json",
                f"gptq_check_{component}.json",
                "gptq_check.json",
            )
        )
        for candidate in candidates:
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


# The quantiser the calibration pass solves and scores its corrections against.
GPTQ_SCHEME = "nvfp4"


def gptq_scheme_refusal(scheme: str, gptq_dir: Optional[str]) -> Optional[str]:
    """Why GPTQ corrections may not be baked under ``scheme``, or None.

    A correction is grid-specific: the pass writes weights that already lie on the NVFP4 grid and
    its held-out do-no-harm scores were measured after NVFP4 rounding. Re-quantising them as fp8 /
    int8 / mxfp8 keeps the perturbation and drops the measurement, and nothing downstream catches
    it -- the second build reproduces the same bytes, so the fingerprint gate passes and the
    artifact publishes."""
    if not gptq_dir or scheme == GPTQ_SCHEME:
        return None
    return (
        f"--gptq-dir cannot be combined with --scheme {scheme}: the corrections lie on the "
        f"{GPTQ_SCHEME} grid and were scored there, so quantising them as {scheme} would bake a "
        "correction nobody measured."
    )


def gptq_meta_base(meta: Any) -> Optional[str]:
    """The base model the calibration pass declares it ran on, or None for a meta predating the stamp."""
    declared = (meta or {}).get("base_model_id") or (meta or {}).get("base")
    declared = str(declared or "").strip()
    return declared or None


def gptq_base_refusal(base: str, meta: Any) -> Optional[str]:
    """Why these corrections may not be replayed against ``base``, or None.

    A correction is solved against ONE checkpoint's activations and scored on ONE checkpoint's
    held-out outputs. Same-shaped variants of a model defeat every other check here: the separately
    trained HunyuanVideo-1.5 480p and 720p transformers share every fqn and shape, so a 720p build
    pointed at the 480p directory passes the per-layer shape check, reproduces itself under
    ``--verify-against``, and publishes 480p corrections under a valid 720p identity.

    A meta that names no base is accepted: the campaigns that predate the stamp declare nothing to
    compare, and refusing them would refuse the artifacts already built from them."""
    declared = gptq_meta_base(meta)
    if not declared:
        return None
    base = (base or "").strip()
    try:
        from core.inference.diffusion_prequant import _same_base_model
        same = _same_base_model(base, declared)
    except Exception:  # noqa: BLE001 -- no registry to ask means no evidence they match
        same = base.lower() == declared.lower()
    if same:
        return None
    return (
        f"the GPTQ meta was calibrated on {declared}, not on {base}. The corrections were solved "
        "and scored against that model's activations, and a same-shaped variant passes every "
        "shape check here, so replaying them would bake another model's corrections. Point "
        "--gptq-dir at this base's own campaign."
    )


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


def base_id_refusal(base: str, base_id: Optional[str]) -> Optional[str]:
    """Why ``--base-id`` may not stand in for ``--base``, or None. It exists for a LOCAL mirror;
    with a remote ``--base`` the weights come from one repo while the family, filename and stamped
    ``base_model_id`` follow another (HunyuanVideo-1.5 480p vs 720p share shapes and a repo)."""
    declared = (base_id or "").strip()
    if not declared:
        return None
    base = (base or "").strip()
    if os.path.isdir(os.path.expanduser(base)):
        return None
    try:
        from core.inference.diffusion_prequant import _same_base_model
        same = _same_base_model(base, declared)
    except Exception:  # noqa: BLE001 -- no registry to ask means no evidence they match
        same = base.lower() == declared.lower()
    if same:
        return None
    return (
        f"--base-id {declared} names a different model than --base {base}, which is not a local "
        "directory. The weights would come from --base while the family, the published filename "
        "and the recorded base_model_id all follow --base-id, so the artifact would carry another "
        "model's identity. Point --base at the local mirror --base-id declares, or drop --base-id."
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
        gptq_scheme_refusal(scheme, args.gptq_dir),
        base_id_refusal(args.base, args.base_id),
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
    # The same GEMM tiling floor the runtime filter applies. Without it an offline build bakes the ragged linears the runtime leaves dense, and the mismatch surfaces only at the first real matmul.
    require_divisible = divisible_for_scheme(scheme)
    filter_fn = make_filter_fn(
        args.min_features,
        exclude_name_tokens = exclude_name_tokens,
        require_bf16 = require_bf16,
        require_divisible = require_divisible,
    )

    # GPTQ, BEFORE quantize_: the corrected weight is a plain bf16 tensor that already lies on the NVFP4 grid, so the quantiser packs it like any other. Only the 4-bit operand is touched.
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
        refusal = gptq_base_refusal(args.base_id or args.base, gptq_pass)
        if refusal:
            print(f"error: {refusal}", flush = True)
            return 2
        if not gptq_meta_base(gptq_pass):
            print(
                f"  warning: {gptq_where['meta']} names no base model, so nothing binds these "
                "corrections to --base; a same-shaped variant's campaign would apply silently",
                flush = True,
            )
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
        admitted = [
            (fqn, module) for fqn, module in transformer.named_modules() if filter_fn(module, fqn)
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
    if gptq_plan is not None:
        # Provenance of every corrected weight. Not part of the fingerprint's identity: it hashes the packed payloads, so a different set of corrections already reads as a different artifact there.
        metadata["gptq"] = {
            # Basenames only: this block travels with the checkpoint to a public repo.
            "source": os.path.basename(os.path.abspath(args.gptq_dir)),
            "meta_path": os.path.basename(gptq_where["meta"]),
            "score_path": os.path.basename(gptq_where["score"]) if gptq_where["score"] else None,
            "score_mode": args.gptq_score_mode,
            "calibrated_on": gptq_meta_base(gptq_pass),
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
