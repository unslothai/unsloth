# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in MiniMax-H3 4-step fast mode from the pi-H3 non-DWD adapter (Lakonik/LakonLab, mirrored as unsloth/pi-H3).

At load, 150 rank-64 LoRA pairs merge in place into the resident pruned denoiser through the quantized-LoRA
requantizer, and 57 small tensors (adaLN curve projections, norm_out, output heads) are overwritten. Renders then run
plain Euler on the pi-H3 grid without CFG. Turning the mode off is a reload, which reads the base weights again."""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Optional

H3_PI_REPO = "unsloth/pi-H3"
H3_PI_REVISION = "main"
H3_PI_FILE = "pi_h3_4step_adapter/diffusion_pytorch_model.safetensors"
# Lakonik/pi-H3 at 4a864589b0fa; a mirror must hold this exact file.
H3_PI_SHA256 = "2dcdbb193f08755e0efb5f3b07585a988e30eb1c38c25283d30635ccf3d6e07f"
H3_PI_SIZE = 503_383_808
H3_PI_LORA_LAYERS = 150
H3_PI_OVERWRITES = 57
# Tests only: a directory holding H3_PI_FILE, read instead of the Hub.
H3_PI_DIR_ENV = "UNSLOTH_H3_PI_ADAPTER_DIR"
H3_PI_GUIDANCE = 1.0

_MERGED_ATTR = "_unsloth_h3_pi"
_VERIFIED: dict = {}


class H3PiUnavailable(ValueError):
    """The adapter cannot be had: the reason is the message."""


def _label(task: Optional[str]) -> str:
    return {"ref2va": "reference-to-video"}.get(str(task or ""), str(task or ""))


def h3_pi_static_refusal(
    fam: Any,
    kind: Optional[str],
    *,
    h3_task: Optional[str] = None,
    gguf_filename: Optional[str] = None,
    transformer_quant: Optional[str] = None,
    has_loras: bool = False,
) -> Optional[str]:
    """Why a load cannot take the 4-step fast mode, or None. Network-free."""
    if getattr(fam, "name", None) != "minimax-h3":
        return "The 4-step fast mode is a MiniMax-H3 option."
    if kind == "gguf":
        return (
            "The 4-step fast mode merges the pi-H3 adapter into the Diffusers denoiser; a GGUF H3 runs on "
            "stable-diffusion.cpp, which cannot take it. Load the MiniMax-H3 pipeline instead."
        )
    if gguf_filename:
        from .video_minimax_h3_pdd import h3_pdd_name
        if h3_pdd_name(gguf_filename):
            return "A PDD checkpoint is already step-distilled; the 4-step fast mode does not stack on it."
    if kind == "single_file" and not h3_task and getattr(fam, "modular_workflow", None):
        # The load infers the partition from the file name later; refuse here, before the worker stages weights.
        from .video_minimax_h3_comfy import h3_comfy_task
        h3_task = h3_comfy_task(gguf_filename)
    if h3_task == "ref2va":
        return (
            "pi-H3 has no reference-to-video adapter: the 4-step fast mode covers text-to-video, image-to-video "
            "and first/last-frame video (the fl2va partition). Reference-to-video keeps its own sampler."
        )
    if has_loras:
        return (
            "User LoRAs merge into the released dense denoiser, and the pi-H3 adapter targets the pruned one; "
            "the two cannot be combined. Load without LoRAs to use the 4-step fast mode."
        )
    tq = str(transformer_quant).strip().lower() if transformer_quant is not None else ""
    if kind == "pipeline" and tq in ("none", "off"):
        return (
            "transformer_quant none keeps the released dense bfloat16 denoiser, and the pi-H3 adapter targets "
            "the pruned one Studio hosts. Leave transformer_quant unset to use the 4-step fast mode."
        )
    return None


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verified(path: str) -> str:
    """``path`` when it is the pinned adapter file, else raises H3PiUnavailable."""
    st = os.stat(path)
    key = (os.path.realpath(path), st.st_size, st.st_mtime_ns)
    if _VERIFIED.get(key):
        return path
    if st.st_size != H3_PI_SIZE or _sha256(path) != H3_PI_SHA256:
        raise H3PiUnavailable(
            f"The pi-H3 adapter at {H3_PI_REPO} is not the expected file (sha256 mismatch), so the 4-step fast "
            "mode is unavailable."
        )
    _VERIFIED[key] = True
    return path


def _override_path() -> Optional[str]:
    root = (os.environ.get(H3_PI_DIR_ENV) or "").strip()
    return os.path.join(root, H3_PI_FILE) if root else None


def _cached_path(cache_dir: Optional[str]) -> Optional[str]:
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(
            H3_PI_REPO, H3_PI_FILE, cache_dir = cache_dir, revision = H3_PI_REVISION
        )
    except Exception:  # noqa: BLE001
        return None
    return hit if isinstance(hit, str) and os.path.isfile(hit) else None


def _unavailable(exc: BaseException) -> H3PiUnavailable:
    name = type(exc).__name__
    if name in (
        "RepositoryNotFoundError",
        "GatedRepoError",
        "EntryNotFoundError",
        "RevisionNotFoundError",
    ):
        why = f"{H3_PI_REPO} is not available on the Hugging Face Hub ({name})"
    elif name in ("LocalEntryNotFoundError", "OfflineModeIsEnabled"):
        why = f"{H3_PI_REPO} is not in the local cache and the Hub cannot be reached"
    else:
        why = f"{H3_PI_REPO} could not be fetched ({name}: {str(exc)[:200]})"
    return H3PiUnavailable(f"The 4-step fast mode is unavailable: {why}.")


def resolve_h3_pi_adapter(
    hf_token: Optional[str] = None,
    *,
    local_files_only: bool = False,
    cache_dir: Optional[str] = None,
) -> str:
    """Local path of the verified 4-step adapter, downloading it from unsloth/pi-H3 when needed. Never reads any
    other repo. Raises H3PiUnavailable with the reason."""
    override = _override_path()
    if override:
        if not os.path.isfile(override):
            raise H3PiUnavailable(f"{H3_PI_DIR_ENV} has no {H3_PI_FILE}.")
        return _verified(override)
    cached = _cached_path(cache_dir)
    if cached:
        return _verified(cached)
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            H3_PI_REPO,
            H3_PI_FILE,
            revision = H3_PI_REVISION,
            cache_dir = cache_dir,
            token = hf_token,
            local_files_only = bool(local_files_only),
        )
    except Exception as exc:  # noqa: BLE001 - every failure is a reason, never a fallback
        raise _unavailable(exc) from exc
    return _verified(path)


def h3_pi_availability(
    hf_token: Optional[str] = None,
    *,
    local_files_only: bool = False,
    cache_dir: Optional[str] = None,
) -> dict:
    """``{available, reason, repo, cached}`` without downloading the weights."""
    out = {"available": False, "reason": None, "repo": H3_PI_REPO, "cached": False}
    override = _override_path()
    if override:
        out.update(available = os.path.isfile(override), cached = os.path.isfile(override))
        if not out["available"]:
            out["reason"] = f"{H3_PI_DIR_ENV} has no {H3_PI_FILE}."
        return out
    if _cached_path(cache_dir):
        out.update(available = True, cached = True)
        return out
    if local_files_only:
        out["reason"] = (
            f"The 4-step fast mode is unavailable: {H3_PI_REPO} is not in the local cache and this load may "
            "not reach the Hub."
        )
        return out
    try:
        from huggingface_hub import HfApi
        info = HfApi().get_paths_info(
            H3_PI_REPO, [H3_PI_FILE], revision = H3_PI_REVISION, token = hf_token
        )
    except Exception as exc:  # noqa: BLE001
        out["reason"] = str(_unavailable(exc))
        return out
    lfs = getattr(info[0], "lfs", None) if info else None
    sha = getattr(lfs, "sha256", None)
    if not info:
        out["reason"] = f"The 4-step fast mode is unavailable: {H3_PI_REPO} has no {H3_PI_FILE}."
    elif sha is not None and sha != H3_PI_SHA256:
        out["reason"] = (
            f"The 4-step fast mode is unavailable: {H3_PI_REPO} holds a different adapter than the one "
            "Studio was validated with."
        )
    else:
        out["available"] = True
    return out


# ── merge ──


def _named(model: Any, items: Any) -> dict:
    return {k.replace("._orig_mod", "").replace("_orig_mod.", ""): v for k, v in items}


def _expected_counts() -> tuple[int, int]:
    return H3_PI_LORA_LAYERS, H3_PI_OVERWRITES


def _split(state_dict: dict) -> tuple[dict, dict]:
    pairs: dict = {}
    over: dict = {}
    for key, value in state_dict.items():
        if ".lora_A." in key:
            pairs.setdefault(key.split(".lora_A.")[0], {})["A"] = value
        elif ".lora_B." in key:
            pairs.setdefault(key.split(".lora_B.")[0], {})["B"] = value
        else:
            over[key] = value
    return pairs, over


def _off_accelerator(tensor: Any) -> bool:
    # A host tensor may be a read-only mmap of the cached file: never written.
    return tensor.device.type in ("cpu", "meta")


def plan_h3_pi_merge(transformer: Any, state_dict: dict) -> tuple[list, list]:
    """Every LoRA target and overwrite checked before anything is written. Raises ValueError naming the first
    mismatch: a missing or non-mergeable layer, a weight with no requantizer, a weight off the accelerator, or a
    tensor whose shape differs (the dense released denoiser, or another model)."""
    from . import diffusion_lora_quant_merge as M

    pairs, over = _split(state_dict)
    if (len(pairs), len(over)) != _expected_counts():
        raise ValueError(
            f"The pi-H3 adapter holds {len(pairs)} LoRA layers and {len(over)} tensors, expected "
            f"{H3_PI_LORA_LAYERS} and {H3_PI_OVERWRITES}."
        )
    modules = _named(transformer, transformer.named_modules())
    lora_plan = []
    for name in sorted(pairs):
        ab = pairs[name]
        module = modules.get(name)
        if module is None or "A" not in ab or "B" not in ab:
            raise ValueError(f"pi-H3 layer '{name}' is not in this denoiser.")
        if not M._mergeable_module(module):
            raise ValueError(
                f"pi-H3 layer '{name}' ({type(module).__name__}) cannot take a merged LoRA."
            )
        kind = M._weight_kind(module.weight)
        if kind is None:
            raise ValueError(
                f"pi-H3 layer '{name}' stores {type(module.weight).__name__} weights, which have no requantizer."
            )
        live = M._tensors(module.weight, kind)
        if any(_off_accelerator(t) for t in live.values()):
            raise ValueError(
                "The 4-step fast mode merges into a denoiser resident on the GPU; this load streams or offloads "
                "it. Use a card that holds the denoiser."
            )
        out_f, in_f = int(module.out_features), int(module.in_features)
        if tuple(ab["B"].shape)[0] != out_f or tuple(ab["A"].shape)[1] != in_f:
            raise ValueError(f"pi-H3 layer '{name}' does not match this denoiser's shape.")
        lora_plan.append((name, module, kind, M._group(module), ab["A"], ab["B"]))
    params = _named(transformer, transformer.named_parameters())
    params.update(_named(transformer, transformer.named_buffers()))
    over_plan = []
    for key in sorted(over):
        value = over[key]
        target = params.get(key)
        if target is None:
            raise ValueError(f"pi-H3 tensor '{key}' is not in this denoiser.")
        if tuple(target.shape) != tuple(value.shape):
            raise ValueError(
                f"pi-H3 tensor '{key}' is {tuple(value.shape)} but this denoiser's is {tuple(target.shape)}: the "
                "adapter targets the pruned MiniMax-H3 denoiser."
            )
        if M._weight_kind(target) != "dense":
            raise ValueError(
                f"pi-H3 tensor '{key}' lands on a {type(target).__name__}, which it cannot overwrite."
            )
        if _off_accelerator(target):
            raise ValueError(
                "The 4-step fast mode merges into a denoiser resident on the GPU; this load streams or offloads "
                "it. Use a card that holds the denoiser."
            )
        over_plan.append((key, target, value))
    return lora_plan, over_plan


def apply_h3_pi_adapter(
    pipe: Any,
    path: str,
    logger: Any = None,
) -> dict:
    """Merge the verified adapter into ``pipe.transformer`` in place. Raises ValueError (nothing written) when it does
    not fit; any later failure leaves a half-merged denoiser, so the caller must drop the pipeline."""
    import torch
    from safetensors.torch import load_file

    from . import diffusion_lora_quant_merge as M

    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        raise ValueError(
            "The 4-step fast mode needs the fl2va denoiser, which this pipeline does not hold."
        )
    from .video_minimax_h3_pdd import h3_pdd_heads

    if h3_pdd_heads(transformer):
        raise ValueError(
            "A PDD checkpoint is already step-distilled; the 4-step fast mode does not stack on it."
        )
    state_dict = load_file(path)
    lora_plan, over_plan = plan_h3_pi_merge(transformer, state_dict)
    started = time.monotonic()
    kinds: dict = {}
    check = None
    tf32 = torch.backends.cuda.matmul.allow_tf32
    # No TF32, as the LoRA merge: the same adapter merges to the same codes on every load.
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad():
            for index, (name, module, kind, group, a, b) in enumerate(lora_plan):
                kinds[f"{kind}/g{group}"] = kinds.get(f"{kind}/g{group}", 0) + 1
                device = next(iter(M._tensors(module.weight, kind).values())).device
                w = M._dequant(module, kind, group, device)
                base = w.clone() if index == 0 else None
                delta = b.to(device, torch.float32) @ a.to(device, torch.float32)
                w.add_(delta)
                M._requant_(module, kind, group, w, seed = index)
                if base is not None:
                    got = M._dequant(module, kind, group, device) - base
                    check = float(
                        torch.nn.functional.cosine_similarity(got.flatten(), delta.flatten(), dim = 0)
                    )
                del w, delta, base
            for key, target, value in over_plan:
                target.data.copy_(value.to(device = target.device, dtype = target.dtype))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    del state_dict
    from .video_minimax_h3_sampling import H3_PI_STEPS

    report = {
        "enabled": True,
        "steps": H3_PI_STEPS,
        "sampler": "euler",
        "sigma_grid": "pi_h3",
        "guidance": H3_PI_GUIDANCE,
        "adapter_repo": H3_PI_REPO,
        "adapter_sha256": H3_PI_SHA256,
        "adapter_source": "override" if _override_path() else "hub",
        "lora_layers": len(lora_plan),
        "overwrites": len(over_plan),
        "weight_kinds": kinds,
        "delta_kept_cos": None if check is None else round(check, 4),
        "merge_s": round(time.monotonic() - started, 2),
        "restore": "reload",
    }
    transformer.__dict__[_MERGED_ATTR] = report
    if logger is not None:
        logger.info(
            "video.h3_pi: merged %d LoRA layers (%s) and %d tensors in %.1f s",
            len(lora_plan),
            kinds,
            len(over_plan),
            report["merge_s"],
        )
    return report


def pipe_h3_pi(pipe: Any) -> Optional[dict]:
    transformer = getattr(pipe, "transformer", None)
    return (getattr(transformer, "__dict__", None) or {}).get(_MERGED_ATTR)


H3_PI_DRAFT_NOTE = (
    "Faster draft mode: about 3.6x faster than the 20-step default at similar sharpness, but prompt adherence can "
    "drop on some prompts."
)


def h3_fast_status(report: Optional[dict]) -> Optional[dict]:
    """The status view of a merge report: labelled a draft mode."""
    if not report:
        return None
    return {**report, "quality": "draft", "note": H3_PI_DRAFT_NOTE}


# ── download plan / progress ──

FETCH_ROLE = "h3_pi"


def h3_pi_fetch_files() -> tuple:
    """``(name, blob key, size)`` of the adapter, as a load's fetch plan lists files."""
    return ((H3_PI_FILE, H3_PI_SHA256, H3_PI_SIZE),)


def h3_pi_plan_entry(cache_dir: Optional[str] = None) -> Optional[dict]:
    """The download-plan entry for the adapter, None when it is already on disk."""
    if _override_path() or _cached_path(cache_dir):
        return None
    return {
        "repo_id": H3_PI_REPO,
        "files": [H3_PI_FILE],
        "bytes": H3_PI_SIZE,
        "gguf_filename": None,
        "checkpoint": False,
    }


def add_h3_pi_to_plan(plan: dict, cache_dir: Optional[str] = None) -> dict:
    """A video download plan with the adapter: an entry and its bytes while it is not on disk, always required."""
    out = dict(plan)
    entry = h3_pi_plan_entry(cache_dir)
    entries = [e for e in out.get("entries") or [] if e.get("repo_id") != H3_PI_REPO]
    if entry is not None:
        entries.append(entry)
    out["entries"] = entries
    out["total_bytes"] = sum(int(e.get("bytes") or 0) for e in entries)
    out["required_bytes"] = int(out.get("required_bytes") or 0) + H3_PI_SIZE
    return out


# ── the mode the user loaded with, for automatic reloads and resident matching ──

_CHOICES: dict = {}


def _choice_keys(names: Any, task: Optional[str]) -> list:
    part = str(task or "fl2va").strip().lower()
    return [(str(n).strip().lower(), part) for n in names if n and str(n).strip()]


def remember_h3_fast(
    names: Any,
    task: Optional[str],
    enabled: bool,
    reason: Optional[str] = None,
) -> None:
    """Record a user load's choice (or an explicit drop with its reason) under every name the model goes by."""
    for key in _choice_keys(names, task):
        _CHOICES[key] = {"enabled": bool(enabled), "reason": reason}


def _choice(names: Any, task: Optional[str]) -> dict:
    for key in _choice_keys(names, task):
        if key in _CHOICES:
            return _CHOICES[key]
    return {"enabled": False, "reason": None}


def wanted_h3_fast(names: Any, task: Optional[str]) -> bool:
    return bool(_choice(names, task)["enabled"])


def h3_fast_drop_reason(names: Any, task: Optional[str]) -> Optional[str]:
    return _choice(names, task)["reason"]


def h3_fast_matches(status: Optional[dict], names: Any, task: Optional[str]) -> bool:
    """A resident build answers for a pick only in the mode that pick's model was loaded with."""
    status = status or {}
    if not status.get("loaded"):
        return True
    return bool(status.get("h3_fast")) == wanted_h3_fast(names, task)


def auto_reload_h3_fast(
    names: Any,
    task: Optional[str],
    hf_token: Optional[str] = None,
) -> bool:
    """Whether an automatic reload keeps the mode: the user's choice while the adapter is on disk; otherwise the
    mode is dropped and the reason recorded for status."""
    if not wanted_h3_fast(names, task):
        return False
    from .diffusion import hub_cache_dir

    available = h3_pi_availability(hf_token, local_files_only = True, cache_dir = hub_cache_dir())
    if available["available"]:
        return True
    remember_h3_fast(
        names,
        task,
        False,
        reason = f"The 4-step fast mode was dropped on an automatic reload: {available['reason']}",
    )
    return False
