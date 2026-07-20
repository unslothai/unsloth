# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared, family-agnostic building blocks for diffusion LoRA training.

Everything here is independent of a specific model architecture, so the SDXL U-Net
trainer (``diffusion_lora_trainer``) and the flow-matching DiT trainers share it:
dataset discovery, the request config + validation, image loading, event emission,
the stop protocol, adapter publishing, and the family/trainer registry.

The pure helpers have no torch/diffusers import at call time and are unit-tested
without a GPU. ``run_diffusion_lora_training`` and its per-family siblings own the
actual training loop; this module only routes a request to the right one.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

from core.inference.diffusion_families import (
    detect_family,
    detect_family_for_pick,
    supported_family_names,
    trainable_family_names,
)

# Default LoRA target modules: the attention projections common to the SDXL U-Net and the DiT
# transformers (the diffusers/kohya convention). A family wanting a wider set overrides this in
# its own defaults; kept here so DiffusionLoraConfig has a sane fallback.
DEFAULT_LORA_TARGETS: tuple[str, ...] = ("to_k", "to_q", "to_v", "to_out.0")

# diffusers' SchedulerType names (diffusers.optimization.get_scheduler). piecewise_constant is
# excluded: it is the only scheduler needing a `step_rules` string, which the trainers never pass
# (and there is no config field for it). Accepting it would pass normalized(), free the resident
# GPU workloads, then crash in the child (get_piecewise_constant_schedule does
# step_rules.split(",") on None) -- the evict-then-fail this validation prevents. The other six
# run with only warmup/training steps.
_LR_SCHEDULERS: frozenset[str] = frozenset(
    {
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    }
)

# DiT families whose fp32 RoPE/embedder overflow fp16, so they train in bf16 only. Must stay
# in sync with the DiT trainer's own specs (kept separate to avoid an import cycle).
_FORCE_BF16_FAMILIES: frozenset[str] = frozenset(
    {"qwen-image", "z-image", "krea-2", "flux.2-klein", "flux.2-dev"}
)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_CAPTION_EXTS = (".txt", ".caption")
# diffusers' canonical single-file LoRA name, so load_lora_weights(dir) finds it.
DEFAULT_LORA_FILENAME = "pytorch_lora_weights.safetensors"

# Architectures Studio can neither train nor load, so they are not in the family registry but are
# recognisable by name. Rejecting them by name turns a confusing mid-run crash into a clear error.
# Registry families (flux / qwen-image / z-image / kontext) are handled by the positive check in
# ``resolve_trainable_family``, so they are intentionally absent here.
_NON_TRAINABLE_RESIDUAL_TOKENS = frozenset({"sd3", "pixart", "sana", "lumina", "cogview"})
_NON_TRAINABLE_RESIDUAL_PHRASES = ("stable-diffusion-3", "hunyuan-dit")

EventCb = Callable[[dict[str, Any]], None]
# Returns a falsy value to keep training, or a truthy stop signal: bare True, or a dict
# that may carry ``save=False`` to cancel without saving a partial adapter.
StopCb = Callable[[], Any]


def _trainable_hint() -> str:
    """A user-facing hint listing the families Studio can train today. Always names SDXL
    explicitly so the message is actionable even as more families become trainable."""
    names = ", ".join(trainable_family_names()) or "sdxl"
    return (
        f"Trainable families right now: {names} "
        f"(for example the SDXL base stabilityai/stable-diffusion-xl-base-1.0). "
        f"Other families can load LoRAs but not train them yet."
    )


def resolve_trainable_family(base_model: str, model_family: Optional[str] = None) -> str:
    """Resolve the trainer family for a base model, or raise ValueError with a clear reason.

    Positive resolution, before anything is downloaded:
    - a ``.gguf`` name can never be a training base -> reject;
    - an explicit ``model_family`` must name a registry family, and that family must be
      trainable;
    - otherwise the family is detected from the base-model name; a KNOWN but non-trainable
      family (e.g. a DiT family before its trainer ships) is rejected;
    - a name that resolves to no registry family but matches a known non-trainable
      architecture (SD3 / PixArt / ...) is rejected;
    - an unclassifiable custom name/path falls through to the SDXL trainer (backwards
      compatible: a genuinely wrong pick still fails cleanly later in from_pretrained).
    """
    name = str(base_model or "").strip().lower()
    # GGUF weights (a ``.gguf`` file or ``*-GGUF`` repo) are inference-only: training needs the
    # full diffusers pipeline (transformer + VAE + text encoders), which a GGUF repo lacks. Reject
    # by name even when the family is trainable. Exempt a local diffusers checkout that merely has
    # "gguf" in its path, identified by its ``model_index.json`` marker, not a bare ``is_dir()``.
    local = Path(base_model).expanduser() if base_model else None
    is_local_diffusers = bool(local and (local / "model_index.json").is_file())
    if name.endswith(".gguf") or ("gguf" in name and not is_local_diffusers):
        raise ValueError(
            f"'{base_model}' is a GGUF checkpoint/repo, which can't be a training base "
            f"(training needs the full diffusers model). {_trainable_hint()}"
        )
    if model_family and str(model_family).strip():
        key = str(model_family).strip().lower()
        fam = detect_family("", override = key)
        if fam is None:
            known = ", ".join(supported_family_names())
            raise ValueError(f"Unknown model_family {model_family!r}. Known families: {known}.")
        if not fam.trainable:
            raise ValueError(f"'{fam.name}' models can't be trained yet. {_trainable_hint()}")
        return fam.name

    fam = detect_family_for_pick(base_model)
    if fam is not None:
        if not fam.trainable:
            raise ValueError(
                f"'{base_model}' looks like a {fam.name} model, which isn't trainable yet. "
                f"{_trainable_hint()}"
            )
        return fam.name

    condensed = re.sub(r"[^a-z0-9]+", "-", name)
    hit = next(
        (p for p in _NON_TRAINABLE_RESIDUAL_PHRASES if p in condensed),
        None,
    ) or next(
        (t for t in condensed.split("-") if t in _NON_TRAINABLE_RESIDUAL_TOKENS),
        None,
    )
    if hit:
        raise ValueError(
            f"'{base_model}' looks like a {hit} model, which isn't trainable. {_trainable_hint()}"
        )
    # Unknown custom name / local path: default to the SDXL trainer (unchanged behaviour).
    return "sdxl"


def repo_is_prequantized(base_model: str) -> bool:
    """Heuristic: a repo whose name marks a bitsandbytes 4-bit build already ships a
    quantized transformer, so it loads as-is for nf4 and cannot serve the dense
    (bf16/int8/fp8/mxfp8) base precisions."""
    name = str(base_model or "").lower()
    return "bnb-4bit" in name or "-4bit" in name or "int4" in name or "nf4" in name


def _module_is_torchao_stub(module: Any) -> bool:
    """True iff ``module`` is the Unsloth Windows-ROCm torchao import stub rather than the
    real package. The stub (core/_torchao_stub.py) satisfies find_spec and even lets
    ``from torchao.quantization import quantize_`` succeed -- but the imported symbols are
    no-op stub types, so the quantization never happens. Every stub module carries the
    ``_unsloth_stub`` sentinel, so match on it (comparing against the stub module's own
    sentinel object, not identity of a re-created one)."""
    if module is None:
        return False
    sentinel = getattr(module, "_unsloth_stub", None)
    if sentinel is None:
        return False
    try:
        from core._torchao_stub import _STUB_SENTINEL
    except Exception:  # noqa: BLE001 -- stub module absent -> nothing to compare against
        return False
    return sentinel is _STUB_SENTINEL


def has_functional_torchao() -> bool:
    """True iff the real torchao quantization API is importable (not the Windows-ROCm stub).

    ``_int8_quantize_base`` needs ``Int8WeightOnlyConfig`` + ``quantize_`` from
    ``torchao.quantization`` and has no runtime fallback, so gate both the auto int8 pick
    and the advertised int8 mode on a FUNCTIONAL import: a plain ``find_spec("torchao")``
    is satisfied by the stub, whose quantize_ is a no-op that leaves the transformer dense
    while compile is disabled as if it were int8. Import the exact symbols the int8 path
    uses and reject the stub module. Never raises."""
    try:
        import importlib

        quant = importlib.import_module("torchao.quantization")
        if _module_is_torchao_stub(quant):
            return False
        # The symbols the int8 path actually imports must exist on the real module.
        return hasattr(quant, "Int8WeightOnlyConfig") and hasattr(quant, "quantize_")
    except Exception:  # noqa: BLE001 -- torchao absent / broken build -> treat as unavailable
        return False


def train_precision_modes() -> tuple[list[str], str]:
    """(supported base_precision modes, recommended pick) for the current machine: nf4
    always works; bf16/auto need a bf16-capable CUDA GPU (Ampere+); int8/fp8/mxfp8 additionally
    need a FUNCTIONAL torchao (their explicit paths import torchao with no fallback, and the
    Windows-ROCm stub only looks installed). fp8 also needs an fp8-capable GPU (sm89+); mxfp8
    (block-scaled fp8 compute) needs the Blackwell tensor cores (sm100+) its cuBLAS kernels
    target. The dense modes all train in bf16 compute, which the DiT trainer requires, so a
    non-bf16 CUDA GPU (T4/V100/RTX 20xx) is offered only nf4 -- otherwise /info would advertise
    a start that evicts resident models and then fails the trainer's bf16 guard. Used by the
    /info endpoint so the UI can gate the precision selector. Never raises."""
    modes = ["nf4"]
    recommended = "nf4"
    try:
        import torch
        if native_bf16_supported():
            modes.append("bf16")
            torchao_ok = has_functional_torchao()
            if torchao_ok:
                modes.append("int8")
            major, minor = torch.cuda.get_device_capability()
            if torchao_ok and (major, minor) >= (8, 9) and hasattr(torch, "float8_e4m3fn"):
                modes.append("fp8")
            if torchao_ok and (major, minor) >= (10, 0):
                modes.append("mxfp8")
            modes.append("auto")
            recommended = "auto"
    except Exception:  # noqa: BLE001 -- no torch / probe failure -> nf4 only
        pass
    return modes, recommended


def get_trainer(family: str) -> Callable[..., str]:
    """Return the training entrypoint for ``family``. Imports the trainer module lazily so
    this shared module stays free of the heavy trainer imports (and any import cycle)."""
    key = (family or "sdxl").strip().lower()
    if key == "sdxl":
        from core.training.diffusion_lora_trainer import run_diffusion_lora_training
        return run_diffusion_lora_training
    if key in ("flux.1", "qwen-image", "z-image", "krea-2", "flux.2-klein", "flux.2-dev"):
        from core.training.diffusion_dit_trainer import run_dit_lora_training
        return run_dit_lora_training
    raise ValueError(f"No trainer is registered for family {family!r}.")


# Per-family training defaults surfaced by the Train UI. Distilled/turbo bases and the big DiTs
# want different rank / learning rate / resolution; these are starting points, not hard limits.
# Families absent here fall back to the DiffusionLoraConfig defaults.
FAMILY_TRAIN_DEFAULTS: dict[str, dict[str, Any]] = {
    "sdxl": {"lora_rank": 16, "learning_rate": 1e-4, "resolution": 1024},
    # Warmup defaults: a short LR ramp keeps the first adapter updates from overshooting on
    # the big flow-matching DiTs (whose logit-normal timestep draw concentrates loss mass
    # mid-schedule); the small warmups below are scaled for Studio's short-run step budgets.
    "flux.1": {"lora_rank": 16, "learning_rate": 1e-4, "resolution": 512, "lr_warmup_steps": 20},
    "qwen-image": {
        "lora_rank": 16, "learning_rate": 5e-5, "resolution": 512, "lr_warmup_steps": 20,
    },
    "z-image": {"lora_rank": 16, "learning_rate": 1e-4, "resolution": 768},
    # The Krea 2 authors' recommended starting point (their DreamBooth script defaults):
    # rank/alpha 32, lr 3e-4, 512px.
    "krea-2": {"lora_rank": 32, "learning_rate": 3e-4, "resolution": 512},
    # The upstream FLUX.2 DreamBooth references default to rank 16 / lr 1e-4; FLUX.2's
    # uniform timestep draw benefits most from a warmup ramp.
    "flux.2-klein": {
        "lora_rank": 16, "learning_rate": 1e-4, "resolution": 512, "lr_warmup_steps": 20,
    },
    "flux.2-dev": {
        "lora_rank": 16, "learning_rate": 1e-4, "resolution": 512, "lr_warmup_steps": 20,
    },
}


def train_defaults(family: str) -> dict[str, Any]:
    """Recommended starting hyperparameters for ``family`` (empty if unknown)."""
    return dict(FAMILY_TRAIN_DEFAULTS.get((family or "").strip().lower(), {}))


# Display labels + a short VRAM/access note per trainable family, surfaced by the Train UI
# so users pick a base with realistic expectations. Kept next to the defaults they pair with.
_FAMILY_LABELS = {
    "sdxl": "SDXL",
    "flux.1": "FLUX.1-dev",
    "qwen-image": "Qwen-Image",
    "z-image": "Z-Image",
    "krea-2": "Krea 2",
    "flux.2-klein": "FLUX.2 Klein",
    "flux.2-dev": "FLUX.2-dev",
}
_FAMILY_VRAM_NOTES = {
    "sdxl": "Trains on ~12 GB+ (bf16 LoRA). The lightest, fastest option.",
    "flux.1": (
        "12B model, QLoRA (nf4) by default (~16 GB+). Gated on Hugging Face: accept the "
        "FLUX.1-dev license and add your HF token before training."
    ),
    "qwen-image": "20B model, QLoRA (nf4) by default (~24 GB+). The heaviest option.",
    "z-image": "6B model, QLoRA (nf4) by default (~12 GB+). bf16 only.",
    "krea-2": (
        "12B model, QLoRA (nf4) by default (~18 GB+). bf16 only. Trains on the "
        "undistilled Krea-2-Raw (Krea's guidance: train on Raw, run adapters on Turbo)."
    ),
    "flux.2-klein": "4B model, QLoRA (nf4) by default (~10 GB+). bf16 only.",
    "flux.2-dev": (
        "32B model, QLoRA (nf4) by default (~28 GB+). bf16 only. Gated on Hugging Face: "
        "accept the FLUX.2-dev license and add your HF token before training."
    ),
}

# The flow-matching DiT families (run by diffusion_dit_trainer). They expose the base_precision /
# compile levers and require bf16 compute on CUDA; SDXL is absent (it uses its own
# mixed_precision path). A set so the UI gate, the bf16 preflight, and any future dispatch stay
# in sync.
_DIT_TRAIN_FAMILIES = frozenset(
    {"flux.1", "qwen-image", "z-image", "krea-2", "flux.2-klein", "flux.2-dev"}
)


def native_bf16_supported() -> bool:
    """True only when the live CUDA GPU provides NATIVE bf16 compute, not pre-Ampere emulation.

    ``torch.cuda.is_bf16_supported()`` defaults to counting EMULATED bf16, which every pre-Ampere
    CUDA card (T4 / V100 / RTX 20xx) reports as supported even though the DiT trainer needs real
    Ampere-or-newer bf16. Gate NVIDIA on compute capability major >= 8 instead -- the same #6658
    fix the inference device resolver (``diffusion_device.py``) already uses; ROCm has no such
    quirk, so ``is_bf16_supported()`` is trustworthy there. Never raises -- a probe failure or a
    no-CUDA host returns False. Shared by the /info modes, the start preflight, and the trainer
    guard so all three stay in sync."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        is_rocm = bool(getattr(getattr(torch, "version", None), "hip", None))
        if is_rocm:
            return bool(torch.cuda.is_bf16_supported())
        return torch.cuda.get_device_capability()[0] >= 8
    except Exception:  # noqa: BLE001 -- no torch / probe failure -> treat as unsupported
        return False


def bf16_unsupported_reason(resolved_family: str) -> Optional[str]:
    """Return a user-facing error string if ``resolved_family`` needs bf16 compute that the
    live GPU cannot provide, else None. The DiT trainer requires a bf16-capable GPU (Ampere
    or newer) and otherwise raises deep in model load; the start route uses this to fail fast
    BEFORE evicting resident GPU workloads. CPU-only hosts (which fall back to fp32 for
    import/unit tests) and SDXL (its own mixed_precision path) are exempt. Never raises."""
    if (resolved_family or "").strip().lower() not in _DIT_TRAIN_FAMILIES:
        return None
    try:
        import torch
        if torch.cuda.is_available() and not native_bf16_supported():
            return (
                "This trainer requires a bfloat16-capable GPU (Ampere or newer); this CUDA "
                "device does not support bf16. Train the DiT families on a newer GPU."
            )
    except Exception:  # noqa: BLE001 -- torch probe failure must not block a start
        return None
    return None


def training_precision_preflight_error(resolved_family: str, base_precision: str) -> Optional[str]:
    """Reason the requested DiT precision cannot run on this host, else None -- checked by the
    start route BEFORE evicting resident GPU workloads (the trainer's own checks fire only in the
    child, after eviction). Four gates, all mirroring _resolve_base_precision so a doomed run is
    rejected before teardown: the bf16-GPU requirement (bf16_unsupported_reason); the dense
    precisions (bf16/int8/fp8/mxfp8) requiring a CUDA GPU; an explicit int8 needing a FUNCTIONAL
    torchao (its _int8_quantize_base has no fallback); and an explicit mxfp8 needing a Blackwell
    (sm100+) GPU (its MX GEMM has no kernel below sm100). Never raises."""
    reason = bf16_unsupported_reason(resolved_family)
    if reason:
        return reason
    fam = (resolved_family or "").strip().lower()
    mode = (base_precision or "").strip().lower()
    if fam in _DIT_TRAIN_FAMILIES and mode in ("bf16", "int8", "fp8", "mxfp8"):
        # The DiT trainer's dense precisions all require CUDA (_resolve_base_precision rejects
        # bf16/int8/fp8/mxfp8 on device != "cuda"). bf16_unsupported_reason exempts a CPU-only host
        # (the fp32 fallback for tests), so without this a dense request on a GPU-less host would
        # pass the preflight, evict residents, then raise only in the child.
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except Exception:  # noqa: BLE001 -- no torch / probe failure -> treat as no CUDA
            has_cuda = False
        if not has_cuda:
            return (
                f"base_precision={mode!r} needs a CUDA GPU; this host has none. "
                "Use base_precision='nf4' or 'auto'."
            )
        if mode == "int8" and not has_functional_torchao():
            return (
                "base_precision='int8' needs a functional torchao install; this host's torchao is "
                "missing or the non-functional Windows-ROCm stub. Use 'nf4', 'bf16', or 'auto'."
            )
        # mxfp8 needs Blackwell (sm100+): its MX GEMM has no kernel below sm100 and raises at the
        # first training step, AFTER a full dense load. Re-check here (mirroring
        # _resolve_base_precision) so a stale/direct client on an older GPU fails fast before
        # eviction instead of crashing mid-run.
        if mode == "mxfp8":
            try:
                import torch
                blackwell = torch.cuda.get_device_capability() >= (10, 0)
            except Exception:  # noqa: BLE001 -- probe failure -> treat as unsupported, fail fast
                blackwell = False
            if not blackwell:
                return (
                    "base_precision='mxfp8' needs a Blackwell (sm100+) GPU; this GPU is older. "
                    "Use base_precision='bf16', 'int8', 'nf4', or 'auto'."
                )
    return None


def family_train_infos() -> list[dict[str, Any]]:
    """Describe every trainable family for the Train UI: name, label, the default + allowed
    base repos, the recommended starting hyperparameters, and a VRAM/access note. Built from
    the family registry so it stays in sync with what the trainers actually support."""
    from core.inference.diffusion_families import detect_family
    from core.inference.diffusion_transformer_quant import _family_denied

    dit_modes, dit_recommended = train_precision_modes()
    infos: list[dict[str, Any]] = []
    for name in trainable_family_names():
        fam = detect_family("", override = name)
        if fam is None:
            continue
        repos = list(fam.train_base_repos) or [fam.base_repo]
        # base_precision applies to the DiT trainer only; SDXL keeps its mixed_precision lever, so
        # the UI hides the precision selector for it. compile applies everywhere: the SDXL trainer
        # regionally compiles the U-Net's transformer blocks too.
        is_dit = name in _DIT_TRAIN_FAMILIES
        # On a non-bf16 CUDA GPU the start preflight rejects EVERY DiT family (even nf4, since the
        # DiT trainer requires bf16 on CUDA), so advertise no precision -- else /info offers an nf4
        # DiT option that always 400s. Otherwise drop any scheme this family's DiT corrupts (fp8 on
        # Qwen-Image: activation outliers exceed fp8's range; the inference path denies the same
        # set), so the UI never offers a mode normalized() would reject.
        dit_block = bf16_unsupported_reason(name) if is_dit else None
        if not is_dit or dit_block:
            fam_modes: list[str] = []
        else:
            fam_modes = [m for m in dit_modes if not _family_denied(name, m)]
        infos.append(
            {
                "name": name,
                "label": _FAMILY_LABELS.get(name, name),
                "default_base": repos[0],
                "base_repos": repos,
                "defaults": train_defaults(name),
                "vram_note": dit_block or _FAMILY_VRAM_NOTES.get(name, ""),
                "precision_modes": fam_modes,
                "recommended_precision": "nf4" if (not is_dit or dit_block) else dit_recommended,
                # compile is offered everywhere (SDXL regional U-Net + DiT), except a DiT family
                # the GPU can't train in bf16 (dit_block), where training is refused outright.
                "supports_compile": bool(not dit_block),
                # Krea trains on Raw but previews adapters on Turbo; None elsewhere.
                "deploy_base": fam.deploy_base_repo,
            }
        )
    return infos


@dataclass
class DiffusionLoraConfig:
    """Everything a diffusion LoRA training run needs. Only ``base_model`` /
    ``data_dir`` / ``output_dir`` are required; the rest have sensible defaults."""

    base_model: str
    data_dir: str
    output_dir: str
    # Dreambooth-style caption applied to any image without its own caption. Required if
    # the dataset has no captions.jsonl / sidecar files.
    instance_prompt: Optional[str] = None
    resolution: int = 1024
    train_steps: int = 500
    # 0 = disabled (train for train_steps). > 0 overrides train_steps with a run length of
    # num_epochs full passes over the dataset, in optimizer steps (see resolve_train_steps).
    num_epochs: int = 0
    learning_rate: float = 1e-4
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    lora_rank: int = 16
    lora_alpha: Optional[int] = None  # defaults to lora_rank
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = DEFAULT_LORA_TARGETS
    seed: int = 42
    mixed_precision: str = "bf16"  # "bf16" | "fp16" | "no"
    snr_gamma: Optional[float] = 5.0  # min-SNR loss weighting; None disables
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    center_crop: bool = False
    random_flip: bool = True
    caption_column: str = "text"  # column in metadata.jsonl
    adapter_name: str = "default"
    hf_token: Optional[str] = None
    # Precompute the VAE latents once (freeing the VAE for the run) instead of re-encoding every
    # step. ``cache_variants`` crop/flip draws are frozen per image; the per-step VAE sampling
    # noise itself is preserved (see the DiT trainer docstring).
    cache_latents: bool = True
    cache_variants: int = 4
    # Persistent on-disk conditioning cache directory (DiT trainer only). None (default)
    # keeps the in-memory-only behavior; a configured directory turns the cache on: latent
    # posterior stats and caption embeddings persist as safetensors keyed by content hash +
    # family + resolution, and a fully warm cache skips loading the VAE and the multi-GB
    # text encoders entirely on the next run.
    cond_cache_dir: Optional[str] = None
    # LoRA EMA decay (DiT trainer only). 0.0 (default) disables it; > 0 keeps an
    # exponential moving average of the trainable LoRA params (warmup-ramped so short runs
    # still absorb the trajectory) and exports it as a second adapter under
    # ``<output_dir>/ema`` next to the primary one.
    ema_decay: float = 0.0
    # Regional torch.compile of the transformer blocks: "off" | "on" | "auto" (auto turns
    # it on only for a dense, non-bitsandbytes base where it is a clean win).
    compile_transformer: str = "auto"
    # TF32 matmuls + high fp32 matmul precision + cudnn autotuning for the run. Near-lossless;
    # disable for strict bit-reproducibility A/Bs.
    enable_tf32: bool = True
    # DiT base transformer precision: "nf4" (bitsandbytes QLoRA, the memory floor and default),
    # "bf16" (dense, fastest eager, compile-friendly), "int8" (torchao weight-only, half of bf16),
    # "fp8" (torchao float8 training on the frozen linears, Ada/Hopper/Blackwell + compile), or
    # "auto" (by free VRAM + GPU class). Non-nf4 modes need a dense base repo. SDXL ignores it.
    base_precision: str = "nf4"
    # Training-time timestep shift applied to the flow-matching sigma draw. None resolves
    # per family in normalized(): "auto" for qwen-image (reproduce the family's inference
    # sigma distribution: the scheduler's exponential time_shift at mu = max_shift, then the
    # shift_terminal stretch), 1.0 (identity, the historical behavior) for every other
    # family. A numeric value applies the standard linear shift s*u/(1+(s-1)*u); 1.0 is a
    # no-op. "auto" on a family without dynamic shifting falls back to identity.
    flow_shift: Optional[Any] = None  # float | "auto" | None
    # Per-sample probability of replacing the caption conditioning with the empty prompt
    # (classifier-free-guidance dropout). 0.0 (default) disables it entirely.
    cfg_dropout: float = 0.0
    # Per-sample loss weighting over the drawn timestep: "none" (default, unweighted MSE)
    # or "bell" (bsmntw-style Gaussian bell centered mid-schedule, normalized to mean 1).
    weighting_scheme: str = "none"
    # How often to emit a progress event (in optimizer steps).
    log_every: int = 1
    # Optional explicit family override ("sdxl" / "flux.1" / ...); None = detect from base_model.
    # ``resolved_family`` is filled by normalized() with the trainer family that will run, so the
    # process adapter can dispatch through the registry.
    model_family: Optional[str] = None
    resolved_family: str = "sdxl"

    def normalized(self) -> "DiffusionLoraConfig":
        """Return a copy with derived/validated fields filled in. Raises ValueError on a
        request that cannot train (bad numbers, or an untrainable base model).

        Also coerces values that arrive as strings/blanks through the Studio config path
        (``learning_rate`` is preserved as a string there; ``hf_token`` defaults to "")."""
        resolved_family = resolve_trainable_family(self.base_model, self.model_family)
        if self.train_steps < 1:
            raise ValueError("train_steps must be >= 1")
        if not 0 <= int(self.num_epochs) <= 1000:
            raise ValueError("num_epochs must be between 0 and 1000 (0 uses train_steps)")
        if self.train_batch_size < 1:
            raise ValueError("train_batch_size must be >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.lora_rank < 1:
            raise ValueError("lora_rank must be >= 1")
        if self.lora_alpha is not None and self.lora_alpha < 1:
            raise ValueError(
                "lora_alpha must be >= 1 (a zero/negative alpha scales the adapter to nothing)"
            )
        if self.resolution < 64 or self.resolution % 8 != 0:
            raise ValueError("resolution must be a multiple of 8 and >= 64")
        if self.mixed_precision not in ("bf16", "fp16", "no"):
            raise ValueError("mixed_precision must be one of bf16 / fp16 / no")
        # Refuse fp16 for a bf16-only DiT family up front, before evicting resident models.
        if self.mixed_precision == "fp16" and resolved_family in _FORCE_BF16_FAMILIES:
            raise ValueError(
                f"'{resolved_family}' LoRA training requires bf16: fp16 overflows its fp32 "
                f"RoPE / embedder internals. Set mixed precision to bf16."
            )
        if str(self.lr_scheduler) not in _LR_SCHEDULERS:
            raise ValueError(
                f"lr_scheduler must be one of {', '.join(sorted(_LR_SCHEDULERS))}; "
                f"got {self.lr_scheduler!r}"
            )
        if not 1 <= int(self.cache_variants) <= 16:
            raise ValueError("cache_variants must be between 1 and 16")
        try:
            ema_decay = float(self.ema_decay or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ema_decay must be a number, got {self.ema_decay!r}") from exc
        # decay = 1.0 would freeze the shadow at its init forever; the EMA update is
        # shadow * decay + param * (1 - decay), so valid decays live in [0, 1).
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1); 0 disables the EMA adapter")
        # A blank cond_cache_dir (the Studio default when unset) means "off", not cwd.
        cond_cache_dir = (
            str(self.cond_cache_dir).strip() if self.cond_cache_dir is not None else ""
        ) or None
        compile_transformer = str(self.compile_transformer or "auto").strip().lower()
        if compile_transformer not in ("off", "on", "auto"):
            raise ValueError("compile_transformer must be one of off / on / auto")
        base_precision = str(self.base_precision or "nf4").strip().lower()
        if base_precision not in ("nf4", "bf16", "int8", "fp8", "mxfp8", "auto"):
            raise ValueError("base_precision must be one of nf4 / bf16 / int8 / fp8 / mxfp8 / auto")
        # base_precision is a DiT-only lever (transformer load precision); SDXL uses its own
        # mixed_precision path and ignores it, so the dense-mode gates (prequant base / non-bf16
        # compute) apply only to the DiT families. The mode-name check above still runs for every
        # family.
        if resolved_family != "sdxl" and base_precision in ("bf16", "int8", "fp8", "mxfp8"):
            if repo_is_prequantized(self.base_model):
                raise ValueError(
                    f"base_precision={base_precision!r} needs a dense base repo, but "
                    f"'{self.base_model}' is already bitsandbytes-quantized. Pick the "
                    f"family's dense (bf16) base repo for this mode, or use nf4/auto."
                )
            if self.mixed_precision != "bf16":
                raise ValueError(
                    f"base_precision={base_precision!r} trains in bf16 compute; set "
                    f"mixed_precision to bf16."
                )
            # Some DiT families are corrupted by fp8's activation range: outliers exceed even
            # per-row fp8's range, so the frozen linears' float8 compute learns against a garbage
            # forward pass. The inference path already denies these schemes; mirror it here so the
            # run fails fast instead of producing a broken adapter. int8 (per-token) is unaffected.
            from core.inference.diffusion_transformer_quant import _family_denied

            if _family_denied(resolved_family, base_precision):
                raise ValueError(
                    f"base_precision={base_precision!r} is not supported for "
                    f"{resolved_family}: its activations exceed fp8's range and corrupt the "
                    f"trained result. Use 'nf4', 'int8', 'bf16', or 'auto'."
                )
        # flow_shift: None resolves to the family default ("auto" only for qwen-image, whose
        # scheduler skips its static shift under use_dynamic_shifting and would otherwise
        # train on unshifted uniform sigmas); an explicit value is validated and kept.
        flow_shift = self.flow_shift
        if flow_shift is None:
            flow_shift = "auto" if resolved_family == "qwen-image" else 1.0
        if isinstance(flow_shift, str):
            flow_shift = flow_shift.strip().lower()
            if flow_shift != "auto":
                try:
                    flow_shift = float(flow_shift)
                except ValueError as exc:
                    raise ValueError(
                        f"flow_shift must be a positive number or 'auto', got {self.flow_shift!r}"
                    ) from exc
        if not isinstance(flow_shift, str):
            flow_shift = float(flow_shift)
            if flow_shift <= 0:
                raise ValueError("flow_shift must be > 0 (1.0 disables the shift), or 'auto'")
        try:
            cfg_dropout = float(self.cfg_dropout or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"cfg_dropout must be a number, got {self.cfg_dropout!r}") from exc
        if not 0.0 <= cfg_dropout <= 1.0:
            raise ValueError("cfg_dropout must be between 0 and 1")
        weighting_scheme = str(self.weighting_scheme or "none").strip().lower()
        if weighting_scheme not in ("none", "bell"):
            raise ValueError("weighting_scheme must be one of none / bell")
        # A zero/negative gamma would zero out (or invert) the min-SNR weight and
        # silently train on a degenerate loss; None is the documented disable.
        if self.snr_gamma is not None and float(self.snr_gamma) <= 0:
            raise ValueError("snr_gamma must be > 0, or null to disable min-SNR weighting")
        # learning_rate can arrive as a string ("1e-4") from the Studio config path, which
        # preserves it as a string after validation; coerce so AdamW receives a float.
        try:
            learning_rate = float(self.learning_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"learning_rate must be a number, got {self.learning_rate!r}") from exc
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        alpha = self.lora_alpha if self.lora_alpha is not None else self.lora_rank
        targets = tuple(self.lora_target_modules) or DEFAULT_LORA_TARGETS
        # A blank Hub token (the Studio default when none is configured) must load
        # anonymously, not as an explicit empty credential.
        token = self.hf_token.strip() if isinstance(self.hf_token, str) else self.hf_token
        return replace(
            self,
            learning_rate = learning_rate,
            lora_alpha = alpha,
            lora_target_modules = targets,
            max_grad_norm = float(self.max_grad_norm),
            hf_token = token or None,
            num_epochs = int(self.num_epochs),
            cache_variants = int(self.cache_variants),
            cond_cache_dir = cond_cache_dir,
            ema_decay = ema_decay,
            compile_transformer = compile_transformer,
            base_precision = base_precision,
            flow_shift = flow_shift,
            cfg_dropout = cfg_dropout,
            weighting_scheme = weighting_scheme,
            resolved_family = resolved_family,
        )


def resolve_train_steps(cfg: "DiffusionLoraConfig", n_images: int) -> int:
    """The effective optimizer-step count for a run. When ``cfg.num_epochs`` is set (> 0),
    one epoch is one full pass over the dataset in optimizer steps -- ceil(N / (batch x
    grad_accum)) steps -- so the run is ``num_epochs`` such passes, capped at 100000. With
    ``num_epochs == 0`` the explicit ``cfg.train_steps`` is used unchanged."""
    if cfg.num_epochs > 0:
        per_step = max(1, cfg.train_batch_size * cfg.gradient_accumulation_steps)
        steps_per_epoch = max(1, math.ceil(n_images / per_step))
        return min(100000, cfg.num_epochs * steps_per_epoch)
    return cfg.train_steps


class PermutationBatchSampler:
    """Yields batch indices as consecutive slices of a reshuffled permutation of
    ``range(n)``, so every index is visited exactly once per cycle before any repeats --
    an epoch-style full pass instead of the with-replacement draw that leaves part of a
    small dataset unseen at low step counts (num_epochs converts to a step budget, but the
    per-batch index draw is what decides coverage). When a cycle is exhausted the order is
    reshuffled from the run's own ``rng`` so the index stream stays seed-deterministic and
    each cycle differs.

    Both trainers share this so the SDXL ``_next_batch`` path and the DiT per-sample draw
    select indices the same way. Only the index selection changes (with-replacement ->
    permutation cycles); step count and batch shapes are unchanged.
    """

    def __init__(self, n: int, rng: random.Random) -> None:
        if n <= 0:
            raise ValueError("PermutationBatchSampler needs at least one item")
        self._n = n
        self._rng = rng
        self._order: list[int] = []
        self._pos = 0

    def _reshuffle(self) -> None:
        self._order = list(range(self._n))
        self._rng.shuffle(self._order)
        self._pos = 0

    def next_batch(self, k: int) -> list[int]:
        # k may exceed n (batch larger than the dataset): the permutation is refilled across as
        # many cycles as needed so the caller always gets exactly k indices and the batch never
        # shrinks, matching the old sampler's fixed batch shape.
        out: list[int] = []
        while len(out) < k:
            if self._pos >= len(self._order):
                self._reshuffle()
            take = min(k - len(out), len(self._order) - self._pos)
            out.extend(self._order[self._pos : self._pos + take])
            self._pos += take
        return out


def discover_image_caption_pairs(
    data_dir: str | os.PathLike[str],
    *,
    instance_prompt: Optional[str] = None,
    caption_column: str = "text",
    verify_images: bool = False,
) -> list[tuple[str, str]]:
    """Resolve ``(image_path, caption)`` pairs from a dataset directory.

    Caption sources, in priority order per image:
      1. a per-image sidecar ``<stem>.txt`` / ``<stem>.caption``,
      2. a ``metadata.jsonl`` / ``captions.jsonl`` row keyed by ``file_name`` (or ``image``)
         carrying the caption in ``caption_column`` (default ``text``),
      3. ``instance_prompt`` (dreambooth) for any remaining image.

    A sidecar wins over the metadata row because it is the user's explicit per-image edit
    (the labeling grid writes a .txt sidecar), which must override the bulk metadata file.
    Must agree with ``routes.training._image_record``, which resolves captions the same way.

    Images with no caption from any source are skipped. Pure filesystem + JSON, so it is
    unit-testable without torch. Raises FileNotFoundError for a missing dir and ValueError
    when nothing is captionable.

    ``verify_images`` (opt-in) additionally runs a cheap PIL header probe on each captioned
    image and raises ValueError on a corrupt/zero-byte/truncated file. The start route enables
    it so a bad upload is rejected BEFORE the resident GPU models are freed, instead of crashing
    the spawned trainer after teardown; the trainers leave it off (they decode every image
    anyway, so a second probe pass would be redundant).
    """
    root = Path(data_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"data_dir is not a directory: {data_dir}")

    images = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)

    # 1. metadata.jsonl / captions.jsonl (either name accepted).
    meta_caption: dict[str, str] = {}
    for meta_name in ("metadata.jsonl", "captions.jsonl"):
        meta_path = root / meta_name
        if not meta_path.is_file():
            continue
        # Tolerate a bad upload (invalid UTF-8, or a line of non-object JSON): skip the record so
        # the instance_prompt fallback still applies rather than crashing the trainer.
        try:
            meta_lines = meta_path.read_text(encoding = "utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        for line in meta_lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(row, dict):
                continue
            key = row.get("file_name") or row.get("image") or row.get("file")
            if key and caption_column in row:
                meta_caption[str(key)] = str(row[caption_column])

    pairs: list[tuple[str, str]] = []
    for img in images:
        caption: Optional[str] = None
        sidecar_present = False
        # 1. per-image sidecar caption file (the user's explicit edit; wins over metadata).
        #    An EMPTY sidecar is a deliberate tombstone (written when a user clears a caption): it
        #    suppresses the metadata caption but leaves the image uncaptioned so the instance_prompt
        #    fallback below still applies, rather than dropping the image.
        for ext in _CAPTION_EXTS:
            sidecar = img.with_suffix(ext)
            if sidecar.is_file():
                sidecar_present = True
                caption = sidecar.read_text(encoding = "utf-8").strip()
                break
        # 2. metadata row keyed by file name (basename or relative path; as_posix so a Windows
        #    backslash path matches the jsonl's forward-slash keys). A sidecar, even empty, wins.
        if not sidecar_present:
            caption = meta_caption.get(img.name) or meta_caption.get(
                img.relative_to(root).as_posix()
            )
        # 3. dreambooth instance prompt for any image still without a caption.
        if not caption and instance_prompt:
            caption = instance_prompt
        if caption:
            if verify_images:
                # Reject a corrupt/zero-byte/truncated image now via a cheap PIL header probe
                # (verify() doesn't decode full pixels): otherwise it passes this filename-only
                # discovery, the start route frees the resident GPU models, and the trainer only
                # then crashes in Image.open -- the eviction this preflight prevents.
                try:
                    from PIL import Image
                    with Image.open(img) as _probe:
                        _probe.verify()
                except Exception as e:  # noqa: BLE001 -- corrupt/zero-byte/truncated file
                    raise ValueError(
                        f"Image cannot be decoded: {img.name} ({e}). Remove or replace the "
                        f"corrupt or zero-byte file before training."
                    ) from e
            pairs.append((str(img), caption))

    if not pairs:
        raise ValueError(
            "No captioned images found. Provide a metadata.jsonl / captions.jsonl, per-image "
            ".txt captions, or an instance prompt."
        )
    return pairs


def _emit(on_event: Optional[EventCb], type_: str, **kw: Any) -> None:
    if on_event is not None:
        on_event({"type": type_, "ts": time.time(), **kw})


def _plan_cache_variants(
    num_images: int, cache_variants: int, center_crop: bool, random_flip: bool, seed: int
) -> list[list[tuple[float, float, bool]]]:
    """Seed-deterministic crop/flip plan for the latent cache: per image, up to
    ``cache_variants`` draws of (u_left, u_top, flip) with the crop as unit fractions the
    loader maps onto its integer crop range. Uses its own rng stream so the training
    loop's draws are untouched. Center-crop / no-flip collapse duplicate variants (a
    center crop without flip is one variant no matter how many draws), so callers encode
    each distinct variant exactly once. Pure (no torch) for CPU unit tests."""
    crop_rng = random.Random(seed)
    plan: list[list[tuple[float, float, bool]]] = []
    for _ in range(max(0, num_images)):
        variants: list[tuple[float, float, bool]] = []
        for _ in range(max(1, cache_variants)):
            u_left, u_top = crop_rng.random(), crop_rng.random()
            flip = bool(random_flip and crop_rng.random() < 0.5)
            if center_crop:
                u_left = u_top = 0.5  # loader ignores the fractions for a center crop
            key = (u_left, u_top, flip)
            if key not in variants:
                variants.append(key)
        plan.append(variants)
    return plan


# Host-memory budget for the AUTOMATIC latent cache. The cache holds two fp32 posterior tensors
# (mean/std, VAE scale folded in) per crop/flip variant per image, pinned on a CUDA host. At
# 1024px an SDXL variant is ~0.5 MiB and a 16-channel DiT variant several times that, so a few
# thousand images x cache_variants can exhaust host or pinned RAM. Over budget the default falls
# back to per-step VAE encoding. A fixed constant (not a psutil RAM fraction) keeps the gate
# dependency-free and identical across hosts; deliberately conservative, well under a typical
# host's RAM.
_LATENT_CACHE_BUDGET_BYTES = 4 * 1024**3  # 4 GiB

# Returned by the cache builders when the estimate exceeds budget: the caller keeps the VAE
# resident and encodes each step's latents in-loop. A distinct sentinel from ``None`` (a stop
# requested mid-build) so the two are not conflated.
LATENT_CACHE_OVER_BUDGET: Any = object()


def _latent_cache_forced() -> bool:
    """The user explicitly forced the latent cache on, bypassing the size gate. This is the
    explicit opt-in counterpart to ``UNSLOTH_DIFFUSION_NO_LATENT_CACHE`` (the explicit
    opt-out); only the automatic default is size-gated, so an explicit choice is honoured
    verbatim in either direction."""
    return os.environ.get("UNSLOTH_DIFFUSION_FORCE_LATENT_CACHE", "") in ("1", "true")


def _latent_cache_over_budget(
    per_variant_bytes: int,
    total_variants: int,
    budget_bytes: Optional[int] = None,
) -> bool:
    """True when a cache of ``total_variants`` entries, each two fp32 tensors totalling
    ``per_variant_bytes``, is estimated to exceed ``budget_bytes``. ``per_variant_bytes`` is
    measured from a real encoded latent, so the estimate tracks the actual per-family tensor
    shape (SDXL 4-channel vs. a packed 16-channel DiT latent) rather than a guess. The budget
    is read from the module constant at call time when not given, so tests can override it."""
    if budget_bytes is None:
        budget_bytes = _LATENT_CACHE_BUDGET_BYTES
    return per_variant_bytes * max(0, total_variants) > budget_bytes


def _apply_perf_flags(
    cfg: "DiffusionLoraConfig",
    device: str,
    cudnn_benchmark: bool = False,
) -> dict:
    """Set the run-scoped torch backend knobs: TF32 matmuls + high fp32 matmul precision
    when ``cfg.enable_tf32`` is on, strict fp32 (all TF32 flags cleared) when it is off,
    plus cudnn autotuning when the caller opts in. Autotune is
    for the conv-heavy SDXL U-Net only: measured on B200, it DOUBLES peak VRAM (fp32 VAE
    conv workspaces) while the DiT loop -- pure matmuls once the latent cache is built --
    gains nothing from it. Returns a snapshot for ``_restore_perf_flags``. Best-effort:
    missing attributes on a CPU/other-vendor build are skipped."""
    from core.inference.diffusion_speed import snapshot_backend_flags

    snap: dict[str, Any] = {"flags": snapshot_backend_flags(), "matmul_precision": None}
    if device != "cuda":
        return snap
    try:
        import torch

        snap["matmul_precision"] = torch.get_float32_matmul_precision()
        if cfg.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        else:
            # The opt-out is a strict-fp32 A/B mode, so actively clear the flags rather
            # than inherit ambient state (cudnn TF32 defaults to ON in torch).
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        if cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        # The cuDNN SDPA backend's TRAINING graph is broken for the FLUX attention shapes on
        # torch 2.10 + cu130 (B200): mha_graph.execute fails, then poisons the context into
        # illegal memory accesses. Flash / mem-efficient SDPA are equivalent, so pin those for the
        # run (restored on exit).
        cuda_backends = getattr(torch.backends, "cuda", None)
        if cuda_backends is not None and hasattr(cuda_backends, "enable_cudnn_sdp"):
            try:
                snap["cudnn_sdp"] = bool(cuda_backends.cudnn_sdp_enabled())
            except Exception:  # noqa: BLE001 -- flag unreadable: skip the tweak entirely
                snap["cudnn_sdp"] = None
            if snap["cudnn_sdp"]:
                cuda_backends.enable_cudnn_sdp(False)
    except Exception:  # noqa: BLE001 -- perf flags are never fatal
        pass
    return snap


def _restore_perf_flags(snap: Optional[dict]) -> None:
    """Undo ``_apply_perf_flags`` (the trainer subprocess is disposable, but in-process
    callers -- tests, notebooks -- must not inherit mutated globals)."""
    if not snap:
        return
    from core.inference.diffusion_speed import restore_backend_flags

    restore_backend_flags(snap.get("flags"))
    try:
        import torch

        if snap.get("matmul_precision"):
            torch.set_float32_matmul_precision(snap["matmul_precision"])
        # Restore the exact pre-run cudnn SDPA state; None means the flag was unreadable
        # (or absent) at apply time and was never touched.
        cuda_backends = getattr(torch.backends, "cuda", None)
        if (
            snap.get("cudnn_sdp") is not None
            and cuda_backends is not None
            and hasattr(cuda_backends, "enable_cudnn_sdp")
        ):
            cuda_backends.enable_cudnn_sdp(bool(snap["cudnn_sdp"]))
    except Exception:  # noqa: BLE001 -- best-effort restore
        pass


# Official safetensors-only TRAINING bases trusted in addition to the inference allowlist
# (_TRUSTED_NON_GGUF_REPOS in core/inference/diffusion.py): the FLUX.2 bases train LoRAs but
# are not (yet) non-GGUF inference bases. Exact-match lowercased, same rules as the loader
# list: extend deliberately; never add pickled weights or remote code.
_TRAIN_EXTRA_TRUSTED_REPOS = frozenset(
    {
        "black-forest-labs/flux.2-dev",
        "black-forest-labs/flux.2-klein-4b",
    }
)


def _assert_trusted_base_model(base_model: str) -> None:
    """Gate the training base model the same way the inference backend gates non-GGUF loads:
    a local path or a trusted repo (``unsloth/*`` or an allowlisted official base). This runs
    BEFORE ``from_pretrained`` so an untrusted remote repo (which could ship pickle weights)
    is never fetched or deserialised."""
    from core.inference.diffusion import _assert_local_base_is_pipeline, _is_trusted_diffusion_repo

    trusted = (
        _is_trusted_diffusion_repo(base_model)
        or str(base_model or "").strip().lower() in _TRAIN_EXTRA_TRUSTED_REPOS
    )
    if not trusted:
        raise ValueError(
            f"Refusing to train from untrusted base model '{base_model}'. Use a local path or "
            f"a trusted repo (an unsloth/* repo or an official base)."
        )
    # An existing LOCAL base is loaded as a full pipeline (from_pretrained(base_model)) by the
    # trainer, which needs a model_index.json. Any existing path is "trusted" above, so reject a
    # non-pipeline local dir here -- before /diffusion/start frees the GPU models -- rather than
    # have the child fail after teardown.
    _assert_local_base_is_pipeline(base_model)


def _publish_to_lora_catalog(lora_path: str, cfg: DiffusionLoraConfig) -> Optional[str]:
    """Best-effort copy of the trained adapter into the Studio diffusion LoRA directory so
    the Images LoRA picker (which scans only files directly under ``loras/diffusion``) finds
    it without the user moving files. Also writes a ``<alias>.json`` metadata sidecar so the
    picker can family-gate the adapter (family, base model, trigger prompt, ...). Returns the
    published path, or None on any failure."""
    try:
        import shutil

        from core.inference.diffusion_lora import loras_dir, sanitize_alias

        base = (
            cfg.adapter_name
            if cfg.adapter_name and cfg.adapter_name != "default"
            else Path(cfg.output_dir).name
        )
        alias = sanitize_alias(base)
        src_resolved = Path(lora_path).resolve()
        dest = loras_dir() / f"{alias}.safetensors"
        # A retrain with the same adapter name must not clobber a prior mirror: pick the next
        # free numeric suffix instead.
        if dest.exists() and dest.resolve() != src_resolved:
            n = 2
            while True:
                candidate = loras_dir() / f"{alias}-{n}.safetensors"
                if not candidate.exists() or candidate.resolve() == src_resolved:
                    dest = candidate
                    break
                n += 1
        if src_resolved != dest.resolve():
            shutil.copy2(lora_path, dest)
        _write_lora_sidecar(dest.with_suffix(".json"), cfg)
        return str(dest)
    except Exception:  # noqa: BLE001 -- the catalog mirror is best-effort, never fatal
        return None


def _write_lora_sidecar(sidecar_path: Path, cfg: DiffusionLoraConfig) -> None:
    """Write the adapter metadata sidecar read back by diffusion_lora._scan_local. Best
    effort: a failure here must not fail publishing, so callers wrap it."""
    meta = {
        "family": cfg.resolved_family,
        "families": [cfg.resolved_family],
        "base_model": cfg.base_model,
        "lora_rank": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "steps": cfg.train_steps,
        "resolution": cfg.resolution,
        "trigger_prompt": cfg.instance_prompt,
        "created_at": time.time(),
        "source": "studio-trained",
    }
    sidecar_path.write_text(json.dumps(meta, indent = 2), encoding = "utf-8")


# Aliases from the generic Studio training payload onto DiffusionLoraConfig fields, so the
# diffusion trainer can also be driven by the shared training request shape (not only its own
# request model, whose keys already match).
_CONFIG_ALIASES = {
    "model_name": "base_model",
    "max_steps": "train_steps",
    # The generic payload's num_epochs already matches the diffusion field name, but list it so
    # the epochs override is threaded through the shared-payload path as explicitly as
    # max_steps -> train_steps is.
    "num_epochs": "num_epochs",
    "batch_size": "train_batch_size",
    "lora_r": "lora_rank",
    "lr_scheduler_type": "lr_scheduler",
    "random_seed": "seed",
    "lr": "learning_rate",
}


def _coerce_gradient_checkpointing(value: Any) -> bool:
    """Studio sends gradient_checkpointing as a string ("none" / "true" / "unsloth"); the
    disable words are False, anything else truthy is True. A real bool passes through."""
    if isinstance(value, str):
        return value.strip().lower() not in ("", "none", "false", "0", "no", "off")
    return bool(value)


def _coerce_bool(value: Any) -> bool:
    """Coerce a flag that may arrive as a string through the generic Studio config path
    (e.g. "false" / "0" / "off"). A non-empty string like "false" is otherwise truthy, so
    an opt-out would silently no-op. A real bool passes through."""
    if isinstance(value, str):
        return value.strip().lower() not in ("", "none", "false", "0", "no", "off")
    return bool(value)


def _config_from_dict(config: dict) -> DiffusionLoraConfig:
    """Build a DiffusionLoraConfig from a plain dict. Unknown keys are ignored so a richer
    request payload (UI form) does not break construction; a small set of generic Studio
    training keys are aliased onto the diffusion field names, and string flags are coerced."""
    valid = DiffusionLoraConfig.__dataclass_fields__.keys()
    kwargs: dict[str, Any] = {}
    # Aliases first (lowest priority); a canonical key present in the payload overrides.
    for src, dst in _CONFIG_ALIASES.items():
        if src in config and config[src] is not None and dst in valid:
            kwargs[dst] = config[src]
    for k, v in config.items():
        if k in valid:
            kwargs[k] = v
    # Epoch-mode payloads from the generic UI carry max_steps: 0 as the "use epochs" sentinel,
    # which the max_steps -> train_steps alias copies as train_steps: 0. Since normalized()
    # rejects train_steps < 1 before resolve_train_steps() applies num_epochs, drop a falsy/0
    # train_steps when num_epochs > 0 so the dataclass default stands in until epoch resolution.
    try:
        _num_epochs = int(kwargs.get("num_epochs") or 0)
    except (TypeError, ValueError):
        _num_epochs = 0
    if _num_epochs > 0 and not kwargs.get("train_steps"):
        kwargs.pop("train_steps", None)
    if kwargs.get("lora_target_modules"):
        kwargs["lora_target_modules"] = tuple(kwargs["lora_target_modules"])
    if "gradient_checkpointing" in kwargs:
        kwargs["gradient_checkpointing"] = _coerce_gradient_checkpointing(
            kwargs["gradient_checkpointing"]
        )
    for flag in ("cache_latents", "enable_tf32"):
        if flag in kwargs:
            kwargs[flag] = _coerce_bool(kwargs[flag])
    return DiffusionLoraConfig(**kwargs)
