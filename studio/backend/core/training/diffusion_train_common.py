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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

from core._torchao_stub import (
    install_torchao_windows_rocm_stub,
    install_xformers_windows_rocm_stub,
    is_stubbed,
    torch_is_rocm,
)
from core.inference.diffusion_families import (
    detect_family,
    detect_family_for_pick,
    supported_family_names,
    trainable_family_names,
)
from core.inference.video_families import detect_video_family, supported_video_family_names
from utils.paths.path_utils import drop_appledouble_metadata

# The trainers run in a spawned child that imports diffusers itself, so the inference-side install
# does not carry over. Both import this module first.
install_xformers_windows_rocm_stub()
install_torchao_windows_rocm_stub()

# Default LoRA target modules: the attention projections common to the SDXL U-Net and the DiTs (the
# diffusers/kohya convention). A family may override this.
DEFAULT_LORA_TARGETS: tuple[str, ...] = ("to_k", "to_q", "to_v", "to_out.0")

# piecewise_constant is excluded: it alone needs a step_rules string the trainers never pass, so
# accepting it would pass normalized(), free the resident GPU workloads, then crash in the child.
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

# DiT families whose fp32 RoPE/embedder overflow fp16, so they train in bf16 only. Keep in sync with
# the DiT trainer's own specs.
_FORCE_BF16_FAMILIES: frozenset[str] = frozenset(
    {"qwen-image", "z-image", "krea-2", "flux.2-klein", "flux.2-dev", "ltx-2", "minimax-h3"}
)

# The video registry has no trainable flag, so the trainable set lives here and every name in it
# MUST resolve through get_trainer; a video base outside it is refused by name.
TRAINABLE_VIDEO_FAMILIES: frozenset[str] = frozenset({"ltx-2", "minimax-h3"})

# Families whose flow_shift default is "auto" (reproduce the family's INFERENCE sigma
# distribution) rather than the identity 1.0. Both schedulers set use_dynamic_shifting, so
# scheduler.sigmas is the unshifted uniform table and training on it draws a distribution the
# model never sees at inference: Qwen-Image pins base_shift = max_shift = log 3 and LTX-2
# evaluates its shift at max_image_seq_len (mu = max_shift = 2.05 at every resolution), so the
# inference mu is a constant "auto" reproduces. MiniMax-H3 instead applies explicit exponential
# shifts (12.0 video, 3.0 audio) whenever flow_shift is not a number, so without this entry a
# default H3 run trained unshifted.
AUTO_FLOW_SHIFT_FAMILIES: frozenset[str] = frozenset({"qwen-image", "ltx-2", "minimax-h3"})

# Video latents are allocated on the family's spatial compression grid, so a training resolution
# off that grid silently changes the latent geometry. LTX-2's VAE compresses 32x spatially,
# matching its resolution_multiple.
_VIDEO_RESOLUTION_MULTIPLE = 32

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_CAPTION_EXTS = (".txt", ".caption")
# diffusers' canonical single-file LoRA name, so load_lora_weights(dir) finds it.
DEFAULT_LORA_FILENAME = "pytorch_lora_weights.safetensors"

# Architectures Unsloth can neither train nor load: not in the family registry but recognisable by
# name, so rejecting them by name gives a clear error instead of a mid-run crash.
_NON_TRAINABLE_RESIDUAL_TOKENS = frozenset({"sd3", "pixart", "sana", "lumina", "cogview"})
_NON_TRAINABLE_RESIDUAL_PHRASES = ("stable-diffusion-3", "hunyuan-dit")

EventCb = Callable[[dict[str, Any]], None]
# Returns a falsy value to keep training, or a truthy stop signal: bare True, or a dict that may
# carry ``save=False`` to cancel without saving a partial adapter.
StopCb = Callable[[], Any]


def _all_trainable_family_names() -> tuple[str, ...]:
    """Every family with a trainer: the image registry's ``trainable`` families plus the
    video families listed in ``TRAINABLE_VIDEO_FAMILIES``. The two registries are separate
    by design (a video checkpoint must never route to an image pipeline), so the trainable
    view has to union them."""
    video = tuple(n for n in supported_video_family_names() if n in TRAINABLE_VIDEO_FAMILIES)
    return tuple(trainable_family_names()) + video


def _trainable_family_spec(name: str) -> Any:
    """The registry entry for a trainable family name, from EITHER registry, or None.

    Every consumer that starts from a resolved family NAME (rather than from a repo id) needs
    this: the image registry does not know a video family, so a plain ``detect_family`` returns
    None for one and the caller silently skips it. That is how ``ltx-2`` stayed out of
    ``/diffusion/info`` and out of the strict pipeline gate."""
    from core.inference.diffusion_families import detect_family

    key = str(name or "").strip().lower()
    fam = detect_family("", override = key)
    if fam is not None:
        return fam
    return detect_video_family("", override = key)


def _refuse_untrainable_video_family(name: str) -> None:
    """Raise for a video family Unsloth has no trainer for.

    Video bases are invisible to the image registry, so before this gate existed every video
    checkpoint fell through ``resolve_trainable_family``'s unknown-name fallback and was
    handed to the SDXL trainer, which then failed somewhere inside from_pretrained. Refuse by
    name instead, with the list of video families that do train."""
    trainable = ", ".join(sorted(TRAINABLE_VIDEO_FAMILIES))
    raise ValueError(
        f"'{name}' is a video model Unsloth can't train yet. Video LoRA training currently "
        f"supports: {trainable}. {_trainable_hint()}"
    )


def _component_only_repos() -> dict[str, tuple[str, str, str]]:
    """Every repo the family registries list only as a COMPONENT source, keyed by lowercased
    repo id -> (family name, component, that family's base repo).

    A pre-cast text-encoder repo (``te_prequant_repos``, present in both the image and the video
    registry) ships a single component archive: no ``model_index.json``, no VAE, no scheduler, no
    pipeline. ``from_pretrained`` on one can only fail. Nothing in the NAME says so, which is the
    whole problem: ``unsloth/LTX-2-FP8`` carries the ``ltx-2`` token, so the family detectors
    claim it and the ``unsloth/*`` trust gate passes it. The registries' own tables are the only
    authority on what a repo actually holds, so read them rather than special-casing repo ids.

    A hosted pre-quantized DENOISER (a video family's ``prequant_repos``) is the same shape: the
    DiT alone, no pipeline around it. The image registry's identically-named table means the
    opposite -- a full quantized pipeline mirror -- so the two are read separately rather than
    together.

    A repo that is ALSO registered as a base somewhere (a full quantized pipeline mirror, a
    deploy base, a train base) is a base and never appears here."""
    from core.inference.diffusion_families import detect_family
    from core.inference.video_families import detect_video_family

    image_families = [detect_family("", override = n) for n in supported_family_names()]
    video_families = [detect_video_family("", override = n) for n in supported_video_family_names()]
    components: dict[str, tuple[str, str, str]] = {}
    bases: set[str] = set()
    for fam, is_video in [(f, False) for f in image_families] + [(f, True) for f in video_families]:
        if fam is None:
            continue
        for attr in ("base_repo", "deploy_base_repo"):
            repo = getattr(fam, attr, None)
            if repo:
                bases.add(str(repo).strip().lower())
        bases.update(str(r).strip().lower() for r in getattr(fam, "train_base_repos", ()) if r)
        # An image family's entry is a full quantized PIPELINE mirror, and so a base; a video family's is a
        # hosted pre-quantized DENOISER, a component.
        for table in ("prequant_repos", "prequant_variant_repos"):
            for row in getattr(fam, table, ()) or ():
                if not row:
                    continue
                repo = str(row[-1]).strip().lower()
                if is_video:
                    components.setdefault(repo, (fam.name, "transformer", str(fam.base_repo)))
                else:
                    bases.add(repo)
        for _scheme, component, repo in getattr(fam, "te_prequant_repos", ()):
            components.setdefault(
                str(repo).strip().lower(), (fam.name, str(component), str(fam.base_repo))
            )
    return {repo: hit for repo, hit in components.items() if repo not in bases}


def _refuse_component_only_repo(base_model: str) -> None:
    """Raise for a base model that is a family's component checkpoint rather than a model.

    Runs from ``resolve_trainable_family``, so it fires in the ``/diffusion/start`` preflight
    BEFORE the resident GPU workloads are freed. Without it the name match resolved a real
    family, the trust gate passed the ``unsloth/*`` repo, the gated-access probe ignored the
    resulting ``model_index.json`` 404 (a 404 is not an access problem), and the run evicted the
    user's loaded model before failing inside ``from_pretrained`` in the child."""
    hit = _component_only_repos().get(str(base_model or "").strip().lower())
    if hit is None:
        return
    family, component, base_repo = hit
    raise ValueError(
        f"'{base_model}' is the pre-cast {component} checkpoint for the '{family}' family, not a "
        f"full model: it ships no model_index.json and no pipeline, so it cannot be a training "
        f"base. Train from '{base_repo}' instead."
    )


def _trainable_hint() -> str:
    """A user-facing hint listing the families Unsloth can train today. Always names SDXL
    explicitly so the message is actionable even as more families become trainable."""
    names = ", ".join(_all_trainable_family_names()) or "sdxl"
    return (
        f"Trainable families right now: {names} "
        f"(for example the SDXL base stabilityai/stable-diffusion-xl-base-1.0). "
        f"Other families can load LoRAs but not train them yet."
    )


def _assert_family_pipeline_available(fam: Any) -> None:
    """Refuse a family whose pipeline class the installed diffusers does not carry.

    ``pyproject`` deliberately leaves the diffusers floor conditional -- diffusers dropped Python
    3.9 in 0.37 and this project still supports 3.9 (``requires-python >= 3.9``), so the pin reads
    ``diffusers>=0.39.0 ; python_version >= '3.10'`` and an unconstrained ``diffusers`` below that.
    A supported install can therefore legitimately predate a family's pipeline class:
    ``Krea2Pipeline`` arrived in 0.39.0 and ``Flux2KleinPipeline`` in 0.37.0, while the newest
    diffusers a 3.9 host can resolve is 0.36.0, and an already-present older one satisfies the
    unconstrained pin outright (``ZImagePipeline`` and ``Flux2Pipeline`` only arrived in 0.36.0).

    The inference paths already assert this before a load (``diffusion.py`` and ``video.py``); the
    training preflight did not, so the family resolved as trainable, ``/diffusion/start`` reserved
    the training slot and freed the resident GPU workloads, and only the spawned child discovered
    the pipeline was missing when it ran its own ``from diffusers import <Pipeline>``. Losing a
    loaded model and THEN failing is the worst ordering available, so assert here, while
    ``resolve_trainable_family`` still runs ahead of every teardown.

    Not strict: an unimportable diffusers is left to ``training_pipeline_import_error`` below.
    ``resolve_trainable_family`` runs from ``normalized()``, which is pure config validation and is
    called in plenty of places that never train, so making it depend on a working diffusers import
    would refuse configs over an unrelated environment problem.

    Family-agnostic on purpose: it reads the probe class off whatever spec it is handed, so the
    image registry and the separate video registry share one gate rather than one each. The class
    comes from ``family_probe_class`` rather than ``fam.pipeline_class`` for the reason that
    helper documents: a modular family's ``pipeline_class`` is the generic ``ModularPipeline``,
    which an older diffusers exports regardless, so probing it accepted a MiniMax-H3 start that
    the listing had already hidden and the child could only fail -- after the teardown."""
    from core.inference.diffusion_families import (
        assert_pipeline_class_available,
        family_probe_class,
    )
    assert_pipeline_class_available(family_probe_class(fam), fam.name)


def training_pipeline_import_error(resolved_family: str) -> Optional[str]:
    """The reason this host cannot import ``resolved_family``'s pipeline class, or None.

    The strict half of the gate above, and it belongs to the ROUTE rather than to config
    validation. ``assert_pipeline_class_available`` deliberately absorbs an unimportable diffusers
    for inference -- the native sd.cpp engine serves GGUF picks on a CPU or Apple host that has
    none. Training has no such fallback: its child is an ``mp.get_context("spawn")`` process in the
    SAME interpreter, so a diffusers that cannot be imported here cannot be imported there either.
    Staying silent bought nothing but the ordering this whole preflight exists to prevent: the slot
    reserved, the resident GPU models freed, and only then the child failing on its own
    ``from diffusers import <Pipeline>``.

    Returns the message instead of raising, matching ``training_precision_preflight_error``, so the
    route maps it to its own 400."""
    from core.inference.diffusion_families import (
        assert_pipeline_class_available,
        family_probe_class,
    )

    # A video family is invisible to detect_family, and returning None would hand the strict half of the
    # gate to the spawned child, after the teardown.
    fam = _trainable_family_spec(resolved_family)
    if fam is None:
        return None
    try:
        assert_pipeline_class_available(family_probe_class(fam), fam.name, strict = True)
    except ValueError as e:
        return str(e)
    return None


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
    - a resolved family whose pipeline class the installed diffusers lacks is rejected
      (``_assert_family_pipeline_available``), so an environment too old for the pick fails
      here rather than in the child, after the GPU residents are gone;
    - an unclassifiable custom name/path falls through to the SDXL trainer (backwards
      compatible: a genuinely wrong pick still fails cleanly later in from_pretrained).
      No pipeline assert on that path: there is no family spec to read a class off, and the
      family it lands on is SDXL, whose pipeline predates every diffusers in play.
    """
    name = str(base_model or "").strip().lower()
    # GGUF weights (a ``.gguf`` file or ``*-GGUF`` repo) are inference-only: training needs the full diffusers pipeline.
    # modular_model_index.json counts: a MODULAR_BASE_FAMILIES checkout has that and no
    # model_index.json, so reading only the conventional name refused the one local form the family has.
    local = Path(base_model).expanduser() if base_model else None
    is_local_diffusers = bool(
        local
        and (
            (local / "model_index.json").is_file() or (local / "modular_model_index.json").is_file()
        )
    )
    if name.endswith(".gguf") or ("gguf" in name and not is_local_diffusers):
        raise ValueError(
            f"'{base_model}' is a GGUF checkpoint/repo, which can't be a training base "
            f"(training needs the full diffusers model). {_trainable_hint()}"
        )
    # A component checkpoint is not a base whatever family the name matches, so this precedes every
    # family branch, including an explicit model_family override.
    _refuse_component_only_repo(base_model)
    # Same shape as the component-only refusal: the name resolves to a real, trainable family, but the repo is
    # not something the training loader can open.
    _refuse_ltx23_training_base(base_model)
    if model_family and str(model_family).strip():
        key = str(model_family).strip().lower()
        fam = detect_family("", override = key)
        if fam is None:
            # Not an image family: it may still name a VIDEO family, which lives in its own registry.
            vid = detect_video_family("", override = key)
            if vid is not None:
                if vid.name not in TRAINABLE_VIDEO_FAMILIES:
                    _refuse_untrainable_video_family(vid.name)
                _assert_family_pipeline_available(vid)
                return vid.name
            known = ", ".join(supported_family_names() + supported_video_family_names())
            raise ValueError(f"Unknown model_family {model_family!r}. Known families: {known}.")
        if not fam.trainable:
            raise ValueError(f"'{fam.name}' models can't be trained yet. {_trainable_hint()}")
        _assert_family_pipeline_available(fam)
        return fam.name

    fam = detect_family_for_pick(base_model)
    if fam is not None:
        if not fam.trainable:
            raise ValueError(
                f"'{base_model}' looks like a {fam.name} model, which isn't trainable yet. "
                f"{_trainable_hint()}"
            )
        _assert_family_pipeline_available(fam)
        return fam.name

    # Checked here rather than first so an image repo the picker already claims keeps its existing route.
    vid = detect_video_family(base_model)
    if vid is not None:
        if vid.name not in TRAINABLE_VIDEO_FAMILIES:
            _refuse_untrainable_video_family(vid.name)
        _assert_family_pipeline_available(vid)
        return vid.name

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
            # ROCm capability values are gfx versions, not the NVIDIA SM levels checked below.
            torchao_ok = has_functional_torchao() and not torch_is_rocm()
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
    if key in ("flux.1", "qwen-image", "z-image", "krea-2", "flux.2-klein", "flux.2-dev", "ltx-2"):
        from core.training.diffusion_dit_trainer import run_dit_lora_training
        return run_dit_lora_training
    # MiniMax-H3 denoises video and audio jointly over one packed sequence on two coupled schedules,
    # which is outside the DiT trainer's _FamilySpec seams.
    if key == "minimax-h3":
        from core.training.diffusion_h3_trainer import run_h3_lora_training
        return run_h3_lora_training
    raise ValueError(f"No trainer is registered for family {family!r}.")


# Per-family training defaults surfaced by the Train UI: starting points, not hard limits. Families
# absent here fall back to the DiffusionLoraConfig defaults.
FAMILY_TRAIN_DEFAULTS: dict[str, dict[str, Any]] = {
    "sdxl": {"lora_rank": 16, "learning_rate": 1e-4, "resolution": 1024},
    # Plain "constant" ignores lr_warmup_steps, so warmup defaults must use a warmup-capable scheduler.
    "flux.1": {
        "lora_rank": 16,
        "learning_rate": 1e-4,
        "resolution": 512,
        "lr_scheduler": "constant_with_warmup",
        "lr_warmup_steps": 20,
    },
    "qwen-image": {
        "lora_rank": 16,
        "learning_rate": 5e-5,
        "resolution": 512,
        "lr_scheduler": "constant_with_warmup",
        "lr_warmup_steps": 20,
    },
    "z-image": {"lora_rank": 16, "learning_rate": 1e-4, "resolution": 768},
    # The Krea 2 authors' recommended starting point (their DreamBooth defaults): rank/alpha 32, lr 3e-4, 512px.
    "krea-2": {"lora_rank": 32, "learning_rate": 3e-4, "resolution": 512},
    # Upstream FLUX.2 DreamBooth references default to rank 16 / lr 1e-4; its uniform timestep draw
    # benefits most from a warmup ramp.
    "flux.2-klein": {
        "lora_rank": 16,
        "learning_rate": 1e-4,
        "resolution": 512,
        "lr_scheduler": "constant_with_warmup",
        "lr_warmup_steps": 20,
    },
    "flux.2-dev": {
        "lora_rank": 16,
        "learning_rate": 1e-4,
        "resolution": 512,
        "lr_scheduler": "constant_with_warmup",
        "lr_warmup_steps": 20,
    },
    # From Lightricks' own ltx-trainer LoRA configs; the resolution must be a multiple of 32, its
    # VAE's spatial compression, and 512 keeps a still at 16x16x1 latents / 256 video tokens.
    "ltx-2": {
        "lora_rank": 32,
        "learning_rate": 1e-4,
        "resolution": 512,
        "lr_scheduler": "constant_with_warmup",
        "lr_warmup_steps": 20,
    },
    # resolution is the canvas SHORT EDGE and 768 is what the released checkpoint generates on;
    # rank 16 not 32 because the adapter also serves the audio rows through one shared stack.
    "minimax-h3": {
        "lora_rank": 16,
        "learning_rate": 1e-4,
        "resolution": 768,
        "lr_scheduler": "constant_with_warmup",
        "lr_warmup_steps": 20,
        "train_batch_size": 1,
    },
}


def train_defaults(family: str) -> dict[str, Any]:
    """Recommended starting hyperparameters for ``family`` (empty if unknown)."""
    return dict(FAMILY_TRAIN_DEFAULTS.get((family or "").strip().lower(), {}))


# Display labels + a short VRAM/access note per trainable family, surfaced by the Train UI so users pick a base
# with realistic expectations.
_FAMILY_LABELS = {
    "sdxl": "SDXL",
    "flux.1": "FLUX.1-dev",
    "qwen-image": "Qwen-Image",
    "z-image": "Z-Image",
    "krea-2": "Krea 2",
    "flux.2-klein": "FLUX.2 Klein",
    "flux.2-dev": "FLUX.2-dev",
    "ltx-2": "LTX-2",
    "minimax-h3": "MiniMax-H3",
}
# params is the transformer size (SDXL is not quoted that way); note is the rest.
_FAMILY_TRAIN_SPECS: dict[str, dict[str, Any]] = {
    "sdxl": {"params": "", "qlora_vram_gb": 12, "gated": False, "note": "The lightest option."},
    "flux.1": {"params": "12B", "qlora_vram_gb": 16, "gated": True, "note": ""},
    "qwen-image": {
        "params": "20B",
        "qlora_vram_gb": 24,
        "gated": False,
        "note": "The heaviest option.",
    },
    "z-image": {"params": "6B", "qlora_vram_gb": 12, "gated": False, "note": ""},
    "krea-2": {
        "params": "12B",
        "qlora_vram_gb": 18,
        "gated": False,
        "note": "Trains on Krea-2-Raw, runs on Turbo.",
    },
    "flux.2-klein": {"params": "4B", "qlora_vram_gb": 10, "gated": False, "note": ""},
    "flux.2-dev": {"params": "32B", "qlora_vram_gb": 28, "gated": True, "note": ""},
    # Measured on a B200: the training LOOP peaks at 11.2 GB, but the RUN peaks at 34.8 GB while the
    # Gemma3-12B conditioning stack is resident, and the quoted figure covers the whole run.
    "ltx-2": {
        "params": "19B",
        "qlora_vram_gb": 36,
        "gated": False,
        "note": "Video: trains a style LoRA on still images.",
    },
    # Measured on a B200: the loop peaks near 44 GB, but a 20-step run peaked at 77.76 GB with the
    # 63 GiB Qwen3-VL conditioner resident, so 72 sizes users onto a card that OOMs later.
    "minimax-h3": {
        "params": "31B",
        "qlora_vram_gb": 80,
        "gated": False,
        "note": "Video with sound: trains on clips that have a soundtrack.",
    },
}
# Keys are canonical upstream ids; family_train_infos also publishes the mirror aliases, and values
# overlay the family facts in the client.
_BASE_TRAIN_SPECS: dict[str, dict[str, Any]] = {
    "black-forest-labs/flux.2-klein-base-9b": {
        "params": "9B",
        # The bf16 text encoder alone measures 16.4 GB, so leave room for the VAE and runtime state rather
        # than inheriting the 4B checkpoint's floor.
        "qlora_vram_gb": 18,
    },
}
_GATED_NOTE = "Gated: needs its license and your HF token."


def _family_vram_note(name: str) -> str:
    """The one-line note, rebuilt from the spec for clients that predate the chip fields."""
    spec = _FAMILY_TRAIN_SPECS.get(name)
    if not spec:
        return ""
    gb = spec["qlora_vram_gb"]
    head = f"{spec['params']}, QLoRA by default (~{gb} GB+)." if spec["params"] else f"~{gb} GB+."
    tail = [t for t in (_GATED_NOTE if spec["gated"] else "", spec["note"]) if t]
    return " ".join([head, *tail])


# The flow-matching DiT families (run by diffusion_dit_trainer): they expose base_precision /
# compile and need bf16 on CUDA. SDXL is absent (own mixed_precision path).
_DIT_TRAIN_FAMILIES = frozenset(
    {"flux.1", "qwen-image", "z-image", "krea-2", "flux.2-klein", "flux.2-dev", "ltx-2"}
)
# Kept separate so the DiT-specific levers (compile, the shared sigma table) do not follow.
_FLOW_TRAIN_FAMILIES = _DIT_TRAIN_FAMILIES | {"minimax-h3"}


def effective_mixed_precision(cfg: Any) -> str:
    """The precision the SDXL trainer will actually run in, resolved the same way it resolves it.

    A pre-Ampere card has no native bf16, so a bf16 request silently becomes fp16 there. Recording
    the REQUEST in a checkpoint's identity let a bundle written in fp16 resume in bf16 on a newer
    card (and the reverse), continuing restored optimizer moments under different frozen-base
    numerics while reporting a clean resume. Shared so the start route and the trainer cannot
    disagree about what a checkpoint was trained as.
    """
    import torch  # noqa: PLC0415 -- keep the import list light for the training subprocess

    requested = str(getattr(cfg, "mixed_precision", "") or "")
    if str(getattr(cfg, "resolved_family", "") or "").strip().lower() in _FLOW_TRAIN_FAMILIES:
        # No flow-matching trainer reads mixed_precision (weight_dtype is bf16 on CUDA, fp32 otherwise),
        # so recording the REQUEST failed a later bf16 resume as a precision mismatch between identical
        # runs. Keyed on _FLOW_TRAIN_FAMILIES so the answer follows the weight dtype.
        return "bf16" if torch.cuda.is_available() else "no"
    if not torch.cuda.is_available():
        return "no"
    if requested == "bf16" and not native_bf16_supported():
        return "fp16"
    return requested


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
    import/unit tests) and SDXL (its own mixed_precision path) are exempt. Never raises.

    Covers every FLOW-matching trainer, not just the DiT one: MiniMax-H3 has the same bf16
    requirement (its checkpoint keeps the patch projections and output heads in fp32) and the
    same eviction ordering to protect."""
    if (resolved_family or "").strip().lower() not in _FLOW_TRAIN_FAMILIES:
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


def dit_accelerator_missing_reason(resolved_family: str) -> Optional[str]:
    """Reason a DiT family cannot train on this host at all, else None. Never raises.

    nf4 is not a CPU fallback: the 4-bit base load goes through diffusers' bitsandbytes
    quantizer, whose validate_environment raises "No GPU found. A GPU is needed for
    quantization." unless CUDA, XPU or MPS is present. Without this gate a GPU-less host
    accepts the default nf4 start, evicts the resident Images pipeline, downloads the text
    encoders, and only then dies in the child. SDXL keeps its own fp32-on-CPU path.

    Covers MiniMax-H3 too, for the same reason: it loads its denoiser through the same 4-bit
    quantizer.
    """
    if (resolved_family or "").strip().lower() not in _FLOW_TRAIN_FAMILIES:
        return None
    try:
        import torch
        def probe(owner: Any) -> bool:
            # Each accelerator is probed on its own: torch.mps.is_available() only exists from torch 2.5 while
            # the floor is 2.4, so a shared try/except would wave a CPU-only host through.
            try:
                fn = getattr(owner, "is_available", None)
                return bool(fn()) if callable(fn) else False
            except Exception:  # noqa: BLE001
                return False

        if (
            probe(torch.cuda)
            # torch.backends.mps has carried is_available() since torch 1.12.
            or probe(getattr(torch, "xpu", None))
            or probe(getattr(torch.backends, "mps", None))
        ):
            return None
    except Exception:  # noqa: BLE001 -- probe failure must not block a start
        return None
    return (
        "Training the DiT families needs a GPU: even the 4-bit (nf4) base load requires "
        "CUDA, XPU or MPS, and this host has none. Train SDXL here, or use a GPU machine."
    )


def training_precision_preflight_error(resolved_family: str, base_precision: str) -> Optional[str]:
    """Reason the requested DiT precision cannot run on this host, else None -- checked by the
    start route BEFORE evicting resident GPU workloads (the trainer's own checks fire only in the
    child, after eviction). Every gate mirrors one in _resolve_base_precision, so a doomed run is
    rejected before teardown: the bf16-GPU requirement (bf16_unsupported_reason); no accelerator at
    all (dit_accelerator_missing_reason, which covers nf4 too); the dense precisions
    (bf16/int8/fp8/mxfp8) needing CUDA; the torchao precisions (int8/fp8/mxfp8) against a ROCm
    build, which clears the stub test and hands the sm100 floor an AMD gfx version; explicit int8
    needing a FUNCTIONAL torchao (its _int8_quantize_base has no fallback); explicit fp8/mxfp8
    against the Windows-ROCm torchao stub; explicit mxfp8 needing Blackwell (sm100+). Add a gate
    there, add it here. Never raises."""
    reason = bf16_unsupported_reason(resolved_family)
    if reason:
        return reason
    # No accelerator at all: every DiT precision is out, nf4 included.
    reason = dit_accelerator_missing_reason(resolved_family)
    if reason:
        return reason
    fam = (resolved_family or "").strip().lower()
    mode = (base_precision or "").strip().lower()
    if fam in _FLOW_TRAIN_FAMILIES and mode in ("bf16", "int8", "fp8", "mxfp8"):
        # The DiT trainer's dense precisions all require CUDA, and bf16_unsupported_reason exempts a CPU-
        # only host, so without this a dense request would evict residents then raise in the child.
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
        # Reject before eviction; _resolve_base_precision repeats this in the child.
        if mode in ("int8", "fp8", "mxfp8") and torch_is_rocm():
            return (
                f"base_precision={mode!r} is a torchao NVIDIA tensor-core path (int8 sm_80+, fp8 "
                "sm_89+, mxfp8 sm_100+) and this is a ROCm/AMD GPU. Use 'nf4', 'bf16', or 'auto'."
            )
        if mode == "int8" and not has_functional_torchao():
            return (
                "base_precision='int8' needs a functional torchao install; this host's torchao is "
                "missing or the non-functional Windows-ROCm stub. Use 'nf4', 'bf16', or 'auto'."
            )
        # Mirrors the child's fp8/mxfp8 stub guard so a doomed run is refused before eviction. Not
        # has_functional_torchao(): that probes int8's symbols.
        if mode in ("fp8", "mxfp8") and is_stubbed("torchao"):
            return (
                f"base_precision={mode!r} is not available on this host: torchao is the "
                "non-functional Windows-ROCm stub. Use 'nf4', 'bf16', or 'auto'."
            )
        # mxfp8 needs Blackwell (sm100+): its MX GEMM raises at the first training step, AFTER a full dense
        # load. Re-check here so a stale client fails fast before eviction.
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
    the family registry so it stays in sync with what the trainers actually support.

    Both registries: a video family with a trainer is as trainable as an image one, and reading
    only the image registry is what kept ``ltx-2`` out of the Train tab entirely -- the trainer,
    the preflight and the start route all accepted it, but nothing ever offered it. A family whose
    pipeline class the installed diffusers lacks is dropped rather than advertised, since the start
    route refuses it (``training_pipeline_import_error``) and no choice in the UI can fix that."""
    from core.inference.diffusion_families import family_pipeline_available, mirror_repo
    from core.inference.diffusion_transformer_quant import _family_train_denied

    dit_modes, dit_recommended = train_precision_modes()
    infos: list[dict[str, Any]] = []
    for name in _all_trainable_family_names():
        fam = _trainable_family_spec(name)
        if fam is None or not family_pipeline_available(fam):
            continue
        # A video family carries no train_base_repos/deploy_base_repo: its own base repo is the one training base.
        repos = list(getattr(fam, "train_base_repos", ()) or ()) or [fam.base_repo]
        # base_precision applies to every flow-matching trainer, not only the shared DiT one: reading
        # _DIT_TRAIN_FAMILIES here reported precision_modes = [] for H3, which the Train panel shows as
        # "Not supported on this GPU".
        is_dit = name in _FLOW_TRAIN_FAMILIES
        # On a non-bf16 CUDA GPU the start preflight rejects EVERY DiT family, so advertise no precision
        # rather than an option that always 400s; also drop schemes the TRAINING bar holds back.
        dit_block = (
            bf16_unsupported_reason(name) or dit_accelerator_missing_reason(name)
            if is_dit
            else None
        )
        # H3 refuses compile_transformer="on" (its packed sequence changes length with every caption, so
        # each step would re-trace), so advertising the control would offer a selection that always 400s.
        supports_compile = bool(not dit_block) and name in _DIT_TRAIN_FAMILIES
        if not is_dit or dit_block:
            fam_modes: list[str] = []
        else:
            fam_modes = [m for m in dit_modes if not _family_train_denied(name, m)]
        spec = _FAMILY_TRAIN_SPECS.get(name, {})
        deploy_bases: dict[str, str] = {}
        base_specs: dict[str, dict[str, Any]] = {}
        for repo in repos:
            base_spec = _BASE_TRAIN_SPECS.get(str(repo).strip().lower())
            if not base_spec:
                continue
            base_specs[repo] = dict(base_spec)
            repo_mirror = mirror_repo(repo)
            if repo_mirror:
                base_specs[repo_mirror] = dict(base_spec)
        for training_repo, inference_repo in getattr(fam, "deploy_base_repos", ()):
            deploy_bases[training_repo] = inference_repo
            # A custom base entered with the public mirror id must follow the same pairing as the advertised
            # vendor id; return the inference mirror too so Deploy stays ungated.
            training_mirror = mirror_repo(training_repo)
            if training_mirror:
                deploy_bases[training_mirror] = mirror_repo(inference_repo) or inference_repo
        infos.append(
            {
                "name": name,
                "label": _FAMILY_LABELS.get(name, name),
                "default_base": repos[0],
                "base_repos": repos,
                "defaults": train_defaults(name),
                "vram_note": dit_block or _family_vram_note(name),
                # Dropped on a dit_block, since vram_note then carries the reason.
                "params": "" if dit_block else spec.get("params", ""),
                "qlora_vram_gb": None if dit_block else spec.get("qlora_vram_gb"),
                "gated": False if dit_block else bool(spec.get("gated", False)),
                "note": "" if dit_block else spec.get("note", ""),
                "precision_modes": fam_modes,
                "recommended_precision": "nf4" if (not is_dit or dit_block) else dit_recommended,
                # compile is offered for SDXL's regional U-Net and the shared DiT trainer, except a family the
                # GPU cannot train in bf16, and except a trainer that cannot compile.
                "supports_compile": supports_compile or name == "sdxl",
                # save_steps is REFUSED for a checkpointless family, not ignored, so a panel that keeps offering
                # "Checkpoint every" turns a nonzero value into a rejected Start with no way to see why.
                "supports_checkpoints": name not in CHECKPOINTLESS_FAMILIES,
                # A batch > 1 is REFUSED for a family whose forward covers one packed sequence, so leaving the
                # control unrestricted turns a reasonable 2 into a rejected Start with nothing to say why.
                "max_train_batch_size": 1 if name in SINGLE_SEQUENCE_FAMILIES else None,
                # Krea trains on Raw but previews adapters on Turbo; None elsewhere (and never for a video family).
                "deploy_base": getattr(fam, "deploy_base_repo", None),
                # Families with several train/deploy pairs cannot use the scalar above.
                "deploy_bases": deploy_bases,
                # Dropped on a dit_block for the same reason as the family chips: the overlay wins and
                # FamilyFacts renders vram_note only when there are no chips, so keeping these would replace the
                # actionable hardware reason with a size the host cannot act on.
                "base_specs": {} if dit_block else base_specs,
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
    # Dreambooth-style caption applied to any image without its own. Required if the dataset has no
    # captions.jsonl / sidecars.
    instance_prompt: Optional[str] = None
    resolution: int = 1024
    train_steps: int = 500
    # 0 = disabled (train for train_steps). Above 0 it overrides train_steps with num_epochs full passes
    # LoRA EMA decay (DiT trainer only). 0.0 disables; a positive value keeps a warmup-ramped EMA of
    # the trainable params and exports it under ema/.
    num_epochs: int = 0
    learning_rate: float = 1e-4
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    lora_rank: int = 16
    lora_alpha: Optional[int] = None
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = DEFAULT_LORA_TARGETS
    seed: int = 42
    mixed_precision: str = "bf16"
    snr_gamma: Optional[float] = 5.0
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    center_crop: bool = False
    random_flip: bool = True
    caption_column: str = "text"
    adapter_name: str = "default"
    hf_token: Optional[str] = None
    # Derived by normalized(): the byte-identical mirror used by from_pretrained, while base_model stays
    # the canonical id in metadata and resume identity.
    fetch_base_model: Optional[str] = None
    # cache_variants crop/flip draws are frozen per image; the per-step VAE sampling noise is preserved.
    cache_latents: bool = True
    cache_variants: int = 4
    # A directory persists latent stats and caption embeddings keyed by content hash, so a warm cache
    # skips the VAE and text encoders.
    cond_cache_dir: Optional[str] = None
    # LoRA EMA decay (DiT trainer only). 0.0 disables; a positive value keeps a warmup-ramped EMA of
    # the trainable params and exports it under ema/.
    ema_decay: float = 0.0
    # Regional torch.compile of the transformer blocks: "off" | "on" | "auto" (auto turns it on only for
    # a dense, non-bitsandbytes base where it is a clean win).
    compile_transformer: str = "auto"
    # TF32 matmuls + cudnn autotuning for the run. Near-lossless; disable for strict bit-reproducibility A/Bs.
    enable_tf32: bool = True
    # DiT base transformer precision: "nf4" (bitsandbytes QLoRA, the memory floor and default), "bf16" (dense, fastest
    base_precision: str = "nf4"
    # None resolves per family in normalized(); a number applies s*u/(1+(s-1)*u), and "auto" without
    # dynamic shifting falls back to identity.
    flow_shift: Optional[Any] = None
    # Per-sample probability of replacing the caption with the empty prompt (classifier-free-guidance dropout).
    # 0.0 disables.
    cfg_dropout: float = 0.0
    # Per-sample loss weighting over the drawn timestep: "none" (unweighted MSE) or "bell" (Gaussian, normalized
    # to mean 1).
    weighting_scheme: str = "none"
    # How often to emit a progress event (in optimizer steps).
    log_every: int = 1
    # 0 writes no periodic checkpoints; a stop-and-save always writes one regardless, so the Resume
    # action stays available.
    save_steps: int = 0
    # How many checkpoint-<N> bundles to keep in the output dir; 0 keeps every one.
    save_total_limit: int = 2
    # The run's output_dir, or one of its checkpoint-<N> directories; the start route resolves and
    # validates it before the trainer spawns.
    resume_from_checkpoint: Optional[str] = None
    # Optional explicit family override; None = detect from base_model. ``resolved_family`` is filled by
    # normalized() with the trainer family that will run.
    model_family: Optional[str] = None
    resolved_family: str = "sdxl"

    def normalized(self) -> "DiffusionLoraConfig":
        """Return a copy with derived/validated fields filled in. Raises ValueError on a
        request that cannot train (bad numbers, or an untrainable base model).

        Also coerces values that arrive as strings/blanks through the Unsloth config path
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
        # A video family's VAE compresses space by 32, so an off-grid resolution changes the latent geometry
        # silently. Refuse it before the GPU models are evicted.
        if (
            resolved_family in TRAINABLE_VIDEO_FAMILIES
            and self.resolution % _VIDEO_RESOLUTION_MULTIPLE != 0
        ):
            raise ValueError(
                f"'{resolved_family}' trains at a resolution that is a multiple of "
                f"{_VIDEO_RESOLUTION_MULTIPLE} (its VAE compresses space by that factor); "
                f"got {self.resolution}."
            )
        if self.mixed_precision not in ("bf16", "fp16", "no"):
            raise ValueError("mixed_precision must be one of bf16 / fp16 / no")
        # torch.manual_seed unpacks int64/uint64, so anything wider raises inside the trainer, after
        # eviction. Catch it here.
        if not -(2**63) <= int(self.seed) <= 2**64 - 1:
            raise ValueError("seed must fit in torch's 64-bit range")
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
        # Do not rewrite the scheduler: it is part of checkpoint identity, so legacy runs with ("constant",
        # warmup > 0) would become unresumable.
        try:
            lr_warmup_steps = int(self.lr_warmup_steps or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"lr_warmup_steps must be a whole number, got {self.lr_warmup_steps!r}"
            ) from exc
        if lr_warmup_steps < 0:
            raise ValueError("lr_warmup_steps must be >= 0")
        if not 1 <= int(self.cache_variants) <= 16:
            raise ValueError("cache_variants must be between 1 and 16")
        try:
            save_steps = int(self.save_steps or 0)
            save_total_limit = int(self.save_total_limit or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"save_steps / save_total_limit must be whole numbers, got "
                f"{self.save_steps!r} / {self.save_total_limit!r}"
            ) from exc
        if save_steps < 0:
            raise ValueError("save_steps must be >= 0 (0 disables periodic checkpoints)")
        if save_total_limit < 0:
            raise ValueError("save_total_limit must be >= 0 (0 keeps every checkpoint)")
        # A blank resume path (the Unsloth default when the field is present but unset) means "fresh run",
        # not the outputs root.
        resume_from_checkpoint = (
            str(self.resume_from_checkpoint).strip()
            if self.resume_from_checkpoint is not None
            else ""
        ) or None
        # The H3 loop neither writes a resume bundle nor restores one, and accepting these silently gave a
        # resume request a FRESH optimization that then overwrote the outputs it was meant to continue.
        if resolved_family in CHECKPOINTLESS_FAMILIES:
            if resume_from_checkpoint:
                raise ValueError(
                    f"resume_from_checkpoint is not supported for {resolved_family}: its trainer "
                    f"writes no checkpoint bundle, so there is nothing to continue from and the "
                    f"run would silently start over and overwrite its output. Start a fresh run."
                )
            if save_steps:
                raise ValueError(
                    f"save_steps is not supported for {resolved_family}: its trainer writes no "
                    f"checkpoint bundle. Leave it at 0; the adapter is still saved at the end."
                )
        try:
            ema_decay = float(self.ema_decay or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ema_decay must be a number, got {self.ema_decay!r}") from exc
        # decay = 1.0 would freeze the shadow at its init forever; the update is shadow * decay + param * (1
        # - decay), so valid decays live in [0, 1).
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1); 0 disables the EMA adapter")
        # A blank cond_cache_dir (the Unsloth default when unset) means "off", not cwd.
        cond_cache_dir = (
            str(self.cond_cache_dir).strip() if self.cond_cache_dir is not None else ""
        ) or None
        compile_transformer = str(self.compile_transformer or "auto").strip().lower()
        if compile_transformer not in ("off", "on", "auto"):
            raise ValueError("compile_transformer must be one of off / on / auto")
        base_precision = str(self.base_precision or "nf4").strip().lower()
        if base_precision not in ("nf4", "bf16", "int8", "fp8", "mxfp8", "auto"):
            raise ValueError("base_precision must be one of nf4 / bf16 / int8 / fp8 / mxfp8 / auto")
        # base_precision is a DiT-only lever, so the dense-mode gates apply only to the DiT families. The
        # mode-name check above still runs for every family.
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
            # qwen-image fp8 renders inside the accuracy gate, but no one has measured whether a LoRA converges
            # against fp8-frozen linears, so training fails fast rather than training on faith.
            # MiniMax-H3 runs all three modalities through one set of linears, so the per-family activation
            # range the fp8 module filter was measured against does not describe it.
            if resolved_family == "minimax-h3" and base_precision in ("fp8", "mxfp8"):
                raise ValueError(
                    f"base_precision={base_precision!r} is not supported for minimax-h3: its "
                    f"packed sequence mixes video, audio and text through one set of linears, "
                    f"so the activation range fp8 was measured against does not apply. Use "
                    f"'nf4', 'int8', 'bf16', or 'auto'."
                )
            # _family_train_denied is the strict superset of _family_denied, so importing the narrower helper
            # here would let a scheme cleared only for rendering reach a trainer.
            from core.inference.diffusion_transformer_quant import _family_train_denied

            if _family_train_denied(resolved_family, base_precision):
                raise ValueError(
                    f"base_precision={base_precision!r} is not validated for training "
                    f"{resolved_family}. Use 'nf4', 'int8', 'bf16', or 'auto'."
                )
        # flow_shift: None resolves to the family default ("auto" only for qwen-image, whose scheduler skips its
        # static shift under use_dynamic_shifting); an explicit value is validated and kept.
        flow_shift = self.flow_shift
        if flow_shift is None:
            flow_shift = "auto" if resolved_family in AUTO_FLOW_SHIFT_FAMILIES else 1.0
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
            # isfinite as well as positive: JSON accepts 1e309, which floats to inf and would poison every
            # sampled sigma while progress looks normal.
            if not math.isfinite(flow_shift) or flow_shift <= 0:
                raise ValueError(
                    "flow_shift must be a finite number > 0 (1.0 disables the shift), or 'auto'"
                )
        try:
            cfg_dropout = float(self.cfg_dropout or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"cfg_dropout must be a number, got {self.cfg_dropout!r}") from exc
        if not 0.0 <= cfg_dropout <= 1.0:
            raise ValueError("cfg_dropout must be between 0 and 1")
        weighting_scheme = str(self.weighting_scheme or "none").strip().lower()
        if weighting_scheme not in ("none", "bell"):
            raise ValueError("weighting_scheme must be one of none / bell")
        # A zero/negative gamma would zero out (or invert) the min-SNR weight and silently train on a
        # degenerate loss; None is the documented disable.
        if self.snr_gamma is not None and float(self.snr_gamma) <= 0:
            raise ValueError("snr_gamma must be > 0, or null to disable min-SNR weighting")
        # learning_rate can arrive as a string ("1e-4") from the Unsloth config path, so coerce it before AdamW sees it.
        try:
            learning_rate = float(self.learning_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"learning_rate must be a number, got {self.learning_rate!r}") from exc
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        alpha = self.lora_alpha if self.lora_alpha is not None else self.lora_rank
        targets = tuple(self.lora_target_modules) or DEFAULT_LORA_TARGETS
        # A blank Hub token (the Unsloth default when none is configured) must load anonymously, not as an
        # explicit empty credential.
        token = self.hf_token.strip() if isinstance(self.hf_token, str) else self.hf_token
        from core.inference.diffusion_families import (
            _is_local_path,
            mirror_repo,
            prefer_ungated_mirror,
            upstream_is_gated,
        )

        if resolved_family == "sdxl":
            fetch_base_model = self.base_model
        else:
            fetch_base_model = prefer_ungated_mirror(self.base_model, token or None)
            # For a GATED upstream with no token, override the cache preference: prefer_ungated_mirror's
            # probe counts any cached weight as a hit, so one leftover shard kept the vendor id.
            # Only gated: the rest of the mirror table is reachable anonymously, and an override there would
            # discard a complete local cache and re-pull gigabytes, or fail offline.
            # UNSLOTH_DIFFUSION_NO_MIRROR still wins, exactly as it does inside prefer_ungated_mirror.
            # A local clone wins over both: a base can be a directory named exactly like the vendor id, and
            # rewriting it sends the fetch to the Hub past weights the user already has.
            if (
                not token
                and upstream_is_gated(self.base_model)
                and not _is_local_path(self.base_model)
                and not os.environ.get("UNSLOTH_DIFFUSION_NO_MIRROR", "").strip()
            ):
                fetch_base_model = mirror_repo(self.base_model) or fetch_base_model
        # A blank caption_column means the default, as the start route's preflight assumes: otherwise
        # route and trainer resolve different captions from a metadata.jsonl and their fingerprints
        # disagree, so an accepted resume is refused in the child.
        caption_column = str(self.caption_column or "").strip() or "text"
        return replace(
            self,
            learning_rate = learning_rate,
            lr_warmup_steps = lr_warmup_steps,
            lora_alpha = alpha,
            lora_target_modules = targets,
            max_grad_norm = float(self.max_grad_norm),
            hf_token = token or None,
            fetch_base_model = fetch_base_model,
            caption_column = caption_column,
            num_epochs = int(self.num_epochs),
            cache_variants = int(self.cache_variants),
            save_steps = save_steps,
            save_total_limit = save_total_limit,
            resume_from_checkpoint = resume_from_checkpoint,
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
        # k may exceed n (batch larger than the dataset): the permutation refills across cycles so the
        # caller always gets exactly k indices.
        out: list[int] = []
        while len(out) < k:
            if self._pos >= len(self._order):
                self._reshuffle()
            take = min(k - len(out), len(self._order) - self._pos)
            out.extend(self._order[self._pos : self._pos + take])
            self._pos += take
        return out

    def state_dict(self) -> dict[str, Any]:
        """The position inside the current permutation cycle, for a resume checkpoint. The
        order itself is stored (not just the position): it was drawn from the run's rng, and
        restoring the rng alone would not reproduce a cycle that was already in progress."""
        return {"n": self._n, "order": list(self._order), "pos": int(self._pos)}

    def load_state_dict(self, state: Optional[dict[str, Any]]) -> bool:
        """Restore a cycle saved by ``state_dict``. True when it was restored.

        Refuses a state for a different dataset size (the resume preflight already rejects a
        changed dataset; this keeps a manually edited checkpoint from indexing out of range)
        and clamps a bad position. The BOOLEAN matters: silently leaving a fresh sampler in
        place looks like a clean resume while the RNG -- already restored to a point after this
        permutation was drawn -- generates a different order, so the run quietly reorders and
        skips images.
        """
        if not isinstance(state, dict) or int(state.get("n") or 0) != self._n:
            return False
        order = state.get("order")
        if not isinstance(order, (list, tuple)):
            return False
        try:
            restored = [int(i) for i in order]
        except (TypeError, ValueError):
            return False
        # A shortened or duplicate-carrying order is in range and plausible, so the cycle reshuffles
        # early or serves the same image twice with the RNG already past the original draw.
        if restored and sorted(restored) != list(range(self._n)):
            return False
        self._order = restored
        try:
            pos = int(state.get("pos") or 0)
        except (TypeError, ValueError):
            return False
        # A position outside 0..len(order) is a damaged manifest, and clamping it re-serves the permutation
        # from the top or ends the cycle early behind an already-restored RNG.
        if not 0 <= pos <= len(self._order):
            return False
        self._pos = pos
        return True


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

    # The caption lookup resolves to the caption's own companion, which exists, so the pair reads as a real one.
    images = drop_appledouble_metadata(
        sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
    )

    # 1. metadata.jsonl / captions.jsonl (either name accepted).
    meta_caption: dict[str, str] = {}
    for meta_name in ("metadata.jsonl", "captions.jsonl"):
        meta_path = root / meta_name
        if not meta_path.is_file():
            continue
        # Tolerate a bad upload (invalid UTF-8, or non-object JSON): skip the record so the instance_prompt
        # fallback still applies.
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
            value = row.get(caption_column)
            # A JSON null is "no caption", not the string "None".
            if key and value is not None:
                meta_caption[str(key)] = str(value)

    pairs: list[tuple[str, str]] = []
    for img in images:
        caption: Optional[str] = None
        sidecar_present = False
        # An EMPTY sidecar is a deliberate tombstone: it suppresses the metadata caption and leaves the
        # image uncaptioned.
        for ext in _CAPTION_EXTS:
            sidecar = img.with_suffix(ext)
            if sidecar.is_file():
                sidecar_present = True
                try:
                    caption = sidecar.read_text(encoding = "utf-8").strip()
                except (OSError, UnicodeError):
                    # An unreadable sidecar reads as the empty tombstone, so the instance_prompt fallback applies
                    # instead of a 500.
                    caption = ""
                break
        # 2. metadata row keyed by file name (basename or relative path, as_posix so Windows paths
        # match). A sidecar, even empty, wins.
        if not sidecar_present:
            caption = meta_caption.get(img.name) or meta_caption.get(
                img.relative_to(root).as_posix()
            )
        # 3. dreambooth instance prompt for any image still without a caption.
        if not caption and instance_prompt:
            caption = instance_prompt
        if caption:
            if verify_images:
                # Reject a corrupt or truncated image now: otherwise it passes filename-only discovery, the start
                # route frees the GPU models, and the trainer crashes in Image.open.
                # Epoch-mode payloads carry max_steps: 0 as the "use epochs" sentinel, which the alias copies as
                # train_steps: 0. normalized() rejects train_steps < 1 before resolve_train_steps() applies
                # num_epochs, so drop a falsy value here.
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


# The shared DiffusionLoraConfig carries save_steps / resume_from_checkpoint for every family, so a
# loop that implements neither has to say so rather than ignore them.
CHECKPOINTLESS_FAMILIES: frozenset[str] = frozenset({"minimax-h3"})

# The batch axis is a pure replication axis for these: the layout, the rotary grid and the row
# timesteps are set by one clip's geometry and its caption's length.
SINGLE_SEQUENCE_FAMILIES: frozenset[str] = frozenset({"minimax-h3"})

# Families whose trainer loads its base through ModularPipeline.from_pretrained. Their local
# layout is modular_model_index.json and no model_index.json, so the conventional shape check
# refuses the only local form the family HAS. One allow_modular set, read by the trainer and by
# the START ROUTE, so the two cannot disagree about the same directory.
MODULAR_BASE_FAMILIES: frozenset[str] = frozenset({"minimax-h3"})

# MiniMax-H3 canvas multiple: a 16x VAE compression and a 2x patch.
_H3_CANVAS_MULTIPLE = 32


def h3_train_unsupported_reason(cfg: Any) -> Optional[str]:
    """Reason this config cannot run the MiniMax-H3 trainer, else None. Never raises.

    Called by the START ROUTE before it frees the resident GPU models, and again by the trainer
    itself so a direct call is refused the same way. Config-only by construction: every check
    here reads the request, never the host, so the route can answer without importing torch or
    touching the GPU. Host capability stays in the precision preflight next to it.
    """
    if (getattr(cfg, "resolved_family", "") or "").strip().lower() != "minimax-h3":
        return None
    if cfg.mixed_precision != "bf16":
        return (
            "MiniMax-H3 LoRA training requires bf16: its checkpoint keeps the patch "
            "projections, the timestep MLP and the output heads in fp32 and fp16 overflows "
            "them. The loop hard-codes the bf16 weight dtype and autocast either way, so any "
            "other setting would be recorded and then not run. Set mixed precision to bf16."
        )
    if cfg.resolution % _H3_CANVAS_MULTIPLE:
        return (
            f"MiniMax-H3 trains on a canvas whose edges are multiples of {_H3_CANVAS_MULTIPLE} "
            f"(a 16x VAE compression and a 2x patch); got resolution {cfg.resolution}."
        )
    if float(getattr(cfg, "cfg_dropout", 0.0) or 0.0) > 0:
        return (
            "MiniMax-H3 is guidance-distilled: it has no unconditional branch and no negative "
            "prompt, so a classifier-free-guidance dropout trains a path the sampler never "
            "takes. Set cfg_dropout to 0."
        )
    if str(getattr(cfg, "weighting_scheme", "none") or "none") != "none":
        return (
            "MiniMax-H3 has no timestep-weighted loss yet: its two schedules put video and "
            "audio at different sigmas in the same step, so a single weight over 'the' "
            "timestep is ambiguous. Use weighting_scheme='none'."
        )
    # Two clips with different captions have different text lengths and therefore different layouts, so
    # a batch > 1 cannot be formed without padding the model has no mask for.
    if cfg.train_batch_size != 1:
        return (
            "MiniMax-H3 trains at batch size 1: one forward covers one packed sequence, whose "
            "row layout is set by the clip's own geometry and its caption's length. Use "
            "gradient_accumulation_steps to raise the effective batch."
        )
    # torch.compile is never invoked here (the packed layout changes shape with every caption length),
    # so an explicit "on" would be accepted and then ignored.
    if str(getattr(cfg, "compile_transformer", "auto") or "auto").strip().lower() == "on":
        return (
            "MiniMax-H3 does not compile: its packed sequence changes length with every clip's "
            "caption, so torch.compile would re-trace each step. Use compile_transformer "
            "'off' or 'auto'."
        )
    # No conditioning cache exists on this path, so accepting the directory would promise a saving that never happens.
    if str(getattr(cfg, "cond_cache_dir", "") or "").strip():
        return (
            "MiniMax-H3 has no persistent conditioning cache yet: each run loads the "
            "conditioner and recomputes its captions and latents, so cond_cache_dir would be "
            "recorded and never read. Leave it unset."
        )
    return None


# These have a DEFAULT the H3 loop disagrees with, so they are normalised rather than refused
# (which would 422 every untouched request): one centre cover-crop, nothing flipped, a plain
# unweighted MSE, and exactly one cached tuple.
_H3_FIXED_RECIPE: dict[str, Any] = {
    "center_crop": True,
    "random_flip": False,
    "snr_gamma": None,
    "cache_latents": True,
    "cache_variants": 1,
}


def train_recipe_overrides(cfg: Any) -> dict[str, Any]:
    """The fields whose REQUESTED value this family's loop replaces, mapped to what it runs.

    Shared for the same reason ``h3_train_unsupported_reason`` is: the trainer applies these in
    the CHILD, while the run record is written by the PARENT from the config handed to
    ``service.start``. Normalising in the trainer alone therefore fixed what ran and left Previous
    Runs describing cropping, flipping and min-SNR weighting that never happened -- exactly the
    recipe drift the normalisation exists to prevent. Both sides read this one table instead.

    Empty for every other family: their loops honour all three."""
    if (getattr(cfg, "resolved_family", "") or "").strip().lower() != "minimax-h3":
        return {}
    return dict(_H3_FIXED_RECIPE)


# LTX-2 is deliberately not here: it trains a style LoRA FROM still images, so it keeps the image discovery.
CLIP_TRAINED_FAMILIES: frozenset[str] = frozenset({"minimax-h3"})


def discover_training_pairs(
    resolved_family: str,
    data_dir: str | os.PathLike[str],
    *,
    instance_prompt: Optional[str] = None,
    caption_column: str = "text",
    verify_images: bool = False,
) -> list[tuple[str, str]]:
    """The ``(path, caption)`` pairs for ``resolved_family``, from whichever discovery its
    trainer runs.

    The /diffusion/start preflight exists so a bad dataset 400s BEFORE the resident GPU models
    are freed, which only holds while it runs the SAME discovery as the trainer. It ran the
    image one unconditionally, so a MiniMax-H3 dataset -- captioned clips, which is the only
    thing its trainer accepts -- was rejected at the route with "No captioned images found" and
    the advertised H3 trainer could not be reached through /diffusion/start at all.

    ``verify_images`` is image-only: the clip discovery has no cheap header probe to match it
    (a container has to be opened to be judged), so it is ignored for a clip family rather than
    quietly implying a check that did not happen."""
    if str(resolved_family or "").strip().lower() in CLIP_TRAINED_FAMILIES:
        from core.training.diffusion_h3_clips import discover_clip_caption_pairs
        return discover_clip_caption_pairs(
            data_dir, instance_prompt = instance_prompt, caption_column = caption_column
        )
    return discover_image_caption_pairs(
        data_dir,
        instance_prompt = instance_prompt,
        caption_column = caption_column,
        verify_images = verify_images,
    )


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


# Two fp32 posterior tensors per crop/flip variant per image, so a few thousand images can exhaust
# pinned RAM; over budget it falls back to per-step VAE encoding.
_LATENT_CACHE_BUDGET_BYTES = 4 * 1024**3  # 4 GiB

# Returned by the cache builders when the estimate exceeds budget: the caller keeps the VAE
# resident. Distinct from ``None`` (a stop requested mid-build).
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
            # The opt-out is a strict-fp32 A/B mode, so actively clear the flags rather than inherit ambient
            # state (cudnn TF32 defaults ON).
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        if cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        # The cuDNN SDPA TRAINING graph is broken for the FLUX attention shapes on torch 2.10 + cu130 (B200)
        # and poisons the context, so pin flash / mem-efficient SDPA for the run (restored on exit).
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
        # Restore the exact pre-run cudnn SDPA state; None means the flag was unreadable at apply time and
        # never touched.
        cuda_backends = getattr(torch.backends, "cuda", None)
        if (
            snap.get("cudnn_sdp") is not None
            and cuda_backends is not None
            and hasattr(cuda_backends, "enable_cudnn_sdp")
        ):
            cuda_backends.enable_cudnn_sdp(bool(snap["cudnn_sdp"]))
    except Exception:  # noqa: BLE001 -- best-effort restore
        pass


# Kept here so the training gate stays independent of the inference allowlist. Exact-match
# lowercased; never add pickles or remote code.
_TRAIN_EXTRA_TRUSTED_REPOS = frozenset(
    {
        "black-forest-labs/flux.2-dev",
        "black-forest-labs/flux.2-klein-4b",
        "black-forest-labs/flux.2-klein-base-4b",
        "black-forest-labs/flux.2-klein-base-9b",
        # A video family, so the image-side inference allowlist never covered it; safetensors-only, no remote code.
        "lightricks/ltx-2",
        # MiniMax-H3's official base, for the same reason: safetensors-only, no remote code.
        "minimaxai/minimax-h3",
    }
)

# LTX-2.3 repos hold SINGLE-FILE checkpoints and no diffusers layout; inference assembles them
# with from_single_file plus 2.3 config overrides (see core/inference/video_ltx2.py), while the
# trainer only knows LTX2Pipeline.from_pretrained. The name still resolves to the ltx-2 family, so
# without an explicit refusal the run evicts residents and only then fails in the child.
_LTX23_TRAIN_UNSUPPORTED = ("lightricks/ltx-2.3", "lightricks/ltx-2.3-fp8")


def _refuse_ltx23_training_base(base_model: str) -> None:
    """Raise for an LTX-2.3 base: the family routing accepts it, the training loader cannot."""
    if str(base_model or "").strip().lower() not in _LTX23_TRAIN_UNSUPPORTED:
        return
    raise ValueError(
        f"'{base_model}' ships LTX-2.3 as single-file checkpoints with no diffusers layout, so it "
        f"cannot be a training base yet: the trainer loads a base with from_pretrained, while 2.3 "
        f"has to be assembled with from_single_file plus components from the 2.0 base. Train from "
        f"'Lightricks/LTX-2' instead."
    )


def _assert_trusted_base_model(base_model: str, *, allow_modular: bool = False) -> None:
    """Gate the training base model the same way the inference backend gates non-GGUF loads:
    a local path or a trusted repo (``unsloth/*`` or an allowlisted official base). This runs
    BEFORE ``from_pretrained`` so an untrusted remote repo (which could ship pickle weights)
    is never fetched or deserialised.

    ``allow_modular`` is for a trainer whose loader is ``ModularPipeline.from_pretrained``: a
    local MiniMax-H3 pipeline carries ``modular_model_index.json`` and no ``model_index.json``,
    so the conventional shape check rejected the one local layout that family HAS."""
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
    # An existing LOCAL base is loaded as a full pipeline, which needs an index; reject a non-pipeline
    # local dir before /diffusion/start frees the GPU models.
    _assert_local_base_is_pipeline(base_model, allow_modular = allow_modular)


# One writer and one reader for BOTH trainers, so an SDXL and a DiT run resume from the same bundle shape.
# ── resume checkpoints ────────────────────────────────────────────────────────
def trainable_state_dict(model: Any) -> dict[str, Any]:
    """The trainable (LoRA) parameters of ``model``, keyed by parameter name.

    Deliberately NOT the peft/diffusers export format: this is the checkpoint's private
    copy of exactly the tensors the optimizer holds moments for, so restoring it and the
    optimizer state together reproduces the run bit-for-bit. Parameter names are stable
    across a re-attach and across regional torch.compile (which compiles submodules in
    place without renaming), which is the same assumption ``LoRAEMA`` already makes."""
    return {name: p.detach() for name, p in model.named_parameters() if p.requires_grad}


def load_trainable_state_dict(model: Any, state: Optional[dict[str, Any]]) -> int:
    """Copy a ``trainable_state_dict`` back into ``model`` in place, returning how many
    parameters were restored. Copies rather than reassigns, so the optimizer's parameter
    references (and their loaded moments) stay valid.

    Every saved tensor must land. A partial match means the parameter NAMES moved (a different
    adapter name, a wrapper that re-prefixes them), and because the optimizer state is keyed by
    parameter INDEX it would still load cleanly -- leaving restored Adam moments and a restored
    LR position driving freshly initialised LoRA weights, while the run reports a normal resume.
    Raise instead."""
    if not state:
        return 0
    import torch

    restored = 0
    trainable: set[str] = set()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            trainable.add(name)
            saved = state.get(name)
            if saved is None:
                continue
            if tuple(saved.shape) != tuple(p.shape):
                raise ValueError(
                    f"Checkpoint tensor '{name}' has shape {tuple(saved.shape)} but this run "
                    f"expects {tuple(p.shape)}; the LoRA configuration does not match."
                )
            p.copy_(saved.to(device = p.device, dtype = p.dtype))
            restored += 1
    # BOTH directions: counting only the checkpoint's own tensors let a truncated adapter holding a
    # strict SUBSET pass, and the optimizer state then loaded on top of freshly initialised weights.
    unsaved = sorted(trainable - set(state))
    unknown = sorted(set(state) - trainable)
    if unsaved or unknown:
        detail = []
        if unsaved:
            detail.append(f"{len(unsaved)} not in the checkpoint (e.g. {', '.join(unsaved[:3])})")
        if unknown:
            detail.append(f"{len(unknown)} not in this run (e.g. {', '.join(unknown[:3])})")
        raise ValueError(
            f"This run has {len(trainable)} trainable tensors and the checkpoint has "
            f"{len(state)}: {'; '.join(detail)}. The LoRA configuration does not match the one "
            "it was saved from."
        )
    return restored


def _json_safe_progress(progress: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Drop non-finite floats from the manifest's progress block. A diverged run pushes
    ``running_loss`` to NaN/inf, which json.dumps writes as the JS-only NaN/Infinity tokens --
    invalid strict JSON that a stricter reader (or a future consumer of these files) rejects."""
    out: dict[str, Any] = {}
    for key, value in (progress or {}).items():
        if isinstance(value, float) and not math.isfinite(value):
            continue
        out[key] = value
    return out


def write_resume_checkpoint(
    cfg: DiffusionLoraConfig,
    *,
    step: int,
    model: Any,
    optimizer: Any,
    lr_scheduler: Any,
    identity: Any,
    on_event: Optional[EventCb] = None,
    ema: Any = None,
    sampler: Any = None,
    rng_streams: Optional[dict[str, Any]] = None,
    progress: Optional[dict[str, Any]] = None,
    discard_existing: bool = False,
    preexisting: Optional[Any] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Write one resume bundle for the run, returning ``(checkpoint_path, error)``.

    Never raises: a checkpoint failure must not lose a training run that is otherwise fine,
    so the error is returned (and emitted as a warning) for the caller to report as
    ``resume_blocked_reason``. Used for BOTH the periodic ``save_steps`` saves and the
    stop-and-save, so the two produce identical state.

    ``discard_existing`` is set on the FIRST write of a run that did not resume, so bundles
    left in the output dir by an earlier run of the same adapter name cannot outrank it."""
    from core.training.diffusion_checkpoint import capture_rng_state, save_checkpoint
    try:
        path = save_checkpoint(
            output_dir = cfg.output_dir,
            step = step,
            adapter_state = trainable_state_dict(model),
            identity = identity,
            target_steps = cfg.train_steps,
            optimizer = optimizer,
            lr_scheduler = lr_scheduler,
            ema_state = ema.state_dict() if ema is not None else None,
            ema_updates = int(getattr(ema, "updates", 0) or 0) if ema is not None else 0,
            rng = capture_rng_state(rng_streams),
            sampler_state = sampler.state_dict() if sampler is not None else None,
            progress = _json_safe_progress(progress),
            save_total_limit = int(cfg.save_total_limit or 0),
            discard_existing = discard_existing,
            # Which bundle THIS is, so the "step already written" shortcut can tell a re-save of our own source
            # apart from another run's bundle at the same number.
            source_checkpoint = getattr(cfg, "resume_from_checkpoint", None),
            # Bundles that predated this run are never pruned to make room for its own: a branched resume used
            # to delete them irreversibly on the first save.
            preexisting = preexisting,
        )
        # Reported per save, so a run that later crashes is still known to have resumable state and one
        # whose write failed is still known to be blocked.
        _emit(on_event, "checkpoint_saved", checkpoint_path = path, step = step)
        return path, None
    except Exception as exc:  # noqa: BLE001 -- reported, never fatal to the run
        message = f"Could not write a resume checkpoint at step {step}: {exc}"
        _emit(on_event, "checkpoint_failed", step = step, message = message)
        return None, message


def _reapply_lr_schedule(optimizer: Any, lr_scheduler: Any) -> None:
    """Recompute the learning rate for the NEXT step from the LIVE schedule.

    ``optimizer.load_state_dict`` restores the rate the checkpoint was written with and
    ``LRScheduler.load_state_dict`` restores only the position, never re-evaluating the lambda,
    so the first step after a resume runs at the OLD schedule's value. That is invisible while
    ``train_steps`` is unchanged (the UI always replays it) but wrong the moment the target
    moves -- continuing a finished cosine run by raising the step count would take its first
    step at lr 0.0, leaving every parameter untouched.

    Only for the closed-form ``LambdaLR`` diffusers' ``get_scheduler`` returns, and evaluated
    from its own public ``lr_lambdas`` / ``base_lrs`` / ``last_epoch`` rather than ``get_lr()``
    (which warns when called outside a step). A chainable scheduler derives its next rate from
    the current one, so re-applying it there would double-step. Best-effort: a schedule we
    cannot re-evaluate keeps the restored value."""
    try:
        import torch

        if not isinstance(lr_scheduler, torch.optim.lr_scheduler.LambdaLR):
            return
        values = [
            float(base * lmbda(lr_scheduler.last_epoch))
            for lmbda, base in zip(lr_scheduler.lr_lambdas, lr_scheduler.base_lrs)
        ]
        for group, value in zip(optimizer.param_groups, values):
            group["lr"] = value
        lr_scheduler._last_lr = values
    except Exception:  # noqa: BLE001 -- keeping the restored rate is a safe fallback
        pass


def restore_resume_state(
    cfg: DiffusionLoraConfig,
    *,
    model: Any,
    optimizer: Any,
    lr_scheduler: Any,
    identity: Any,
    on_event: Optional[EventCb] = None,
    ema: Any = None,
    sampler: Any = None,
    rng_streams: Optional[dict[str, Any]] = None,
) -> Any:
    """Load ``cfg.resume_from_checkpoint`` into a freshly built run.

    Returns the ``LoadedCheckpoint`` (whose ``.step`` is the last COMPLETED optimizer step,
    so the loop restarts at ``.step``), or None when no resume was requested.

    Re-runs the full preflight in the child: the start route already validated the request
    before evicting the GPU, but the trainer must not trust a config it did not check
    itself. Raises ResumeError (a ValueError) on a mismatch, which the process adapter
    surfaces as the run's error message."""
    if not cfg.resume_from_checkpoint:
        return None
    from core.training.diffusion_checkpoint import (
        ResumeError,
        load_checkpoint,
        optimizer_key,
        preflight_resume,
        restore_rng_state,
    )

    path, step = preflight_resume(
        cfg.resume_from_checkpoint, identity = identity, target_steps = cfg.train_steps
    )
    ckpt = load_checkpoint(path)
    load_trainable_state_dict(model, ckpt.tensors("adapter"))
    optimizer_state = ckpt.torch_state("optimizer")
    if optimizer_state is not None:
        # The trainers pick their optimizer from the HOST (bnb present, fused kernel available,
        # UNSLOTH_DIFFUSION_FP32_OPTIM), so foreign moments arrive legitimately (state1/state2 versus
        # exp_avg/exp_avg_sq): shapes match, load_state_dict accepts them, and the first step dies on a
        # bare KeyError.
        saved_optimizer = ckpt.optimizer_class
        live_optimizer = optimizer_key(optimizer)
        if not saved_optimizer:
            # An optimizer file with no class beside it is a hand-edited or half-written bundle, and foreign
            # moments load cleanly then die on the first step, after the preflight evicted the residents.
            raise ResumeError(
                "This checkpoint does not record which optimizer wrote its state, so its "
                "moments cannot be safely restored. Resume from an earlier checkpoint, or "
                "start a new run."
            )
        if saved_optimizer != live_optimizer:
            raise ResumeError(
                f"This checkpoint's optimizer state was written by {saved_optimizer}, but this "
                f"machine builds {live_optimizer}. Install the same optimizer backend (or unset "
                f"UNSLOTH_DIFFUSION_FP32_OPTIM) to continue this run."
            )
        # Optimizer state is keyed by parameter POSITION while the adapter was restored by NAME, so a
        # PEFT/diffusers upgrade that changes traversal order loads every moment cleanly onto a
        # same-shaped wrong tensor.
        saved_names = ckpt.optimizer_param_names
        live_names = list(trainable_state_dict(model))
        if saved_names is not None and saved_names != live_names:
            raise ResumeError(
                "This checkpoint's optimizer state was written for a different parameter order "
                "than this build produces, so its moments cannot be matched to this run's "
                "tensors. Start a new run, or resume on the version that wrote it."
            )
        # load_state_dict replaces the param groups too, so the checkpoint's learning rate wins over a
        # changed cfg, the same semantics as HF Trainer's resume.
        optimizer.load_state_dict(optimizer_state)
    elif optimizer is not None:
        # Every bundle this writer produces has an optimizer, so continuing without one restarts Adam's
        # moments from zero at step N while reporting a clean resume.
        raise ResumeError(
            "This checkpoint carries no optimizer state, so the run cannot be continued from "
            "it: the moments would restart from zero at the resumed step. Start a new run."
        )
    scheduler_state = ckpt.torch_state("scheduler")
    if scheduler_state is not None:
        lr_scheduler.load_state_dict(scheduler_state)
        _reapply_lr_schedule(optimizer, lr_scheduler)
    elif lr_scheduler is not None:
        # A fresh LambdaLR at step 0 would re-warm the learning rate the restored optimizer already moved past.
        raise ResumeError(
            "This checkpoint carries no learning-rate scheduler state, so the schedule would "
            "restart from step 0. Start a new run."
        )
    if sampler is not None and not sampler.load_state_dict(ckpt.sampler_state):
        # Every image checkpoint carries sampler state, so a missing or malformed one leaves a fresh sampler
        # behind an RNG restored past the saved permutation: images silently skipped or repeated.
        raise ResumeError(
            "This checkpoint's dataset sampler state is missing or unreadable, so the image "
            "order cannot be continued. Start a new run."
        )
    if ema is not None:
        ema_state = ckpt.tensors("ema")
        if ema_state:
            missing = ema.missing_from(ema_state)
            if missing:
                # load_state_dict keeps the freshly initialised shadow for anything it cannot match, so an
                # incomplete set blends restored EMA weights with initialisation noise under a clean resume.
                named = ", ".join(missing[:3]) + ("..." if len(missing) > 3 else "")
                raise ResumeError(
                    f"This checkpoint's EMA state is missing or mis-shaped for {len(missing)} "
                    f"of this run's adapter tensors ({named}), so the averaged weights cannot "
                    f"be continued. Resume with EMA disabled, or start a new run."
                )
            ema.load_state_dict(ema_state, updates = ckpt.ema_updates)
        else:
            # EMA is not part of the validated identity, so a resume may turn it on: the object is built
            # BEFORE the adapter is restored, so its shadow holds freshly initialised weights.
            ema.reseed_from(model)
    restore_rng_state(ckpt.rng_json, ckpt.torch_state("rng"), rng_streams)
    _emit(
        on_event,
        "resumed",
        checkpoint_path = str(path),
        step = step,
        total_steps = cfg.train_steps,
        # Which bundle THIS is, not just where it sat: another run can write its own checkpoint over the
        # same slot, and the pathname alone would offer that replacement back as this run's lineage.
        source_created_at = ckpt.manifest.get("created_at"),
    )
    return ckpt


def _publish_to_lora_catalog(
    lora_path: str,
    cfg: DiffusionLoraConfig,
    steps: Optional[int] = None,
) -> Optional[str]:
    """Best-effort copy of the trained adapter into the Unsloth diffusion LoRA directory so
    the Images LoRA picker (which scans only files directly under ``loras/diffusion``) finds
    it without the user moving files. Also writes a ``<alias>.json`` metadata sidecar so the
    picker can family-gate the adapter (family, base model, trigger prompt, ...). Returns the
    published path, or None on any failure.

    ``steps`` is the step count the run actually REACHED, which is what the sidecar must
    record: a run stopped at step 11 of 500 published an adapter claiming 500 steps. None
    keeps the old behaviour for callers that do not know the reached step.

    A VIDEO family publishes nothing. ``loras/diffusion`` is read by the Images LoRA picker
    alone, and ``core/inference/video.py`` has no LoRA surface at all, so mirroring a video
    adapter there would copy a large file into a catalog nothing can load and hand the UI a
    deployment path Unsloth cannot honour. The run still reports ``lora_path`` (and ``ema_path``),
    which is the adapter a caller loads directly. Giving video its own catalog and a load path
    on the Video tab is a separate, larger piece of work."""
    if detect_video_family("", override = cfg.resolved_family) is not None:
        return None
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
        # A retrain with the same adapter name must not clobber a prior mirror: pick the next free numeric suffix.
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
        _write_lora_sidecar(dest.with_suffix(".json"), cfg, steps)
        return str(dest)
    except Exception:  # noqa: BLE001 -- the catalog mirror is best-effort, never fatal
        return None


def _write_lora_sidecar(
    sidecar_path: Path,
    cfg: DiffusionLoraConfig,
    steps: Optional[int] = None,
) -> None:
    """Write the adapter metadata sidecar read back by diffusion_lora._scan_local. Best
    effort: a failure here must not fail publishing, so callers wrap it.

    ``steps`` records the step the run REACHED. It used to record ``cfg.train_steps``, the
    CONFIGURED length, so an adapter saved by stopping at step 11 of 500 advertised 500."""
    meta = {
        "family": cfg.resolved_family,
        "families": [cfg.resolved_family],
        "base_model": cfg.base_model,
        "lora_rank": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "steps": int(steps) if steps is not None else cfg.train_steps,
        "resolution": cfg.resolution,
        "trigger_prompt": cfg.instance_prompt,
        "created_at": time.time(),
        "source": "studio-trained",
    }
    sidecar_path.write_text(json.dumps(meta, indent = 2), encoding = "utf-8")


# Aliases from the generic Unsloth training payload onto DiffusionLoraConfig fields, so the shared
# request shape can also drive this trainer.
_CONFIG_ALIASES = {
    "model_name": "base_model",
    "max_steps": "train_steps",
    # num_epochs already matches the diffusion field name, but list it so the epochs override is
    # threaded through explicitly.
    "num_epochs": "num_epochs",
    "batch_size": "train_batch_size",
    "lora_r": "lora_rank",
    "lr_scheduler_type": "lr_scheduler",
    "random_seed": "seed",
    "lr": "learning_rate",
}


def _coerce_gradient_checkpointing(value: Any) -> bool:
    """Unsloth sends gradient_checkpointing as a string ("none" / "true" / "unsloth"); the
    disable words are False, anything else truthy is True. A real bool passes through."""
    if isinstance(value, str):
        return value.strip().lower() not in ("", "none", "false", "0", "no", "off")
    return bool(value)


def _coerce_bool(value: Any) -> bool:
    """Coerce a flag that may arrive as a string through the generic Unsloth config path
    (e.g. "false" / "0" / "off"). A non-empty string like "false" is otherwise truthy, so
    an opt-out would silently no-op. A real bool passes through."""
    if isinstance(value, str):
        return value.strip().lower() not in ("", "none", "false", "0", "no", "off")
    return bool(value)


def _config_from_dict(config: dict) -> DiffusionLoraConfig:
    """Build a DiffusionLoraConfig from a plain dict. Unknown keys are ignored so a richer
    request payload (UI form) does not break construction; a small set of generic Unsloth
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
    # Epoch-mode payloads carry max_steps: 0 as the "use epochs" sentinel, and normalized() rejects
    # train_steps < 1 before resolve_train_steps() applies num_epochs.
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
