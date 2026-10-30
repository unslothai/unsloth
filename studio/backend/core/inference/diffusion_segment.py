# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text-prompted SAM 3 masks like ComfyUI SAM3_Detect, score = sigmoid(logit) * sigmoid(presence);
dropped when the GPU arbiter loads, evicts the image model or training starts; CPU while training"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import sys
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from loggers import get_logger

logger = get_logger(__name__)

# ── weights ─────────────────────────────────────────────────────────────────────────────────────────────────────────
SEGMENTER_WEIGHTS_FILE = "model.safetensors"
# Only what Sam3Model and its tokenizer read: facebook/sam3 also holds sam3.pt, the same weights in the original format.
SEGMENTER_CONFIG_FILES = (
    "config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)
SEGMENTER_FILES = SEGMENTER_CONFIG_FILES + (SEGMENTER_WEIGHTS_FILE,)


@dataclass(frozen = True)
class SegmenterPrequant:
    """One pre-quantized detector file of a variant (scripts/build_segmenter_prequant_checkpoint.py): it replaces
    model.safetensors, next to the same configs and tokenizer. ``tag`` names its download spec."""

    precision: str
    filename: str
    size_bytes: int
    rotated: bool = False

    @property
    def tag(self) -> str:
        return self.precision + ("-convrot" if self.rotated else "")


@dataclass(frozen = True)
class SegmenterVariant:
    """One downloadable transformers-format SAM 3 checkpoint. ``revision`` is a full commit sha, or "main" for a repo
    whose pin is not set yet (the download then records the commit it resolved)."""

    id: str
    label: str
    repo: str
    revision: str
    size_bytes: int
    gated: bool
    license_url: str
    files: tuple = SEGMENTER_FILES
    # Pre-quantized detector files and the commit that holds them (with the configs and tokenizer); "main" until
    # that commit is pinned. A file not on the Hub there yet reads as absent, and bf16 is used.
    prequant: tuple = ()
    prequant_revision: Optional[str] = None


# SAM 3.1's detector in transformers' Sam3Model format, BF16 (converted from facebook/sam3.1's sam3.1_multiplex.pt).
SAM31_REPO = "unsloth/sam3.1"
# The unsloth/sam3.1 commit the files below were verified at.
SAM31_REVISION = "6cfffd097f83c94fdedf56e8d09d4f89174b733e"
# 1_680_935_372 bytes of bf16 safetensors plus the configs and tokenizer.
SAM31_CONFIG_BYTES = 5_037_537
SAM31_SIZE_BYTES = 1_680_935_372 + SAM31_CONFIG_BYTES
# The commit of unsloth/sam3.1 that holds the pre-quantized files (with the configs and tokenizer). None offers no
# pre-quantized download and every load uses bf16 unless a file is already on disk; a file missing at this commit
# downloads bf16 instead.
SAM31_PREQUANT_REVISION: Optional[str] = "6cfffd097f83c94fdedf56e8d09d4f89174b733e"
# Built by scripts/build_segmenter_prequant_checkpoint.py from the bf16 model.safetensors above; accuracy gate,
# exclusions and the auto rule in diffusion_segment_quant. int8 is the ConvRot build unless UNSLOTH_SAM3_INT8_CONVROT=0.
SAM31_PREQUANT = (
    SegmenterPrequant("int8", "sam3.1-INT8-ConvRot.safetensors", 1_047_746_476, rotated = True),
    SegmenterPrequant("int8", "sam3.1-INT8.safetensors", 992_371_564),
    SegmenterPrequant("fp8", "sam3.1-FP8.safetensors", 1_047_320_804),
)

SAM3_REPO = "facebook/sam3"
SAM3_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
# fp32 safetensors as published (half of it once loaded in bf16 / fp16), plus the tokenizer and configs.
SAM3_SIZE_BYTES = 3_439_938_512 + 5_060_000

SEGMENTER_VARIANTS: dict[str, SegmenterVariant] = {
    "sam3.1": SegmenterVariant(
        id = "sam3.1",
        label = "SAM 3.1",
        repo = SAM31_REPO,
        revision = SAM31_REVISION,
        size_bytes = SAM31_SIZE_BYTES,
        gated = False,
        license_url = f"https://huggingface.co/{SAM31_REPO}/blob/main/LICENSE",
        prequant = SAM31_PREQUANT,
        prequant_revision = SAM31_PREQUANT_REVISION,
    ),
    "sam3": SegmenterVariant(
        id = "sam3",
        label = "SAM 3",
        repo = SAM3_REPO,
        revision = SAM3_REVISION,
        size_bytes = SAM3_SIZE_BYTES,
        gated = True,
        license_url = f"https://huggingface.co/{SAM3_REPO}",
    ),
}
DEFAULT_VARIANT = "sam3.1"
ENV_VARIANT = "UNSLOTH_SAM3_VARIANT"

ENV_PATH_OVERRIDE = "UNSLOTH_SAM3_PATH"
ENV_KEEP_ALIVE = "UNSLOTH_SAM3_KEEP_ALIVE"
# Seconds a host-RAM parked model is kept; 0 never parks.
ENV_PARK_SECONDS = "UNSLOTH_SAM3_PARK_SECONDS"
PARK_SECONDS = 600.0
# "1" pins the parked copy (0.03 s unpark vs ~0.6 s); opt-in since pinning 1.7 GB took minutes on a loaded host.
ENV_PARK_PIN = "UNSLOTH_SAM3_PARK_PIN"

# ── ComfyUI SAM3_Detect defaults ────────────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE = 1008
DEFAULT_THRESHOLD = 0.5
MAX_CONCEPTS = 16
# The detector proposes 200 queries per concept.
MAX_DETECTIONS = 200
TEXT_MAX_LENGTH = 32

# ── memory ──────────────────────────────────────────────────────────────────────────────────────────────────────────
KEEP_ALIVE_SECONDS = 120.0
# bf16 weights (~1.7 GB) plus the 1008x1008 ViT activations and the allocator's slack.
_GPU_NEED_MIB = 4096
# After a call, below this much free (or this fraction of the card) the model leaves at once, so the image model
# never denoises next to it. The prompt upsampler's tight rule.
_TIGHT_FREE_MIB = 6144
_TIGHT_FREE_FRACTION = 0.20
# bf16's measured drop in mem_get_info free after a Find (reserve + workspace, above its allocated peak), so the bf16
# pick is not read as tight by _after_call and unloaded. auto goes 8-bit only if free minus this reads tight.
_BF16_RESIDENT_MIB = 2560


class SegmenterUnavailableError(RuntimeError):
    """This install cannot run SAM 3 (transformers without it)."""


class SegmenterNotDownloadedError(RuntimeError):
    """The weights are not in the cache; the UI offers the download."""


class SegmenterBusyError(RuntimeError):
    """The image model is rendering or loading; the GPU is not free for the segmenter."""


class SegmenterReleaseTimeoutError(RuntimeError):
    """A GPU handover waited out a call still running on the accelerator: the card is not free yet."""


# ── prompt syntax ───────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen = True)
class Concept:
    text: str
    max_detections: int


_MAX_SUFFIX = re.compile(r"^(.+?)\s*:\s*([\d.]+)\s*$")


def parse_prompt(text: Optional[str]) -> list[Concept]:
    """ComfyUI's ``_parse_prompts`` (comfy/text_encoders/sam3_clip.py): drop parentheses, split on commas, and read a
    trailing ``:N`` as that concept's maximum detections (rounded, at least 1; 1 when absent). Duplicates are kept,
    as ComfyUI keeps them. Capped at MAX_CONCEPTS concepts and MAX_DETECTIONS per concept."""
    text = (text or "").replace("(", "").replace(")", "")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    concepts: list[Concept] = []
    for part in parts:
        match = _MAX_SUFFIX.match(part)
        if match:
            label = match.group(1).strip()
            try:
                count = max(1, round(float(match.group(2))))
            except (ValueError, OverflowError):  # "1.2.3", or hundreds of digits (float -> inf)
                label, count = part, 1
        else:
            label, count = part, 1
        if label:
            concepts.append(Concept(label, min(count, MAX_DETECTIONS)))
    if len(concepts) > MAX_CONCEPTS:
        raise ValueError(f"At most {MAX_CONCEPTS} concepts per mask (got {len(concepts)}).")
    return concepts


# ── availability and files ──────────────────────────────────────────────────────────────────────────────────────────
def transformers_supports_sam3() -> bool:
    """Whether the installed transformers ships SAM 3 (5.x), read from the package layout: the status route must not
    import the model code, and an attribute probe on transformers' lazy module during another thread's import of it
    can read as missing."""
    try:
        import importlib.util

        # A --no-torch install ships transformers without torch: SAM 3 has the code there and still cannot run.
        if importlib.util.find_spec("torch") is None:
            return False
        return importlib.util.find_spec("transformers.models.sam3.modeling_sam3") is not None
    except Exception:  # noqa: BLE001
        return False


def _override_dir() -> Optional[str]:
    """The override directory when it holds what Sam3Model and its tokenizer load (a config, the weights, single or
    sharded, and a tokenizer), else None: a partial or unrelated folder must not read as downloaded."""
    value = (os.environ.get(ENV_PATH_OVERRIDE) or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    # Only the variant the override is reported (and loaded) as: another variant's file would plan no weights.
    variant = SEGMENTER_VARIANTS[requested_variant_id() or DEFAULT_VARIANT]
    has = lambda *names: any((path / name).is_file() for name in names)  # noqa: E731
    complete = (
        has("config.json")
        and (
            has(SEGMENTER_WEIGHTS_FILE)
            or _shards_complete(path / (SEGMENTER_WEIGHTS_FILE + ".index.json"))
            # or only a pre-quantized detector file the precision setting may load (dequantized where it cannot
            # run quantized); one the setting rejects (fp8 under auto, any under bf16) is not a loadable model
            or any(
                _loadable(variant, a) and has(a.filename) and not _broken(str(path), a)
                for a in variant.prequant
            )
        )
        # AutoTokenizer (CLIPTokenizer for sam3) loads tokenizer.json, else needs vocab.json and merges.txt together.
        and (has("tokenizer.json") or (has("vocab.json") and has("merges.txt")))
    )
    return str(path) if complete else None


def _shards_complete(index: Path) -> bool:
    """A sharded checkpoint's index is present and every shard its weight_map names is beside it."""
    try:
        weight_map = json.loads(index.read_text(encoding = "utf-8")).get("weight_map")
    except Exception:  # noqa: BLE001 -- missing or unreadable: not a checkpoint
        return False
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    shards = set(weight_map.values())
    return all(isinstance(name, str) and (index.parent / name).is_file() for name in shards)


_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


def _commit(revision: Optional[str]) -> Optional[str]:
    return revision if _COMMIT_SHA.fullmatch(revision or "") else None


def pinned_revision(variant: SegmenterVariant) -> Optional[str]:
    """The variant's commit pin, or None while it is a branch ("main"): the download then pins what it resolves."""
    return _commit(variant.revision)


# A pre-quantized file's download spec id: "<variant>@<tag>", e.g. "sam3.1@int8-convrot".
_SPEC_SEP = "@"


def prequant_spec_id(variant: SegmenterVariant, artifact: SegmenterPrequant) -> str:
    return f"{variant.id}{_SPEC_SEP}{artifact.tag}"


def split_spec_id(model_id: str) -> tuple[SegmenterVariant, Optional[SegmenterPrequant]]:
    """(variant, pre-quantized file or None) for a variant id or a pre-quantized spec id; KeyError when unknown."""
    variant_id, _, tag = str(model_id).partition(_SPEC_SEP)
    variant = SEGMENTER_VARIANTS[variant_id]
    if not tag:
        return variant, None
    for artifact in variant.prequant:
        if artifact.tag == tag:
            return variant, artifact
    raise KeyError(model_id)


def variant_artifact(variant: SegmenterVariant, precision: str) -> Optional[SegmenterPrequant]:
    """The file ``precision`` loads for ``variant``, or None: int8 is the ConvRot build unless
    UNSLOTH_SAM3_INT8_CONVROT=0 picks the plain one (never the other as a stand-in)."""
    from core.inference.diffusion_segment_quant import PRECISION_INT8, int8_convrot_enabled

    rotated = int8_convrot_enabled() if precision == PRECISION_INT8 else False
    for artifact in variant.prequant:
        if artifact.precision == precision and artifact.rotated == rotated:
            return artifact
    return None


def requested_variant_id() -> Optional[str]:
    """The variant UNSLOTH_SAM3_VARIANT names (its id or repo, any case), else None. An unknown name is logged and
    ignored, so a typo falls back to the default instead of breaking the status route."""
    value = (os.environ.get(ENV_VARIANT) or "").strip().lower()
    if not value:
        return None
    for variant in SEGMENTER_VARIANTS.values():
        if value in (variant.id, variant.repo.lower()):
            return variant.id
    logger.warning(
        "images.segment: unknown %s=%r (choices: %s), using the default",
        ENV_VARIANT,
        value,
        ", ".join(SEGMENTER_VARIANTS),
    )
    return None


def _cached_snapshot(
    model_id: str = DEFAULT_VARIANT,
    *,
    hub_cache: Optional[Path] = None,
    revision: Optional[str] = None,
) -> Optional[str]:
    """The snapshot directory holding every file of ``model_id`` (a variant's bf16 files, or a pre-quantized spec's
    configs and file) at one revision (the pin first, then the recorded one, then any cached commit), else None."""
    try:
        from huggingface_hub import hf_hub_download

        from core.inference.stt_sidecar import (
            _active_hf_hub_cache,
            _fallback_revisions,
            _read_revision_record,
        )

        spec = SEGMENTER_SPECS[model_id]
        root = hub_cache if hub_cache is not None else _active_hf_hub_cache()
        if revision is not None:
            revisions = [revision]
        else:
            candidates = [_commit(spec.revision), _read_revision_record(spec.repo)]
            candidates += _fallback_revisions(spec.repo, hub_cache = root)
            revisions = list(dict.fromkeys(rev for rev in candidates if rev))
        for rev in revisions:
            paths = []
            for filename in spec.files:
                try:
                    paths.append(
                        hf_hub_download(
                            repo_id = spec.repo,
                            filename = filename,
                            revision = rev,
                            local_files_only = True,
                            cache_dir = str(root),
                        )
                    )
                except Exception:  # noqa: BLE001 -- missing at this revision
                    paths = []
                    break
            if paths:
                return str(Path(paths[0]).parent)
    except Exception:  # noqa: BLE001 -- an unreadable cache reads as not downloaded
        return None
    return None


# Pre-quantized files that failed to load even dequantized (corrupt, truncated, built for another rule) in this
# process. They stop counting as downloaded, so status and the download route offer the bf16 file instead.
_BROKEN_PREQUANT: set[str] = set()


def _prequant_key(folder: str, artifact: SegmenterPrequant) -> str:
    return os.path.realpath(os.path.join(folder, artifact.filename))


def _broken(folder: Optional[str], artifact: SegmenterPrequant) -> bool:
    return folder is not None and _prequant_key(folder, artifact) in _BROKEN_PREQUANT


def _variant_snapshot(variant_id: str) -> Optional[str]:
    """Cached Sam3Model snapshot: bf16 files, else a pre-quantized file plus configs and tokenizer."""
    path = _cached_snapshot(variant_id)
    if path is not None:
        return path
    variant = SEGMENTER_VARIANTS[variant_id]
    for artifact in variant.prequant:
        if not _loadable(variant, artifact):
            continue
        path = _cached_snapshot(prequant_spec_id(variant, artifact))
        if path is not None and not _broken(path, artifact):
            return path
    return None


def _resolve_active() -> tuple[SegmenterVariant, Optional[str]]:
    """(the active variant, its cached snapshot or None). UNSLOTH_SAM3_VARIANT decides when set; else the default,
    unless only another variant is already downloaded (an install that fetched facebook/sam3 keeps using it). With
    UNSLOTH_SAM3_PATH set the cache is never read: the chosen (or default) variant is reported, with no snapshot."""
    chosen = requested_variant_id()
    if (os.environ.get(ENV_PATH_OVERRIDE) or "").strip():
        return SEGMENTER_VARIANTS[chosen or DEFAULT_VARIANT], None
    if chosen is not None:
        return SEGMENTER_VARIANTS[chosen], _variant_snapshot(chosen)
    default = SEGMENTER_VARIANTS[DEFAULT_VARIANT]
    path = _variant_snapshot(default.id)
    if path is None:
        for variant in SEGMENTER_VARIANTS.values():
            if variant.id != default.id:
                other = _variant_snapshot(variant.id)
                if other is not None:
                    return variant, other
    return default, path


def active_variant() -> SegmenterVariant:
    return _resolve_active()[0]


def model_dir() -> Optional[str]:
    """Env override if set, so a typo'd path is never replaced by the cache; else the active snapshot."""
    if (os.environ.get(ENV_PATH_OVERRIDE) or "").strip():
        return _override_dir()
    return _resolve_active()[1]


@dataclass(frozen = True)
class _SegmenterSpec:
    repo: str
    files: tuple
    label: str
    # A commit sha pins the download; "main" resolves (and records) the newest commit.
    revision: Optional[str] = None
    model_file: str = SEGMENTER_WEIGHTS_FILE
    mmproj_file: str = "config.json"
    # Falls back to the bf16 spec when the prequant file is not on the Hub at that revision.
    fallback: Optional[str] = None


def _build_specs() -> dict:
    specs = {}
    for variant in SEGMENTER_VARIANTS.values():
        specs[variant.id] = _SegmenterSpec(
            repo = variant.repo,
            files = variant.files,
            label = f"{variant.label} segmenter",
            revision = variant.revision,
        )
        for artifact in variant.prequant:
            specs[prequant_spec_id(variant, artifact)] = _SegmenterSpec(
                repo = variant.repo,
                files = SEGMENTER_CONFIG_FILES + (artifact.filename,),
                label = f"{variant.label} {artifact.precision.upper()} segmenter",
                revision = variant.prequant_revision,
                model_file = artifact.filename,
                fallback = variant.id,
            )
    return specs


SEGMENTER_SPECS = _build_specs()


def _resolve_segmenter_id(model_id: Optional[str]) -> str:
    """A variant id (the active one when None) or a pre-quantized spec id."""
    if model_id is None:
        return active_variant().id
    if model_id not in SEGMENTER_SPECS:
        raise ValueError(f"Unknown segmenter '{model_id}'.")
    return model_id


def describe_download_error(model_id: str, error: object) -> Optional[str]:
    """Only a gated refusal names the license to accept; the rest keep the generic message."""
    try:
        variant = split_spec_id(model_id)[0]
    except KeyError:
        return None
    if not variant.gated:
        return None
    name = type(error).__name__
    text = str(error)
    if (
        name == "GatedRepoError"
        or "gated" in text.lower()
        or "403" in text
        or "401" in text
        or "awaiting a review" in text
    ):
        return (
            f"{variant.repo} is gated on Hugging Face. Accept the {variant.label} license at {variant.license_url} "
            "with your account, add a Hugging Face token in Settings, then download again."
        )
    return None


_download_state_lock = threading.Lock()
_download_state = None


def _cached_download_paths(
    model_id: str,
    *,
    hub_cache: Optional[Path] = None,
    revision: Optional[str] = None,
):
    path = _cached_snapshot(model_id, hub_cache = hub_cache, revision = revision)
    return None if path is None else (path, path)


def _get_download_state():
    global _download_state
    with _download_state_lock:
        if _download_state is None:
            from core.inference.stt_mtmd_sidecar import _MtmdDownloadState
            _download_state = _MtmdDownloadState(
                specs = SEGMENTER_SPECS,
                cached_paths = _cached_download_paths,
                resolve = _resolve_segmenter_id,
                label = "Mask segmenter",
                describe_error = describe_download_error,
            )
        return _download_state


def start_download(hf_token: Optional[str] = None, variant: Optional[str] = None) -> str:
    """Download ``variant`` (the active one when None) in the background: the configs, the tokenizer and the one
    weights file this host would run (download_plan), else the bf16 files."""
    model_id = _resolve_segmenter_id(variant)
    if _SPEC_SEP not in model_id:
        model_id = download_plan(SEGMENTER_VARIANTS[model_id])["spec"]
    return _get_download_state().start(model_id, hf_token)


def download_status() -> dict:
    return _get_download_state().status()


def cancel_download(download_id: Optional[str] = None) -> bool:
    return _get_download_state().cancel(None, download_id)


# ── precision ──────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen = True)
class WeightsPlan:
    """Which weights a load uses. ``weights`` is a pre-quantized file (None: the bf16 model.safetensors in
    ``model_dir``); ``dequantize`` loads that file as dense weights (no bf16 file on disk, and the device cannot run
    it quantized, or its quantized load failed)."""

    precision: str
    model_dir: Optional[str]
    weights: Optional[str] = None
    artifact: Optional[SegmenterPrequant] = None
    reason: str = ""
    dequantize: bool = False


def _loadable(variant: SegmenterVariant, artifact: SegmenterPrequant) -> bool:
    """``artifact`` is a file the current UNSLOTH_SAM3_PRECISION may load: the file its precision loads here, and
    under auto one that passed the accuracy gate, under fp8 / int8 that precision's, under bf16 none. Anything else
    on disk is ignored (not used, not even dequantized), so the selection's own file is offered for download."""
    from core.inference.diffusion_segment_quant import (
        PRECISION_AUTO,
        PRECISION_BF16,
        auto_may_load,
        requested_precision,
    )

    from core.inference.prequant_safetensors import safetensors_prequant_supported

    if artifact != variant_artifact(variant, artifact.precision):
        return False
    # An install that cannot read the files (no torchao, or the Windows ROCm stub) cannot even dequantize them.
    if not safetensors_prequant_supported():
        return False
    requested = requested_precision()
    if requested == PRECISION_AUTO:
        return auto_may_load(artifact.tag)
    return requested != PRECISION_BF16 and artifact.precision == requested


def _has_dense_weights(path: Optional[str]) -> bool:
    if not path:
        return False
    folder = Path(path)
    return (folder / SEGMENTER_WEIGHTS_FILE).is_file() or _shards_complete(
        folder / (SEGMENTER_WEIGHTS_FILE + ".index.json")
    )


def _local_artifacts(
    variant: SegmenterVariant, path: Optional[str], *, use_cache: bool
) -> list[tuple[SegmenterPrequant, str]]:
    """(file, its folder) for every pre-quantized file of ``variant`` beside ``path`` or, with ``use_cache``, in its
    own cached snapshot (which may be another commit than the bf16 one)."""
    found = []
    for artifact in variant.prequant:
        if path and (Path(path) / artifact.filename).is_file():
            if not _broken(path, artifact):
                found.append((artifact, path))
        elif use_cache:
            snapshot = _cached_snapshot(prequant_spec_id(variant, artifact))
            if snapshot is not None and not _broken(snapshot, artifact):
                found.append((artifact, snapshot))
    return found


def plan_weights(
    path: Optional[str],
    device: str,
    dtype: Any = None,
    *,
    variant: Optional[SegmenterVariant] = None,
    use_cache: bool = True,
    memory_tight: bool = False,
) -> WeightsPlan:
    """The weights a load on ``device`` uses: the precision UNSLOTH_SAM3_PRECISION asks for (auto: bf16, or the gated
    order in diffusion_segment_quant when ``memory_tight``) when its file is on disk and the device runs it, else
    bf16. bf16 with no bf16 file on disk dequantizes a pre-quantized one the selection may load (never one it
    rejected)."""
    from core.inference.diffusion_segment_quant import PRECISION_BF16, resolve_precision

    variant = variant or active_variant()
    local = _local_artifacts(variant, path, use_cache = use_cache)
    usable: dict = {}
    for artifact, folder in local:
        if _loadable(variant, artifact):
            usable.setdefault(artifact.precision, (artifact, folder))
    precision, why = resolve_precision(
        device, dtype, available = tuple(usable), memory_tight = memory_tight
    )
    if precision != PRECISION_BF16:
        artifact, folder = usable[precision]
        return WeightsPlan(precision, folder, str(Path(folder) / artifact.filename), artifact, why)
    if _has_dense_weights(path) or not usable:
        return WeightsPlan(PRECISION_BF16, path, reason = why)
    artifact, folder = next(iter(usable.values()))
    return WeightsPlan(
        PRECISION_BF16,
        folder,
        str(Path(folder) / artifact.filename),
        artifact,
        f"{why}; no bf16 file on disk, {artifact.filename} is dequantized",
        dequantize = True,
    )


def _host_device() -> str:
    """The device a call would most likely use, for planning without a call: the current CUDA card, else Metal,
    else the CPU. Reads properties only (no CUDA context)."""
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def download_plan(
    variant: SegmenterVariant,
    device: Optional[str] = None,
    *,
    memory_tight: Optional[bool] = None,
) -> dict:
    """Pre-quantized files only when published and runnable here; the spec falls back to bf16 if absent."""
    from core.inference.diffusion_segment_quant import PRECISION_BF16, resolve_precision

    offered = {}
    if variant.prequant_revision:
        for artifact in variant.prequant:
            # A cached copy that already failed to load would only be "downloaded" again: offer bf16 instead.
            if _loadable(variant, artifact) and not _broken(
                _cached_snapshot(prequant_spec_id(variant, artifact)), artifact
            ):
                offered.setdefault(artifact.precision, artifact)
    device = device or _host_device()
    if memory_tight is None:
        memory_tight = auto_memory_tight(device)
    precision, why = resolve_precision(device, available = tuple(offered), memory_tight = memory_tight)
    if precision == PRECISION_BF16:
        return {
            "spec": variant.id,
            "precision": precision,
            "reason": why,
            "file": SEGMENTER_WEIGHTS_FILE,
            "size_bytes": variant.size_bytes,
        }
    artifact = offered[precision]
    config_bytes = SAM31_CONFIG_BYTES if variant.id == "sam3.1" else 0
    return {
        "spec": prequant_spec_id(variant, artifact),
        "precision": precision,
        "reason": why,
        "file": artifact.filename,
        "size_bytes": artifact.size_bytes + config_bytes,
    }


def variant_summary(variant: SegmenterVariant) -> dict:
    """What the status and download routes report about a variant."""
    return {
        "variant": variant.id,
        "label": variant.label,
        "repo": variant.repo,
        "revision": variant.revision,
        "gated": variant.gated,
        "license_url": variant.license_url,
        "size_bytes": variant.size_bytes,
    }


def status() -> dict:
    """The status route body: can it run, which variant, is it downloaded, is it loaded (and where), and the download
    progress."""
    supported = transformers_supports_sam3()
    override = bool((os.environ.get(ENV_PATH_OVERRIDE) or "").strip())
    if supported:
        variant, path = _resolve_active()
        if override:
            path = _override_dir()
    else:
        variant, path = SEGMENTER_VARIANTS[requested_variant_id() or DEFAULT_VARIANT], None
    segmenter = _segmenter
    precision = _precision_summary(variant, path, segmenter, supported = supported, override = override)
    return {
        "supported": supported,
        "reason": None
        if supported
        else "Mask from text needs PyTorch and transformers 5 (Python 3.10 or newer); this install lacks one of them.",
        "downloaded": path is not None,
        "loaded": bool(segmenter is not None and segmenter.loaded),
        "device": segmenter.device if segmenter is not None and segmenter.loaded else None,
        **variant_summary(variant),
        **precision,
        "override": override,
        "download": download_status(),
    }


def _precision_summary(
    variant: SegmenterVariant,
    path: Optional[str],
    segmenter: Optional["MaskSegmenter"],
    *,
    supported: bool,
    override: bool,
) -> dict:
    """The status route's precision fields: what runs (the loaded model), else what a load on this host would use
    (on disk), else what a download would fetch. ``size_bytes`` is that download's size until it is on disk."""
    from core.inference.diffusion_segment_quant import requested_precision

    body = {"precision_requested": requested_precision()}
    try:
        if segmenter is not None and segmenter.loaded and segmenter.precision:
            body.update(
                precision = segmenter.precision,
                precision_reason = segmenter.precision_reason,
                weights_file = segmenter.weights_file,
            )
        elif not supported:
            body.update(precision = None, precision_reason = None, weights_file = None)
        elif path is not None:
            device = _host_device()
            plan = plan_weights(
                path,
                device,
                variant = variant,
                use_cache = not override,
                memory_tight = auto_memory_tight(device),
            )
            body.update(
                precision = plan.precision,
                precision_reason = plan.reason,
                weights_file = os.path.basename(plan.weights)
                if plan.weights
                else SEGMENTER_WEIGHTS_FILE,
            )
        else:
            plan = download_plan(variant)
            body.update(
                precision = plan["precision"],
                precision_reason = plan["reason"],
                weights_file = plan["file"],
                size_bytes = plan["size_bytes"],
            )
    except Exception as exc:  # noqa: BLE001 -- the status route must answer
        logger.debug("images.segment: precision summary failed: %s", exc)
        body.update(precision = None, precision_reason = None, weights_file = None)
    return body


# ── device choice ───────────────────────────────────────────────────────────────────────────────────────────────────
def _resident_cuda_ordinal() -> Optional[int]:
    """The CUDA card the resident image model sits on, else None."""
    try:
        from core.inference.diffusion_engine_router import get_active_diffusion_engine
        ordinal = getattr(
            getattr(get_active_diffusion_engine(), "_state", None), "placed_ordinal", None
        )
        return None if ordinal is None else int(ordinal)
    except Exception:  # noqa: BLE001
        return None


def _cuda_free_mib(ordinal: int) -> tuple[Optional[int], Optional[int]]:
    """(free, total) MiB on CUDA card ``ordinal``: driver free plus this process's reusable allocator cache, the same
    credit the render's memory check gives (diffusion_memory.cuda_reclaimable_bytes), less the user's VRAM Budget
    reserve (raw at the default), so the segmenter neither lands on nor stays resident in that reserve. Parking leaves
    SAM's emptied weights in that cache on purpose, so driver free alone would send the next detection to the CPU on a
    warm card (ComfyUI's get_free_memory likewise adds reserved minus active bytes)."""
    try:
        import torch
        free, total = torch.cuda.mem_get_info(ordinal)
    except Exception:  # noqa: BLE001
        return None, None
    try:
        from core.inference.diffusion_memory import cuda_reclaimable_bytes
        reclaimable = cuda_reclaimable_bytes(ordinal) or 0
    except Exception:  # noqa: BLE001 -- no allocator reading: driver free still stands
        reclaimable = 0
    free = min(int(free) + max(0, int(reclaimable)), int(total))
    try:
        from core.inference.diffusion_memory import user_budget_reserve_bytes
        reserve = int(user_budget_reserve_bytes("cuda", total) or 0)
    except Exception:  # noqa: BLE001 -- no Budget reading: the raw figure stands
        reserve = 0
    free = max(0, free - reserve)
    return int(free // (1024 * 1024)), int(total // (1024 * 1024))


def vram_is_tight(free_mib: Optional[int], total_mib: Optional[int]) -> bool:
    if free_mib is None:
        return False
    floor = _TIGHT_FREE_MIB
    if total_mib:
        floor = max(floor, int(total_mib * _TIGHT_FREE_FRACTION))
    return free_mib < floor


def auto_memory_tight(device: str) -> bool:
    """The CUDA card ``device`` would read as tight with the bf16 segmenter on it. False off CUDA, when unreadable, and
    before this process initialised CUDA (never create a context just to plan)."""
    if not device.startswith("cuda"):
        return False
    try:
        import torch
        if not torch.cuda.is_initialized():
            return False
        ordinal = int(device.split(":", 1)[1]) if ":" in device else torch.cuda.current_device()
    except Exception:  # noqa: BLE001
        return False
    free_mib, total_mib = _cuda_free_mib(ordinal)
    if free_mib is None:
        return False
    return vram_is_tight(free_mib - _BF16_RESIDENT_MIB, total_mib)


def _other_owner_on(ordinal: Optional[int]) -> Optional[str]:
    """The GPU arbiter's resident other than the image model on CUDA card ``ordinal`` (one with unknown GPUs counts
    for every card; ``ordinal`` None is Apple's single Metal device, which every resident shares), else None. Called under the run lock after the handover fence: a later handover waits for the
    call (or aborts) instead."""
    try:
        from core.inference import gpu_arbiter
        held = gpu_arbiter.residents()
        physical = None if ordinal is None else gpu_arbiter.physical_gpu_ids([ordinal])
    except Exception:  # noqa: BLE001 -- no arbiter: nothing else is placed through it
        return None
    for owner, gpus in held.items():
        if owner == gpu_arbiter.DIFFUSION:
            continue
        if gpus is None or physical is None or gpus & physical:
            return owner
    return None


def choose_device(allow_gpu: bool = True) -> tuple[str, Any]:
    """(torch device string, dtype): the image model's CUDA card (else the current one) in bf16 (fp16 where bf16 is
    not supported) when it has room for the segmenter, Apple's MPS in fp32, else the CPU in fp32."""
    import torch

    if allow_gpu and torch.cuda.is_available():
        ordinal = _resident_cuda_ordinal()
        if ordinal is None or ordinal >= torch.cuda.device_count():
            ordinal = torch.cuda.current_device()
        other = _other_owner_on(ordinal)
        if other is not None:
            # A chat / video / other model owns the card: it grows its working memory without knowing about SAM 3.
            logger.info("images.segment: cuda:%s is held by %s, running on the CPU", ordinal, other)
            return "cpu", torch.float32
        free_mib, _total = _cuda_free_mib(ordinal)
        if free_mib is None or free_mib >= _GPU_NEED_MIB:
            bf16 = False
            with suppress(Exception):
                # is_bf16_supported reads the CURRENT card; the segmenter runs on ``ordinal``.
                with torch.cuda.device(ordinal):
                    bf16 = bool(torch.cuda.is_bf16_supported())
            return f"cuda:{ordinal}", torch.bfloat16 if bf16 else torch.float16
        logger.info("images.segment: %s MiB free on cuda:%s, running on the CPU", free_mib, ordinal)
    if (
        allow_gpu
        and getattr(getattr(torch, "backends", None), "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        other = _other_owner_on(None)
        if other is None:
            return "mps", torch.float32
        logger.info("images.segment: Metal is held by %s, running on the CPU", other)
    return "cpu", torch.float32


# ── inference ───────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Instance:
    concept: str
    score: float
    box: tuple[float, float, float, float]


@dataclass
class SegmentResult:
    mask_png: bytes
    width: int
    height: int
    instances: list[Instance] = field(default_factory = list)
    concepts: list[dict] = field(default_factory = list)
    device: str = "cpu"
    elapsed_ms: float = 0.0
    precision: Optional[str] = None
    # Per-instance CPU mask logits (instance_masks only, never in the route body); None for filtered planes.
    instance_logits: list = field(default_factory = list)

    def to_json(self) -> dict:
        return {
            "mask": "data:image/png;base64," + base64.b64encode(self.mask_png).decode("ascii"),
            "width": self.width,
            "height": self.height,
            "instances": [
                {
                    "prompt": i.concept,
                    "score": round(i.score, 4),
                    "box": [round(v, 1) for v in i.box],
                }
                for i in self.instances
            ],
            "prompts": self.concepts,
            "device": self.device,
            "precision": self.precision,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


def model_image_size(model: Any) -> int:
    """The square input the checkpoint was built for (1008 for SAM 3)."""
    try:
        return int(model.config.vision_config.backbone_config.image_size)
    except Exception:  # noqa: BLE001
        return IMAGE_SIZE


def preprocess(
    image: Any,
    device: str,
    dtype: Any,
    size: int = IMAGE_SIZE,
) -> Any:
    """ComfyUI's input: RGB in [0, 1] stretched to 1008x1008 with bilinear interpolation (no crop, no antialias),
    then SAM 3's normalisation (mean 0.5, std 0.5)."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    array = np.array(image.convert("RGB"), dtype = np.uint8)
    tensor = torch.from_numpy(array).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = F.interpolate(tensor, size = (size, size), mode = "bilinear", align_corners = False)
    return ((tensor - 0.5) / 0.5).to(dtype)


def select_detections(
    pred_logits: Any,
    pred_boxes: Any,
    pred_masks: Any,
    *,
    max_detections: int,
    threshold: float,
    presence_logit: Any = None,
) -> tuple[Any, Any, Any]:
    """ComfyUI's keep rule for one concept: score > threshold, best first, at most ``max_detections``, with the score
    sigmoid(logit) * sigmoid(presence) when the presence logit is given. Takes one image's (Q,), (Q, 4) normalised
    xyxy and (Q, h, w) logits; returns the kept scores, boxes, masks."""
    probs = pred_logits.float().sigmoid()
    if presence_logit is not None:
        probs = probs * presence_logit.float().sigmoid().reshape(-1)[0]
    keep = probs > threshold
    probs, boxes, masks = probs[keep], pred_boxes[keep], pred_masks[keep]
    order = probs.argsort(descending = True)[:max_detections]
    return probs[order], boxes[order], masks[order]


def union_mask(masks: Any, height: int, width: int) -> Any:
    """(N, h, w) mask logits -> one (height, width) bool mask: bilinear upsample, > 0, any."""
    import torch
    import torch.nn.functional as F

    if masks is None or masks.shape[0] == 0:
        return torch.zeros(
            height, width, dtype = torch.bool, device = None if masks is None else masks.device
        )
    union = torch.zeros(height, width, dtype = torch.bool, device = masks.device)
    # One instance at a time keeps a 4K source to a single full-size float plane.
    for logits in masks:
        up = F.interpolate(
            logits[None, None].float(), size = (height, width), mode = "bilinear", align_corners = False
        )[0, 0]
        union |= up > 0
    return union


def mask_to_png(mask: Any) -> bytes:
    from PIL import Image

    array = (mask.to("cpu").numpy().astype("uint8")) * 255
    buffer = io.BytesIO()
    # Every zlib level is lossless; optimize=True only buys a few KB for several times the encode time.
    Image.fromarray(array, mode = "L").save(buffer, format = "PNG", compress_level = 1)
    return buffer.getvalue()


def _free_device_cache(device: Optional[str]) -> None:
    """Collect what nothing references any more, then hand ``device``'s cached blocks back to the driver."""
    import gc

    gc.collect()
    if device and device.startswith("cuda"):
        with suppress(Exception):
            import torch
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    elif device == "mps":
        with suppress(Exception):
            import torch
            torch.mps.empty_cache()


class _HostWeights:
    """Parked weights as one host buffer and one device storage every weight views (via ``Tensor.set_``), so
    park/unpark is one copy instead of ~1,500 allocations, which on a busy host is slower than rereading the file."""

    _ALIGN = 256

    def __init__(self, pin: bool) -> None:
        self.pin = pin
        self.buffer: Any = None
        self.storage: Any = None
        self.total = 0

    @staticmethod
    def _leaves(model: Any) -> list:
        import torch

        out: list = []

        def _visit(t: Any) -> None:
            if type(t) not in (torch.Tensor, torch.nn.Parameter):
                flatten = getattr(t, "__tensor_flatten__", None)
                if flatten is None:
                    raise TypeError(f"cannot park a {type(t).__name__} tensor")
                names, _ctx = flatten()
                for name in names:
                    _visit(getattr(t, name))
                return
            out.append(t)

        # The registered objects themselves, not ``.data`` aliases: set_ has to re-point the tensor the module uses.
        for param in model.parameters():
            _visit(param)
        for buffer in model.buffers():
            _visit(buffer)
        return out

    def _bytes(
        self,
        storage: Any,
        nbytes: int,
        offset: int = 0,
    ) -> Any:
        import torch
        view = torch.empty(0, dtype = torch.uint8, device = storage.device)
        return view.set_(storage, offset, (nbytes,), (1,))

    def _repack(self, model: Any) -> None:
        import torch

        leaves = self._leaves(model)
        slots: dict = {}
        offset = 0
        for leaf in leaves:
            storage = leaf.untyped_storage()
            key = storage.data_ptr()
            if key in slots or storage.nbytes() == 0:
                continue
            slots[key] = (storage, offset)
            offset += -(-storage.nbytes() // self._ALIGN) * self._ALIGN
        self.total = offset
        self.buffer = torch.empty(self.total, dtype = torch.uint8, pin_memory = self.pin)
        storages = [st for st, _off in slots.values()]
        if storages:
            torch._foreach_copy_(
                [self.buffer[off : off + st.nbytes()] for st, off in slots.values()],
                [self._bytes(st, st.nbytes()) for st in storages],
                non_blocking = self.pin,
            )
            if storages[0].device.type == "cuda":
                torch.cuda.synchronize(storages[0].device)
        device = storages[0].device if storages else torch.device("cpu")
        packed = torch.empty(self.total, dtype = torch.uint8, device = device).untyped_storage()
        placed = []
        for leaf in leaves:
            storage = leaf.untyped_storage()
            slot = slots.get(storage.data_ptr())
            if slot is None:
                continue
            size, stride, start = leaf.size(), leaf.stride(), leaf.storage_offset()
            placed.append((leaf, slot[1] // leaf.element_size() + start, size, stride))
        # Re-pointed only once every offset is known; the old storages go with their last reference.
        for leaf, start, size, stride in placed:
            with torch.no_grad():
                leaf.set_(packed, start, size, stride)
        del storages, slots, placed
        self.storage = packed

    def park(self, model: Any) -> None:
        import torch
        if self.storage is None:
            self._repack(model)
        elif self.total:
            self.buffer.copy_(self._bytes(self.storage, self.total), non_blocking = self.pin)
            if self.storage.device.type == "cuda":
                torch.cuda.synchronize(self.storage.device)
        self.storage.resize_(0)

    def unpark(self, model: Any, device: str) -> None:
        import torch

        self.storage.resize_(self.total)
        if self.total:
            self._bytes(self.storage, self.total).copy_(self.buffer, non_blocking = self.pin)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)


class MaskSegmenter:
    """One lazily loaded Sam3Model + its tokenizer. ``_run_lock`` serialises loads, calls and unloads."""

    def __init__(self, keep_alive_seconds: Optional[float] = None) -> None:
        if keep_alive_seconds is None:
            try:
                keep_alive_seconds = float(os.environ.get(ENV_KEEP_ALIVE) or KEEP_ALIVE_SECONDS)
            except ValueError:
                keep_alive_seconds = KEEP_ALIVE_SECONDS
        self.keep_alive_seconds = max(0.0, keep_alive_seconds)
        self._run_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._path: Optional[str] = None
        self._weights_key: Optional[tuple] = None
        self.device: Optional[str] = None
        self._dtype = None
        # What the loaded model runs: "bf16" / "fp8" / "int8", why, and the weights file it came from.
        self.precision: Optional[str] = None
        self.precision_reason: Optional[str] = None
        self.weights_file: Optional[str] = None
        # Why the last pre-quantized load fell back to bf16, if one did.
        self.last_prequant_failure: Optional[str] = None
        # Reused while loaded: a reading taken with the model resident would flip auto's choice and reload it.
        self._memory_tight: Optional[bool] = None
        # Planning reads the Hub cache only for the active variant's own snapshot, not an explicit folder.
        self._use_cache = True
        self._timer: Optional[threading.Timer] = None
        # Batch holds (hold()): while one is open a call keeps the model loaded; the holder releases it at the end.
        self._holds = 0
        self._release_pending = False
        self._release_epoch = 0
        # Bumped by every GPU handover's release: a call that queued before one never runs on the accelerator.
        self._handover_epoch = 0
        # The device of the call in flight: "" until it is chosen, None when no call runs.
        self._call_device: Optional[str] = None
        # Set (under _state_lock) once the call in flight has read its release flag: a release that times out after
        # that is not honoured by the call, so it waits out the call's short tail itself.
        self._call_settled = False
        # A GPU model moved to host RAM by release(park = True), with what _load matches it against; the next call on
        # the same card, weights and dtype moves it back instead of reading the files again. Never counts as loaded.
        self._parked: Optional[dict] = None
        self._park_timer: Optional[threading.Timer] = None
        # The weights' host copy, kept while the same model is loaded so a later park copies nothing back.
        self._host: Optional[_HostWeights] = None
        try:
            self.park_seconds = max(0.0, float(os.environ.get(ENV_PARK_SECONDS) or PARK_SECONDS))
        except ValueError:
            self.park_seconds = PARK_SECONDS

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def holds_vram(self) -> bool:
        return self._model is not None and bool(self.device) and self.device != "cpu"

    @property
    def parked(self) -> bool:
        return self._parked is not None

    # -- lifecycle -------------------------------------------------------------------------------------------------
    def _load(self, path: str, device: str, dtype: Any) -> None:
        if self._model is not None and self.device == device and self._memory_tight is not None:
            memory_tight = self._memory_tight
        else:
            memory_tight = auto_memory_tight(device)
        plan = plan_weights(
            path, device, dtype, use_cache = self._use_cache, memory_tight = memory_tight
        )
        key = (plan.model_dir, plan.weights, plan.dequantize)
        if (
            self._model is not None
            and self._path == path
            and self._weights_key == key
            and self.device == device
            and self._dtype == dtype
        ):
            return
        if self._model is not None:
            self._unload_locked("reload")
        if self._unpark_locked(path, key, device, dtype):
            return
        from transformers import AutoTokenizer

        started = time.perf_counter()
        # The tokenizer first: a broken one fails before any weights are placed on the device.
        tokenizer = AutoTokenizer.from_pretrained(plan.model_dir or path)
        model = None
        try:
            model, plan = self._load_weights(plan, path, device, dtype)
        except BaseException:
            # Weights placed before the failure are referenced nowhere else: drop them and their cached blocks.
            model = None
            _free_device_cache(device)
            raise
        self._model, self._tokenizer = model, tokenizer
        self._path, self.device, self._dtype = path, device, dtype
        self._weights_key = key
        self._memory_tight = memory_tight
        self.precision, self.precision_reason = plan.precision, plan.reason
        self.weights_file = (
            os.path.basename(plan.weights) if plan.weights else SEGMENTER_WEIGHTS_FILE
        )
        logger.info(
            "images.segment: loaded %s on %s (%s, precision %s from %s: %s) in %.1fs",
            plan.model_dir or path,
            device,
            str(dtype).replace("torch.", ""),
            plan.precision,
            self.weights_file,
            plan.reason,
            time.perf_counter() - started,
        )

    def _load_weights(
        self, plan: WeightsPlan, path: str, device: str, dtype: Any
    ) -> tuple[Any, WeightsPlan]:
        """(model, the plan that loaded). A pre-quantized file that fails to load falls back to the bf16 file, else
        to the same file dequantized; nothing is left half-placed on the device."""
        from transformers import Sam3Model

        from core.inference.diffusion_segment_quant import (
            PRECISION_BF16,
            load_prequantized_segmenter,
        )

        variant = active_variant()
        base = variant.repo
        if plan.weights is not None and not plan.dequantize:
            try:
                model, _info = load_prequantized_segmenter(
                    plan.model_dir,
                    plan.weights,
                    scheme = plan.artifact.precision,
                    rotated = plan.artifact.rotated,
                    device = device,
                    dtype = dtype,
                    base_model_id = base,
                )
                self.last_prequant_failure = None
                return model, plan
            except Exception as exc:  # noqa: BLE001 -- bf16 is slower and bigger, never wrong
                self.last_prequant_failure = f"{type(exc).__name__}: {str(exc)[:200]}"
                logger.warning(
                    "images.segment: pre-quantized %s file %s did not load (%s); using bf16",
                    plan.precision,
                    os.path.basename(plan.weights),
                    self.last_prequant_failure,
                )
                _free_device_cache(device)
                why = f"{plan.artifact.filename} did not load"
                if _has_dense_weights(path):
                    plan = WeightsPlan(PRECISION_BF16, path, reason = why)
                else:
                    plan = WeightsPlan(
                        PRECISION_BF16,
                        plan.model_dir,
                        plan.weights,
                        plan.artifact,
                        f"{why}; dequantized",
                        dequantize = True,
                    )
        if plan.weights is not None:
            try:
                model, _info = load_prequantized_segmenter(
                    plan.model_dir,
                    plan.weights,
                    scheme = plan.artifact.precision,
                    rotated = plan.artifact.rotated,
                    device = device,
                    dtype = dtype,
                    base_model_id = base,
                    dequantize = True,
                )
            except Exception as exc:
                # Not a quantized-kernel problem: the file itself does not load. Stop counting it as downloaded so
                # the next status offers the bf16 download rather than failing every request on the same file.
                _BROKEN_PREQUANT.add(_prequant_key(plan.model_dir, plan.artifact))
                _free_device_cache(device)
                raise RuntimeError(
                    f"{plan.artifact.filename} could not be loaded ({type(exc).__name__}: {str(exc)[:200]}); "
                    f"download the {variant.label} BF16 weights from the Mask from text panel"
                ) from exc
            return model, plan
        if device.startswith("cuda"):
            # Straight onto the card: about a third faster than loading on the CPU and moving it.
            model = Sam3Model.from_pretrained(
                plan.model_dir or path, dtype = dtype, device_map = device
            )
        else:
            model = Sam3Model.from_pretrained(plan.model_dir or path, dtype = dtype)
            model.to(device)
        model.eval()
        return model, plan

    def _unpark_locked(self, path: str, key: tuple, device: str, dtype: Any) -> bool:
        """Unpark when weights, card and dtype match (else drop the parked copy); False when nothing was usable."""
        parked = self._parked
        if parked is None:
            return False
        if (parked["path"], parked["key"], parked["device"], parked["dtype"]) != (
            path,
            key,
            device,
            dtype,
        ):
            self._drop_parked_locked("another load")
            return False
        self._parked = None
        self._cancel_park_timer()
        started = time.perf_counter()
        model = parked["model"]
        try:
            parked["host"].unpark(model, device)
        except BaseException:
            model = None
            parked.clear()
            self._host = None
            _free_device_cache(device)
            logger.warning(
                "images.segment: the parked model did not move back to %s; loading again", device
            )
            return False
        self._model, self._tokenizer = model, parked["tokenizer"]
        self._host = parked["host"]
        self._path, self.device, self._dtype, self._weights_key = path, device, dtype, key
        self.precision, self.precision_reason, self.weights_file = (
            parked["precision"],
            parked["precision_reason"],
            parked["weights_file"],
        )
        logger.info(
            "images.segment: moved back from host RAM to %s in %.2fs",
            device,
            time.perf_counter() - started,
        )
        return True

    def _park_locked(self, reason: str) -> bool:
        """Move the GPU model to host RAM and free its VRAM, keeping it for the next call. Falls back to an unload
        on a CPU / MPS model, with parking off, or when the move fails. True when the weights left the device."""
        if self._model is None:
            return False
        device = self.device
        if self.park_seconds <= 0 or not (device or "").startswith("cuda"):
            return self._unload_locked(reason)
        with self._state_lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        started = time.perf_counter()
        model = self._model
        from .host_pinning import host_pinning_allowed

        host = self._host or _HostWeights(
            pin = (os.environ.get(ENV_PARK_PIN) or "").strip() == "1" and host_pinning_allowed(device)
        )
        try:
            host.park(model)
            copied = time.perf_counter() - started
        except Exception as exc:  # noqa: BLE001 -- a module that will not move is simply unloaded
            logger.warning("images.segment: could not park in host RAM (%s); unloading", exc)
            self._host = None
            return self._unload_locked(reason)
        self._drop_parked_locked("replaced")
        self._host = None
        self._parked = {
            "model": model,
            "host": host,
            "tokenizer": self._tokenizer,
            "path": self._path,
            "key": self._weights_key,
            "device": device,
            "dtype": self._dtype,
            "precision": self.precision,
            "precision_reason": self.precision_reason,
            "weights_file": self.weights_file,
        }
        self._model = self._tokenizer = None
        self._path = self._weights_key = self.device = self._dtype = None
        self.precision = self.precision_reason = self.weights_file = None
        del model
        # No empty_cache: the flush took minutes on a busy card; the cache counts as free for the render's check.
        self._arm_park_timer()
        logger.info(
            "images.segment: parked in host RAM from %s in %.2fs (copy %.2fs) (%s)",
            device,
            time.perf_counter() - started,
            copied,
            reason,
        )
        return True

    def _drop_parked_locked(self, reason: str) -> bool:
        self._cancel_park_timer()
        if self._parked is None:
            return False
        device = self._parked.get("device")
        self._parked = None
        self._host = None
        _free_device_cache(device)
        logger.info("images.segment: dropped the host RAM copy (%s)", reason)
        return True

    def _cancel_park_timer(self) -> None:
        with self._state_lock:
            if self._park_timer is not None:
                self._park_timer.cancel()
                self._park_timer = None

    def _arm_park_timer(self) -> None:
        with self._state_lock:
            if self._park_timer is not None:
                self._park_timer.cancel()
            timer = threading.Timer(self.park_seconds, self._park_expire)
            timer.daemon = True
            self._park_timer = timer
            timer.start()

    def _park_expire(self) -> None:
        if not self._run_lock.acquire(blocking = False):
            # A call is running: it either takes the parked copy or leaves it for a new timer.
            if self._parked is not None:
                self._arm_park_timer()
            return
        try:
            self._drop_parked_locked("park expired")
        finally:
            self._run_lock.release()

    def _unload_locked(self, reason: str) -> bool:
        with self._state_lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        if self._model is None:
            return False
        device = self.device
        self._model = None
        self._tokenizer = None
        self._host = None
        self._path = None
        self._weights_key = None
        self.device = None
        self._dtype = None
        self._memory_tight = None
        self.precision = self.precision_reason = self.weights_file = None
        _free_device_cache(device)
        logger.info("images.segment: unloaded from %s (%s)", device, reason)
        return True

    def _call_on_accelerator(self) -> bool:
        """A call is in flight on the GPU / MPS, or has not chosen its device yet."""
        device = self._call_device
        return device is not None and device != "cpu"

    def release(
        self,
        reason: str = "release",
        *,
        wait: bool = False,
        timeout: float = 30.0,
        handover: bool = False,
        park: bool = False,
    ) -> bool:
        """Unload now if idle. A call in flight unloads when it ends (``wait`` waits up to ``timeout`` for it
        instead). True when the weights left now. ``handover`` (the GPU is changing hands) raises
        SegmenterReleaseTimeoutError when the wait ran out on a call still on the accelerator. ``park`` (an image
        render takes the GPU back) moves a GPU model to host RAM for the next call instead of dropping it; without
        it any parked copy is dropped too."""
        with self._state_lock:
            self._release_epoch += 1
            if handover:
                self._handover_epoch += 1
        acquired = (
            self._run_lock.acquire(timeout = timeout)
            if wait
            else self._run_lock.acquire(blocking = False)
        )
        if not acquired:
            with self._state_lock:
                self._release_pending = True
                # Read with the flag: a call that has not settled yet sees the flag and unloads when it ends.
                honoured = self._call_device is not None and not self._call_settled
                on_accelerator = self._call_on_accelerator()
            if wait and not honoured:
                # The call ended (or is ending) without seeing the flag: its tail is short, so wait it out and unload
                # here rather than leave the weights resident for the next owner.
                acquired = self._run_lock.acquire(timeout = _SETTLE_TIMEOUT_SECONDS)
        if not acquired:
            if handover and on_accelerator:
                raise SegmenterReleaseTimeoutError(
                    "A mask from text is still running on the GPU; GPU ownership cannot be transferred yet."
                )
            return False
        try:
            with self._state_lock:
                self._release_pending = False
            if park:
                return self._park_locked(reason)
            dropped = self._drop_parked_locked(reason)
            return self._unload_locked(reason) or dropped
        finally:
            self._run_lock.release()

    @contextmanager
    def hold(self) -> Iterator[None]:
        with self._state_lock:
            self._holds += 1
        try:
            yield
        finally:
            with self._state_lock:
                self._holds -= 1

    def _arm_keep_alive(self) -> None:
        with self._state_lock:
            if self._timer is not None:
                self._timer.cancel()
            epoch = self._release_epoch
            timer = threading.Timer(self.keep_alive_seconds, self._expire, args = (epoch,))
            timer.daemon = True
            self._timer = timer
            timer.start()

    def _expire(self, epoch: int) -> None:
        if not self._run_lock.acquire(blocking = False):
            return  # in use: the call re-arms the keep-alive when it ends
        try:
            with self._state_lock:
                if epoch != self._release_epoch and self._model is None:
                    return
            self._unload_locked("keep-alive expired")
        finally:
            self._run_lock.release()

    def _after_call(self) -> None:
        """Called with ``_run_lock`` held: honour a release that arrived mid-call, drop a GPU copy when the card is
        tight, else keep it for the keep-alive."""
        with self._state_lock:
            pending, self._release_pending = self._release_pending, False
            self._call_settled = True
        if pending:
            self._unload_locked("released during a call")
            return
        with self._state_lock:
            held = self._holds > 0
        if held:
            return  # a batch pass: its holder parks or unloads once, after the last image
        if self.device and self.device.startswith("cuda"):
            free_mib, total_mib = _cuda_free_mib(int(self.device.split(":", 1)[1]))
            if vram_is_tight(free_mib, total_mib):
                self._unload_locked(f"tight: {free_mib} of {total_mib} MiB free")
                return
        if self.keep_alive_seconds <= 0:
            self._unload_locked("no keep-alive")
            return
        self._arm_keep_alive()

    # -- inference -------------------------------------------------------------------------------------------------
    def segment(
        self,
        image: Any,
        prompt: str,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        allow_gpu: bool = True,
        path: Optional[str] = None,
        instance_masks: bool = False,
        plane_filter: Optional[Callable[[tuple[float, float, float, float]], bool]] = None,
        max_planes: Optional[int] = None,
    ) -> SegmentResult:
        """With ``plane_filter`` / ``max_planes``, only those planes leave the device and no union mask is built."""
        concepts = parse_prompt(prompt)
        if not concepts:
            raise ValueError("Describe what to mask, for example: sofa, red lamp, pillow:2")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        if not transformers_supports_sam3():
            raise SegmenterUnavailableError(
                "This install's transformers has no SAM 3. It needs transformers 5 (Python 3.10 or newer)."
            )
        self._use_cache = path is None and not (os.environ.get(ENV_PATH_OVERRIDE) or "").strip()
        path = path or model_dir()
        if path is None:
            variant = active_variant()
            raise SegmenterNotDownloadedError(
                f"The {variant.label} segmenter is not downloaded. Download {variant.repo} first."
            )
        queued_at = self._handover_epoch
        with self._run_lock:
            if allow_gpu and (self._handover_epoch != queued_at or _gpu_handover_in_progress()):
                # The GPU changed hands while this call waited, or is changing hands now: the new owner may still be
                # loading onto the card.
                raise SegmenterBusyError("The GPU is changing hands.")
            with self._state_lock:
                self._call_device, self._call_settled = "", False
            try:
                return self._segment_locked(
                    image,
                    concepts,
                    threshold,
                    path,
                    allow_gpu,
                    instance_masks,
                    plane_filter,
                    max_planes,
                )
            finally:
                with self._state_lock:
                    self._call_device, self._call_settled = None, False

    def _segment_locked(
        self,
        image: Any,
        concepts: list[Concept],
        threshold: float,
        path: str,
        allow_gpu: bool,
        instance_masks: bool = False,
        plane_filter: Optional[Callable[[tuple[float, float, float, float]], bool]] = None,
        max_planes: Optional[int] = None,
    ) -> SegmentResult:
        started = time.perf_counter()
        device, dtype = choose_device(allow_gpu)
        self._call_device = device
        extra: dict = {"instance_masks": True} if instance_masks else {}
        if instance_masks and (plane_filter is not None or max_planes is not None):
            extra.update(plane_filter = plane_filter, max_planes = max_planes)
        oom = False
        try:
            result = self._segment_on(image, concepts, threshold, path, device, dtype, **extra)
        except Exception as exc:
            if not _is_oom(exc) or device == "cpu":
                self._after_call()
                raise
            oom = True
        if oom:
            # Out of the except block: the OOM's traceback (and the frames pinning the failed attempt's tensors) is
            # gone, so the unload and the cache release below actually free the card before the call reads CPU-only.
            logger.warning("images.segment: out of memory on %s, retrying on the CPU", device)
            self._unload_locked("out of memory")
            _free_device_cache(device)  # also when the OOM hit mid-load, before any model was kept
            import torch

            device, dtype = "cpu", torch.float32
            self._call_device = device
            try:
                result = self._segment_on(image, concepts, threshold, path, device, dtype, **extra)
            except Exception:
                # The CPU copy and any release asked for mid-call are settled here too, not left resident.
                self._after_call()
                raise
        result.elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._after_call()
        return result

    def _segment_on(
        self,
        image: Any,
        concepts: list[Concept],
        threshold: float,
        path: str,
        device: str,
        dtype: Any,
        instance_masks: bool = False,
        plane_filter: Optional[Callable[[tuple[float, float, float, float]], bool]] = None,
        max_planes: Optional[int] = None,
    ) -> SegmentResult:
        import torch

        self._load(path, device, dtype)
        model, tokenizer = self._model, self._tokenizer
        width, height = image.size
        scale = torch.tensor([width, height, width, height], dtype = torch.float32)
        budgeted = instance_masks and (plane_filter is not None or max_planes is not None)
        slots = 0
        union = None if budgeted else torch.zeros(height, width, dtype = torch.bool, device = device)
        instances: list[Instance] = []
        logits_out: list = []
        summary: list[dict] = []
        with torch.inference_mode():
            pixels = preprocess(image, device, dtype, model_image_size(model))
            vision = model.get_vision_features(pixel_values = pixels)
            # Duplicates are encoded once; each concept keeps its own cap, as ComfyUI runs them one by one.
            texts = list(dict.fromkeys(c.text for c in concepts))
            tokens = tokenizer(
                texts,
                padding = "max_length",
                max_length = TEXT_MAX_LENGTH,
                truncation = True,
                return_tensors = "pt",
            ).to(device)
            # Only each text's kept detections (best first, up to its largest cap) outlive its forward, never the raw
            # output's 200 mask planes; a repeat with a smaller cap takes the best of them, as its own run would.
            caps: dict[str, int] = {}
            for concept in concepts:
                caps[concept.text] = max(caps.get(concept.text, 0), concept.max_detections)
            # Per text only its kept scores and boxes outlive the forward: the union of a text's best-``cap`` masks
            # covers every occurrence's best-``k`` (k <= cap), so the masks are OR-ed in at once and dropped.
            kept: dict[str, tuple[Any, Any, list]] = {}
            for concept in concepts:
                if concept.text not in kept:
                    index = texts.index(concept.text)
                    out = model(
                        vision_embeds = vision,
                        input_ids = tokens["input_ids"][index : index + 1],
                        attention_mask = tokens["attention_mask"][index : index + 1],
                    )
                    presence = getattr(out, "presence_logits", None)
                    scores, boxes, masks = select_detections(
                        out.pred_logits[0],
                        out.pred_boxes[0],
                        out.pred_masks[0],
                        max_detections = caps[concept.text],
                        threshold = threshold,
                        presence_logit = None if presence is None else presence[0],
                    )
                    del out, presence
                    if budgeted:
                        kept[concept.text] = (scores, boxes, (masks, {}))
                    else:
                        union |= union_mask(masks, height, width)
                        planes = [m.float().cpu() for m in masks] if instance_masks else []
                        del masks
                        kept[concept.text] = (scores, boxes, planes)
                scores, boxes, planes = kept[concept.text]
                scores, boxes = scores[: concept.max_detections], boxes[: concept.max_detections]
                if not budgeted:
                    logits_out.extend(planes[: concept.max_detections])
                boxes_px = (boxes.float().cpu() * scale).tolist()
                for j, (score, box) in enumerate(zip(scores.tolist(), boxes_px)):
                    x1, y1, x2, y2 = (float(v) for v in box)
                    clipped = (
                        max(0.0, x1),
                        max(0.0, y1),
                        min(float(width), x2),
                        min(float(height), y2),
                    )
                    instances.append(Instance(concept.text, float(score), clipped))
                    if budgeted:
                        device_masks, moved = planes
                        plane = None
                        if (max_planes is None or slots < max_planes) and (
                            plane_filter is None or plane_filter(clipped)
                        ):
                            slots += 1
                            if j not in moved:
                                moved[j] = device_masks[j].float().cpu()
                            plane = moved[j]
                        logits_out.append(plane)
                summary.append(
                    {
                        "text": concept.text,
                        "max_detections": concept.max_detections,
                        "found": int(scores.shape[0]),
                    }
                )
            del kept, vision, pixels
        return SegmentResult(
            mask_png = b"" if union is None else mask_to_png(union),
            width = width,
            height = height,
            instances = instances,
            concepts = summary,
            device = device,
            precision = self.precision,
            instance_logits = logits_out,
        )


# How long a timed-out release waits for a call that already settled (unload / keep-alive, then the lock) to let go.
_SETTLE_TIMEOUT_SECONDS = 5.0


def _gpu_handover_in_progress() -> bool:
    """The arbiter is handing the GPU to a new owner (its load still registering). Never imports the arbiter."""
    arbiter = sys.modules.get("core.inference.gpu_arbiter")
    probe = getattr(arbiter, "handover_in_progress", None)
    try:
        return bool(probe()) if callable(probe) else False
    except Exception:  # noqa: BLE001
        return False


def _is_oom(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name in ("OutOfMemoryError",) or "out of memory" in str(exc).lower()


_segmenter: Optional[MaskSegmenter] = None
_segmenter_lock = threading.Lock()


def get_segmenter() -> MaskSegmenter:
    global _segmenter
    with _segmenter_lock:
        if _segmenter is None:
            _segmenter = MaskSegmenter()
        return _segmenter


def release_segmenter(
    reason: str = "release",
    *,
    wait: bool = False,
    handover: bool = False,
    park: bool = False,
) -> bool:
    """Drops the weights if loaded; never creates the segmenter. See MaskSegmenter.release."""
    segmenter = _segmenter
    if segmenter is None:
        return False
    if park:
        return segmenter.release(reason, wait = wait, handover = handover, park = True)
    return segmenter.release(reason, wait = wait, handover = handover)


_renders_pending = 0
_renders_lock = threading.Lock()


@contextmanager
def render_fence() -> Iterator[None]:
    """Held by an image render from before it releases the segmenter until it ends: no Find starts meanwhile."""
    global _renders_pending
    with _renders_lock:
        _renders_pending += 1
    try:
        yield
    finally:
        with _renders_lock:
            _renders_pending -= 1


def _take_render_slot() -> tuple[Any, str]:
    """(the active image engine's generation lock, acquired) or (None, why not). ``generating`` while a render or
    a model transition holds it, ``loading`` while a load is in flight; an engine with no lock (or none readable)
    runs nothing on torch, so the call proceeds with no slot. Never blocks."""
    try:
        from core.inference.diffusion_engine_router import get_active_diffusion_engine
        engine = get_active_diffusion_engine()
    except Exception:  # noqa: BLE001 -- no engine: nothing to collide with
        return None, ""
    loading = getattr(engine, "_loading", None)
    if loading is not None and not getattr(loading, "error", None):
        return None, "loading"
    lock = getattr(engine, "_generate_lock", None)
    if lock is None:
        return None, ""
    if not lock.acquire(blocking = False):
        return None, "generating"
    loading = getattr(engine, "_loading", None)
    if loading is not None and not getattr(loading, "error", None):
        lock.release()
        return None, "loading"
    return lock, ""


def segment_image(
    image_b64: str,
    prompt: str,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    allow_gpu: bool = True,
) -> dict:
    """The segment route's work: decode the upload (EXIF-upright, as inpaint decodes it), refuse while the image
    model renders or loads, run SAM 3, answer the route body."""
    from core.inference.diffusion import decode_b64_image

    if not parse_prompt(prompt):
        raise ValueError("Describe what to mask, for example: sofa, red lamp, pillow:2")
    image = decode_b64_image(image_b64, mode = "RGB")
    slot, why = _take_render_slot()
    if not why and _renders_pending:
        # A render released SAM 3 and is on its way to the slot: a Find taking it first would reload SAM 3 under it.
        if slot is not None:
            slot.release()
        slot, why = None, "generating"
    if why:
        raise SegmenterBusyError(
            "The image model is generating."
            if why == "generating"
            else "An image model is loading."
        )
    try:
        return (
            get_segmenter()
            .segment(image, prompt, threshold = threshold, allow_gpu = allow_gpu)
            .to_json()
        )
    finally:
        if slot is not None:
            slot.release()


__all__ = [
    "Concept",
    "DEFAULT_THRESHOLD",
    "DEFAULT_VARIANT",
    "MaskSegmenter",
    "SEGMENTER_VARIANTS",
    "SegmenterVariant",
    "SegmenterBusyError",
    "SegmenterNotDownloadedError",
    "SegmenterReleaseTimeoutError",
    "SegmenterUnavailableError",
    "active_variant",
    "get_segmenter",
    "parse_prompt",
    "release_segmenter",
    "segment_image",
    "status",
]
