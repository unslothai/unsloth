# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which drafter an MLX load speculates with: ``speculative_type`` and ``spec_draft_model`` resolved
against the target's own MTP head and the companion drafters already in the local Hugging Face cache."""

from __future__ import annotations

import inspect
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)

MLX_DRAFTER_KINDS = ("mtp", "dflash", "dspark", "eagle3")
MLX_SPEC_MODES = frozenset({"auto", "off", "ngram", *MLX_DRAFTER_KINDS})
_LEGACY_MODES = {
    "mtp+ngram": "mtp",
    "default": "auto",
    "draft-mtp": "mtp",
    "draft-dspark": "dspark",
    "draft-dflash": "dflash",
    "ngram-mod": "ngram",
    "ngram-simple": "ngram",
    "none": "off",
    "disable": "off",
    "disabled": "off",
}
_COMPANION_ORDER = ("mtp", "dflash2", "dflash", "dspark", "eagle3")

DRAFTER_NOT_FOUND = "drafter_not_found"
DRAFTER_INCOMPATIBLE = "drafter_incompatible"
DRAFTER_NO_MEMORY = "drafter_no_memory"
RUNTIME_ERROR = "runtime_error"
KV_QUANT = "kv_quant"
AUTO_CONTEXT_COST = "auto_context_cost"


def mlx_spec_mode(value) -> str:
    """``speculative_type`` as an MLX load reads it: absent or unknown is ``auto``, as on llama.cpp."""
    if not isinstance(value, str) or not value.strip():
        return "auto"
    mode = value.strip().lower()
    mode = _LEGACY_MODES.get(mode, mode)
    return mode if mode in MLX_SPEC_MODES else "auto"


@dataclass(frozen = True)
class DrafterSource:
    kind: str
    path: str
    builtin: bool


@dataclass(frozen = True)
class SpecResolution:
    mode: str
    sources: tuple = ()
    copies: bool = False  # n-gram copies alongside a drafter, or alone when an explicit kind's drafter does not attach
    reason: Optional[str] = None

    @property
    def speculative(self) -> bool:
        return bool(self.sources) or self.copies


def _read_config(path) -> dict:
    try:
        with open(os.path.join(path, "config.json"), encoding = "utf-8") as handle:
            config = json.load(handle)
        return config if isinstance(config, dict) else {}
    except (OSError, ValueError):
        return {}


def companion_kind(config: dict) -> Optional[str]:
    """``dflash2``, ``dflash``, ``dspark``, ``eagle3`` or ``mtp`` (a separate MTP head or a Gemma assistant) for a companion drafter config."""
    # mlx-vlm executes a checkpoint's own model_file on load; a drafter never gets the target's remote-code grant.
    if config.get("model_file"):
        return None
    architectures = " ".join(str(name) for name in config.get("architectures") or ())
    # mlx-vlm loads EAGLE-3 only in the speculators format, not e.g. llama-typed Eagle3 exports.
    if "eagle3" in (config.get("model_type"), config.get("speculators_model_type")):
        return "eagle3"
    model_type = str(config.get("model_type", ""))
    if model_type.endswith("_mtp") or (
        model_type.endswith("_assistant") and "Assistant" in architectures
    ):
        return "mtp"
    if "dflash_config" in config:
        return (
            "dspark"
            if "DSpark" in architectures
            else "dflash2"
            if "DFlash2" in architectures
            else "dflash"
        )
    return None


def _reported(kind: str) -> str:
    return "dflash" if kind == "dflash2" else kind


_DRAFTER_WORDS = frozenset(
    {"dflash", "dflash2", "dspark", "eagle3", "speculator", "assistant", "mtp", "drafter"}
)
_WEIGHT_WORDS = re.compile(r"^(mlx|bf16|fp16|fp8|mxfp4|nvfp4|qat|unquantized|\d+bit|dwq)$")
_GGUF_QUANT = re.compile(r"(?<![a-z0-9])q\d(_[a-z0-9]+)+(?![a-z0-9])")


def _alnum(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", os.path.basename(str(name).rstrip("/")).lower())


def _stem(name: str) -> str:
    """A drafter repo's name without its organization, drafter and weight-format words, as alphanumerics."""
    base = _GGUF_QUANT.sub("", os.path.basename(str(name).rstrip("/")).lower())
    words = re.split(r"[-_]", base.replace("speculator.eagle3", ""))
    return _alnum(
        "".join(w for w in words if w not in _DRAFTER_WORDS and not _WEIGHT_WORDS.match(w))
    )


def _text(config: dict) -> dict:
    scope = config.get("text_config")
    return scope if isinstance(scope, dict) else config


def _vocab_size(config: dict) -> Optional[int]:
    for scope in (
        config,
        config.get("text_config") or {},
        config.get("transformer_layer_config") or {},
    ):
        if isinstance(scope, dict) and isinstance(scope.get("vocab_size"), int):
            return scope["vocab_size"]
    return None


def _fits(kind: str, config: dict, target: dict) -> bool:
    """Whether a drafter's config is built for a target of this shape. A dimension either side does not state cannot refuse."""
    text, own = _text(target), _text(config)
    layers, flash = text.get("num_hidden_layers"), config.get("dflash_config") or {}
    if kind == "eagle3":
        wants = config.get("target_hidden_size") or (
            config.get("transformer_layer_config") or {}
        ).get("hidden_size")
        taps = config.get("eagle_aux_hidden_state_layer_ids") or flash.get("target_layer_ids")
    elif kind == "mtp":
        wants = config.get("backbone_hidden_size") or own.get("hidden_size")
        taps = None
        if "backbone_hidden_size" not in config:
            pairs = [
                (own.get("num_hidden_layers"), layers),
                (own.get("model_type"), text.get("model_type")),
            ]
            if any(a is not None and b is not None and a != b for a, b in pairs):
                return False
    else:
        wants = config.get("target_hidden_size") or config.get("hidden_size")
        taps = flash.get("target_layer_ids") or config.get("target_layer_ids")
        stated = config.get("num_target_layers") or flash.get("num_target_layers")
        if stated is not None and layers is not None and stated != layers:
            return False
    vocab = _vocab_size(target)
    return not (
        (wants is not None and text.get("hidden_size") not in (None, wants))
        # Families number tapped layers from 0 or from 1; only an id past both is out of range.
        or (taps and layers is not None and max(taps) > layers)
        or (vocab is not None and _vocab_size(config) not in (None, vocab))
    )


def _has_weights(snapshot) -> bool:
    try:
        return any(name.endswith(".safetensors") for name in os.listdir(snapshot))
    except OSError:
        return False


def _cached_repos():
    from utils.utils import _hf_cache_roots
    seen = set()
    for root in _hf_cache_roots():
        try:
            entries = sorted(os.listdir(root))
        except OSError:
            continue
        for entry in entries:
            if entry.startswith("models--") and entry not in seen:
                seen.add(entry)
                yield entry[len("models--") :].replace("--", "/")


def cached_drafters(target_name: str, target_config: dict) -> list:
    """Drafters named for the target come first, then same-architecture ones that fit by shape alone."""
    from utils.utils import hf_cache_snapshot_dir_for_repo

    target, found = _alnum(target_name), []
    for repo in _cached_repos():
        if repo.lower() == str(target_name).lower():
            continue
        snapshot = hf_cache_snapshot_dir_for_repo(repo)
        config = _read_config(snapshot) if snapshot is not None else {}
        kind = companion_kind(config)
        if kind is None or not _fits(kind, config, target_config) or not _has_weights(snapshot):
            continue
        stem = _stem(repo)
        named = bool(stem) and target.startswith(stem)
        source = DrafterSource(_reported(kind), str(snapshot), False)
        found.append((not named, _COMPANION_ORDER.index(kind), repo, source))
    return [(repo, source, not unnamed) for unnamed, _, repo, source in sorted(found)]


def discover_companions(
    target_name: str,
    target_config: dict,
    allowed = None,
) -> list:
    """The cached drafters auto attaches: those named for the target, within ``allowed`` repos when given."""
    found = cached_drafters(target_name, target_config)
    return [
        source for repo, source, named in found if named and (allowed is None or repo in allowed)
    ]


def _named_companion(spec_draft_model: str) -> Optional[DrafterSource]:
    from utils.utils import hf_cache_snapshot_dir_for_repo

    path = Path(os.path.expanduser(spec_draft_model))
    snapshot = path if path.is_dir() else hf_cache_snapshot_dir_for_repo(spec_draft_model)
    kind = companion_kind(_read_config(snapshot)) if snapshot is not None else None
    return None if kind is None else DrafterSource(_reported(kind), str(snapshot), False)


def has_builtin_head(model_dir: Optional[str]) -> bool:
    if not model_dir:
        return False
    try:
        from mlx_vlm.speculative.drafters.mtp_split import detect_mtp_splitter
        return detect_mtp_splitter(Path(model_dir)) is not None
    except Exception as exc:
        logger.debug("MLX speculative: no MTP splitter check for %s: %s", model_dir, exc)
        return False


def resolve_speculation(
    speculative_type,
    spec_draft_model: Optional[str],
    *,
    model_dir: Optional[str],
    target_name: str,
    allowed = None,
) -> SpecResolution:
    """The drafters an MLX load tries: a named ``spec_draft_model``, then cached companions (any kind under auto) and the built-in head, in ``_COMPANION_ORDER``."""
    mode = mlx_spec_mode(speculative_type)
    if mode == "off":
        return SpecResolution(mode)
    if mode == "ngram":
        return SpecResolution(mode, copies = True)
    auto = mode == "auto"
    kind = "mtp" if auto else mode
    sources, reason = [], None
    if spec_draft_model:
        named = _named_companion(spec_draft_model)
        if named is None:
            reason = DRAFTER_NOT_FOUND
        elif named.kind != kind and not auto:
            reason = DRAFTER_INCOMPATIBLE
        else:
            sources.append(named)
    companions = discover_companions(
        target_name, _read_config(model_dir) if model_dir else {}, allowed
    )
    companions = [
        source for source in companions if (auto or source.kind == kind) and source not in sources
    ]
    if kind == "mtp" and has_builtin_head(model_dir):
        companions.insert(0, DrafterSource("mtp", str(model_dir), True))
    sources += companions
    if not sources and auto:
        return SpecResolution(mode, reason = reason)
    if not sources and reason is None:
        reason = DRAFTER_NOT_FOUND
    return SpecResolution(mode, tuple(sources), copies = True, reason = reason)


def speculates_on_route(
    mode: str,
    vision: bool,
    named = None,
) -> bool:
    """Whether a load in ``mode`` looks for a drafter: auto only where mlx-vlm serves it anyway, or for a drafter the request names."""
    return mode != "off" and (mode != "auto" or vision or bool(named))


_MAX_DEPTH = (
    8  # a drafter without its own block size; the controller picks each round's depth below it
)
_MAX_COPY = 16
_EXACT_HEAD_DEPTH = 3  # MTP heads and assistants chain a forward per drafted token


def speculation_refusal(*, kv_quant: bool, distributed: bool, lora: bool) -> Optional[str]:
    """The reason an MLX load cannot speculate at all, or None."""
    if kv_quant:
        return KV_QUANT
    if distributed or lora:
        return RUNTIME_ERROR
    try:
        from unsloth_zoo.mlx.speculative import speculative_unavailable_reason
    except ImportError as exc:
        logger.info("MLX speculative decoding unavailable: %s", exc)
        return RUNTIME_ERROR
    unavailable = speculative_unavailable_reason()
    if unavailable is not None:
        logger.info("MLX speculative decoding unavailable: %s", unavailable)
        return RUNTIME_ERROR
    return None


def _draft(drafter, copies: bool, draft_n_max: Optional[int]):
    from unsloth_zoo.mlx.speculative import (
        DraftController,
        install_speculative_seam,
        SpeculativeDraft,
    )

    depth = 0 if drafter is None else int(getattr(drafter, "max_depth", _MAX_DEPTH))
    copy, exact = _MAX_COPY, False
    if draft_n_max:
        exact = draft_n_max <= getattr(drafter, "max_depth", _EXACT_HEAD_DEPTH)
        depth, copy = min(depth, draft_n_max), min(copy, draft_n_max)
    install_speculative_seam()
    return SpeculativeDraft(
        DraftController(max_depth = depth, max_copy = copy, can_copy = copies, fixed_depth = exact), drafter
    )


def _carries_encoder_state(target) -> bool:
    # generate_step threads these from forward to forward (Mllama, Florence2, Nemotron Parse); the
    # speculative rounds refuse them only after prefill, too late to decode the reply unspeculated.
    for module in (target, getattr(target, "language_model", None)):
        try:
            parameters = inspect.signature(module).parameters
        except (TypeError, ValueError):
            continue
        if "cross_attention_states" in parameters or "encoder_outputs" in parameters:
            return True
    return False


def build_draft(
    target,
    resolution: SpecResolution,
    *,
    fits,
    draft_n_max: Optional[int] = None,
) -> tuple:
    """``(draft, kind, reason, context)`` for the first source ``fits`` accepts that builds, else copies
    alone when the mode allows. ``reason`` names a passed-over choice, so a substitute never attaches silently."""
    from unsloth_zoo.mlx.speculative import companion_drafter, native_mtp_drafter

    if _carries_encoder_state(target):
        logger.info("MLX speculative decoding: %s carries encoder state", type(target).__name__)
        return None, None, None if resolution.mode == "auto" else RUNTIME_ERROR, None
    reason = resolution.reason
    for source in resolution.sources:
        ok, context = fits(source)
        if not ok:
            if resolution.mode != "auto":
                reason = DRAFTER_NO_MEMORY
            elif ok is not None:  # priced, and the drafter would shrink it
                reason = AUTO_CONTEXT_COST
            continue
        try:
            drafter = (native_mtp_drafter if source.builtin else companion_drafter)(
                source.path, target
            )
            if drafter is None:
                raise ValueError("no MTP splitter knows this checkpoint")
        except Exception as exc:
            logger.warning("MLX drafter %s (%s) not attached: %s", source.path, source.kind, exc)
            reason = DRAFTER_INCOMPATIBLE
            continue
        return _draft(drafter, resolution.copies, draft_n_max), source.kind, reason, context
    if resolution.copies and resolution.mode != "auto":  # auto copies only alongside a drafter
        return _draft(None, True, draft_n_max), "ngram", reason, None
    return None, None, reason, None
