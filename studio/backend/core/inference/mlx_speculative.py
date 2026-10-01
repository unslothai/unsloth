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
MLX_SPEC_MODES = frozenset(
    {"auto", "off", "ngram", *MLX_DRAFTER_KINDS, *(f"{kind}+ngram" for kind in MLX_DRAFTER_KINDS)}
)
_LEGACY_MODES = {
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
# Auto's order among cached companions of one target.
_COMPANION_ORDER = ("dflash2", "dspark", "dflash", "eagle3", "mtp")

# spec_fallback_reason codes an MLX load reports.
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
    kind: str  # as reported: mtp, dflash, dspark or eagle3
    path: str
    builtin: bool


@dataclass(frozen = True)
class SpecResolution:
    mode: str
    sources: tuple = ()  # DrafterSource candidates, tried in order at load
    copies: bool = False  # n-gram copies alongside a drafter, or alone when no source loads
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
_WEIGHT_WORDS = re.compile(
    r"^(mlx|bf16|fp16|fp8|mxfp4|nvfp4|qat|unquantized|q\d(_\d)?|\d+bit|dwq)$"
)


def _stem(name: str) -> str:
    """A repo or directory name without its organization, drafter and weight-format words, as alphanumerics."""
    words = re.split(r"[-_]", os.path.basename(str(name).rstrip("/")).lower())
    kept = [word for word in words if word not in _DRAFTER_WORDS and not _WEIGHT_WORDS.match(word)]
    return re.sub(r"[^a-z0-9]", "", "".join(kept).replace("speculator.eagle3", ""))


def _vocab_size(config: dict) -> Optional[int]:
    for scope in (
        config,
        config.get("text_config") or {},
        config.get("transformer_layer_config") or {},
    ):
        if isinstance(scope, dict) and isinstance(scope.get("vocab_size"), int):
            return scope["vocab_size"]
    return None


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


def discover_companions(target_name: str, target_config: dict) -> list:
    """Cached companion drafters whose repo names ``target_name``'s model and whose vocabulary matches, in auto's order."""
    from utils.utils import hf_cache_snapshot_dir_for_repo

    stem, vocab, found = _stem(target_name), _vocab_size(target_config), []
    if not stem:
        return found
    for repo in _cached_repos():
        if _stem(repo) != stem or repo.lower() == str(target_name).lower():
            continue
        snapshot = hf_cache_snapshot_dir_for_repo(repo)
        config = _read_config(snapshot) if snapshot is not None else {}
        kind = companion_kind(config)
        if kind is None or (vocab is not None and _vocab_size(config) not in (None, vocab)):
            continue
        found.append(
            (
                _COMPANION_ORDER.index(kind),
                repo,
                DrafterSource(_reported(kind), str(snapshot), False),
            )
        )
    return [source for _, _, source in sorted(found)]


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
    speculative_type, spec_draft_model: Optional[str], *, model_dir: Optional[str], target_name: str
) -> SpecResolution:
    """The drafters an MLX load tries: a named ``spec_draft_model``, then cached companions (any kind under auto) and the built-in head, in ``_COMPANION_ORDER``."""
    mode = mlx_spec_mode(speculative_type)
    if mode == "off":
        return SpecResolution(mode)
    if mode == "ngram":
        return SpecResolution(mode, copies = True)
    auto = mode == "auto"
    kind, copies = ("mtp", True) if auto else (mode.split("+")[0], mode.endswith("+ngram"))
    sources, reason = [], None
    if spec_draft_model:
        named = _named_companion(spec_draft_model)
        if named is None:
            reason = DRAFTER_NOT_FOUND
        elif named.kind != kind and not auto:
            reason = DRAFTER_INCOMPATIBLE
        else:
            sources.append(named)
    companions = discover_companions(target_name, _read_config(model_dir) if model_dir else {})
    companions = [
        source for source in companions if (auto or source.kind == kind) and source not in sources
    ]
    heads = next(
        (at for at, source in enumerate(companions) if source.kind == "mtp"), len(companions)
    )
    if kind == "mtp" and has_builtin_head(model_dir):
        companions.insert(heads, DrafterSource("mtp", str(model_dir), True))
    sources += companions
    if not sources and auto:
        return SpecResolution(mode, reason = reason)
    if not sources and reason is None:
        reason = DRAFTER_NOT_FOUND
    return SpecResolution(mode, tuple(sources), copies = copies, reason = reason)


def speculates_on_route(mode: str, vision: bool) -> bool:
    """Whether a load in ``mode`` looks for a drafter: auto only where mlx-vlm serves it anyway."""
    return mode != "off" and (mode != "auto" or vision)


_MAX_DEPTH = (
    8  # a drafter without its own block size; the controller picks each round's depth below it
)
_MAX_COPY = 16


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
    from unsloth_zoo.mlx.speculative import DraftController, install_speculative_seam, SpeculativeDraft

    depth = 0 if drafter is None else int(getattr(drafter, "max_depth", _MAX_DEPTH))
    copy = _MAX_COPY
    if draft_n_max:
        depth, copy = min(depth, draft_n_max), min(copy, draft_n_max)
    install_speculative_seam()
    return SpeculativeDraft(
        DraftController(max_depth = depth, max_copy = copy, can_copy = copies), drafter
    )


def _carries_encoder_state(target) -> bool:
    # generate_step threads these from forward to forward (Mllama, Florence2, Nemotron Parse); the
    # speculative rounds refuse them only after prefill, too late to decode the reply unspeculated.
    for module in (target, getattr(target, "language_model", None)):
        try:
            parameters = inspect.signature(module).parameters
        except (TypeError, ValueError):  # e.g. a language model its wrapper calls piecewise
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
    """``(draft, kind, reason, context)``: the first source whose drafter-inclusive fit ``fits(source)``
    accepts and that builds against ``target``, else copies alone when the mode allows them. ``fits``
    returns ``(ok, fitted context or None)``, ``ok`` None when nothing could be priced. The reason names why an earlier choice was passed over,
    so a substitute never attaches silently."""
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
