# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Discovery, recommendation, and preflight policy for MLX speculative decoding."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version
import inspect
import itertools
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Optional


MLX_SPECULATIVE_METHODS = frozenset({"mtp", "dflash", "eagle3", "dspark", "dflash2"})

# DSpark and DFlash2 are architectures over the DFlash loop rather than loops of their own, and
# mlx-vlm refuses any kind outside KNOWN_DRAFTER_KINDS, so only a kind from here may reach
# ``load_drafter``.
MLX_SPECULATIVE_DRAFT_KINDS: dict[str, str] = {
    "mtp": "mtp",
    "dflash": "dflash",
    "eagle3": "eagle3",
    "dspark": "dflash",
    "dflash2": "dflash",
}

# Methods carried by a drafter module a runtime can predate; absent means as old as the API.
_MLX_METHOD_MODULES: dict[str, str] = {
    "dspark": "mlx_vlm.speculative.drafters.dspark",
    "dflash2": "mlx_vlm.speculative.drafters.dflash2",
}
MLX_SPECULATIVE_MODES = MLX_SPECULATIVE_METHODS | {"auto"}

# Each method joins this set with the load path that can run it, so a request for a
# method the worker cannot execute is refused before the active model is torn down.
ENABLED_MLX_SPECULATIVE_METHODS: frozenset[str] = frozenset(
    {"mtp", "dflash", "eagle3", "dspark", "dflash2"}
)

# Refusals reach the client as prose, while the codes stay the vocabulary the response
# schema will use to say why a resolved method differs from the one requested.
MLX_SPECULATIVE_REFUSALS: dict[str, str] = {
    "method_not_integrated": "This build cannot run the requested MLX speculative decoding method.",
    "checkpoint_required": "Choose a draft checkpoint to use this MLX speculative decoding method.",
    "checkpoint_not_compatible": "The chosen draft checkpoint cannot pair with this model.",
    "checkpoint_config_mismatch": "The chosen draft checkpoint was built for a different model.",
    "checkpoint_not_downloaded": "The chosen draft checkpoint is not downloaded yet.",
    "runtime_unavailable": "The installed MLX runtime does not support speculative decoding.",
    "runtime_missing_speculative_api": "The installed mlx-vlm does not expose the speculative decoding API.",
    "mlx_requires_apple_silicon": "MLX speculative decoding needs Apple silicon.",
    "method_runtime_unavailable": "The installed MLX runtime cannot run this speculative decoding method.",
    "insufficient_unified_memory": "There is not enough unified memory for this model and its draft checkpoint together.",
    "target_weights_unmeasured": "This model is not downloaded yet, so the memory it needs with a draft checkpoint cannot be measured.",
    "target_config_unavailable": "This model's configuration could not be read.",
    "verifier_contract_unavailable": "The chosen draft checkpoint does not say which model it verifies.",
    "tokenizer_contract_unavailable": "The chosen draft checkpoint's tokenizer could not be compared with this model's.",
    "no_cached_drafter": "No draft checkpoint for this model is downloaded.",
    "checkpoint_quantization_unsupported": "This draft checkpoint is quantized in a format the MLX runtime cannot load.",
    "target_too_small_to_draft": "This model is small enough that drafting for it would not speed it up.",
    "auto_drafter_load_failed": "The draft checkpoint could not be loaded, so this model is running without speculative decoding.",
    "auto_preferred_candidate_unavailable": "The chosen draft checkpoint is not available, so this model runs without speculative decoding.",
    "mlx_vlm_target_required": "Speculative decoding runs on the MLX vision-language path, which this model does not use.",
    "mlx_speculative_lora_unsupported": "Speculative decoding does not run on a LoRA adapter.",
    "mlx_speculative_distributed_unsupported": "Speculative decoding does not run on a distributed load.",
}

# A code with no entry of its own still has to reach the client as a 400 rather than as the
# KeyError a subscript would raise.
MLX_SPECULATIVE_GENERIC_REFUSAL = "This model cannot use the requested speculative decoding method."


_RECOMMENDATION_TARGET_SHAPES = {
    ("qwen3_5", 2560, 32, None, 248320): frozenset({"qwen3.5-4b"}),
    ("qwen3_5", 4096, 32, None, 248320): frozenset({"qwen3.5-9b"}),
    ("qwen3_5", 5120, 64, None, 248320): frozenset({"qwen3.5-27b", "qwen3.6-27b", "qwen3.8-27b"}),
    ("qwen3_5_moe", 2048, 40, 256, 248320): frozenset({"qwen3.5-35b-a3b", "qwen3.6-35b-a3b"}),
    ("qwen3_5_moe", 3072, 48, 256, 248320): frozenset({"qwen3.5-122b-a10b"}),
    ("qwen3_5_moe", 4096, 60, 512, 248320): frozenset({"qwen3.5-397b-a17b"}),
    ("gemma4", 1536, 35, None, 262144): frozenset({"gemma4-e2b"}),
    ("gemma4", 2560, 42, None, 262144): frozenset({"gemma4-e4b"}),
    ("gemma4_unified", 3840, 48, None, 262144): frozenset({"gemma4-12b"}),
    ("gemma4", 2816, 30, 128, 262144): frozenset({"gemma4-26b-a4b"}),
    ("gemma4", 5376, 60, None, 262144): frozenset({"gemma4-31b"}),
    ("muse_glimmer", 6656, 52, None, 202048): frozenset({"muse-glimmer-30b"}),
    ("deepseek_v4", 4096, 43, 256, 129280): frozenset({"deepseek-v4-flash"}),
    ("lfm2", 2048, 30, None, 128000): frozenset({"lfm2.5-2.6b"}),
    ("lfm2_moe", 2048, 24, 32, 128000): frozenset({"lfm2.5-8b-a1b"}),
}


@dataclass(frozen = True)
class _RecommendationSeed:
    target_key: str
    method: str
    repo_id: str
    label: str
    approximate_size_bytes: int
    verifier_id: Optional[str] = None
    target_owner: Optional[str] = None
    requires_missing_native_mtp: bool = False


_RECOMMENDATIONS = (
    _RecommendationSeed(
        "qwen3.5-4b",
        "mtp",
        "mlx-community/Qwen3.5-4B-MTP-bf16",
        "Qwen 3.5 4B MTP",
        241_200_628,
        target_owner = "mlx-community",
        requires_missing_native_mtp = True,
    ),
    _RecommendationSeed(
        "qwen3.5-4b",
        "dflash",
        "z-lab/Qwen3.5-4B-DFlash",
        "Qwen 3.5 4B DFlash",
        1_268_859_081,
    ),
    _RecommendationSeed(
        "qwen3.5-9b",
        "mtp",
        "mlx-community/Qwen3.5-9B-MTP-bf16",
        "Qwen 3.5 9B MTP",
        486_582_779,
        target_owner = "mlx-community",
        requires_missing_native_mtp = True,
    ),
    _RecommendationSeed(
        "qwen3.5-9b",
        "dflash",
        "z-lab/Qwen3.5-9B-DFlash",
        "Qwen 3.5 9B DFlash",
        2_583_816_465,
    ),
    _RecommendationSeed(
        "qwen3.5-27b",
        "dflash",
        "z-lab/Qwen3.5-27B-DFlash",
        "Qwen 3.5 27B DFlash",
        4_257_372_459,
    ),
    _RecommendationSeed(
        "qwen3.5-35b-a3b",
        "dflash",
        "z-lab/Qwen3.5-35B-A3B-DFlash",
        "Qwen 3.5 35B-A3B DFlash",
        771_819_674,
    ),
    _RecommendationSeed(
        "qwen3.5-122b-a10b",
        "dflash",
        "z-lab/Qwen3.5-122B-A10B-DFlash",
        "Qwen 3.5 122B-A10B DFlash",
        1_547_794_655,
    ),
    _RecommendationSeed(
        "qwen3.5-397b-a17b",
        "dflash",
        "z-lab/Qwen3.5-397B-A17B-DFlash",
        "Qwen 3.5 397B-A17B DFlash",
        2_583_816_472,
    ),
    _RecommendationSeed(
        "qwen3.6-27b",
        "mtp",
        "mlx-community/Qwen3.6-27B-MTP-bf16",
        "Qwen 3.6 27B MTP",
        849_400_337,
        target_owner = "mlx-community",
        requires_missing_native_mtp = True,
    ),
    _RecommendationSeed(
        "qwen3.6-27b",
        "dflash",
        "z-lab/Qwen3.6-27B-DFlash",
        "Qwen 3.6 27B DFlash",
        3_460_432_504,
    ),
    _RecommendationSeed(
        "qwen3.6-35b-a3b",
        "mtp",
        "mlx-community/Qwen3.6-35B-A3B-MTP-bf16",
        "Qwen 3.6 35B-A3B MTP",
        1_689_283_752,
        target_owner = "mlx-community",
        requires_missing_native_mtp = True,
    ),
    _RecommendationSeed(
        "qwen3.6-35b-a3b",
        "dflash",
        "z-lab/Qwen3.6-35B-A3B-DFlash",
        "Qwen 3.6 35B-A3B DFlash",
        771_819_674,
    ),
    _RecommendationSeed(
        "qwen3.8-27b",
        "dflash2",
        "z-lab/Qwen3.8-27B-DFlash2",
        "Qwen 3.8 27B DFlash2",
        3_848_817_896,
    ),
    _RecommendationSeed(
        "qwen3.8-27b",
        "dspark",
        "RadixArk/Qwen3.8-27B-DSpark",
        "Qwen 3.8 27B DSpark",
        2_718_576_122,
    ),
    _RecommendationSeed(
        "qwen3.8-27b",
        "mtp",
        "mlx-community/Qwen3.8-27B-MTP-bf16",
        "Qwen 3.8 27B MTP",
        849_400_335,
        target_owner = "mlx-community",
        requires_missing_native_mtp = True,
    ),
    _RecommendationSeed(
        "gemma4-e2b",
        "mtp",
        "google/gemma-4-E2B-it-assistant",
        "Gemma 4 E2B MTP assistant",
        157_565_344,
    ),
    _RecommendationSeed(
        "gemma4-e4b",
        "mtp",
        "google/gemma-4-E4B-it-assistant",
        "Gemma 4 E4B MTP assistant",
        159_138_208,
    ),
    _RecommendationSeed(
        "gemma4-12b",
        "mtp",
        "google/gemma-4-12B-it-assistant",
        "Gemma 4 12B MTP assistant",
        845_719_296,
    ),
    _RecommendationSeed(
        "gemma4-12b",
        "dflash",
        "z-lab/gemma4-12B-it-DFlash",
        "Gemma 4 12B DFlash",
        1_455_000_120,
    ),
    _RecommendationSeed(
        "gemma4-26b-a4b",
        "mtp",
        "google/gemma-4-26B-A4B-it-assistant",
        "Gemma 4 26B-A4B MTP assistant",
        839_427_840,
    ),
    _RecommendationSeed(
        "gemma4-26b-a4b",
        "dflash",
        "z-lab/gemma-4-26B-A4B-it-DFlash",
        "Gemma 4 26B-A4B DFlash",
        859_384_328,
    ),
    _RecommendationSeed(
        "gemma4-26b-a4b",
        "eagle3",
        "RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3",
        "Gemma 4 26B-A4B EAGLE-3",
        1_855_768_160,
        "google/gemma-4-26b-a4b-it",
    ),
    _RecommendationSeed(
        "gemma4-31b",
        "mtp",
        "google/gemma-4-31B-it-assistant",
        "Gemma 4 31B MTP assistant",
        939_042_560,
    ),
    _RecommendationSeed(
        "gemma4-31b",
        "dflash",
        "z-lab/gemma-4-31B-it-DFlash",
        "Gemma 4 31B DFlash",
        3_071_941_240,
    ),
    _RecommendationSeed(
        "gemma4-31b",
        "eagle3",
        "RedHatAI/gemma-4-31B-it-speculator.eagle3",
        "Gemma 4 31B EAGLE-3",
        4_470_642_280,
        "google/gemma-4-31b-it",
    ),
    _RecommendationSeed(
        "muse-glimmer-30b",
        "dflash",
        "meta-models/Muse-Glimmer-30B-assistant",
        "Muse Glimmer 30B DFlash assistant",
        5_111_976_608,
    ),
    _RecommendationSeed(
        "deepseek-v4-flash",
        "mtp",
        "mlx-community/DeepSeek-V4-Flash-MTP-bf16",
        "DeepSeek V4 Flash MTP",
        3_598_959_572,
        target_owner = "mlx-community",
        requires_missing_native_mtp = True,
    ),
    _RecommendationSeed(
        "lfm2.5-2.6b",
        "dspark",
        "LiquidAI/LFM2.5-2.6B-DSpark",
        "LFM2.5 2.6B DSpark",
        655_421_522,
    ),
    _RecommendationSeed(
        "lfm2.5-8b-a1b",
        "dspark",
        "LiquidAI/LFM2.5-8B-A1B-DSpark",
        "LFM2.5 8B-A1B DSpark",
        655_421_522,
    ),
)


def mlx_speculative_refusal_text(reason: str) -> str:
    """The sentence for a refusal, as an error detail. Unknown reasons read generically."""
    return MLX_SPECULATIVE_REFUSALS.get(reason, MLX_SPECULATIVE_GENERIC_REFUSAL)


def mlx_speculative_reason_text(reason: Optional[str]) -> Optional[str]:
    """The sentence a client shows for a load that ended without a drafter.

    Auto never fails a load, so its outcomes arrive on a successful response rather than as
    an error detail.
    """
    return None if not reason else mlx_speculative_refusal_text(reason)


def normalize_mlx_speculative_mode(value: Any) -> str:
    mode = str(value or "off").strip().lower()
    return mode if mode in MLX_SPECULATIVE_MODES else "off"


def normalize_mlx_speculative_method(value: Any) -> str:
    """The concrete method ``value`` names, or "off". Auto is a request, not a method."""
    mode = normalize_mlx_speculative_mode(value)
    return mode if mode in MLX_SPECULATIVE_METHODS else "off"


def mlx_speculative_request_identity(
    mode: Any, draft_model: Optional[str], block_size: Optional[int]
) -> tuple[str, Optional[str], Optional[int]]:
    """The speculative settings a resident model must already have to be reused.

    Off carries no drafter, and a repository id differing only in case or surrounding
    space names the same checkpoint. Every place that decides whether two requests name
    the same drafter folds through here, so they cannot normalize differently, which
    reads as a settings change on every request and reloads the model forever.
    """
    normalized = normalize_mlx_speculative_mode(mode)
    if normalized == "off":
        return normalized, None, None
    named = draft_model.strip().casefold() if isinstance(draft_model, str) else None
    return normalized, named or None, block_size


_RUNTIME_CAPABILITIES: Optional[dict[str, Any]] = None


def mlx_speculative_runtime_capabilities() -> dict[str, Any]:
    """What the installed MLX stack can draft with.

    Self-heal installs and upgrades the stack inside this running process, so nothing it can
    still change is answered from before it ran. It is asked first, from distribution metadata:
    importing a stack it is about to replace pins the old modules in ``sys.modules``, where the
    upgraded ones can no longer be reached. Only a usable stack, or a platform that can never
    have one, is remembered.
    """
    global _RUNTIME_CAPABILITIES
    if _RUNTIME_CAPABILITIES is not None:
        return _RUNTIME_CAPABILITIES
    result: dict[str, Any] = {
        "common": False,
        "methods": {method: False for method in sorted(MLX_SPECULATIVE_METHODS)},
        "reason": "runtime_unavailable",
    }
    if sys.platform != "darwin":
        result["reason"] = "mlx_requires_apple_silicon"
        _RUNTIME_CAPABILITIES = result
        return result
    try:
        from utils.mlx_repair import mlx_stack_available

        if not mlx_stack_available():
            return result
        drafters = importlib.import_module("mlx_vlm.speculative.drafters")
        ar = importlib.import_module("mlx_vlm.generate.ar")
        utils = importlib.import_module("mlx_vlm.speculative.utils")
    except Exception:
        return result
    probed = _runtime_capabilities_from_modules(drafters, ar, utils)
    if probed["common"]:
        _RUNTIME_CAPABILITIES = probed
    return probed


def _runtime_capabilities_from_modules(drafters: Any, ar: Any, utils: Any) -> dict[str, Any]:
    signature = inspect.signature(ar.generate_step)
    common = (
        callable(getattr(drafters, "load_drafter", None))
        and callable(getattr(utils, "run_speculative_rounds", None))
        and {"draft_model", "draft_kind", "draft_block_size"}.issubset(signature.parameters)
    )
    known = set(getattr(drafters, "KNOWN_DRAFTER_KINDS", ()))
    methods = {}
    for method in sorted(MLX_SPECULATIVE_METHODS):
        kind = MLX_SPECULATIVE_DRAFT_KINDS[method]
        try:
            dispatch = utils.get_speculative_rounds_batch(kind)
        except Exception:
            dispatch = None
        available = bool(common and kind in known and callable(dispatch))
        module = _MLX_METHOD_MODULES.get(method)
        if available and module is not None:
            try:
                importlib.import_module(module)
            except Exception:
                available = False
        methods[method] = available
    result = {"common": common, "methods": methods}
    result["reason"] = None if common else "runtime_missing_speculative_api"
    return result


def _local_path(model_id: str) -> Optional[Path]:
    """A "~unknown-user" prefix has no home to expand into, and callers are asking
    whether a checkpoint sits on disk rather than asserting that it does.
    """
    try:
        return Path(model_id).expanduser()
    except RuntimeError:
        return None


def _public_target_model_id(target_id: str) -> str:
    from core.inference.model_ids import public_model_id

    local = _local_path(target_id)
    if local is None:
        return "local-model"
    if local.is_absolute() or local.exists() or target_id.startswith("."):
        return local.name or "local-model"
    return public_model_id(target_id) or "local-model"


_MATCH = "match"  # structurally verified against the target
_MISMATCH = "mismatch"  # structurally refuted
_INDETERMINATE = "indeterminate"  # contract unreadable, neither proved nor refuted
_UNVERIFIED = "unverified"  # source asserts no structural verdict (seeds)


class _CandidateRow:
    """One source's claim about one drafter repository. ``inherit`` names fields the
    merge must take from the row being replaced rather than from this one.
    """

    __slots__ = ("key", "status", "fields", "reason", "inherit")

    def __init__(
        self,
        key,
        status,
        fields,
        reason = None,
        inherit = (),
    ):
        self.key = key
        self.status = status
        self.fields = fields
        self.reason = reason
        self.inherit = inherit


def _merge_candidate_rows(rows):
    """One repository yields a row per cached revision. A verified match freezes it and an
    unreadable one blocks a later refutation, giving match > indeterminate > mismatch
    whatever order the snapshots are read in.
    """
    candidates: list[dict] = []
    index: dict[str, int] = {}
    frozen: set[str] = set()
    indeterminate: set[str] = set()

    for row in rows:
        key = row.key
        if key in frozen:
            continue
        existing = index.get(key)

        fields = dict(row.fields)
        if existing is not None:
            for field in row.inherit:
                fields[field] = candidates[existing].get(field, fields.get(field))

        # A refuted row says why a drafter another source offered does not fit; with
        # nothing to attach it to there is no candidate to describe, so it is dropped.
        if row.status == _MISMATCH:
            if key in indeterminate or existing is None:
                continue
            candidates[existing].update(fields, compatible = False, loadable = False, reason = row.reason)
            continue

        if row.status == _INDETERMINATE:
            indeterminate.add(key)
        elif row.status == _MATCH:
            frozen.add(key)

        if existing is None:
            index[key] = len(candidates)
            candidates.append({**fields, "reason": row.reason})
        else:
            candidates[existing] = {**fields, "reason": row.reason}

    return candidates


def _active_hf_cache_root() -> Optional[Path]:
    try:
        from utils.hf_cache_settings import get_hf_cache_paths
        return Path(get_hf_cache_paths().hub_cache)
    except Exception:
        return None


def _known_hf_cache_roots() -> list[Path]:
    """Every hub cache Studio has pointed at, which is the same set local model resolution
    trusts. Outside them a ``models--`` directory is a name anyone can write."""
    try:
        from utils.hf_cache_settings import known_hf_hub_caches
        roots = known_hf_hub_caches()
    except Exception:
        return []
    resolved = []
    for root in roots:
        try:
            resolved.append(root.resolve())
        except OSError:
            continue
    return resolved


def _cached_config_path(repo_id: str) -> Optional[Path]:
    path = _local_path(repo_id)
    if path is None:
        return None
    if path.is_dir() and (path / "config.json").is_file():
        return path / "config.json"
    if path.is_file() and path.name == "config.json":
        return path
    try:
        from huggingface_hub import try_to_load_from_cache
        root = _active_hf_cache_root()
        cached = try_to_load_from_cache(
            repo_id, "config.json", cache_dir = str(root) if root is not None else None
        )
    except Exception:
        return None
    return Path(cached) if isinstance(cached, str) else None


def _read_config(repo_id: str) -> Optional[dict[str, Any]]:
    path = _cached_config_path(repo_id)
    return _config_from_path(path) if path is not None else None


def mlx_target_config_is_cached(target_id: str) -> bool:
    """Whether this target's configuration is already on disk, so reading it needs no Hub call."""
    return _cached_config_path(target_id) is not None


def _snapshot_weight_bytes(repo_id: str) -> int:
    config_path = _cached_config_path(repo_id)
    if config_path is None:
        return 0
    return _snapshot_weight_bytes_at(config_path.parent)


def _snapshot_weight_bytes_at(snapshot: Path) -> int:
    try:
        return sum(
            path.stat().st_size
            for pattern in ("*.safetensors", "*.bin")
            for path in snapshot.glob(pattern)
            if path.is_file()
        )
    except OSError:
        return 0


def _snapshot_complete_at(snapshot: Path, *, require_tokenizer: bool = False) -> bool:
    try:
        has_weights = any(
            path.is_file()
            for pattern in ("*.safetensors", "*.bin")
            for path in snapshot.glob(pattern)
        )
        if not has_weights or not (snapshot / "config.json").is_file():
            return False
        if require_tokenizer and not (snapshot / "tokenizer.json").is_file():
            return False
        from hub.utils.inventory_scan import snapshot_holds_a_complete_payload

        return snapshot_holds_a_complete_payload(snapshot, quants = False)
    except (OSError, RuntimeError, ValueError):
        return False


# repository id, configuration, snapshot directory, weight bytes
_DrafterRows = tuple[tuple[str, dict[str, Any], Path, int], ...]


def _config_from_path(path: Path) -> Optional[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _scan_active_cached_drafter_configs(root: Path) -> Optional[_DrafterRows]:
    """The drafters under ``root``, or None where the cache itself could not be read."""
    try:
        from hub.utils.hf_cache_state import latest_snapshot_dir, ref_snapshot_dir

        # Listed rather than matched with a glob, which reports a directory it cannot read as one
        # holding nothing and would have this return an empty cache instead of no answer.
        repo_dirs = sorted(
            (path for path in root.iterdir() if path.name.startswith("models--")),
            key = lambda path: path.name.casefold(),
        )
    except (OSError, RuntimeError, ValueError):
        return None
    rows = []
    for repo_dir in repo_dirs:
        encoded = repo_dir.name.removeprefix("models--")
        if not encoded or "--" not in encoded:
            continue
        repo_id = encoded.replace("--", "/")
        snapshots = []
        preferred = ref_snapshot_dir(repo_dir)
        latest = latest_snapshot_dir(repo_dir)
        for snapshot in (preferred, latest):
            if snapshot is not None and snapshot not in snapshots:
                snapshots.append(snapshot)
        try:
            snapshots.extend(
                snapshot
                for snapshot in sorted(
                    (repo_dir / "snapshots").iterdir(), key = lambda path: path.name, reverse = True
                )
                if snapshot.is_dir() and snapshot not in snapshots
            )
        except OSError:
            pass
        for snapshot in snapshots:
            config = _config_from_path(snapshot / "config.json")
            if config is None:
                continue
            # Completeness reads shard indexes off disk, so reject on the configuration first.
            method = _drafter_method(config)
            if method is None:
                continue
            # An architecture this runtime cannot load looks like a config that is not a
            # drafter. A method the probe tracks is the exception: its row survives to name why.
            if method not in _MLX_METHOD_MODULES and (
                not _drafter_architecture_available(config)
                or _normalized_drafter_config(config) is None
            ):
                continue
            if not _snapshot_complete_at(snapshot):
                continue
            if method == "mtp" and not _snapshot_complete_at(snapshot, require_tokenizer = True):
                continue
            rows.append((repo_id, config, snapshot, _snapshot_weight_bytes_at(snapshot)))
    return tuple(rows)


# A scan ages from the moment it ran, so the lookups a request makes within that age share one
# scan rather than falling either side of a deadline and scanning the cache twice. It also bounds
# how long the scan's view may disagree with disk about a change Studio did not make, in either
# direction: a checkpoint written behind its back stays invisible, and one deleted behind its back
# stays selectable. Changes Studio does make bump the scan epoch, and are seen by the first
# lookup that starts after the bump.
_DRAFTER_SCAN_MAX_AGE_SECONDS = 15.0
# A floor, raised by callers that traverse: a cap below the number of roots evicts the entries
# the traversal is still walking towards, and every request rescans every cache.
_DRAFTER_SCANS_KEPT = 8
# (cache root, scan epoch) -> when that scan finished, and the drafters it found
_drafter_scans: dict[tuple[str, int], tuple[float, _DrafterRows]] = {}
_drafter_scans_lock = threading.Lock()


def _serve_kept_scan(key: tuple[str, int]) -> _DrafterRows:
    """The rows held for ``key``, moved youngest-last because they were just asked for."""
    # Reinserted rather than read in place: assigning to a key a dictionary already holds leaves
    # it where it was, which is first in line to be dropped when scans arrive behind it.
    held = _drafter_scans.pop(key)
    _drafter_scans[key] = held
    return held[1]


def _cached_active_drafter_configs(
    root: str,
    epoch: int,
    keep: int = 0,
) -> _DrafterRows:
    key = (root, epoch)
    started = time.monotonic()
    with _drafter_scans_lock:
        held = _drafter_scans.get(key)
        if held is not None and started - held[0] < _DRAFTER_SCAN_MAX_AGE_SECONDS:
            return _serve_kept_scan(key)
    rows = _scan_active_cached_drafter_configs(Path(root))
    finished = time.monotonic()
    with _drafter_scans_lock:
        held = _drafter_scans.get(key)
        fresh = held is not None and finished - held[0] < _DRAFTER_SCAN_MAX_AGE_SECONDS
        # A cache that could not be read is not a cache holding no drafters. Remembering it as one
        # would keep every request off the accelerator until the entry aged out, and reporting it
        # is worse still while a scan that did read the cache is standing right there.
        if rows is None:
            return _serve_kept_scan(key) if fresh else ()
        # The first scan of an epoch to finish is the one kept until it ages out, so one that
        # started earlier but arrived later does not replace it, and every caller is answered with
        # the scan that was kept rather than its own. Scans of other epochs, or of another cache
        # root, are kept beside it rather than displacing it.
        if fresh:
            return _serve_kept_scan(key)
        # Ages from the moment it finished, so a scan slower than the age it is kept for does not
        # arrive already expired.
        _drafter_scans.pop(key, None)
        _drafter_scans[key] = (finished, rows)
        while len(_drafter_scans) > max(_DRAFTER_SCANS_KEPT, keep):
            del _drafter_scans[next(iter(_drafter_scans))]
    return rows


def _cached_drafter_configs() -> Iterator[tuple[str, dict[str, Any], Path, int]]:
    """Every cache Studio knows, active first, so a target left in a previously configured one
    still finds the drafters beside it."""
    try:
        from hub.utils.inventory_scan import hf_cache_scans_epoch
        epoch = hf_cache_scans_epoch()
    except Exception:
        return iter(())
    roots = _known_hf_cache_roots()
    # Two epochs' worth, so one turning over mid-traversal does not evict the roots behind it.
    keep = len(roots) * 2
    return iter(
        [row for root in roots for row in _cached_active_drafter_configs(str(root), epoch, keep)]
    )


def _mlx_memory_budget() -> Optional[int]:
    """The ceiling the worker will enforce, or None when the device cannot report one."""
    try:
        import mlx.core as mx
        if not mx.metal.is_available():
            return None
        recommended = mx.device_info().get("max_recommended_working_set_size") or 0
    except Exception:
        return None
    # The same fraction of the recommended working set the worker applies before it loads.
    return int(recommended * 0.85) if recommended > 0 else None


def _mlx_speculative_memory_ready(estimated_bytes: int) -> bool:
    """Whether a measured target and drafter fit the budget the load is held to.

    Metal caps allocations below physical memory, so sizing against RAM offers a pair the cap
    then refuses -- and an explicit method takes the resident model down with it. Answered from
    checkpoint files, which a load quantizing them holds less of: a pair this refuses may fit.
    """
    if estimated_bytes <= 0:
        return True
    budget = _mlx_memory_budget()
    if budget is not None:
        return estimated_bytes <= budget
    try:
        import psutil
        return estimated_bytes <= int(psutil.virtual_memory().total * 0.85)
    except Exception:
        return True


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("text_config")
    return value if isinstance(value, dict) else config


def _draft_model_type(config: dict[str, Any]) -> Optional[str]:
    value = config.get("model_type") or config.get("speculators_model_type")
    return value if isinstance(value, str) else None


def _tokenizer_path(repo_id: str) -> Path:
    config_path = _cached_config_path(repo_id)
    return config_path.parent / "tokenizer.json" if config_path else Path()


def _token_id_map(repo_id: str) -> Optional[dict[str, int]]:
    tokenizer_path = _tokenizer_path(repo_id)
    return _token_id_map_from_path(tokenizer_path)


def _token_id_map_from_path(tokenizer_path: Path) -> Optional[dict[str, int]]:
    if not tokenizer_path.is_file():
        return None
    try:
        stat = tokenizer_path.stat()
    except OSError:
        return None
    return _cached_token_id_map(tokenizer_path, stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize = 32)
def _cached_token_id_map(
    tokenizer_path: Path, mtime_ns: int, size: int
) -> Optional[dict[str, int]]:
    del mtime_ns, size
    try:
        payload = json.loads(tokenizer_path.read_text(encoding = "utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    model = payload.get("model") if isinstance(payload, dict) else None
    vocab = model.get("vocab") if isinstance(model, dict) else None
    if isinstance(vocab, dict):
        try:
            mapping = {str(token): int(token_id) for token, token_id in vocab.items()}
        except (TypeError, ValueError):
            return None
    elif isinstance(vocab, list):
        mapping = {str(token): token_id for token_id, token in enumerate(vocab)}
    else:
        return None
    added_tokens = payload.get("added_tokens", [])
    if not isinstance(added_tokens, list):
        return None
    for token in added_tokens:
        if isinstance(token, dict) and isinstance(token.get("content"), str):
            try:
                mapping[token["content"]] = int(token["id"])
            except (KeyError, TypeError, ValueError):
                return None
    return mapping


def _dflash_family_method(config: dict[str, Any]) -> Optional[str]:
    """These checkpoints leave ``model_type`` as their backbone's, so mlx-vlm separates them in
    ``get_model_and_args`` by architecture and projector. Read the same way rather than by loading
    the drafter module, so a runtime too old to load one can still name the method.
    """
    architectures = config.get("architectures")
    if "DFlash2DraftModel" in set(architectures if isinstance(architectures, list) else ()):
        return "dflash2"
    nested = config.get("dflash_config")
    if not isinstance(nested, dict):
        return None
    try:
        markov = int(config.get("markov_rank") or nested.get("markov_rank") or 0)
    except (TypeError, ValueError):
        markov = 0
    return "dspark" if nested.get("projector_type") == "dspark" or markov > 0 else "dflash"


def _drafter_method(config: dict[str, Any]) -> Optional[str]:
    model_type = _draft_model_type(config)
    try:
        from mlx_vlm.speculative.drafters import DRAFTER_KIND_BY_MODEL_TYPE
        kind = DRAFTER_KIND_BY_MODEL_TYPE.get(model_type)
    except Exception:
        kind = None
    if kind == "dflash" or (kind is None and isinstance(config.get("dflash_config"), dict)):
        return _dflash_family_method(config) or "dflash"
    if kind in MLX_SPECULATIVE_METHODS:
        return kind
    speculators = config.get("speculators_config")
    if isinstance(speculators, dict) and speculators.get("algorithm") == "eagle3":
        return "eagle3"
    return None


def _token_maps_compatible(target: dict[str, int], draft: dict[str, int]) -> bool:
    return bool(
        draft
        and len(draft) >= int(len(target) * 0.99)
        and all(target.get(token) == token_id for token, token_id in draft.items())
    )


def _drafter_architecture_available(config: dict[str, Any]) -> bool:
    try:
        from mlx_vlm.utils import get_model_and_args
        module, _model_type = get_model_and_args(config)
        return module.__name__.startswith("mlx_vlm.speculative.drafters.") and callable(
            getattr(module, "Model", None)
        )
    except Exception:
        return False


def _accepts_mtp_captures(target_class: Any) -> bool:
    """Whether MTP's capture keywords reach this class's call. Mirrors the check the loaded
    pair repeats; a class taking ``**kwargs`` passes them through."""
    try:
        parameters = inspect.signature(target_class.__call__).parameters
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    ) or all(name in parameters for name in ("return_hidden", "return_shared_kv"))


def _target_method_contract_available(method: str, config: dict[str, Any]) -> bool:
    """Every method rewinds the target's cache, so a model class without that entry point is
    filtered during discovery rather than refused once both models are resident. An
    unrecognized target resolves to the generic text model, which lacks it too.

    MTP also reads hidden states and shared KV back out of the target's call. The loaded pair is
    checked for that again, which is after the resident model has gone, so it is asked here too.
    """
    try:
        from mlx_vlm.utils import get_model_and_args

        module, _model_type = get_model_and_args(config)
        model = getattr(module, "Model")
        language_model = getattr(module, "LanguageModel", None)
        if language_model is None:
            language_model = getattr(
                importlib.import_module(model.__module__), "LanguageModel", None
            )
        return any(
            callable(getattr(candidate, "rollback_speculative_cache", None))
            and (method != "mtp" or _accepts_mtp_captures(candidate))
            for candidate in (model, language_model)
            if candidate is not None
        )
    except Exception:
        return False


def _normalized_drafter_config(config: dict[str, Any]) -> Optional[Any]:
    try:
        from mlx_vlm.utils import get_model_and_args

        module, _model_type = get_model_and_args(config)
        model_config = getattr(module, "ModelConfig")
        factory = getattr(model_config, "from_dict", None) or getattr(
            model_config, "from_hf_dict", None
        )
        return factory(config) if callable(factory) else model_config(**config)
    except Exception:
        return None


def _config_value(
    config: Any,
    key: str,
    default: Any = None,
) -> Any:
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def _same_eos(left: Any, right: Any) -> bool:
    def values(value: Any) -> set[int]:
        return (
            {value}
            if isinstance(value, int)
            else (
                {item for item in value if isinstance(item, int)}
                if isinstance(value, (list, tuple))
                else set()
            )
        )

    return bool(values(left).intersection(values(right)))


def _drafter_eos_serves(draft_eos: Any, target_config: dict[str, Any], text: Any) -> bool:
    """A drafter naming none does not discriminate on it, and a family can declare several
    while its text config names one, so both places count.
    """
    if draft_eos is None:
        return True
    return _same_eos(draft_eos, target_config.get("eos_token_id")) or _same_eos(
        draft_eos, _config_value(text, "eos_token_id")
    )


_RECOMMENDATION_COMMUNITY_OWNERS = frozenset({"mlx-community", "unsloth"})


def _target_identity_key(value: str) -> Optional[str]:
    compact = "".join(character for character in value.casefold() if character.isalnum())
    matches = set()
    for generation in ("35", "36", "38"):
        for variant in ("397ba17b", "122ba10b", "35ba3b", "27b", "9b", "4b"):
            if f"qwen{generation}{variant}" in compact:
                dotted_generation = f"3.{generation[-1]}"
                dashed_variant = variant.replace("ba", "b-a", 1) if "ba" in variant else variant
                matches.add(f"qwen{dotted_generation}-{dashed_variant}")
    for variant in ("26ba4b", "31b", "12b", "e4b", "e2b"):
        if f"gemma4{variant}" in compact:
            dashed_variant = variant.replace("ba", "b-a", 1) if "ba" in variant else variant
            matches.add(f"gemma4-{dashed_variant}")
    for variant, dashed in (("8ba1b", "8b-a1b"), ("26b", "2.6b")):
        if f"lfm25{variant}" in compact:
            matches.add(f"lfm2.5-{dashed}")
    if "museglimmer30b" in compact:
        matches.add("muse-glimmer-30b")
    if "deepseekv4flash" in compact:
        matches.add("deepseek-v4-flash")
    return next(iter(matches)) if len(matches) == 1 else None


def _target_repository_owner(target_id: str) -> Optional[str]:
    normalized = target_id.strip().replace("\\", "/")
    path = _local_path(normalized)
    if path is None:
        return None
    if not path.is_absolute() and (
        (path.is_dir() and (path / "config.json").is_file())
        or (path.is_file() and path.name == "config.json")
    ):
        return None
    if path.is_absolute():
        # The layout under any cache Studio knows, not only the active one: a target picked out
        # of a previously configured cache loads by snapshot path, and the repository that
        # snapshot came from is still what a recommendation is allowed against. Outside those
        # roots the same three components are directory names that vouch for nobody.
        try:
            layout_path = path.parent.resolve() / path.name
        except OSError:
            return None
        for root in _known_hf_cache_roots():
            try:
                parts = layout_path.relative_to(root).parts
            except ValueError:
                continue
            target_is_snapshot = len(parts) == 3 or (
                len(parts) == 4 and parts[3].casefold() == "config.json" and path.is_file()
            )
            if not (target_is_snapshot and parts[1].casefold() == "snapshots" and parts[2]):
                continue
            prefix = "models--"
            if parts[0].casefold().startswith(prefix):
                owner, separator, model = parts[0][len(prefix) :].partition("--")
                if separator and owner and model:
                    return owner.casefold()
        return None
    parts = normalized.split("/")
    if len(parts) == 2 and all(parts):
        return parts[0].casefold()
    return None


def _recommendation_target_owner_allowed(target_id: str, target_key: str) -> bool:
    owner = _target_repository_owner(target_id)
    vendor = _target_vendor_owner(target_key)
    return owner == vendor or owner in _RECOMMENDATION_COMMUNITY_OWNERS


def _target_vendor_owner(target_key: str) -> str:
    if target_key.startswith("qwen"):
        return "qwen"
    if target_key.startswith("gemma"):
        return "google"
    if target_key.startswith("muse-glimmer"):
        return "meta-models"
    if target_key.startswith("deepseek"):
        return "deepseek-ai"
    if target_key.startswith("lfm"):
        return "liquidai"
    return "poolside"


def _recommendation_target_key(target_id: str, config: Optional[dict[str, Any]]) -> Optional[str]:
    identity_key = _target_identity_key(target_id)
    if config is None or identity_key is None:
        return None
    text = _text_config(config)
    model_type = config.get("model_type")
    hidden = text.get("hidden_size")
    layers = text.get("num_hidden_layers")
    experts = text.get("num_experts")
    if experts is None:
        experts = text.get("n_routed_experts")
    vocab = text.get("vocab_size")
    if (
        not isinstance(model_type, str)
        or any(type(value) is not int for value in (hidden, layers, vocab))
        or (experts is not None and type(experts) is not int)
    ):
        return None
    signature = (
        model_type,
        hidden,
        layers,
        experts,
        vocab,
    )
    return (
        identity_key
        if identity_key in _RECOMMENDATION_TARGET_SHAPES.get(signature, frozenset())
        else None
    )


def _verifier_matches_target(
    verifier_id: Any, target_id: str, target_config: dict[str, Any]
) -> Optional[bool]:
    if not isinstance(verifier_id, str) or not verifier_id.strip():
        return None
    for config in (target_config, _text_config(target_config)):
        if "_name_or_path" not in config:
            continue
        declared = config["_name_or_path"]
        if declared is None or declared == "":
            continue
        if not isinstance(declared, str):
            return False
        return verifier_id.casefold() == declared.casefold()
    if verifier_id.casefold() == target_id.casefold():
        return True
    target_key = _recommendation_target_key(target_id, target_config)
    return bool(
        target_key
        and _recommendation_target_owner_allowed(target_id, target_key)
        and _target_repository_owner(verifier_id) == _target_vendor_owner(target_key)
        and _target_identity_key(verifier_id) == target_key
    )


def _identity_conflict(target_id: str, draft_id: str) -> bool:
    """A same-shape successor — Qwen3.5-27B and Qwen3.6-27B agree on model type, width,
    depth, vocabulary and tokenizer — leaves the published names as the only evidence. An
    unresolved name is silence, not disagreement.
    """
    target_key = _target_identity_key(target_id)
    draft_key = _target_identity_key(draft_id)
    return target_key is not None and draft_key is not None and target_key != draft_key


def _dynamic_candidate_config_matches(
    method: str,
    target_id: str,
    target_config: dict[str, Any],
    draft_config: dict[str, Any],
    draft_snapshot: Path,
    draft_id: str,
) -> Optional[bool]:
    if not _target_method_contract_available(method, target_config):
        return False
    if _identity_conflict(target_id, draft_id):
        return False
    target = _text_config(target_config)
    draft = _text_config(draft_config)
    target_hidden = target.get("hidden_size")
    target_layers = target.get("num_hidden_layers")
    target_vocab = target.get("vocab_size")
    if not all(
        isinstance(value, int) and value > 0
        for value in (target_hidden, target_layers, target_vocab)
    ):
        return False

    if method == "mtp":
        binding_hidden = (
            draft_config.get("backbone_hidden_size")
            or draft_config.get("target_hidden_size")
            or draft.get("hidden_size")
        )
        if binding_hidden != target_hidden or draft.get("vocab_size") != target_vocab:
            return False
        if _draft_model_type(draft_config) == "qwen3_5_mtp" and (
            draft.get("num_hidden_layers") != target_layers
            or draft.get("mtp_num_hidden_layers") in (None, 0)
        ):
            return False
        target_tokens = _token_id_map(target_id)
        draft_tokens = _token_id_map_from_path(draft_snapshot / "tokenizer.json")
        if target_tokens is None or draft_tokens is None:
            return None
        return _token_maps_compatible(target_tokens, draft_tokens)

    normalized = _normalized_drafter_config(draft_config)
    if normalized is None:
        return False

    if MLX_SPECULATIVE_DRAFT_KINDS.get(method) == "dflash":
        # Where a field lives is the config class's decision, not the raw file's: newer
        # checkpoints state num_target_layers under dflash_config.
        captures = _config_value(normalized, "target_layer_ids")
        hidden = _config_value(normalized, "hidden_size")
        vocab = _config_value(normalized, "vocab_size")
        target_count = _config_value(normalized, "num_target_layers")
        eos_matches = _draft_model_type(draft_config) == "laguna" or _drafter_eos_serves(
            draft_config.get("eos_token_id"), target_config, target
        )
        return bool(
            hidden == target_hidden
            and vocab == target_vocab
            and target_count == target_layers
            and eos_matches
            and isinstance(captures, (list, tuple))
            and captures
            and all(type(layer) is int and 0 <= layer < target_layers for layer in captures)
            and len(captures) == len(set(captures))
        )

    transformer = _config_value(normalized, "transformer_layer_config", {})
    captures = draft_config.get("target_layer_ids") or draft_config.get(
        "eagle_aux_hidden_state_layer_ids"
    )
    runtime_captures = _config_value(normalized, "capture_layer_ids")
    speculators = draft_config.get("speculators_config")
    verifier = speculators.get("verifier") if isinstance(speculators, dict) else None
    verifier_id = verifier.get("name_or_path") if isinstance(verifier, dict) else None
    width = _config_value(transformer, "hidden_size")
    structurally_compatible = bool(
        _config_value(normalized, "target_hidden_size") == target_hidden
        and isinstance(width, int)
        and width > 0
        and _config_value(transformer, "vocab_size") == target_vocab
        and isinstance(captures, (list, tuple))
        and len(captures) == 3
        and all(type(layer) is int for layer in captures)
        and len(set(captures)) == 3
        and isinstance(runtime_captures, (list, tuple))
        and len(runtime_captures) == 3
        and all(type(layer) is int and 0 <= layer < target_layers for layer in runtime_captures)
        and len(set(runtime_captures)) == 3
    )
    if not structurally_compatible:
        return False
    return _verifier_matches_target(verifier_id, target_id, target_config)


def _dynamic_materialization_bytes(config: dict[str, Any]) -> int:
    if _draft_model_type(config) not in {
        "gemma4_assistant",
        "gemma4_unified_assistant",
    } or not config.get("use_ordered_embeddings"):
        return 0
    text = _text_config(config)
    vocab, hidden = text.get("vocab_size"), text.get("hidden_size")
    return int(vocab * hidden * 2) if isinstance(vocab, int) and isinstance(hidden, int) else 0


BUILTIN_MTP_ID = "builtin://mtp"

# Draft tokens Auto runs each method at. The drafter's own configuration declares either
# nothing or its block size, neither near where the method pays off.
MLX_AUTO_DRAFT_TOKENS: dict[str, int] = {
    "mtp": 3,
    "dflash": 3,
    # Drafts exactly what it is asked for, and acceptance falls off faster than a third token
    # repays: measured slower at three on every target.
    "dspark": 2,
    # Adapts below the request -- measured two per round when asked for three -- so the third
    # is headroom for sequences that accept enough to use it, not a cost paid up front.
    "dflash2": 3,
    "eagle3": 1,
}

# A target's own head ranks with MTP because it is MTP, and needs no download to get there.
# What a round returns against what it costs: a drafter's weights are read once per drafted
# token, the target once for the whole round. Neither term orders these methods alone.
_AUTO_METHOD_RANK: dict[str, int] = {
    # Within a few hundredths of DFlash2's acceptance per round, from a fifth of the weights.
    "mtp": 0,
    # Accepts enough more than DSpark to cover the weights it adds. That margin narrows with
    # the target and reverses under about 10 GB of it, which a low-bit large target can reach.
    "dflash2": 1,
    "dspark": 2,
    # Both lead the original DFlash by family, ordering checkpoints no target currently spans.
    "dflash": 3,
    "eagle3": 4,
}


def _precision_rank(bits: Optional[int]) -> tuple[int, int]:
    """How a drafter's width ranks: 8-bit, then full precision, then the rest widest-first.

    8-bit first because a narrower drafter leaves more memory to the target it drafts for
    at little cost in acceptance; below full precision the widths order themselves.
    """
    if bits == 8:
        return (0, 0)
    if bits is None:
        return (1, 0)
    return (2, -bits)


def mlx_auto_draft_block_size(method: str) -> Optional[int]:
    """Block size for the depth Auto runs ``method`` at, or None for a method it does not run.

    mlx-vlm counts the verified token alongside the drafted ones, so a block is one longer
    than the depth.
    """
    depth = MLX_AUTO_DRAFT_TOKENS.get(normalize_mlx_speculative_method(method))
    return None if depth is None else depth + 1


# Below this, verification costs about what the drafted tokens save.
MLX_AUTO_MIN_TARGET_PARAMETERS = 4_000_000_000


def _largest_int(value: Any) -> Optional[int]:
    """The largest of a field some configurations state per layer and others state once.

    Over-counting leaves a target the drafter it has; under-counting takes one away.
    """
    if type(value) is int:
        return value
    if isinstance(value, list):
        widths = [entry for entry in value if type(entry) is int]
        return max(widths) if widths else None
    return None


def _target_parameter_estimate(config: dict[str, Any]) -> Optional[int]:
    """Roughly how large the target is, or None when its shape does not say.

    From declared dimensions, so a quantized checkpoint and its full-precision twin estimate
    alike. Tables indexed by token id are excluded, since counting Gemma's per-layer embeddings
    would rate an E2B checkpoint above 4B; every expert is counted, since counting only the
    routed ones would rate a 35B mixture below a dense 4B.
    """
    text = _text_config(config)
    hidden = _largest_int(text.get("hidden_size"))
    layers = _largest_int(text.get("num_hidden_layers"))
    vocab = _largest_int(text.get("vocab_size"))
    if hidden is None or layers is None or vocab is None:
        return None
    experts = _largest_int(text.get("num_experts")) or _largest_int(text.get("n_routed_experts"))
    dense_width = _largest_int(text.get("intermediate_size")) or 4 * hidden
    if experts:
        width = _largest_int(text.get("moe_intermediate_size")) or dense_width
        feed_forward = experts * 3 * hidden * width
    else:
        feed_forward = 3 * hidden * dense_width
    return 2 * vocab * hidden + layers * (4 * hidden * hidden + feed_forward)


def _target_carries_quantization(config: dict[str, Any]) -> bool:
    """Whether the checkpoint is already quantized on disk.

    A load asked for 4-bit leaves such a checkpoint alone and quantizes a full-precision one
    as it loads, which is what decides whether the target's own head still matches it.
    """
    return bool(config.get("quantization") or config.get("quantization_config"))


def _fitting_cached_revision(
    repo_id: str,
    target_id: Optional[str],
    target_config: Optional[dict[str, Any]],
    method: Optional[str],
    revisions: Optional[tuple[tuple[str, dict[str, Any], Path, int], ...]] = None,
) -> tuple[Optional[dict[str, Any]], Optional[Path]]:
    """The cached revision of ``repo_id`` a load would choose for this target.

    One repository is often cached in several revisions, so ranking and loading share this:
    ranking one revision and loading another picks a drafter for a precision it lacks. A caller
    asking about several repositories hands the scan over rather than reaching for it each time,
    which is what can otherwise expire mid-request and scan the cache twice.
    """
    if revisions is None:
        revisions = tuple(_cached_drafter_configs())
    for cached_repo_id, config, snapshot, _size in revisions:
        if cached_repo_id.casefold() != repo_id.casefold():
            continue
        if target_config is None or (
            _drafter_method(config) == method
            and _dynamic_candidate_config_matches(
                method, target_id, target_config, config, snapshot, cached_repo_id
            )
            is True
        ):
            return config, snapshot
    return None, None


# Widths mlx-vlm derives from a method name, so a checkpoint ranks at the width it loads at.
_QUANT_METHOD_BITS: dict[str, int] = {"mxfp4": 4, "compressed-tensors": 4}
# The runtime warns, leaves the model dense, then loads the packed tensors into it strictly.
_QUANT_METHODS_REFUSED = frozenset({"awq", "gptq", "bitnet"})


def _loader_quantization(config: dict[str, Any]) -> tuple[Optional[dict[str, Any]], bool]:
    """The one declaration mlx-vlm will apply, and whether it settled the question by itself.

    A top-level "quantization" is applied as it stands, so a stale method left elsewhere decides
    nothing. Without one it falls to "quantization_config", top level then nested; the nested
    "quantization" spelling it never reads.
    """
    if "quantization" in config:
        block = config["quantization"]
        return (block if isinstance(block, dict) else None), True
    for source in (config, _text_config(config)):
        block = source.get("quantization_config")
        if isinstance(block, dict):
            return block, False
    return None, False


def _declared_quant_method(block: dict[str, Any]) -> Optional[str]:
    """The method a block names, or None where it names nothing usable.

    Configurations carry whatever they carry: a non-string used as a lookup key raises and takes
    the resolution with it, failing the target's load over a drafter it did not need.
    """
    method = block.get("quant_method")
    return method if isinstance(method, str) else None


def _refuses_quantization(config: dict[str, Any]) -> bool:
    """Whether the loader will decline this checkpoint's quantization, or break on it."""
    block, settled = _loader_quantization(config)
    if settled:
        # The loader subscripts both fields, so a block missing either fails the load.
        return not (
            isinstance(block, dict)
            and type(block.get("bits")) is int
            and type(block.get("group_size")) is int
        )
    if block is None:
        return False
    return _declared_quant_method(block) in _QUANT_METHODS_REFUSED


def _sidecar_quantization_bits(snapshot: Optional[Path]) -> Optional[int]:
    """The width a checkpoint declares beside its configuration rather than inside it."""
    if snapshot is None:
        return None
    try:
        with open(snapshot / "hf_quant_config.json", encoding = "utf-8") as handle:
            beside = json.load(handle)
    except (OSError, ValueError):
        return None
    # Well-formed JSON that is not an object still has to leave the target loadable.
    declared = beside.get("quantization") if isinstance(beside, dict) else None
    return 4 if isinstance(declared, dict) and declared.get("quant_algo") == "NVFP4" else None


def _config_precision_rank(
    config: dict[str, Any], snapshot: Optional[Path] = None
) -> tuple[int, int]:
    """How the width one cached revision would load at ranks.

    Read every way the loader reads it: a declaration missed reads as full precision, which is
    how a quantized drafter outranks the one it should lose to. DeepSeek's fp8 and a bare width
    under "quantization_config" rank full precision, neither being one it can answer.
    """
    block, settled = _loader_quantization(config)
    if block is None:
        # Only where the configuration declares nothing does the loader look beside it.
        return _precision_rank(None if settled else _sidecar_quantization_bits(snapshot))
    if not settled:
        # The runtime replaces the whole block, so a derived width beats one declared beside it.
        implied = _QUANT_METHOD_BITS.get(_declared_quant_method(block))
        if implied is not None:
            return _precision_rank(implied)
    bits = block.get("bits")
    return _precision_rank(bits if type(bits) is int else None)


@dataclass(frozen = True)
class NativeMtpEvidence:
    handler: str
    weight_bytes: int


@dataclass(frozen = True)
class _NativeMtpHandler:
    name: str
    module: str
    function: str
    prefixes: tuple[str, ...] | Callable[[dict[str, Any]], tuple[str, ...]]
    complete: Callable[[dict[str, Any], dict[str, int]], bool]


def _exact_layers(sizes: dict[str, int], prefix: str, expected: range) -> bool:
    tokens = [key[len(prefix) :].split(".", 1)[0] for key in sizes if key.startswith(prefix)]
    return all(token.isdigit() for token in tokens) and {int(token) for token in tokens} == set(
        expected
    )


def _only_expected_tensors(
    sizes: dict[str, int], prefix: str, required: set[str], optional: set[str]
) -> bool:
    return all(not key.startswith(prefix) or key in required or key in optional for key in sizes)


def _sidecars(weights: set[str], suffixes: tuple[str, ...]) -> set[str]:
    return {
        f"{key[: -len('.weight')]}{suffix}"
        for key in weights
        if key.endswith(".weight")
        for suffix in suffixes
    }


def _quantization_for_path(quantization: dict[str, Any], path: str) -> dict[str, Any]:
    override = quantization.get(path)
    return override if isinstance(override, dict) else quantization


def _qwen_complete(config: dict[str, Any], sizes: dict[str, int]) -> bool:
    text = config.get("text_config")
    if not isinstance(text, dict):
        return False
    depth = text.get("mtp_num_hidden_layers", 1)
    if type(depth) is not int or not 1 <= depth <= 16:
        return False
    if not _exact_layers(sizes, "mtp.layers.", range(depth)):
        return False
    required = {
        "mtp.fc.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    }
    attention = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    )
    target_moe = "moe" in str(config.get("model_type", "")).lower()
    moe = "moe" in str(text.get("model_type", "")).lower()
    if target_moe != moe:
        return False
    biases = (
        (
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
            "self_attn.o_proj.bias",
        )
        if text.get("attention_bias")
        else ()
    )
    for layer in range(depth):
        prefix = f"mtp.layers.{layer}."
        required.update(prefix + suffix for suffix in (*attention, *biases))
        if not moe:
            required.update(
                prefix + suffix
                for suffix in ("mlp.down_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight")
            )
            continue
        required.update(
            prefix + suffix
            for suffix in (
                "mlp.gate.weight",
                "mlp.shared_expert_gate.weight",
                "mlp.shared_expert.down_proj.weight",
                "mlp.shared_expert.gate_proj.weight",
                "mlp.shared_expert.up_proj.weight",
            )
        )
        sanitized = {
            prefix + f"mlp.switch_mlp.{projection}_proj.weight"
            for projection in ("down", "gate", "up")
        }
        fused = {prefix + "mlp.experts.gate_up_proj", prefix + "mlp.experts.down_proj"}
        selected = sanitized if sanitized.issubset(sizes) else fused
        if not selected.issubset(sizes):
            return False
        required.update(selected)
    quantized = {key for key in required if key.endswith(".weight") and "norm" not in key}
    optional = _sidecars(quantized, (".scales", ".biases"))
    sidecars = {key for key in sizes if key.endswith((".scales", ".biases"))}
    if sidecars:
        quantization = config.get("mtplx_mtp_quantization")
        if quantization is None:
            quantization = config.get("quantization")
        if (
            not isinstance(quantization, dict)
            or type(quantization.get("group_size")) is not int
            or type(quantization.get("bits")) is not int
        ):
            return False
        for scale in (key for key in sidecars if key.endswith(".scales")):
            base = scale.removesuffix(".scales")
            params = _quantization_for_path(quantization, base.removeprefix("mtp."))
            mode = params.get("mode", "affine")
            has_biases = f"{base}.biases" in sizes
            if mode not in {"affine", "mxfp4", "mxfp8", "nvfp4"}:
                return False
            if mode == "affine" and not has_biases:
                return False
            if mode != "affine" and has_biases:
                return False
        if any(
            f"{key.removesuffix('.biases')}.scales" not in sizes
            for key in sidecars
            if key.endswith(".biases")
        ):
            return False
    return required.issubset(sizes) and _only_expected_tensors(sizes, "mtp.", required, optional)


_QWEN = _NativeMtpHandler(
    "qwen3_5",
    "mlx_vlm.speculative.drafters.qwen3_5_mtp.split",
    "split_qwen3_5_mtp",
    ("mtp.",),
    _qwen_complete,
)


_DEEPSEEK_SHARED_TENSORS = (
    "enorm.weight",
    "hnorm.weight",
    "e_proj.weight",
    "h_proj.weight",
    "norm.weight",
    "attn_norm.weight",
    "ffn_norm.weight",
    "attn.q_norm.weight",
    "attn.kv_norm.weight",
    "attn.wq_a.weight",
    "attn.wq_b.weight",
    "attn.wkv.weight",
    "attn.wo_a.weight",
    "attn.wo_b.weight",
    "attn.attn_sink",
    "ffn.gate.weight",
    "ffn.gate.bias",
    "ffn.shared_experts.w1.weight",
    "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w3.weight",
    "hc_attn_base",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_ffn_base",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_head_base",
    "hc_head_fn",
    "hc_head_scale",
)

_DEEPSEEK_UNQUANTIZED_WEIGHTS = frozenset(
    {
        "enorm.weight",
        "hnorm.weight",
        "norm.weight",
        "attn_norm.weight",
        "ffn_norm.weight",
        "attn.q_norm.weight",
        "attn.kv_norm.weight",
        "ffn.gate.weight",
    }
)


def _deepseek_complete(config: dict[str, Any], sizes: dict[str, int]) -> bool:
    text = config.get("text_config")
    text = text if isinstance(text, dict) else config
    depth = text.get("num_nextn_predict_layers", 1)
    experts = text.get("n_routed_experts")
    if type(depth) is not int or not 1 <= depth <= 16:
        return False
    if type(experts) is not int or not 1 <= experts <= 2048:
        return False
    layer_tokens = [key.split(".", 2)[1] for key in sizes if key.startswith("mtp.")]
    if any(not token.isdigit() for token in layer_tokens):
        return False
    if {int(token) for token in layer_tokens} != {0}:
        return False
    prefix = "mtp.0."
    required = {prefix + suffix for suffix in _DEEPSEEK_SHARED_TENSORS}
    required.update(
        prefix + f"ffn.experts.{expert}.w{projection}.weight"
        for expert in range(experts)
        for projection in (1, 2, 3)
    )
    if not required.issubset(sizes):
        return False
    quantized_weights = {
        key
        for key in required
        if key.endswith(".weight") and key.split(".", 2)[-1] not in _DEEPSEEK_UNQUANTIZED_WEIGHTS
    }
    expected_experts = set(range(experts))
    for projection in (1, 2, 3):
        scaled_experts = {
            expert
            for expert in expected_experts
            if f"{prefix}ffn.experts.{expert}.w{projection}.scales" in sizes
        }
        if scaled_experts and scaled_experts != expected_experts:
            return False
    optional = _sidecars(quantized_weights, (".scale", ".scales"))
    if any(key.endswith(".scale") for key in sizes):
        for key in quantized_weights:
            scale = key[: -len(".weight")]
            if f"{scale}.scale" not in sizes and f"{scale}.scales" not in sizes:
                return False
    return _only_expected_tensors(sizes, "mtp.", required, optional)


_DEEPSEEK = _NativeMtpHandler(
    "deepseek_v4",
    "mlx_vlm.speculative.drafters.deepseek_v4_mtp.split",
    "split_deepseek_v4_mtp",
    ("mtp.",),
    _deepseek_complete,
)

_INKLING_LAYER_TENSORS = (
    "embed_norm.weight",
    "hidden_norm.weight",
    "input_proj.weight",
    "transformer_block.attn.k_norm.weight",
    "transformer_block.attn.k_sconv.weight",
    "transformer_block.attn.q_norm.weight",
    "transformer_block.attn.rel_logits_proj.proj",
    "transformer_block.attn.v_sconv.weight",
    "transformer_block.attn.wk_dv.weight",
    "transformer_block.attn.wo_ud.weight",
    "transformer_block.attn.wq_du.weight",
    "transformer_block.attn.wr_du.weight",
    "transformer_block.attn.wv_dv.weight",
    "transformer_block.attn_norm.weight",
    "transformer_block.attn_sconv.weight",
    "transformer_block.mlp.global_scale",
    "transformer_block.mlp.w13_dn.weight",
    "transformer_block.mlp.w2_md.weight",
    "transformer_block.mlp_norm.weight",
    "transformer_block.mlp_sconv.weight",
)

_INKLING_BLOCK_TENSORS = (
    "embed_norm.weight",
    "hidden_norm.weight",
    "input_proj.weight",
    "transformer_block.self_attn.qkvr_proj.weight",
    "transformer_block.self_attn.o_proj.weight",
    "transformer_block.self_attn.k_sconv.conv.weight",
    "transformer_block.self_attn.v_sconv.conv.weight",
    "transformer_block.self_attn.q_norm.weight",
    "transformer_block.self_attn.k_norm.weight",
    "transformer_block.self_attn.rel_proj",
    "transformer_block.mlp.gate_proj.weight",
    "transformer_block.mlp.up_proj.weight",
    "transformer_block.mlp.down_proj.weight",
    "transformer_block.mlp.global_scale",
    "transformer_block.input_layernorm.weight",
    "transformer_block.post_attention_layernorm.weight",
    "transformer_block.attn_sconv.conv.weight",
    "transformer_block.mlp_sconv.conv.weight",
)


def _inkling_complete(config: dict[str, Any], sizes: dict[str, int]) -> bool:
    mtp = config.get("mtp_config")
    text = config.get("text_config")
    if not isinstance(text, dict) or not text:
        return False
    mtp_depth = mtp.get("num_nextn_predict_layers") if isinstance(mtp, dict) else None
    if "num_mtp_layers" in text:
        depth = text.get("num_mtp_layers") or text.get("num_nextn_predict_layers") or 1
    else:
        depth = mtp_depth or text.get("num_nextn_predict_layers") or 1
    if type(depth) is not int or not 1 <= depth <= 16:
        return False
    raw_prefix = "model.mtp.layers."
    block_prefix = "model.mtp.blocks."
    raw = any(key.startswith(raw_prefix) for key in sizes)
    blocks = any(key.startswith(block_prefix) for key in sizes)
    if raw == blocks:
        return False
    layer_prefix, tensors = (
        (raw_prefix, _INKLING_LAYER_TENSORS) if raw else (block_prefix, _INKLING_BLOCK_TENSORS)
    )
    if not _exact_layers(sizes, layer_prefix, range(depth)):
        return False
    required = {"model.llm.norm.weight"}
    for index in range(depth):
        prefix = f"{layer_prefix}{index}."
        required.update(prefix + suffix for suffix in tensors)
    return required.issubset(sizes) and _only_expected_tensors(sizes, "model.mtp.", required, set())


_INKLING = _NativeMtpHandler(
    "inkling",
    "mlx_vlm.speculative.drafters.inkling_mtp.split",
    "split_inkling_mtp",
    ("model.mtp.", "model.llm.norm.weight"),
    _inkling_complete,
)


_GLM_HEAD_TENSORS = (
    "eh_proj.weight",
    "enorm.weight",
    "hnorm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "shared_head.head.weight",
    "shared_head.norm.weight",
    "self_attn.q_a_proj.weight",
    "self_attn.q_a_layernorm.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_a_layernorm.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate.weight",
    "mlp.shared_experts.gate_proj.weight",
    "mlp.shared_experts.up_proj.weight",
    "mlp.shared_experts.down_proj.weight",
)
_GLM_EXPERT_TENSORS = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
# Quantization sidecars are absent deliberately: this family's splitter reads the source
# as plain weights and only quantizes what it writes, so a quantized head cannot be split
# and must not be offered as one.
_GLM_OPTIONAL_TENSORS = ("embed_tokens.weight", "mlp.gate.e_score_correction_bias")


def _glm_text_config(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config") or config
    return text if isinstance(text, dict) else {}


def _glm_prefixes(config: dict[str, Any]) -> tuple[str, ...]:
    """This family's head is stored as one layer past the last real layer."""
    layers = _glm_text_config(config).get("num_hidden_layers")
    return (f"model.layers.{layers}.",) if type(layers) is int and layers >= 0 else ()


def _glm_complete(config: dict[str, Any], sizes: dict[str, int]) -> bool:
    text = _glm_text_config(config)
    prefixes = _glm_prefixes(config)
    experts = text.get("n_routed_experts")
    depth = text.get("num_nextn_predict_layers", 1)
    if not prefixes or type(experts) is not int or not 0 < experts <= 1024:
        return False
    if type(depth) is not int or not 1 <= depth <= 16:
        return False
    prefix = prefixes[0]
    required = {prefix + tensor for tensor in _GLM_HEAD_TENSORS}
    required |= {
        f"{prefix}mlp.experts.{index}.{tensor}"
        for index in range(experts)
        for tensor in _GLM_EXPERT_TENSORS
    }
    optional = {prefix + tensor for tensor in _GLM_OPTIONAL_TENSORS}
    return required.issubset(sizes) and _only_expected_tensors(sizes, prefix, required, optional)


_GLM = _NativeMtpHandler(
    "glm4_moe_lite",
    "mlx_vlm.speculative.drafters.glm4_moe_lite_mtp.split",
    "split_glm4_moe_lite_mtp",
    _glm_prefixes,
    _glm_complete,
)


_HANDLERS = {
    "qwen3_5": _QWEN,
    "qwen3_5_moe": _QWEN,
    "deepseek_v4": _DEEPSEEK,
    "inkling": _INKLING,
    "inkling_mm_model": _INKLING,
    "glm4_moe_lite": _GLM,
}


@lru_cache(maxsize = 8)
def _splitter(module: str, function: str) -> Optional[Callable[..., Path]]:
    try:
        value = getattr(importlib.import_module(module), function)
    except Exception:
        return None
    return value if callable(value) else None


def _handler(config: dict[str, Any]) -> Optional[_NativeMtpHandler]:
    handler = _handler_definition(config)
    if not handler or not _splitter(handler.module, handler.function):
        return None
    try:
        model = getattr(importlib.import_module(handler.module.rsplit(".", 1)[0]), "Model")
    except Exception:
        return None
    return handler if callable(model) else None


def _handler_definition(config: dict[str, Any]) -> Optional[_NativeMtpHandler]:
    text = config.get("text_config")
    model_type = config.get("model_type") or (
        text.get("model_type") if isinstance(text, dict) else None
    )
    return _HANDLERS.get(model_type)


def _safetensor_header(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            size = int.from_bytes(handle.read(8), "little")
            if size <= 0 or size > 64 * 1024 * 1024:
                return {}
            value = json.loads(handle.read(size))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _tensor_sizes(snapshot: Path, prefixes: tuple[str, ...]) -> Optional[dict[str, int]]:
    index = snapshot / "model.safetensors.index.json"
    selected: dict[Path, Optional[set[str]]]
    try:
        payload = json.loads(index.read_text(encoding = "utf-8")) if index.is_file() else None
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if isinstance(weight_map, dict):
            indexed: dict[Path, set[str]] = {}
            for key, name in weight_map.items():
                if not isinstance(key, str):
                    return None
                if not key.startswith(prefixes):
                    continue
                if not isinstance(name, str) or Path(name).name != name:
                    return None
                indexed.setdefault(snapshot / name, set()).add(key)
            selected = indexed
        else:
            selected = {}
        if not selected:
            selected = {
                path: None
                for path in snapshot.glob("*.safetensors")
                if path.name != "consolidated.safetensors"
            }
        if not selected or any(not path.is_file() for path in selected):
            return None
    except (OSError, RuntimeError, ValueError):
        return None
    sizes: dict[str, int] = {}
    for path, keys in selected.items():
        for key, metadata in _safetensor_header(path).items():
            if keys is not None and key not in keys:
                continue
            offsets = metadata.get("data_offsets") if isinstance(metadata, dict) else None
            if (
                key != "__metadata__"
                and isinstance(offsets, list)
                and len(offsets) == 2
                and all(type(offset) is int for offset in offsets)
                and 0 <= offsets[0] <= offsets[1]
            ):
                sizes[key] = offsets[1] - offsets[0]
    return sizes or None


def native_mtp_evidence(snapshot: Path, config: dict[str, Any]) -> Optional[NativeMtpEvidence]:
    handler = _handler(config)
    return _native_mtp_evidence(snapshot, config, handler)


def _native_mtp_evidence(
    snapshot: Path, config: dict[str, Any], handler: Optional[_NativeMtpHandler]
) -> Optional[NativeMtpEvidence]:
    prefixes = handler.prefixes if handler else ()
    if callable(prefixes):
        prefixes = prefixes(config)
    identity = _snapshot_identity(snapshot, handler) if handler and prefixes else None
    sizes = _cached_tensor_sizes(str(snapshot), identity, prefixes) if identity else None
    if not handler or not prefixes or not sizes or not handler.complete(config, sizes):
        return None
    selected = [size for key, size in sizes.items() if key.startswith(prefixes)]
    return NativeMtpEvidence(handler.name, sum(selected))


@lru_cache(maxsize = 32)
def _cached_tensor_sizes(
    snapshot: str, identity: str, prefixes: tuple[str, ...]
) -> Optional[dict[str, int]]:
    del identity
    return _tensor_sizes(Path(snapshot), prefixes)


def _snapshot_identity(snapshot: Path, handler: _NativeMtpHandler) -> str:
    try:
        runtime = version("mlx-vlm")
    except PackageNotFoundError:
        runtime = "unavailable"
    digest = hashlib.sha256(
        f"{handler.name}\0{handler.module}\0{handler.function}\0mlx-vlm:{runtime}"
        f"\0{snapshot.resolve()}".encode()
    )
    for path in sorted(
        (
            snapshot / "config.json",
            snapshot / "model.safetensors.index.json",
            *snapshot.glob("*.safetensors"),
        )
    ):
        try:
            stat = path.stat()
        except OSError:
            continue
        digest.update(f"\0{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


@dataclass(frozen = True)
class MlxSpeculativeResolution:
    """Concrete drafter pinned for one requested load."""

    method: str
    draft_model: Optional[str]
    reason: Optional[str] = None


def _complete_sidecar(path: Path, identity: str) -> bool:
    try:
        return (
            (path / "config.json").is_file()
            and (path / "model.safetensors").stat().st_size > 8
            and json.loads((path / "source.json").read_text(encoding = "utf-8")).get("identity")
            == identity
        )
    except (OSError, AttributeError, json.JSONDecodeError):
        return False


def cleanup_native_mtp_staging() -> None:
    from utils.paths.storage_roots import cache_root

    root = cache_root() / "mlx-speculative" / "mtp"
    try:
        paths = tuple(root.iterdir())
    except OSError:
        return
    for path in paths:
        try:
            fingerprint, owner, _suffix = path.name[1:].split("-", 2)
            if len(fingerprint) != 12:
                continue
            int(fingerprint, 16)
            os.kill(int(owner), 0)
        except ProcessLookupError:
            shutil.rmtree(path, ignore_errors = True)
        except (OSError, ValueError):
            continue


# A sibling is inside the critical section, so the destructive half is declined.
MLX_SIDECAR_LOCK_BUSY = "busy"


def _sidecar_lock_debug(what: str, exc: BaseException) -> None:
    from loggers import get_logger
    get_logger(__name__).debug("The MLX sidecar lock %s (%s)", what, exc)


@contextlib.contextmanager
def native_mtp_sidecar_lock(timeout: float = 10.0):
    """Serialize handing a sidecar out against reclaiming one, across this install's backends.

    Two backends of one install share this cache (see ``live_sibling_backend`` in ``run.py``), so
    one can reclaim a copy the other resolved a moment earlier and is about to open. Hold it until
    the drafter's files are open, after which unlinking them is harmless.

    Yields ``compiled_cache_lock``'s three states: only busy proves a sibling, and only busy must
    decline the destructive half.
    """
    from utils.cache_cleanup import (
        _CONTENTION_ERRNOS,
        LOCK_BUSY,
        LOCK_HELD,
        LOCK_UNAVAILABLE,
        _try_lock,
        _unlock,
        cache_coordination_dir,
    )

    try:
        directory = cache_coordination_dir()
        directory.mkdir(parents = True, exist_ok = True)
        fd = os.open(
            str(directory / "mlx-speculative-sidecars.lock"), os.O_CREAT | os.O_RDWR, 0o600
        )
    except Exception as exc:  # noqa: BLE001 -- opening it is part of taking it
        _sidecar_lock_debug("could not be opened", exc)
        yield LOCK_UNAVAILABLE
        return

    # _unlock closes the descriptor it is given, so only an acquired one may reach it.
    acquired: list[int] = []
    state = LOCK_HELD
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                _try_lock(fd)
                acquired.append(fd)
                break
            except OSError as exc:
                if exc.errno not in _CONTENTION_ERRNOS:
                    # Not contention: this filesystem cannot lock at all. Waiting out the timeout
                    # to answer "busy" would stall every load, then decline reclaiming forever.
                    _sidecar_lock_debug("is unavailable", exc)
                    with contextlib.suppress(OSError):
                        os.close(fd)
                    state = LOCK_UNAVAILABLE
                    break
                if time.monotonic() >= deadline:
                    _sidecar_lock_debug("is busy", exc)
                    with contextlib.suppress(OSError):
                        os.close(fd)
                    state = LOCK_BUSY
                    break
                time.sleep(0.05)
            except Exception as exc:  # noqa: BLE001
                _sidecar_lock_debug("is unavailable", exc)
                with contextlib.suppress(OSError):
                    os.close(fd)
                state = LOCK_UNAVAILABLE
                break
        yield state
    finally:
        for held in acquired:
            _unlock(held)


def _mark_sidecar_in_use(path: Path) -> None:
    try:
        os.utime(path, None)
    except OSError:
        pass


# A caller opens the directory just after it is handed back, so one touched this recently may be
# about to be read and is left for a later pass rather than deleted out from under it.
_SIDECAR_IN_USE_SECONDS = 300.0


def reclaim_superseded_native_mtp(root: Path, source: Path, identity: str) -> None:
    """Drop the sidecars split from ``source`` that this materialization replaces.

    A sidecar's name digests the splitter, the mlx-vlm version and the source's own files, so
    upgrading the runtime or re-splitting a re-downloaded target mints a new one and strands the
    old. Standing on this same source is what makes one superseded, so nothing about the target
    is guessed at: a sidecar whose target went away names a source never materialized again, and
    stays.
    """
    try:
        paths = tuple(root.iterdir())
    except OSError:
        return
    for path in paths:
        if path.name.startswith(".") or path.name == identity or not path.is_dir():
            continue
        try:
            recorded = json.loads((path / "source.json").read_text(encoding = "utf-8"))
            idle = time.time() - path.stat().st_mtime
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(recorded, dict) or recorded.get("source") != str(source):
            continue
        if idle > _SIDECAR_IN_USE_SECONDS:
            shutil.rmtree(path, ignore_errors = True)


def materialize_native_mtp(snapshot: Path, *, reclaim: bool = False) -> Path:
    """Split the target's built-in head into a sidecar, reusing one already on disk.

    ``reclaim`` drops the copies this split supersedes. It defaults off because that is safe only
    under ``native_mtp_sidecar_lock``, so the caller holding it is the one that asks.
    """
    config = json.loads((snapshot / "config.json").read_text(encoding = "utf-8"))
    handler = _handler(config) if isinstance(config, dict) else None
    if not handler or native_mtp_evidence(snapshot, config) is None:
        raise ValueError("mlx_builtin_mtp_unavailable")
    identity = _snapshot_identity(snapshot, handler)
    source = snapshot.resolve()
    from utils.paths.storage_roots import cache_root

    root = cache_root() / "mlx-speculative" / "mtp"
    final = root / identity
    if _complete_sidecar(final, identity):
        _mark_sidecar_in_use(final)
        # Swept here too, not only when a split runs: a copy this one superseded may have been
        # in use at that moment and skipped, and after that every request is this early return.
        if reclaim:
            reclaim_superseded_native_mtp(root, source, identity)
        return final
    cleanup_native_mtp_staging()
    if reclaim:
        reclaim_superseded_native_mtp(root, source, identity)
    root.mkdir(parents = True, exist_ok = True)
    staging = Path(tempfile.mkdtemp(prefix = f".{identity[:12]}-{os.getpid()}-", dir = root))
    try:
        splitter = _splitter(handler.module, handler.function)
        if splitter is None:
            raise RuntimeError("mlx_builtin_mtp_splitter_unavailable")
        splitter(str(snapshot), str(staging))
        if not (staging / "config.json").is_file() or not (staging / "model.safetensors").is_file():
            raise RuntimeError("mlx_builtin_mtp_materialization_incomplete")
        (staging / "source.json").write_text(
            json.dumps({"identity": identity, "source": str(source)}), encoding = "utf-8"
        )
        try:
            staging.replace(final)
        except OSError:
            if _complete_sidecar(final, identity):
                return final
            raise
        return final
    finally:
        shutil.rmtree(staging, ignore_errors = True)


def mlx_target_snapshot_path(target_id: str) -> Path:
    config_path = _cached_config_path(target_id)
    snapshot = config_path.parent if config_path is not None else None
    if snapshot is None or not _snapshot_complete_at(snapshot):
        raise FileNotFoundError(f"MLX target checkpoint is not downloaded: {target_id}")
    return snapshot


def mlx_speculative_snapshot_path(
    repo_id: str,
    target_id: Optional[str] = None,
    method: Optional[str] = None,
) -> Path:
    """The cached snapshot for ``repo_id`` that fits ``target_id``.

    With a target named this requires a fit rather than preferring one: a repository
    present in more than one revision resolves to the revision that matches, and one with
    no matching revision raises, so a stale snapshot cannot be handed to the worker.
    """
    target_config = _read_config(target_id) if target_id else None
    _config, snapshot = _fitting_cached_revision(repo_id, target_id, target_config, method)
    if snapshot is not None:
        return snapshot
    if target_config is not None:
        raise FileNotFoundError(f"No compatible MLX speculative checkpoint: {repo_id}")
    config_path = _cached_config_path(repo_id)
    if config_path is None or not _snapshot_complete_at(config_path.parent):
        raise FileNotFoundError(f"Incomplete MLX speculative checkpoint: {repo_id}")
    return config_path.parent


def native_mtp_tensors_present(snapshot: Path, config: dict[str, Any]) -> bool:
    """Return whether the target carries a complete native head, independent of runtime support."""
    return _native_mtp_evidence(snapshot, config, _handler_definition(config)) is not None


def _builtin_candidate_rows(target_id, target_config, caps, enabled):
    """A target that drafts for itself needs no companion, so this precedes every
    downloadable candidate. Carrying a head is not enough: one whose model class cannot
    rewind its cache is filtered here rather than refused at load.
    """
    if not _target_method_contract_available("mtp", target_config):
        return
    try:
        snapshot = mlx_target_snapshot_path(target_id)
    except (OSError, RuntimeError, ValueError):
        return
    if snapshot is None:
        return
    try:
        evidence = native_mtp_evidence(snapshot, target_config)
    except (OSError, RuntimeError, ValueError):
        return
    if evidence is None:
        return

    upstream_ready = bool(caps["methods"].get("mtp"))
    locally_ready = "mtp" in enabled
    estimated_memory_bytes = _snapshot_weight_bytes(target_id)
    if not upstream_ready:
        reason = caps["reason"] or "method_runtime_unavailable"
    elif not locally_ready:
        reason = "method_not_integrated"
    elif not _mlx_speculative_memory_ready(estimated_memory_bytes):
        reason = "insufficient_unified_memory"
    else:
        reason = None
    yield _CandidateRow(
        BUILTIN_MTP_ID,
        _UNVERIFIED,
        {
            "method": "mtp",
            "repo_id": BUILTIN_MTP_ID,
            "label": "Built-in MTP",
            "source": "builtin",
            "recommended": False,
            "approximate_size_bytes": evidence.weight_bytes,
            "estimated_memory_bytes": estimated_memory_bytes,
            "materialization_bytes": 0,
            "downloaded": True,
            "compatible": True,
            "runtime_supported": upstream_ready,
            "integration_ready": locally_ready,
            "loadable": reason is None,
        },
        reason,
    )


def _recommendation_offered(seed, target_id, target_key, target_config, native_head) -> bool:
    """A checkpoint is proposed only for the family it was built for and only from an owner
    the target vouches for, so an unrelated repository sharing a name cannot steer one.
    """
    if target_key != seed.target_key or not _recommendation_target_owner_allowed(
        target_id, target_key
    ):
        return False
    if seed.target_owner is not None and _target_repository_owner(target_id) != seed.target_owner:
        return False
    if target_config is not None and not _target_method_contract_available(
        seed.method, target_config
    ):
        return False
    # A target that already carries a head needs no companion for the same job.
    return not (seed.requires_missing_native_mtp and native_head)


def _recommended_candidate_rows(
    target_id, target_config, caps, enabled, native_head, builtin_offered
):
    """One proposal is marked, so the badge names a choice instead of restating the list."""
    target_key = _recommendation_target_key(target_id, target_config)
    offered = [
        seed
        for seed in _RECOMMENDATIONS
        if _recommendation_offered(seed, target_id, target_key, target_config, native_head)
    ]
    # The method Auto would reach for first. A resident head no proposal outranks is only
    # better than downloading one where it is actually offered: carrying the tensors is not
    # enough when this runtime cannot split them into a drafter.
    best = (
        None
        if builtin_offered
        else min(offered, key = lambda seed: _AUTO_METHOD_RANK[seed.method], default = None)
    )
    for seed in offered:
        target_matches = target_config is not None and (
            seed.verifier_id is None
            or _verifier_matches_target(seed.verifier_id, target_id, target_config) is True
        )
        upstream_ready = bool(caps["methods"].get(seed.method))
        locally_ready = seed.method in enabled
        estimated_memory_bytes = _snapshot_weight_bytes(target_id) + seed.approximate_size_bytes
        if target_config is None:
            reason = "target_config_unavailable"
        elif not target_matches:
            reason = "checkpoint_config_mismatch"
        elif not upstream_ready:
            reason = caps["reason"] or "method_runtime_unavailable"
        elif not locally_ready:
            reason = "method_not_integrated"
        else:
            reason = "checkpoint_not_downloaded"
        yield _CandidateRow(
            seed.repo_id.casefold(),
            _UNVERIFIED,
            {
                "method": seed.method,
                "repo_id": seed.repo_id,
                "label": seed.label,
                "source": "recommended",
                "recommended": seed is best,
                "approximate_size_bytes": seed.approximate_size_bytes,
                "estimated_memory_bytes": estimated_memory_bytes,
                "materialization_bytes": 0,
                "downloaded": False,
                "compatible": target_matches,
                "runtime_supported": upstream_ready,
                "integration_ready": locally_ready,
                "loadable": False,
            },
            reason,
        )


def _cached_candidate_rows(target_id, target_config, caps, enabled):
    """One row per snapshot directory, so the merge — not this source — picks the revision."""
    target_bytes = _snapshot_weight_bytes(target_id)
    for repo_id, draft_config, snapshot, weight_bytes in _cached_drafter_configs():
        method = _drafter_method(draft_config)
        if method is None:
            continue

        materialization_bytes = _dynamic_materialization_bytes(draft_config)
        estimated_memory_bytes = target_bytes + weight_bytes + materialization_bytes
        upstream_ready = bool(caps["methods"].get(method))
        locally_ready = method in enabled
        fields = {
            "method": method,
            "repo_id": repo_id,
            "label": repo_id.rsplit("/", 1)[-1],
            "source": "cached",
            "recommended": False,
            "approximate_size_bytes": weight_bytes,
            "estimated_memory_bytes": estimated_memory_bytes,
            "materialization_bytes": materialization_bytes,
            "downloaded": True,
            "runtime_supported": upstream_ready,
            "integration_ready": locally_ready,
            "compatible": True,
        }

        match = _dynamic_candidate_config_matches(
            method, target_id, target_config, draft_config, snapshot, repo_id
        )
        # A runtime that cannot load the drafter cannot judge its configuration either.
        if match is False and upstream_ready:
            yield _CandidateRow(
                repo_id.casefold(),
                _MISMATCH,
                fields,
                "checkpoint_config_mismatch",
                inherit = ("recommended",),
            )
            continue

        if not upstream_ready:
            reason = caps["reason"] or "method_runtime_unavailable"
        elif _refuses_quantization(draft_config):
            reason = "checkpoint_quantization_unsupported"
        elif not locally_ready:
            reason = "method_not_integrated"
        elif match is None:
            reason = (
                "verifier_contract_unavailable"
                if method == "eagle3"
                else "tokenizer_contract_unavailable"
            )
        elif not target_bytes:
            reason = "target_weights_unmeasured"
        elif not _mlx_speculative_memory_ready(estimated_memory_bytes):
            reason = "insufficient_unified_memory"
        else:
            reason = None

        yield _CandidateRow(
            repo_id.casefold(),
            _MATCH if match is True else _INDETERMINATE,
            {**fields, "loadable": reason is None},
            reason,
            inherit = ("recommended",),
        )


def _canonical_target_id(target_id: str) -> str:
    """The load path strips the request, expands a bare name to the default owner and reuses
    a cached spelling's case. Matching the raw request instead scans a cache entry that does
    not exist, so the target is told it has no drafter while the load then finds one.
    """
    from utils.models.model_config import is_local_path
    from utils.paths.path_utils import resolve_cached_repo_id_case

    identifier = (target_id or "").strip()
    if not identifier or is_local_path(identifier):
        return identifier or target_id
    if "/" not in identifier:
        identifier = f"unsloth/{identifier}"
    return resolve_cached_repo_id_case(identifier)


def mlx_speculative_options(target_id: str) -> dict[str, Any]:
    """Speculative drafters usable with ``target_id``, with local paths redacted.

    A runtime without speculative support still answers, with every candidate
    carrying the reason it cannot run rather than being omitted.
    """
    capabilities = mlx_speculative_runtime_capabilities()
    target_id = _canonical_target_id(target_id)
    target_config = _read_config(target_id)
    rows = []
    if target_config is not None:
        args = (target_id, target_config, capabilities, ENABLED_MLX_SPECULATIVE_METHODS)
        # Read from the checkpoint itself, not from whether this runtime could drive it, so
        # a runtime without speculative support does not become advice to download a head
        # the target already has. A target not on disk simply suppresses nothing.
        native_head = False
        try:
            snapshot = mlx_target_snapshot_path(target_id)
            if snapshot is not None:
                native_head = native_mtp_tensors_present(snapshot, target_config)
        except (OSError, RuntimeError, ValueError):
            native_head = False
        builtin = list(_builtin_candidate_rows(*args))
        rows = itertools.chain(
            builtin,
            _recommended_candidate_rows(*args, native_head, bool(builtin)),
            _cached_candidate_rows(*args),
        )
    return {
        "target_model": _public_target_model_id(target_id),
        "experimental": True,
        "runtime_supported": bool(capabilities["common"]),
        "runtime_reason": capabilities["reason"],
        "candidates": _merge_candidate_rows(rows),
    }


def _pinned_drafter(
    mode: str, draft_model: Optional[str], candidates: list[dict[str, Any]]
) -> tuple[Optional[str], Optional[str]]:
    """An accepted drafter carries the candidate's own repository id, not the spelling the
    request used, because the loader matches names exactly: accepting one it cannot resolve
    moves the failure from a refusal to a crash after the resident model is gone.
    """
    if mode not in ENABLED_MLX_SPECULATIVE_METHODS:
        return draft_model, "method_not_integrated"
    _, named, _ = mlx_speculative_request_identity(mode, draft_model, None)
    if not named:
        return draft_model, "checkpoint_required"
    for candidate in candidates:
        if candidate["method"] == mode and candidate["repo_id"].casefold() == named:
            return candidate["repo_id"], candidate["reason"]
    return draft_model, "checkpoint_not_compatible"


def mlx_speculative_request_reason(
    target_id: str,
    mode: Any,
    draft_model: Optional[str] = None,
    *,
    is_vision: bool = True,
    is_lora: bool = False,
) -> Optional[str]:
    """Why an MLX speculative request cannot be served, or None when it can.

    Off and Auto always resolve; an explicit method is refused unless the candidate list reports
    its drafter loadable, carrying that candidate's own reason.
    """
    mode = normalize_mlx_speculative_mode(mode)
    if mode in {"off", "auto"}:
        return None
    return resolve_mlx_speculative_request(
        target_id,
        mode,
        draft_model,
        is_vision = is_vision,
        is_lora = is_lora,
    ).reason


# Missing an input the target load supplies. The verifier contract is not one: it is read from
# the cached drafter, which no download changes.
_UNPROVEN_REASONS = frozenset(
    {
        "tokenizer_contract_unavailable",
        "target_config_unavailable",
        "target_weights_unmeasured",
    }
)


def mlx_speculative_reason_is_unproven(reason: Optional[str]) -> bool:
    """Whether ``reason`` records a comparison still missing its inputs, rather than a pair known
    not to fit.

    Refusing one rejects a pair that would have loaded, because downloading the target is what
    makes the comparison possible. The worker performs it once both checkpoints are resident,
    and an explicit method still fails its load there.
    """
    return reason in _UNPROVEN_REASONS


def mlx_speculative_refusal(mode: Any, resolution: "MlxSpeculativeResolution") -> Optional[str]:
    """Why ``resolution`` cannot be loaded, or None to load it.

    Auto never fails a load. Its reason is a diagnosis of why the request runs without
    speculation, not a refusal, so a model that cannot be accelerated still generates. A
    comparison still missing its inputs is carried the same way: the download this load is about
    to perform supplies them, and the worker judges the pair with both checkpoints resident.
    """
    if normalize_mlx_speculative_mode(mode) in {"off", "auto"} or resolution.reason is None:
        return None
    if mlx_speculative_reason_is_unproven(resolution.reason):
        return None
    return mlx_speculative_refusal_text(resolution.reason)


def mlx_speculative_target_is_adapter(target_id: str) -> bool:
    """Whether the target is a LoRA adapter, read from the files its load would open.

    The candidate list is built for callers holding no model configuration, so the adapter is
    recognised from the snapshot rather than asked of one, on the same markers the rest of the
    model layer recognises it by.
    """
    from utils.models.model_config import _looks_like_lora_adapter

    path = _cached_config_path(_canonical_target_id(target_id))
    return path is not None and _looks_like_lora_adapter(path.parent)


def mlx_speculative_target_ineligible(
    *,
    is_vision: bool,
    is_lora: bool,
    is_distributed: bool = False,
) -> Optional[str]:
    """Why this launch can run no drafter, or None when it can.

    Speculation rides the mlx-vlm path, which a text-only target never takes; an adapter or a
    sharded placement does take it but has no drafter support. Asked wherever a drafter is
    resolved, so the answer a request is given is the one its load reaches.
    """
    if not is_vision:
        return "mlx_vlm_target_required"
    if is_lora:
        return "mlx_speculative_lora_unsupported"
    if is_distributed:
        return "mlx_speculative_distributed_unsupported"
    return None


def mlx_speculative_load_resolution(
    target_id: str,
    mode: Any,
    draft_model: Optional[str],
    *,
    resolved_mode: Any,
    resolved_draft_model: Optional[str],
    resolved_reason: Optional[str],
    is_vision: bool,
    is_lora: bool,
    is_distributed: bool,
) -> "MlxSpeculativeResolution":
    """The drafter a load will use, reusing the caller's pinned choice when it has one.

    A caller that already resolved passes its choice through unchanged, so the decision is
    not made twice against two different views of the cache. Only Auto with nothing pinned
    scans here, which is the path a caller that never resolved takes.
    """
    requested, _, _ = mlx_speculative_request_identity(mode, draft_model, None)
    ineligible = mlx_speculative_target_ineligible(
        is_vision = is_vision, is_lora = is_lora, is_distributed = is_distributed
    )
    if requested == "auto" and ineligible is not None:
        return MlxSpeculativeResolution("off", None, ineligible)
    if requested == "auto" and resolved_mode is None:
        return resolve_mlx_speculative_request(target_id, "auto", draft_model)
    method = resolved_mode if requested == "auto" else requested
    return MlxSpeculativeResolution(
        normalize_mlx_speculative_method(method), resolved_draft_model, resolved_reason
    )


def resolve_mlx_speculative_request(
    target_id: str,
    mode: Any,
    draft_model: Optional[str] = None,
    *,
    is_vision: bool = True,
    is_lora: bool = False,
    options: Optional[dict[str, Any]] = None,
) -> MlxSpeculativeResolution:
    """Pin one concrete local drafter, or ordinary MLX when Auto finds none.

    Auto never fails a load: with no loadable candidate it resolves to Off carrying the reason.

    ``is_vision`` and ``is_lora`` describe the target the load will build. Omitted, the answer is
    about the drafters alone; passed, a target no drafter can attach to is answered here rather
    than after the resident model has been torn down for it.
    """
    requested = normalize_mlx_speculative_mode(mode)
    if requested == "off":
        return MlxSpeculativeResolution("off", None)
    # Ahead of every cache read: no drafter changes an answer the target itself settles.
    ineligible = mlx_speculative_target_ineligible(is_vision = is_vision, is_lora = is_lora)
    if ineligible is not None:
        return MlxSpeculativeResolution(
            "off" if requested == "auto" else requested, None, ineligible
        )
    # Both the scan and the compatibility checks match on the canonical id.
    target_id = _canonical_target_id(target_id)
    target_config = _read_config(target_id)
    if requested != "auto":
        available = (options or mlx_speculative_options(target_id))["candidates"]
        pinned, reason = _pinned_drafter(requested, draft_model, available)
        # Candidates match on the target's identity, so an unfetched target offers none and
        # would be refused for having none. The load resolves again once it can.
        if reason == "checkpoint_not_compatible" and not available and target_config is None:
            # Built anyway, but carrying why it is unjudged: the settlement after the
            # download has nothing to recognise otherwise.
            return MlxSpeculativeResolution(requested, draft_model, "target_config_unavailable")
        return MlxSpeculativeResolution(requested, pinned, reason)

    # No readable configuration is a different answer from having matched no drafter.
    if target_config is None:
        return MlxSpeculativeResolution("off", None, "target_config_unavailable")

    # Answered about the list the caller holds, rather than rescanning the cache.
    available = (options or mlx_speculative_options(target_id))["candidates"]
    _, preferred, _ = mlx_speculative_request_identity(requested, draft_model, None)
    # Ahead of the pin below, since a drafter named by hand is still one this target cannot
    # profit from. An unmeasurable target is not refused on ignorance.
    parameters = _target_parameter_estimate(target_config)
    if parameters is not None and parameters < MLX_AUTO_MIN_TARGET_PARAMETERS:
        return MlxSpeculativeResolution("off", None, "target_too_small_to_draft")

    builtin = next((row for row in available if row["source"] == "builtin"), None)

    if preferred:
        selected = next(
            (candidate for candidate in available if candidate["repo_id"].casefold() == preferred),
            None,
        )
        if selected is None or not selected["loadable"]:
            return MlxSpeculativeResolution(
                "off",
                None,
                (selected or {}).get("reason") or "auto_preferred_candidate_unavailable",
            )
        return MlxSpeculativeResolution(selected["method"], selected["repo_id"])

    if builtin is not None and builtin["loadable"]:
        return MlxSpeculativeResolution("mtp", builtin["repo_id"])

    downloaded = [row for row in available if row["source"] != "recommended"]
    # Ranked by the revision a load would take. A row with none that fits is one the loader
    # would refuse, whatever the list said when it was built.
    precision = {}
    rankable = [row for row in downloaded if row["loadable"] and row is not builtin]
    revisions = tuple(_cached_drafter_configs()) if rankable else ()
    for row in rankable:
        config, snapshot = _fitting_cached_revision(
            row["repo_id"], target_id, target_config, row["method"], revisions
        )
        if snapshot is not None:
            precision[row["repo_id"]] = _config_precision_rank(config, snapshot)
    candidates = [row for row in downloaded if row["repo_id"] in precision]
    if candidates:

        def priority(candidate: dict[str, Any]) -> tuple[int, tuple[int, int], str]:
            return (
                _AUTO_METHOD_RANK[candidate["method"]],
                precision[candidate["repo_id"]],
                candidate["repo_id"].casefold(),
            )

        selected = min(candidates, key = priority)
        return MlxSpeculativeResolution(selected["method"], selected["repo_id"])

    # Which way it failed matters: too large asks different action than never downloaded.
    refused = next((row["reason"] for row in downloaded if row.get("reason")), None)
    return MlxSpeculativeResolution("off", None, refused or "no_cached_drafter")
