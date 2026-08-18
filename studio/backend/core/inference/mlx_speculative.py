# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Discovery, recommendation, and preflight policy for MLX speculative decoding."""

from functools import lru_cache
import importlib
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional


MLX_SPECULATIVE_METHODS = frozenset({"mtp", "dflash", "eagle3"})
MLX_SPECULATIVE_MODES = MLX_SPECULATIVE_METHODS | {"auto"}

# Each method joins this set with the load path that can run it, so a request for a
# method the worker cannot execute is refused before the active model is torn down.
ENABLED_MLX_SPECULATIVE_METHODS: frozenset[str] = frozenset()

# Refusals reach the client as prose, while the codes stay the vocabulary the response
# schema will use to say why a resolved method differs from the one requested.
MLX_SPECULATIVE_REFUSALS: dict[str, str] = {
    "method_not_integrated": "This build cannot run the requested MLX speculative decoding method.",
}

# A code with no entry of its own still has to reach the client as a 400 rather than
# as the KeyError a subscript would raise.
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
}


def mlx_speculative_refusal_text(reason: str) -> str:
    """The sentence for a refusal, as an error detail. Unknown reasons read generically."""
    return MLX_SPECULATIVE_REFUSALS.get(reason, MLX_SPECULATIVE_GENERIC_REFUSAL)


def normalize_mlx_speculative_mode(value: Any) -> str:
    mode = str(value or "off").strip().lower()
    return mode if mode in MLX_SPECULATIVE_MODES else "off"


@lru_cache(maxsize = 1)
def mlx_speculative_runtime_capabilities() -> dict[str, Any]:
    result: dict[str, Any] = {
        "common": False,
        "methods": {method: False for method in sorted(MLX_SPECULATIVE_METHODS)},
        "reason": "runtime_unavailable",
    }
    if sys.platform != "darwin":
        result["reason"] = "mlx_requires_apple_silicon"
        return result
    try:
        drafters = importlib.import_module("mlx_vlm.speculative.drafters")
        ar = importlib.import_module("mlx_vlm.generate.ar")
        utils = importlib.import_module("mlx_vlm.speculative.utils")
        return _runtime_capabilities_from_modules(drafters, ar, utils)
    except Exception:
        return result


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
        try:
            dispatch = utils.get_speculative_rounds_batch(method)
        except Exception:
            dispatch = None
        methods[method] = bool(common and method in known and callable(dispatch))
    result = {"common": common, "methods": methods}
    result["reason"] = None if common else "runtime_missing_speculative_api"
    return result


def _local_path(model_id: str) -> Optional[Path]:
    """``model_id`` as a local path, or None when it cannot name one.

    A "~unknown-user" prefix has no home to expand into, and every caller here is
    asking whether a checkpoint sits on disk rather than asserting that it does.
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
    """One source's claim about one drafter repository.

    ``inherit`` names fields the merge must take from the row being replaced rather
    than from this one.
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
    """Resolve source rows into the candidate list, first verified match winning.

    One repository can produce several rows, because a cache holds a snapshot per
    revision. A verified match freezes the repository so no later snapshot can
    displace it, and an unreadable one blocks a later refutation, giving
    match > indeterminate > mismatch regardless of the order snapshots are read.
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


def _config_from_path(path: Path) -> Optional[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _scan_active_cached_drafter_configs(
    root: Path,
) -> tuple[tuple[str, dict[str, Any], Path, int], ...]:
    try:
        from hub.utils.hf_cache_state import latest_snapshot_dir, ref_snapshot_dir
        repo_dirs = sorted(root.glob("models--*"), key = lambda path: path.name.casefold())
    except (OSError, RuntimeError, ValueError):
        return ()
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
            if config is None or not _snapshot_complete_at(snapshot):
                continue
            method = _drafter_method(config)
            if (
                method is None
                or not _drafter_architecture_available(config)
                or _normalized_drafter_config(config) is None
                or (
                    method == "mtp"
                    and (
                        not _snapshot_complete_at(snapshot, require_tokenizer = True)
                        or _token_id_map_from_path(snapshot / "tokenizer.json") is None
                    )
                )
            ):
                continue
            rows.append((repo_id, config, snapshot, _snapshot_weight_bytes_at(snapshot)))
    return tuple(rows)


@lru_cache(maxsize = 8)
def _cached_active_drafter_configs(
    root: str, epoch: int, ttl_bucket: int
) -> tuple[tuple[str, dict[str, Any], Path, int], ...]:
    del epoch, ttl_bucket
    return _scan_active_cached_drafter_configs(Path(root))


def _active_cached_drafter_configs() -> Iterator[tuple[str, dict[str, Any], Path, int]]:
    try:
        from hub.utils.inventory_scan import hf_cache_scans_epoch
        from utils.hf_cache_settings import get_hf_cache_paths

        root = get_hf_cache_paths().hub_cache
        epoch = hf_cache_scans_epoch()
    except Exception:
        return iter(())
    return iter(_cached_active_drafter_configs(str(root), epoch, int(time.monotonic() / 15)))


def _mlx_speculative_memory_ready(estimated_bytes: int) -> bool:
    if estimated_bytes <= 0:
        return True
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


def _drafter_method(config: dict[str, Any]) -> Optional[str]:
    model_type = _draft_model_type(config)
    try:
        from mlx_vlm.speculative.drafters import DRAFTER_KIND_BY_MODEL_TYPE
        method = DRAFTER_KIND_BY_MODEL_TYPE.get(model_type)
    except Exception:
        method = None
    if method in MLX_SPECULATIVE_METHODS:
        return method
    if isinstance(config.get("dflash_config"), dict):
        return "dflash"
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


def _target_method_contract_available(method: str, config: dict[str, Any]) -> bool:
    if method != "dflash":
        return True
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
        root = _active_hf_cache_root()
        try:
            layout_path = path.parent.resolve() / path.name
            relative = layout_path.relative_to(root.resolve()) if root is not None else None
        except (OSError, ValueError):
            relative = None
        parts = relative.parts if relative is not None else ()
        target_is_snapshot = len(parts) == 3 or (
            len(parts) == 4 and parts[3].casefold() == "config.json" and path.is_file()
        )
        if target_is_snapshot and parts[1].casefold() == "snapshots":
            encoded = parts[0]
            prefix = "models--"
            if encoded.casefold().startswith(prefix):
                owner, separator, model = encoded[len(prefix) :].partition("--")
                if separator and owner and model and parts[2]:
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
    """Whether the two names resolve to different models.

    A same-shape successor — Qwen3.5-27B and Qwen3.6-27B share model type, width, depth,
    vocabulary and tokenizer — is indistinguishable from its predecessor in every value a
    config carries, so the published names are the only remaining evidence. An unresolved
    name is silence rather than disagreement, and leaves the structural verdict alone.
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

    if method == "dflash":
        if _draft_model_type(draft_config) == "laguna":
            captures = _config_value(normalized, "target_layer_ids")
            hidden = _config_value(normalized, "hidden_size")
            vocab = _config_value(normalized, "vocab_size")
            target_count = _config_value(normalized, "num_target_layers")
            eos_matches = True
        else:
            dflash = draft_config.get("dflash_config")
            captures = dflash.get("target_layer_ids") if isinstance(dflash, dict) else None
            hidden = draft_config.get("hidden_size")
            vocab = draft_config.get("vocab_size")
            target_count = draft_config.get("num_target_layers")
            eos_matches = _same_eos(draft_config.get("eos_token_id"), target.get("eos_token_id"))
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


def _cached_candidate_rows(target_id, target_config, caps, enabled):
    """Rows for drafters already materialized in the local cache.

    One repository yields one row per snapshot directory, so the merge — not this
    source — decides which revision wins.
    """
    target_bytes = _snapshot_weight_bytes(target_id)
    for repo_id, draft_config, snapshot, weight_bytes in _active_cached_drafter_configs():
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
        if match is False:
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
        elif not locally_ready:
            reason = "method_not_integrated"
        elif match is None:
            reason = (
                "verifier_contract_unavailable"
                if method == "eagle3"
                else "tokenizer_contract_unavailable"
            )
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


def mlx_speculative_options(target_id: str) -> dict[str, Any]:
    """Speculative drafters usable with ``target_id``, with local paths redacted.

    A runtime without speculative support still answers, with every candidate
    carrying the reason it cannot run rather than being omitted.
    """
    capabilities = mlx_speculative_runtime_capabilities()
    target_config = _read_config(target_id)
    rows = (
        _cached_candidate_rows(
            target_id, target_config, capabilities, ENABLED_MLX_SPECULATIVE_METHODS
        )
        if target_config is not None
        else ()
    )
    return {
        "target_model": _public_target_model_id(target_id),
        "experimental": True,
        "runtime_supported": bool(capabilities["common"]),
        "runtime_reason": capabilities["reason"],
        "candidates": _merge_candidate_rows(rows),
    }


def mlx_speculative_request_reason(method: Any) -> Optional[str]:
    """Why an MLX speculative request cannot be served, or None when it can.

    Off and Auto always resolve: Auto falls back to ordinary MLX generation when no
    drafter can run. An explicit method is refused unless its execution path is
    available, so an unsupported request never reaches model teardown.
    """
    mode = normalize_mlx_speculative_mode(method)
    if mode in {"off", "auto"}:
        return None
    if mode not in ENABLED_MLX_SPECULATIVE_METHODS:
        return "method_not_integrated"
    return None
