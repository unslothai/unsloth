# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Opt-in REAP expert pruning before an export (Core's ``FastLanguageModel.reap_prune``)."""

from __future__ import annotations

import functools
import json
import os
import weakref
from pathlib import Path
from typing import Any, List, Optional, Tuple

from loggers import get_logger

logger = get_logger(__name__)

# Core's unsloth/models/reap.py keys, plus the raw config.json names transformers maps onto them
# through attribute_map (HunYuan moe_topk, Ernie moe_k), which Core reads as num_experts_per_tok
# but a raw config dict does not; checked on text_config first, then the top-level config.
_NUM_EXPERTS_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
_TOP_K_KEYS = ("num_experts_per_tok", "top_k", "moe_top_k", "top_k_experts", "moe_topk", "moe_k")
# Decoder layers that hand state to the next layer (prev_* arguments), which Core refuses since it
# calibrates one layer at a time: GLM-5 / HY-V4 sparse-attention top-k indices, ZAYA router states.
_STATEFUL_MODEL_TYPES = frozenset({"glm_moe_dsa", "glm5_next", "glm5_next_text", "hy_v4", "zaya"})
# Hash-routed MoE (a fixed token to expert table, tid2eid), which Core refuses since dropped experts
# cannot be remapped.
_HASH_ROUTED_MODEL_TYPES = frozenset({"deepseek_v4"})
_N_GROUP_KEYS = ("n_group", "num_expert_group")
_TOPK_GROUP_KEYS = ("topk_group", "top_k_group")
REPORT_NAME = "reap_report.json"
# Written into each staging copy (the export worker's pid), so a parent can remove what a killed
# worker left behind without trusting a path it did not create.
_OWNER_FILE = ".unsloth_reap_owner"


def _first_int(
    config: dict,
    keys,
    per_layer = False,
) -> Optional[int]:
    for key in keys:
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if (
            per_layer
            and isinstance(value, list)
            and value
            and all(isinstance(v, int) for v in value)
        ):
            return max(value)
    return None


def _model_parallel(distributed) -> bool:
    """A saved distributed_config Core refuses: expert / tensor / pipeline / FSDP / sequence parallel."""
    if not isinstance(distributed, dict):
        return False
    if distributed.get("enable_expert_parallel") or distributed.get("enable_sequence_parallel"):
        return True
    return any(
        isinstance(distributed.get(k), int) and distributed.get(k) > 1
        for k in ("tp_size", "pp_size", "fsdp_size")
    )


# Serialized quantization REAP can still prune once Studio loads the model in 16-bit; any other
# quant_method (FP8, compressed-tensors / SparseML, ModelOpt, GPTQ, AWQ, ...) stays packed or is
# rewritten to FP8 by the loader, and Core's 16 / 32-bit weight check refuses it only after a full
# load. mxfp4: gpt-oss dequantizes to bf16 without native kernels (Core refuses it clearly
# otherwise). bitsandbytes only for the Hub ids Unsloth swaps for their 16-bit original.
_UNPACKED_QUANT_METHODS = {"mxfp4"}
# Older ModelOpt exports declare their quantization here instead of in config.json; the loader
# reads it the same way (attach_hf_quant_config).
_HF_QUANT_CONFIG = "hf_quant_config.json"


@functools.lru_cache(maxsize = 1)
def _int_to_float_mapper() -> dict:
    """Unsloth's quantized-to-16-bit Hub id table, read from mapper.py (plain data) without importing
    unsloth, which would initialize the GPU stack in the server process."""
    import importlib.util

    spec = importlib.util.find_spec("unsloth")
    roots = list(spec.submodule_search_locations or []) if spec else []
    for root in roots:
        path = Path(root) / "models" / "mapper.py"
        if path.is_file():
            module_spec = importlib.util.spec_from_file_location("_unsloth_mapper_data", path)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            return dict(module.INT_TO_FLOAT_MAPPER)
    return {}


def _maps_to_16_bit(model: str) -> bool:
    """A quantized Hub id Unsloth loads as its 16-bit original when asked for 16-bit."""
    try:
        return str(model).lower() in _int_to_float_mapper()
    except Exception as exc:
        logger.debug("Could not read Unsloth's model mapper: %s", exc)
        return False


class MoeProbeError(RuntimeError):
    """The config could not be read (network, Hub, I/O): unknown, not a dense answer, so an export
    with REAP on must wait or have REAP turned off rather than go out unpruned."""


def _cached_hub_file(repo: str, filename: str, offline: bool) -> Optional[bool]:
    """What the local Hub cache knows: True when the file is cached, False when its absence is
    recorded (or, offline, when the repo's snapshot is cached without it, so the loader could not
    fetch it either), else None."""
    try:
        from huggingface_hub import try_to_load_from_cache
        from utils.hf_cache_settings import active_hf_hub_cache

        cache_dir = active_hf_hub_cache()
        found = try_to_load_from_cache(repo, filename, cache_dir = cache_dir)
        if isinstance(found, str):
            return True
        if found is not None:  # _CACHED_NO_EXIST
            return False
        if offline and any(
            isinstance(try_to_load_from_cache(repo, name, cache_dir = cache_dir), str)
            for name in ("config.json", "adapter_config.json")
        ):
            return False
    except Exception as exc:
        logger.debug("Could not read the Hub cache for '%s': %s", repo, exc)
    return None


def hub_file_exists(
    repo: str,
    filename: str,
    hf_token = None,
) -> bool:
    """Whether a Hub repo has ``filename``. file_exists answers False only for a definite absence
    (repo, file or revision not found). Offline, or when the request fails, the local cache answers
    if it knows; anything still unknown (timeout, auth, gated) raises ``MoeProbeError``, so a file
    that would make REAP fail after the load is never assumed away."""
    error = None
    try:
        from huggingface_hub import HfApi, constants
        offline = bool(constants.HF_HUB_OFFLINE)
        if not offline:
            return bool(HfApi(token = hf_token).file_exists(repo, filename))
    except Exception as exc:
        offline, error = False, exc
    cached = _cached_hub_file(repo, filename, offline)
    if cached is not None:
        return cached
    raise MoeProbeError(f"Could not check '{repo}' for {filename}: {error or 'offline'}") from error


def _export_runs_transformers_4(model: str, hf_token = None) -> bool:
    """Whether Studio would load ``model`` for export under transformers 4.x (the 4.57 tier, or the
    default tier where the default is 4.57, e.g. Python 3.9). Core prunes only transformers 5's
    fused experts; 4.x keeps a ModuleList, which it refuses after the full load. If the tier
    cannot be resolved, the option stays (Core still refuses clearly, never exporting unpruned)."""
    try:
        from utils.transformers_version import (
            TRANSFORMERS_DEFAULT_VERSION,
            get_transformers_activation_tier,
        )
        tier = get_transformers_activation_tier(model, hf_token)
        return tier == "457" or (
            tier == "default" and str(TRANSFORMERS_DEFAULT_VERSION).startswith("4.")
        )
    except Exception as exc:
        logger.debug("Could not resolve the transformers tier of '%s': %s", model, exc)
        return False


def hub_adapter_has_mtp_head(repo: str, hf_token = None) -> bool:
    """Unsloth's loader re-attaches a Hub LoRA repo's trained MTP head, which REAP refuses."""
    return hub_file_exists(repo, "mtp_head.safetensors", hf_token)


def _quantization_allows_reap(holders, bnb_maps_to_16_bit: bool) -> bool:
    for holder in holders:
        quant = holder.get("quantization_config")
        if not quant:
            continue
        if not isinstance(quant, dict):
            return False
        method = str(quant.get("quant_method") or "").lower()
        if not method and (quant.get("load_in_4bit") or quant.get("load_in_8bit")):
            method = "bitsandbytes"  # configs saved before quant_method existed
        if method in _UNPACKED_QUANT_METHODS or (method == "bitsandbytes" and bnb_maps_to_16_bit):
            continue
        return False
    return True


def moe_info_from_config(
    config: Optional[dict], bnb_maps_to_16_bit: bool = False
) -> Optional[dict]:
    """``{num_experts, top_k}`` (plus ``n_group`` / ``topk_group`` for group-limited routers) for a
    routed-MoE config dict, else None."""
    if not isinstance(config, dict):
        return None
    holders = [h for h in (config.get("text_config"), config) if isinstance(h, dict)]

    # Like Core's _first_attr: each field from the first level that has it, text_config first.
    def first(keys, per_layer = False):
        return next((v for h in holders if (v := _first_int(h, keys, per_layer)) is not None), None)

    # Core refuses routers with identity (zero) experts and bidirectional / diffusion models.
    if any(h.get("zero_expert_num") for h in holders) or any(
        h.get("use_bidirectional_attention") == "all"
        or str(h.get("model_type", "")).startswith("diffusion")
        for h in holders
    ):
        return None
    if any(_model_parallel(h.get("distributed_config")) for h in holders):
        return None
    if not _quantization_allows_reap(holders, bnb_maps_to_16_bit):
        return None
    if any(
        str(h.get("model_type", "")) in _STATEFUL_MODEL_TYPES | _HASH_ROUTED_MODEL_TYPES
        for h in holders
    ):
        return None
    experts = first(_NUM_EXPERTS_KEYS)
    top_k = first(_TOP_K_KEYS, per_layer = True)
    if not (experts and top_k and experts > top_k):
        return None
    info = {"num_experts": experts, "top_k": top_k}
    n_group, topk_group = first(_N_GROUP_KEYS), first(_TOPK_GROUP_KEYS)
    if n_group and n_group > 1:
        if not topk_group:
            # Core refuses grouped routing it cannot prune per group.
            return None
        # Core prunes per group, so the UI must count the same way.
        info.update(n_group = n_group, topk_group = topk_group)
    return info


def moe_export_info(
    model: str,
    hf_token = None,
    local_files_only = True,
    _depth = 0,
    adapter_base: Optional[str] = None,
) -> Optional[dict]:
    """MoE info for a local checkpoint / model folder or a Hub id (a LoRA folder resolves to its
    base, or to ``adapter_base`` when the caller already authorized it). None for a dense model, a
    repo without config.json or a malformed one; ``MoeProbeError`` when the config could not be read."""
    try:
        from utils.models.diffusion_lm import _local_adapter_base
        from utils.models.model_config import _read_raw_config_json
        from utils.paths import is_local_path, normalize_path

        if is_local_path(model):
            local = Path(normalize_path(model)).expanduser()
            has_config = (local / "config.json").is_file()
            # Unsloth's loader re-attaches a LoRA folder's trained MTP head, which REAP refuses. A
            # full checkpoint's head is attached only per its training manifest (checked below).
            if (local / "adapter_config.json").is_file() and (
                local / "mtp_head.safetensors"
            ).is_file():
                return None
            # An adapter folder may carry a config.json too; the export loader follows
            # adapter_config.json, so the probe does as well.
            if not has_config or (local / "adapter_config.json").is_file():
                base = adapter_base or _local_adapter_base(str(local))
                if base and _depth < 2 and base != model:
                    # A Hub base of a local adapter may not be cached yet: read it online.
                    return moe_export_info(base, hf_token, is_local_path(base), _depth + 1)
                if not has_config:
                    return None
            if not (local / "adapter_config.json").is_file():
                from core.export.drafter_export import training_manifest

                # The export attaches a full checkpoint's trained MTP head, which REAP refuses.
                manifest = training_manifest(str(local))
                if manifest and manifest.get("strategy") == "mtp":
                    return None
        local = is_local_path(model)
        if local and (Path(normalize_path(model)).expanduser() / _HF_QUANT_CONFIG).is_file():
            return None
        config = _read_raw_config_json(model, hf_token, local_files_only, None)
        info = moe_info_from_config(config, bnb_maps_to_16_bit = not local and _maps_to_16_bit(model))
        if info is not None and _export_runs_transformers_4(model, hf_token):
            return None
        if (
            info is not None
            and not local
            and not (isinstance(config, dict) and config.get("quantization_config"))
            and hub_file_exists(model, _HF_QUANT_CONFIG, hf_token)
        ):
            return None
        return info
    except MoeProbeError:
        raise
    except ValueError as exc:
        # A malformed config.json (JSONDecodeError and friends) stays malformed: hide the option.
        logger.debug("MoE detection skipped for '%s': %s", model, exc)
        return None
    except Exception as exc:
        raise MoeProbeError(f"Could not read the config of '{model}': {exc}") from exc


def _options_key(options: dict) -> Tuple[float, int, Optional[str]]:
    return (
        float(options.get("sparsity", 0.25)),
        int(options.get("num_calibration_samples", 512)),
        (options.get("dataset") or "").strip() or None,
    )


def _loaded_in_4bit(model: Any) -> bool:
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        return True
    config = getattr(model, "config", None)
    quant = getattr(config, "quantization_config", None)
    if quant is not None:
        get = quant.get if isinstance(quant, dict) else lambda k, d = None: getattr(quant, k, d)
        if get("load_in_4bit", False) or get("load_in_8bit", False):
            return True
    return False


def _calibration_dataset(dataset: Optional[str], hf_token):
    """A custom dataset opened with the request's token (export workers keep tokens out of their
    environment, so Core's own load would be anonymous); the default stays Core's."""
    if not dataset:
        return None
    from datasets import load_dataset

    kwargs = {} if hf_token is None else {"token": hf_token}
    # Streaming reads only the rows Core uses. No split is forced: like Core, take `train`, else the
    # only split (calibration sets often ship just `validation` or `test`).
    splits = load_dataset(dataset, streaming = True, **kwargs)
    if not isinstance(splits, dict):
        return splits
    if "train" in splits:
        return splits["train"]
    if len(splits) == 1:
        return next(iter(splits.values()))
    raise ValueError(
        f"the calibration dataset '{dataset}' has no train split but several others "
        f"({', '.join(sorted(splits))}); use a dataset with a train split or a single split"
    )


def current_reap_key(backend: Any):
    """The REAP settings the loaded model was pruned with, else None. A key whose model is no
    longer the loaded one (an unload that failed part way before a new load) is not this model's."""
    key = getattr(backend, "_reap_key", None)
    ref = getattr(backend, "_reap_model", None)
    model = getattr(backend, "current_model", None)
    if key is None or ref is None or model is None or ref() is not model:
        return None
    return key


def apply_reap(
    backend: Any,
    options: Optional[dict],
    staging_parent,
    hf_token = None,
) -> Tuple[bool, str]:
    """Prune the loaded model once per load; a multi-format export reuses the pruned model.
    ``hf_token`` opens a private or gated calibration dataset."""
    if not options:
        return True, ""
    key = _options_key(options)
    applied = current_reap_key(backend)
    if applied is None and getattr(backend, "_reap_key", None) is not None:
        # Left by an unload that failed part way: it describes an older model, not this load.
        discard_reap_staging(backend)
    if applied == key:
        return True, ""
    if applied is not None:
        return False, (
            "This model was already pruned with different REAP settings. Start the export again "
            "to load a fresh copy."
        )
    model, tokenizer = backend.current_model, backend.current_tokenizer
    if model is None:
        return False, "No model loaded. Please select a checkpoint first."
    if _loaded_in_4bit(model):
        return False, (
            "REAP expert pruning needs the model loaded in 16-bit, but it was loaded in 4-bit. "
            "Turn off 4-bit loading and export again."
        )
    if getattr(model, "__dict__", {}).get("_unsloth_mtp_head") is not None:
        # Core refuses this: the attached MTP head would keep the old expert count.
        return False, (
            "REAP cannot prune a checkpoint with an attached MTP head. Export it without expert "
            "pruning."
        )
    from core.export.export import _staging_dir
    from unsloth import FastLanguageModel

    try:
        if getattr(backend, "is_peft", False):
            # REAP scores the fine-tuned experts, so the adapters are merged first (16-bit, in memory).
            logger.info("REAP: merging the LoRA adapters into the 16-bit base before pruning...")
            model = model.merge_and_unload()
            backend.current_model = model
            backend.is_peft = False
            # Kept until the next load: the base export must still refuse this LoRA checkpoint.
            backend._reap_merged_peft = True
        sweep_reap_staging()
        staging = _staging_dir(staging_parent)
        _register_staging(staging.name)
        # Attached now, so an unload after a failure below also removes it.
        backend._reap_staging = staging
        sparsity, samples, dataset = key
        # save_directory repoints the model at the pruned copy, so every later save, merge or GGUF
        # conversion reads the pruned weights and never the original checkpoint. GGUF files are named
        # after that folder, so it carries the model's own name.
        source = str(getattr(model.config, "_name_or_path", "") or "")
        name = Path(source.rstrip("/\\") or "model").name
        FastLanguageModel.reap_prune(
            model,
            tokenizer,
            sparsity = sparsity,
            dataset = _calibration_dataset(dataset, hf_token),
            num_calibration_samples = samples,
            save_directory = str(Path(staging.name) / f"{name or 'model'}-REAP"),
        )
    except Exception as exc:
        # A failure part way leaves the loaded model merged and partly pruned; never export it.
        logger.warning("REAP failed, unloading the partly pruned model: %s", exc)
        unloaded = False
        try:
            unloaded = backend.cleanup_memory() is not False
        except Exception as cleanup_exc:
            logger.warning("Could not unload after a failed REAP run: %s", cleanup_exc)
        if not unloaded:
            # cleanup_memory reports its own failures as False; the model must still never be
            # exported again from this worker, so drop it here.
            backend.current_model = None
            backend.current_tokenizer = None
            backend.current_checkpoint = None
            discard_reap_staging(backend)
        raise RuntimeError(
            f"REAP expert pruning failed: {exc}. The model was unloaded since it may be partly "
            "pruned; load it again to export."
        ) from exc
    # The model now points at the staging copy; Hub cards should name the model it came from.
    model._unsloth_reap_source = source or None
    backend._reap_key = key
    # The settings belong to this model object only (see current_reap_key).
    backend._reap_model = weakref.ref(model)
    report = getattr(model, "_unsloth_reap_report", None) or {}
    kept = len(next(iter(report.values()), []))
    return True, f"REAP kept {kept} experts per MoE layer in {len(report)} layers."


def gguf_report_name(gguf: str) -> str:
    """The report beside one GGUF file (``model.Q4_K_M.gguf`` -> ``model.Q4_K_M.reap_report.json``):
    a reused GGUF folder keeps files from other loads, so each file carries its own."""
    name = os.path.basename(str(gguf))
    stem = name[: -len(".gguf")] if name.lower().endswith(".gguf") else name
    return f"{stem}.reap_report.json"


def sync_reap_report(
    backend: Any,
    output_path: Optional[str],
    name: str = REPORT_NAME,
    files: Optional[List[str]] = None,
) -> None:
    """After an export: the report when the exported model is pruned, else remove one a previous
    REAP export left in a reused folder, so the folder never describes other weights."""
    if current_reap_key(backend) is not None:
        write_reap_report(backend, output_path, name, files)
        return
    if output_path and Path(output_path).is_dir():
        stale = Path(output_path) / name
        try:
            stale.unlink(missing_ok = True)
        except OSError as exc:
            logger.warning(f"Could not remove the stale {name}: {exc}")


def write_reap_report(
    backend: Any,
    output_path: Optional[str],
    name: str = REPORT_NAME,
    files: Optional[List[str]] = None,
) -> None:
    """The report (kept expert ids per layer and the settings) next to the export; ``files`` names
    the artifacts it describes when the folder can hold others."""
    model = getattr(backend, "current_model", None)
    report = getattr(model, "_unsloth_reap_report", None)
    key = current_reap_key(backend)
    if not report or key is None or not output_path or not Path(output_path).is_dir():
        return
    try:
        payload = {
            "method": "REAP (arXiv:2510.13999)",
            "sparsity": key[0],
            "num_calibration_samples": key[1],
            "dataset": key[2] or "mlabonne/open-perfectblend",
            "kept_experts": report,
        }
        if files:
            payload["files"] = files
        (Path(output_path) / name).write_text(json.dumps(payload, indent = 1), encoding = "utf-8")
    except Exception as exc:
        logger.warning(f"Could not write {name}: {exc}")


def discard_reap_staging(backend: Any) -> None:
    staging = getattr(backend, "_reap_staging", None)
    backend._reap_staging = None
    backend._reap_key = None
    backend._reap_model = None
    backend._reap_merged_peft = False
    if staging is not None:
        try:
            staging.cleanup()
        except Exception as exc:
            logger.warning(f"Could not remove the REAP staging copy: {exc}")
        sweep_reap_staging()


def reap_staging_parent(save_directory: Optional[str]) -> Path:
    """Where the pruned copy may stage: the chosen save folder's filesystem (an absolute path on
    another drive is allowed), else the exports root. ``_staging_dir`` then takes whichever of
    this and the temp dir has more room."""
    from utils.paths.storage_roots import resolve_export_dir, resolve_export_write_dir

    try:
        target = resolve_export_write_dir(save_directory)
    except Exception:
        return resolve_export_dir()
    # The save folder itself may not exist yet; its nearest existing ancestor is on the same drive.
    for candidate in (target, *target.parents):
        if candidate.is_dir():
            return candidate
    return resolve_export_dir()


def _registry() -> Path:
    from utils.paths.storage_roots import cache_root
    return cache_root() / "reap_staging.json"


def _read_registry() -> list:
    try:
        entries = json.loads(_registry().read_text(encoding = "utf-8"))
        return [e for e in entries if isinstance(e, dict) and "path" in e and "pid" in e]
    except Exception:
        return []


def _write_registry(entries: list) -> None:
    try:
        path = _registry()
        path.parent.mkdir(parents = True, exist_ok = True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries), encoding = "utf-8")
        tmp.replace(path)
    except Exception as exc:
        logger.debug("Could not update the REAP staging registry: %s", exc)


def _register_staging(path: str) -> None:
    import os

    pid = os.getpid()
    try:
        (Path(path) / _OWNER_FILE).write_text(str(pid), encoding = "utf-8")
    except OSError as exc:
        logger.debug("Could not mark the REAP staging copy: %s", exc)
    _write_registry(_read_registry() + [{"path": str(path), "pid": pid}])


def sweep_reap_staging(dead_pid: Optional[int] = None) -> None:
    """Remove staging copies whose worker is gone: a cancel kills the worker, so neither
    ``cleanup_memory`` nor the TemporaryDirectory finalizer runs. ``dead_pid`` is a worker the
    caller just killed; otherwise an owner counts as gone when its process no longer exists.
    Only folders carrying their owner marker are touched. Never raises."""
    import shutil

    try:
        import psutil
        alive = psutil.pid_exists
    except Exception:
        alive = None
    entries = _read_registry()
    if not entries:
        # Nothing was ever staged: no file is created or rewritten for users who never use REAP.
        return
    keep = []
    for entry in entries:
        path, pid = Path(entry["path"]), entry["pid"]
        if not path.exists():
            continue
        gone = pid == dead_pid or (dead_pid is None and alive is not None and not alive(pid))
        try:
            owned = (path / _OWNER_FILE).read_text(encoding = "utf-8").strip() == str(pid)
        except OSError:
            owned = False
        if gone and owned:
            shutil.rmtree(path, ignore_errors = True)
            if not path.exists():
                logger.info("Removed a REAP staging copy left by a stopped export: %s", path)
                continue
        keep.append(entry)
    if keep != entries:
        _write_registry(keep)
