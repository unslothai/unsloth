# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which multi-GPU layout a Studio training run uses, decided before anything loads.

`plan(request, host, model)` is pure: the host (platform, GPUs, collective backend, what the installed Unsloth can
split) and the model (VRAM estimate, MoE, family) come in as dicts, so tests spoof them. `detect_host()` and
`model_facts()` read the real ones. A multi-process plan becomes the UNSLOTH_PARALLEL spec every rank gets
(core/training/parallel_launch.py starts the ranks); `single` and `layer_split` keep today's one process.
A layout Unsloth's capability table refuses is refused here, with Unsloth's own reason, before any VRAM moves.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
from typing import Any, Optional

__all__ = [
    "MODES",
    "MULTI_PROCESS_MODES",
    "ParallelPlanError",
    "plan",
    "detect_host",
    "model_facts",
    "capabilities",
    "plan_for_request",
]

AUTO, SINGLE, LAYER_SPLIT, CUSTOM = "auto", "single", "layer_split", "custom"
DDP, FSDP2, TP, PP, CP, EP = "ddp", "fsdp2", "tp", "pp", "cp", "ep"
MODES = (AUTO, SINGLE, LAYER_SPLIT, DDP, FSDP2, TP, PP, CP, EP, CUSTOM)
MULTI_PROCESS_MODES = (DDP, FSDP2, TP, PP, CP, EP, CUSTOM)
_AXIS_OF_MODE = {DDP: "dp", FSDP2: "dp", TP: "tp", PP: "pp", CP: "cp", EP: "ep"}
_LABEL = {
    SINGLE: "one GPU",
    LAYER_SPLIT: "layers split across GPUs",
    DDP: "data parallel (DDP)",
    FSDP2: "sharded data parallel (FSDP2)",
    TP: "tensor parallel",
    PP: "pipeline parallel",
    CP: "context parallel",
    EP: "expert parallel",
    CUSTOM: "custom layout",
}
# A GPU is judged on its free memory less this share, the same headroom Core's UNSLOTH_PARALLEL=auto keeps for
# allocator fragmentation and the collective's buffers.
_FIT_MARGIN = 0.9


class ParallelPlanError(ValueError):
    """A refused layout: `.payload` is the contract's error object (code, mode, message, alternative)."""

    def __init__(
        self,
        code: str,
        mode: Optional[str],
        message: str,
        alternative: Optional[str] = None,
    ):
        super().__init__(message)
        self.payload = {"code": code, "mode": mode, "message": message, "alternative": alternative}


# --------------------------------------------------------------------------------------------- host


def _core_file(*parts: str) -> Optional[str]:
    """Path of a file inside the installed unsloth package, found without importing it (that patches torch)."""
    try:
        spec = importlib.util.find_spec("unsloth")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for root in spec.submodule_search_locations:
        path = os.path.join(root, *parts)
        if os.path.isfile(path):
            return path
    return None


_CORE_MODULES: dict = {}


def _core_module(name: str):
    """unsloth/<name>.py executed standalone (both modules import nothing heavy at module level); None if absent."""
    if name in _CORE_MODULES:
        return _CORE_MODULES[name]
    module = None
    path = _core_file(f"{name}.py")
    if path is not None:
        try:
            spec = importlib.util.spec_from_file_location(f"_studio_unsloth_{name}", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = (
                module  # dataclasses resolves the module of its classes by name
            )
            spec.loader.exec_module(module)
        except Exception:  # noqa: BLE001 -- an unreadable module only narrows what Studio offers
            sys.modules.pop(f"_studio_unsloth_{name}", None)
            module = None
    _CORE_MODULES[name] = module
    return module


def core_axes() -> dict:
    """Which parallel axes the installed Unsloth implements (files present, ep not reserved)."""
    env = _core_module("parallel_env")
    has_env = env is not None
    reserved = getattr(env, "_RESERVED", {}) if has_env else {}
    return {
        DDP: has_env,
        FSDP2: has_env and _core_file("models", "fsdp2_native.py") is not None,
        TP: has_env and _core_file("models", "tp_native.py") is not None,
        PP: has_env and _core_file("models", "pp_native.py") is not None,
        CP: has_env and _core_file("context_parallel.py") is not None,
        EP: has_env and "ep" not in reserved and _core_file("models", "ep_native.py") is not None,
    }


def _distributed_backend(device: str) -> tuple[Optional[str], Optional[str]]:
    """(backend, reason it is None)."""
    if sys.platform == "win32":
        return None, (
            "Windows has no NCCL, so multi-process training (DDP, FSDP2, tensor / pipeline / context / expert "
            "parallel) runs on Linux only. On Windows, Layer split trains one model across several GPUs."
        )
    if sys.platform == "darwin":
        return None, "Apple Silicon trains on one device."
    if device not in ("cuda", "rocm"):
        return (
            None,
            f"Multi-process training needs NVIDIA or AMD GPUs (this host trains on {device}).",
        )
    try:
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_nccl_available():
            return None, (
                "This PyTorch build has no NCCL / RCCL, which multi-process training needs. "
                "Layer split still trains one model across several GPUs."
            )
    except Exception as exc:  # noqa: BLE001
        return None, f"torch.distributed is unavailable ({exc})."
    return ("rccl" if device == "rocm" else "nccl"), None


def _gpu_rows() -> list:
    from utils.hardware import hardware as hw

    rows = []
    try:
        utilization = hw.get_visible_gpu_utilization()
    except Exception:  # noqa: BLE001
        utilization = {}
    for device in utilization.get("devices", []) or []:
        total = device.get("vram_total_gb")
        used = device.get("vram_used_gb")
        if device.get("index") is None or total is None:
            continue
        rows.append(
            {
                "index": int(device["index"]),
                "name": device.get("name") or device.get("gpu_name") or "GPU",
                "total_gb": round(float(total), 2),
                "free_gb": round(max(float(total) - float(used or 0.0), 0.0), 2),
            }
        )
    return rows


_HOST_CACHE: dict = {"at": 0.0, "value": None}
_HOST_LOCK = threading.Lock()


def detect_host(max_age_s: float = 5.0) -> dict:
    """The host half of the plan: platform, device, collective backend, GPUs, Unsloth's axes."""
    with _HOST_LOCK:
        cached = _HOST_CACHE["value"]
        if cached is not None and time.monotonic() - _HOST_CACHE["at"] < max_age_s:
            return cached
    from utils.hardware import hardware as hw

    device_type = hw.get_device()
    device = getattr(device_type, "value", str(device_type)).lower()
    if device == "cuda" and getattr(hw, "IS_ROCM", False):
        device = "rocm"
    backend, reason = _distributed_backend(device)
    gpus = _gpu_rows() if device in ("cuda", "rocm", "xpu") else []
    host = {
        "platform": {"win32": "windows", "darwin": "darwin"}.get(sys.platform, "linux"),
        "device": device,
        "distributed_backend": backend,
        "multi_process_reason": reason,
        "gpus": gpus,
        "core": core_axes(),
        "torch": _torch_version(),
        "sm": _sm(),
    }
    with _HOST_LOCK:
        _HOST_CACHE.update(at = time.monotonic(), value = host)
    return host


def _torch_version() -> tuple:
    torch = sys.modules.get("torch")
    try:
        major, minor = str(torch.__version__).split("+")[0].split(".")[:2]
        return (int(major), int(minor))
    except Exception:  # noqa: BLE001
        return (0, 0)


def _sm() -> Optional[int]:
    torch = sys.modules.get("torch")
    try:
        if (
            torch is not None
            and torch.cuda.is_available()
            and not getattr(torch.version, "hip", None)
        ):
            return int(torch.cuda.get_device_capability(0)[0])
    except Exception:  # noqa: BLE001
        pass
    return None


# -------------------------------------------------------------------------------------------- model


def _is_moe(config) -> bool:
    for owner in (config, getattr(config, "text_config", None)):
        if owner is None:
            continue
        for key in ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts"):
            value = getattr(owner, key, None)
            if isinstance(value, int) and value > 1:
                return True
    return False


def model_facts(
    model_name: str,
    *,
    hf_token: Optional[str] = None,
    training_type: str = "LoRA/QLoRA",
    load_in_4bit: bool = True,
    batch_size: int = 4,
    max_seq_length: int = 2048,
    lora_rank: int = 16,
    target_modules: Optional[list] = None,
    gradient_checkpointing: str = "unsloth",
    optimizer: str = "adamw_8bit",
) -> dict:
    """The model half: one-GPU training estimate, per-GPU sharded estimates, MoE / family / multimodal."""
    from utils.hardware import hardware as hw

    facts: dict = {
        "required_gb": None,
        "per_gpu_gb": {},
        "is_moe": False,
        "family": "",
        "multimodal": False,
        "mode": "full"
        if training_type == "Full Finetuning"
        else ("qlora" if load_in_4bit else "lora"),
        "trainer": "sft",
    }
    try:
        required_gb, meta = hw.estimate_required_model_memory_gb(
            model_name,
            hf_token = hf_token,
            training_type = training_type,
            load_in_4bit = load_in_4bit,
            batch_size = batch_size,
            max_seq_length = max_seq_length,
            lora_rank = lora_rank,
            target_modules = target_modules,
            gradient_checkpointing = gradient_checkpointing,
            optimizer = optimizer,
        )
    except Exception:  # noqa: BLE001 -- an unestimable model plans without a size
        required_gb, meta = None, {}
    facts["required_gb"] = None if required_gb is None else round(float(required_gb), 2)
    for key, value in (meta.get("vram_breakdown") or {}).items():
        if key.startswith("min_per_gpu_") and value is not None:
            facts["per_gpu_gb"][int(key.rsplit("_", 1)[1])] = float(value)
    try:
        estimate_model = hw._resolve_model_identifier_for_gpu_estimate(
            model_name, hf_token = hf_token
        )
        config = hw._load_config_for_gpu_estimate(estimate_model, hf_token = hf_token)
    except Exception:  # noqa: BLE001
        config = None
    if config is not None:
        facts["is_moe"] = _is_moe(config)
        facts["family"] = str(getattr(config, "model_type", "") or "")
        facts["multimodal"] = getattr(config, "vision_config", None) is not None
    return facts


# --------------------------------------------------------------------------------------------- plan

_SINGLE_PROCESS_FEATURES = (
    ("is_embedding", "embedding models"),
    ("is_decision", "decision models"),
    ("batch_members", "batched multi-LoRA runs"),
    ("drafter", "drafter training"),
    ("is_dataset_audio", "audio datasets"),
)


def _single_process_feature(request: dict) -> Optional[str]:
    for key, label in _SINGLE_PROCESS_FEATURES:
        if request.get(key):
            return label
    offload = request.get("offload_layers")
    if offload not in (None, 0, "0", "", False):
        return "Offload layers (block swap)"
    return None


def _spec_for(mode: str, world: int, custom: Optional[str]) -> str:
    if mode == DDP:
        return f"dp={world}"
    if mode == FSDP2:
        return f"dp={world},fsdp=1"
    if mode in (TP, PP, CP, EP):
        return f"{_AXIS_OF_MODE[mode]}={world}"
    return str(custom or "").strip()


def _parse_spec(raw: str, world: int) -> dict:
    """Core's parser, plus the world check; {'dp','tp','pp','cp','ep','fsdp'} sizes."""
    env = _core_module("parallel_env")
    if env is None:
        raise ParallelPlanError(
            "parallel_mode_unavailable", CUSTOM, "This Unsloth version has no parallel launcher."
        )
    if not raw:
        raise ParallelPlanError(
            "parallel_bad_spec", CUSTOM, 'Custom layout needs a parallel_spec such as "dp=2,tp=2".'
        )
    ep_requested = any(
        part.split("=", 1)[0].strip().lower() in ("ep", "ep_size")
        for part in raw.replace(";", ",").split(",")
    )
    try:
        spec = env.parse_parallel_spec(raw, source = "parallel_spec")
    except ValueError as exc:
        code = "parallel_mode_unavailable" if ep_requested else "parallel_bad_spec"
        raise ParallelPlanError(code, CUSTOM, str(exc)) from None
    if spec.get("auto"):
        raise ParallelPlanError(
            "parallel_bad_spec", CUSTOM, 'Use parallel_mode "auto" instead of the spec "auto".'
        )
    sizes = {k: int(spec.get(k, 1)) for k in ("tp", "pp", "cp")}
    sizes["ep"] = int(spec.get("ep", 1)) if not isinstance(spec.get("ep"), bool) else 1
    model_ranks = sizes["tp"] * sizes["pp"] * sizes["cp"]
    if world % model_ranks:
        raise ParallelPlanError(
            "parallel_world_mismatch",
            CUSTOM,
            f"The layout {raw!r} needs a multiple of {model_ranks} GPUs; this run has {world}.",
        )
    dp = int(
        spec.get("dp") or spec.get("dp_shard") or spec.get("dp_replicate") or world // model_ranks
    )
    if dp * model_ranks != world:
        raise ParallelPlanError(
            "parallel_world_mismatch",
            CUSTOM,
            f"The layout {raw!r} uses {dp * model_ranks} GPUs; this run has {world}.",
        )
    sizes["dp"] = dp
    sizes["fsdp"] = bool(spec.get("fsdp")) or int(spec.get("dp_shard", 1) or 1) > 1
    return sizes


def _core_check(sizes: dict, model: dict, host: dict, request: dict) -> None:
    """Unsloth's capability table on what is known before load; NotImplementedError -> parallel_unsupported."""
    caps = _core_module("parallel_capabilities")
    if caps is None or not hasattr(caps, "Query"):
        return
    fields = getattr(caps.Query, "__dataclass_fields__", {})
    values = {
        "trainer": model.get("trainer", "sft"),
        "family": model.get("family", ""),
        "multimodal": bool(model.get("multimodal")),
        "mode": model.get("mode", "lora"),
        "packing": bool(request.get("packing")),
        "torch": tuple(host.get("torch") or (0, 0)),
        "sm": host.get("sm"),
        "dp": sizes["dp"],
        "tp": sizes["tp"],
        "pp": sizes["pp"],
        "cp": sizes["cp"],
        "fsdp2": bool(sizes["fsdp"]),
        "moe": bool(model.get("is_moe")),
        "optim": str(request.get("optim") or "").lower(),
    }
    query = caps.Query(**{k: v for k, v in values.items() if k in fields})
    try:
        caps.check(query)
    except NotImplementedError as exc:
        text = str(exc)
        message, _, alternative = text.partition(" Instead: ")
        alternative = alternative.rsplit(" (layout", 1)[0].strip().rstrip(".") or None
        raise ParallelPlanError(
            "parallel_unsupported", request.get("parallel_mode"), message.strip(), alternative
        )


def _fits(need_gb: Optional[float], gpus: list) -> bool:
    if need_gb is None:
        return False
    return all(g["free_gb"] * _FIT_MARGIN >= need_gb for g in gpus)


def _usable_gpus(host: dict, gpu_ids: Optional[list]) -> list:
    rows = list(host.get("gpus") or [])
    if not gpu_ids:
        return rows
    by_index = {g["index"]: g for g in rows}
    missing = [i for i in gpu_ids if i not in by_index]
    if missing:
        raise ParallelPlanError(
            "parallel_gpu_ids",
            None,
            f"GPU(s) {missing} are not visible to Studio; visible: {sorted(by_index)}.",
        )
    return [by_index[i] for i in gpu_ids]


def _mode_rows(host: dict, model: Optional[dict], request: dict, n_gpus: int) -> list:
    """Per mode: can it run here (and, with a model, for this model)."""
    rows = []
    multi_reason = host.get("multi_process_reason")
    feature = _single_process_feature(request)
    for mode in MODES:
        reason = None
        if mode == LAYER_SPLIT and host.get("device") not in ("cuda", "rocm", "xpu"):
            reason = "Layer split needs several CUDA, ROCm or XPU GPUs."
        elif mode in MULTI_PROCESS_MODES:
            if multi_reason:
                reason = multi_reason
            elif mode != CUSTOM and not (host.get("core") or {}).get(mode, False):
                reason = (
                    "This Unsloth version has no expert parallel yet."
                    if mode == EP
                    else f"This Unsloth version has no {_LABEL[mode]}."
                )
            elif feature:
                reason = f"{feature} train in one process; use One GPU or Layer split."
            elif mode == EP and model is not None and not model.get("is_moe"):
                reason = "Expert parallel needs a mixture-of-experts model."
        min_gpus = 1 if mode in (AUTO, SINGLE) else 2
        if reason is None and min_gpus > 1 and n_gpus < min_gpus:
            reason = f"Needs at least {min_gpus} GPUs; {n_gpus} usable."
        if reason is None and model is not None and mode in (TP, PP, CP, EP, FSDP2) and n_gpus >= 2:
            try:
                _core_check(
                    _sizes_for(mode, n_gpus), model, host, {**request, "parallel_mode": mode}
                )
            except ParallelPlanError as exc:
                reason = exc.payload["message"]
        rows.append(
            {"mode": mode, "available": reason is None, "reason": reason, "min_gpus": min_gpus}
        )
    return rows


def _sizes_for(mode: str, world: int) -> dict:
    sizes = {"dp": 1, "tp": 1, "pp": 1, "cp": 1, "ep": 1, "fsdp": False}
    if mode in (DDP, FSDP2):
        sizes["dp"] = world
        sizes["fsdp"] = mode == FSDP2
    elif mode == EP:
        sizes["dp"] = world  # experts split over the data parallel ranks
        sizes["ep"] = world
    else:
        sizes[_AXIS_OF_MODE[mode]] = world
    return sizes


def _estimate_row(model: dict, n: int) -> dict:
    single = model.get("required_gb")
    sharded = (model.get("per_gpu_gb") or {}).get(max(n, 1))
    return {"single": single, "ddp": single, "fsdp2": sharded, "layer_split": sharded}


def plan(request: dict, host: dict, model: Optional[dict]) -> dict:
    """The contract's plan object for one request. Raises ParallelPlanError for a refused explicit mode."""
    requested = request.get("parallel_mode") or None
    if requested is not None and requested not in MODES:
        raise ParallelPlanError(
            "parallel_bad_spec",
            requested,
            f"Unknown parallel_mode {requested!r}; one of {', '.join(MODES)}.",
        )
    custom = request.get("parallel_spec")
    if custom and requested != CUSTOM:
        raise ParallelPlanError(
            "parallel_bad_spec",
            requested,
            'parallel_spec is only read with parallel_mode "custom".',
        )
    gpu_ids = list(request.get("gpu_ids") or [])
    gpus = _usable_gpus(host, gpu_ids)
    model = model or {}
    warnings: list = []
    feature = _single_process_feature(request)
    multi_reason = host.get("multi_process_reason") or (
        f"{feature} train in one process." if feature else None
    )
    need = model.get("required_gb")

    def _result(
        mode: str,
        chosen: list,
        reason: str,
        spec: Optional[str] = None,
    ) -> dict:
        multi = mode in MULTI_PROCESS_MODES
        n = len(chosen)
        if multi and len({(g["name"], round(g["total_gb"])) for g in chosen}) > 1:
            warnings.append("These GPUs differ; data parallel runs at the pace of the slowest.")
        return {
            "requested_mode": requested or AUTO,
            "mode": mode,
            "world_size": n if multi else 1,
            "gpu_ids": [g["index"] for g in chosen] if chosen else (gpu_ids or None),
            "spec": spec if multi else None,
            "launcher": "multi_process" if multi else "single_process",
            "distributed_backend": host.get("distributed_backend") if multi else None,
            "reason": reason,
            "estimate": {"required_gb": need, "per_gpu_gb": _estimate_row(model, n or 1)},
            "is_moe": bool(model.get("is_moe")),
            "warnings": warnings,
            "modes": _mode_rows(host, model or None, request, len(gpus)),
            "error": None,
        }

    def _legacy(why: str) -> dict:
        # Today's behaviour: explicit gpu_ids as given; otherwise the start-time auto selection (the fewest GPUs that
        # fit, layers split when that is more than one) still picks, so gpu_ids stays unset.
        chosen = gpus if gpu_ids else []
        if (
            len(gpus) <= 1
            or (gpu_ids and len(gpu_ids) == 1)
            or (need is not None and _fits(need, gpus[:1]))
        ):
            return _result(SINGLE, chosen[:1], why)
        return _result(
            LAYER_SPLIT,
            chosen,
            why + " The model does not fit one GPU, so its layers are split across them.",
        )

    if requested in (None, AUTO):
        if host.get("device") not in ("cuda", "rocm", "xpu") or len(gpus) <= 1:
            return _result(SINGLE, gpus[:1] if gpu_ids else [], "One device: single-GPU training.")
        if multi_reason:
            return _legacy(multi_reason)
        core = host.get("core") or {}
        if model.get("is_moe") and core.get(EP):
            try:
                _core_check(
                    _sizes_for(EP, len(gpus)), model, host, {**request, "parallel_mode": EP}
                )
                return _result(
                    EP,
                    gpus,
                    "Mixture-of-experts model: its experts are split across the GPUs (expert parallel).",
                    _spec_for(EP, len(gpus), None),
                )
            except ParallelPlanError:
                pass
        if need is None:
            return _legacy("The model size could not be estimated, so Studio keeps one process.")
        if _fits(need, gpus) and core.get(DDP):
            return _result(
                DDP,
                gpus,
                f"The model needs about {need:.1f} GB per GPU and every GPU has room, so each GPU trains a full copy (DDP).",
                _spec_for(DDP, len(gpus), None),
            )
        shard = (model.get("per_gpu_gb") or {}).get(len(gpus))
        if core.get(FSDP2) and shard is not None and _fits(shard, gpus):
            try:
                _core_check(
                    _sizes_for(FSDP2, len(gpus)), model, host, {**request, "parallel_mode": FSDP2}
                )
                return _result(
                    FSDP2,
                    gpus,
                    f"The model needs about {need:.1f} GB, more than one GPU holds; sharded over {len(gpus)} GPUs it needs about {shard:.1f} GB each (FSDP2).",
                    _spec_for(FSDP2, len(gpus), None),
                )
            except ParallelPlanError:
                pass
        return _legacy(
            f"The model needs about {need:.1f} GB and does not fit each GPU whole or sharded."
        )

    if requested == SINGLE:
        if len(gpu_ids) > 1:
            raise ParallelPlanError(
                "parallel_gpu_ids", SINGLE, "One GPU mode takes exactly one GPU id."
            )
        chosen = gpus[:1] if gpus else []
        if not gpu_ids and len(gpus) > 1:
            chosen = [max(gpus, key = lambda g: (g["free_gb"], -g["index"]))]
        return _result(SINGLE, chosen, "Single-GPU training, as requested.")

    if requested == LAYER_SPLIT:
        if host.get("device") not in ("cuda", "rocm", "xpu"):
            raise ParallelPlanError(
                "parallel_platform",
                LAYER_SPLIT,
                "Layer split needs several CUDA, ROCm or XPU GPUs.",
            )
        if len(gpus) < 2:
            raise ParallelPlanError(
                "parallel_gpu_ids",
                LAYER_SPLIT,
                f"Layer split needs at least 2 GPUs; {len(gpus)} usable.",
            )
        return _result(
            LAYER_SPLIT, gpus, f"Layers split across {len(gpus)} GPUs in one process, as requested."
        )

    # Multi-process modes.
    if host.get("multi_process_reason"):
        code = (
            "parallel_no_backend"
            if "NCCL" in host["multi_process_reason"] and host.get("platform") == "linux"
            else "parallel_platform"
        )
        raise ParallelPlanError(
            code, requested, host["multi_process_reason"], 'parallel_mode "layer_split" or "single"'
        )
    if feature:
        raise ParallelPlanError(
            "parallel_feature",
            requested,
            f"{feature} train in one process.",
            'parallel_mode "single" or "layer_split"',
        )
    if len(gpus) < 2:
        raise ParallelPlanError(
            "parallel_not_enough_gpus",
            requested,
            f"{_LABEL[requested].capitalize()} needs at least 2 GPUs; {len(gpus)} usable.",
        )
    if requested != CUSTOM and not (host.get("core") or {}).get(requested):
        msg = (
            "This Unsloth version has no expert parallel yet."
            if requested == EP
            else f"This Unsloth version has no {_LABEL[requested]}."
        )
        raise ParallelPlanError(
            "parallel_mode_unavailable", requested, msg, 'parallel_mode "ddp" or "fsdp2"'
        )
    if requested == EP and not model.get("is_moe", True):
        raise ParallelPlanError(
            "parallel_unsupported",
            EP,
            "Expert parallel needs a mixture-of-experts model.",
            'parallel_mode "ddp" or "fsdp2"',
        )
    world = len(gpus)
    spec = _spec_for(requested, world, custom)
    sizes = _parse_spec(spec, world) if requested == CUSTOM else _sizes_for(requested, world)
    if requested == CUSTOM and sizes.get("ep", 1) > 1 and not (host.get("core") or {}).get(EP):
        raise ParallelPlanError(
            "parallel_mode_unavailable", CUSTOM, "This Unsloth version has no expert parallel yet."
        )
    _core_check(sizes, model, host, request)
    if requested == DDP and need is not None and not _fits(need, gpus):
        warnings.append(
            f"The model needs about {need:.1f} GB per GPU, which may not fit; FSDP2 shards it."
        )
    return _result(
        requested, gpus, f"{_LABEL[requested].capitalize()} on {world} GPUs, as requested.", spec
    )


# ----------------------------------------------------------------------------------------- wrappers


def capabilities(host: Optional[dict] = None) -> dict:
    host = host or detect_host()
    gpus = host.get("gpus") or []
    return {
        "platform": host.get("platform"),
        "device": host.get("device"),
        "distributed_backend": host.get("distributed_backend"),
        "multi_process": host.get("multi_process_reason") is None,
        "multi_process_reason": host.get("multi_process_reason"),
        "gpus": gpus,
        "identical_gpus": len({(g["name"], round(g["total_gb"])) for g in gpus}) <= 1,
        "core": dict(host.get("core") or {}),
        "modes": _mode_rows(host, None, {}, len(gpus)),
    }


def plan_for_request(
    request: dict,
    *,
    hf_token: Optional[str] = None,
    host: Optional[dict] = None,
    inline_error: bool = False,
) -> dict:
    """plan() on the live host and model. A refusal raises ParallelPlanError (/start), or with inline_error comes back
    in `error` beside the auto plan (/plan, so the Train page can show it next to the default)."""
    host = host or detect_host()
    model = None
    if request.get("model_name"):
        model = model_facts(
            request["model_name"],
            hf_token = hf_token,
            training_type = request.get("training_type") or "LoRA/QLoRA",
            load_in_4bit = bool(request.get("load_in_4bit", True)),
            batch_size = int(request.get("batch_size") or 4),
            max_seq_length = int(request.get("max_seq_length") or 2048),
            lora_rank = int(request.get("lora_r") or 16),
            target_modules = request.get("target_modules"),
            gradient_checkpointing = request.get("gradient_checkpointing") or "unsloth",
            optimizer = request.get("optim") or "adamw_8bit",
        )
    try:
        return plan(request, host, model)
    except ParallelPlanError as exc:
        if not inline_error:
            raise
        fallback = {**request, "parallel_mode": AUTO, "parallel_spec": None}
        try:
            result = plan(fallback, host, model)
        except ParallelPlanError:
            result = plan({**fallback, "gpu_ids": None}, host, model)
        return {**result, "requested_mode": request.get("parallel_mode"), "error": exc.payload}
