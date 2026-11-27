# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Picks a Studio training run's multi-GPU layout before anything loads.

`plan(request, host, model)` is pure (host / model facts come in as dicts). Layouts Unsloth's capability table
refuses are refused here, with Unsloth's reason, before any VRAM moves.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import threading
import time
import types
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
# Same headroom Core's UNSLOTH_PARALLEL=auto keeps.
_FIT_MARGIN = 0.9


class ParallelPlanError(ValueError):
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
    """Found without importing unsloth (that patches torch)."""
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
    names: list = []
    devices = utilization.get("devices", []) or []
    if any(not d.get("name") for d in devices):
        try:
            from utils.hardware import nvidia

            # nvidia-smi: no CUDA context in the server.
            names = [row[3] if row else None for row in (nvidia.get_cuda_device_info() or [])]
        except Exception:  # noqa: BLE001
            names = []
    try:
        # ROCm cards torch has no kernels for, dropped as the legacy auto selection does.
        uncovered = (
            set(hw.rocm_gpu_ids_without_torch_kernels()) if getattr(hw, "IS_ROCM", False) else set()
        )
    except Exception:  # noqa: BLE001
        uncovered = set()
    try:
        from utils.hardware import nvidia
        capabilities = {} if getattr(hw, "IS_ROCM", False) else nvidia.get_compute_capabilities()
    except Exception:  # noqa: BLE001
        capabilities = {}
    for ordinal, device in enumerate(devices):
        total = device.get("vram_total_gb")
        used = device.get("vram_used_gb")
        if device.get("index") is None or total is None or device["index"] in uncovered:
            continue
        visible = device.get("visible_ordinal")
        visible = ordinal if visible is None else visible
        index = int(device["index"])
        rows.append(
            {
                "index": index,
                "name": device.get("name")
                or device.get("gpu_name")
                or (names[visible] if 0 <= visible < len(names) else None)
                or "GPU",
                "total_gb": round(float(total), 2),
                "free_gb": round(max(float(total) - float(used or 0.0), 0.0), 2),
                "sm": (capabilities.get(index) or (None,))[0],
            }
        )
    return rows


_HOST_CACHE: dict = {"at": 0.0, "value": None}
_HOST_LOCK = threading.Lock()


def detect_host(max_age_s: float = 5.0) -> dict:
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
        # False on a UUID / MIG mask: /start refuses a plan that pins a subset of it.
        "pins_gpu_ids": _mask_takes_gpu_ids(),
        "transformers": _transformers_version(),
        "torch": _torch_version(),
    }
    with _HOST_LOCK:
        _HOST_CACHE.update(at = time.monotonic(), value = host)
    return host


def _mask_takes_gpu_ids() -> bool:
    try:
        from utils.hardware import hardware as hw
        return bool(hw._get_parent_visible_gpu_spec().get("supports_explicit_gpu_ids", True))
    except Exception:  # noqa: BLE001 -- the same default as the /start check
        return True


def _torch_version() -> tuple:
    torch = sys.modules.get("torch")
    try:
        major, minor = str(torch.__version__).split("+")[0].split(".")[:2]
        return (int(major), int(minor))
    except Exception:  # noqa: BLE001
        return (0, 0)


def _transformers_version() -> tuple:
    try:
        from importlib.metadata import version
        major, minor = version("transformers").split(".")[:2]
        return (int(major), int("".join(ch for ch in minor if ch.isdigit()) or 0))
    except Exception:  # noqa: BLE001
        return (0, 0)


def _sm(gpus: list) -> Optional[int]:
    majors = [g.get("sm") for g in gpus if g.get("sm") is not None]
    return min(majors) if majors else None


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
    from utils.hardware import hardware as hw

    facts: dict = {
        "required_gb": None,
        "per_gpu_gb": {},
        "is_moe": False,
        "family": "",
        "multimodal": False,
        "hybrid": False,
        "windowed": False,
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
        facts.update(_config_traits(config))
        text = (
            config.get_text_config()
            if callable(getattr(config, "get_text_config", None))
            else config
        )
        experts = getattr(text, "num_local_experts", None) or getattr(text, "num_experts", None)
        if isinstance(experts, int):
            facts["num_experts"] = experts
    return facts


def _config_traits(config) -> dict:
    """Core's own model traits (family, hybrid, multimodal, windowed) from a config, so its refusals apply."""
    caps = _core_module("parallel_capabilities")
    traits = getattr(caps, "_model_traits", None)
    if traits is None:
        return {}
    try:
        family, hybrid, multimodal, windowed = traits(types.SimpleNamespace(config = config))
    except Exception:  # noqa: BLE001
        return {}
    return {
        "family": str(family or ""),
        "hybrid": bool(hybrid),
        "multimodal": bool(multimodal),
        "windowed": bool(windowed),
    }


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


def _spec_example(world: int) -> str:
    """dp over every GPU: a hint every run accepts (a tp axis is refused for some models)."""
    return f"dp={max(world, 1)}"


def _parse_spec(raw: str, world: int) -> dict:
    env = _core_module("parallel_env")
    if env is None:
        raise ParallelPlanError(
            "parallel_mode_unavailable", CUSTOM, "This Unsloth version has no parallel launcher."
        )
    if not raw:
        raise ParallelPlanError(
            "parallel_bad_spec",
            CUSTOM,
            f'Custom layout needs a layout such as "{_spec_example(world)}".',
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
    if not spec:
        raise ParallelPlanError(
            "parallel_bad_spec",
            CUSTOM,
            f'Custom layout needs a layout such as "{_spec_example(world)}".',
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
    sizes["cp_backend"] = spec.get("cp_backend")
    sizes["pp_schedule"] = spec.get("pp_schedule")
    return sizes


def _core_check(
    sizes: dict,
    model: dict,
    host: dict,
    request: dict,
    gpus: Optional[list] = None,
) -> None:
    caps = _core_module("parallel_capabilities")
    if caps is None or not hasattr(caps, "Query"):
        return
    fields = getattr(caps.Query, "__dataclass_fields__", {})
    cp_backend = sizes.get("cp_backend")
    if cp_backend == "auto":
        cp_backend = None
    sm = _sm(gpus or []) if any(g.get("sm") is not None for g in gpus or []) else host.get("sm")
    ep = int(sizes.get("ep") or 1)
    ep_blocker = model.get("ep_blocker")
    if ep_blocker is None and ep > 1 and callable(getattr(caps, "_ep_blocker", None)):
        # Only the model_type is known here: Core's table of families expert parallel cannot shard.
        # A 4-bit load can change the experts layout (gpt-oss: per-expert Linear4bit lists).
        family = str(model.get("family") or "")
        four_bit = model.get("mode") == "qlora" or (
            model.get("mode") != "full"
            and "bnb-4bit" in str(request.get("model_name") or "").lower()
        )
        try:
            ep_blocker = caps._ep_blocker(
                family, ep, load_in_4bit = four_bit, experts = model.get("num_experts")
            )
        except TypeError:  # a Core before the load_in_4bit / experts arguments
            try:
                ep_blocker = caps._ep_blocker(family, ep, load_in_4bit = four_bit)
            except TypeError:
                ep_blocker = caps._ep_blocker(family, ep)
    values = {
        "trainer": model.get("trainer", "sft"),
        "family": model.get("family", ""),
        "multimodal": bool(model.get("multimodal")),
        "hybrid": bool(model.get("hybrid")),
        "windowed": bool(model.get("windowed")),
        "mode": model.get("mode", "lora"),
        "packing": bool(request.get("packing")),
        "torch": tuple(host.get("torch") or (0, 0)),
        "sm": sm,
        "dp": sizes["dp"],
        "tp": sizes["tp"],
        "pp": sizes["pp"],
        "cp": sizes["cp"],
        "ep": ep,
        "ep_blocker": ep_blocker,
        "fsdp2": bool(sizes["fsdp"]),
        "moe": bool(model.get("is_moe")),
        "optim": str(request.get("optim") or "").lower(),
        "cp_backend": cp_backend,
        "pp_schedule": sizes.get("pp_schedule"),
    }
    query = caps.Query(**{k: v for k, v in values.items() if k in fields and v is not None})
    try:
        # cp_backend "auto" resolves only at load, so the cell check is left to Unsloth then.
        caps.check(query, cells = not (sizes["cp"] > 1 and cp_backend is None))
    except NotImplementedError as exc:
        text = str(exc)
        message, _, alternative = text.partition(" Instead: ")
        layout = text.rsplit(" (layout ", 1)[1] if " (layout " in text else ""
        alternative = alternative.rsplit(" (layout", 1)[0].strip().rstrip(".") or None
        if _REASON_BELOW in message:
            # Core names the model's own reason only in its layout tail ("...; models/tp_native.py: <why>)").
            tail = layout[:-1] if layout.endswith(")") else layout
            detail = tail.rsplit("; ", 1)[-1].partition(": ")[2].strip()
            if not detail and ep > 1 and ep_blocker and message.startswith("Unsloth: ep x model:"):
                detail = str(ep_blocker).strip().rstrip(".")
            message = message.split(": models/", 1)[0] + (f": {detail}." if detail else ".")
        raise ParallelPlanError(
            "parallel_unsupported",
            request.get("parallel_mode"),
            _plain_core_message(message),
            alternative,
        )


_REASON_BELOW = "(its reason follows the layout below)"
# Core's rule prefix ("Unsloth: pp x packing: ..."), dropped from the Train page message.
_CORE_RULE_PREFIX = re.compile(r"^(?:Unsloth:\s*)?(?:[a-z0-9_]+(?: x [a-z0-9_]+)+:\s*)?")


def _plain_core_message(message: str) -> str:
    text = _CORE_RULE_PREFIX.sub("", message.strip(), count = 1)
    return text[:1].upper() + text[1:] if text else message.strip()


def _too_big(need: float, largest: dict, host: dict) -> str:
    """Why a run that DDP could not take is sharded, claiming only what was checked: the freest card."""
    if not _fits(need, [largest]):
        return "more than any one GPU has room for"
    if not host.get("pins_gpu_ids", True):
        return "and not every GPU has room for a full copy, while this GPU mask cannot keep a run to one card"
    return "and not every GPU has room for a full copy"


def _fits(need_gb: Optional[float], gpus: list) -> bool:
    if need_gb is None:
        return False
    return all(g["free_gb"] * _FIT_MARGIN >= need_gb for g in gpus)


def _usable_gpus(host: dict, gpu_ids: Optional[list]) -> list:
    rows = list(host.get("gpus") or [])
    if not gpu_ids:
        return rows
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ParallelPlanError(
            "parallel_gpu_ids", None, f"gpu_ids {list(gpu_ids)} names a GPU twice."
        )
    by_index = {g["index"]: g for g in rows}
    missing = [i for i in gpu_ids if i not in by_index]
    if missing:
        raise ParallelPlanError(
            "parallel_gpu_ids",
            None,
            f"GPU(s) {missing} are not visible to Studio; visible: {sorted(by_index)}.",
        )
    return [by_index[i] for i in gpu_ids]


def _mode_rows(host: dict, model: Optional[dict], request: dict, gpus: list) -> list:
    n_gpus = len(gpus)
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
        if (
            reason is None
            and model is not None
            and mode in (DDP, TP, PP, CP, EP, FSDP2)
            and n_gpus >= 2
        ):
            try:
                _core_check(
                    _sizes_for(mode, n_gpus), model, host, {**request, "parallel_mode": mode}, gpus
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
        sizes["dp"] = world
        sizes["ep"] = world
    else:
        sizes[_AXIS_OF_MODE[mode]] = world
    return sizes


def _estimate_row(model: dict, n: int) -> dict:
    single = model.get("required_gb")
    sharded = (model.get("per_gpu_gb") or {}).get(max(n, 1))
    return {"single": single, "ddp": single, "fsdp2": sharded, "layer_split": sharded}


def _gpus_differ(gpus: list) -> bool:
    return len({(g["name"], round(g["total_gb"])) for g in gpus}) > 1


def _mixed_gpu_note(mode: str) -> str:
    if mode in (DDP, FSDP2, EP):
        return "These GPUs differ; data parallel runs at the pace of the slowest."
    return "These GPUs differ; every GPU holds an equal share, so the slowest and smallest set the pace and the limit."


def plan(request: dict, host: dict, model: Optional[dict]) -> dict:
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
    notes: list = []  # host facts, shown muted; warnings are risks of this run
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
        if multi and _gpus_differ(chosen):
            notes.append(_mixed_gpu_note(mode))
        env: dict = {}
        sizes = None
        if mode == FSDP2:
            sizes = {"fsdp": True}
        elif mode == CUSTOM and spec:
            try:
                sizes = _parse_spec(spec, n)
            except ParallelPlanError:
                sizes = None
        if sizes and sizes.get("fsdp") and need is not None and chosen and not _fits(need, chosen):
            # Core keeps FSDP2 blocks gathered when the weights alone fit a card, ignoring optimizer state.
            env["UNSLOTH_FSDP2_LAYOUT"] = "sharded"
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
            "notes": notes,
            "modes": _mode_rows(host, model or None, request, gpus),
            "error": None,
            "env": env if multi else {},
        }

    def _legacy(why: str) -> dict:
        # Pre-parallel behaviour: without explicit gpu_ids the start-time auto selection still picks.
        chosen = gpus if gpu_ids else []
        if gpu_ids and len(gpu_ids) > 1:
            return _result(LAYER_SPLIT, chosen, why + " Layers are split across the chosen GPUs.")
        freest = sorted(gpus, key = lambda g: -g["free_gb"])[:1]
        if len(gpus) <= 1 or gpu_ids or (need is not None and _fits(need, freest)):
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
        if need is None:
            return _legacy("The model size could not be estimated, so Studio keeps one process.")
        # Never auto-pick expert parallel: DDP has no per-layer all-to-all (OLMoE-1B-7B, 2 GPUs: DDP 240 s,
        # ep 570 s), and only Core knows what an ep rank holds, while FSDP2 shards any model.
        if _fits(need, gpus) and core.get(DDP):
            try:
                _core_check(
                    _sizes_for(DDP, len(gpus)), model, host, {**request, "parallel_mode": DDP}, gpus
                )
                return _result(
                    DDP,
                    gpus,
                    f"The model needs about {need:.1f} GB per GPU and every GPU has room, so each GPU trains a full copy (DDP).",
                    _spec_for(DDP, len(gpus), None),
                )
            except ParallelPlanError:
                pass
        # Multi-process layouts give equal shares, so the smallest card caps them: train on the largest alone.
        largest = max(gpus, key = lambda g: (g["free_gb"], -g["index"]))
        if host.get("pins_gpu_ids", True) and _fits(need, [largest]):
            return _result(
                SINGLE,
                [largest],
                f"The model needs about {need:.1f} GB"
                + ("" if _fits(need, gpus) else ", and not every GPU has room for a full copy")
                + f"; GPU {largest['index']} ({largest['name']}) holds it whole, so it trains there alone.",
            )
        shard = (model.get("per_gpu_gb") or {}).get(len(gpus))
        if core.get(FSDP2) and shard is not None and _fits(shard, gpus):
            try:
                _core_check(
                    _sizes_for(FSDP2, len(gpus)),
                    model,
                    host,
                    {**request, "parallel_mode": FSDP2},
                    gpus,
                )
                return _result(
                    FSDP2,
                    gpus,
                    f"The model needs about {need:.1f} GB, {_too_big(need, largest, host)}; sharded over "
                    f"{len(gpus)} GPUs it needs about {shard:.1f} GB each (FSDP2).",
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
    if requested == CUSTOM:
        core = host.get("core") or {}
        for axis, mode in (("ep", EP), ("tp", TP), ("pp", PP), ("cp", CP)):
            if sizes.get(axis, 1) > 1 and not core.get(mode):
                msg = (
                    "This Unsloth version has no expert parallel yet."
                    if mode == EP
                    else f"This Unsloth version has no {_LABEL[mode]}."
                )
                raise ParallelPlanError("parallel_mode_unavailable", CUSTOM, msg)
        if sizes.get("fsdp") and not core.get(FSDP2):
            raise ParallelPlanError(
                "parallel_mode_unavailable", CUSTOM, f"This Unsloth version has no {_LABEL[FSDP2]}."
            )
    _core_check(sizes, model, host, request, gpus)
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
        "modes": _mode_rows(host, None, {}, gpus),
    }


def plan_for_request(
    request: dict,
    *,
    hf_token: Optional[str] = None,
    host: Optional[dict] = None,
    inline_error: bool = False,
) -> dict:
    """With inline_error a refusal comes back in `error` beside the auto plan instead of raising."""
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
