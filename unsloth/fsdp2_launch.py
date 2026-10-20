# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FSDP launch detection from the launcher env only (no torch): identical on every rank."""

import importlib
import importlib.util
import os
import sys

__all__ = [
    "unsloth_fsdp2_requested",
    "unsloth_fsdp2_native_requested",
    "fsdp_version_from_env",
    "apply_unsloth_fsdp2_launch_env",
    "NATIVE_BACKEND_TABLE",
    "native_backend_conflicts",
    "refuse_native_backend_conflicts",
]

_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("0", "false", "no", "off")


def _env_flag(name):
    return str(os.environ.get(name, "")).strip().lower() in _TRUTHY


def _env_world_size():
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return 1


def unsloth_fsdp2_requested():
    return _env_flag("UNSLOTH_FSDP2") or _env_value("UNSLOTH_FSDP2") == "native"


def _env_value(name):
    return str(os.environ.get(name, "")).strip().lower()


def unsloth_fsdp2_native_requested():
    """UNSLOTH_FSDP2=1 / native under WORLD_SIZE > 1 selects Unsloth's own fully_shard wrap
    (`unsloth.models.fsdp2_native`). UNSLOTH_FSDP2_BACKEND=accelerate, or an `accelerate launch`
    FSDP config (ACCELERATE_USE_FSDP) without an explicit backend, keeps accelerate's wrap."""
    if not unsloth_fsdp2_requested() or _env_world_size() <= 1:
        return False
    backend = _env_value("UNSLOTH_FSDP2_BACKEND")
    if backend in ("accelerate", "hf", "transformers"):
        return False
    if backend not in ("", "native", "unsloth", "torch"):
        raise ValueError(
            f"Unsloth: UNSLOTH_FSDP2_BACKEND={os.environ.get('UNSLOTH_FSDP2_BACKEND')} is not one "
            "of 'native' (default) or 'accelerate'."
        )
    if backend == "" and _env_flag("ACCELERATE_USE_FSDP"):
        return False
    return True


def fsdp_version_from_env():
    """0 for no FSDP, else 1 or 2. accelerate builds FSDP1 when ACCELERATE_USE_FSDP names no
    version; FSDP_VERSION="0" means "not FSDP"."""
    if unsloth_fsdp2_requested() and _env_world_size() > 1:
        return 2
    version = str(os.environ.get("FSDP_VERSION", "")).strip()
    use_fsdp = _env_flag("ACCELERATE_USE_FSDP")
    if version in ("", "0"):
        return 1 if use_fsdp else 0
    try:
        return int(version)
    except ValueError:
        return 1 if use_fsdp else 0


def _accelerate_fsdp2_support():
    """None when accelerate is not found, else whether it has FSDP2 (1.6) and FSDP2 state dict
    options (1.7; 1.6 saves FULL_STATE_DICT checkpoints as each rank's DTensor shard). Read from
    the source: importing accelerate here would import torch."""
    try:
        spec = importlib.util.find_spec("accelerate")
    except (ImportError, ValueError):
        return None
    for location in getattr(spec, "submodule_search_locations", None) or ():
        try:
            with open(os.path.join(location, "utils", "fsdp_utils.py"), encoding = "utf-8") as f:
                source = f.read()
        except OSError:
            continue
        return "def fsdp2_prepare_model" in source and "def _prepare_sd_options" in source
    return None


# Every distributed backend accelerate 1.15 can select (accelerate/utils/dataclasses.py:602-638
# DistributedType, chosen in accelerate/state.py PartialState._prepare_backend :755-816 and
# AcceleratorState.__init__ :960-1023), plus the launcher-only layouts beside it, and what native FSDP2
# does with each. "handle" rows are supported or turned off elsewhere (named in `how`); "refuse" rows
# are detected from
#   env:      a flag `accelerate launch` / an Accelerator plugin exports (true / 1 / yes / on),
#   args:     a TrainingArguments attribute that is set,
#   package:  an installed package accelerate switches on by itself (find_spec, no import),
#   device:   (vendor package or None, torch attribute, checked before CUDA like accelerate): an
#             available device of that kind; needs torch, so the launch step checks it only once torch
#             is imported, Trainer setup and shard_model always,
#   sizes:    accelerate's ParallelismConfig env (accelerate/parallelism_config.py:277-289).
# The launch step (apply_unsloth_fsdp2_launch_env) and Trainer setup both run every row, before any
# model change.
NATIVE_BACKEND_TABLE = (
    {"type": "NO", "action": "handle", "how": "native FSDP2 needs WORLD_SIZE > 1"},
    {
        "type": "MULTI_CPU",
        "action": "handle",
        "how": "gloo CPU mesh (use_cpu / ACCELERATE_USE_CPU)",
    },
    {"type": "MULTI_GPU", "action": "handle", "how": "the supported CUDA / ROCm launch"},
    {
        "type": "FSDP",
        "action": "handle",
        "how": "accelerate's FSDP wins unless UNSLOTH_FSDP2_BACKEND=native, which turns it off "
        "(ACCELERATE_USE_FSDP here, TrainingArguments(fsdp) in native_fsdp2_defers_to_trainer_fsdp)",
    },
    {
        "type": "DEEPSPEED",
        "action": "refuse",
        "label": "DeepSpeed",
        # accelerate/utils/launch.py:566, state.py:974, accelerator.py:368 (a DeepSpeedPlugin sets it)
        "env": ("ACCELERATE_USE_DEEPSPEED",),
        "args": ("deepspeed", "deepspeed_plugin", "hf_deepspeed_config"),
    },
    {
        "type": "MEGATRON_LM",
        "action": "refuse",
        "label": "Megatron-LM",
        # accelerate/utils/launch.py:337, state.py:1017-1023, accelerator.py:405-410 (MegatronLMPlugin)
        "env": ("ACCELERATE_USE_MEGATRON_LM",),
        "args": ("megatron_lm_plugin",),
    },
    {
        "type": "XLA",
        "action": "refuse",
        "label": "torch_xla (TPU / XLA)",
        # accelerate/utils/imports.py:30-41 (USE_TORCH_XLA, default on), state.py:766-767: XLA wins
        # whenever torch_xla imports.
        "package": "torch_xla",
        "unless_env_false": "USE_TORCH_XLA",
    },
    {"type": "MULTI_MLU", "action": "refuse", "label": "MLU", "device": ("torch_mlu", "mlu", True)},
    {
        "type": "MULTI_SDAA",
        "action": "refuse",
        "label": "SDAA",
        "device": ("torch_sdaa", "sdaa", True),
    },
    {
        "type": "MULTI_MUSA",
        "action": "refuse",
        "label": "MUSA",
        "device": ("torch_musa", "musa", True),
    },
    {
        "type": "MULTI_NPU",
        "action": "refuse",
        "label": "Ascend NPU",
        "device": ("torch_npu", "npu", True),
    },
    {
        "type": "MULTI_HPU",
        "action": "refuse",
        "label": "Gaudi HPU",
        "device": ("habana_frameworks.torch", "hpu", True),
    },
    {"type": "MULTI_XPU", "action": "refuse", "label": "Intel XPU", "device": (None, "xpu", False)},
    {
        "type": "MULTI_NEURON",
        "action": "refuse",
        "label": "AWS Neuron",
        "device": ("torch_neuronx", "neuron", False),
    },
    {
        "type": "SAGEMAKER_DP",
        "action": "refuse",
        "label": "SageMaker data parallel",
        # accelerate/state.py:186-191: smddp backend
        "env_pair": ("ACCELERATE_USE_SAGEMAKER", "ACCELERATE_SAGEMAKER_DISTRIBUTED_TYPE"),
    },
    {
        "type": "PARALLELISM_CONFIG",
        "action": "refuse",
        "label": "an accelerate ParallelismConfig (TP / CP / SP / replicate)",
        # accelerate/utils/launch.py:403-420, accelerator.py:455-460; TrainingArguments(
        # parallelism_config) and tp_plan models are refused in fsdp2_native (_refuse_parallelism_config,
        # _refuse_tensor_parallel_model).
        "sizes": "PARALLELISM_CONFIG_",
    },
)


def _set_arg(training_args, name):
    value = getattr(training_args, name, None)
    return value is not None and value is not False and value != "" and value != {}


def _device_available(spec):
    package, attr, before_cuda = spec
    torch = sys.modules.get("torch")
    if torch is None:
        return False  # launch step before torch: Trainer setup / shard_model check it
    try:
        if not before_cuda and torch.cuda.is_available():
            return False  # accelerate prefers CUDA over this one
        if package is not None:
            if importlib.util.find_spec(package.split(".")[0]) is None:
                return False
            importlib.import_module(package)
        backend = getattr(torch, attr, None)
        return bool(backend is not None and backend.is_available())
    except Exception:
        return False


def _parallelism_sizes(prefix):
    if not _env_flag("ACCELERATE_USE_PARALLELISM_CONFIG"):
        return None
    sizes = {}
    for name in ("DP_REPLICATE_SIZE", "DP_SHARD_SIZE", "TP_SIZE", "CP_SIZE", "SP_SIZE"):
        try:
            sizes[name.lower()] = int(os.environ.get(prefix + name, "1") or 1)
        except ValueError:
            sizes[name.lower()] = 1
    world = _env_world_size()
    extra = {k: v for k, v in sizes.items() if k != "dp_shard_size" and v > 1}
    if extra or sizes["dp_shard_size"] not in (1, world):
        return sizes
    return None


def native_backend_conflicts(training_args = None, uses_cpu = None):
    """[(DistributedType, label, reason)] for every NATIVE_BACKEND_TABLE refuse row this launch or
    these TrainingArguments select. `uses_cpu` (the Trainer's use_cpu) skips the device rows."""
    if uses_cpu is None:
        uses_cpu = _env_flag("ACCELERATE_USE_CPU")
    found = []
    for row in NATIVE_BACKEND_TABLE:
        if row["action"] != "refuse":
            continue
        reasons = [name for name in row.get("env", ()) if _env_flag(name)]
        if training_args is not None:
            reasons += [
                f"TrainingArguments.{name}"
                for name in row.get("args", ())
                if _set_arg(training_args, name)
            ]
        package = row.get("package")
        if package and importlib.util.find_spec(package) is not None:
            off = os.environ.get(row.get("unless_env_false", ""), "")
            if str(off).strip().lower() not in _FALSY:
                reasons.append(f"{package} installed")
        if "device" in row and not uses_cpu and _device_available(row["device"]):
            reasons.append(f"a {row['label']} device")
        pair = row.get("env_pair")
        if pair and _env_flag(pair[0]) and _env_value(pair[1]) not in ("", "no"):
            reasons.append(f"{pair[0]} / {pair[1]}={os.environ.get(pair[1])}")
        if "sizes" in row:
            sizes = _parallelism_sizes(row["sizes"])
            if sizes:
                reasons.append(
                    "ACCELERATE_USE_PARALLELISM_CONFIG with "
                    + ", ".join(f"{k}={v}" for k, v in sizes.items())
                )
        if reasons:
            found.append((row["type"], row["label"], reasons))
    return found


def refuse_native_backend_conflicts(training_args = None, uses_cpu = None):
    """Raise, before any model change, when native FSDP2 meets a backend it cannot run under."""
    found = native_backend_conflicts(training_args, uses_cpu)
    if found:
        what = "; ".join(f"{label} ({', '.join(reasons)})" for _, label, reasons in found)
        raise RuntimeError(
            "Unsloth: UNSLOTH_FSDP2=1 shards the model with native FSDP2 (CUDA / ROCm, or the CPU), "
            f"which cannot run under {what}. Drop that backend, or unset UNSLOTH_FSDP2."
        )


def apply_unsloth_fsdp2_launch_env():
    """UNSLOTH_FSDP2=1 with WORLD_SIZE > 1: export what `accelerate launch` exports for an FSDP2
    config (explicit FSDP_* values win). Must run before transformers / accelerate read the env."""
    if not unsloth_fsdp2_requested() or _env_world_size() <= 1:
        return False
    if unsloth_fsdp2_native_requested():
        # Unsloth shards the model itself; accelerate must see a plain multi-GPU launch.
        refuse_native_backend_conflicts()
        if _env_flag("ACCELERATE_USE_FSDP"):
            # `accelerate launch` with an FSDP config and an explicit UNSLOTH_FSDP2_BACKEND=native:
            # accelerate would build its own FSDP plugin from this env and wrap the sharded model
            # again (and with FSDP_CPU_RAM_EFFICIENT_LOADING, from_pretrained leaves the weights
            # unloaded off local rank 0). Same env on every rank, so every rank drops it alike.
            os.environ.pop("ACCELERATE_USE_FSDP", None)
            os.environ.pop("FSDP_VERSION", None)
            if str(os.environ.get("RANK", "0")).strip() in ("", "0"):
                print(
                    "Unsloth: UNSLOTH_FSDP2_BACKEND=native on an `accelerate launch` FSDP config: "
                    "Unsloth shards the model itself, so accelerate's FSDP is turned off "
                    "(ACCELERATE_USE_FSDP and FSDP_VERSION unset)."
                )
        return False
    if _env_flag("ACCELERATE_USE_DEEPSPEED"):
        raise RuntimeError(
            "Unsloth: UNSLOTH_FSDP2=1 was set on a DeepSpeed launch. Unset one of them."
        )
    version = str(os.environ.get("FSDP_VERSION", "")).strip()
    if version not in ("", "2"):
        raise ValueError(
            f"Unsloth: UNSLOTH_FSDP2=1 asks for FSDP2, but FSDP_VERSION={version}. Unset one of them."
        )
    if _accelerate_fsdp2_support() is False:
        # Older accelerate ignores FSDP_VERSION and builds FSDP1 from these variables.
        raise RuntimeError(
            "Unsloth: UNSLOTH_FSDP2=1 needs accelerate >= 1.7, which this accelerate is older than. "
            "Upgrade it with `pip install -U accelerate`."
        )
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    os.environ["FSDP_VERSION"] = "2"
    # accelerate's default is the root only.
    os.environ.setdefault("FSDP_AUTO_WRAP_POLICY", "TRANSFORMER_BASED_WRAP")
    reshard = os.environ.get("UNSLOTH_FSDP2_RESHARD_AFTER_FORWARD", None)
    if reshard is not None and reshard.strip() != "":
        os.environ.setdefault(
            "FSDP_RESHARD_AFTER_FORWARD",
            "false" if reshard.strip().lower() in _FALSY else "true",
        )
    os.environ.setdefault("FSDP_STATE_DICT_TYPE", "FULL_STATE_DICT")
    return True
