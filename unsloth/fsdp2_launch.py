# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FSDP launch detection from the launcher env only (no torch): identical on every rank."""

import ast
import importlib.util
import os
import re
import sys

__all__ = [
    "unsloth_fsdp2_requested",
    "unsloth_fsdp2_native_requested",
    "non_cuda_install",
    "reclaim_native_fsdp2_for_cpu_trainer",
    "fsdp_version_from_env",
    "apply_unsloth_fsdp2_launch_env",
    "NATIVE_BACKEND_TABLE",
    "native_backend_conflicts",
    "refuse_native_backend_conflicts",
    "install_raw_accelerator_fsdp2_fallback",
    "install_native_fsdp2_trainer_init",
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
    if _non_cuda_accelerate_route() is not None:
        # Native FSDP2 is CUDA / ROCm / CPU only: an XPU / NPU / ... install keeps accelerate's FSDP2
        # (pre-native behaviour); an explicit native pick stays native and is refused there.
        return False
    return True


# {key: value before} of the accelerate env exported only because this is a non-CUDA install, so a
# CPU Trainer can be handed back to native FSDP2.
_NON_CUDA_EXPORT = None
_CPU_TRAINER = False


def _non_cuda_accelerate_route():
    """The non_cuda_install() label when the generic UNSLOTH_FSDP2=1 request (no explicit native pick,
    no `accelerate launch` FSDP config) goes to accelerate's FSDP2, else None."""
    if not _env_flag("UNSLOTH_FSDP2") or _env_value("UNSLOTH_FSDP2_BACKEND") != "":
        return None
    if _env_flag("ACCELERATE_USE_FSDP"):
        return None
    return non_cuda_install()


def _trainer_args_use_cpu(training_args):
    return bool(
        getattr(training_args, "use_cpu", False) or getattr(training_args, "no_cuda", False)
    )


def reclaim_native_fsdp2_for_cpu_trainer(training_args):
    """TrainingArguments(use_cpu = True) (or `no_cuda`) on a non-CUDA install whose launch step routed
    UNSLOTH_FSDP2=1 to accelerate's FSDP2: accelerate runs that Trainer as MULTI_CPU, which never builds
    its FSDP plugin (accelerate/state.py AcceleratorState.__init__), so the sharding would be lost. Undo
    the exported env before the Accelerator exists and let native FSDP2 shard on the gloo CPU mesh, as
    it does for the same Trainer on a CUDA install. True when native FSDP2 now applies."""
    global _NON_CUDA_EXPORT, _CPU_TRAINER
    if _NON_CUDA_EXPORT is None or not _trainer_args_use_cpu(training_args):
        return False
    for key, value in _NON_CUDA_EXPORT.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    _NON_CUDA_EXPORT = None
    _CPU_TRAINER = True
    if str(os.environ.get("RANK", "0")).strip() in ("", "0"):
        print(
            "Unsloth: UNSLOTH_FSDP2=1 with TrainingArguments(use_cpu = True): native FSDP2 shards the "
            "model on the CPU (accelerate's FSDP2 does not run on a CPU launch)."
        )
    return unsloth_fsdp2_native_requested()


def _torch_build():
    """{"cuda", "hip", "xpu"}: the torch build's `torch.version` strings (None when absent), read from
    torch/version.py without importing torch (the launch step runs before torch is imported), or None
    when torch is not found. Same answer at the launch step and at Trainer setup."""
    torch = sys.modules.get("torch")
    version = getattr(torch, "version", None) if torch is not None else None
    if version is not None:
        build = {name: getattr(version, name, None) for name in ("cuda", "hip", "xpu")}
        label = getattr(version, "__version__", None) or getattr(torch, "__version__", None)
        return _xpu_from_label(build, label)
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return None
    for location in getattr(spec, "submodule_search_locations", None) or ():
        try:
            with open(os.path.join(location, "version.py"), encoding = "utf-8") as f:
                source = f.read()
        except OSError:
            continue
        build = {}
        for name in ("cuda", "hip", "xpu", "__version__"):
            match = re.search(rf"^{name}\s*(?::[^=\n]*)?=\s*(.+?)\s*$", source, re.M)
            try:
                build[name] = ast.literal_eval(match.group(1)) if match else None
            except (ValueError, SyntaxError):
                build[name] = None
        return _xpu_from_label(build, build.pop("__version__"))
    return None


def _xpu_from_label(build, label):
    """A `+xpu` wheel can ship torch.version.xpu = None (and torch < 2.7 has no xpu attribute at all):
    the version label is then the only build marker, read the same way with and without torch."""
    if not build.get("xpu") and "+xpu" in str(label or "").lower():
        build["xpu"] = str(label)
    return build


def non_cuda_install():
    """The label of the non-CUDA accelerator accelerate would pick on this install (a
    NATIVE_BACKEND_TABLE device row), from the torch build and the installed vendor packages only, so
    every rank and every call agrees; None on a CUDA / ROCm torch build, for a CPU launch
    (ACCELERATE_USE_CPU), or when nothing matches."""
    if _CPU_TRAINER or _env_flag("ACCELERATE_USE_CPU"):
        return None
    build = _torch_build()
    if not build or build.get("cuda") or build.get("hip"):
        return None
    for row in NATIVE_BACKEND_TABLE:
        if row["action"] != "refuse" or "device" not in row:
            continue
        package, attr, _ = row["device"]
        if package is None:
            if build.get(attr):
                return row["label"]
            continue
        try:
            if importlib.util.find_spec(package.split(".")[0]) is not None:
                return row["label"]
        except (ImportError, ValueError):
            continue
    return None


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


def _import_vendor(package):
    """The vendor extension that registers its device with torch, as accelerate imports it
    (accelerate/utils/imports.py is_*_available). Written-out names only."""
    if package == "torch_mlu":
        import torch_mlu  # noqa: F401
    elif package == "torch_sdaa":
        import torch_sdaa  # noqa: F401
    elif package == "torch_musa":
        import torch_musa  # noqa: F401
    elif package == "torch_npu":
        import torch_npu  # noqa: F401
    elif package == "habana_frameworks.torch":
        import habana_frameworks.torch  # noqa: F401
    elif package == "torch_neuronx":
        import torch_neuronx  # noqa: F401
    else:
        raise ValueError(package)


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
            _import_vendor(package)
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
        devices = {row["type"] for row in NATIVE_BACKEND_TABLE if "device" in row}
        hint = ""
        if any(kind in devices for kind, _, _ in found):
            hint = (
                " For accelerate's FSDP2 on this device, set UNSLOTH_FSDP2=1 without a native pick "
                "(or UNSLOTH_FSDP2_BACKEND=accelerate)."
            )
        raise RuntimeError(
            "Unsloth: UNSLOTH_FSDP2=1 shards the model with native FSDP2 (CUDA / ROCm, or the CPU), "
            f"which cannot run under {what}. Drop that backend, or unset UNSLOTH_FSDP2.{hint}"
        )


_LAUNCH_EXPORTS = (
    "ACCELERATE_USE_FSDP",
    "FSDP_VERSION",
    "FSDP_AUTO_WRAP_POLICY",
    "FSDP_RESHARD_AFTER_FORWARD",
    "FSDP_STATE_DICT_TYPE",
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
    return _export_accelerate_fsdp2_env()


def _export_accelerate_fsdp2_env():
    """What `accelerate launch` exports for an FSDP2 config (explicit FSDP_* values win)."""
    global _NON_CUDA_EXPORT
    if _env_flag("ACCELERATE_USE_DEEPSPEED"):
        raise RuntimeError(
            "Unsloth: UNSLOTH_FSDP2=1 was set on a DeepSpeed launch. Unset one of them."
        )
    device = _non_cuda_accelerate_route()
    if device is not None and str(os.environ.get("RANK", "0")).strip() in ("", "0"):
        print(
            f"Unsloth: UNSLOTH_FSDP2=1 with {device} (not a CUDA / ROCm torch build): native FSDP2 "
            "runs on CUDA / ROCm or the CPU, so accelerate's FSDP2 shards the model."
        )
    before = {key: os.environ.get(key) for key in _LAUNCH_EXPORTS}
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
    if device is not None:
        _NON_CUDA_EXPORT = {
            key: value for key, value in before.items() if os.environ.get(key) != value
        }
    return True


_RAW_ACCELERATOR_PATCHED = "_unsloth_fsdp2_raw_accelerator_fallback"


def _built_by_trainer(frame, max_depth = 64):
    """An Accelerator built inside Trainer.create_accelerator_and_postprocess (native FSDP2 shards the
    Trainer's model itself), told apart from one a custom loop builds."""
    while frame is not None and max_depth > 0:
        if frame.f_code.co_name == "create_accelerator_and_postprocess":
            owner = frame.f_locals.get("self", None)
            if owner is not None and hasattr(owner, "args"):
                return True
        frame = frame.f_back
        max_depth -= 1
    return False


def _raw_accelerator_wants_fsdp2(kwargs):
    """UNSLOTH_FSDP2=1 picked native FSDP2 by default (no UNSLOTH_FSDP2_BACKEND), and this Accelerator
    is the first one, built by a custom loop that chose no distributed plugin itself."""
    if _env_value("UNSLOTH_FSDP2_BACKEND") != "" or not unsloth_fsdp2_native_requested():
        return False
    if any(
        kwargs.get(name) is not None
        for name in (
            "fsdp_plugin",
            "deepspeed_plugin",
            "deepspeed_plugins",
            "megatron_lm_plugin",
            "torch_tp_plugin",
            "parallelism_config",
        )
    ):
        return False
    if kwargs.get("cpu") or _env_flag("ACCELERATE_USE_CPU"):
        return False  # accelerate has no FSDP on the CPU
    try:
        from accelerate.state import AcceleratorState
        if getattr(AcceleratorState, "_shared_state", {}):
            return False  # an earlier Accelerator already fixed the launch for this process
    except Exception:
        return False
    native = sys.modules.get("unsloth.models.fsdp2_native", None)
    if native is not None and getattr(native, "_SHARDED_ANY", False):
        return False  # shard_model already sharded a model natively
    return True


def install_raw_accelerator_fsdp2_fallback():
    """UNSLOTH_FSDP2=1 shards natively inside the Trainer only. A custom loop that builds its own
    Accelerator and calls accelerator.prepare(model, optimizer) on an unsharded model would get a
    plain DDP wrap (no parameter or optimizer sharding). The first such Accelerator instead gets the
    accelerate FSDP2 launch env that UNSLOTH_FSDP2=1 exported before native FSDP2 existed (the same as
    UNSLOTH_FSDP2_BACKEND=accelerate). An explicit UNSLOTH_FSDP2_BACKEND is left alone. Idempotent."""
    if not unsloth_fsdp2_requested() or _env_world_size() <= 1:
        return False
    try:
        from accelerate import Accelerator
    except Exception:
        return False
    if getattr(Accelerator, _RAW_ACCELERATOR_PATCHED, False):
        return True
    import functools
    import inspect

    original_init = Accelerator.__init__

    try:
        signature = inspect.signature(original_init)
    except (TypeError, ValueError):
        signature = None

    def _named(self, args, kwargs):
        # Options passed positionally count as explicit too.
        if signature is None or not args:
            return kwargs
        try:
            return dict(signature.bind_partial(self, *args, **kwargs).arguments)
        except TypeError:
            return kwargs

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        if _raw_accelerator_wants_fsdp2(_named(self, args, kwargs)) and not _built_by_trainer(
            sys._getframe(1)
        ):
            _export_accelerate_fsdp2_env()
            if str(os.environ.get("RANK", "0")).strip() in ("", "0"):
                print(
                    "Unsloth: UNSLOTH_FSDP2=1 with an Accelerator built outside a Trainer: accelerate's "
                    "FSDP2 shards this loop (as UNSLOTH_FSDP2_BACKEND=accelerate). Set "
                    "UNSLOTH_FSDP2_BACKEND explicitly to choose the backend."
                )
        original_init(self, *args, **kwargs)

    Accelerator.__init__ = __init__
    setattr(Accelerator, _RAW_ACCELERATOR_PATCHED, True)
    return True


def install_native_fsdp2_trainer_init():
    """UNSLOTH_FSDP2=1 under WORLD_SIZE > 1 shards natively inside Unsloth's Trainer.__init__ wrapper
    (installed by `_utils.patch_gradient_accumulation_fix`). FastLanguageModel / FastModel loads and
    DecisionTrainer install it; a model built without them (a FastSentenceTransformer encoder) left the
    Trainer unwrapped, so it trained under DDP with nothing sharded and no message. Installs the wrapper
    once the flag selects native FSDP2, or while a non-CUDA accelerate route is pending (a use_cpu
    Trainer reclaims native FSDP2 inside the wrapper). Returns whether it applies."""
    if not unsloth_fsdp2_native_requested():
        if _NON_CUDA_EXPORT is None:
            return False
        from transformers import Trainer

        from .models._utils import patch_gradient_accumulation_fix

        patch_gradient_accumulation_fix(Trainer)
        return False
    from transformers import Trainer

    from .models._utils import patch_gradient_accumulation_fix

    patch_gradient_accumulation_fix(Trainer)
    return True
