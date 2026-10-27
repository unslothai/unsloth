# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`optim = "adamw_flash"`: FlashAdamW from databricks/flashoptim (arXiv 2602.23349).

bf16 / fp16 parameters keep a 16 bit error correction term in the optimizer state, so
the effective master weight is fp32. The momentum is stored in the parameter dtype and the
bf16 variance in fp32 (a bf16 variance cannot decay at beta2 = 0.999): about 12 bytes per
parameter (with bf16 grads) instead of 16 for bf16 weights plus fp32 master plus fp32
moments. This default tracked fp32 master AdamW on a real Qwen2.5-0.5B full finetune.
optim_args = "quantize=True" stores the moments in 8 bits (about 8 bytes per parameter;
with "master_weight_bits=24", flashoptim's own default, about 7), but its loss curve moved
measurably away from AdamW there.
fp32 parameters (LoRA adapters) have no correction term; for them only quantize=True
saves memory. Trainable fp16 parameters need quantize=True: unquantized fp16 moments
underflow, so the optimizer refuses them.

transformers validates `optim` against OptimizerNames, so the config stores
`adamw_torch` and marks itself with `_unsloth_flash_adamw` plus a token in `optim_args`
(a dataclass field, so `dataclasses.replace` keeps it); the patched
`Trainer.get_optimizer_cls_and_kwargs` reads either mark. Because the placeholder is a real
optim name, `dataclasses.replace(args, optim = "adamw_torch")` still selects FlashAdamW;
switch back with `optim_args = None` (or `unmark_flash_adamw(args)`).
"""

__all__ = [
    "FLASH_ADAMW_NAME",
    "is_flash_adamw",
    "mark_flash_adamw",
    "unmark_flash_adamw",
    "get_flash_adamw_cls_and_kwargs",
]

import os
import sys

import torch

FLASH_ADAMW_NAME = "adamw_flash"
_FLASH_ADAMW_PLACEHOLDER = "adamw_torch"
_FLASH_ADAMW_MARKER = "unsloth_adamw_flash"
_FLASH_ADAMW_CLASS = None
_VARIANCE_UPDATE_KERNEL = None
_FLASH_ADAMW_MIN_VERSION = "0.1.4"
_TRUE_VALUES = ("1", "true", "yes")
_FALSE_VALUES = ("0", "false", "no")
_announced = False


def is_flash_adamw(optim):
    return str(getattr(optim, "value", optim)).strip().lower() == FLASH_ADAMW_NAME


def _parse_optim_args(optim_args):
    parsed = {}
    if not optim_args:
        return parsed
    for item in str(optim_args).replace(" ", "").split(","):
        if not item:
            continue
        key, _, value = item.partition("=")
        if key == _FLASH_ADAMW_MARKER:
            continue
        if key == "master_weight_bits":
            parsed[key] = None if value.lower() in ("none", "0", "") else int(value)
        elif key in ("quantize", "compress_state_dict", "check_numerics", "decouple_lr"):
            if value.lower() not in _TRUE_VALUES + _FALSE_VALUES:
                raise ValueError(
                    f"Unsloth: optim = 'adamw_flash' optim_args {key}={value!r} is not a boolean. "
                    "Use true / false, 1 / 0 or yes / no."
                )
            parsed[key] = value.lower() in _TRUE_VALUES
        else:
            raise ValueError(
                f"Unsloth: optim = 'adamw_flash' does not understand optim_args key {key!r}. "
                "Supported: master_weight_bits=24|32|none, quantize, compress_state_dict, check_numerics, decouple_lr."
            )
    return parsed


def _get_flash_adamw_class():
    global _FLASH_ADAMW_CLASS
    if _FLASH_ADAMW_CLASS is not None:
        return _FLASH_ADAMW_CLASS
    try:
        import flashoptim
        from flashoptim import FlashAdamW
    except ImportError as error:
        raise ImportError(
            "Unsloth: optim = 'adamw_flash' needs the flashoptim package. "
            f"Install it via `pip install 'flashoptim>={_FLASH_ADAMW_MIN_VERSION}'`."
        ) from error
    from packaging.version import Version

    version = getattr(flashoptim, "__version__", None)
    if version is None:
        import importlib.metadata
        version = importlib.metadata.version("flashoptim")
    # 0.1.4 made compress_state_dict default to True, which the resume guard below relies on.
    if Version(version) < Version(_FLASH_ADAMW_MIN_VERSION):
        raise ImportError(
            f"Unsloth: optim = 'adamw_flash' needs flashoptim>={_FLASH_ADAMW_MIN_VERSION}, found {version}. "
            f"Upgrade via `pip install -U 'flashoptim>={_FLASH_ADAMW_MIN_VERSION}'`."
        )

    class UnslothFlashAdamW(FlashAdamW):
        def __init__(
            self,
            params,
            *args,
            master_weight_bits = 32,
            **kwargs,
        ):
            params = [
                {**group, "params": _materialize_params(group["params"])}
                if isinstance(group, dict) and "params" in group
                else group
                for group in params
            ]
            # flashoptim seeds every group's initial_lr with the constructor lr, and the LR
            # scheduler then uses it as the group's base lr (setdefault keeps it), which would
            # drop embedding_learning_rate and loraplus_lr_ratio, and skew decouple_lr decay.
            params = [
                {**group, "initial_lr": group.get("initial_lr", group["lr"])}
                if isinstance(group, dict) and "lr" in group
                else group
                for group in params
            ]
            # flashoptim raises when every parameter is fp32 and master_weight_bits is set,
            # which is the normal LoRA case: there only the 8 bit moments apply.
            tensors = [
                p
                for group in params
                for p in (group["params"] if isinstance(group, dict) else [group])
            ]
            if tensors and all(p.dtype == torch.float32 for p in tensors):
                master_weight_bits = None
            # flashoptim's load_state_dict re-quantizes uncompressed moments with one
            # optimizer wide bf16 check, which corrupts the other dtype's scales on resume.
            if (
                kwargs.get("quantize", True)
                and not kwargs.get("compress_state_dict", True)
                and len({p.dtype == torch.bfloat16 for p in tensors}) > 1
            ):
                raise ValueError(
                    "Unsloth: optim = 'adamw_flash' with quantize=True and compress_state_dict=False "
                    "cannot resume a model mixing bf16 and other trainable dtypes. "
                    "Drop compress_state_dict=False or quantize=True."
                )
            super().__init__(params, *args, master_weight_bits = master_weight_bits, **kwargs)

        # flashoptim stores unquantized moments in the param dtype, so an fp16 variance
        # (grad**2 * (1 - beta2)) underflows to 0 for small grads and the update becomes
        # momentum / eps. quantize=True keeps the variance as sqrt with per-group scales.
        # Checked per group after flashoptim fills its defaults (so positional and per-group
        # quantize both count), for groups added at construction and by add_param_group later.
        def _group_quantizes(self, group):
            # flashoptim resolves a group without the key (a vanilla AdamW state dict) to the
            # optimizer-wide setting.
            return group.get("quantize", getattr(self, "_quantize", True))

        def add_param_group(self, param_group):
            super().add_param_group(param_group)
            group = self.param_groups[-1]
            if not self._group_quantizes(group) and any(
                p.dtype == torch.float16 and p.requires_grad for p in group["params"]
            ):
                # Not kept: a caller that catches this can add the params again with quantize=True.
                self.param_groups.pop()
                self.__dict__.pop("quantized_state_spec", None)
                _refuse_unquantized_fp16()

        # A bf16 variance cannot move at beta2 = 0.999 (a step is under half an ulp), so it never
        # decays and stalls near grad**2 / 3: the unquantized bf16 variance is kept in fp32.
        def _ensure_state_initialized(self, p, hparams):
            super()._ensure_state_initialized(p, hparams)
            _widen_variance(self.state[p].get("exp_avg_sq"))

        def _load_state_for_param(self, param, opt_state, hparams):
            state = super()._load_state_for_param(param, opt_state, hparams)
            _widen_variance(state.get("exp_avg_sq"))
            return state

        def load_state_dict(self, state_dict):
            # torch.optim casts floating state to the param dtype: restore the fp32 variance.
            saved_ids = [i for g in state_dict.get("param_groups", []) for i in g["params"]]
            params = dict(zip(saved_ids, (p for g in self.param_groups for p in g["params"])))
            fp32_vars = {
                params[i]: v
                for i, st in state_dict.get("state", {}).items()
                if i in params
                and torch.is_tensor(v := st.get("exp_avg_sq"))
                and v.dtype == torch.float32
            }
            super().load_state_dict(state_dict)
            for p, v in fp32_vars.items():
                var = getattr(self.state.get(p, {}).get("exp_avg_sq"), "_data", None)
                v = self._get_local_tensor(v)
                if var is not None and var.dtype == torch.float32 and var.shape == v.shape:
                    var.copy_(v)

        def _do_step(self, param, grad, errors, param_state, hparams):
            var = getattr(param_state.get("exp_avg_sq"), "_data", None)
            if (
                not self._fused
                or var is None
                or var.dtype != torch.float32
                or param.dtype == torch.float32
            ):
                return super()._do_step(param, grad, errors, param_state, hparams)
            # The fused kernel stores the variance rounded to the param dtype: keep an fp32 copy.
            new_var = _variance_update(var, grad, hparams["betas"][1])
            out = super()._do_step(param, grad, errors, param_state, hparams)
            param_state["exp_avg_sq"]._data = new_var
            return out

        # Every update goes through step_param: step() loops over it and gradient release calls
        # it per parameter. Checked there against the group's current quantize (load_state_dict
        # can restore an older group's quantize=False), so a frozen fp16 param unfrozen later is
        # refused before it gets moments. A param frozen after backward keeps its .grad, but
        # flashoptim skips it (requires_grad False), so it is not refused. Only fp16 params pay
        # for the group lookup.
        def step_param(
            self,
            p,
            group = None,
        ):
            if p.dtype == torch.float16 and p.requires_grad and p.grad is not None:
                if group is None:
                    group = next(
                        (g for g in self.param_groups if any(q is p for q in g["params"])), None
                    )
                if group is not None and not self._group_quantizes(group):
                    _refuse_unquantized_fp16()
            return super().step_param(p, group)

    _FLASH_ADAMW_CLASS = UnslothFlashAdamW
    return _FLASH_ADAMW_CLASS


def _variance_update(var, grad, beta2):
    """fp32 `var * beta2 + grad**2 * (1 - beta2)`, as flashoptim's fused kernel computes it."""
    global _VARIANCE_UPDATE_KERNEL
    import triton

    if _VARIANCE_UPDATE_KERNEL is None:
        import triton.language as tl

        @triton.jit
        def kernel(var_ptr, grad_ptr, out_ptr, n, beta2, BLOCK: tl.constexpr):
            offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < n
            v = tl.load(var_ptr + offsets, mask = mask)
            g = tl.load(grad_ptr + offsets, mask = mask).to(tl.float32)
            tl.store(out_ptr + offsets, v * beta2 + g * g * (1.0 - beta2), mask = mask)

        _VARIANCE_UPDATE_KERNEL = kernel
    out = torch.empty_like(var)
    n = var.numel()
    _VARIANCE_UPDATE_KERNEL[(triton.cdiv(n, 1024),)](var, grad, out, n, beta2, BLOCK = 1024)
    return out


def _widen_variance(state):
    # flashoptim's private _MaybeQuantizedTensor fields; absent ones leave its storage as is.
    data = getattr(state, "_data", None)
    if data is not None and data.dtype == torch.bfloat16 and hasattr(state, "_storage_dtype"):
        state._storage_dtype = torch.float32
        state._data = data.float()


def _refuse_unquantized_fp16():
    raise ValueError(
        "Unsloth: optim = 'adamw_flash' with quantize=False stores float16 Adam moments in "
        "float16, where the variance underflows (wrong updates). Found trainable float16 "
        "parameters: cast them to bfloat16 (its variance is kept in float32) or float32, or "
        "set optim_args = 'quantize=True'."
    )


def _materialize_params(params):
    # A one-shot iterable (model.parameters()) would be exhausted by the dtype scan below.
    return [params] if isinstance(params, torch.Tensor) else list(params)


def _has_marker_token(args):
    optim_args = str(getattr(args, "optim_args", None) or "").replace(" ", "")
    return any(item.partition("=")[0] == _FLASH_ADAMW_MARKER for item in optim_args.split(","))


def _has_placeholder_optim(args):
    optim = getattr(args, "optim", None)
    return str(getattr(optim, "value", optim)).strip().lower() == _FLASH_ADAMW_PLACEHOLDER


def _is_marked_in_optim_args(args):
    # The mark only counts while optim is still the placeholder, so a later optim change wins.
    return _has_placeholder_optim(args) and _has_marker_token(args)


def _is_marked(args):
    # The optim_args token alone decides, so an in place `args.optim_args = None` switches back.
    return _is_marked_in_optim_args(args)


def _strip_marker(optim_args):
    items = str(optim_args or "").split(",")
    kept = [
        i
        for i in items
        if i.strip() and i.replace(" ", "").partition("=")[0] != _FLASH_ADAMW_MARKER
    ]
    return ",".join(kept) or None


def _is_truthy(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def _fsdp_cpu_offload(args):
    fsdp = getattr(args, "fsdp", None)
    items = (
        fsdp.split() if isinstance(fsdp, str) else (fsdp if isinstance(fsdp, (list, tuple)) else [])
    )
    if any(str(getattr(item, "value", item)).lower() == "offload" for item in items):
        return True
    config = getattr(args, "fsdp_config", None)
    config = config if isinstance(config, dict) else {}
    if any(
        _is_truthy(config.get(key, False))
        for key in ("cpu_offload", "offload_params", "fsdp_offload_params")
    ):
        return True
    return _is_truthy(os.environ.get("FSDP_OFFLOAD_PARAMS", ""))


def _fsdp_version(args):
    if getattr(args, "fsdp", None):
        config = getattr(args, "fsdp_config", None)
        config = config if isinstance(config, dict) else {}
        version = config.get("version", config.get("fsdp_version", None))
        if version is None:
            import transformers
            from packaging.version import Version
            version = 2 if Version(transformers.__version__) >= Version("5.10.0") else 1
        return int(version)
    if os.environ.get("ACCELERATE_USE_FSDP", "").strip().lower() in ("1", "true", "yes"):
        return int(os.environ.get("FSDP_VERSION", "").strip() or 1)
    return None


def _deepspeed_offloads_optimizer(args):
    if _is_truthy(os.environ.get("ACCELERATE_USE_DEEPSPEED", "")) and os.environ.get(
        "ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE", ""
    ).strip().lower() in ("cpu", "nvme"):
        return True
    config = getattr(getattr(args, "hf_deepspeed_config", None), "config", None)
    if config is None:
        config = getattr(args, "deepspeed", None)
        if isinstance(config, str) and config:
            import json
            with open(config, "r", encoding = "utf-8") as f:
                config = json.load(f)
    if not isinstance(config, dict):
        return False
    offload = (config.get("zero_optimization", None) or {}).get("offload_optimizer", None) or {}
    return str(offload.get("device", "none")).lower() in ("cpu", "nvme")


def get_flash_adamw_cls_and_kwargs(args):
    global _announced
    # flashoptim supports only Linux with NVIDIA CUDA; ROCm also reports torch.cuda.is_available().
    if (
        not torch.cuda.is_available()
        or getattr(torch.version, "hip", None)
        or not sys.platform.startswith("linux")
    ):
        raise NotImplementedError(
            "Unsloth: optim = 'adamw_flash' runs flashoptim's Triton kernels, which support only "
            "Linux with NVIDIA CUDA GPUs. Use optim = 'adamw_8bit' or 'adamw_torch' instead."
        )
    if _deepspeed_offloads_optimizer(args):
        raise ValueError(
            "Unsloth: optim = 'adamw_flash' cannot be used with DeepSpeed ZeRO offload_optimizer, "
            "since flashoptim has no CPU implementation. Turn off offload_optimizer or use another optim."
        )
    if _fsdp_version(args) == 1:
        raise ValueError(
            "Unsloth: optim = 'adamw_flash' does not support FSDP1 (flashoptim supports DDP and FSDP2). "
            "Set fsdp_config = {'version': 2} or use another optim."
        )
    if _fsdp_version(args) is not None and _fsdp_cpu_offload(args):
        raise ValueError(
            "Unsloth: optim = 'adamw_flash' cannot step FSDP2 CPU offloaded parameters, since "
            "flashoptim's Triton kernels need CUDA tensors. Turn off cpu_offload or use another optim."
        )
    optimizer_cls = _get_flash_adamw_class()
    if not _announced:
        _announced = True
        print(
            "Unsloth: using FlashAdamW (optim = 'adamw_flash'). To train this config with plain "
            "AdamW, set optim = 'adamw_torch' and optim_args = None."
        )
    optimizer_kwargs = {
        "lr": args.learning_rate,
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.adam_epsilon,
        "master_weight_bits": 32,
        "quantize": False,
    }
    optimizer_kwargs.update(_parse_optim_args(getattr(args, "optim_args", None)))
    return optimizer_cls, optimizer_kwargs


def _patch_trainer_get_optimizer_cls_and_kwargs():
    from transformers import Trainer

    original = Trainer.__dict__["get_optimizer_cls_and_kwargs"]
    original = getattr(original, "__func__", original)
    if getattr(original, "_unsloth_flash_adamw", False):
        return

    def get_optimizer_cls_and_kwargs(args, *other, **kwargs):
        if _is_marked(args):
            return get_flash_adamw_cls_and_kwargs(args)
        if _has_marker_token(args):
            import copy
            args = copy.copy(args)
            args.optim_args = _strip_marker(args.optim_args)
        return original(args, *other, **kwargs)

    get_optimizer_cls_and_kwargs._unsloth_flash_adamw = True
    get_optimizer_cls_and_kwargs.__wrapped__ = original
    get_optimizer_cls_and_kwargs.__doc__ = original.__doc__
    Trainer.get_optimizer_cls_and_kwargs = staticmethod(get_optimizer_cls_and_kwargs)


def is_flash_adamw_marked(training_args):
    """A config rebuilt from to_dict / to_json_file carries only the optim_args marker."""
    return _is_marked_in_optim_args(training_args)


def unmark_flash_adamw(training_args):
    """A config that selected adamw_flash and was then rebuilt with another optim."""
    if getattr(training_args, "_unsloth_flash_adamw", False):
        training_args._unsloth_flash_adamw = False
    if _has_marker_token(training_args):
        training_args.optim_args = _strip_marker(training_args.optim_args)
    return training_args


def _patch_hf_argument_parser():
    # HfArgumentParser turns `optim: OptimizerNames | str` into an OptimizerNames choice list,
    # so `--optim adamw_flash` is rejected before an Unsloth config can translate it.
    try:
        from transformers import HfArgumentParser
    except ImportError:
        return
    original = getattr(HfArgumentParser, "_add_dataclass_arguments", None)
    if original is None or getattr(original, "_unsloth_flash_adamw", False):
        return

    def _add_dataclass_arguments(self, dtype, *args, **kwargs):
        result = original(self, dtype, *args, **kwargs)
        if not getattr(dtype, "_unsloth_accepts_adamw_flash", False):
            return result
        for action in self._actions:
            if action.dest != "optim" or not action.choices or FLASH_ADAMW_NAME in action.choices:
                continue
            action.choices = list(action.choices) + [FLASH_ADAMW_NAME]
            convert = action.type
            if callable(convert):
                action.type = lambda value, convert = convert: (
                    FLASH_ADAMW_NAME if is_flash_adamw(value) else convert(value)
                )
        return result

    _add_dataclass_arguments._unsloth_flash_adamw = True
    HfArgumentParser._add_dataclass_arguments = _add_dataclass_arguments


_patch_hf_argument_parser()


def mark_flash_adamw(training_args):
    """Called after TrainingArguments.__init__ ran with the placeholder optim."""
    training_args._unsloth_flash_adamw = True
    if not _is_marked_in_optim_args(training_args):
        optim_args = getattr(training_args, "optim_args", None)
        # key=value form: transformers' own optim_args parser splits every item on "=".
        marker = f"{_FLASH_ADAMW_MARKER}=1"
        training_args.optim_args = f"{marker},{optim_args}" if optim_args else marker
    _patch_trainer_get_optimizer_cls_and_kwargs()
    return training_args
