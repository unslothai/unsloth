# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`optim = "adamw_flash"`: FlashAdamW from databricks/flashoptim (arXiv 2602.23349).

bf16 / fp16 parameters keep a 16 bit error correction term in the optimizer state, so
the effective master weight is fp32, and the Adam moments are stored in the parameter
dtype: about 10 bytes per parameter (with bf16 grads) instead of 16 for bf16 weights
plus fp32 master plus fp32 moments. This default tracked fp32 master AdamW on a real
Qwen2.5-0.5B full finetune. optim_args = "quantize=True" stores the moments in 8 bits
(about 8 bytes per parameter; with "master_weight_bits=24", flashoptim's own default,
about 7), but its loss curve moved measurably away from AdamW there.
fp32 parameters (LoRA adapters) have no correction term; for them only quantize=True
saves memory.

transformers validates `optim` against OptimizerNames, so the config stores
`adamw_torch` and marks itself with `_unsloth_flash_adamw` plus a token in `optim_args`
(a dataclass field, so `dataclasses.replace` keeps it); the patched
`Trainer.get_optimizer_cls_and_kwargs` reads either mark.
"""

__all__ = [
    "FLASH_ADAMW_NAME",
    "is_flash_adamw",
    "mark_flash_adamw",
    "get_flash_adamw_cls_and_kwargs",
]

import torch

FLASH_ADAMW_NAME = "adamw_flash"
_FLASH_ADAMW_PLACEHOLDER = "adamw_torch"
_FLASH_ADAMW_MARKER = "unsloth_adamw_flash"
_FLASH_ADAMW_CLASS = None


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
            parsed[key] = value.lower() in ("1", "true", "yes")
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
        from flashoptim import FlashAdamW
    except ImportError as error:
        raise ImportError(
            "Unsloth: optim = 'adamw_flash' needs the flashoptim package. "
            "Install it via `pip install flashoptim`."
        ) from error

    class UnslothFlashAdamW(FlashAdamW):
        def __init__(self, params, *args, master_weight_bits = 32, **kwargs):
            params = list(params)
            # flashoptim raises when every parameter is fp32 and master_weight_bits is set,
            # which is the normal LoRA case: there only the 8 bit moments apply.
            tensors = [
                p
                for group in params
                for p in (group["params"] if isinstance(group, dict) else [group])
            ]
            if tensors and all(p.dtype == torch.float32 for p in tensors):
                master_weight_bits = None
            super().__init__(params, *args, master_weight_bits = master_weight_bits, **kwargs)

    _FLASH_ADAMW_CLASS = UnslothFlashAdamW
    return _FLASH_ADAMW_CLASS


def _is_marked_in_optim_args(args):
    optim_args = str(getattr(args, "optim_args", None) or "").replace(" ", "")
    return any(item.partition("=")[0] == _FLASH_ADAMW_MARKER for item in optim_args.split(","))


def _is_marked(args):
    return getattr(args, "_unsloth_flash_adamw", False) or _is_marked_in_optim_args(args)


def get_flash_adamw_cls_and_kwargs(args):
    optimizer_cls = _get_flash_adamw_class()
    if not torch.cuda.is_available():
        raise NotImplementedError(
            "Unsloth: optim = 'adamw_flash' runs fused Triton kernels and needs an NVIDIA GPU. "
            "Use optim = 'adamw_8bit' or 'adamw_torch' instead."
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
        return original(args, *other, **kwargs)

    get_optimizer_cls_and_kwargs._unsloth_flash_adamw = True
    get_optimizer_cls_and_kwargs.__doc__ = original.__doc__
    Trainer.get_optimizer_cls_and_kwargs = staticmethod(get_optimizer_cls_and_kwargs)


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
