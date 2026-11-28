# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""bitsandbytes optimizers for Unsloth's bitsandbytes stand-in. AdamW8bit and friends build the
plain torch optimizer with the same update rule (8-bit and paged state are bitsandbytes native
features), after a one time notice. The result is a real torch.optim instance, so code that
checks for torch optimizers (checkpointing, FSDP2, stochastic rounding) treats it as one."""

import torch

__all__ = [
    "GlobalOptimManager",
    "Optimizer1State",
    "Optimizer2State",
    "Optimizer8bit",
    "torch_optimizer_fallback",
]

_NOTICED = set()
# bitsandbytes-only constructor arguments, dropped for the torch optimizer.
_BNB_ONLY = (
    "optim_bits",
    "args",
    "min_8bit_size",
    "percentile_clipping",
    "block_wise",
    "is_paged",
    "max_unorm",
    "skip_zeros",
)
# Keys only bitsandbytes optimizer state has: such a checkpoint cannot resume a torch optimizer.
_BNB_STATE_KEYS = ("state1", "state2", "qmap1", "qmap2", "absmax1", "absmax2")


class GlobalOptimManager:
    """Records per-parameter overrides (Trainer pins embeddings to 32-bit); full precision already."""

    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.pid2config = {}
        self.index2config = {}
        self.optimizer = None
        self.uses_config_override = False
        self.module_weight_config_triple = []

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def register_parameters(self, params):
        pass

    def override_config(
        self,
        parameters,
        key = None,
        value = None,
        key_value_dict = None,
    ):
        pass

    def register_module_override(self, module, param_name, config):
        # Full precision state already: nothing to override, and nothing kept alive.
        pass


class Optimizer8bit(torch.optim.Optimizer):
    """isinstance target only: no stand-in optimizer keeps 8-bit state."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Unsloth: 8-bit optimizer state needs the bitsandbytes package; the stand-in optimizers "
            "(bitsandbytes.optim.AdamW and friends) run the torch optimizer instead."
        )


class Optimizer1State(Optimizer8bit):
    pass


class Optimizer2State(Optimizer8bit):
    pass


def _materialize(params):
    """List of parameters or of param-group dicts, every generator consumed exactly once."""
    out = []
    for item in params:
        if isinstance(item, dict):
            group = dict(item)
            group_params = group.get("params", ())
            if isinstance(group_params, torch.Tensor):
                group_params = [group_params]
            group["params"] = list(group_params)
            out.append(group)
        else:
            out.append(item)
    return out


def _parameters(params):
    for item in params:
        if isinstance(item, dict):
            yield from item["params"]
        else:
            yield item


def _refuse_bnb_state(optimizer, state_dict):
    for state in (state_dict.get("state") or {}).values():
        if isinstance(state, dict) and any(key in state for key in _BNB_STATE_KEYS):
            raise RuntimeError(
                "Unsloth: this optimizer checkpoint holds bitsandbytes 8-bit state, which the "
                "bitsandbytes stand-in's torch optimizer cannot resume. Resume with bitsandbytes "
                "installed, or start the optimizer fresh."
            )


def torch_optimizer_fallback(name, base):
    """A `base` subclass named `name` whose construction returns a plain `base` (torch) optimizer,
    ignoring bitsandbytes-only arguments."""

    def __new__(cls, params, *args, **kwargs):
        for key in _BNB_ONLY:
            kwargs.pop(key, None)
        params = _materialize([params] if isinstance(params, torch.Tensor) else params)
        if any(
            getattr(p, "dtype", None) is torch.float16 and p.requires_grad
            for p in _parameters(params)
        ):
            # bitsandbytes keeps fp32 math for fp16 weights; torch state in fp16 overflows.
            raise RuntimeError(
                f"Unsloth: `{name}` without bitsandbytes runs torch.optim.{base.__name__}, which "
                "cannot step float16 parameters safely. Train float32 / bfloat16 parameters, "
                "or install bitsandbytes."
            )
        if name not in _NOTICED:
            _NOTICED.add(name)
            # print, not warnings.warn: training scripts commonly silence warnings.
            print(
                f"Unsloth: bitsandbytes is unavailable, so `{name}` runs torch.optim."
                f"{base.__name__} with full precision optimizer states (more memory than 8-bit "
                "states, same update rule). Install bitsandbytes for 8-bit or paged optimizers."
            )
        optimizer = base(params, *args, **kwargs)
        hook = getattr(optimizer, "register_load_state_dict_pre_hook", None)
        if hook is not None:
            hook(_refuse_bnb_state)
        return optimizer

    # A torch.optim subclass, so transformers >= 5 (which calls anything else a factory) builds it
    # like a class; __new__ hands back the plain `base` instance, so __init__ here never runs.
    return type(
        name,
        (base,),
        {"__new__": __new__, "__module__": "bitsandbytes.optim", "__qualname__": name},
    )
