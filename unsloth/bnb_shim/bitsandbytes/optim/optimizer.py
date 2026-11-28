# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""bitsandbytes optimizers for Unsloth's bitsandbytes stand-in: the torch optimizer with the same
update (8-bit and paged state are bitsandbytes native features), after a one time notice."""

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
        self.module_weight_config_triple.append((module, param_name, config))


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


def torch_optimizer_fallback(
    name,
    base,
    rename = None,
    drop = (),
):
    """A `base` subclass named `name` that accepts and ignores bitsandbytes-only arguments."""

    def __init__(self, params, *args, **kwargs):
        if name not in _NOTICED:
            _NOTICED.add(name)
            # print, not warnings.warn: training scripts commonly silence warnings.
            print(
                f"Unsloth: bitsandbytes is not installed, so `{name}` runs torch.optim."
                f"{base.__name__} with full precision optimizer states (more memory than 8-bit "
                "states, same update rule). Install bitsandbytes for 8-bit or paged optimizers."
            )
        for key in _BNB_ONLY + tuple(drop):
            kwargs.pop(key, None)
        for old, new in (rename or {}).items():
            if old in kwargs:
                kwargs[new] = kwargs.pop(old)
        base.__init__(self, params, *args, **kwargs)

    return type(name, (base,), {"__init__": __init__, "__module__": "bitsandbytes.optim.optimizer"})
