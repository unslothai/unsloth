# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""bitsandbytes.optim names for Unsloth's bitsandbytes stand-in. Each runs the torch optimizer with
the same update rule and full precision state (see optimizer.py); Lion has no torch counterpart and
refuses with a clear error."""

import torch

from .optimizer import (
    GlobalOptimManager,
    Optimizer1State,
    Optimizer2State,
    Optimizer8bit,
    torch_optimizer_fallback,
)

_ADAM = ("Adam", "Adam8bit", "Adam32bit", "PagedAdam", "PagedAdam8bit", "PagedAdam32bit")
_ADAMW = ("AdamW", "AdamW8bit", "AdamW32bit", "PagedAdamW", "PagedAdamW8bit", "PagedAdamW32bit")
_SGD = ("SGD", "SGD8bit", "SGD32bit")
_RMSPROP = ("RMSprop", "RMSprop8bit", "RMSprop32bit")
_ADAGRAD = ("Adagrad", "Adagrad8bit", "Adagrad32bit")

for _name in _ADAM:
    globals()[_name] = torch_optimizer_fallback(_name, torch.optim.Adam)
for _name in _ADAMW:
    globals()[_name] = torch_optimizer_fallback(_name, torch.optim.AdamW)
for _name in _SGD:
    globals()[_name] = torch_optimizer_fallback(_name, torch.optim.SGD)
for _name in _RMSPROP:
    globals()[_name] = torch_optimizer_fallback(_name, torch.optim.RMSprop)
for _name in _ADAGRAD:
    globals()[_name] = torch_optimizer_fallback(_name, torch.optim.Adagrad)


def _unsupported(name):
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            f"Unsloth: `{name}` needs the bitsandbytes package (no torch equivalent). Use "
            "optim='adamw_torch' or install bitsandbytes."
        )

    return type(
        name, (torch.optim.Optimizer,), {"__init__": __init__, "__module__": "bitsandbytes.optim"}
    )


for _name in (
    "Lion",
    "Lion8bit",
    "Lion32bit",
    "PagedLion",
    "PagedLion8bit",
    "PagedLion32bit",
    "AdEMAMix",
    "AdEMAMix8bit",
    "AdEMAMix32bit",
    "PagedAdEMAMix",
    "PagedAdEMAMix8bit",
    "PagedAdEMAMix32bit",
    "LAMB",
    "LAMB8bit",
    "LAMB32bit",
    "LARS",
    "LARS8bit",
    "LARS32bit",
):
    globals()[_name] = _unsupported(_name)

del _name
