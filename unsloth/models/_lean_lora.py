# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Plain LoRA without PEFT's per-call checks, fp32 round trip and scale-1 multiply."""

import types

import torch


def lean_lora_forward(self, x, *args, **kwargs):
    adapter = self._unsloth_adapter
    if (
        self.disable_adapters
        or self.merged
        or args
        or kwargs
        or len(self.lora_A)
        != 1  # another adapter was added or selected after install: PEFT routes it
        or self.active_adapters != [adapter]
        or (
            not torch.is_autocast_enabled(x.device.type)
            and x.dtype != self.lora_A[adapter].weight.dtype
        )
    ):
        return self._unsloth_peft_forward(x, *args, **kwargs)
    lora = self.lora_B[adapter](self.lora_A[adapter](x))
    scaling = self.scaling[adapter]
    return self.base_layer(x) + (lora if scaling == 1 else lora * scaling)


def apply_lean_lora(model) -> int:
    from peft.tuners.lora.layer import Linear

    # Unsloth's compiler (a FastModel load earlier in this process) already gave PEFT a compiled forward.
    if Linear.forward.__name__ == "unsloth_forward":
        return 0
    count = 0
    for module in model.modules():
        if type(module) is not Linear or len(module.lora_A) != 1 or module.lora_variant:
            continue
        adapter = next(iter(module.lora_A))
        if not isinstance(module.lora_dropout[adapter], torch.nn.Identity):
            continue
        # Bound methods, so the deepcopy that merges for saving rebinds them to the copy.
        module._unsloth_adapter = adapter
        module._unsloth_peft_forward = module.forward
        module.forward = types.MethodType(lean_lora_forward, module)
        count += 1
    return count
