# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LoRA for block-diagonal grouped linears.

DeepSeek-V4's grouped output projection (`DeepseekV4GroupedLinear`, and the
`FP8GroupedLinear` transformers swaps in for FP8 checkpoints) subclasses
`nn.Linear` and stores one `(n_groups * out_per_group, in_per_group)` weight,
but its forward is block diagonal: input `(..., n_groups, in_per_group)` gives
`(..., n_groups, out_per_group)`. PEFT sees an `nn.Linear`, wraps it with the
dense LoRA layer and adds `lora_B(lora_A(x))` of shape
`(..., n_groups, n_groups * out_per_group)` to it, which fails with
`The size of tensor a (1024) must match the size of tensor b (8192)`.

`GroupedLinearLoRA` keeps PEFT's dense parameters (`lora_A: in -> r`,
`lora_B: r -> n_groups * out_per_group`) so checkpoints, `merge_and_unload`
and every PEFT version from 0.18 load and save it as a plain LoRA, and only
changes the forward: group g reads rows `g * out_per_group : (g + 1) * out_per_group`
of `lora_B`, which is exactly the block that merging `lora_B @ lora_A` into the
weight would add to that group.
"""

import torch

__all__ = [
    "is_grouped_linear",
    "grouped_linear_classes",
    "register_grouped_linear_lora",
]


def is_grouped_linear(module):
    """An `nn.Linear` whose forward is block diagonal over `n_groups`."""
    return (
        isinstance(module, torch.nn.Linear)
        and isinstance(getattr(module, "n_groups", None), int)
        and type(module).forward is not torch.nn.Linear.forward
    )


def grouped_linear_classes(model):
    classes = []
    for module in model.modules():
        if is_grouped_linear(module) and type(module) not in classes:
            classes.append(type(module))
    return classes


def _grouped_lora_layer():
    from peft.tuners.lora.layer import Linear as LoraLinear
    class GroupedLinearLoRA(LoraLinear):
        """PEFT's dense LoRA layer with a block-diagonal LoRA forward."""

        def forward(self, x, *args, **kwargs):
            self._check_forward_args(x, *args, **kwargs)
            adapter_names = kwargs.pop("adapter_names", None)
            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                return self.base_layer(x, *args, **kwargs)
            if adapter_names is not None:
                raise NotImplementedError(
                    "Unsloth: mixed adapter batches are not supported on grouped linears."
                )
            if self.merged:
                return self.base_layer(x, *args, **kwargs)

            result = self.base_layer(x, *args, **kwargs)
            result_dtype = result.dtype
            n_groups = self.base_layer.n_groups
            out_per_group = result.shape[-1]
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A:
                    continue
                if active_adapter in getattr(self, "lora_variant", {}):
                    # DoRA and the other PEFT variants replace the plain LoRA sum with their
                    # own forward; this layer computes only the plain sum, so a variant would
                    # train without its magnitude vector and merge to a different weight.
                    raise NotImplementedError(
                        "Unsloth: DoRA and other LoRA variants are not supported on grouped "
                        "linears (block-diagonal `o_a_proj`). Use plain LoRA for these layers."
                    )
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x_cast = self._cast_input_dtype(x, lora_A.weight.dtype)
                # (..., n_groups, r)
                low_rank = lora_A(dropout(x_cast))
                # lora_B.weight is (n_groups * out_per_group, r): block g is rows of group g.
                weight_B = lora_B.weight.view(n_groups, out_per_group, -1)
                delta = torch.einsum("...gr,gor->...go", low_rank, weight_B)
                if lora_B.bias is not None:
                    delta = delta + lora_B.bias.view(n_groups, out_per_group)
                result = result + delta.to(result_dtype) * scaling
            return result.to(result_dtype)

    return GroupedLinearLoRA


def register_grouped_linear_lora(lora_config, model):
    """Map every grouped-linear class in `model` to `GroupedLinearLoRA` on `lora_config`.

    Returns the classes registered, empty when the model has none or PEFT
    predates custom module registration.
    """
    classes = grouped_linear_classes(model)
    if not classes or not hasattr(lora_config, "_register_custom_module"):
        return []
    if getattr(lora_config, "use_dora", False):
        raise NotImplementedError(
            "Unsloth: `use_dora = True` is not supported on a model with block-diagonal grouped "
            "linears (" + ", ".join(cls.__name__ for cls in classes) + "): the grouped LoRA "
            "forward computes the plain LoRA sum only, so the DoRA magnitude vector would never "
            "train. Use plain LoRA."
        )
    layer = _grouped_lora_layer()
    lora_config._register_custom_module({cls: layer for cls in classes})
    return classes
