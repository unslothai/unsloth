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
"""LoRA for block-diagonal grouped linears (DeepSeek-V4 `o_a_proj`, `FP8GroupedLinear`).

These subclass `nn.Linear` with one `(n_groups * out_per_group, in_per_group)` weight but a
block-diagonal forward, so PEFT's dense LoRA sum has the wrong shape. `GroupedLinearLoRA`
keeps PEFT's dense parameters, so saving, loading and merging are unchanged, and only
changes the forward: group g uses rows `g * out_per_group : (g + 1) * out_per_group` of `lora_B`.
"""

import functools

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

        _unsloth_grouped_lora = True

        def merge(self, *args, **kwargs):
            _refuse_fp8_grouped_merge([self])
            return super().merge(*args, **kwargs)

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
                    # Variants (DoRA) replace the plain LoRA sum, which is all this layer computes.
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


def _refuse_fp8_grouped_merge(modules):
    """PEFT adds B @ A to the stored weight; an FP8GroupedLinear's weight is fp8 read with an
    unchanged weight_scale_inv, so that merge is wrong or raises."""
    for module in modules:
        if not getattr(module, "_unsloth_grouped_lora", False):
            continue
        weight = getattr(module.get_base_layer(), "weight", None)
        if weight is not None and weight.dtype.itemsize == 1 and weight.is_floating_point():
            raise NotImplementedError(
                "Unsloth: cannot merge LoRA into an fp8 grouped linear in place. Load the "
                "model in 16-bit to merge, or save the adapter on its own."
            )


def _preflight_peft_merges():
    """Refuse before PEFT merges anything: it merges layer by layer, so a refusal from the
    grouped layer itself would leave the earlier layers already merged."""
    try:
        from peft.tuners.tuners_utils import BaseTuner
    except Exception:
        return
    for name in ("_unload_and_optionally_merge", "merge_adapter"):
        original = BaseTuner.__dict__.get(name)
        if original is None or getattr(original, "_unsloth_fp8_grouped_preflight", False):
            continue

        def make(original, name):
            def wrapped(self, *args, **kwargs):
                merging = name == "merge_adapter" or kwargs.get("merge", args[0] if args else True)
                if merging:
                    _refuse_fp8_grouped_merge(self.model.modules())
                return original(self, *args, **kwargs)

            wrapped._unsloth_fp8_grouped_preflight = True
            return functools.wraps(original)(wrapped)

        setattr(BaseTuner, name, make(original, name))


def _targets_module(lora_config, name):
    """Whether `lora_config` selects the module called `name`, by PEFT's own matcher.

    `target_modules = None` (PEFT resolves it later) or a config PEFT cannot read
    counts as targeted: keep the grouped forward rather than drop it.
    """
    target_modules = getattr(lora_config, "target_modules", None)
    if target_modules is None or target_modules == "all-linear":
        return True
    try:
        from peft.tuners.tuners_utils import check_target_module_exists
        return bool(check_target_module_exists(lora_config, name))
    except Exception:
        return True


def targeted_grouped_linear_classes(lora_config, model):
    """The grouped-linear classes `lora_config.target_modules` actually selects in `model`."""
    classes = []
    for name, module in model.named_modules():
        if (
            is_grouped_linear(module)
            and type(module) not in classes
            and _targets_module(lora_config, name)
        ):
            classes.append(type(module))
    return classes


def register_grouped_linear_lora(lora_config, model):
    """Map every targeted grouped-linear class in `model` to `GroupedLinearLoRA` on `lora_config`.

    Returns the classes registered, empty when `target_modules` selects no grouped
    linear or PEFT predates custom module registration.
    """
    classes = targeted_grouped_linear_classes(lora_config, model)
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
    _preflight_peft_merges()
    lora_config._register_custom_module({cls: layer for cls in classes})
    return classes


def register_grouped_linear_lora_for_adapter(model, adapter_path, **hub_kwargs):
    """The `PeftConfig` of a saved adapter with the grouped mapping registered, or None.

    The custom module mapping holds class objects, so it is not in the saved adapter
    config. None when nothing grouped is targeted or the config cannot be read.
    """
    if not grouped_linear_classes(model):
        return None
    try:
        from peft import PeftConfig
        peft_config = PeftConfig.from_pretrained(adapter_path, **hub_kwargs)
    except Exception:
        return None
    if not register_grouped_linear_lora(peft_config, model):
        return None
    return peft_config
