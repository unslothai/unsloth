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

"""Layer-wise learning-rate decay: lr(i) = base_lr * decay ** (depth - 1 - i), per layer stack.

Stdlib-only so the grouping logic is testable without importing unsloth.
"""

import re

__all__ = ["make_layerwise_lr_param_groups"]

_LAYER_INDEX_RE = re.compile(r"(?:^|\.)(?:layers|blocks)\.(\d+)\.")
_MODULES_TO_SAVE_SUFFIX = "modules_to_save.default.weight"


def get_layer_key(name):
    # (stack prefix, index): decoder and vision towers are independent stacks.
    match = _LAYER_INDEX_RE.search(name)
    return (name[: match.start()], int(match.group(1))) if match is not None else None


def make_layerwise_lr_param_groups(
    model,
    lr,
    weight_decay,
    layerwise_lr_decay,
    embedding_lr = None,
    num_layers = None,
    decay_parameter_names = None,
    verbose = True,
):
    if not (0.0 < layerwise_lr_decay <= 1.0):
        raise ValueError(
            f"Unsloth: layerwise_lr_decay must be in (0, 1], got {layerwise_lr_decay}."
        )

    trainable = [
        (name, param)
        for name, param in model.named_parameters()
        if getattr(param, "requires_grad", False)
    ]
    if decay_parameter_names is not None:
        decay_parameter_names = set(decay_parameter_names)

    stack_depth = {}
    for name, _ in trainable:
        key = get_layer_key(name)
        if key is not None:
            prefix, idx = key
            stack_depth[prefix] = max(stack_depth.get(prefix, 0), idx + 1)
    # Config depth is only unambiguous with a single stack; it is a floor for partial adapters.
    if num_layers is not None and len(stack_depth) == 1:
        prefix = next(iter(stack_depth))
        stack_depth[prefix] = max(stack_depth[prefix], num_layers)
    max_depth = max(stack_depth.values(), default = 0)
    shallowest_lr = lr * (layerwise_lr_decay ** (max_depth - 1)) if max_depth else lr

    groups = {}
    for name, param in trainable:
        key = get_layer_key(name)
        if key is not None:
            prefix, idx = key
            group_lr = lr * (layerwise_lr_decay ** (stack_depth[prefix] - 1 - idx))
        elif name.endswith(_MODULES_TO_SAVE_SUFFIX) and embedding_lr is not None:
            group_lr = embedding_lr
        elif "embed" in name and "lm_head" not in name:
            group_lr = shallowest_lr
        else:
            # Final norm, lm_head and other output-side params train at the top rate.
            group_lr = lr
        decays = decay_parameter_names is None or name in decay_parameter_names
        group = groups.get((group_lr, decays))
        if group is None:
            group = {"params": [], "lr": group_lr, "weight_decay": weight_decay if decays else 0.0}
            groups[(group_lr, decays)] = group
        group["params"].append(param)

    param_groups = [groups[key] for key in sorted(groups, key = lambda k: (k[0], not k[1]))]

    if verbose and param_groups:
        rates = [g["lr"] for g in param_groups]
        print(
            f"Unsloth: Layer-wise LR decay = {layerwise_lr_decay} across {len(stack_depth)} "
            f"stack(s), max depth {max_depth} -> {len(param_groups)} groups, "
            f"lr in [{min(rates):.2e}, {max(rates):.2e}]."
        )
    return param_groups
