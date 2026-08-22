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

"""Adapter-level sidecar for UEmbed's trainable SPLADE heads.

The sparse heads are trained alongside the LoRA adapter, so they are part of the result of
a fine-tune and have to be written out with it. They are kept as a SIDECAR pair -
`sparse_weights.pt` (UEmbed's own `sparse_lm_heads` / `sparse_bias` layout) plus
`sparse_info.json` (`num_eos_tokens`) - written next to the adapter rather than folded into
the model's safetensors. That is what lets them survive a `merged_16bit` export: the merge
rewrites the backbone weights and never touches these two files, and the saved directory
then loads back as a UEmbed checkpoint through the very same path as the original.

Both halves are opt-in: `save_uembed_sparse_sidecar` returns False (writing nothing) for a
model without a SPLADE head, and `load_uembed_sparse_sidecar` returns False for a directory
without the sidecar, so plain dense embedders are untouched.

The reload is attach-AWARE. A directory saved from sentence-transformers may already
rebuild the sparse module from its own subfolder, so blindly attaching again would leave
the pipeline with two heads and a doubled sparse dimension. When a head is already present
this repopulates it from the sidecar (which holds the newest, trained values) instead.

Torch-only, so it imports without an accelerator and without importing `unsloth`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from typing import Any

import torch


def _sibling(module_name: str, standalone_name: str):
    """Import a sibling `uembed_*` module, by package or (standalone) by file path.

    The standalone names match the ones the siblings use for each other, so every entry
    point ends up sharing ONE copy of each module and `isinstance` keeps working.
    """
    if __package__:
        try:
            return importlib.import_module(f".{module_name}", __package__)
        except ImportError:
            pass

    if standalone_name in sys.modules:
        return sys.modules[standalone_name]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(standalone_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[standalone_name] = module
    spec.loader.exec_module(module)
    return module


def _uembed_splade():
    return _sibling("uembed_splade", "unsloth_uembed_splade_direct")


def _uembed_wiring():
    return _sibling("uembed_wiring", "unsloth_uembed_wiring_direct")


def _uembed_pooling():
    return _sibling("uembed_pooling", "unsloth_uembed_pooling_direct")


def _sidecar_weights_path(directory: Any) -> str | None:
    """Path of the directory's `sparse_weights.pt`, or None when it has none."""
    if not isinstance(directory, str) or not os.path.isdir(directory):
        return None
    path = os.path.join(directory, _uembed_splade().SPARSE_WEIGHTS_FILENAME)
    return path if os.path.isfile(path) else None


def _saved_num_eos_tokens(directory: str) -> int | None:
    """`num_eos_tokens` from the directory's own `sparse_info.json`, else None.

    None means "the directory does not say", which leaves an already-loaded head's value
    alone rather than resetting it to 0 and disabling `splade.last`.
    """
    pooling = _uembed_pooling()
    if not os.path.isfile(os.path.join(directory, pooling.SPARSE_INFO_FILENAME)):
        return None
    return pooling.read_num_eos_tokens(directory)


def _write_sparse_info(directory: str, num_eos_tokens: int) -> None:
    """Emit `sparse_info.json`, keeping any other keys the checkpoint already carried."""
    pooling = _uembed_pooling()
    path = os.path.join(directory, pooling.SPARSE_INFO_FILENAME)
    sparse_info: dict[str, Any] = {}
    if os.path.isfile(path):
        with open(path, encoding = "utf-8") as file:
            existing = json.load(file)
        if isinstance(existing, dict):
            sparse_info = existing
    sparse_info[pooling.NUM_EOS_TOKENS_KEY] = int(num_eos_tokens)
    with open(path, "w", encoding = "utf-8") as file:
        json.dump(sparse_info, file, indent = 2)


def save_uembed_sparse_sidecar(model: Any, save_directory: str) -> bool:
    """Write the model's trained SPLADE heads next to the adapter in `save_directory`.

    Returns False without writing anything when the model carries no sparse head, which is
    every non-UEmbed embedder.
    """
    module = _uembed_wiring().find_uembed_sparse_output(model)
    if module is None:
        return False

    splade = _uembed_splade()
    head = module.head
    os.makedirs(save_directory, exist_ok = True)
    torch.save(
        {
            splade.SPARSE_LM_HEADS_KEY: [
                weight.detach().cpu().clone() for weight in head.sparse_lm_heads
            ],
            splade.SPARSE_BIAS_KEY: [
                bias.detach().cpu().clone() for bias in head.sparse_bias
            ],
        },
        os.path.join(save_directory, splade.SPARSE_WEIGHTS_FILENAME),
    )
    _write_sparse_info(save_directory, head.num_eos_tokens)
    return True


def _restore_in_place(head: Any, path: str, num_eos_tokens: int | None) -> bool:
    """Copy the sidecar's tensors into an existing head; False when the layout differs.

    Nothing is written until every shape has been checked, so a mismatched sidecar leaves
    the head as it was instead of half-overwritten.
    """
    splade = _uembed_splade()
    state = torch.load(path, map_location = "cpu", weights_only = True)
    missing = [
        key
        for key in (splade.SPARSE_LM_HEADS_KEY, splade.SPARSE_BIAS_KEY)
        if not isinstance(state, dict) or key not in state
    ]
    if missing:
        raise ValueError(
            f"Unsloth: `{path}` is missing the key(s) {missing}; expected a dict with "
            f"`{splade.SPARSE_LM_HEADS_KEY}` and `{splade.SPARSE_BIAS_KEY}`."
        )

    saved = list(state[splade.SPARSE_LM_HEADS_KEY]) + list(state[splade.SPARSE_BIAS_KEY])
    current = list(head.sparse_lm_heads) + list(head.sparse_bias)
    if len(saved) != len(current):
        return False
    if any(tuple(a.shape) != tuple(b.shape) for a, b in zip(current, saved)):
        return False

    with torch.no_grad():
        for parameter, value in zip(current, saved):
            parameter.copy_(value.to(device = parameter.device, dtype = parameter.dtype))
    if num_eos_tokens is not None:
        head.num_eos_tokens = int(num_eos_tokens)
    return True


def load_uembed_sparse_sidecar(
    model: Any,
    model_dir: str,
    mode: str | None = None,
    num_eos_tokens: int | None = None,
) -> bool:
    """Reload `model_dir`'s sparse sidecar into `model`'s pipeline.

    Attach-aware: a model that already carries a sparse head has that head repopulated from
    the sidecar, so a reload can never stack a second one. Returns False when the directory
    ships no `sparse_weights.pt`.
    """
    path = _sidecar_weights_path(model_dir)
    if path is None:
        return False

    splade = _uembed_splade()
    wiring = _uembed_wiring()
    if num_eos_tokens is None:
        num_eos_tokens = _saved_num_eos_tokens(model_dir)

    existing = wiring.find_uembed_sparse_output(model)
    if existing is None:
        head = splade.SpladeHead.from_checkpoint(model_dir, num_eos_tokens = num_eos_tokens)
        return wiring.attach_uembed_sparse_output(model, head, mode)

    # A module reconstructed from modules.json has no process-local encode wrapper.
    # Restore it before repopulating the existing head; the helper is idempotent.
    wiring.patch_uembed_sparse_encode(model)
    if mode is not None:
        existing.set_mode(mode)
    if _restore_in_place(existing.head, path, num_eos_tokens):
        return True

    # A sidecar from a differently shaped checkpoint: swap the head rather than append a
    # second module, keeping the pipeline at exactly one sparse head.
    head = splade.SpladeHead.from_checkpoint(model_dir, num_eos_tokens = num_eos_tokens)
    reference = next(iter(existing.head.parameters()), None)
    if reference is not None:
        if reference.is_floating_point():
            head.to(device = reference.device, dtype = reference.dtype)
        else:
            head.to(device = reference.device)
    existing.head = head
    return True
