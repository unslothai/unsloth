# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""`_offload_frozen_module_for_training` must not drag the copy off its card.

Continued pretraining puts `embed_tokens` and `lm_head` in `target_modules`,
which makes PEFT build a `ModulesToSaveWrapper` around each. `llama.py` then
calls this helper to move the trainable copy onto the accelerator and offload
the frozen original to CPU, passing `DEVICE_TYPE_TORCH`, which is the bare
string "cuda".

A bare "cuda" has no index:

    >>> torch.device("cuda").index is None
    True
    >>> torch.zeros(2).to("cuda").device
    device(type='cuda', index=0)

So on a model the loader split across cards, a copy sitting on cuda:1 was moved
to cuda:0 while the rest of its layer stayed behind, and the forward then mixed
devices. Single-GPU runs never saw it, because there cuda:0 is where the copy
already was.

These tests drive the real helper with a stub that RECORDS the device it is
handed rather than allocating anything. That is deliberate: the case under test
is a second card, and recording the argument tests the decision on any box,
including CI runners with no GPU at all. The one test that does allocate is
skipped without CUDA and only covers the no-regression direction.
"""

import types
import pytest
import torch


def _load_helper():
    """Import the helper without dragging in the whole loader.

    `unsloth.models.llama` is expensive to import and pulls hardware probes, so
    the suite would become a GPU test by accident. The function is self
    contained, so read it out of the module source and exec it in a namespace
    holding only what it closes over.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    src = (root / "unsloth" / "models" / "llama.py").read_text()
    tree = ast.parse(src)
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_offload_frozen_module_for_training"
    )
    # The annotations are evaluated when the def executes, so the names they
    # mention have to be real here, not placeholders.
    from typing import Optional

    ns = {"torch": torch, "Optional": Optional, "ModulesToSaveWrapper": object}
    module = ast.Module(body = [fn], type_ignores = [])
    exec(compile(ast.fix_missing_locations(module), "<llama>", "exec"), ns)
    return ns["_offload_frozen_module_for_training"]


offload = _load_helper()


class _Recorder:
    """Stands in for a submodule, recording `.to()` instead of allocating."""

    def __init__(self, device, dtype = torch.float32):
        self.weight = types.SimpleNamespace(
            device = torch.device(device), dtype = dtype,
        )
        self.to_calls = []
        self.requires_grad_calls = []

    def to(self, **kwargs):
        self.to_calls.append(kwargs)
        if "device" in kwargs:
            self.weight.device = torch.device(kwargs["device"])
        if "dtype" in kwargs:
            self.weight.dtype = kwargs["dtype"]
        return self

    def requires_grad_(self, flag):
        self.requires_grad_calls.append(flag)
        return self


class _Wrapper:
    def __init__(self, copy_device, original_device = None, dtype = torch.float32):
        self.modules_to_save = types.SimpleNamespace(
            default = _Recorder(copy_device, dtype),
        )
        self.original_module = _Recorder(original_device or copy_device, dtype)


def _moved_to(wrapper):
    call = wrapper.modules_to_save.default.to_calls[-1]
    return torch.device(call["device"])


def test_a_copy_on_the_second_card_stays_on_the_second_card():
    """The regression. Reverting the fix sends this to cuda:0."""
    w = _Wrapper("cuda:1")
    offload(w, "cuda")
    assert _moved_to(w) == torch.device("cuda", 1)


def test_a_copy_on_the_third_card_stays_on_the_third_card():
    """Not special cased to index 1: any index the copy already has is kept."""
    w = _Wrapper("cuda:3")
    offload(w, "cuda")
    assert _moved_to(w) == torch.device("cuda", 3)


def test_a_copy_on_the_first_card_is_unchanged():
    w = _Wrapper("cuda:0")
    offload(w, "cuda")
    assert _moved_to(w) == torch.device("cuda", 0)


def test_a_copy_on_cpu_is_still_moved_onto_the_accelerator():
    """The single-GPU path, which must keep behaving exactly as before."""
    w = _Wrapper("cpu")
    offload(w, "cuda")
    assert torch.device(_moved_to(w)).type == "cuda"


def test_a_copy_on_meta_is_still_moved_onto_the_accelerator():
    """A deferred-init module has no index to preserve."""
    w = _Wrapper("meta")
    offload(w, "cuda")
    assert torch.device(_moved_to(w)).type == "cuda"


def test_mps_keeps_its_own_device_rather_than_being_sent_to_cuda():
    """Apple silicon reaches here with DEVICE_TYPE_TORCH == "mps"."""
    w = _Wrapper("mps")
    offload(w, "mps")
    assert torch.device(_moved_to(w)).type == "mps"


def test_float16_is_still_promoted_to_float32():
    """PR #1200: Tesla T4 must train these in float32."""
    w = _Wrapper("cuda:1", dtype = torch.float16)
    offload(w, "cuda")
    assert w.modules_to_save.default.to_calls[-1]["dtype"] == torch.float32


def test_bfloat16_is_left_alone():
    w = _Wrapper("cuda:1", dtype = torch.bfloat16)
    offload(w, "cuda")
    assert w.modules_to_save.default.to_calls[-1]["dtype"] == torch.bfloat16


def test_the_trainable_copy_is_the_only_one_that_requires_grad():
    w = _Wrapper("cuda:1")
    offload(w, "cuda")
    assert w.modules_to_save.default.requires_grad_calls == [True]
    assert w.original_module.requires_grad_calls == [False]


def test_the_frozen_original_is_offloaded_to_cpu():
    w = _Wrapper("cuda:1")
    offload(w, "cuda")
    assert torch.device(w.original_module.to_calls[-1]["device"]) == torch.device("cpu")


def test_offload_device_none_leaves_the_frozen_original_in_place():
    w = _Wrapper("cuda:1")
    offload(w, "cuda", offload_device = None)
    assert w.original_module.to_calls == []
    assert w.original_module.requires_grad_calls == [False]


def test_a_module_without_modules_to_save_is_untouched():
    plain = _Recorder("cuda:1")
    assert offload(plain, "cuda") is None
    assert plain.to_calls == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_on_real_hardware_a_module_on_the_current_card_is_unchanged():
    """No-regression check with real allocation. Single card is enough."""

    class _Real:
        def __init__(self):
            self.modules_to_save = types.SimpleNamespace(
                default = torch.nn.Linear(4, 4).cuda(),
            )
            self.original_module = torch.nn.Linear(4, 4).cuda()

    w = _Real()
    before = w.modules_to_save.default.weight.device
    offload(w, "cuda")
    assert w.modules_to_save.default.weight.device == before
    assert w.original_module.weight.device == torch.device("cpu")
    assert w.modules_to_save.default.weight.requires_grad
    assert not w.original_module.weight.requires_grad
