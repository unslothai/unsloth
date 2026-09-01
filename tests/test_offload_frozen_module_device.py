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

Continued pretraining puts `embed_tokens` and `lm_head` in `target_modules`, so
PEFT wraps each in a `ModulesToSaveWrapper` and llama.py moves the trainable
copy onto the accelerator with `DEVICE_TYPE_TORCH`, the bare string "cuda".
A bare "cuda" carries no index and resolves to the CURRENT device, so a copy on
cuda:1 was moved to cuda:0 while the rest of its layer stayed behind.

The stub RECORDS the device it is handed rather than allocating, so the second
card is testable on any box, including CI runners with no GPU.
"""

import types
import pytest
import torch


def _load_helper():
    """Exec the helper from source: importing llama.py would pull hardware probes."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    src = (root / "unsloth" / "models" / "llama.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_offload_frozen_module_for_training"
    )
    # The annotations are evaluated when the def executes, so the names they mention have to be real here, not
    from typing import Optional

    ns = {"torch": torch, "Optional": Optional, "ModulesToSaveWrapper": object}
    module = ast.Module(body = [fn], type_ignores = [])
    exec(compile(ast.fix_missing_locations(module), "<llama>", "exec"), ns)
    return ns["_offload_frozen_module_for_training"]


offload = _load_helper()


class _Recorder:
    """Stands in for a submodule, recording `.to()` instead of allocating."""

    def __init__(
        self,
        device,
        dtype = torch.float32,
    ):
        self.weight = types.SimpleNamespace(
            device = torch.device(device),
            dtype = dtype,
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
    def __init__(
        self,
        copy_device,
        original_device = None,
        dtype = torch.float32,
    ):
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
    w = _Wrapper("meta")
    offload(w, "cuda")
    assert torch.device(_moved_to(w)).type == "cuda"


def test_mps_keeps_its_own_device_rather_than_being_sent_to_cuda():
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




# The DEFAULT path.
# `use_gradient_checkpointing = "unsloth"` offloads the trained embedding and head to disk BEFORE `_get_peft_model`
# runs, so PEFT builds `modules_to_save.default` on CPU and the copy has no index left to preserve.
# llama.py records the real placement first (`input_embeddings_device` / `output_embeddings_device`, captured just above
# that offload) and hands it over.
def test_a_copy_rebuilt_on_cpu_by_the_disk_offload_uses_the_recorded_device():
    """The default-path regression: without this the copy lands on cuda:0."""
    w = _Wrapper("cpu")
    offload(w, "cuda", original_device = torch.device("cuda", 1))
    assert _moved_to(w) == torch.device("cuda", 1)


def test_the_recorded_device_is_used_for_the_first_card_too():
    w = _Wrapper("cpu")
    offload(w, "cuda", original_device = torch.device("cuda", 0))
    assert _moved_to(w) == torch.device("cuda", 0)


def test_a_copy_that_still_has_an_index_beats_the_recorded_device():
    """The copy is live truth; the record only covers a copy that lost its index."""
    w = _Wrapper("cuda:1")
    offload(w, "cuda", original_device = torch.device("cuda", 0))
    assert _moved_to(w) == torch.device("cuda", 1)


def test_a_recorded_cpu_device_does_not_pin_the_copy_to_cpu():
    """A model loaded on CPU records cpu; it must still reach the accelerator."""
    w = _Wrapper("cpu")
    offload(w, "cuda", original_device = torch.device("cpu"))
    assert torch.device(_moved_to(w)).type == "cuda"


def test_no_recorded_device_behaves_exactly_as_before():
    """The two call sites that have nothing to record must be unaffected."""
    w = _Wrapper("cpu")
    offload(w, "cuda", original_device = None)
    assert torch.device(_moved_to(w)).type == "cuda"


def test_a_meta_copy_also_uses_the_recorded_device():
    w = _Wrapper("meta")
    offload(w, "cuda", original_device = torch.device("cuda", 2))
    assert _moved_to(w) == torch.device("cuda", 2)


def test_the_continued_pretraining_call_sites_pass_the_recorded_device():
    """A helper supporting it is worthless if the callers drop it, which was the bug."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "unsloth" / "models" / "llama.py").read_text(encoding = "utf-8"))

    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) != "_offload_frozen_module_for_training":
            continue
        kwargs = {kw.arg for kw in node.keywords}
        if "offload_device" not in kwargs:
            continue  # the pre-wrapped PEFT branch, which records nothing
        assert "original_device" in kwargs, (
            "a continued-pretraining call dropped original_device, so a copy "
            "rebuilt on CPU by the disk offload lands on cuda:0"
        )
        checked += 1
    assert checked == 2, f"expected 2 continued-pretraining call sites, found {checked}"
