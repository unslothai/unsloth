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

"""_attach_bnb_multidevice_hooks must leave a one-device model alone. No GPU needed.

Measured on 2x Tesla T4, unsloth/Qwen3-0.6B, load_in_4bit = True:

    device_map = "balanced"  ->  ValueError: You can't train a model that has been
    loaded in 8-bit or 4-bit precision on a different device than the one you're
    training on

with `model.hf_device_map == {"": 1}` and 396 modules carrying an `_hf_hook`.
Plain transformers + peft + trl, same box, same checkpoint, same flag, reaches
the identical placement (every parameter on cuda:1) and TRAINS, because
transformers leaves `hf_device_map` unset once the map collapses to one device.

The difference was ours. `dispatch_model` ends by assigning
`model.hf_device_map`, and `Accelerator.prepare_model` rejects a 4bit model
whose map has exactly one entry that is not device 0
(accelerate/accelerator.py, `elif len(model_devices) == 1:`). The hooks were
added for genuinely split weights (#5068, "Expected all tensors to be on the
same device"); a model that sits on one device has no cross-device edge for
them to bridge, so it now returns before dispatch.

`device_map = "balanced"` reaches that placement for any small quantized model:
`get_balanced_memory` caps every device except the last at about
`model_size / n_devices`, and `infer_auto_device_map` then subtracts the largest
layer from cuda:0 alone, leaving it too small to take the embedding.
"""

import ast
import os
import types
import warnings

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISION = os.path.join(HERE, "unsloth", "models", "vision.py")

_SRC = open(VISION, encoding = "utf-8").read()


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(msg)


def _load(device_count, dispatch_recorder, inferred_map = None):
    """Exec the two functions under test with their module globals supplied.

    `DEVICE_COUNT` is a module constant read at call time, so the number of
    devices the host really has never decides the outcome of a test.
    """
    ns = {
        "torch": torch, "os": os, "warnings": warnings,
        "logger": _Logger(), "DEVICE_COUNT": device_count,
    }
    mod = ast.parse(_SRC)
    # `_repair_dispatch_hooks` (#9995) is called at the top of the function under
    # test, so exec the real one rather than stubbing it: these tests should fail
    # if the two ever stop composing.
    wanted = {
        "_attach_bnb_multidevice_hooks",
        "_infer_device_map_from_loaded_model",
        "_repair_dispatch_hooks",
    }
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(ast.get_source_segment(_SRC, node), ns)
            wanted.discard(node.name)
    assert not wanted, f"not found in vision.py: {sorted(wanted)}"
    if inferred_map is not None:
        ns["_infer_device_map_from_loaded_model"] = lambda model: dict(inferred_map)

    # The function imports dispatch_model from `accelerate` at call time.
    import accelerate
    real = getattr(accelerate, "dispatch_model", None)
    accelerate.dispatch_model = dispatch_recorder
    ns["_restore_accelerate"] = lambda: setattr(accelerate, "dispatch_model", real)
    return ns


class _Param:
    """Only `.device` is read before the placement decision is made."""

    def __init__(self, device):
        self.device = device


def _model(devices):
    return types.SimpleNamespace(
        parameters = lambda: iter([_Param(d) for d in devices]),
        named_parameters = lambda: iter(
            [(f"p{i}", _Param(d)) for i, d in enumerate(devices)]),
        is_loaded_in_4bit = True,
        _skip_keys_device_placement = None,
    )


def _run(devices, device_count = 2, inferred_map = None):
    calls = []

    def recorder(*args, **kwargs):
        calls.append(kwargs)

    ns = _load(device_count, recorder, inferred_map)
    try:
        ns["_attach_bnb_multidevice_hooks"](
            _model(devices), load_in_4bit = True, load_in_8bit = False,
            offload_embedding = False, fast_inference = False,
        )
    finally:
        ns["_restore_accelerate"]()
    return calls


CUDA0 = torch.device("cuda", 0)
CUDA1 = torch.device("cuda", 1)
CPU = torch.device("cpu")


def test_whole_model_on_the_second_card_is_not_dispatched():
    # The measured failure: `balanced` puts every parameter on cuda:1.
    # Dispatching sets hf_device_map = {"": 1}, which is what the trainer rejects.
    assert _run([CUDA1, CUDA1, CUDA1]) == []


def test_whole_model_on_the_first_card_is_not_dispatched():
    # The pre-existing no-op, unchanged.
    assert _run([CUDA0, CUDA0]) == []


def test_whole_model_on_a_fourth_card_is_not_dispatched():
    assert _run([torch.device("cuda", 3)], device_count = 4) == []


def test_a_real_split_is_still_dispatched():
    # #5068: weights spread over two cards crash on the first forward without
    # AlignDevicesHook. That case must keep its hooks.
    calls = _run([CUDA0, CUDA1], inferred_map = {"model.embed_tokens": CUDA0,
                                                 "model.layers": CUDA1})
    assert len(calls) == 1, calls
    assert calls[0]["force_hooks"] is True
    assert calls[0]["device_map"] == {"model.embed_tokens": 0, "model.layers": 1}


def test_cpu_offload_alongside_a_gpu_is_still_dispatched():
    calls = _run([CUDA1, CPU], inferred_map = {"model.layers": CUDA1,
                                               "model.embed_tokens": CPU})
    assert len(calls) == 1, calls
    assert calls[0]["device_map"] == {"model.layers": 1, "model.embed_tokens": "cpu"}
    # A cpu entry must not become the main device.
    assert calls[0]["main_device"] == 1


def test_a_cpu_only_model_is_not_dispatched():
    assert _run([CPU, CPU]) == []


def test_an_already_dispatched_model_is_left_alone():
    calls = []
    ns = _load(2, lambda *a, **k: calls.append(k))
    model = _model([CUDA1])
    model.hf_device_map = {"": 1}
    try:
        ns["_attach_bnb_multidevice_hooks"](
            model, load_in_4bit = True, load_in_8bit = False,
            offload_embedding = False, fast_inference = False,
        )
    finally:
        ns["_restore_accelerate"]()
    assert calls == []


def test_a_non_bnb_model_is_not_dispatched():
    calls = []
    ns = _load(2, lambda *a, **k: calls.append(k))
    model = types.SimpleNamespace(parameters = lambda: iter([_Param(CUDA1)]))
    try:
        ns["_attach_bnb_multidevice_hooks"](
            model, load_in_4bit = False, load_in_8bit = False,
            offload_embedding = False, fast_inference = False,
        )
    finally:
        ns["_restore_accelerate"]()
    assert calls == []


def test_the_collapse_is_reported_only_when_it_cost_the_user_a_card():
    import io
    from contextlib import redirect_stdout

    def said_for(devices, device_count):
        buf = io.StringIO()
        with redirect_stdout(buf):
            _run(devices, device_count = device_count)
        return buf.getvalue()

    # Several cards, everything on a later one: the user asked for a split and
    # did not get one, so say so.
    said = said_for([CUDA1], 2)
    assert "cuda:1" in said and "balanced" in said
    # The one behaviour the guard gives up is named where the user will read it.
    assert ".to(model.device)" in said
    # One card, or the first card: nothing was lost, so stay quiet.
    assert said_for([CUDA0], 2) == ""
    assert said_for([CUDA0], 1) == ""


def test_the_guard_is_a_device_count_not_a_cuda0_comparison():
    """The old predicate `all_devs == {torch.device("cuda", 0)}` skipped only
    cuda:0. Keeping it would silently reinstate the failure."""
    assert 'all_devs == {default_cuda}' not in _SRC
    assert "if len(all_devs) == 1:" in _SRC
