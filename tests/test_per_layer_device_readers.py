# Unsloth - Fast finetuning of LLMs
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The pipeline-parallel inference readers must survive a layer with no device index.

`unsloth_zoo.patching_utils.verify_and_set_device` records where each decoder
layer lives. It used to record `device.index` verbatim, and
`torch.device("cpu").index` is None, so a CPU-offloaded or not-yet-materialised
layer ended `move_to_device` with "ValueError: Invalid target device: None"
(#3538). The five readers now resolve the placement through
`unsloth.models._utils.per_layer_device`, which prefers the `torch.device`
unsloth_zoo publishes and falls back through the index and then the layer's own
parameters.

CPU-only; the one CUDA assertion is skipped without a GPU.
"""

from __future__ import annotations

import pathlib
import re

import pytest
import torch


# Every file that reads the per-layer device, with the reader it lives in and the
# per-accelerator tuples that same reader subscripts with the buffer index. Keep
# in step with a grep for `_per_layer_device` across the repository (#3538).
READERS = {
    # llama's reader is the nested LlamaModel_fast_forward_inference_custom,
    # which is why the scope is located from the call site below and not by name.
    "unsloth/models/llama.py": ("temp_gates", "temp_ups"),
    "unsloth/models/granite.py": (),
    "unsloth/models/gemma.py": ("out_weights",),
    "unsloth/models/gemma2.py": ("out_weights",),
    "unsloth/models/cohere.py": ("out_weights",),
}

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _device_or_none(value):
    """`torch.device(value)`, or None when this build cannot name that device."""
    try:
        return torch.device(value)
    except (RuntimeError, TypeError, ValueError):
        return None


# torch 2.6 raises "RuntimeError: Cannot access accelerator device when none is
# available" from the bare `torch.device(0)` form on a host with nothing visible,
# which is exactly how the CPU lanes run. Where that is the case per_layer_device
# falls back to the layer itself, so the expectations below have to follow rather
# than assume cuda:0 is always constructible.
INDEXED_DEVICES_CONSTRUCTIBLE = _device_or_none(0) is not None


class _Layer(torch.nn.Module):
    """A decoder layer stand-in carrying whatever unsloth_zoo published."""

    def __init__(self, device=None, index="absent", parameter_device="cpu"):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.zeros(2, device=torch.device(parameter_device)),
            requires_grad=False,
        )
        if device is not None:
            self._per_layer_device = device
        if index != "absent":
            self._per_layer_device_index = index


def test_current_unsloth_zoo_device_wins():
    from unsloth.models._utils import per_layer_device

    layer = _Layer(device=torch.device("cuda:2"), index=2)
    device, buffer_index = per_layer_device(layer)
    assert device == torch.device("cuda:2")
    assert buffer_index == 2


def test_cpu_offloaded_layer_resolves_to_cpu_not_cuda_zero():
    """The #3538 case as current unsloth_zoo publishes it."""
    from unsloth.models._utils import per_layer_device

    layer = _Layer(device=torch.device("cpu"), index="cpu")
    device, buffer_index = per_layer_device(layer)
    assert device == torch.device("cpu")
    assert isinstance(buffer_index, int), (
        "gemma, gemma2 and cohere subscript a per-device tuple with this"
    )


def test_older_unsloth_zoo_integer_index_is_unchanged():
    """An unsloth_zoo that publishes only the index still works."""
    from unsloth.models._utils import per_layer_device

    layer = _Layer(index=1, parameter_device="cpu")
    device, buffer_index = per_layer_device(layer)
    if INDEXED_DEVICES_CONSTRUCTIBLE:
        assert device == torch.device(1)
    else:
        assert device == torch.device("cpu")
    assert buffer_index == 1, "the buffer subscript must survive either way"


def test_older_unsloth_zoo_none_index_reads_the_layer_instead():
    """The exact #3538 state: `_per_layer_device_index = None` and nothing else."""
    from unsloth.models._utils import per_layer_device

    layer = _Layer(index=None, parameter_device="cpu")
    device, buffer_index = per_layer_device(layer)
    assert device == torch.device("cpu"), (
        "a None index must not resolve to cuda:0, which would move a CPU layer's "
        "activations off the layer"
    )
    assert isinstance(buffer_index, int)


def test_a_layer_with_no_attributes_keeps_the_historical_default():
    """Every reader used to spell this `getattr(layer, ..., 0)`."""
    from unsloth.models._utils import per_layer_device

    layer = _Layer()
    device, buffer_index = per_layer_device(layer)
    if INDEXED_DEVICES_CONSTRUCTIBLE:
        assert device == torch.device(0)
    else:
        assert device == torch.device("cpu")
    assert buffer_index == 0


def test_a_garbage_index_falls_back_rather_than_raising():
    from unsloth.models._utils import per_layer_device

    layer = _Layer(index="not a device")
    device, buffer_index = per_layer_device(layer)
    assert isinstance(device, torch.device)
    if INDEXED_DEVICES_CONSTRUCTIBLE:
        assert device == torch.device(0)
    else:
        assert device == torch.device("cpu")
    assert buffer_index == 0


@pytest.mark.parametrize(
    "device,index",
    [
        (torch.device("cpu"), "cpu"),
        (torch.device("cuda:0"), 0),
        (None, None),
        (None, 3),
        (None, "absent"),
    ],
)
def test_move_to_device_accepts_every_resolution(device, index):
    """The contract that broke: move_to_device only takes int, str or torch.device."""
    from unsloth.models._utils import move_to_device, per_layer_device

    layer = _Layer(device=device, index=index)
    resolved, buffer_index = per_layer_device(layer)
    assert isinstance(resolved, torch.device)

    per_device_buffers = tuple(range(8))
    assert per_device_buffers[buffer_index] == buffer_index

    # move_to_device itself only promises int, str and torch.device, and a
    # torch.device is what it was handed, so the type contract is satisfied. The
    # move is only performed where it cannot need hardware.
    if resolved.type == "cpu":
        moved = move_to_device(resolved, torch.zeros(2))
        assert moved.device == torch.device("cpu")


def test_cpu_offloaded_layer_no_longer_raises_invalid_target_device():
    """The verbatim #3538 failure, driven through move_to_device."""
    from unsloth.models._utils import move_to_device, per_layer_device

    layer = _Layer(device=torch.device("cpu"), index="cpu")
    resolved, _ = per_layer_device(layer)
    hidden_states = torch.zeros(1, 2, 4)
    position_ids = torch.zeros(1, 2, dtype=torch.long)
    hidden_states, position_ids = move_to_device(resolved, hidden_states, position_ids)
    assert hidden_states.device == torch.device("cpu")
    assert position_ids.device == torch.device("cpu")

    # And the shape that used to reach move_to_device is still rejected loudly.
    with pytest.raises(ValueError, match="Invalid target device"):
        move_to_device(None, hidden_states)


def test_unsloth_zoo_setter_and_reader_agree():
    """Round trip through the writer that publishes the attributes."""
    from unsloth.models._utils import per_layer_device

    try:
        from unsloth_zoo.patching_utils import verify_and_set_device
    except Exception as exception:  # pragma: no cover
        pytest.skip(f"unsloth_zoo not importable: {exception}")

    layer = torch.nn.Linear(4, 4)
    verify_and_set_device(layer)
    device, buffer_index = per_layer_device(layer)
    assert device == torch.device("cpu")
    assert isinstance(buffer_index, int)


@pytest.mark.parametrize("path", sorted(READERS))
def test_every_reader_goes_through_the_helper(path):
    """Wiring check: no reader may index with the raw attribute again."""
    source = (REPOSITORY_ROOT / path).read_text()
    assert 'getattr(decoder_layer, "_per_layer_device_index"' not in source, (
        f"{path} reads the raw index again; a None there is #3538"
    )
    assert "per_layer_device(decoder_layer)" in source, (
        f"{path} no longer resolves the layer device through per_layer_device"
    )
    assert re.search(r"move_to_device\(\s*layer_device", source), (
        f"{path} does not hand the resolved device to move_to_device"
    )


@pytest.mark.parametrize("path", sorted(READERS))
def test_every_reader_imports_the_helper_explicitly(path):
    """Do not rely on `from .llama import *` to carry the name across."""
    source = (REPOSITORY_ROOT / path).read_text()
    assert re.search(r"^from \._utils import .*per_layer_device", source, re.MULTILINE), (
        f"{path} must import per_layer_device from ._utils explicitly"
    )


def _reader_scopes(source: str, path: str):
    """(name, lineno) of the innermost function around each per_layer_device call."""
    import ast

    tree = ast.parse(source)
    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) != "per_layer_device":
            continue
        enclosing = [
            function for function in functions
            if function.lineno <= node.lineno <= (function.end_lineno or function.lineno)
        ]
        assert enclosing, f"{path}: per_layer_device call at module level?"
        innermost = max(enclosing, key=lambda function: function.lineno)
        found.append((innermost.name, innermost.lineno))
    return found


@pytest.mark.parametrize("path", sorted(READERS))
def test_no_reader_reads_a_name_it_never_binds(path):
    """The reader functions must not reference an unbound local.

    gemma, gemma2, cohere and llama all pull a second value out of the pair
    `per_layer_device` returns and use it to subscript a per-accelerator tuple
    (`out_weights`, `temp_gates`, `temp_ups`). Binding only the device and
    discarding the index leaves those subscripts reading a name that no longer
    exists, which Python only reports at the first decode step, deep inside
    generation. symtable classifies such a name as an implicit global, so
    checking it against the module's real attributes catches it statically, star
    imports and all.
    """
    import builtins
    import importlib
    import symtable

    module = importlib.import_module("unsloth.models." + pathlib.Path(path).stem)
    source = (REPOSITORY_ROOT / path).read_text()
    wanted = _reader_scopes(source, path)
    assert wanted, f"{path}: no per_layer_device call found"

    def scopes(table):
        yield table
        for child in table.get_children():
            yield from scopes(child)

    checked = []
    unbound = []
    for scope in scopes(symtable.symtable(source, path, "exec")):
        if scope.get_type() != "function":
            continue
        if (scope.get_name(), scope.get_lineno()) not in wanted:
            continue
        checked.append(scope.get_name())
        for symbol in scope.get_symbols():
            name = symbol.get_name()
            if not symbol.is_global():
                continue
            if hasattr(module, name) or hasattr(builtins, name):
                continue
            unbound.append(name)

    assert len(checked) == len(wanted), (
        f"{path}: located {wanted} from the call sites but symtable matched {checked}"
    )
    assert not unbound, (
        f"{path}:{checked} reads {sorted(set(unbound))} without binding it and "
        f"without {module.__name__} defining it, so generation raises NameError"
    )


@pytest.mark.parametrize("path", sorted(READERS))
def test_per_accelerator_tuples_are_still_subscripted_by_an_int(path):
    """A reader that uses a per-accelerator tuple must keep the buffer index."""
    source = (REPOSITORY_ROOT / path).read_text()
    for tuple_name in READERS[path]:
        assert f"{tuple_name}[device_index]" in source, (
            f"{path} subscripts {tuple_name} with something other than the "
            "buffer index per_layer_device returns"
        )
        assert "layer_device, device_index = per_layer_device(decoder_layer)" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real GPU")
def test_cuda_layer_path_is_unchanged():
    from unsloth.models._utils import move_to_device, per_layer_device

    try:
        from unsloth_zoo.patching_utils import verify_and_set_device
    except Exception as exception:  # pragma: no cover
        pytest.skip(f"unsloth_zoo not importable: {exception}")

    layer = torch.nn.Linear(4, 4).cuda()
    verify_and_set_device(layer)
    device, buffer_index = per_layer_device(layer)

    assert device.type == "cuda"
    assert buffer_index == torch.cuda.current_device()
    moved = move_to_device(device, torch.zeros(2))
    assert moved.device == next(layer.parameters()).device
