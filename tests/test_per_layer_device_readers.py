# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The pipeline-parallel inference readers must survive a layer with no device index.

`verify_and_set_device` used to record `device.index` verbatim, and
`torch.device("cpu").index` is None, so a CPU-offloaded or not-yet-materialised layer ended
`move_to_device` with "ValueError: Invalid target device: None" (#3538). The five readers
now go through `unsloth.models._utils.per_layer_device`.
"""

from __future__ import annotations

import functools
import pathlib
import re

import pytest
from real_accelerator import (
    has_real_cuda,
)  # tests/_shared, on sys.path via tests/conftest.py
import torch


# Each reader file, with the per-accelerator tuples it subscripts with the buffer index.
# Keep in step with a grep for `_per_layer_device` across the repository (#3538).
READERS = {
    # llama's reader is nested, which is why the scope is located from the call site.
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


@functools.lru_cache(maxsize = 1)
def default_device_is_usable() -> bool:
    """Can `torch.device(0)`, the historical default, actually hold a tensor here.

    torch 2.6 raises from the bare `torch.device(0)` on an accelerator-less host while
    torch 2.11 returns `cuda:0` and only fails at the move, so the expectations below
    follow a real allocation rather than either torch version's answer.
    """
    try:
        torch.zeros(1, device = torch.device(0))
        return True
    except Exception:
        return False


class _Layer(torch.nn.Module):
    """A decoder layer stand-in carrying whatever unsloth_zoo published."""

    def __init__(
        self,
        device = None,
        index = "absent",
        parameter_device = "cpu",
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.zeros(2, device = torch.device(parameter_device)),
            requires_grad = False,
        )
        if device is not None:
            self._per_layer_device = device
        if index != "absent":
            self._per_layer_device_index = index


def test_current_unsloth_zoo_device_wins():
    from unsloth.models._utils import per_layer_device

    layer = _Layer(device = torch.device("cuda:2"), index = 2)
    device, buffer_index = per_layer_device(layer)
    assert device == torch.device("cuda:2")
    assert buffer_index == 2


def test_cpu_offloaded_layer_resolves_to_cpu_not_cuda_zero():
    """The #3538 case as current unsloth_zoo publishes it."""
    from unsloth.models._utils import per_layer_device

    layer = _Layer(device = torch.device("cpu"), index = "cpu")
    device, buffer_index = per_layer_device(layer)
    assert device == torch.device("cpu")
    assert isinstance(
        buffer_index, int
    ), "gemma, gemma2 and cohere subscript a per-device tuple with this"


def test_older_unsloth_zoo_integer_index_is_unchanged():
    """An unsloth_zoo that publishes only the index still works."""
    from unsloth.models._utils import per_layer_device

    layer = _Layer(index = 1, parameter_device = "cpu")
    device, buffer_index = per_layer_device(layer)
    if default_device_is_usable():
        assert device == torch.device(1)
    else:
        assert device == torch.device("cpu")
    assert buffer_index == 1, "the buffer subscript must survive either way"


def test_older_unsloth_zoo_none_index_reads_the_layer_instead():
    """The exact #3538 state: `_per_layer_device_index = None` and nothing else."""
    from unsloth.models._utils import per_layer_device

    layer = _Layer(index = None, parameter_device = "cpu")
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
    if default_device_is_usable():
        assert device == torch.device(0)
    else:
        assert device == torch.device("cpu")
    assert buffer_index == 0


def test_an_unavailable_accelerator_is_not_a_usable_device():
    """`torch.device(0)` succeeding is not the same as the device existing: torch 2.11
    builds `cuda:0` on a host with no CUDA and only fails at the move, which turned the
    "fall back to the layer" branch into "RuntimeError: No CUDA GPUs are available"."""
    from unsloth.models import _utils

    assert _utils._device_type_is_usable("cpu")
    assert _utils._device_type_is_usable(
        "meta"
    ), "meta has no is_available to ask, and an offloaded layer sits on it"
    assert _utils._device_type_is_usable(
        "not-a-backend"
    ), "an unknown backend must be taken at its word, not refused"
    assert _utils._device_type_is_usable("cuda") == torch.cuda.is_available()


def test_a_default_pointing_at_a_missing_accelerator_reads_the_layer(monkeypatch):
    """The CPU-only half of #3538: no attributes, no accelerator, still a device.
    Spoofed rather than measured, so the assertion is the same on the GPU legs."""
    from unsloth.models import _utils

    monkeypatch.setattr(_utils, "_device_type_is_usable", lambda device_type: device_type == "cpu")
    assert _utils._as_torch_device(0) is None
    assert _utils._as_torch_device("cpu") == torch.device("cpu")

    layer = _Layer(parameter_device = "cpu")
    device, buffer_index = _utils.per_layer_device(layer)
    assert device == torch.device(
        "cpu"
    ), "with no accelerator the layer's own parameters are the only real answer"
    assert buffer_index == 0, "the historical subscript must survive the fallback"


def test_an_available_accelerator_still_wins_the_default(monkeypatch):
    """The control: the probe must not steal the historical default on a real GPU."""
    from unsloth.models import _utils

    monkeypatch.setattr(_utils, "_device_type_is_usable", lambda device_type: True)
    layer = _Layer(parameter_device = "cpu")
    device, buffer_index = _utils.per_layer_device(layer)
    expected = _device_or_none(0)
    if expected is not None:
        assert device == expected
    assert buffer_index == 0


def test_a_garbage_index_falls_back_rather_than_raising():
    from unsloth.models._utils import per_layer_device

    layer = _Layer(index = "not a device")
    device, buffer_index = per_layer_device(layer)
    assert isinstance(device, torch.device)
    if default_device_is_usable():
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

    layer = _Layer(device = device, index = index)
    resolved, buffer_index = per_layer_device(layer)
    assert isinstance(resolved, torch.device)

    per_device_buffers = tuple(range(8))
    assert per_device_buffers[buffer_index] == buffer_index

    # The move is only performed where it cannot need hardware.
    if resolved.type == "cpu":
        moved = move_to_device(resolved, torch.zeros(2))
        assert moved.device == torch.device("cpu")


def test_cpu_offloaded_layer_no_longer_raises_invalid_target_device():
    """The verbatim #3538 failure, driven through move_to_device."""
    from unsloth.models._utils import move_to_device, per_layer_device

    layer = _Layer(device = torch.device("cpu"), index = "cpu")
    resolved, _ = per_layer_device(layer)
    hidden_states = torch.zeros(1, 2, 4)
    position_ids = torch.zeros(1, 2, dtype = torch.long)
    hidden_states, position_ids = move_to_device(resolved, hidden_states, position_ids)
    assert hidden_states.device == torch.device("cpu")
    assert position_ids.device == torch.device("cpu")

    # And the shape that used to reach move_to_device is still rejected loudly.
    with pytest.raises(ValueError, match = "Invalid target device"):
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
    source = (REPOSITORY_ROOT / path).read_text(encoding = "utf-8")
    assert (
        'getattr(decoder_layer, "_per_layer_device_index"' not in source
    ), f"{path} reads the raw index again; a None there is #3538"
    assert (
        "per_layer_device(decoder_layer)" in source
    ), f"{path} no longer resolves the layer device through per_layer_device"
    assert re.search(
        r"move_to_device\(\s*layer_device", source
    ), f"{path} does not hand the resolved device to move_to_device"


@pytest.mark.parametrize("path", sorted(READERS))
def test_every_reader_imports_the_helper_explicitly(path):
    """Do not rely on `from .llama import *` to carry the name across."""
    source = (REPOSITORY_ROOT / path).read_text(encoding = "utf-8")
    assert re.search(
        r"^from \._utils import .*per_layer_device", source, re.MULTILINE
    ), f"{path} must import per_layer_device from ._utils explicitly"


def _reader_scopes(source: str, path: str):
    """(name, lineno) of the innermost function around each per_layer_device call."""
    import ast

    tree = ast.parse(source)
    functions = [
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) != "per_layer_device":
            continue
        enclosing = [
            function
            for function in functions
            if function.lineno <= node.lineno <= (function.end_lineno or function.lineno)
        ]
        assert enclosing, f"{path}: per_layer_device call at module level?"
        innermost = max(enclosing, key = lambda function: function.lineno)
        found.append((innermost.name, innermost.lineno))
    return found


@pytest.mark.parametrize("path", sorted(READERS))
def test_no_reader_reads_a_name_it_never_binds(path):
    """The reader functions must not reference an unbound local.

    Binding only the device and discarding the index leaves the per-accelerator tuple
    subscripts reading a name that no longer exists, which Python reports only at the first
    decode step. symtable classifies such a name as an implicit global, so checking it
    against the module's real attributes catches it statically, star imports and all.
    """
    import builtins
    import importlib
    import symtable

    module = importlib.import_module("unsloth.models." + pathlib.Path(path).stem)
    source = (REPOSITORY_ROOT / path).read_text(encoding = "utf-8")
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

    assert len(checked) == len(
        wanted
    ), f"{path}: located {wanted} from the call sites but symtable matched {checked}"
    assert not unbound, (
        f"{path}:{checked} reads {sorted(set(unbound))} without binding it and "
        f"without {module.__name__} defining it, so generation raises NameError"
    )


@pytest.mark.parametrize("path", sorted(READERS))
def test_per_accelerator_tuples_are_still_subscripted_by_an_int(path):
    """A reader that uses a per-accelerator tuple must keep the buffer index."""
    source = (REPOSITORY_ROOT / path).read_text(encoding = "utf-8")
    for tuple_name in READERS[path]:
        assert f"{tuple_name}[device_index]" in source, (
            f"{path} subscripts {tuple_name} with something other than the "
            "buffer index per_layer_device returns"
        )
        assert "layer_device, device_index = per_layer_device(decoder_layer)" in source


@pytest.mark.skipif(not has_real_cuda(), reason = "needs a real GPU")
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


def test_the_backend_probe_is_asked_once_per_device_type():
    """The reader reaches `torch.cuda.is_available()` on every layer of every token
    wherever unsloth_zoo publishes only the index, and the answer cannot change inside a
    process, so it is memoised."""
    from unsloth.models import _utils

    assert hasattr(
        _utils._device_type_is_usable, "cache_clear"
    ), "_device_type_is_usable must stay memoised; it is on the per-token path"

    _utils._device_type_is_usable.cache_clear()
    calls = []
    real = torch.cuda.is_available

    def counted():
        calls.append(1)
        return real()

    try:
        torch.cuda.is_available = counted
        first = _utils._device_type_is_usable("cuda")
        for _ in range(50):
            assert _utils._device_type_is_usable("cuda") == first
    finally:
        torch.cuda.is_available = real
        _utils._device_type_is_usable.cache_clear()

    assert len(calls) <= 1, (
        f"the backend was probed {len(calls)} times for one device type; "
        "that is a per-layer per-token cost"
    )


def test_the_published_attributes_are_read_without_a_failed_getattr():
    """`nn.Module.__getattr__` searches _parameters, _buffers and _modules and then
    raises, and an unsloth_zoo publishing only the index leaves `_per_layer_device` missing
    on every layer, so the reader must not discover that through getattr."""
    from unsloth.models._utils import per_layer_device

    class _Counting(_Layer):
        misses = 0

        def __getattr__(self, name):
            if name in ("_per_layer_device", "_per_layer_device_index"):
                type(self).misses += 1
            return super().__getattr__(name)

    # Current unsloth_zoo: both names published.
    layer = _Counting(device = torch.device("cpu"), index = "cpu")
    _Counting.misses = 0
    per_layer_device(layer)
    assert _Counting.misses == 0

    # unsloth_zoo from before the device attribute existed: the index alone.
    layer = _Counting(index = 0)
    _Counting.misses = 0
    per_layer_device(layer)
    assert _Counting.misses == 0, (
        "the missing device attribute must not be discovered through "
        "nn.Module.__getattr__ on every layer of every token"
    )


def test_a_class_level_attribute_is_still_honoured():
    """Reading the instance dictionary must not lose a layer that publishes the placement
    on its class or through a property, which is what the getattr fallback is for."""
    from unsloth.models._utils import per_layer_device

    class _ClassAttribute(_Layer):
        _per_layer_device = torch.device("cpu")
        _per_layer_device_index = "cpu"

    device, buffer_index = per_layer_device(_ClassAttribute())
    assert device == torch.device("cpu")
    assert buffer_index == 0

    class _Property(_Layer):
        @property
        def _per_layer_device_index(self):
            return "cpu"

    device, buffer_index = per_layer_device(_Property())
    assert device == torch.device("cpu")
    assert buffer_index == 0


def test_moving_an_activation_to_meta_destroys_it_silently():
    """`tensor.to("meta")` succeeds and drops the data, and mixing the result with a real
    tensor propagates meta instead of raising, so a decode that resolved a layer to meta
    would run to completion and return nothing. That is why meta is excluded everywhere."""
    activation = torch.ones(2, 4)
    moved = activation.to("meta")
    assert moved.device.type == "meta"
    assert torch.matmul(moved, torch.ones(4, 4)).device.type == "meta"


def test_a_meta_layer_never_resolves_to_meta():
    from unsloth.models._utils import per_layer_device
    for layer in (
        _Layer(parameter_device = "meta", index = None),
        _Layer(device = torch.device("meta"), index = "meta"),
        _Layer(parameter_device = "meta", index = "meta"),
    ):
        device, buffer_index = per_layer_device(layer)
        assert device.type != "meta", device
        assert isinstance(buffer_index, int), buffer_index


def test_a_meta_layer_uses_the_accelerate_hook_execution_device():
    """AlignDevicesHook.pre_forward moves the layer's inputs to `execution_device`."""
    from types import SimpleNamespace

    from unsloth.models._utils import per_layer_device

    layer = _Layer(parameter_device = "meta", index = None)
    layer._hf_hook = SimpleNamespace(execution_device = "cpu")
    device, _buffer_index = per_layer_device(layer)
    assert device == torch.device("cpu")


def test_a_hook_that_itself_says_meta_is_not_believed():
    """accelerate sets execution_device to meta while a model is still being built."""
    from types import SimpleNamespace

    from unsloth.models._utils import per_layer_device

    for execution_device in (None, "meta", torch.device("meta")):
        layer = _Layer(parameter_device = "meta", index = None)
        layer._hf_hook = SimpleNamespace(execution_device = execution_device)
        device, _buffer_index = per_layer_device(layer)
        assert device.type != "meta", (execution_device, device)


def test_a_non_meta_layer_is_unchanged_by_the_meta_guard():
    """The control: nothing above may touch the ordinary resolutions."""
    from unsloth.models._utils import per_layer_device

    device, buffer_index = per_layer_device(_Layer(parameter_device = "cpu", index = None))
    assert device == torch.device("cpu")
    assert buffer_index == 0

    device, buffer_index = per_layer_device(_Layer(device = torch.device("cuda:1"), index = 1))
    assert device == torch.device("cuda:1")
    assert buffer_index == 1
