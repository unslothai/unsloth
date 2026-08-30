# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A dispatched module must carry a hook, including the tied ones.

`accelerate.dispatch_model` moves a TIED weight but installs no
`AlignDevicesHook` on its module, because a tied parameter is reachable under
more than one name. On a `tie_word_embeddings` model the unsloth planner's map
puts `model.embed_tokens` and `lm_head` on the second card, so the two modules
it deliberately moved are exactly the two left with nothing to move their
inputs after them.

Measured on 2x Tesla T4, `unsloth/Qwen3-0.6B` in 4bit, map placing
`embed_tokens` and `lm_head` on cuda:1 and the 30 layers on cuda:0:

    modules_with_hf_hook   395
    embedding_has_hook     False
    train                  RuntimeError ... index is on cuda:0, different from
                           other tensors on cuda:1 ... wrapper_CUDA__index_select

while the same model at `device_map = {"": 0}` trained to loss 2.662875493367513.

These RUN `_repair_tied_module_hooks` against stub modules. A rule fed a
hand-written dict would pass on a function that repairs nothing, which is the
failure this file exists to prevent.
"""

import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")

from accelerate.hooks import AlignDevicesHook, add_hook_to_module  # noqa: E402


def _repair():
    """The real function, imported without dragging in unsloth's CUDA stack."""
    from unsloth.models.vision import _repair_tied_module_hooks
    return _repair_tied_module_hooks


# `init_hook` MOVES the module's tensors to the execution device, so naming a
# CUDA device needs that card to exist -- and the CPU runner this suite also
# runs on has none. "meta" is a real device everywhere and accelerate handles it
# explicitly, so the attach is exercised for real with no GPU at all; "cpu"
# stands in for a module the repair must leave alone. The integer -> "cuda:N"
# mapping is checked separately, without attaching anything.
FAR = "meta"
NEAR = "cpu"


class _Model(torch.nn.Module):
    """A stand-in with the shape that matters: named submodules and a map."""

    def __init__(
        self,
        device_map,
        hooked = (),
    ):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(4, 2)
        self.lm_head = torch.nn.Linear(2, 4)
        self.layer = torch.nn.Linear(2, 2)
        self.hf_device_map = dict(device_map)
        for name in hooked:
            add_hook_to_module(self.get_submodule(name), AlignDevicesHook())


def _hooked(model):
    return {n for n, m in model.named_modules() if hasattr(m, "_hf_hook")}


def test_a_tied_module_the_map_placed_gets_its_hook_back():
    """The measured failure: in the map, on the far card, no hook."""
    model = _Model({"embed_tokens": FAR, "lm_head": FAR, "layer": NEAR})
    assert not hasattr(model.embed_tokens, "_hf_hook"), "fixture is not the broken state"

    repaired = _repair()(model)

    assert repaired == 2, (
        "the two modules the map put on the far card were not repaired, so the "
        "first embedding lookup still crosses devices unaided"
    )
    assert {"embed_tokens", "lm_head"} <= _hooked(model)


def test_a_bare_integer_names_the_card_the_map_meant(monkeypatch):
    """`hf_device_map` gives CUDA entries as bare ints.

    torch reads an int as a device index only when paired with a type, so an
    unconverted `1` would send the ids somewhere the map never named. Captured
    rather than attached, since attaching would need that second card present.
    """
    import accelerate.hooks as ah

    seen = {}

    def capture(module, hook, **kwargs):
        seen[id(module)] = hook
        return module

    monkeypatch.setattr(ah, "add_hook_to_module", capture)
    model = _Model({"embed_tokens": 1, "layer": 0})
    assert _repair()(model) == 2

    hook = seen[id(model.embed_tokens)]
    assert str(hook.execution_device) == "cuda:1", (
        f"the ids would be sent to {hook.execution_device!r}, not the card the "
        "map put the weight on"
    )
    assert hook.io_same_device is True, (
        "without io_same_device the output is left on the far card and the "
        "mismatch simply moves one operation downstream"
    )


def test_a_failed_attach_is_reported_not_swallowed():
    """A repair that counts a module it never hooked is worse than none.

    The original crash stands and its reason is hidden. `init_hook` moves the
    module's tensors, so a map naming a device this machine lacks lands here.
    """
    model = _Model({"embed_tokens": "cuda:99", "layer": NEAR})
    with pytest.warns(RuntimeWarning, match = "could not re-attach"):
        repaired = _repair()(model)
    assert repaired == 0, "the unattachable module was still counted as repaired"
    assert "embed_tokens" not in _hooked(model)


def test_a_module_that_already_has_a_hook_is_left_alone():
    """Double-hooking a module moves its inputs twice per forward."""
    model = _Model({"embed_tokens": FAR, "layer": NEAR}, hooked = ["embed_tokens"])
    original = model.embed_tokens._hf_hook

    assert _repair()(model) == 0, "nothing else here is repairable"
    assert model.embed_tokens._hf_hook is original, (
        "an already-dispatched module was hooked again, so its inputs move "
        "twice on every forward"
    )


def test_a_single_device_map_is_left_completely_alone():
    """The overwhelmingly common case must cost nothing and change nothing."""
    # Entries that WOULD attach, so dropping the early return changes the
    # count. A map of unattachable devices passes either way and proves nothing.
    model = _Model({"embed_tokens": FAR, "lm_head": FAR, "layer": FAR})
    assert _repair()(model) == 0
    assert _hooked(model) == set(), "hooks were attached on a single-device load"


def test_no_map_at_all_is_not_an_error():
    """`device_map = {"": 0}` leaves `hf_device_map` None, and that path trains."""
    model = _Model({})
    model.hf_device_map = None
    assert _repair()(model) == 0


def test_a_cpu_or_disk_entry_is_never_hooked_here():
    """Offload is a different mechanism with hooks of its own."""
    model = _Model({"embed_tokens": "cpu", "lm_head": "disk", "layer": FAR})
    assert _repair()(model) == 1, "only `layer` is repairable here"
    assert "embed_tokens" not in _hooked(model) and "lm_head" not in _hooked(model), (
        "an offloaded module was given a dispatch hook, which fights the "
        "offload path's own pre-hook"
    )


def test_a_name_the_model_does_not_have_is_skipped_not_invented():
    """A map naming a module this build lacks must not raise during a load."""
    model = _Model({"embed_tokens": FAR, "does.not.exist": FAR, "layer": NEAR})
    assert _repair()(model) == 1, "the one real far-device name was not repaired"
    assert "embed_tokens" in _hooked(model)


def test_the_repair_stands_aside_for_an_offloaded_embedding():
    """`_embedding_dispatch_device` READS this hook to place the ids.

    With `offload_embedding` the weight is on the CPU and the offload pre-hook
    has already sent the ids there; a hook naming the card the map wanted would
    answer that question with the wrong device. Read off the caller, since the
    decision lives there rather than in the repair.
    """
    import ast
    import inspect

    from unsloth.models import vision

    src = inspect.getsource(vision._attach_bnb_multidevice_hooks)
    tree = ast.parse(src.lstrip())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "_repair_tied_module_hooks"
    ]
    assert calls, "the loader no longer repairs tied hooks at all"

    guarded = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and getattr(node.test.operand, "id", None) == "offload_embedding"
        and any(
            isinstance(c, ast.Call) and getattr(c.func, "id", None) == "_repair_tied_module_hooks"
            for c in ast.walk(node)
        )
    ]
    assert guarded, (
        "the repair is no longer behind `if not offload_embedding`, so it "
        "attaches a hook naming a card while the offload path sends the ids "
        "to the CPU weight"
    )
