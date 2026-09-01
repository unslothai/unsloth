# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The GRPO hidden-states wrapper must not pay for logits it throws away.

`Kaggle-Muse_Glimmer_(30B)-GRPO` died on a 2 x T4 kernel with

    accelerate/hooks.py:429  AlignDevicesHook.post_forward
      -> send_to_device(output, self.input_device)
    OutOfMemoryError: Tried to allocate 1002.00 MiB.
    GPU 0 has 14.56 GiB capacity, of which 768.81 MiB is free.

Two costs met there.

The first is the lm_head. `UnslothEfficientGRPO` never sees a logits tensor --
it takes per-token logps plus `lm_head` and chunks the projection itself -- and
this wrapper exists to hand it hidden states instead of logits. But it forwarded
the caller's `logits_to_keep` unchanged, and the GRPO trainer does not pass one:

    outputs = unwrapped_model(
        input_ids = input_ids_chunk,
        attention_mask = attention_mask_chunk,
        ...
    )
    logits_chunk = outputs.logits

transformers reads a missing or zero value as `slice(-0, None)`, which is
`slice(0, None)`: every position. So the model projected the whole prompt and
completion over a 202048-wide vocabulary, softcapped it twice, and the wrapper
then overwrote the result with hidden states.

The second is the other layers. `output_hidden_states = True` returns every
layer; only `[-1]` is read, and the rest stayed attached to the output.

On one card neither cost is visible: the trainer's `del outputs` frees both a
line later. Under an accelerate layer-split dispatch, `io_same_device` walks the
whole returned object and copies every tensor in it to the input device first,
so both ride across the bus and the first card runs out.

These tests drive the real helpers, lifted from `unsloth/models/rl.py` with
`ast` so they track the shipped source without importing unsloth.
"""

from __future__ import annotations

import collections
import inspect
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _rl_source import load_rl_wrapper  # noqa: E402

WRAPPER = load_rl_wrapper()
_minimise_logits_kwarg = WRAPPER["_minimise_logits_kwarg"]
_drop_spare_hidden_states = WRAPPER["_drop_spare_hidden_states"]
_install = WRAPPER["_install_grpo_hidden_states_forward_wrapper"]


# a stand-in for transformers' ModelOutput, including the trap
# --------------------------------------------------------------------------
class FakeModelOutput(collections.OrderedDict):
    """`ModelOutput`'s actual assignment semantics, which are the whole point.

        def __setattr__(self, name, value):
            if name in field_names and value is not None:
                super().__setitem__(name, value)
            super().__setattr__(name, value)

    so assigning None updates the attribute and leaves the mapping entry alone,
    and `__delitem__` / `pop` / `update` / `setdefault` all raise.
    """

    _fields = ("logits", "hidden_states")

    def __init__(self, **kwargs):
        super().__init__()
        for name in self._fields:
            value = kwargs.get(name)
            object.__setattr__(self, name, value)
            if value is not None:
                collections.OrderedDict.__setitem__(self, name, value)

    def __setattr__(self, name, value):
        if name in self._fields and value is not None:
            collections.OrderedDict.__setitem__(self, name, value)
        object.__setattr__(self, name, value)

    def __delitem__(self, *a, **kw):
        raise Exception("You cannot use ``__delitem__`` on a FakeModelOutput instance.")

    def pop(self, *a, **kw):
        raise Exception("You cannot use ``pop`` on a FakeModelOutput instance.")

    def update(self, *a, **kw):
        raise Exception("You cannot use ``update`` on a FakeModelOutput instance.")

    def setdefault(self, *a, **kw):
        raise Exception("You cannot use ``setdefault`` on a FakeModelOutput instance.")


def test_the_fake_output_really_does_reproduce_the_trap():
    """If this stops holding, the test below proves nothing."""
    out = FakeModelOutput(logits = "L", hidden_states = ("a", "b", "c"))
    out.hidden_states = None
    assert out.hidden_states is None, "the attribute should have taken the None"
    assert out["hidden_states"] == (
        "a",
        "b",
        "c",
    ), "the mapping entry must survive -- that is the bug being guarded"
    with pytest.raises(Exception):
        out.pop("hidden_states")


# --------------------------------------------------------------------------
def test_the_spare_layers_leave_the_mapping_not_just_the_attribute():
    """A consumer that walks the object as a mapping is the one that matters."""
    out = FakeModelOutput(logits = "L", hidden_states = ("a", "b", "c"))
    _drop_spare_hidden_states(out)
    assert out.hidden_states is None
    assert out["hidden_states"] is None, (
        "accelerate's send_to_device iterates .items(); a stale tuple here is "
        "copied to the other device"
    )
    assert dict(out.items())["hidden_states"] is None


def test_a_plain_dict_output_is_handled_too():
    out = {"logits": "L", "hidden_states": ("a", "b")}
    _drop_spare_hidden_states(out)
    assert out["hidden_states"] is None


def test_a_plain_object_output_is_handled_too():
    @dataclass
    class Out:
        logits: str
        hidden_states: tuple

    out = Out(logits = "L", hidden_states = ("a", "b"))
    _drop_spare_hidden_states(out)
    assert out.hidden_states is None


def test_an_output_without_hidden_states_is_left_alone():
    out = {"logits": "L"}
    _drop_spare_hidden_states(out)
    assert out == {"logits": "L"}


def test_an_unassignable_output_does_not_take_the_step_down():
    """The caller already has the layer it needs; this is not worth raising."""

    class Frozen:
        __slots__ = ()

        @property
        def hidden_states(self):
            return ("a",)

    _drop_spare_hidden_states(Frozen())  # must not raise


# not paying for the lm_head
# --------------------------------------------------------------------------
def _sig(fn):
    import inspect
    return inspect.signature(fn)


def test_the_modern_kwarg_is_pinned_to_one():
    def forward(
        input_ids = None,
        logits_to_keep = 0,
        **kwargs,
    ): ...

    kwargs = {"input_ids": "x"}
    name = _minimise_logits_kwarg(_sig(forward), (), kwargs)
    assert name == "logits_to_keep"
    assert (
        kwargs["logits_to_keep"] == 1
    ), "0 means every position, which is the whole cost being avoided"


def test_the_legacy_kwarg_is_used_when_that_is_what_the_model_takes():
    """Not transformers -- measured, 4.57.6 through 5.15.0 all take the modern
    name and none takes this one. It is Unsloth's own patched forwards
    (`models/llama.py`, `models/mistral.py`) and the VLM stacks
    `models/vision.py` probes the old name for.
    """

    def forward(input_ids = None, num_logits_to_keep = 0): ...

    kwargs = {"input_ids": "x"}
    name = _minimise_logits_kwarg(_sig(forward), (), kwargs)
    assert name == "num_logits_to_keep"
    assert kwargs["num_logits_to_keep"] == 1
    assert "logits_to_keep" not in kwargs, "must not send a kwarg this forward rejects"


def test_a_positionally_bound_width_is_not_worked_around_via_the_other_name():
    """The dangerous shape: positional modern name, plus a `**kwargs` sink.

    Falling through to the legacy name here would set a kwarg this forward does
    not declare. `**kwargs` accepts it silently, the model ignores it, no logits
    are saved, and the non-None return arms the absent-hidden-states re-run --
    a second full forward bought for nothing. Give up instead.
    """

    def forward(
        input_ids = None,
        logits_to_keep = 0,
        **kwargs,
    ): ...

    kwargs = {}
    name = _minimise_logits_kwarg(_sig(forward), ("x", 512), kwargs)
    assert name is None
    assert kwargs == {}, "the caller's positional width must stand untouched"


def test_the_modern_name_wins_when_a_forward_takes_both():
    def forward(
        input_ids = None,
        logits_to_keep = 0,
        num_logits_to_keep = 0,
    ): ...

    kwargs = {"input_ids": "x"}
    assert _minimise_logits_kwarg(_sig(forward), (), kwargs) == "logits_to_keep"
    assert "num_logits_to_keep" not in kwargs


def test_a_caller_supplied_value_is_overridden_because_we_discard_the_logits():
    def forward(input_ids = None, logits_to_keep = 0): ...

    kwargs = {"input_ids": "x", "logits_to_keep": 512}
    _minimise_logits_kwarg(_sig(forward), (), kwargs)
    assert kwargs["logits_to_keep"] == 1


def test_a_positional_value_is_left_alone():
    """Passing it positionally and again by keyword is a TypeError."""

    def forward(input_ids = None, logits_to_keep = 0): ...

    kwargs = {}
    name = _minimise_logits_kwarg(_sig(forward), ("x", 512), kwargs)
    assert name is None
    assert kwargs == {}


def test_a_forward_that_cannot_take_it_is_left_alone():
    def forward(input_ids = None, attention_mask = None): ...

    kwargs = {"input_ids": "x"}
    assert _minimise_logits_kwarg(_sig(forward), (), kwargs) is None
    assert kwargs == {"input_ids": "x"}


def test_var_keyword_counts_as_accepting_it():
    def forward(input_ids = None, **kwargs): ...

    kwargs = {"input_ids": "x"}
    assert _minimise_logits_kwarg(_sig(forward), (), kwargs) == "logits_to_keep"


def test_a_forward_given_labels_keeps_its_logits():
    """A model that computes its own loss needs the real thing."""

    def forward(
        input_ids = None,
        labels = None,
        logits_to_keep = 0,
    ): ...

    kwargs = {"input_ids": "x", "labels": "y"}
    assert _minimise_logits_kwarg(_sig(forward), (), kwargs) is None
    assert "logits_to_keep" not in kwargs


def test_labels_of_none_does_not_count_as_labels():
    def forward(
        input_ids = None,
        labels = None,
        logits_to_keep = 0,
    ): ...

    kwargs = {"input_ids": "x", "labels": None}
    assert _minimise_logits_kwarg(_sig(forward), (), kwargs) == "logits_to_keep"


# --------------------------------------------------------------------------
class _Recorder:
    """A model whose lm_head cost is proportional to the positions it is asked for."""

    def __init__(
        self,
        n_layers = 4,
        seq = 8,
        accepts = "logits_to_keep",
    ):
        self.n_layers, self.seq, self.accepts = n_layers, seq, accepts
        self.projected_positions = None
        self.calls = 0
        if accepts == "logits_to_keep":

            def forward(
                input_ids = None,
                logits_to_keep = 0,
                output_hidden_states = False,
                return_dict = False,
            ):
                return self._run(logits_to_keep, output_hidden_states)
        elif accepts == "num_logits_to_keep":

            def forward(
                input_ids = None,
                num_logits_to_keep = 0,
                output_hidden_states = False,
                return_dict = False,
            ):
                return self._run(num_logits_to_keep, output_hidden_states)
        else:

            def forward(
                input_ids = None,
                output_hidden_states = False,
                return_dict = False,
            ):
                return self._run(0, output_hidden_states)

        self.forward = forward

    def _run(self, keep, output_hidden_states):
        self.calls += 1
        kept = self.seq if not keep else keep
        self.projected_positions = kept
        return FakeModelOutput(
            logits = [["logit"] * 202048] * kept,
            hidden_states = tuple(f"layer{i}" for i in range(self.n_layers + 1))
            if output_hidden_states
            else None,
        )


@pytest.fixture
def hidden_states_on(monkeypatch):
    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "1")


@pytest.mark.parametrize("accepts", ["logits_to_keep", "num_logits_to_keep"])
def test_the_wrapper_asks_for_one_position_not_the_whole_sequence(hidden_states_on, accepts):
    model = _Recorder(accepts = accepts)
    assert _install(model) is not False or True
    model.forward(input_ids = "x")
    assert model.projected_positions == 1, (
        f"projected {model.projected_positions} of {model.seq} positions over the "
        f"full vocabulary, then discarded them"
    )


def test_the_wrapper_returns_the_last_layer_as_logits(hidden_states_on):
    model = _Recorder()
    _install(model)
    out = model.forward(input_ids = "x")
    assert out.logits == "layer4"


def test_the_wrapper_leaves_no_spare_layers_on_the_output(hidden_states_on):
    model = _Recorder()
    _install(model)
    out = model.forward(input_ids = "x")
    assert (
        out["hidden_states"] is None
    ), "5 layer tensors would each be copied to the input device by accelerate"


def test_a_model_that_takes_no_such_kwarg_still_gets_hidden_states(hidden_states_on):
    model = _Recorder(accepts = None)
    _install(model)
    out = model.forward(input_ids = "x")
    assert out.logits == "layer4"
    assert out["hidden_states"] is None


def test_nothing_changes_when_hidden_states_were_not_asked_for(monkeypatch):
    monkeypatch.delenv("UNSLOTH_RETURN_HIDDEN_STATES", raising = False)
    model = _Recorder()
    _install(model)
    out = model.forward(input_ids = "x")
    assert (
        model.projected_positions == model.seq
    ), "outside the hidden-states path this must be the untouched forward"
    assert out.logits != "layer4"


def test_a_forward_that_rejects_the_value_is_retried_without_it(hidden_states_on):
    """Signature says yes, forward says no. Losing hidden states over an
    optimisation would be a worse outcome than not optimising."""
    seen = []

    class Fussy:
        def forward(
            self,
            input_ids = None,
            logits_to_keep = 0,
            output_hidden_states = False,
            return_dict = False,
        ):
            seen.append(logits_to_keep)
            if logits_to_keep == 1:
                raise TypeError("logits_to_keep must be 0 for this model")
            return FakeModelOutput(
                logits = "raw",
                hidden_states = ("a", "b") if output_hidden_states else None,
            )

    model = Fussy()
    _install(model)
    out = model.forward(input_ids = "x")
    assert seen == [1, 0], f"expected a retry without the kwarg, got {seen}"
    assert out.logits == "b"


def test_an_unrelated_type_error_still_propagates(hidden_states_on):
    class Broken:
        def forward(
            self,
            input_ids = None,
            logits_to_keep = 0,
            output_hidden_states = False,
            return_dict = False,
        ):
            raise TypeError("something else entirely")

    model = Broken()
    _install(model)
    with pytest.raises(TypeError, match = "something else entirely"):
        model.forward(input_ids = "x")


def test_a_retry_that_is_then_refused_hidden_states_still_falls_back(hidden_states_on):
    """A forward can refuse the logits limiter and the hidden states one after
    the other -- a wrapper that splats **kwargs into a sub-module does exactly
    that. The second refusal must reach the same raw-logits fallback, not
    escape the wrapper and take the GRPO step down with it."""
    seen = []

    class Splatter:
        def forward(
            self,
            input_ids = None,
            logits_to_keep = 0,
            **kwargs,
        ):
            seen.append((logits_to_keep, kwargs.get("output_hidden_states", False)))
            if logits_to_keep:
                raise TypeError("sub_forward() got an unexpected keyword argument 'logits_to_keep'")
            if kwargs.get("output_hidden_states"):
                raise TypeError(
                    "sub_forward() got an unexpected keyword argument 'output_hidden_states'"
                )
            return FakeModelOutput(logits = "raw", hidden_states = None)

    model = Splatter()
    _install(model)
    out = model.forward(input_ids = "x")
    assert seen == [(1, True), (0, True), (0, False)], seen
    assert out.logits == "raw"


def test_a_forward_that_ignores_output_hidden_states_gets_its_logits_back(hidden_states_on):
    """`logits_to_keep = 1` is only safe because the logits get overwritten. If
    no hidden states come back they ARE the return value, and one position is
    not the completion window GRPO asked for."""
    seen = []

    class Deaf:
        def forward(
            self,
            input_ids = None,
            logits_to_keep = 0,
            output_hidden_states = False,
            return_dict = False,
        ):
            seen.append(logits_to_keep)
            kept = 4 if not logits_to_keep else logits_to_keep
            return FakeModelOutput(logits = ["logit"] * kept, hidden_states = None)

    model = Deaf()
    _install(model)
    out = model.forward(input_ids = "x", logits_to_keep = 4)
    assert seen == [1, 4], f"expected a re-run on the caller's own value, got {seen}"
    assert len(out.logits) == 4, "GRPO slices the completion window out of these"


def test_labels_passed_positionally_still_keep_their_logits():
    """The keyword lookup misses a positional `labels`, but the model's own loss
    needs every position just the same."""

    def forward(
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        logits_to_keep = 0,
        output_hidden_states = False,
        return_dict = True,
    ):
        pass

    signature = inspect.signature(forward)
    forward_kwargs = {"output_hidden_states": True, "return_dict": True}
    used = _minimise_logits_kwarg(
        signature,
        ("ids", None, None, None, None, "labels"),
        forward_kwargs,
    )
    assert used is None, "a one-position logits tensor cannot serve a labels loss"
    assert "logits_to_keep" not in forward_kwargs
