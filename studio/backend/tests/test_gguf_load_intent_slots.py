# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GgufLoadIntent must not use slots=True: it removes the instance dict that vars() reads."""

import copy
import dataclasses
import pickle
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.inference.llama_cpp import (  # noqa: E402
    CountAborted,
    GgufLoadIntent,
    LlamaServerNotFoundError,
    _LlamaStreamCancelled,
)


@pytest.fixture
def intent():
    # __post_init__ freezes lists to tuples so the intent stays hashable.
    return GgufLoadIntent(
        model_identifier = "unsloth/gemma-4-E2B-it-GGUF",
        gpu_ids = [0, 1],
        extra_args = ["--flash-attn"],
        tensor_split = [0.5, 0.5],
    )


def test_intent_keeps_its_instance_dict(intent):
    """The revert, pinned: several callers enumerate the intent through ``vars()``."""
    assert not hasattr(GgufLoadIntent, "__slots__")
    assert set(vars(intent)) == {f.name for f in dataclasses.fields(intent)}


def test_post_init_freezes_sequences(intent):
    assert intent.gpu_ids == (0, 1)
    assert intent.extra_args == ("--flash-attn",)
    assert intent.tensor_split == (0.5, 0.5)


def test_empty_gpu_ids_normalizes_to_none():
    assert GgufLoadIntent(model_identifier = "m", gpu_ids = []).gpu_ids is None


def test_replace_preserves_the_rest(intent):
    """Retries rebuild the intent with one field changed."""
    replaced = dataclasses.replace(intent, n_ctx = 8192)

    assert replaced.n_ctx == 8192
    assert replaced.model_identifier == intent.model_identifier
    assert replaced.gpu_ids == intent.gpu_ids


def test_equality_ignores_whether_sequences_arrived_as_lists(intent):
    assert intent == GgufLoadIntent(
        model_identifier = "unsloth/gemma-4-E2B-it-GGUF",
        gpu_ids = (0, 1),
        extra_args = ("--flash-attn",),
        tensor_split = (0.5, 0.5),
    )


def test_copy_and_pickle_round_trip(intent):
    """``slots=True`` changes the pickle protocol used, so both are exercised."""
    assert copy.deepcopy(intent) == intent
    assert pickle.loads(pickle.dumps(intent)) == intent


def test_still_frozen(intent):
    with pytest.raises(dataclasses.FrozenInstanceError):
        intent.n_ctx = 1


def test_fields_and_asdict_are_unaffected(intent):
    names = [f.name for f in dataclasses.fields(intent)]

    assert "model_identifier" in names
    assert dataclasses.asdict(intent)["gpu_ids"] == (0, 1)


@pytest.mark.parametrize(
    "exception", [CountAborted, LlamaServerNotFoundError, _LlamaStreamCancelled]
)
def test_signal_exceptions_carry_empty_slots(exception):
    """Signal exceptions declare empty __slots__ only to record they carry no slot of their own."""
    assert exception.__slots__ == ()
    assert "__dict__" in vars(BaseException)

    with pytest.raises(exception, match = "boom"):
        raise exception("boom")


def test_llama_server_not_found_is_still_a_runtime_error():
    """Existing handlers catch RuntimeError; adding slots must not change the MRO."""
    assert issubclass(LlamaServerNotFoundError, RuntimeError)


def test_field_names_are_enumerable_without_reflection(intent):
    """dataclasses.fields enumerates intent fields without relying on the instance __dict__."""
    assert [f.name for f in dataclasses.fields(intent)] == list(vars(intent))
