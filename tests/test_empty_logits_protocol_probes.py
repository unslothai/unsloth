# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""unsloth#409: EMPTY_LOGITS must not answer hasattr for protocol dunders it cannot honour."""

from __future__ import annotations

import dataclasses

import pytest

from unsloth.models._utils import EMPTY_LOGITS, LOGITS_ERROR_STRING


# No tensor dunders: the loop under EMPTY_LOGITS binds those as real instance
# attributes, so only these ever reach `__getattr__`.
PROTOCOL_DUNDERS = (
    "__dataclass_fields__",
    "__fields__",
    "__attrs_attrs__",
    "__get_validators__",
    "__pydantic_fields__",
    "__dataclass_params__",
)


@pytest.mark.parametrize("name", PROTOCOL_DUNDERS)
def test_the_sentinel_does_not_claim_a_protocol_dunder(name):
    assert not hasattr(EMPTY_LOGITS, name), (
        f"EMPTY_LOGITS answers hasattr({name!r}); a library that duck-types on it "
        f"will take a branch the sentinel cannot honour"
    )


def test_the_torch_distributed_output_walk_leaves_the_sentinel_alone():
    """The exact probe `_apply_to_tensors` makes, and the call it makes next."""
    assert not hasattr(EMPTY_LOGITS, "__dataclass_fields__")
    with pytest.raises(TypeError):
        dataclasses.replace(EMPTY_LOGITS)


def test_an_ordinary_attribute_still_explains_how_to_get_real_logits():
    """Dunders only: a tensor attribute must still name UNSLOTH_RETURN_LOGITS, not `AttributeError: shape`."""
    raiser = EMPTY_LOGITS.shape
    with pytest.raises(NotImplementedError) as excinfo:
        raiser()
    assert "UNSLOTH_RETURN_LOGITS" in str(excinfo.value)
    assert LOGITS_ERROR_STRING in str(excinfo.value)


def test_to_is_still_the_no_op_accelerate_needs():
    """accelerate calls `.to(device)` on model outputs, so it is special-cased ahead of the dunder check."""
    assert EMPTY_LOGITS.to("cpu") is None


def test_dunder_names_the_class_really_defines_are_untouched():
    """`__getattr__` runs only after normal lookup fails, so what the class defines is reached as before."""
    assert EMPTY_LOGITS == EMPTY_LOGITS
    assert repr(EMPTY_LOGITS) == LOGITS_ERROR_STRING
    assert EMPTY_LOGITS.__reduce__() == (type(EMPTY_LOGITS), ())
