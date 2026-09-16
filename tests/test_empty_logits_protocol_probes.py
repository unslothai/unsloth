# Unsloth - 2x faster, 60% less VRAM LLM training and finetuning
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

"""unsloth#409: the EMPTY_LOGITS sentinel must not claim protocol dunders.

``EmptyLogits.__getattr__`` answers every name, so ``hasattr`` on the sentinel
used to be true for every name too. Libraries duck-type on dunders:
``torch.distributed.utils._apply_to_tensors`` tests
``hasattr(x, "__dataclass_fields__")`` and then calls ``dataclasses.replace(x)``
on whatever said yes, so FSDP2's mixed-precision output cast
(``_fsdp_state._cast_output_dtype``) killed every training step with
``TypeError: replace() should be called on dataclass instances``.

Measured on 2 GPUs with ``accelerate launch --config_file <fsdp2>``: before, all
four steps of a Qwen2.5-0.5B LoRA SFT died in that TypeError; after, the same
run logged ``[3.5781, 3.9453, 3.5156, 3.4297]`` on both ranks.

No GPU and no distributed launcher needed: the probe below is exactly the one
torch makes.
"""

from __future__ import annotations

import dataclasses

import pytest

from unsloth.models._utils import EMPTY_LOGITS, LOGITS_ERROR_STRING


# Dunders that libraries duck-type on when they walk a model output and that the
# sentinel has no business claiming. Deliberately none of `dir(torch.Tensor)`:
# the module-level loop under EMPTY_LOGITS binds every tensor dunder as a real
# instance attribute, so those are answered by the instance dict and never reach
# `__getattr__`. These are the ones only `__getattr__` could have invented.
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
        # Reached only when the hasattr above is true. Asserting it still raises
        # documents why the hasattr has to be false rather than why replace works.
        dataclasses.replace(EMPTY_LOGITS)


def test_an_ordinary_attribute_still_explains_how_to_get_real_logits():
    """The AttributeError is for protocol probes only. A user reaching for a
    tensor attribute must still be told about UNSLOTH_RETURN_LOGITS, or the fix
    trades a TypeError for a silent `AttributeError: shape`."""
    raiser = EMPTY_LOGITS.shape
    with pytest.raises(NotImplementedError) as excinfo:
        raiser()
    assert "UNSLOTH_RETURN_LOGITS" in str(excinfo.value)
    assert LOGITS_ERROR_STRING in str(excinfo.value)


def test_to_is_still_the_no_op_accelerate_needs():
    """`.to(device)` is called on model outputs by accelerate; it must stay a
    no-op rather than raise, which is why it is special-cased ahead of both the
    dunder check and the raiser."""
    assert EMPTY_LOGITS.to("cpu") is None


def test_dunder_names_the_class_really_defines_are_untouched():
    """The guard is on `__getattr__`, which Python only consults after the normal
    lookup fails, so anything the class defines is reached as before."""
    assert EMPTY_LOGITS == EMPTY_LOGITS
    assert repr(EMPTY_LOGITS) == LOGITS_ERROR_STRING
    assert EMPTY_LOGITS.__reduce__() == (type(EMPTY_LOGITS), ())
