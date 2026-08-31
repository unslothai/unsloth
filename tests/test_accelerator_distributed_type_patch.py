# SPDX-License-Identifier: Apache-2.0
"""#10028: Accelerator.distributed_type must be a property, not a bare function.

`accelerate.accelerator.Accelerator.distributed_type` is a @property upstream.
Assigning a bare lambda creates a bound method, so every
`self.distributed_type != DistributedType.NO` guard flips to True on a
single-device setup — the exact case the module-level patch exists to rule out.

Wrapping in `property()` replaces the eager descriptor so an instance reads
`DistributedType.NO` instead of a bound method.
"""

import pytest

accelerate = pytest.importorskip("accelerate")
from accelerate.utils.dataclasses import DistributedType
from accelerate.accelerator import Accelerator

# The _utils module patches this at import time when DEVICE_COUNT == 1.
# We import it here to trigger the code path, then inspect the result.
from unsloth.models import _utils


def test_distributed_type_is_not_a_bound_method():
    """After patching, `accelerator.distributed_type` returns a value, not a method."""
    acc = Accelerator()
    dt = acc.distributed_type
    assert not callable(dt), (
        f"distributed_type is a bound method ({type(dt).__name__}), "
        "not an enum value — the lambda was not wrapped in property()"
    )


def test_distributed_type_is_no_on_single_device():
    """On a single device with DEVICE_COUNT == 1, distributed_type must be NO."""
    acc = Accelerator()
    assert acc.distributed_type == DistributedType.NO, (
        f"expected DistributedType.NO, got {acc.distributed_type}"
    )


def test_class_attribute_is_a_property():
    """Verifying class attribute: the descriptor on the class is a property."""
    desc = Accelerator.__dict__["distributed_type"]
    assert isinstance(desc, property), (
        f"Class-level `distributed_type` is a {type(desc).__name__}, not a property — "
        "the patch did not take effect"
    )
