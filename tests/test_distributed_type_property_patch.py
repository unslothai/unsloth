# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""On a single-GPU / single-process host, `unsloth/models/_utils.py` patches
`Accelerator.distributed_type` to always report `DistributedType.NO`, so accelerate's
"you can't train a device_map='auto' model in distributed mode" guards don't fire when
there is nothing distributed going on.

A bare lambda assigned to a class attribute is a function, and functions are descriptors:
reading it off an instance (`self.distributed_type`) binds it as a method instead of
calling it. Every read then returns a bound-method object, which is never equal to
`DistributedType.NO`, so the guards fire unconditionally as soon as a model legitimately
spans devices (e.g. a 4-bit checkpoint with part of it offloaded to CPU) -- even on a
single GPU, single process host. `Accelerator.distributed_type` is a `@property` upstream,
so the patch has to be one too.
"""

import re

_PATCH_LINE = re.compile(r"accelerate\.accelerator\.Accelerator\.distributed_type\s*=\s*(.+)")


def _find_patch_expression():
    """The exact right-hand side unsloth assigns to Accelerator.distributed_type,
    read from source rather than duplicated here, so this test fails if the patch
    is edited back to a bare lambda instead of testing a copy that could drift."""
    import pathlib

    utils_path = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_utils.py"
    text = utils_path.read_text(encoding = "utf-8")
    match = _PATCH_LINE.search(text)
    assert match is not None, "could not find the Accelerator.distributed_type patch in _utils.py"
    return match.group(1).strip()


def test_patch_is_a_property_not_a_bare_lambda():
    expression = _find_patch_expression()
    assert expression.startswith("property("), (
        "Accelerator.distributed_type is patched with a bare function again, which "
        f"accelerate binds as a method rather than calling: {expression!r}. Wrap it in "
        "property(...) so instance access returns the value, not a bound method."
    )


def test_patched_attribute_reads_as_distributed_type_no_not_a_bound_method():
    accelerate = __import__("accelerate")
    from accelerate.utils.dataclasses import DistributedType

    original = accelerate.accelerator.Accelerator.distributed_type
    try:
        expression = _find_patch_expression()
        accelerate.accelerator.Accelerator.distributed_type = eval(
            expression, {"DistributedType": DistributedType}
        )
        instance = accelerate.Accelerator()
        value = instance.distributed_type
        assert value == DistributedType.NO, (
            f"self.distributed_type read back {value!r} instead of DistributedType.NO; "
            "accelerate's distributed-mode guards (`if self.distributed_type != "
            "DistributedType.NO: raise ...`) would fire even on a single GPU / single "
            "process host."
        )
    finally:
        accelerate.accelerator.Accelerator.distributed_type = original
