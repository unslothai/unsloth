# SPDX-License-Identifier: AGPL-3.0-only
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

"""`UNSLOTH_FORCE_CUSTOM_DTYPE` carries executable code in two of its five fields.

Every value unsloth ships is a literal in `models/loader.py` and both readers run in
the process that set it, so a value this process did not set has no legitimate source
and its code fields are dropped rather than executed. The wire format is unchanged,
since `unsloth_zoo` parses the same five fields on its own release schedule.
"""

__all__ = [
    "register_custom_dtype",
    "trusted_custom_dtype",
    "neutralize_inherited_custom_dtype",
    "resolve_dtype",
    "DTYPE_ALIASES",
]

import os

import torch

# A table instead of eval, so the field NAMES a dtype rather than being an arbitrary expression.
DTYPE_ALIASES = {
    "None": None,
    "none": None,
    "": None,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

# unsloth_zoo==2026.8.15, which this package's floor resolves to, still evals the dtype field, and
# eval("fp16") is a NameError, so canonicalise to the one spelling both readers evaluate.
_CANONICAL_DTYPE_NAMES = {
    None: "None",
    torch.float16: "torch.float16",
    torch.bfloat16: "torch.bfloat16",
    torch.float32: "torch.float32",
    torch.float64: "torch.float64",
}

_ENV_KEY = "UNSLOTH_FORCE_CUSTOM_DTYPE"

# Values this process set. Not cleared: a model can be loaded more than once.
_REGISTERED = set()


def resolve_dtype(text):
    """Maps a dtype field of `UNSLOTH_FORCE_CUSTOM_DTYPE` onto a torch dtype"""
    key = str(text).strip()
    if key not in DTYPE_ALIASES:
        raise ValueError(
            f"Unsloth: `{_ENV_KEY}` names an unsupported dtype `{key}`.\n"
            f"Supported: {sorted(x for x in DTYPE_ALIASES if x)}"
        )
    return DTYPE_ALIASES[key]


def register_custom_dtype(value):
    """Sets `UNSLOTH_FORCE_CUSTOM_DTYPE` and records it as ours"""
    _REGISTERED.add(value)
    os.environ[_ENV_KEY] = value
    return value


def neutralize_inherited_custom_dtype():
    """Rewrites an INHERITED `UNSLOTH_FORCE_CUSTOM_DTYPE` so no reader can evaluate it.

    `unsloth_zoo==2026.8.15`, which this package's floor resolves to, still `eval`s the
    dtype field. The five fields stay: both packages assert on the separator count.
    """
    value = os.environ.get(_ENV_KEY, "")
    if value == "" or value in _REGISTERED:
        return value
    if value.count(";") < 4:
        # Both readers assert on the layout, so this can only crash an unrelated run.
        os.environ.pop(_ENV_KEY, None)
        return ""
    checker, dtype, bnb_compute_dtype, _custom_datatype, _execute_code = value.split(";", 4)

    def named(field):
        # Empty is what an unset field already looks like to both readers.
        key = field.strip()
        if key == "":
            return ""
        if key not in DTYPE_ALIASES:
            return "None"
        return _CANONICAL_DTYPE_NAMES.get(DTYPE_ALIASES[key], "None")

    # Emptied rather than removed, for the same reason.
    sanitized = ";".join([checker, named(dtype), named(bnb_compute_dtype), "", ""])
    os.environ[_ENV_KEY] = sanitized
    return sanitized


def trusted_custom_dtype():
    """Returns (value, code_is_trusted); False means honour the dtype fields and drop
    the two code fields."""
    value = os.environ.get(_ENV_KEY, "")
    if value == "":
        return "", False
    return value, value in _REGISTERED


# On import, before either package reads the variable.
neutralize_inherited_custom_dtype()
