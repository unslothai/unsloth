# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The AOTriton gate: opened by default, never argued with when already set, and opened
before anything can import torch."""

import re
from pathlib import Path

from unsloth._rocm_attention import AOTRITON_ENV, enable_rocm_aotriton_attention

_INIT = Path(__file__).resolve().parents[2] / "unsloth" / "__init__.py"


def test_opens_the_gate_when_unset():
    env = {}
    assert enable_rocm_aotriton_attention(env) is True
    assert env[AOTRITON_ENV] == "1"


def test_zero_is_the_opt_out_and_survives():
    """ "0" is a deliberate choice by someone who hit an AOTriton bug, not an empty slot."""
    env = {AOTRITON_ENV: "0"}
    assert enable_rocm_aotriton_attention(env) is False
    assert env[AOTRITON_ENV] == "0"


def test_existing_value_is_never_rewritten():
    env = {AOTRITON_ENV: "whatever-the-user-set"}
    assert enable_rocm_aotriton_attention(env) is False
    assert env[AOTRITON_ENV] == "whatever-the-user-set"


def test_idempotent():
    env = {}
    assert enable_rocm_aotriton_attention(env) is True
    assert enable_rocm_aotriton_attention(env) is False
    assert env[AOTRITON_ENV] == "1"


def test_defaults_to_the_real_environment(monkeypatch):
    import os

    monkeypatch.delenv(AOTRITON_ENV, raising = False)
    assert enable_rocm_aotriton_attention() is True
    assert os.environ[AOTRITON_ENV] == "1"


def test_init_opens_the_gate_before_importing_torch():
    """The whole fix is ordering: torch latches the variable while loading its C++
    extension, so a set that lands after any torch import is dead code."""
    source = _INIT.read_text(encoding = "utf-8")
    gate = source.index("_enable_rocm_aotriton_attention()")
    before = source[:gate]
    # No torch import of any spelling may precede the gate.
    assert re.search(r"^\s*(import torch|from torch)\b", before, re.MULTILINE) is None
    # Nor an unsloth submodule that pulls torch on the way in.
    for module in ("_gpu_init", "import_fixes", "models", "kernels"):
        assert f"from .{module}" not in before
        assert f"from unsloth.{module}" not in before


def test_init_imports_the_helper_from_the_package():
    """Guards the import line itself: a refactor that drops it would silently reinstate
    the O(N^2) path with every test above still green."""
    source = _INIT.read_text(encoding = "utf-8")
    assert "from ._rocm_attention import enable_rocm_aotriton_attention" in source
