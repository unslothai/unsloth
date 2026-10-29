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
"""torch's has_cuda / has_mps / ... warnings keep the caller's location through the wrapper."""

import types
import warnings

import pytest

from unsloth import import_fixes


@pytest.fixture
def patched_torch():
    import torch

    assert torch.__dict__.get("__getattr__") is not None
    previous = torch.__dict__["__getattr__"]
    if getattr(previous, "__unsloth_patched__", False):
        torch.__getattr__ = previous.__unsloth_original__
    assert import_fixes.patch_torch_missing_attribute_error() is True
    try:
        yield torch
    finally:
        torch.__getattr__ = previous


_ALIASES = ("has_cuda", "has_cudnn", "has_mkldnn", "has_mps")


def _deprecated_names(torch):
    names = tuple(sorted(getattr(torch, "_deprecated_attrs", {})))
    assert names == _ALIASES, f"torch._deprecated_attrs changed: {names}"
    return names


def _access_from(module_name, filename, attribute):
    import torch

    module = types.ModuleType(module_name)
    module.__dict__["torch"] = torch
    exec(compile(f"value = torch.{attribute}\n", filename, "exec"), module.__dict__)
    return module.__dict__["value"]


def test_get_ignored_functions_stays_silent(patched_torch):
    torch = patched_torch
    _deprecated_names(torch)
    torch.overrides.get_ignored_functions.cache_clear()
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("default")
        torch.overrides.get_ignored_functions()
    deprecated = [w for w in caught if "is deprecated, please use" in str(w.message)]
    assert deprecated == [], [f"{w.filename}:{w.lineno} {w.message}" for w in deprecated]


@pytest.mark.parametrize("name", _ALIASES)
def test_direct_access_warns_at_the_caller(patched_torch, name):
    torch = patched_torch
    _deprecated_names(torch)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        value = _access_from("user_code", "<user_code>", name)
    assert value == torch._deprecated_attrs[name]()
    assert len(caught) == 1
    assert caught[0].category is UserWarning
    assert f"'{name}' is deprecated" in str(caught[0].message)
    assert caught[0].filename == "<user_code>"
    assert caught[0].filename != import_fixes.__file__


def test_message_matches_torch(patched_torch):
    torch = patched_torch
    original = torch.__dict__["__getattr__"].__unsloth_original__
    for name in _deprecated_names(torch):
        with warnings.catch_warnings(record = True) as ours:
            warnings.simplefilter("always")
            getattr(torch, name)
        with warnings.catch_warnings(record = True) as theirs:
            warnings.simplefilter("always")
            original(name)
        assert [str(w.message) for w in ours] == [str(w.message) for w in theirs]
        assert [w.category for w in ours] == [w.category for w in theirs]


def test_repeated_access_warns_once_per_location(patched_torch):
    torch = patched_torch
    name = _deprecated_names(torch)[0]
    module = types.ModuleType("repeat_user")
    module.__dict__["torch"] = torch
    code = compile(f"for _ in range(5): torch.{name}\n", "<repeat_user>", "exec")
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("default")
        exec(code, module.__dict__)
        exec(code, module.__dict__)
    assert len(caught) == 1


def test_module_filter_on_the_caller_applies(patched_torch):
    torch = patched_torch
    name = _deprecated_names(torch)[0]
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category = UserWarning, module = "quiet_pkg")
        _access_from("quiet_pkg.mod", "<quiet_pkg>", name)
    assert caught == []


def test_other_names_are_unchanged(patched_torch):
    torch = patched_torch
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        assert torch._dynamo is not None
        assert not hasattr(torch, "unsloth_not_a_real_attribute")
    assert caught == []
    with pytest.raises(AttributeError):
        torch.unsloth_not_a_real_attribute


def test_repatch_is_a_noop(patched_torch):
    torch = patched_torch
    installed = torch.__dict__["__getattr__"]
    assert import_fixes.patch_torch_missing_attribute_error() is True
    assert torch.__dict__["__getattr__"] is installed
