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
"""On MLX, trl.__all__ is filtered on first read: resolving every name at import pulled torch in."""

import importlib.util
import pathlib
import platform
import subprocess
import sys
import types

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "_mlx_trl_all.py"


@pytest.fixture
def lazy_all():
    spec = importlib.util.spec_from_file_location("_unsloth_mlx_trl_all_under_test", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _filter(trl, names):
    """Same contract as unsloth's _safe_mlx_trl_star_exports."""
    kept = []
    for name in names:
        try:
            getattr(trl, name)
        except Exception:
            continue
        kept.append(name)
    for name in ("SFTConfig", "SFTTrainer"):
        if name not in kept:
            kept.append(name)
    return kept


class _LazyLike(types.ModuleType):
    """trl's _LazyModule shape: a class-level __getattr__ importing on demand."""

    def __init__(self, name, lazy):
        super().__init__(name)
        self._lazy = lazy
        self.resolved = []
        self.__all__ = list(lazy)

    def __getattr__(self, name):
        if name not in self._lazy:
            raise AttributeError(name)
        self.resolved.append(name)
        value = self._lazy[name]()
        setattr(self, name, value)
        return value


def _broken():
    raise RuntimeError("optional dependency missing")


def test_nothing_resolves_until_all_is_read(lazy_all, monkeypatch):
    trl = _LazyLike("trl", {"DataCollator": object, "Broken": _broken, "GRPOTrainer": object})
    trl.SFTTrainer = trl.SFTConfig = object
    lazy_all.install_lazy_trl_all(trl, _filter)
    assert trl.resolved == []
    assert lazy_all.raw_trl_all(trl) == ["DataCollator", "Broken", "GRPOTrainer"]
    assert trl.resolved == []
    assert trl.__all__ == ["DataCollator", "GRPOTrainer", "SFTConfig", "SFTTrainer"]
    assert sorted(trl.resolved) == ["Broken", "DataCollator", "GRPOTrainer"]
    # Cached: a second read resolves nothing more.
    trl.resolved.clear()
    assert trl.__all__[-1] == "SFTTrainer" and trl.resolved == []
    monkeypatch.setitem(sys.modules, "trl", trl)
    namespace = {}
    exec("from trl import *", namespace)
    assert set(namespace) - {"__builtins__"} == {
        "DataCollator",
        "GRPOTrainer",
        "SFTConfig",
        "SFTTrainer",
    }


def test_class_level_getattr_still_serves_other_names(lazy_all):
    trl = _LazyLike("trl", {"DataCollator": lambda: "collator"})
    lazy_all.install_lazy_trl_all(trl, _filter)
    assert trl.DataCollator == "collator"
    with pytest.raises(AttributeError):
        trl.missing_name
    assert "DataCollator" in dir(trl)


def test_module_dict_getattr_is_chained(lazy_all):
    trl = types.ModuleType("trl")
    trl.ExistingExport = object()
    trl.__all__ = ["ExistingExport", "BrokenExport"]

    def _module_getattr(name):
        if name == "BrokenExport":
            raise RuntimeError("optional dependency missing")
        raise AttributeError(name)

    trl.__getattr__ = _module_getattr
    lazy_all.install_lazy_trl_all(trl, _filter)
    with pytest.raises(RuntimeError):
        trl.BrokenExport
    assert trl.__all__ == ["ExistingExport", "SFTConfig", "SFTTrainer"]


def test_reinstall_keeps_the_original_names_and_getattr(lazy_all):
    trl = types.ModuleType("trl")
    trl.__all__ = ["A"]
    trl.A = 1
    original = lambda name: (_ for _ in ()).throw(AttributeError(name))  # noqa: E731
    trl.__getattr__ = original
    lazy_all.install_lazy_trl_all(trl, _filter)
    assert trl.__all__ == ["A", "SFTConfig", "SFTTrainer"]
    lazy_all.install_lazy_trl_all(trl, _filter)
    assert lazy_all.raw_trl_all(trl) == ["A"]
    assert trl.__getattr__._unsloth_previous is original
    assert trl.__all__ == ["A", "SFTConfig", "SFTTrainer"]


def test_module_without_all(lazy_all):
    trl = types.ModuleType("trl")
    lazy_all.install_lazy_trl_all(trl, _filter)
    assert trl.__all__ == ["SFTConfig", "SFTTrainer"]


def test_a_reentrant_all_read_during_filtering_sees_the_raw_names(lazy_all):
    install_lazy_trl_all = lazy_all.install_lazy_trl_all

    trl = types.ModuleType("fake_trl_reentrant")
    trl.__all__ = ["A", "B"]
    seen = []

    def filter_exports(module, names):
        # An export whose import reads trl.__all__ again mid-filter.
        seen.append(list(module.__all__))
        return [n for n in names if n != "B"]

    install_lazy_trl_all(trl, filter_exports)
    assert trl.__all__ == ["A"]
    assert seen == [["A", "B"]]
    assert trl.__all__ == ["A"]


class _LazyDirLike(_LazyLike):
    """trl's _LazyModule.__dir__: the namespace listing, then any __all__ name not in it."""

    def __dir__(self):
        result = super().__dir__()
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result


@pytest.mark.parametrize("dir_first", [True, False])
def test_dir_matches_the_eager_filter(lazy_all, dir_first):
    def make():
        trl = _LazyDirLike(
            "trl", {"DataCollator": object, "Broken": _broken, "GRPOTrainer": object}
        )
        trl.SFTTrainer = trl.SFTConfig = object
        return trl

    eager = make()
    eager.__all__ = _filter(eager, list(eager.__all__))
    lazy = make()
    lazy_all.install_lazy_trl_all(lazy, _filter)
    if dir_first:
        listed = dir(lazy)
        assert lazy.__all__ == eager.__all__
    else:
        assert lazy.__all__ == eager.__all__
        listed = dir(lazy)
    assert listed == dir(eager)
    # Built: no hook or bookkeeping name is left in trl's namespace.
    assert not {"__getattr__", "__dir__", "_unsloth_mlx_trl_all"} & set(vars(lazy))


@pytest.mark.skipif(
    platform.system() != "Darwin"
    or platform.machine() != "arm64"
    or importlib.util.find_spec("mlx") is None
    or importlib.util.find_spec("trl") is None,
    reason = "the real MLX import path needs Apple Silicon with mlx and trl",
)
def test_real_mlx_import_leaves_trl_all_unbuilt():
    probe = (
        "import sys, unsloth, trl\n"
        "assert unsloth.DEVICE_TYPE == 'mlx', unsloth.DEVICE_TYPE\n"
        "assert '__all__' not in vars(trl), 'import unsloth filtered trl.__all__ eagerly'\n"
        "names = trl.__all__\n"
        "assert {'SFTConfig', 'SFTTrainer'} <= set(names), names\n"
        "ns = {}\n"
        "exec('from trl import *', ns)\n"
        "assert set(names) <= set(ns), set(names) - set(ns)\n"
        "assert ns['SFTTrainer'] is unsloth.UnslothTrainer\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd = pathlib.Path(__file__).resolve().parents[1],
        capture_output = True,
        text = True,
        timeout = 600,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
