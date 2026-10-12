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
"""#2760: triton < 3.4 reused a driver module another Python version compiled."""

import importlib.util
import sys
import sysconfig
import textwrap
import types

from unsloth import import_fixes


_TRITON_33_DRIVER = textwrap.dedent(
    """
    seen = []

    def compile_module_from_src(src, name):
        seen.append((src, name))
        return name

    class CudaDriver:
        def __init__(self):
            self.utils = compile_module_from_src("driver.c source", "cuda_utils")
    """
)

_TRITON_34_DRIVER = textwrap.dedent(
    """
    from keyed_build import compile_module_from_src

    class CudaDriver:
        def __init__(self):
            self.utils = compile_module_from_src("driver.c source", "cuda_utils")
    """
)


def _install_backend(monkeypatch, tmp_path, source):
    # Load the way triton.backends does: spec_from_file_location, never in sys.modules.
    path = tmp_path / "driver.py"
    path.write_text(source, encoding = "utf-8")
    spec = importlib.util.spec_from_file_location("nvidia", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registry = types.ModuleType("triton.backends")
    registry.backends = {"nvidia": types.SimpleNamespace(driver = module.CudaDriver)}
    monkeypatch.setitem(sys.modules, "triton.backends", registry)
    monkeypatch.setattr(import_fixes.importlib.util, "find_spec", lambda name: object())
    return module


def test_triton_33_driver_modules_are_keyed_by_python(monkeypatch, tmp_path):
    module = _install_backend(monkeypatch, tmp_path, _TRITON_33_DRIVER)
    import_fixes.fix_triton_module_cache_python_key()
    import_fixes.fix_triton_module_cache_python_key()

    assert module.CudaDriver().utils == "cuda_utils"
    ((src, name),) = module.seen
    assert name == "cuda_utils"
    assert src.startswith("driver.c source")
    assert sysconfig.get_config_var("EXT_SUFFIX") in src
    # Idempotent: one tag, one wrapper.
    assert src.count("unsloth:") == 1
    assert not getattr(module.compile_module_from_src.__wrapped__, "__unsloth_patched__", False)


def test_triton_34_keyed_build_is_left_alone(monkeypatch, tmp_path):
    keyed = types.ModuleType("keyed_build")

    def compile_module_from_src(src, name):
        return src

    compile_module_from_src.__module__ = "keyed_build"
    keyed.compile_module_from_src = compile_module_from_src
    monkeypatch.setitem(sys.modules, "keyed_build", keyed)
    module = _install_backend(monkeypatch, tmp_path, _TRITON_34_DRIVER)

    import_fixes.fix_triton_module_cache_python_key()
    assert module.compile_module_from_src is compile_module_from_src
    assert module.CudaDriver().utils == "driver.c source"
