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
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""A flash-attn / causal-conv1d build for another torch fails with "undefined symbol" and Unsloth
falls back to slower kernels; the hint says how to get matching ones back."""

import importlib.util
import pathlib
import sys
import types

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_STALE = ImportError(
    "/venv/lib/python3.13/site-packages/flash_attn_2_cuda.cpython-313-x86_64-linux-gnu.so: "
    "undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib"
)


def _load_import_fixes():
    # By path: importing the unsloth package needs an accelerator.
    spec = importlib.util.spec_from_file_location(
        "unsloth_import_fixes_stale_kernel", _REPO_ROOT / "unsloth" / "import_fixes.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope = "module")
def fixes():
    return _load_import_fixes()


def _host(
    monkeypatch,
    fixes,
    *,
    torch_version = "2.14.0+cu130",
    cuda = "13.0",
    py = (3, 13),
    machine = "x86_64",
    platform = "linux",
    cxx11 = True,
):
    fake_torch = types.SimpleNamespace(
        __version__ = torch_version,
        version = types.SimpleNamespace(cuda = cuda),
        _C = types.SimpleNamespace(_GLIBCXX_USE_CXX11_ABI = cxx11),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(fixes.sys, "platform", platform)
    monkeypatch.setattr(fixes.sys, "version_info", (*py, 0, "final", 0))
    monkeypatch.setattr(fixes.platform, "machine", lambda: machine)


@pytest.mark.parametrize(
    "package, wheel",
    [
        ("flash_attn", "flash_attn-2.8.4+cu13torch2.14cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"),
        (
            "causal_conv1d",
            "causal_conv1d-1.7.0+cu13torch2.14cxx11abiTRUE-cp313-cp313-linux_x86_64.whl",
        ),
    ],
)
def test_a_published_cell_names_our_matching_wheel(monkeypatch, fixes, package, wheel):
    _host(monkeypatch, fixes)
    hint = fixes.stale_kernel_hint(package, _STALE)
    assert "built for a different torch than 2.14.0+cu130" in hint
    assert hint.endswith(
        "pip install --no-deps --force-reinstall "
        f"https://github.com/unslothai/unsloth/releases/download/prebuilt-wheels-cu13/{wheel}"
    )


def test_the_wheel_names_are_the_published_ones(fixes):
    # Kept in step with the release the prebuilt workflow publishes (studio/backend/utils/wheel_utils.py).
    source = (_REPO_ROOT / "studio" / "backend" / "utils" / "wheel_utils.py").read_text(
        encoding = "utf-8"
    )
    for package, version in fixes._PREBUILT_KERNEL_VERSIONS.items():
        assert (
            f'"{package}": "{version}"' in source or f"'{package}': '{version}'" in source
        ), package
    assert fixes._PREBUILT_KERNEL_RELEASE_URL.endswith("/prebuilt-wheels-cu13")


@pytest.mark.parametrize(
    "host",
    [
        {"torch_version": "2.12.1+cu130"},
        {"cuda": "12.8", "torch_version": "2.14.0+cu128"},
        {"py": (3, 12)},
        {"machine": "aarch64"},
        {"platform": "win32"},
        {"cxx11": False},
    ],
)
def test_anywhere_else_it_names_a_source_rebuild(monkeypatch, fixes, host):
    _host(monkeypatch, fixes, **host)
    hint = fixes.stale_kernel_hint("flash_attn", _STALE)
    assert "releases/download" not in hint
    assert hint.endswith(
        "pip install --no-deps --no-build-isolation --force-reinstall --no-binary flash-attn flash-attn"
    )


def test_a_chained_cause_is_found(monkeypatch, fixes):
    _host(monkeypatch, fixes)
    try:
        try:
            raise _STALE
        except ImportError as inner:
            raise RuntimeError("causal_conv1d import failed") from inner
    except RuntimeError as outer:
        assert "causal_conv1d-1.7.0" in fixes.stale_kernel_hint("causal_conv1d", outer)


@pytest.mark.parametrize(
    "error",
    [
        ImportError("No module named 'flash_attn_2_cuda'"),
        OSError("libcudart.so.13: cannot open shared object file"),
    ],
)
def test_other_failures_add_nothing(monkeypatch, fixes, error):
    _host(monkeypatch, fixes)
    assert fixes.stale_kernel_hint("flash_attn", error) == ""


def test_both_fallback_messages_print_the_hint():
    utils = (_REPO_ROOT / "unsloth" / "models" / "_utils.py").read_text(encoding = "utf-8")
    fixes_src = (_REPO_ROOT / "unsloth" / "import_fixes.py").read_text(encoding = "utf-8")
    assert 'if hint := stale_kernel_hint("flash_attn", error):' in utils
    assert 'hint = stale_kernel_hint("causal_conv1d", error)' in fixes_src
