# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Intel XPU Triton's compiler on Windows (#14807). Discovery is stubbed, so these also run on CI Linux."""

import os
import sys
import types
import logging

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from core import _msvc_env
from core import _xpu_triton_toolchain as toolchain

CL = os.path.join("C:", "VS", "VC", "Tools", "MSVC", "bin", "Hostx64", "x64", "cl.exe")


@pytest.fixture
def xpu_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(toolchain, "_triton_is_xpu", lambda: True)
    monkeypatch.delenv("CXX", raising = False)


def _which(table):
    return lambda name: table.get(name)


def test_triton_spelling_matches_its_msvc_test():
    spelled = toolchain._triton_cl_spelling(CL)
    assert "cl.EXE" in spelled
    assert os.path.dirname(spelled) == os.path.dirname(CL)


def test_loads_visual_studio_and_points_triton_at_cl(xpu_windows, monkeypatch):
    table = {}

    def load():
        table["cl"] = CL
        return True

    monkeypatch.setattr(toolchain.shutil, "which", _which(table))
    monkeypatch.setattr(toolchain, "_msvc_environment_loaded", lambda: False)
    monkeypatch.setattr(toolchain, "_load_msvc_environment", load)

    toolchain.prepare_xpu_triton_compiler(logging.getLogger("t"))

    assert os.environ["CXX"] == toolchain._triton_cl_spelling(CL)


def test_a_lowercase_cl_in_cxx_is_respelled(xpu_windows, monkeypatch):
    monkeypatch.setenv("CXX", CL)
    monkeypatch.setattr(toolchain.shutil, "which", _which({CL: CL}))
    monkeypatch.setattr(toolchain, "_msvc_environment_loaded", lambda: True)

    def never():
        raise AssertionError("the environment was already loaded")

    monkeypatch.setattr(toolchain, "_load_msvc_environment", never)

    toolchain.prepare_xpu_triton_compiler(logging.getLogger("t"))

    assert "cl.EXE" in os.environ["CXX"]


def test_a_gnu_driver_in_cxx_is_left_alone_with_a_warning(xpu_windows, monkeypatch, caplog):
    monkeypatch.setenv("CXX", "clang++.exe")
    monkeypatch.setattr(toolchain, "_load_msvc_environment", lambda: pytest.fail("not wanted"))

    with caplog.at_level(logging.WARNING):
        toolchain.prepare_xpu_triton_compiler(logging.getLogger("t"))

    assert os.environ["CXX"] == "clang++.exe"
    assert "/LIBPATH:" in caplog.text


@pytest.mark.parametrize("where", ["LEVEL_ZERO_V1_SDK_PATH", "ZE_PATH", "INCLUDE", None])
def test_a_missing_level_zero_sdk_is_named(xpu_windows, monkeypatch, caplog, tmp_path, where):
    for var in ("LEVEL_ZERO_V1_SDK_PATH", "ZE_PATH", "INCLUDE"):
        monkeypatch.delenv(var, raising = False)
    (tmp_path / "include" / "level_zero").mkdir(parents = True)
    (tmp_path / "include" / "level_zero" / "ze_api.h").write_text("")
    if where == "INCLUDE":
        monkeypatch.setenv("INCLUDE", str(tmp_path / "include"))
    elif where:
        monkeypatch.setenv(where, str(tmp_path))
    monkeypatch.setattr(toolchain.shutil, "which", _which({"cl": CL}))
    monkeypatch.setattr(toolchain, "_msvc_environment_loaded", lambda: True)
    monkeypatch.setattr(toolchain, "_provide_level_zero_sdk", lambda: None)

    with caplog.at_level(logging.WARNING):
        toolchain.prepare_xpu_triton_compiler(logging.getLogger("t"))

    assert os.environ["CXX"] == toolchain._triton_cl_spelling(CL)
    assert ("LEVEL_ZERO_V1_SDK_PATH" in caplog.text) is (where is None)


def test_a_missing_level_zero_sdk_is_provided(xpu_windows, monkeypatch, caplog):
    for var in ("LEVEL_ZERO_V1_SDK_PATH", "ZE_PATH", "INCLUDE"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr(toolchain.shutil, "which", _which({"cl": CL}))
    monkeypatch.setattr(toolchain, "_msvc_environment_loaded", lambda: True)
    monkeypatch.setattr(toolchain, "_provide_level_zero_sdk", lambda: r"C:\studio\cache\l0")

    with caplog.at_level(logging.WARNING):
        toolchain.prepare_xpu_triton_compiler(logging.getLogger("t"))

    assert os.environ["LEVEL_ZERO_V1_SDK_PATH"] == r"C:\studio\cache\l0"
    assert "Level Zero" not in caplog.text


def _sdk_zip(names):
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name in names:
            z.writestr(name, "x")
    return buf.getvalue()


def _serve(
    monkeypatch,
    blob,
    pin = True,
):
    import hashlib
    import contextlib
    import urllib.request

    if pin:
        monkeypatch.setattr(toolchain, "_L0_SHA256", hashlib.sha256(blob).hexdigest())

    @contextlib.contextmanager
    def urlopen(url, timeout):
        yield types.SimpleNamespace(read = lambda: blob)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)


def test_the_sdk_archive_is_unpacked_from_its_backslash_names(tmp_path, monkeypatch):
    # The real archive's layout: Windows separators inside the zip.
    _serve(monkeypatch, _sdk_zip([r"include\level_zero\ze_api.h", r"lib\ze_loader.lib"]))
    dest = tmp_path / "level-zero-sdk"

    toolchain._fetch_level_zero_sdk(dest)

    assert toolchain._sdk_complete(dest)
    assert [p.name for p in tmp_path.iterdir()] == ["level-zero-sdk"]


@pytest.mark.parametrize(
    "names, pin",
    [
        ([r"include\level_zero\ze_api.h", r"lib\ze_loader.lib"], False),
        ([r"..\evil.dll", r"lib\ze_loader.lib"], True),
    ],
)
def test_a_bad_sdk_archive_leaves_nothing_behind(tmp_path, monkeypatch, names, pin):
    _serve(monkeypatch, _sdk_zip(names), pin = pin)
    dest = tmp_path / "level-zero-sdk"

    with pytest.raises(ValueError):
        toolchain._fetch_level_zero_sdk(dest)

    assert list(tmp_path.iterdir()) == []


def test_icpx_on_path_is_left_to_triton(xpu_windows, monkeypatch):
    monkeypatch.setattr(toolchain.shutil, "which", _which({"icpx": "icpx.exe"}))
    monkeypatch.setattr(toolchain, "_load_msvc_environment", lambda: pytest.fail("not wanted"))

    toolchain.prepare_xpu_triton_compiler(logging.getLogger("t"))

    assert "CXX" not in os.environ


def test_no_compiler_anywhere_warns_and_sets_nothing(xpu_windows, monkeypatch, caplog):
    monkeypatch.setattr(toolchain.shutil, "which", _which({}))
    monkeypatch.setattr(toolchain, "_msvc_environment_loaded", lambda: False)
    monkeypatch.setattr(toolchain, "_load_msvc_environment", lambda: False)

    with caplog.at_level(logging.WARNING):
        toolchain.prepare_xpu_triton_compiler(logging.getLogger("t"))

    assert "CXX" not in os.environ
    assert "Build Tools" in caplog.text


@pytest.mark.parametrize(
    "backends, expected", [(("amd", "nvidia"), False), (("amd", "intel", "nvidia"), True)]
)
def test_only_a_triton_with_an_intel_backend_is_xpu(tmp_path, monkeypatch, backends, expected):
    pkg = tmp_path / "triton"
    for name in backends:
        (pkg / "backends" / name).mkdir(parents = True)
    fake = types.ModuleType("triton")
    fake.__file__ = str(pkg / "__init__.py")
    monkeypatch.setitem(sys.modules, "triton", fake)

    assert toolchain._triton_is_xpu() is expected


def test_triton_windows_is_not_touched(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(toolchain, "_triton_is_xpu", lambda: False)
    monkeypatch.delenv("CXX", raising = False)
    monkeypatch.setattr(toolchain, "_prepare", lambda log: pytest.fail("not wanted"))

    toolchain.prepare_xpu_triton_compiler(logging.getLogger("t"))

    assert "CXX" not in os.environ


def test_off_windows_nothing_happens(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(toolchain, "_prepare", lambda log: pytest.fail("not wanted"))

    toolchain.prepare_xpu_triton_compiler(logging.getLogger("t"))


def test_a_failure_never_reaches_the_worker(xpu_windows, monkeypatch):
    def boom(log):
        raise RuntimeError("vswhere blew up")

    monkeypatch.setattr(toolchain, "_prepare", boom)

    toolchain.prepare_xpu_triton_compiler(logging.getLogger("t"))


def test_vcvars_environment_keeps_only_what_cl_needs(monkeypatch):
    out = "\n".join(
        [
            r"Path=C:\VS\bin;C:\Users\José\bin",
            r"INCLUDE=C:\VS\include",
            r"LIB=C:\VS\lib",
            r"LIBPATH=C:\VS\libpath",
            r"VSCMD_ARG_TGT_ARCH=x64",
            r"USERNAME=someone",
        ]
    )
    monkeypatch.setattr(
        toolchain.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode = 0, stdout = out.encode("utf-16-le")),
    )

    env = toolchain._vcvars_environment(r"C:\VS\vcvars64.bat")

    assert env == {
        "PATH": r"C:\VS\bin;C:\Users\José\bin",
        "INCLUDE": r"C:\VS\include",
        "LIB": r"C:\VS\lib",
        "LIBPATH": r"C:\VS\libpath",
    }


def test_a_failed_vcvars_yields_nothing(monkeypatch):
    monkeypatch.setattr(
        toolchain.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode = 1, stdout = b""),
    )

    assert toolchain._vcvars_environment(r"C:\VS\vcvars64.bat") is None


def test_the_windows_gate_prepares_xpu_triton(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(_msvc_env, "_torch_is_rocm_build", lambda: False)
    monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))
    monkeypatch.setattr(_msvc_env, "crt_headers_reachable", lambda: True)
    calls = []
    monkeypatch.setattr(toolchain, "prepare_xpu_triton_compiler", lambda log: calls.append(log))

    _msvc_env.gate_torch_compile_on_windows(logging.getLogger("t"))

    assert len(calls) == 1
