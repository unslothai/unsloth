# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Triton C-toolchain gate on Windows (#7595). Discovery is stubbed, so these also run on CI Linux."""

import os
import sys
import types
import logging

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from core import _msvc_env


def _fake_triton(
    monkeypatch,
    inc_dirs,
    cc = "clang-cl.exe",
    with_is_clang_cl = True,
):
    pkg = types.ModuleType("triton")
    utils = types.ModuleType("triton.windows_utils")
    utils.find_msvc_winsdk = lambda *a, **k: (None, list(inc_dirs), [])
    pkg.windows_utils = utils
    runtime = types.ModuleType("triton.runtime")
    build = types.ModuleType("triton.runtime.build")
    build.get_cc = lambda: cc
    build.is_msvc = lambda c: os.path.basename(c).lower() in ("cl", "cl.exe")
    if with_is_clang_cl:
        build.is_clang_cl = lambda c: os.path.basename(c).lower() in ("clang-cl", "clang-cl.exe")
    runtime.build = build
    pkg.runtime = runtime
    monkeypatch.setitem(sys.modules, "triton", pkg)
    monkeypatch.setitem(sys.modules, "triton.windows_utils", utils)
    monkeypatch.setitem(sys.modules, "triton.runtime", runtime)
    monkeypatch.setitem(sys.modules, "triton.runtime.build", build)


def test_have_crt_headers_true_when_include_has_both_headers(tmp_path, monkeypatch):
    (tmp_path / "stdlib.h").write_text("/* stub */")
    (tmp_path / "vcruntime.h").write_text("/* stub */")
    monkeypatch.setenv("INCLUDE", str(tmp_path))
    assert _msvc_env._have_crt_headers() is True


def test_have_crt_headers_false_when_include_lacks_stdlib(tmp_path, monkeypatch):
    monkeypatch.setenv("INCLUDE", str(tmp_path))
    assert _msvc_env._have_crt_headers() is False


def test_have_crt_headers_false_when_include_has_only_stdlib(tmp_path, monkeypatch):
    (tmp_path / "stdlib.h").write_text("/* stub */")
    monkeypatch.setenv("INCLUDE", str(tmp_path))
    assert _msvc_env._have_crt_headers() is False


def test_have_crt_headers_false_when_include_unset(monkeypatch):
    monkeypatch.delenv("INCLUDE", raising = False)
    assert _msvc_env._have_crt_headers() is False


def _sdk_dirs(tmp_path, *, with_toolset):
    """The real layout: `stdlib.h` in the SDK's ucrt, `vcruntime.h` in the VC toolset include."""
    ucrt = tmp_path / "ucrt"
    ucrt.mkdir()
    (ucrt / "stdlib.h").write_text("/* stub */")
    dirs = [str(tmp_path / "um"), str(ucrt)]
    (tmp_path / "um").mkdir()
    if with_toolset:
        msvc = tmp_path / "msvc"
        msvc.mkdir()
        (msvc / "vcruntime.h").write_text("/* stub */")
        dirs.append(str(msvc))
    return dirs


def test_triton_discovery_true_when_dirs_carry_both_headers(tmp_path, monkeypatch):
    _fake_triton(monkeypatch, _sdk_dirs(tmp_path, with_toolset = True))
    assert _msvc_env._triton_finds_crt_headers() is True


def test_triton_discovery_false_when_sdk_lacks_the_vc_toolset(tmp_path, monkeypatch):
    """A standalone SDK passes the `stdlib.h` check and the compile still dies on `vcruntime.h`
    (measured on an R9700 with the toolset dir removed). This must gate."""
    _fake_triton(monkeypatch, _sdk_dirs(tmp_path, with_toolset = False))
    assert _msvc_env._triton_finds_crt_headers() is False


def test_triton_discovery_false_when_no_dir_has_stdlib(tmp_path, monkeypatch):
    _fake_triton(monkeypatch, [str(tmp_path)])
    assert _msvc_env._triton_finds_crt_headers() is False


def test_triton_discovery_false_when_search_returns_nothing(monkeypatch):
    _fake_triton(monkeypatch, [])
    assert _msvc_env._triton_finds_crt_headers() is False


def test_triton_discovery_is_unknown_when_triton_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton.windows_utils", None)
    assert _msvc_env._triton_finds_crt_headers() is None


def test_triton_discovery_survives_a_raising_search(monkeypatch):
    pkg = types.ModuleType("triton")
    utils = types.ModuleType("triton.windows_utils")

    def boom(*a, **k):
        raise RuntimeError("vswhere blew up")

    utils.find_msvc_winsdk = boom
    pkg.windows_utils = utils
    monkeypatch.setitem(sys.modules, "triton", pkg)
    monkeypatch.setitem(sys.modules, "triton.windows_utils", utils)
    assert _msvc_env._triton_finds_crt_headers() is None


def test_triton_discovery_is_unknown_when_the_search_changes_arity(monkeypatch):
    """`find_msvc_winsdk` is private. A 2- or 4-tuple must read as unknown, never as "no headers"."""
    for shape in ((["c:/inc"], []), ("", ["c:/inc"], [], [])):
        pkg = types.ModuleType("triton")
        utils = types.ModuleType("triton.windows_utils")
        utils.find_msvc_winsdk = lambda s = shape: s
        pkg.windows_utils = utils
        monkeypatch.setitem(sys.modules, "triton", pkg)
        monkeypatch.setitem(sys.modules, "triton.windows_utils", utils)
        assert _msvc_env._triton_finds_crt_headers() is None


def test_triton_discovery_is_unknown_when_the_search_returns_none(monkeypatch):
    pkg = types.ModuleType("triton")
    utils = types.ModuleType("triton.windows_utils")
    utils.find_msvc_winsdk = lambda *a, **k: None
    pkg.windows_utils = utils
    monkeypatch.setitem(sys.modules, "triton", pkg)
    monkeypatch.setitem(sys.modules, "triton.windows_utils", utils)
    assert _msvc_env._triton_finds_crt_headers() is None


def test_reachable_is_true_off_win32(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        _msvc_env,
        "_triton_finds_crt_headers",
        lambda: (_ for _ in ()).throw(AssertionError("probed off win32")),
    )
    assert _msvc_env.crt_headers_reachable() is True


def test_reachable_via_triton_even_when_include_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("INCLUDE", raising = False)
    _fake_triton(monkeypatch, _sdk_dirs(tmp_path, with_toolset = True))
    assert _msvc_env.crt_headers_reachable() is True


def test_reachable_via_include_when_triton_finds_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    (tmp_path / "stdlib.h").write_text("/* stub */")
    (tmp_path / "vcruntime.h").write_text("/* stub */")
    monkeypatch.setenv("INCLUDE", str(tmp_path))
    _fake_triton(monkeypatch, [])
    assert _msvc_env.crt_headers_reachable() is True


def test_not_reachable_when_neither_source_has_headers(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("INCLUDE", str(tmp_path))
    _fake_triton(monkeypatch, [])
    assert _msvc_env.crt_headers_reachable() is False


def test_unreachable_discovery_does_not_gate_a_visual_studio_box(tmp_path, monkeypatch):
    """The dangerous direction. Discovery that cannot be asked is not evidence of a missing SDK, and
    clang-cl locates MSVC itself, so gating here would disable torch.compile on a box that compiles."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("INCLUDE", raising = False)
    _fake_triton(monkeypatch, [], cc = "clang-cl.exe")
    monkeypatch.setitem(sys.modules, "triton.windows_utils", None)
    assert _msvc_env.crt_headers_reachable() is True


def test_a_search_that_ran_and_found_nothing_still_gates(tmp_path, monkeypatch):
    """The other half: without this, the #7595 crash is no longer prevented at all."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("INCLUDE", raising = False)
    _fake_triton(monkeypatch, [], cc = "clang-cl.exe")
    assert _msvc_env.crt_headers_reachable() is False


def test_tinycc_does_not_need_msvc_headers(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("INCLUDE", str(tmp_path))
    _fake_triton(monkeypatch, [], cc = "tcc.exe")
    assert _msvc_env._needs_msvc_headers() is False
    assert _msvc_env.crt_headers_reachable() is True


def test_clang_cl_does_need_msvc_headers(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("INCLUDE", str(tmp_path))
    _fake_triton(monkeypatch, [], cc = "clang-cl.exe")
    assert _msvc_env._needs_msvc_headers() is True
    assert _msvc_env.crt_headers_reachable() is False


def test_xpu_triton_without_the_private_api_is_not_gated(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("INCLUDE", str(tmp_path))
    monkeypatch.setitem(sys.modules, "triton.runtime.build", None)
    monkeypatch.setattr(_msvc_env, "_rocm_clang_cl_present", lambda: False)
    assert _msvc_env._needs_msvc_headers() is False
    assert _msvc_env.crt_headers_reachable() is True


def test_stale_rocm_clang_cl_under_xpu_triton_is_not_gated(tmp_path, monkeypatch):
    """setup.ps1:4162 repairs in place without pruning, so `_rocm_sdk_core` outlives its Triton."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("INCLUDE", str(tmp_path))
    monkeypatch.setitem(sys.modules, "triton.runtime.build", None)
    monkeypatch.setattr(_msvc_env, "_rocm_clang_cl_present", lambda: True)
    monkeypatch.setattr(_msvc_env, "_triton_is_triton_windows", lambda: False)
    assert _msvc_env._needs_msvc_headers() is False
    assert _msvc_env.crt_headers_reachable() is True


def test_tinycc_is_not_gated_on_a_release_without_is_clang_cl(tmp_path, monkeypatch):
    """triton-windows 3.2.0.post18 through 3.5.1.post22 ship `get_cc` and `is_msvc` but no
    `is_clang_cl`. Importing the three together lost `get_cc` too, so an AMD box that still had the
    ROCm wheel on disk fell through to the wheel-layout guess and was gated, even though those
    releases predate the ROCm clang-cl branch and pick TinyCC, which compiles."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("INCLUDE", raising = False)
    _fake_triton(monkeypatch, [], cc = "tcc.exe", with_is_clang_cl = False)
    monkeypatch.setattr(_msvc_env, "_rocm_clang_cl_present", lambda: True)
    monkeypatch.setattr(_msvc_env, "_triton_is_triton_windows", lambda: True)
    assert _msvc_env._needs_msvc_headers() is False
    assert _msvc_env.crt_headers_reachable() is True


def test_clang_cl_is_still_gated_without_is_clang_cl(tmp_path, monkeypatch):
    """The other half: dropping the predicate must not drop the detection."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("INCLUDE", raising = False)
    _fake_triton(monkeypatch, [], cc = "clang-cl.exe", with_is_clang_cl = False)
    assert _msvc_env._needs_msvc_headers() is True
    assert _msvc_env.crt_headers_reachable() is False


def test_a_raising_predicate_does_not_escape(tmp_path, monkeypatch):
    """`is_msvc(None)` is a TypeError. The call used to sit outside the try, so it escaped."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("INCLUDE", raising = False)
    _fake_triton(monkeypatch, [], cc = "tcc.exe")
    build = sys.modules["triton.runtime.build"]

    def boom(_c):
        raise TypeError("expected str, bytes or os.PathLike object, not NoneType")

    monkeypatch.setattr(build, "is_msvc", boom)
    assert _msvc_env._needs_msvc_headers() is False
    assert _msvc_env.crt_headers_reachable() is True


def _fake_triton_38(monkeypatch, inc_dirs, cc):
    """triton-windows 3.8.0.post28: `get_cc` is gone, `_find_compiler(language)` replaces it."""
    _fake_triton(monkeypatch, inc_dirs, cc = cc)
    build = sys.modules["triton.runtime.build"]
    monkeypatch.delattr(build, "get_cc")
    build._find_compiler = lambda language: cc
    return build


def test_the_get_cc_rename_is_followed(tmp_path, monkeypatch):
    """3.8.0.post28 is what a bare `pip install triton-windows` resolves to, and it has no `get_cc`.
    Falling through to the wheel-layout guess stops asking Triton which compiler it will run."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("INCLUDE", raising = False)
    _fake_triton_38(monkeypatch, [], cc = "tcc.exe")
    monkeypatch.setattr(_msvc_env, "_rocm_clang_cl_present", lambda: True)
    monkeypatch.setattr(_msvc_env, "_triton_is_triton_windows", lambda: True)
    assert _msvc_env._triton_cc() == "tcc.exe"
    assert _msvc_env._needs_msvc_headers() is False
    assert _msvc_env.crt_headers_reachable() is True


def test_the_get_cc_rename_still_gates_clang_cl(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("INCLUDE", raising = False)
    _fake_triton_38(monkeypatch, [], cc = "clang-cl.exe")
    assert _msvc_env._needs_msvc_headers() is True
    assert _msvc_env.crt_headers_reachable() is False


def test_neither_compiler_name_present_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("INCLUDE", str(tmp_path))
    _fake_triton(monkeypatch, [], cc = "clang-cl.exe")
    build = sys.modules["triton.runtime.build"]
    monkeypatch.delattr(build, "get_cc")
    monkeypatch.setattr(_msvc_env, "_rocm_clang_cl_present", lambda: False)
    assert _msvc_env._needs_msvc_headers() is False
    assert _msvc_env.crt_headers_reachable() is True


def test_triton_is_triton_windows_reads_the_distribution_name(monkeypatch):
    import importlib.metadata as md

    monkeypatch.setattr(md, "packages_distributions", lambda: {"triton": ["triton-windows"]})
    assert _msvc_env._triton_is_triton_windows() is True
    monkeypatch.setattr(md, "packages_distributions", lambda: {"triton": ["pytorch-triton-xpu"]})
    assert _msvc_env._triton_is_triton_windows() is False
    monkeypatch.setattr(md, "packages_distributions", lambda: {})
    assert _msvc_env._triton_is_triton_windows() is False

    def boom():
        raise RuntimeError("metadata is unreadable")

    monkeypatch.setattr(md, "packages_distributions", boom)
    assert _msvc_env._triton_is_triton_windows() is False


def test_no_private_api_but_rocm_clang_cl_on_disk_is_gated(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("INCLUDE", str(tmp_path))
    monkeypatch.setitem(sys.modules, "triton.runtime.build", None)
    monkeypatch.setattr(_msvc_env, "_rocm_clang_cl_present", lambda: True)
    monkeypatch.setattr(_msvc_env, "_triton_is_triton_windows", lambda: True)
    # Left real, this asks the runner's own Visual Studio and answers for the wrong machine.
    monkeypatch.setattr(_msvc_env, "_triton_finds_crt_headers", lambda: False)
    assert _msvc_env._needs_msvc_headers() is True
    assert _msvc_env.crt_headers_reachable() is False


def test_rocm_clang_cl_present_probes_the_platlib_path(tmp_path, monkeypatch):
    import sysconfig

    monkeypatch.setattr(sysconfig, "get_path", lambda name: str(tmp_path))
    assert _msvc_env._rocm_clang_cl_present() is False
    exe = tmp_path / "_rocm_sdk_core" / "lib" / "llvm" / "bin"
    exe.mkdir(parents = True)
    (exe / "clang-cl.exe").write_text("")
    assert _msvc_env._rocm_clang_cl_present() is True


def test_toolchain_summary_separates_no_vs_from_a_partial_sdk(tmp_path, monkeypatch):
    _fake_triton(monkeypatch, [], cc = "clang-cl.exe")
    monkeypatch.delenv("INCLUDE", raising = False)
    summary = _msvc_env._toolchain_summary()
    assert "compiler=clang-cl.exe" in summary
    assert "include dirs=0" in summary
    assert "missing headers=stdlib.h,vcruntime.h" in summary
    assert "INCLUDE=unset" in summary

    _fake_triton(monkeypatch, _sdk_dirs(tmp_path, with_toolset = False), cc = "clang-cl.exe")
    monkeypatch.setenv("INCLUDE", str(tmp_path))
    summary = _msvc_env._toolchain_summary()
    assert "include dirs=2" in summary
    assert "missing headers=vcruntime.h" in summary
    assert "INCLUDE=set" in summary


def test_toolchain_summary_never_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton.runtime.build", None)
    monkeypatch.setitem(sys.modules, "triton.windows_utils", None)
    assert "compiler=unknown" in _msvc_env._toolchain_summary()


def test_gate_survives_a_probe_that_raises(monkeypatch):
    """This gate exists to stop a crash at worker startup; it must not become one."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "")
    monkeypatch.delenv("TORCHDYNAMO_DISABLE")
    monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))

    def boom():
        raise RuntimeError("sysconfig has no platlib on this scheme")

    monkeypatch.setattr(_msvc_env, "crt_headers_reachable", boom)
    _msvc_env.gate_torch_compile_on_windows(logging.getLogger("test_gate_7595"))
    assert "TORCHDYNAMO_DISABLE" not in os.environ


def _gate(monkeypatch, *, triton_importable, headers_ok):
    monkeypatch.setattr(sys, "platform", "win32")
    # setenv first: delenv(raising = False) records nothing when absent, so the gate's write leaks.
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "")
    monkeypatch.delenv("TORCHDYNAMO_DISABLE")
    # A None entry in sys.modules makes `import triton` raise ImportError.
    monkeypatch.setitem(
        sys.modules, "triton", types.ModuleType("triton") if triton_importable else None
    )
    monkeypatch.setattr(_msvc_env, "crt_headers_reachable", lambda: headers_ok)
    records = []
    logger = logging.getLogger("test_gate_7595")
    monkeypatch.setattr(logger, "warning", lambda msg, *a: records.append(("warning", msg)))
    monkeypatch.setattr(logger, "info", lambda msg, *a: records.append(("info", msg)))
    _msvc_env.gate_torch_compile_on_windows(logger)
    return records


def test_gate_is_noop_off_win32(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "")
    monkeypatch.delenv("TORCHDYNAMO_DISABLE")
    monkeypatch.setattr(
        _msvc_env,
        "crt_headers_reachable",
        lambda: (_ for _ in ()).throw(AssertionError("gate ran off win32")),
    )
    _msvc_env.gate_torch_compile_on_windows(logging.getLogger("test_gate_7595"))
    assert "TORCHDYNAMO_DISABLE" not in os.environ


def test_gate_disables_when_triton_missing(monkeypatch):
    records = _gate(monkeypatch, triton_importable = False, headers_ok = True)
    assert os.environ["TORCHDYNAMO_DISABLE"] == "1"
    assert any("Triton not found" in msg for _, msg in records)


def test_gate_disables_when_headers_unreachable(monkeypatch):
    records = _gate(monkeypatch, triton_importable = True, headers_ok = False)
    assert os.environ["TORCHDYNAMO_DISABLE"] == "1"
    assert any("'stdlib.h'" in msg for _, msg in records)


def test_gate_enables_when_triton_and_headers_present(monkeypatch):
    records = _gate(monkeypatch, triton_importable = True, headers_ok = True)
    assert "TORCHDYNAMO_DISABLE" not in os.environ
    assert records == [("info", "Triton available — torch.compile enabled")]


def test_gate_does_not_disable_where_triton_already_compiles(monkeypatch, tmp_path):
    """Visual Studio installed, INCLUDE unset: judging on INCLUDE alone would gate a working box."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "")
    monkeypatch.delenv("TORCHDYNAMO_DISABLE")
    monkeypatch.delenv("INCLUDE", raising = False)
    _fake_triton(monkeypatch, _sdk_dirs(tmp_path, with_toolset = True))

    _msvc_env.gate_torch_compile_on_windows(logging.getLogger("test_gate_7595"))
    assert "TORCHDYNAMO_DISABLE" not in os.environ
