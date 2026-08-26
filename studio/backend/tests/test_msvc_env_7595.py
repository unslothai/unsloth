# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Triton C-toolchain gate on Windows (#7595).

Platform-independent: Triton's discovery is stubbed through a fake
`triton.windows_utils` in sys.modules, so these run identically on CI Linux and
on a Windows box with or without Visual Studio.
"""

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
):
    """Install a stub triton.windows_utils whose search returns inc_dirs.

    `cc` is what the stubbed `get_cc()` reports, since the gate only applies to the compilers
    that need the SDK headers.
    """
    pkg = types.ModuleType("triton")
    utils = types.ModuleType("triton.windows_utils")
    utils.find_msvc_winsdk = lambda *a, **k: (None, list(inc_dirs), [])
    pkg.windows_utils = utils
    runtime = types.ModuleType("triton.runtime")
    build = types.ModuleType("triton.runtime.build")
    build.get_cc = lambda: cc
    build.is_msvc = lambda c: os.path.basename(c).lower() in ("cl", "cl.exe")
    build.is_clang_cl = lambda c: os.path.basename(c).lower() in ("clang-cl", "clang-cl.exe")
    runtime.build = build
    pkg.runtime = runtime
    monkeypatch.setitem(sys.modules, "triton", pkg)
    monkeypatch.setitem(sys.modules, "triton.windows_utils", utils)
    monkeypatch.setitem(sys.modules, "triton.runtime", runtime)
    monkeypatch.setitem(sys.modules, "triton.runtime.build", build)


# ── the INCLUDE fallback ──


def test_have_crt_headers_true_when_include_has_stdlib(tmp_path, monkeypatch):
    (tmp_path / "stdlib.h").write_text("/* stub */")
    monkeypatch.setenv("INCLUDE", str(tmp_path))
    assert _msvc_env._have_crt_headers() is True


def test_have_crt_headers_false_when_include_lacks_stdlib(tmp_path, monkeypatch):
    monkeypatch.setenv("INCLUDE", str(tmp_path))
    assert _msvc_env._have_crt_headers() is False


def test_have_crt_headers_false_when_include_unset(monkeypatch):
    monkeypatch.delenv("INCLUDE", raising = False)
    assert _msvc_env._have_crt_headers() is False


# ── Triton's own discovery, which is what the real compile uses ──


def test_triton_discovery_true_when_ucrt_dir_has_stdlib(tmp_path, monkeypatch):
    ucrt = tmp_path / "ucrt"
    ucrt.mkdir()
    (ucrt / "stdlib.h").write_text("/* stub */")
    other = tmp_path / "um"
    other.mkdir()
    _fake_triton(monkeypatch, [str(other), str(ucrt)])
    assert _msvc_env._triton_finds_crt_headers() is True


def test_triton_discovery_false_when_no_dir_has_stdlib(tmp_path, monkeypatch):
    _fake_triton(monkeypatch, [str(tmp_path)])
    assert _msvc_env._triton_finds_crt_headers() is False


def test_triton_discovery_false_when_search_returns_nothing(monkeypatch):
    """No Visual Studio at all: find_msvc_winsdk yields an empty /I list."""
    _fake_triton(monkeypatch, [])
    assert _msvc_env._triton_finds_crt_headers() is False


def test_triton_discovery_false_when_triton_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton.windows_utils", None)
    assert _msvc_env._triton_finds_crt_headers() is False


def test_triton_discovery_survives_a_raising_search(monkeypatch):
    pkg = types.ModuleType("triton")
    utils = types.ModuleType("triton.windows_utils")

    def boom(*a, **k):
        raise RuntimeError("vswhere blew up")

    utils.find_msvc_winsdk = boom
    pkg.windows_utils = utils
    monkeypatch.setitem(sys.modules, "triton", pkg)
    monkeypatch.setitem(sys.modules, "triton.windows_utils", utils)
    assert _msvc_env._triton_finds_crt_headers() is False


# ── crt_headers_reachable: the two sources combined ──


def test_reachable_is_true_off_win32(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        _msvc_env,
        "_triton_finds_crt_headers",
        lambda: (_ for _ in ()).throw(AssertionError("probed off win32")),
    )
    assert _msvc_env.crt_headers_reachable() is True


def test_reachable_via_triton_even_when_include_is_empty(tmp_path, monkeypatch):
    """The case a plain INCLUDE probe gets wrong: Triton passes explicit /I dirs."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("INCLUDE", raising = False)
    ucrt = tmp_path / "ucrt"
    ucrt.mkdir()
    (ucrt / "stdlib.h").write_text("/* stub */")
    _fake_triton(monkeypatch, [str(ucrt)])
    assert _msvc_env.crt_headers_reachable() is True


def test_reachable_via_include_when_triton_finds_nothing(tmp_path, monkeypatch):
    """Older triton-windows passes no /I and leans on the compiler's INCLUDE."""
    monkeypatch.setattr(sys, "platform", "win32")
    (tmp_path / "stdlib.h").write_text("/* stub */")
    monkeypatch.setenv("INCLUDE", str(tmp_path))
    _fake_triton(monkeypatch, [])
    assert _msvc_env.crt_headers_reachable() is True


def test_not_reachable_when_neither_source_has_headers(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("INCLUDE", str(tmp_path))  # no stdlib.h
    _fake_triton(monkeypatch, [])
    assert _msvc_env.crt_headers_reachable() is False


# ── the compiler Triton picks decides whether any of this applies ──


def test_tinycc_does_not_need_msvc_headers(tmp_path, monkeypatch):
    """The ordinary Windows NVIDIA install: no ROCm wheel, so Triton falls through to its
    bundled TinyCC, which carries its own headers and is never passed an SDK `/I`. Gating it
    on MSVC would turn off torch.compile on a machine that compiles fine."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("INCLUDE", str(tmp_path))  # no stdlib.h anywhere
    _fake_triton(monkeypatch, [], cc = "tcc.exe")
    assert _msvc_env._needs_msvc_headers() is False
    assert _msvc_env.crt_headers_reachable() is True


def test_clang_cl_does_need_msvc_headers(tmp_path, monkeypatch):
    """The AMD case: get_cc() prefers the ROCm wheel's clang-cl, which does need them."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("INCLUDE", str(tmp_path))
    _fake_triton(monkeypatch, [], cc = "clang-cl.exe")
    assert _msvc_env._needs_msvc_headers() is True
    assert _msvc_env.crt_headers_reachable() is False


def test_xpu_triton_without_the_private_api_is_not_gated(tmp_path, monkeypatch):
    """setup.ps1 swaps triton-windows for torch's pytorch-triton-xpu on Intel, and that owns the
    same top-level `triton` package without `runtime.build`'s helpers. Reading a missing private
    API as "MSVC required" would disable torch.compile on every Windows XPU install, which is a
    regression against main, where the gate was only `import triton`."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("INCLUDE", str(tmp_path))  # no stdlib.h
    monkeypatch.setitem(sys.modules, "triton.runtime.build", None)
    monkeypatch.setattr(_msvc_env, "_rocm_clang_cl_present", lambda: False)
    assert _msvc_env._needs_msvc_headers() is False
    assert _msvc_env.crt_headers_reachable() is True


def test_stale_rocm_clang_cl_under_xpu_triton_is_not_gated(tmp_path, monkeypatch):
    """A ROCm-to-XPU move under a pinned index is repaired in place (setup.ps1:4162) and does
    not prune orphans, so `_rocm_sdk_core` can outlive the Triton that would have selected it.
    The file being there is not evidence the active Triton will run it."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("INCLUDE", str(tmp_path))  # no stdlib.h
    monkeypatch.setitem(sys.modules, "triton.runtime.build", None)
    monkeypatch.setattr(_msvc_env, "_rocm_clang_cl_present", lambda: True)  # the orphan
    monkeypatch.setattr(_msvc_env, "_triton_is_triton_windows", lambda: False)  # but XPU Triton
    assert _msvc_env._needs_msvc_headers() is False
    assert _msvc_env.crt_headers_reachable() is True


def test_triton_is_triton_windows_reads_the_distribution_name(monkeypatch):
    """Resolved by distribution, the same question setup.ps1 asks when it swaps the two."""
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
    """The other half: an AMD box whose Triton does not expose the API still gets the gate,
    because get_cc() would pick the ROCm clang-cl sitting right there."""
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
    """The path is the one get_cc() builds, so a typo here silently disables the whole gate."""
    import sysconfig

    monkeypatch.setattr(sysconfig, "get_path", lambda name: str(tmp_path))
    assert _msvc_env._rocm_clang_cl_present() is False
    exe = tmp_path / "_rocm_sdk_core" / "lib" / "llvm" / "bin"
    exe.mkdir(parents = True)
    (exe / "clang-cl.exe").write_text("")
    assert _msvc_env._rocm_clang_cl_present() is True


# ── gate_torch_compile_on_windows: the arm the workers actually call ──


def test_toolchain_summary_separates_no_vs_from_a_partial_sdk(tmp_path, monkeypatch):
    """The two look identical in a bug report otherwise. Zero dirs is no Visual Studio; several
    dirs with no stdlib.h among them is an SDK missing pieces, which the gate cannot tell apart
    but a reader can once the counts are in the log."""
    _fake_triton(monkeypatch, [], cc = "clang-cl.exe")
    monkeypatch.delenv("INCLUDE", raising = False)
    summary = _msvc_env._toolchain_summary()
    assert "compiler=clang-cl.exe" in summary
    assert "include dirs=0" in summary
    assert "INCLUDE=unset" in summary

    _fake_triton(monkeypatch, [str(tmp_path), str(tmp_path)], cc = "clang-cl.exe")
    monkeypatch.setenv("INCLUDE", str(tmp_path))
    summary = _msvc_env._toolchain_summary()
    assert "include dirs=2" in summary
    assert "INCLUDE=set" in summary


def test_toolchain_summary_never_raises(monkeypatch):
    """It is evaluated as an argument to the warning, after TORCHDYNAMO_DISABLE is already set."""
    monkeypatch.setitem(sys.modules, "triton.runtime.build", None)
    monkeypatch.setitem(sys.modules, "triton.windows_utils", None)
    assert "compiler=unknown" in _msvc_env._toolchain_summary()


def test_gate_survives_a_probe_that_raises(monkeypatch):
    """The probe runs at worker startup, so an exception out of it kills the worker. This gate
    exists to stop a crash; it must not become one. Leave torch.compile alone and carry on."""
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
    """Drive the gate with a stubbed triton + header outcome; return the log."""
    monkeypatch.setattr(sys, "platform", "win32")
    # setenv first: delenv(raising = False) records nothing when the var is
    # absent, so the gate's write would leak into the rest of the session.
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
    """Visual Studio installed, INCLUDE unset: the common Windows AMD setup.

    The false-positive direction, and the one an INCLUDE-only probe gets wrong.
    Triton finds its own headers here and the compile succeeds, so a gate that
    judged on INCLUDE alone would turn off torch.compile on a working machine.
    """
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "")
    monkeypatch.delenv("TORCHDYNAMO_DISABLE")
    monkeypatch.delenv("INCLUDE", raising = False)
    ucrt = tmp_path / "ucrt"
    ucrt.mkdir()
    (ucrt / "stdlib.h").write_text("/* stub */")
    _fake_triton(monkeypatch, [str(ucrt)])

    _msvc_env.gate_torch_compile_on_windows(logging.getLogger("test_gate_7595"))
    assert "TORCHDYNAMO_DISABLE" not in os.environ
