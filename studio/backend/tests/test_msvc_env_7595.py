# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MSVC-env gate for Triton on Windows (#7595).

Platform-independent unit coverage: the header probe reads INCLUDE, and the
public entrypoints are no-ops off win32 and when the CRT headers are already
present. The vcvarsall discovery/import path needs a real VS install, so it is
exercised only through its observable outcomes here. The worker-facing gate is
driven with a stubbed `triton` in sys.modules.
"""

import os
import sys
import types
import logging
import subprocess

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from core import _msvc_env


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


def test_ensure_is_noop_off_win32(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    # Must not touch discovery on non-Windows.
    monkeypatch.setattr(
        _msvc_env,
        "_find_vcvarsall",
        lambda: (_ for _ in ()).throw(AssertionError("discovery ran off win32")),
    )
    assert _msvc_env.ensure_msvc_env_for_triton() is True


def test_ensure_true_when_headers_already_present(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    (tmp_path / "stdlib.h").write_text("/* stub */")
    monkeypatch.setenv("INCLUDE", str(tmp_path))
    monkeypatch.setattr(
        _msvc_env,
        "_find_vcvarsall",
        lambda: (_ for _ in ()).throw(AssertionError("discovery ran with headers present")),
    )
    assert _msvc_env.ensure_msvc_env_for_triton() is True


def test_ensure_false_when_no_vs_found(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("INCLUDE", str(tmp_path))  # no stdlib.h
    monkeypatch.setattr(_msvc_env, "_find_vcvarsall", lambda: None)
    assert _msvc_env.ensure_msvc_env_for_triton() is False


def test_ensure_imports_env_then_rechecks(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    empty = tmp_path / "empty"
    empty.mkdir()
    crt = tmp_path / "crt"
    crt.mkdir()
    (crt / "stdlib.h").write_text("/* stub */")
    monkeypatch.setenv("INCLUDE", str(empty))
    monkeypatch.setattr(_msvc_env, "_find_vcvarsall", lambda: r"C:\fake\vcvarsall.bat")

    def fake_import(vcvarsall, arch = "x64"):
        os.environ["INCLUDE"] = str(crt)  # simulate vcvarsall populating INCLUDE
        return True

    monkeypatch.setattr(_msvc_env, "_import_vcvars_env", fake_import)
    assert _msvc_env.ensure_msvc_env_for_triton() is True


# ── _find_vcvarsall: discovery order ──


def _fake_fs(monkeypatch, present):
    """Make os.path.isfile answer True for exactly `present` (case-insensitive)."""
    want = {p.lower() for p in present}
    monkeypatch.setattr(_msvc_env.os.path, "isfile", lambda p: str(p).lower() in want)


def test_find_prefers_vswhere_result(monkeypatch):
    vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    found = r"D:\VS\Any\VC\Auxiliary\Build\vcvarsall.bat"
    monkeypatch.setenv("ProgramFiles(x86)", r"C:\Program Files (x86)")
    monkeypatch.setenv("ProgramFiles", r"C:\Program Files")
    _fake_fs(monkeypatch, [vswhere, found])
    monkeypatch.setattr(
        _msvc_env.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(stdout = "D:\\VS\\Any\n", returncode = 0),
    )
    monkeypatch.setattr(_msvc_env.glob, "glob", lambda pattern: [])
    assert _msvc_env._find_vcvarsall() == found


def test_find_falls_back_to_scan_when_vswhere_absent(monkeypatch):
    monkeypatch.setenv("ProgramFiles(x86)", r"C:\PFx86")
    monkeypatch.setenv("ProgramFiles", r"C:\PF")
    newest = os.path.join(
        r"C:\PF", "Microsoft Visual Studio", "18", "BuildTools", "VC", "Auxiliary", "Build",
        "vcvarsall.bat",
    )
    older = os.path.join(
        r"C:\PF", "Microsoft Visual Studio", "2019", "Community", "VC", "Auxiliary", "Build",
        "vcvarsall.bat",
    )
    _fake_fs(monkeypatch, [newest, older])
    monkeypatch.setattr(_msvc_env.glob, "glob", lambda pattern: [])
    # 18 (VS 2026) is first in _VS_YEAR_DIRS, so it must win over 2019.
    assert _msvc_env._find_vcvarsall() == newest


def test_find_returns_none_when_nothing_installed(monkeypatch):
    monkeypatch.setenv("ProgramFiles(x86)", r"C:\PFx86")
    monkeypatch.setenv("ProgramFiles", r"C:\PF")
    _fake_fs(monkeypatch, [])
    monkeypatch.setattr(_msvc_env.glob, "glob", lambda pattern: [])
    assert _msvc_env._find_vcvarsall() is None


def test_vs_sort_key_ranks_vs2026_ahead_of_2022():
    """Reverse-lexicographic would invert this: "2022" > "18" as strings."""
    p18 = os.path.join(r"C:\PF", "Microsoft Visual Studio", "18", "BuildTools", "vcvarsall.bat")
    p22 = os.path.join(r"C:\PF", "Microsoft Visual Studio", "2022", "BuildTools", "vcvarsall.bat")
    assert sorted([p22, p18], key = _msvc_env._vs_sort_key) == [p18, p22]


def test_vs_sort_key_puts_unknown_layout_last():
    known = os.path.join(r"C:\PF", "Microsoft Visual Studio", "2019", "X", "vcvarsall.bat")
    unknown = os.path.join(r"C:\PF", "Microsoft Visual Studio", "weird", "X", "vcvarsall.bat")
    assert sorted([unknown, known], key = _msvc_env._vs_sort_key) == [known, unknown]


# ── _import_vcvars_env: parsing and failure handling ──


def test_import_carries_only_toolchain_keys(monkeypatch):
    dumped = "\n".join(
        [
            "INCLUDE=C:\\crt",
            "LIB=C:\\lib",
            "LIBPATH=C:\\libpath",
            "PATH=C:\\vc;C:\\windows",
            "UNRELATED=should-not-be-copied",
            "not a key-value line",
        ]
    )
    monkeypatch.setattr(
        _msvc_env.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(stdout = dumped, stderr = "", returncode = 0),
    )
    for key in (*_msvc_env._CARRY, "UNRELATED"):
        monkeypatch.delenv(key, raising = False)

    assert _msvc_env._import_vcvars_env(r"C:\fake\vcvarsall.bat") is True
    assert os.environ["INCLUDE"] == "C:\\crt"
    assert os.environ["LIB"] == "C:\\lib"
    assert os.environ["LIBPATH"] == "C:\\libpath"
    assert os.environ["PATH"] == "C:\\vc;C:\\windows"
    assert "UNRELATED" not in os.environ


def test_import_decodes_with_console_codepage(monkeypatch):
    """cmd emits OEM, not UTF-8; decoding it as UTF-8 corrupts non-ASCII paths to U+FFFD."""
    seen = {}

    def fake_run(*a, **k):
        seen.update(k)
        return types.SimpleNamespace(stdout = "INCLUDE=C:\\crt", stderr = "", returncode = 0)

    monkeypatch.setattr(_msvc_env.subprocess, "run", fake_run)
    monkeypatch.delenv("INCLUDE", raising = False)
    _msvc_env._import_vcvars_env(r"C:\fake\vcvarsall.bat")
    assert seen["encoding"] == _msvc_env._CONSOLE_ENCODING
    if sys.platform == "win32":
        assert seen["encoding"] == "oem"


def test_import_returns_false_and_logs_reason_on_nonzero_exit(monkeypatch, caplog):
    monkeypatch.setattr(
        _msvc_env.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(
            stdout = "", stderr = "ERROR: missing the VC tools\n", returncode = 1
        ),
    )
    with caplog.at_level(logging.WARNING, logger = _msvc_env.__name__):
        assert _msvc_env._import_vcvars_env(r"C:\fake\vcvarsall.bat") is False
    assert "missing the VC tools" in caplog.text


def test_import_returns_false_when_subprocess_raises(monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd = "cmd", timeout = 120)

    monkeypatch.setattr(_msvc_env.subprocess, "run", boom)
    assert _msvc_env._import_vcvars_env(r"C:\fake\vcvarsall.bat") is False


def test_ensure_false_when_env_import_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("INCLUDE", str(tmp_path))  # no stdlib.h
    monkeypatch.setattr(_msvc_env, "_find_vcvarsall", lambda: r"C:\fake\vcvarsall.bat")
    monkeypatch.setattr(_msvc_env, "_import_vcvars_env", lambda *a, **k: False)
    assert _msvc_env.ensure_msvc_env_for_triton() is False


# ── gate_torch_compile_on_windows: the arm the workers actually call ──


def _gate(monkeypatch, *, triton_importable, msvc_ok):
    """Drive the gate with a stubbed triton + MSVC outcome; return the log."""
    monkeypatch.setattr(sys, "platform", "win32")
    # setenv first: delenv(raising = False) records nothing when the var is
    # absent, so the gate's write would leak into the rest of the session.
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "")
    monkeypatch.delenv("TORCHDYNAMO_DISABLE")
    # A None entry in sys.modules makes `import triton` raise ImportError.
    monkeypatch.setitem(
        sys.modules, "triton", types.ModuleType("triton") if triton_importable else None
    )
    monkeypatch.setattr(_msvc_env, "ensure_msvc_env_for_triton", lambda: msvc_ok)
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
        "ensure_msvc_env_for_triton",
        lambda: (_ for _ in ()).throw(AssertionError("gate ran off win32")),
    )
    _msvc_env.gate_torch_compile_on_windows(logging.getLogger("test_gate_7595"))
    assert "TORCHDYNAMO_DISABLE" not in os.environ


def test_gate_disables_when_triton_missing(monkeypatch):
    records = _gate(monkeypatch, triton_importable = False, msvc_ok = True)
    assert os.environ["TORCHDYNAMO_DISABLE"] == "1"
    assert any("Triton not found" in msg for _, msg in records)


def test_gate_disables_when_msvc_missing(monkeypatch):
    records = _gate(monkeypatch, triton_importable = True, msvc_ok = False)
    assert os.environ["TORCHDYNAMO_DISABLE"] == "1"
    assert any("no MSVC toolchain" in msg for _, msg in records)


def test_gate_enables_when_triton_and_msvc_present(monkeypatch):
    records = _gate(monkeypatch, triton_importable = True, msvc_ok = True)
    assert "TORCHDYNAMO_DISABLE" not in os.environ
    assert records == [("info", "Triton available — torch.compile enabled")]


# ── the invocation form itself: this is what silently broke the whole fix ──


def test_import_invokes_cmd_as_a_string_not_a_list(monkeypatch):
    """Regression: a list arg sends list2cmdline's \\" escaping to cmd.

    Every real VS lives under "Program Files", so the quoted path is not
    optional, and as a list element cmd receives \\"C:\\Program Files...\\" and
    reports "is not recognized as an internal or external command". The import
    then fails on EVERY machine and the gate wrongly reports "no MSVC toolchain".
    """
    seen = {}

    def fake_run(cmd, *a, **k):
        seen["cmd"] = cmd
        seen["shell"] = k.get("shell")
        return types.SimpleNamespace(stdout = "INCLUDE=C:\\crt", stderr = "", returncode = 0)

    monkeypatch.setattr(_msvc_env.subprocess, "run", fake_run)
    monkeypatch.delenv("INCLUDE", raising = False)
    _msvc_env._import_vcvars_env(r"C:\Program Files\VS\vcvarsall.bat")

    assert isinstance(seen["cmd"], str), "must be a string; a list gets quote-escaped"
    assert seen["shell"] is True
    assert '"C:\\Program Files\\VS\\vcvarsall.bat"' in seen["cmd"]


@pytest.mark.skipif(sys.platform != "win32", reason = "needs a real Windows VS install")
def test_import_actually_populates_include_on_this_machine():
    """End-to-end against the real vcvarsall, when this box has one.

    The unit tests all stub the subprocess, so only this one can catch the
    invocation being malformed. Skips where no VS is installed.
    """
    vcvarsall = _msvc_env._find_vcvarsall()
    if not vcvarsall:
        pytest.skip("no Visual Studio / Build Tools install on this machine")

    before = os.environ.get("INCLUDE")
    try:
        assert _msvc_env._import_vcvars_env(vcvarsall) is True
        assert _msvc_env._have_crt_headers(), "vcvarsall ran but INCLUDE has no stdlib.h"
    finally:
        if before is None:
            os.environ.pop("INCLUDE", None)
        else:
            os.environ["INCLUDE"] = before
