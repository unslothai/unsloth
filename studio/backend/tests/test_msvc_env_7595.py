# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MSVC-env gate for Triton on Windows (#7595).

Platform-independent unit coverage: the header probe reads INCLUDE, and the
public entrypoint is a no-op off win32 and when the CRT headers are already
present. The vcvarsall discovery/import path needs a real VS install, so it is
exercised only through its observable outcomes here.
"""

import os
import sys

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
    monkeypatch.setattr(_msvc_env, "_find_vcvarsall",
                        lambda: (_ for _ in ()).throw(AssertionError("discovery ran off win32")))
    assert _msvc_env.ensure_msvc_env_for_triton() is True


def test_ensure_true_when_headers_already_present(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    (tmp_path / "stdlib.h").write_text("/* stub */")
    monkeypatch.setenv("INCLUDE", str(tmp_path))
    monkeypatch.setattr(_msvc_env, "_find_vcvarsall",
                        lambda: (_ for _ in ()).throw(AssertionError("discovery ran with headers present")))
    assert _msvc_env.ensure_msvc_env_for_triton() is True


def test_ensure_false_when_no_vs_found(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("INCLUDE", str(tmp_path))  # no stdlib.h
    monkeypatch.setattr(_msvc_env, "_find_vcvarsall", lambda: None)
    assert _msvc_env.ensure_msvc_env_for_triton() is False


def test_ensure_imports_env_then_rechecks(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    empty = tmp_path / "empty"; empty.mkdir()
    crt = tmp_path / "crt"; crt.mkdir(); (crt / "stdlib.h").write_text("/* stub */")
    monkeypatch.setenv("INCLUDE", str(empty))
    monkeypatch.setattr(_msvc_env, "_find_vcvarsall", lambda: r"C:\fake\vcvarsall.bat")

    def fake_import(vcvarsall, arch = "x64"):
        os.environ["INCLUDE"] = str(crt)  # simulate vcvarsall populating INCLUDE
        return True

    monkeypatch.setattr(_msvc_env, "_import_vcvars_env", fake_import)
    assert _msvc_env.ensure_msvc_env_for_triton() is True
