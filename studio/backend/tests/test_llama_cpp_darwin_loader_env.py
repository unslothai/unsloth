# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the llama-server child environment, per platform.

_llama_server_env_for_binary used to branch win32 vs "everything else", and
that else branch is Linux: WSL/ROCm probing, pip nvidia wheel globs, CUDA
toolkit paths, and LD_LIBRARY_PATH. dyld ignores LD_LIBRARY_PATH, so on macOS
llama-server was launched with no library search path at all, while the
installer's own staged-binary validation sets DYLD_LIBRARY_PATH and therefore
passed on a path the real launch never took (issue #8566).

These tests pin the Darwin branch and, just as importantly, pin the Linux and
Windows branches so the fix cannot change them.
"""

from __future__ import annotations

import os
import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Match sibling tests' stubbing so the module imports without fastapi.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("structlog")
sys.modules.setdefault("structlog", _structlog_stub)
if not hasattr(sys.modules["structlog"], "get_logger"):
    sys.modules["structlog"].get_logger = _structlog_stub.get_logger

from core.inference import llama_cpp as llama_module  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402


@pytest.fixture
def binary(tmp_path):
    path = tmp_path / "llama.cpp" / "build" / "bin" / "llama-server"
    path.parent.mkdir(parents = True)
    path.write_text("")
    return path


def _env_for(binary_path):
    return LlamaCppBackend._llama_server_env_for_binary(str(binary_path))


def _no_linux_discovery(monkeypatch):
    """Make the Linux-only helpers fatal, so entering that branch is caught."""

    def _boom(*_a, **_k):
        raise AssertionError("Linux discovery ran on a non-Linux platform")

    monkeypatch.setattr(llama_module, "_wsl_system_rocm_lib_dirs", _boom)
    monkeypatch.setattr(llama_module, "_native_linux_system_rocm_lib_dirs", _boom)


class TestDarwin:
    def test_binary_dir_lands_on_dyld_library_path(self, monkeypatch, binary):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.delenv("DYLD_LIBRARY_PATH", raising = False)
        _no_linux_discovery(monkeypatch)
        env = _env_for(binary)
        assert env["DYLD_LIBRARY_PATH"].split(os.pathsep)[0] == str(binary.parent)

    def test_inherited_dyld_entries_are_kept_after_ours(self, monkeypatch, binary):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setenv("DYLD_LIBRARY_PATH", "/opt/inherited")
        _no_linux_discovery(monkeypatch)
        entries = _env_for(binary)["DYLD_LIBRARY_PATH"].split(os.pathsep)
        assert entries == [str(binary.parent), "/opt/inherited"]

    def test_our_dir_is_not_duplicated(self, monkeypatch, binary):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setenv("DYLD_LIBRARY_PATH", f"{binary.parent}{os.pathsep}/opt/inherited")
        _no_linux_discovery(monkeypatch)
        entries = _env_for(binary)["DYLD_LIBRARY_PATH"].split(os.pathsep)
        assert entries == [str(binary.parent), "/opt/inherited"]

    def test_an_inherited_ld_library_path_is_left_alone(self, monkeypatch, binary):
        # Repurposing it would be pointless (dyld ignores it) and would leak a
        # llama.cpp dir into anything the user set it for.
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/sentinel")
        _no_linux_discovery(monkeypatch)
        assert _env_for(binary)["LD_LIBRARY_PATH"] == "/opt/sentinel"

    def test_a_wrapper_entrypoint_resolves_to_the_real_bin_dir(self, monkeypatch, tmp_path):
        # The managed install can put a shell entrypoint in front of the real
        # binary; the dylibs sit next to the target, not next to the wrapper.
        # This pins the env dict only. macOS SIP purges DYLD_* when exec'ing a
        # protected binary, and a #!/bin/sh wrapper is exactly that, so on a
        # real Mac the variable would not survive into llama-server through
        # this path. The installer only writes a wrapper when it cannot make
        # the symlink (install_llama_prebuilt.create_exec_entrypoint), so the
        # mainline is unaffected; do not read this test as covering the
        # wrapper case end to end.
        real_dir = tmp_path / "llama.cpp" / "build" / "bin"
        real_dir.mkdir(parents = True)
        (real_dir / "llama-server-real").write_text("")
        wrapper = real_dir / "llama-server"
        wrapper.write_text('#!/bin/sh\nexec "$(dirname "$0")/llama-server-real" "$@"\n')
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.delenv("DYLD_LIBRARY_PATH", raising = False)
        _no_linux_discovery(monkeypatch)
        env = LlamaCppBackend._llama_server_env_for_binary(str(wrapper))
        assert env["DYLD_LIBRARY_PATH"].split(os.pathsep)[0] == str(real_dir)

    def test_loader_path_var_matches_the_sibling_engines(self, monkeypatch):
        for platform, expected in (
            ("darwin", "DYLD_LIBRARY_PATH"),
            ("win32", "PATH"),
            ("linux", "LD_LIBRARY_PATH"),
        ):
            monkeypatch.setattr(sys, "platform", platform)
            assert llama_module._loader_path_var() == expected


class TestLinuxUnchanged:
    def test_binary_dir_still_lands_on_ld_library_path(self, monkeypatch, binary):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("LD_LIBRARY_PATH", raising = False)
        monkeypatch.setattr(llama_module, "_wsl_system_rocm_lib_dirs", lambda: [])
        monkeypatch.setattr(llama_module, "_native_linux_system_rocm_lib_dirs", lambda _d: [])
        env = _env_for(binary)
        assert env["LD_LIBRARY_PATH"].split(":")[0] == str(binary.parent)
        assert "DYLD_LIBRARY_PATH" not in env

    def test_inherited_entries_stay_last(self, monkeypatch, binary):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/inherited")
        monkeypatch.setattr(llama_module, "_wsl_system_rocm_lib_dirs", lambda: [])
        monkeypatch.setattr(llama_module, "_native_linux_system_rocm_lib_dirs", lambda _d: [])
        assert _env_for(binary)["LD_LIBRARY_PATH"].split(":")[-1] == "/opt/inherited"

    def test_system_rocm_still_precedes_the_bundle(self, monkeypatch, binary):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("LD_LIBRARY_PATH", raising = False)
        monkeypatch.setattr(llama_module, "_wsl_system_rocm_lib_dirs", lambda: ["/wsl/rocm"])
        monkeypatch.setattr(llama_module, "_native_linux_system_rocm_lib_dirs", lambda _d: [])
        env = _env_for(binary)
        assert env["LD_LIBRARY_PATH"].split(":")[0] == "/wsl/rocm"
        assert env.get("HSA_ENABLE_DXG_DETECTION") == "1"


class TestWindowsUnchanged:
    def test_path_is_semicolon_joined_and_no_unix_vars_appear(self, monkeypatch, binary):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("PATH", "C:\\existing")
        monkeypatch.delenv("LD_LIBRARY_PATH", raising = False)
        monkeypatch.delenv("DYLD_LIBRARY_PATH", raising = False)
        _no_linux_discovery(monkeypatch)
        env = _env_for(binary)
        assert env["PATH"].startswith(f"{binary.parent};")
        assert env["PATH"].endswith("C:\\existing")
        assert "LD_LIBRARY_PATH" not in env
        assert "DYLD_LIBRARY_PATH" not in env
