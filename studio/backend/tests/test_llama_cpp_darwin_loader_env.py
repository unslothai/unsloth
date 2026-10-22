# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""dyld ignores LD_LIBRARY_PATH, so macOS children need DYLD_LIBRARY_PATH; others unchanged."""

from __future__ import annotations

import os
import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

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
        # dyld ignores LD_LIBRARY_PATH, so it must not be repurposed on macOS.
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/sentinel")
        _no_linux_discovery(monkeypatch)
        assert _env_for(binary)["LD_LIBRARY_PATH"] == "/opt/sentinel"

    def test_a_wrapper_entrypoint_resolves_to_the_real_bin_dir(self, monkeypatch, tmp_path):
        # Dylibs sit next to the wrapper's target; SIP purges DYLD_* under /bin/sh,
        # so the env dict alone is not enough on macOS.
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
        # startswith, not split(':'): on a Windows host the first entry is 'C:\...'.
        assert env["LD_LIBRARY_PATH"].startswith(str(binary.parent))
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

    def test_use_system_rocm_false_skips_native_linux_prepend(self, monkeypatch, binary):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("LD_LIBRARY_PATH", raising = False)
        monkeypatch.setattr(llama_module, "_wsl_system_rocm_lib_dirs", lambda: [])
        monkeypatch.setattr(
            llama_module, "_native_linux_system_rocm_lib_dirs", lambda _d: ["/opt/rocm/lib"]
        )
        with_system = LlamaCppBackend._llama_server_env_for_binary(str(binary))
        bundle_only = LlamaCppBackend._llama_server_env_for_binary(
            str(binary), use_system_rocm = False
        )
        assert with_system["LD_LIBRARY_PATH"].split(":")[0] == "/opt/rocm/lib"
        assert "/opt/rocm/lib" not in bundle_only["LD_LIBRARY_PATH"].split(":")
        assert bundle_only["LD_LIBRARY_PATH"].startswith(str(binary.parent))

    def test_a_proved_bundle_only_host_stops_prepending_for_every_child(self, monkeypatch, binary):
        # Latch so the STT sidecar (same helper) stops reusing the disproved prepend.
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("LD_LIBRARY_PATH", raising = False)
        monkeypatch.setattr(llama_module, "_wsl_system_rocm_lib_dirs", lambda: [])
        monkeypatch.setattr(
            llama_module, "_native_linux_system_rocm_lib_dirs", lambda _d: ["/opt/rocm/lib"]
        )
        monkeypatch.setattr(LlamaCppBackend, "_bundle_only_rocm_dirs", {})
        before = LlamaCppBackend._llama_server_env_for_binary(str(binary))
        assert before["LD_LIBRARY_PATH"].split(":")[0] == "/opt/rocm/lib"

        LlamaCppBackend._remember_bundle_only_rocm(str(binary))

        after = LlamaCppBackend._llama_server_env_for_binary(str(binary))
        assert "/opt/rocm/lib" not in after["LD_LIBRARY_PATH"].split(":")
        from core.inference.stt_mtmd_sidecar import _llama_server_child_env

        sidecar = _llama_server_child_env(str(binary))
        assert "/opt/rocm/lib" not in sidecar["LD_LIBRARY_PATH"].split(":")

    def test_a_replaced_runtime_retests_the_system_rocm_prepend(
        self, monkeypatch, binary, tmp_path
    ):
        # The updater swaps a new install into the same path; old proof must not stick.
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("LD_LIBRARY_PATH", raising = False)
        monkeypatch.setattr(llama_module, "_wsl_system_rocm_lib_dirs", lambda: [])
        monkeypatch.setattr(
            llama_module, "_native_linux_system_rocm_lib_dirs", lambda _d: ["/opt/rocm/lib"]
        )
        monkeypatch.setattr(LlamaCppBackend, "_bundle_only_rocm_dirs", {})
        LlamaCppBackend._remember_bundle_only_rocm(str(binary))
        assert LlamaCppBackend._prefers_bundle_only_rocm(str(binary))

        replacement = tmp_path / "replacement-llama-server"
        replacement.write_bytes(b"new runtime binary")
        os.replace(replacement, binary)

        assert not LlamaCppBackend._prefers_bundle_only_rocm(str(binary))
        env = LlamaCppBackend._llama_server_env_for_binary(str(binary))
        assert env["LD_LIBRARY_PATH"].split(":")[0] == "/opt/rocm/lib"

    def test_a_different_build_dir_still_gets_the_prepend(self, monkeypatch, binary, tmp_path):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("LD_LIBRARY_PATH", raising = False)
        monkeypatch.setattr(llama_module, "_wsl_system_rocm_lib_dirs", lambda: [])
        monkeypatch.setattr(
            llama_module, "_native_linux_system_rocm_lib_dirs", lambda _d: ["/opt/rocm/lib"]
        )
        monkeypatch.setattr(LlamaCppBackend, "_bundle_only_rocm_dirs", {})
        LlamaCppBackend._remember_bundle_only_rocm(str(binary))
        other = tmp_path / "other" / "llama-server"
        other.parent.mkdir(parents = True)
        other.write_bytes(b"binary")
        env = LlamaCppBackend._llama_server_env_for_binary(str(other))
        assert env["LD_LIBRARY_PATH"].split(":")[0] == "/opt/rocm/lib"

    def test_use_system_rocm_false_keeps_the_wsl_prepend(self, monkeypatch, binary):
        # librocdxg is WSL; the native-Linux flag must not drop it.
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("LD_LIBRARY_PATH", raising = False)
        monkeypatch.setattr(llama_module, "_wsl_system_rocm_lib_dirs", lambda: ["/wsl/rocm"])
        monkeypatch.setattr(
            llama_module, "_native_linux_system_rocm_lib_dirs", lambda _d: ["/opt/rocm/lib"]
        )
        env = LlamaCppBackend._llama_server_env_for_binary(str(binary), use_system_rocm = False)
        parts = env["LD_LIBRARY_PATH"].split(":")
        assert parts[0] == "/wsl/rocm"
        assert "/opt/rocm/lib" not in parts
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


class TestDarwinSpawnsTheResolvedBinary:
    """macOS SIP purges DYLD_* while starting the protected /bin/sh that a
    wrapper entrypoint runs under, so the loader path we build would not
    survive the wrapper's own exec. load_model resolves the entrypoint first."""

    def test_resolve_llama_binary_follows_the_wrapper(self, tmp_path):
        real_dir = tmp_path / "build" / "bin"
        real_dir.mkdir(parents = True)
        target = real_dir / "llama-server-real"
        target.write_text("")
        wrapper = real_dir / "llama-server"
        wrapper.write_text('#!/bin/sh\nexec "$(dirname "$0")/llama-server-real" "$@"\n')
        assert llama_module._resolve_llama_binary(str(wrapper)) == target

    def test_a_plain_binary_is_returned_unchanged(self, binary):
        assert llama_module._resolve_llama_binary(str(binary)) == binary.resolve()


class TestExecPathForLaunch:
    """Only OUR entrypoint is resolved past. A user's LLAMA_SERVER_PATH wrapper
    may export backend variables before its exec line, and jumping straight to
    the target would silently drop that setup."""

    def _wrapper(self, tmp_path):
        real_dir = tmp_path / "llama.cpp" / "build" / "bin"
        real_dir.mkdir(parents = True)
        (real_dir / "llama-server-real").write_text("")
        wrapper = real_dir / "llama-server"
        wrapper.write_text('#!/bin/sh\nexec "$(dirname "$0")/llama-server-real" "$@"\n')
        return wrapper, real_dir / "llama-server-real"

    def test_a_managed_entrypoint_is_resolved(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
        wrapper, target = self._wrapper(tmp_path)
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(tmp_path / "llama.cpp"))
        assert LlamaCppBackend._exec_path_for_launch(str(wrapper)) == str(target)

    def test_a_selected_installer_entrypoint_still_resolves_by_shape(self, monkeypatch, tmp_path):
        from utils import llama_cpp_path_settings as path_settings

        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(path_settings, "custom_llama_cpp_path_source", lambda: "studio")
        wrapper, target = self._wrapper(tmp_path)
        # LF-only template even on a Windows host.
        wrapper.write_bytes(b'#!/bin/sh\nexec "$(dirname "$0")/llama-server-real" "$@"\n')

        assert LlamaCppBackend._is_unsloth_managed_binary(str(wrapper)) is False
        assert LlamaCppBackend._exec_path_for_launch(str(wrapper)) == str(target)

    def test_a_pinned_custom_wrapper_is_launched_as_given(self, monkeypatch, tmp_path):
        wrapper, _ = self._wrapper(tmp_path)
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setenv("LLAMA_SERVER_PATH", str(wrapper))
        assert LlamaCppBackend._exec_path_for_launch(str(wrapper)) == str(wrapper)

    def test_other_platforms_are_untouched(self, monkeypatch, tmp_path):
        wrapper, _ = self._wrapper(tmp_path)
        for platform in ("linux", "win32"):
            monkeypatch.setattr(sys, "platform", platform)
            assert LlamaCppBackend._exec_path_for_launch(str(wrapper)) == str(wrapper)

    def test_no_binary_is_passed_through(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        assert LlamaCppBackend._exec_path_for_launch(None) is None


class TestBinaryRevisionPathSpace:
    """_binary_revision keys on the path string, so both sides must resolve the entrypoint the same way."""

    def _managed_wrapper(self, monkeypatch, tmp_path):
        root = tmp_path / "llama.cpp"
        real_dir = root / "build" / "bin"
        real_dir.mkdir(parents = True)
        (real_dir / "llama-server-real").write_text("real")
        wrapper = root / "llama-server"
        wrapper.write_text('#!/bin/sh\nexec "$(dirname "$0")/build/bin/llama-server-real" "$@"\n')
        monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(root))
        monkeypatch.setattr(sys, "platform", "darwin")
        return wrapper

    def test_a_missing_baseline_does_not_rediscover_the_binary(self, monkeypatch):
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._launch_binary_revision = ()

        # *args/**kwargs, or a caller passing include_denied fails on the signature.
        def _unexpected_discovery(*args, **kwargs):  # pragma: no cover - must never run
            raise AssertionError("binary discovery ran without a revision to compare")

        monkeypatch.setattr(backend, "_find_llama_server_binary", _unexpected_discovery)
        assert backend._binary_changed_since_launch() is False

    def test_an_unchanged_managed_install_does_not_look_updated(self, monkeypatch, tmp_path):
        wrapper = self._managed_wrapper(monkeypatch, tmp_path)
        monkeypatch.setattr(
            LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda **_k: str(wrapper))
        )
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._launch_binary_revision = LlamaCppBackend._binary_revision(
            LlamaCppBackend._exec_path_for_launch(str(wrapper))
        )
        assert backend._launch_binary_revision
        assert backend._binary_changed_since_launch() is False

    def test_a_real_update_is_still_detected(self, monkeypatch, tmp_path):
        wrapper = self._managed_wrapper(monkeypatch, tmp_path)
        monkeypatch.setattr(
            LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda **_k: str(wrapper))
        )
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._launch_binary_revision = LlamaCppBackend._binary_revision(
            LlamaCppBackend._exec_path_for_launch(str(wrapper))
        )
        target = tmp_path / "llama.cpp" / "build" / "bin" / "llama-server-real"
        target.write_text("a different build entirely")
        assert backend._binary_changed_since_launch() is True


class TestLocalLinkInstalls:
    """--with-llama-cpp-dir points the canonical llama.cpp dir at the user's own
    checkout. The updater refuses to write through that link, so the install is
    theirs: no updater advice, and no resolving past their entrypoint."""

    def _local_link_tree(self, tmp_path, monkeypatch):
        external = tmp_path / "my-llama-checkout"
        bin_dir = external / "build" / "bin"
        bin_dir.mkdir(parents = True)
        (bin_dir / "llama-server-real").write_text("")
        wrapper = external / "llama-server"
        wrapper.write_text(
            '#!/bin/sh\nexport MY_BACKEND=1\nexec "$(dirname "$0")/build/bin/llama-server-real" "$@"\n'
        )
        studio_root = tmp_path / "studio"
        studio_root.mkdir()
        link = studio_root / "llama.cpp"
        link.symlink_to(external)
        monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(link))
        return link / "llama-server"

    def test_a_local_link_is_not_reported_as_managed(self, monkeypatch, tmp_path):
        binary = self._local_link_tree(tmp_path, monkeypatch)
        assert LlamaCppBackend._is_unsloth_managed_binary(str(binary)) is False

    def test_the_users_entrypoint_is_launched_as_given(self, monkeypatch, tmp_path):
        binary = self._local_link_tree(tmp_path, monkeypatch)
        monkeypatch.setattr(sys, "platform", "darwin")
        assert LlamaCppBackend._exec_path_for_launch(str(binary)) == str(binary)

    def test_the_remedy_does_not_send_them_to_the_updater(self, monkeypatch, tmp_path):
        binary = self._local_link_tree(tmp_path, monkeypatch)
        remedy = LlamaCppBackend._runtime_remedy(str(binary))
        assert "unsloth studio update" not in remedy
        assert "custom llama.cpp" in remedy


class TestTheLoaderPathPrependIsAFixedPoint:
    """Building the child env again on retry must not prepend the runtime dir a second time."""

    def test_the_binary_dir_comes_first(self):
        assert llama_module._prepend_loader_dir("", "/x/bin") == "/x/bin"
        assert llama_module._prepend_loader_dir("/a", "/x/bin") == f"/x/bin{os.pathsep}/a"

    def test_repeating_it_changes_nothing(self):
        once = llama_module._prepend_loader_dir("/a", "/x/bin")
        assert llama_module._prepend_loader_dir(once, "/x/bin") == once
        assert llama_module._prepend_loader_dir(once, "/x/bin") == once

    @pytest.mark.parametrize("spelling", ["/x/bin/", "/x/./bin", "/x/y/../bin", "/x//bin"])
    def test_the_same_dir_spelled_differently_is_not_kept_twice(self, spelling):
        got = llama_module._prepend_loader_dir(spelling, "/x/bin")
        assert got == "/x/bin"

    def test_an_unrelated_entry_survives_with_its_own_spelling(self):
        got = llama_module._prepend_loader_dir(f"/opt/other/{os.pathsep}/x/bin/", "/x/bin")
        assert got == f"/x/bin{os.pathsep}/opt/other/"

    def test_empty_segments_are_dropped(self):
        got = llama_module._prepend_loader_dir(f"{os.pathsep}{os.pathsep}/a{os.pathsep}", "/x/bin")
        assert got == f"/x/bin{os.pathsep}/a"

    def test_building_the_env_twice_gives_the_same_value(self, monkeypatch, tmp_path):
        binary = tmp_path / "build" / "bin" / "llama-server"
        binary.parent.mkdir(parents = True)
        binary.write_bytes(b"\xcf\xfa\xed\xfe")
        monkeypatch.setattr(sys, "platform", "darwin")
        first = LlamaCppBackend._llama_server_env_for_binary(str(binary))
        monkeypatch.setenv("DYLD_LIBRARY_PATH", first["DYLD_LIBRARY_PATH"])
        second = LlamaCppBackend._llama_server_env_for_binary(str(binary))
        assert second["DYLD_LIBRARY_PATH"] == first["DYLD_LIBRARY_PATH"]


class TestAnExplicitPinOutranksInferredOwnership:
    """A wrapper pinned in LLAMA_SERVER_PATH is the user's, even inside a managed install tree."""

    @staticmethod
    def _managed_tree_with_a_pinned_wrapper(tmp_path):
        root = tmp_path / "llama.cpp"
        (root / "build" / "bin").mkdir(parents = True)
        (root / "build" / "bin" / "llama-server").write_bytes(b"\xcf\xfa\xed\xfe")
        for marker in (
            "UNSLOTH_PREBUILT_INFO.json",
            ".unsloth_llama_install",
            "unsloth_install.json",
        ):
            (root / marker).write_text("{}")
        wrapper = root / "my-wrapper"
        wrapper.write_text(
            '#!/bin/sh\nexport MY_TUNING=1\nexec "$(dirname "$0")/build/bin/llama-server" "$@"\n'
        )
        wrapper.chmod(0o755)
        return wrapper

    def test_the_pinned_wrapper_is_launched_not_its_target(self, monkeypatch, tmp_path):
        wrapper = self._managed_tree_with_a_pinned_wrapper(tmp_path)
        monkeypatch.setenv("LLAMA_SERVER_PATH", str(wrapper))
        monkeypatch.setattr(sys, "platform", "darwin")
        assert LlamaCppBackend._exec_path_for_launch(str(wrapper)) == str(wrapper)

    def test_the_managed_entrypoint_beside_it_is_still_resolved(self, monkeypatch, tmp_path):
        wrapper = self._managed_tree_with_a_pinned_wrapper(tmp_path)
        entry = wrapper.parent / "llama-server"
        entry.write_text('#!/bin/sh\nexec "$(dirname "$0")/build/bin/llama-server" "$@"\n')
        entry.chmod(0o755)
        monkeypatch.setenv("LLAMA_SERVER_PATH", str(wrapper))
        monkeypatch.setattr(sys, "platform", "darwin")
        got = LlamaCppBackend._exec_path_for_launch(str(entry))
        assert got == str(wrapper.parent / "build" / "bin" / "llama-server")


class TestTheCpuFallbackGateIsUnchangedForLinkedTrees:
    """Linked trees keep the CPU fallback gate unchanged: managed-binary ownership is asked separately."""

    @staticmethod
    def _linked_tree(tmp_path, monkeypatch):
        external = tmp_path / "my-llama-checkout"
        (external / "build" / "bin").mkdir(parents = True)
        (external / "build" / "bin" / "llama-server").write_bytes(b"\xcf\xfa\xed\xfe")
        studio_root = tmp_path / "studio"
        studio_root.mkdir()
        link = studio_root / "llama.cpp"
        link.symlink_to(external)
        monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(link))
        return str(link / "build" / "bin" / "llama-server")

    def test_the_updater_cannot_repair_it(self, monkeypatch, tmp_path):
        binary = self._linked_tree(tmp_path, monkeypatch)
        assert LlamaCppBackend._is_unsloth_managed_binary(binary) is False

    def test_but_it_is_still_an_install_tree_we_can_stage_from(self, monkeypatch, tmp_path):
        binary = self._linked_tree(tmp_path, monkeypatch)
        assert LlamaCppBackend._is_llama_install_tree(binary) is True


class TestAWrapperChainIsFollowedToTheEnd:
    """Wrapper chains resolve to the end, since SIP drops DYLD_* through any intermediate shell script."""

    @staticmethod
    def _chain(tmp_path, depth):
        real_dir = tmp_path / "build" / "bin"
        real_dir.mkdir(parents = True)
        real = real_dir / "llama-server"
        real.write_bytes(b"\xcf\xfa\xed\xfe")
        target = "build/bin/llama-server"
        for i in range(depth):
            name = f"hop{i}.sh" if i < depth - 1 else "llama-server"
            p = tmp_path / name
            p.write_text(f'#!/bin/sh\nexec "$(dirname "$0")/{target}" "$@"\n')
            p.chmod(0o755)
            target = name
        return tmp_path / "llama-server", real

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_the_real_binary_is_reached(self, tmp_path, depth):
        entry, real = self._chain(tmp_path, depth)
        assert llama_module._resolve_llama_binary(str(entry)) == real.resolve()
        assert llama_module._llama_lib_dir(str(entry)) == real.parent.resolve()

    def test_a_wrapper_loop_terminates(self, tmp_path):
        a = tmp_path / "llama-server"
        b = tmp_path / "other.sh"
        a.write_text('#!/bin/sh\nexec "$(dirname "$0")/other.sh" "$@"\n')
        b.write_text('#!/bin/sh\nexec "$(dirname "$0")/llama-server" "$@"\n')
        a.chmod(0o755)
        b.chmod(0o755)
        got = llama_module._resolve_llama_binary(str(a))
        assert got in (a.resolve(), b.resolve())


class TestOnlyTheInstallersOwnEntrypointIsSkipped:
    """Only the installer's fixed three-line wrapper is skipped as ours; any other wrapper is preserved."""

    @staticmethod
    def _tree(tmp_path, wrapper_body):
        root = tmp_path / "my-llama-checkout"
        (root / "build" / "bin").mkdir(parents = True)
        (root / "build" / "bin" / "llama-server").write_bytes(b"\xcf\xfa\xed\xfe")
        entry = root / "llama-server"
        entry.write_text(wrapper_body)
        entry.chmod(0o755)
        return root, entry

    _INSTALLER = '#!/bin/sh\nexec "$(dirname "$0")/build/bin/llama-server" "$@"\n'
    _USERS = '#!/bin/sh\nexport MY_TUNING=1\nexec "$(dirname "$0")/build/bin/llama-server" "$@"\n'

    def test_a_users_wrapper_in_a_custom_dir_is_launched_as_written(self, monkeypatch, tmp_path):
        root, entry = self._tree(tmp_path, self._USERS)
        monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(root))
        monkeypatch.setattr(sys, "platform", "darwin")
        assert LlamaCppBackend._is_unsloth_managed_binary(str(entry)) is True
        assert LlamaCppBackend._exec_path_for_launch(str(entry)) == str(entry)

    def test_the_installers_own_wrapper_is_still_resolved(self, monkeypatch, tmp_path):
        root, entry = self._tree(tmp_path, self._INSTALLER)
        monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(root))
        monkeypatch.setattr(sys, "platform", "darwin")
        got = LlamaCppBackend._exec_path_for_launch(str(entry))
        assert got == str((root / "build" / "bin" / "llama-server").resolve())

    def test_the_library_dir_still_comes_from_the_target(self, monkeypatch, tmp_path):
        root, entry = self._tree(tmp_path, self._USERS)
        assert llama_module._llama_lib_dir(str(entry)) == (root / "build" / "bin").resolve()

    def test_a_symlink_entrypoint_is_ours_to_resolve(self, monkeypatch, tmp_path):
        root = tmp_path / "t"
        (root / "build" / "bin").mkdir(parents = True)
        real = root / "build" / "bin" / "llama-server"
        real.write_bytes(b"\xcf\xfa\xed\xfe")
        link = root / "llama-server"
        link.symlink_to(real)
        monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(root))
        monkeypatch.setattr(sys, "platform", "darwin")
        assert LlamaCppBackend._exec_path_for_launch(str(link)) == str(real.resolve())


class TestPinningTheInstallersOwnEntrypoint:
    """A wrapper's shape, not how it was reached, decides: templates resolve, custom wrappers are kept."""

    @staticmethod
    def _managed_tree(tmp_path, body):
        root = tmp_path / "llama.cpp"
        (root / "build" / "bin").mkdir(parents = True)
        (root / "build" / "bin" / "llama-server").write_bytes(b"\xcf\xfa\xed\xfe")
        for marker in (
            "UNSLOTH_PREBUILT_INFO.json",
            ".unsloth_llama_install",
            "unsloth_install.json",
        ):
            (root / marker).write_text("{}")
        entry = root / "llama-server"
        entry.write_text(body)
        entry.chmod(0o755)
        return root, entry

    _INSTALLER = '#!/bin/sh\nexec "$(dirname "$0")/build/bin/llama-server" "$@"\n'
    _CUSTOM = '#!/bin/sh\nexport MY_TUNING=1\nexec "$(dirname "$0")/build/bin/llama-server" "$@"\n'

    def test_a_pinned_installer_wrapper_still_resolves(self, monkeypatch, tmp_path):
        root, entry = self._managed_tree(tmp_path, self._INSTALLER)
        monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising = False)
        monkeypatch.setenv("LLAMA_SERVER_PATH", str(entry))
        monkeypatch.setattr(sys, "platform", "darwin")
        got = LlamaCppBackend._exec_path_for_launch(str(entry))
        assert got == str((root / "build" / "bin" / "llama-server").resolve())

    def test_a_pinned_custom_wrapper_is_still_preserved(self, monkeypatch, tmp_path):
        _root, entry = self._managed_tree(tmp_path, self._CUSTOM)
        monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising = False)
        monkeypatch.setenv("LLAMA_SERVER_PATH", str(entry))
        monkeypatch.setattr(sys, "platform", "darwin")
        assert LlamaCppBackend._exec_path_for_launch(str(entry)) == str(entry)


class TestLaunchStopsAtSomebodyElsesWrapper:
    """Launch stops at a foreign wrapper; the library-dir lookup still follows the whole chain."""

    @staticmethod
    def _tree(tmp_path, inner_body):
        import json

        root = tmp_path / ".unsloth" / "llama.cpp"
        (root / "build" / "bin").mkdir(parents = True)
        (root / "UNSLOTH_PREBUILT_INFO.json").write_text(json.dumps({"tag": "b9415"}))
        real = root / "build" / "bin" / "llama-server-real"
        real.write_bytes(b"\x00\x00\x00\x00")
        real.chmod(0o755)
        inner = root / "build" / "bin" / "llama-server"
        inner.write_text(inner_body)
        inner.chmod(0o755)
        outer = root / "llama-server"
        outer.write_text('#!/bin/sh\nexec "$(dirname "$0")/build/bin/llama-server" "$@"\n')
        outer.chmod(0o755)
        return outer, inner, real

    _CUSTOM = (
        "#!/bin/sh\nexport GGML_METAL_PATH_RESOURCES=/custom\n"
        'exec "$(dirname "$0")/llama-server-real" "$@"\n'
    )
    _TEMPLATE = '#!/bin/sh\nexec "$(dirname "$0")/llama-server-real" "$@"\n'

    def test_a_custom_inner_wrapper_is_launched_not_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        outer, inner, real = self._tree(tmp_path, self._CUSTOM)
        assert LlamaCppBackend._exec_path_for_launch(str(outer)) == str(inner)

    def test_an_all_template_chain_still_resolves_to_the_end(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        outer, inner, real = self._tree(tmp_path, self._TEMPLATE)
        assert LlamaCppBackend._exec_path_for_launch(str(outer)) == str(real)

    def test_the_library_directory_still_follows_the_whole_chain(self, tmp_path, monkeypatch):
        """Stopping early for launch must not move the dylib search path."""
        monkeypatch.setattr(sys, "platform", "darwin")
        outer, inner, real = self._tree(tmp_path, self._CUSTOM)
        assert llama_module._llama_lib_dir(str(outer)) == real.parent
