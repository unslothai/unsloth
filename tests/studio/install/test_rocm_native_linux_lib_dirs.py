# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Coverage for the native-Linux system-ROCm library prepend (PR #7233).

#7233 fixed the segfault-on-launch class of AMD reports (#7208, #7310, #6276 and
the native-Linux half of #7307): a prebuilt llama.cpp ships its own libggml-hip /
HIP runtime, and on a bare-metal ROCm box that bundled runtime can disagree with
the host amdkfd driver, so the server dies the moment a model is loaded. The fix
prepends the *system* ROCm lib dirs ahead of the bundle on LD_LIBRARY_PATH.

It landed as two hand-copied helpers, one in the installer (validation-time) and
one in the serve-time launcher:

  studio/install_llama_prebuilt.py  _bundled_hip_present / _native_linux_system_rocm_lib_dirs
  studio/backend/core/inference/llama_cpp.py   same two, "mirrors" comment only

and shipped with no tests at all: the WSL sibling helper added earlier has
TestWslSystemRocmLibDirs / TestBinaryEnvWslOrdering / TestLlamaCppRuntimeWslOrdering,
the native-Linux one has nothing. Every gate here is a false-positive risk that
would silently reorder LD_LIBRARY_PATH for users the fix was never meant to touch
(WSL, NVIDIA hosts, macOS, containers without /dev/kfd), so each gate gets a test,
and both copies are run against the same fake host and required to agree.

llama_cpp.py cannot be imported from the test suite (module-level structlog /
backend imports), so its two helpers are lifted out with ast and exec'd standalone.
"""

import ast
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]

_PREBUILT_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
_LLAMA_CPP_PATH = PACKAGE_ROOT / "studio" / "backend" / "core" / "inference" / "llama_cpp.py"

_HELPERS = ("_bundled_hip_present", "_native_linux_system_rocm_lib_dirs")


def _load_prebuilt_module():
    spec = importlib.util.spec_from_file_location(
        "studio_install_llama_prebuilt_native", _PREBUILT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _extract_functions(path: Path, names) -> dict:
    """exec just the named top-level functions out of a module that is too
    heavy to import."""
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    wanted = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    found = {n.name for n in wanted}
    assert found == set(names), f"{path.name}: missing {sorted(set(names) - found)}"
    module = ast.Module(body = wanted, type_ignores = [])
    ns: dict = {"os": os, "sys": sys, "Path": Path}
    exec(compile(module, str(path), "exec"), ns)
    return ns


prebuilt_mod = _load_prebuilt_module()
llama_ns = _extract_functions(_LLAMA_CPP_PATH, _HELPERS)


def _impls():
    """The two copies of the helper, by the file they live in."""
    return {
        "studio/install_llama_prebuilt.py": prebuilt_mod._native_linux_system_rocm_lib_dirs,
        "studio/backend/core/inference/llama_cpp.py": llama_ns[
            "_native_linux_system_rocm_lib_dirs"
        ],
    }


def _norm(paths):
    """os.path.join emits '\\' on the Windows test host; compare POSIX-style."""
    return [str(p).replace("\\", "/") for p in paths]


def _fake_exists(present):
    """os.path.exists stub over a set of POSIX paths."""
    present = {p.replace("\\", "/") for p in present}

    def _exists(p):
        return str(p).replace("\\", "/") in present

    return _exists


@pytest.fixture
def bundle_dir(tmp_path):
    """A prebuilt directory that does contain a bundled HIP runtime."""
    d = tmp_path / "bundle"
    d.mkdir()
    (d / "libggml-hip.so").write_text("")
    return d


@pytest.fixture(autouse = True)
def _clean_rocm_env(monkeypatch):
    for var in ("UNSLOTH_LLAMA_NO_SYSTEM_ROCM", "HIP_PATH", "HIP_PATH_57", "ROCM_PATH"):
        monkeypatch.delenv(var, raising = False)


def _call(
    impl,
    bundle,
    present,
    platform = "linux",
):
    """Run one copy of the helper against a fake host.

    sys.platform is patched inside the call rather than in a fixture: pytest's own
    tmp_path factory branches on it, so a session-wide patch breaks the fixture on
    a Windows test host."""
    with patch.object(sys, "platform", platform):
        # isdir too: the llvm probe requires a directory, so a fake host that only answers exists() would report every
        with (
            patch("os.path.exists", _fake_exists(present)),
            patch("os.path.isdir", _fake_exists(present)),
        ):
            return _norm(impl(str(bundle)))


class TestBundledHipPresent:
    """The prepend only makes sense when the prebuilt actually bundles HIP; a
    CPU or CUDA build must be left alone."""

    @pytest.mark.parametrize(
        "where", ["studio/install_llama_prebuilt.py", "studio/backend/core/inference/llama_cpp.py"]
    )
    def test_detects_versioned_and_plain_sonames(self, tmp_path, where):
        impl = (
            prebuilt_mod._bundled_hip_present
            if where == "studio/install_llama_prebuilt.py"
            else llama_ns["_bundled_hip_present"]
        )
        plain = tmp_path / "plain"
        plain.mkdir()
        (plain / "libggml-hip.so").write_text("")
        versioned = tmp_path / "versioned"
        versioned.mkdir()
        (versioned / "libggml-hip.so.0.0.1").write_text("")
        cpu_only = tmp_path / "cpu"
        cpu_only.mkdir()
        (cpu_only / "libggml-cpu.so").write_text("")
        assert impl(str(plain)) is True, f"{where}: plain soname not detected"
        assert impl(str(versioned)) is True, f"{where}: versioned soname not detected"
        assert impl(str(cpu_only)) is False, f"{where}: CPU-only build treated as HIP"
        assert impl("") is False, f"{where}: empty binary_dir must be falsy"
        assert (
            impl(str(tmp_path / "does-not-exist")) is False
        ), f"{where}: missing dir must be falsy"


class TestNativeLinuxGates:
    """Each gate, on both copies. A gate that stops working silently reorders
    LD_LIBRARY_PATH for a platform the fix was never aimed at."""

    _ROCM_LIB = "/opt/rocm/lib"
    _HOST = {"/dev/kfd", "/opt/rocm/lib/libhsa-runtime64.so"}

    _run = staticmethod(_call)

    def test_returns_system_rocm_lib_on_a_native_rocm_host(self, bundle_dir):
        for where, impl in _impls().items():
            assert self._run(impl, bundle_dir, self._HOST) == [self._ROCM_LIB], where

    def test_accepts_versioned_hsa_runtime_soname(self, bundle_dir):
        present = {"/dev/kfd", "/opt/rocm/lib/libhsa-runtime64.so.1"}
        for where, impl in _impls().items():
            assert self._run(impl, bundle_dir, present) == [self._ROCM_LIB], where

    @pytest.mark.parametrize("platform", ["win32", "darwin"])
    def test_no_op_off_linux(self, bundle_dir, platform):
        for where, impl in _impls().items():
            assert self._run(impl, bundle_dir, self._HOST, platform = platform) == [], where

    def test_no_op_on_wsl(self, bundle_dir):
        """WSL has its own ordering path (plus HSA_ENABLE_DXG_DETECTION); /dev/dxg
        must hand off to it, not double-prepend here."""
        for where, impl in _impls().items():
            assert self._run(impl, bundle_dir, self._HOST | {"/dev/dxg"}) == [], where

    def test_no_op_without_amdkfd(self, bundle_dir):
        """No /dev/kfd: NVIDIA host, CPU host, or a container without the AMD
        device node. Prepending system ROCm there would be pure breakage."""
        present = {"/opt/rocm/lib/libhsa-runtime64.so"}
        for where, impl in _impls().items():
            assert self._run(impl, bundle_dir, present) == [], where

    def test_no_op_when_prebuilt_bundles_no_hip(self, tmp_path):
        cpu_bundle = tmp_path / "cpu-bundle"
        cpu_bundle.mkdir()
        for where, impl in _impls().items():
            assert self._run(impl, cpu_bundle, self._HOST) == [], where

    def test_no_op_when_system_rocm_has_no_hsa_runtime(self, bundle_dir):
        """ROCm dir exists but is not a usable runtime install."""
        for where, impl in _impls().items():
            assert self._run(impl, bundle_dir, {"/dev/kfd"}) == [], where

    def test_opt_out_env_wins_over_everything(self, bundle_dir, monkeypatch):
        monkeypatch.setenv("UNSLOTH_LLAMA_NO_SYSTEM_ROCM", "1")
        for where, impl in _impls().items():
            assert self._run(impl, bundle_dir, self._HOST) == [], where

    def test_opt_out_env_only_honours_exactly_one(self, bundle_dir, monkeypatch):
        """Documented switch is =1; "0"/"" must not disable the fix."""
        for value in ("0", "", "false"):
            monkeypatch.setenv("UNSLOTH_LLAMA_NO_SYSTEM_ROCM", value)
            for where, impl in _impls().items():
                assert self._run(impl, bundle_dir, self._HOST) == [
                    self._ROCM_LIB
                ], f"{where} ({value!r})"


class TestNativeLinuxRootResolution:
    """Which ROCm roots are searched, in what order."""

    _run = staticmethod(_call)

    def test_env_roots_take_precedence_over_opt_rocm(self, bundle_dir, monkeypatch):
        """A user with a side-by-side ROCm (HIP_PATH) must get theirs first: the
        one matching their driver, not whatever /opt/rocm happens to be."""
        monkeypatch.setenv("HIP_PATH", "/usr/local/rocm7")
        present = {
            "/dev/kfd",
            "/usr/local/rocm7/lib/libhsa-runtime64.so",
            "/opt/rocm/lib/libhsa-runtime64.so",
        }
        for where, impl in _impls().items():
            assert self._run(impl, bundle_dir, present) == [
                "/usr/local/rocm7/lib",
                "/opt/rocm/lib",
            ], where

    def test_all_three_env_roots_are_consulted_in_order(self, bundle_dir, monkeypatch):
        monkeypatch.setenv("HIP_PATH", "/a")
        monkeypatch.setenv("HIP_PATH_57", "/b")
        monkeypatch.setenv("ROCM_PATH", "/c")
        present = {
            "/dev/kfd",
            "/a/lib/libhsa-runtime64.so",
            "/b/lib/libhsa-runtime64.so",
            "/c/lib/libhsa-runtime64.so",
        }
        for where, impl in _impls().items():
            assert self._run(impl, bundle_dir, present) == ["/a/lib", "/b/lib", "/c/lib"], where

    def test_lib64_layout_is_found(self, bundle_dir):
        """RHEL / SUSE ROCm packages install to lib64."""
        present = {"/dev/kfd", "/opt/rocm/lib64/libhsa-runtime64.so"}
        for where, impl in _impls().items():
            assert self._run(impl, bundle_dir, present) == ["/opt/rocm/lib64"], where

    def test_nested_llvm_runtime_follows_system_rocm_lib(self, bundle_dir):
        """#7446: libamd_comgr depends on ROCm's versioned LLVM runtime, which is
        installed below lib/llvm/lib rather than directly in lib."""
        present = {
            "/dev/kfd",
            "/opt/rocm/lib/libhsa-runtime64.so",
            "/opt/rocm/lib/llvm/lib",
        }
        for where, impl in _impls().items():
            assert self._run(impl, bundle_dir, present) == [
                "/opt/rocm/lib",
                "/opt/rocm/lib/llvm/lib",
            ], where

    def test_lib64_host_still_finds_llvm_under_lib(self, bundle_dir):
        """ROCm puts LLVM under <root>/lib/llvm even where HSA lives in lib64, so
        deriving the llvm dir from the HSA dir alone would miss it and leave
        libamd_comgr binding to the bundle's libLLVM."""
        present = {
            "/dev/kfd",
            "/opt/rocm/lib64/libhsa-runtime64.so",
            "/opt/rocm/lib/llvm/lib",
        }
        for where, impl in _impls().items():
            assert self._run(impl, bundle_dir, present) == [
                "/opt/rocm/lib64",
                "/opt/rocm/lib/llvm/lib",
            ], where

    def test_lib64_host_prefers_its_own_llvm_dir_when_both_exist(self, bundle_dir):
        present = {
            "/dev/kfd",
            "/opt/rocm/lib64/libhsa-runtime64.so",
            "/opt/rocm/lib64/llvm/lib",
            "/opt/rocm/lib/llvm/lib",
        }
        for where, impl in _impls().items():
            assert self._run(impl, bundle_dir, present) == [
                "/opt/rocm/lib64",
                "/opt/rocm/lib64/llvm/lib",
                "/opt/rocm/lib/llvm/lib",
            ], where

    def test_llvm_path_that_is_a_file_is_not_prepended(self, tmp_path, bundle_dir):
        """Real filesystem: the serve-time caller joins these straight into
        LD_LIBRARY_PATH without an is-dir filter, so a non-directory must not
        reach it."""
        root = tmp_path / "rocm"
        (root / "lib").mkdir(parents = True)
        (root / "lib" / "libhsa-runtime64.so").write_text("")
        (root / "lib" / "llvm").mkdir()
        (root / "lib" / "llvm" / "lib").write_text("not a directory")
        real_exists = os.path.exists
        # take the WSL early-return and make this pass for the wrong reason.
        # Pin both device nodes: a WSL test host really has /dev/dxg, which would take the WSL early-return and make
        pinned = {"/dev/kfd": True, "/dev/dxg": False}

        def _exists(p):
            return pinned.get(str(p), None) if str(p) in pinned else real_exists(p)

        # A test host may itself have a real /opt/rocm (the default candidate), so assert on the bogus entry rather
        for where, impl in _impls().items():
            with (
                patch.object(sys, "platform", "linux"),
                patch.dict(os.environ, {"ROCM_PATH": str(root)}, clear = True),
                patch("os.path.exists", _exists),
            ):
                out = impl(str(bundle_dir))
            assert str(root / "lib") in out, where
            assert str(root / "lib" / "llvm" / "lib") not in out, where

    def test_lib_precedes_lib64_when_both_exist(self, bundle_dir):
        present = {
            "/dev/kfd",
            "/opt/rocm/lib/libhsa-runtime64.so",
            "/opt/rocm/lib64/libhsa-runtime64.so",
        }
        for where, impl in _impls().items():
            assert self._run(impl, bundle_dir, present) == [
                "/opt/rocm/lib",
                "/opt/rocm/lib64",
            ], where

    def test_duplicate_roots_are_deduped(self, bundle_dir, monkeypatch):
        """ROCM_PATH=/opt/rocm is the common setup; it must not emit the dir twice."""
        monkeypatch.setenv("ROCM_PATH", "/opt/rocm")
        present = {"/dev/kfd", "/opt/rocm/lib/libhsa-runtime64.so"}
        for where, impl in _impls().items():
            assert self._run(impl, bundle_dir, present) == ["/opt/rocm/lib"], where

    def test_empty_env_var_is_ignored(self, bundle_dir, monkeypatch):
        monkeypatch.setenv("HIP_PATH", "")
        present = {"/dev/kfd", "/opt/rocm/lib/libhsa-runtime64.so"}
        for where, impl in _impls().items():
            assert self._run(impl, bundle_dir, present) == ["/opt/rocm/lib"], where


class TestHelperParity:
    """The two copies carry a "mirrors ..." comment and nothing enforced it."""

    @pytest.mark.parametrize("name", _HELPERS)
    def test_bodies_are_identical(self, name):
        a = _function_ast(_PREBUILT_PATH, name)
        b = _function_ast(_LLAMA_CPP_PATH, name)
        assert ast.dump(a) == ast.dump(b), (
            f"{name} has drifted between install_llama_prebuilt.py and llama_cpp.py; "
            "the install-time and serve-time launchers must resolve the same lib dirs"
        )


def _function_ast(path: Path, name: str) -> ast.FunctionDef:
    """The function's executable body, with docstring and type annotations
    stripped: llama_cpp.py quotes its annotations ('list[str]') for the
    older-typing lint and documents itself as mirroring the installer. Neither is
    drift; the code is."""
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            for child in ast.walk(node):
                if isinstance(child, (ast.AnnAssign, ast.arg)):
                    child.annotation = None
                elif isinstance(child, ast.FunctionDef):
                    child.returns = None
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
            return node
    raise AssertionError(f"{name} not found in {path.name}")


class TestBinaryEnvNativeOrdering:
    """install-time validation launches the binary through binary_env."""

    @staticmethod
    def _linux_host():
        return prebuilt_mod.HostInfo(
            system = "Linux",
            machine = "x86_64",
            is_windows = False,
            is_linux = True,
            is_macos = False,
            is_x86_64 = True,
            is_arm64 = False,
            nvidia_smi = None,
            driver_cuda_version = None,
            compute_caps = [],
            visible_cuda_devices = None,
            has_physical_nvidia = False,
            has_usable_nvidia = False,
            has_rocm = True,
        )

    def test_system_rocm_precedes_bundle_dir(self, tmp_path):
        binary = tmp_path / "bundle" / "llama-server"
        binary.parent.mkdir(parents = True)
        binary.write_text("")
        sys_rocm = tmp_path / "sysrocm"  # dedupe_existing_dirs drops missing dirs
        sys_rocm.mkdir()
        with patch.object(prebuilt_mod, "_wsl_system_rocm_lib_dirs", return_value = []):
            with patch.object(
                prebuilt_mod, "_native_linux_system_rocm_lib_dirs", return_value = [str(sys_rocm)]
            ):
                with patch.dict(os.environ, {}, clear = True):
                    env = prebuilt_mod.binary_env(binary, tmp_path, self._linux_host())
        ld = [str(Path(p).resolve()) for p in env["LD_LIBRARY_PATH"].split(os.pathsep)]
        assert ld.index(str(sys_rocm.resolve())) < ld.index(str(binary.parent.resolve()))

    def test_native_path_does_not_enable_dxg_detection(self, tmp_path):
        """HSA_ENABLE_DXG_DETECTION belongs to the WSL branch only; setting it on
        bare metal changes HSA agent enumeration for every native AMD user."""
        binary = tmp_path / "bundle" / "llama-server"
        binary.parent.mkdir(parents = True)
        binary.write_text("")
        sys_rocm = tmp_path / "sysrocm"
        sys_rocm.mkdir()
        with patch.object(prebuilt_mod, "_wsl_system_rocm_lib_dirs", return_value = []):
            with patch.object(
                prebuilt_mod, "_native_linux_system_rocm_lib_dirs", return_value = [str(sys_rocm)]
            ):
                with patch.dict(os.environ, {}, clear = True):
                    env = prebuilt_mod.binary_env(binary, tmp_path, self._linux_host())
        assert "HSA_ENABLE_DXG_DETECTION" not in env

    def test_helper_is_asked_about_the_binary_dir_not_the_install_dir(self, tmp_path):
        """_bundled_hip_present globs the directory it is handed; passing
        install_dir would look for libggml-hip.so in the wrong place and no-op."""
        binary = tmp_path / "bundle" / "llama-server"
        binary.parent.mkdir(parents = True)
        binary.write_text("")
        seen = []

        def _spy(binary_dir = ""):
            seen.append(binary_dir)
            return []

        with patch.object(prebuilt_mod, "_wsl_system_rocm_lib_dirs", return_value = []):
            with patch.object(prebuilt_mod, "_native_linux_system_rocm_lib_dirs", _spy):
                with patch.dict(os.environ, {}, clear = True):
                    prebuilt_mod.binary_env(binary, tmp_path, self._linux_host())
        assert seen == [str(binary.parent)]

    def test_no_prepend_leaves_bundle_dir_first(self, tmp_path):
        binary = tmp_path / "bundle" / "llama-server"
        binary.parent.mkdir(parents = True)
        binary.write_text("")
        with patch.object(prebuilt_mod, "_wsl_system_rocm_lib_dirs", return_value = []):
            with patch.object(prebuilt_mod, "_native_linux_system_rocm_lib_dirs", return_value = []):
                with patch.dict(os.environ, {}, clear = True):
                    env = prebuilt_mod.binary_env(binary, tmp_path, self._linux_host())
        assert env["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(binary.parent)


class TestLlamaCppRuntimeNativeOrdering:
    """The serve-time launcher builds LD_LIBRARY_PATH inline inside a large
    function, so this half stays a source check (as the WSL sibling does)."""

    def test_prepends_before_binary_dir(self):
        source = _LLAMA_CPP_PATH.read_text(encoding = "utf-8")
        idx_helper = source.find("lib_dirs.extend(_native_linux_system_rocm_lib_dirs(binary_dir))")
        idx_binary = source.find("lib_dirs.append(binary_dir)")
        assert (
            idx_helper != -1
        ), "serve-time launcher must call the native-Linux helper with binary_dir"
        assert idx_binary != -1
        assert (
            idx_helper < idx_binary
        ), "system ROCm must be searched before the bundled HIP runtime"

    def test_dxg_detection_stays_on_the_wsl_branch(self):
        """HSA_ENABLE_DXG_DETECTION must be set from the WSL helper's result only."""
        source = _LLAMA_CPP_PATH.read_text(encoding = "utf-8")
        idx_wsl = source.find("lib_dirs.extend(_wsl_system_rocm_lib_dirs())")
        idx_dxg = source.find('env.setdefault("HSA_ENABLE_DXG_DETECTION", "1")', idx_wsl)
        idx_native = source.find("lib_dirs.extend(_native_linux_system_rocm_lib_dirs(binary_dir))")
        assert idx_wsl != -1 and idx_dxg != -1 and idx_native != -1
        assert idx_wsl < idx_dxg < idx_native, (
            "HSA_ENABLE_DXG_DETECTION must be decided from the WSL dirs alone, before the "
            "native-Linux dirs are appended to lib_dirs"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
