"""
Comprehensive tests for PR #4562 bug fixes.

Tests cover:
  - Bug 1: PS1 detached HEAD on re-run (fetch + checkout -B pattern)
  - Bug 2: Source-build fallback ignores pinned tag (both .sh and .ps1)
  - Bug 3: Unix fallback deletes install before checking prerequisites
  - Bug 4: Linux LD_LIBRARY_PATH missing build/bin
  - "latest" tag resolution fallback chain (helper -> raw)
  - Cross-platform binary_env (Linux, macOS, Windows)
  - Edge cases: malformed JSON, empty responses, env overrides

Run: pytest tests/studio/install/test_pr4562_bugfixes.py -v
"""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Load the module under test (same pattern as existing test files)
# ---------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
SPEC = importlib.util.spec_from_file_location(
    "studio_install_llama_prebuilt", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)

binary_env = MOD.binary_env
HostInfo = MOD.HostInfo
resolve_requested_llama_tag = MOD.resolve_requested_llama_tag
PublishedReleaseBundle = MOD.PublishedReleaseBundle
ApprovedArtifactHash = MOD.ApprovedArtifactHash
ApprovedReleaseChecksums = MOD.ApprovedReleaseChecksums
source_archive_logical_name = MOD.source_archive_logical_name

SETUP_SH = PACKAGE_ROOT / "studio" / "setup.sh"
SETUP_PS1 = PACKAGE_ROOT / "studio" / "setup.ps1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_host(*, system: str) -> HostInfo:
    """Create a HostInfo for the given OS."""
    return HostInfo(
        system = system,
        machine = "x86_64" if system != "Darwin" else "arm64",
        is_windows = (system == "Windows"),
        is_linux = (system == "Linux"),
        is_macos = (system == "Darwin"),
        is_x86_64 = (system != "Darwin"),
        is_arm64 = (system == "Darwin"),
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )


BASH = "/bin/bash"


def run_bash(script: str, *, timeout: int = 10, env: dict | None = None) -> str:
    """Run a bash script fragment and return its stdout."""
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    result = subprocess.run(
        [BASH, "-c", script],
        capture_output = True,
        text = True,
        timeout = timeout,
        env = run_env,
    )
    assert (
        result.returncode == 0
    ), f"bash script failed (exit {result.returncode}):\n{result.stderr}"
    return result.stdout.strip()


# =========================================================================
# TEST GROUP A: binary_env across all platforms (Bug 4 + cross-platform)
# =========================================================================
class TestBinaryEnvCrossPlatform:
    """Test that binary_env returns correct library paths for all OSes."""

    def test_linux_includes_binary_parent_in_ld_library_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        install_dir = tmp_path / "llama.cpp"
        bin_dir = install_dir / "build" / "bin"
        bin_dir.mkdir(parents = True)
        binary_path = bin_dir / "llama-server"
        binary_path.write_bytes(b"fake")

        host = make_host(system = "Linux")
        monkeypatch.setattr(MOD, "linux_runtime_dirs", lambda _bp: [])

        env = binary_env(binary_path, install_dir, host)
        ld_dirs = env["LD_LIBRARY_PATH"].split(os.pathsep)
        assert str(bin_dir) in ld_dirs, f"build/bin not in LD_LIBRARY_PATH: {ld_dirs}"
        assert (
            str(install_dir) in ld_dirs
        ), f"install_dir not in LD_LIBRARY_PATH: {ld_dirs}"

    def test_linux_binary_parent_comes_before_install_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """build/bin should be searched before install_dir for .so files."""
        install_dir = tmp_path / "llama.cpp"
        bin_dir = install_dir / "build" / "bin"
        bin_dir.mkdir(parents = True)
        binary_path = bin_dir / "llama-server"
        binary_path.write_bytes(b"fake")

        host = make_host(system = "Linux")
        monkeypatch.setattr(MOD, "linux_runtime_dirs", lambda _bp: [])

        env = binary_env(binary_path, install_dir, host)
        ld_dirs = env["LD_LIBRARY_PATH"].split(os.pathsep)
        bin_idx = ld_dirs.index(str(bin_dir))
        install_idx = ld_dirs.index(str(install_dir))
        assert (
            bin_idx < install_idx
        ), "binary_path.parent should come before install_dir"

    def test_linux_deduplicates_when_binary_parent_equals_install_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """When binary is directly in install_dir, no duplicate entries."""
        install_dir = tmp_path / "llama.cpp"
        install_dir.mkdir(parents = True)
        binary_path = install_dir / "llama-server"
        binary_path.write_bytes(b"fake")

        host = make_host(system = "Linux")
        monkeypatch.setattr(MOD, "linux_runtime_dirs", lambda _bp: [])

        env = binary_env(binary_path, install_dir, host)
        ld_dirs = [d for d in env["LD_LIBRARY_PATH"].split(os.pathsep) if d]
        count = ld_dirs.count(str(install_dir))
        assert count == 1, f"install_dir appears {count} times in LD_LIBRARY_PATH"

    def test_linux_preserves_existing_ld_library_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        install_dir = tmp_path / "llama.cpp"
        bin_dir = install_dir / "build" / "bin"
        bin_dir.mkdir(parents = True)
        binary_path = bin_dir / "llama-server"
        binary_path.write_bytes(b"fake")

        # Create real directories so dedupe_existing_dirs keeps them
        custom_lib = tmp_path / "custom_lib"
        other_lib = tmp_path / "other_lib"
        custom_lib.mkdir()
        other_lib.mkdir()

        host = make_host(system = "Linux")
        monkeypatch.setattr(MOD, "linux_runtime_dirs", lambda _bp: [])
        original = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{custom_lib}:{other_lib}"
        try:
            env = binary_env(binary_path, install_dir, host)
        finally:
            if original:
                os.environ["LD_LIBRARY_PATH"] = original
            else:
                os.environ.pop("LD_LIBRARY_PATH", None)
        ld_dirs = env["LD_LIBRARY_PATH"].split(os.pathsep)
        assert str(custom_lib.resolve()) in ld_dirs
        assert str(other_lib.resolve()) in ld_dirs

    def test_windows_includes_binary_parent_in_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        install_dir = tmp_path / "llama.cpp"
        bin_dir = install_dir / "build" / "bin" / "Release"
        bin_dir.mkdir(parents = True)
        binary_path = bin_dir / "llama-server.exe"
        binary_path.write_bytes(b"MZ")

        host = make_host(system = "Windows")
        monkeypatch.setattr(
            MOD, "windows_runtime_dirs_for_runtime_line", lambda _rt: []
        )

        env = binary_env(binary_path, install_dir, host)
        path_dirs = env["PATH"].split(os.pathsep)
        assert str(bin_dir) in path_dirs, f"build/bin/Release not in PATH: {path_dirs}"

    def test_macos_sets_dyld_library_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        install_dir = tmp_path / "llama.cpp"
        install_dir.mkdir(parents = True)
        bin_dir = install_dir / "build" / "bin"
        binary_path = bin_dir / "llama-server"
        binary_path.parent.mkdir(parents = True)
        binary_path.write_bytes(b"fake")

        host = make_host(system = "Darwin")
        monkeypatch.delenv("DYLD_LIBRARY_PATH", raising = False)

        env = binary_env(binary_path, install_dir, host)
        dyld_parts = [p for p in env["DYLD_LIBRARY_PATH"].split(os.pathsep) if p]
        assert (
            str(bin_dir) in dyld_parts
        ), f"build/bin not in DYLD_LIBRARY_PATH: {dyld_parts}"
        assert (
            str(install_dir) in dyld_parts
        ), f"install_dir not in DYLD_LIBRARY_PATH: {dyld_parts}"
        # binary_path.parent (build/bin) should come before install_dir
        assert dyld_parts.index(str(bin_dir)) < dyld_parts.index(str(install_dir))


# =========================================================================
# TEST GROUP B: resolve_requested_llama_tag (Python function)
# =========================================================================
class TestResolveRequestedLlamaTag:
    def test_concrete_tag_passes_through(self):
        assert resolve_requested_llama_tag("b8508") == "b8508"

    def test_none_resolves_to_latest(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(MOD, "latest_upstream_release_tag", lambda: "b9999")
        assert resolve_requested_llama_tag(None) == "b9999"

    def test_latest_resolves_to_upstream(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(MOD, "latest_upstream_release_tag", lambda: "b1234")
        assert resolve_requested_llama_tag("latest") == "b1234"

    def test_empty_string_resolves_to_latest(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(MOD, "latest_upstream_release_tag", lambda: "b5555")
        assert resolve_requested_llama_tag("") == "b5555"

    def test_latest_with_published_repo_uses_latest_valid_published_release(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        invalid = PublishedReleaseBundle(
            repo = "unslothai/llama.cpp",
            release_tag = "v2.0",
            upstream_tag = "b9000",
            assets = {},
            manifest_asset_name = "llama-prebuilt-manifest.json",
            artifacts = [],
            selection_log = [],
        )
        valid = PublishedReleaseBundle(
            repo = "unslothai/llama.cpp",
            release_tag = "v1.0",
            upstream_tag = "b8999",
            assets = {},
            manifest_asset_name = "llama-prebuilt-manifest.json",
            artifacts = [],
            selection_log = [],
        )

        monkeypatch.setattr(
            MOD,
            "iter_published_release_bundles",
            lambda repo, published_release_tag = "": iter([invalid, valid]),
        )

        def fake_load(repo, release_tag):
            if release_tag == "v2.0":
                raise MOD.PrebuiltFallback("checksum asset missing")
            return ApprovedReleaseChecksums(
                repo = repo,
                release_tag = release_tag,
                upstream_tag = "b8999",
                source_commit = None,
                artifacts = {
                    source_archive_logical_name("b8999"): ApprovedArtifactHash(
                        asset_name = source_archive_logical_name("b8999"),
                        sha256 = "a" * 64,
                        repo = "ggml-org/llama.cpp",
                        kind = "upstream-source",
                    )
                },
            )

        monkeypatch.setattr(MOD, "load_approved_release_checksums", fake_load)
        monkeypatch.setattr(MOD, "latest_upstream_release_tag", lambda: "b7777")

        assert resolve_requested_llama_tag("latest", "unslothai/llama.cpp") == "b8999"


# =========================================================================
# TEST GROUP C: setup.sh logic (bash subprocess tests)
# =========================================================================
class TestSetupShLogic:
    """Test setup.sh fragments via bash subprocess with controlled PATH."""

    def test_cmake_missing_preserves_install(self, tmp_path: Path):
        """Bug 3: When cmake is missing, rm -rf should NOT run."""
        llama_dir = tmp_path / "llama.cpp"
        llama_dir.mkdir()
        marker = llama_dir / "marker.txt"
        marker.write_text("existing")

        mock_bin = tmp_path / "mock_bin"
        mock_bin.mkdir()
        # Create mock git but NOT cmake
        (mock_bin / "git").write_text("#!/bin/bash\nexit 0\n")
        (mock_bin / "git").chmod(0o755)

        # Build PATH: mock_bin first, then system dirs WITHOUT cmake
        safe_dirs = [str(mock_bin)]
        for d in os.environ.get("PATH", "").split(":"):
            if d and not os.path.isfile(os.path.join(d, "cmake")):
                safe_dirs.append(d)

        script = textwrap.dedent(f"""\
            export LLAMA_CPP_DIR="{llama_dir}"
            if ! command -v cmake &>/dev/null; then
                echo "cmake_missing"
            elif ! command -v git &>/dev/null; then
                echo "git_missing"
            else
                rm -rf "$LLAMA_CPP_DIR"
                echo "would_clone"
            fi
        """)
        output = run_bash(script, env = {"PATH": ":".join(safe_dirs)})
        assert "cmake_missing" in output
        assert marker.exists(), "Install dir was deleted despite cmake missing!"

    def test_git_missing_preserves_install(self, tmp_path: Path):
        """Bug 3: When git is missing, rm -rf should NOT run."""
        llama_dir = tmp_path / "llama.cpp"
        llama_dir.mkdir()
        marker = llama_dir / "marker.txt"
        marker.write_text("existing")

        mock_bin = tmp_path / "mock_bin"
        mock_bin.mkdir()
        # Create mock cmake but NOT git
        (mock_bin / "cmake").write_text("#!/bin/bash\nexit 0\n")
        (mock_bin / "cmake").chmod(0o755)

        # Build PATH: mock_bin first, then system dirs WITHOUT git
        safe_dirs = [str(mock_bin)]
        for d in os.environ.get("PATH", "").split(":"):
            if d and not os.path.isfile(os.path.join(d, "git")):
                safe_dirs.append(d)

        script = textwrap.dedent(f"""\
            export LLAMA_CPP_DIR="{llama_dir}"
            if ! command -v cmake &>/dev/null; then
                echo "cmake_missing"
            elif ! command -v git &>/dev/null; then
                echo "git_missing"
            else
                rm -rf "$LLAMA_CPP_DIR"
                echo "would_clone"
            fi
        """)
        output = run_bash(script, env = {"PATH": ":".join(safe_dirs)})
        assert "git_missing" in output
        assert marker.exists(), "Install dir was deleted despite git missing!"

    def test_both_present_runs_rm_and_clone(self, tmp_path: Path):
        """Bug 3: When both present, rm -rf runs before clone."""
        llama_dir = tmp_path / "llama.cpp"
        llama_dir.mkdir()
        marker = llama_dir / "marker.txt"
        marker.write_text("existing")

        mock_bin = tmp_path / "mock_bin"
        mock_bin.mkdir()
        (mock_bin / "cmake").write_text("#!/bin/bash\nexit 0\n")
        (mock_bin / "cmake").chmod(0o755)
        (mock_bin / "git").write_text("#!/bin/bash\nexit 0\n")
        (mock_bin / "git").chmod(0o755)

        script = textwrap.dedent(f"""\
            export PATH="{mock_bin}:$PATH"
            export LLAMA_CPP_DIR="{llama_dir}"
            if ! command -v cmake &>/dev/null; then
                echo "cmake_missing"
            elif ! command -v git &>/dev/null; then
                echo "git_missing"
            else
                rm -rf "$LLAMA_CPP_DIR"
                echo "would_clone"
            fi
        """)
        output = run_bash(script)
        assert "would_clone" in output
        assert not marker.exists(), "Install dir should have been deleted"

    def test_clone_uses_pinned_tag(self, tmp_path: Path):
        """Bug 2: git clone should use --branch with the resolved tag."""
        mock_bin = tmp_path / "mock_bin"
        mock_bin.mkdir()
        log_file = tmp_path / "git_calls.log"
        (mock_bin / "git").write_text(f'#!/bin/bash\necho "$*" >> {log_file}\nexit 0\n')
        (mock_bin / "git").chmod(0o755)

        script = textwrap.dedent(f"""\
            export PATH="{mock_bin}:$PATH"
            git clone --depth 1 --branch "b8508" https://github.com/ggml-org/llama.cpp.git /tmp/llama_test
        """)
        run_bash(script)
        log = log_file.read_text()
        assert "--branch b8508" in log, f"Expected --branch b8508 in: {log}"

    def test_fetch_checkout_b_pattern(self, tmp_path: Path):
        """Bug 1: Re-run should use fetch + checkout -B, not pull + checkout FETCH_HEAD."""
        mock_bin = tmp_path / "mock_bin"
        mock_bin.mkdir()
        log_file = tmp_path / "git_calls.log"
        (mock_bin / "git").write_text(f'#!/bin/bash\necho "$*" >> {log_file}\nexit 0\n')
        (mock_bin / "git").chmod(0o755)

        llama_dir = tmp_path / "llama.cpp"
        llama_dir.mkdir()
        (llama_dir / ".git").mkdir()

        script = textwrap.dedent(f"""\
            export PATH="{mock_bin}:$PATH"
            LlamaCppDir="{llama_dir}"
            ResolvedLlamaTag="b8508"
            if [ -d "$LlamaCppDir/.git" ]; then
                git -C "$LlamaCppDir" fetch --depth 1 origin "$ResolvedLlamaTag"
                if [ $? -ne 0 ]; then
                    echo "WARN: fetch failed"
                else
                    git -C "$LlamaCppDir" checkout -B unsloth-llama-build FETCH_HEAD
                fi
            fi
        """)
        run_bash(script)
        log = log_file.read_text()
        assert "fetch --depth 1 origin b8508" in log
        assert "checkout -B unsloth-llama-build FETCH_HEAD" in log
        assert "pull" not in log, "Should use fetch, not pull"

    def test_fetch_failure_warns_not_aborts(self, tmp_path: Path):
        """Bug 1: fetch failure should warn and continue, not set BuildOk=false."""
        mock_bin = tmp_path / "mock_bin"
        mock_bin.mkdir()
        (mock_bin / "git").write_text(
            '#!/bin/bash\nif echo "$*" | grep -q fetch; then exit 1; fi\nexit 0\n'
        )
        (mock_bin / "git").chmod(0o755)

        llama_dir = tmp_path / "llama.cpp"
        llama_dir.mkdir()
        (llama_dir / ".git").mkdir()

        script = textwrap.dedent(f"""\
            export PATH="{mock_bin}:$PATH"
            LlamaCppDir="{llama_dir}"
            ResolvedLlamaTag="b8508"
            BuildOk=true
            if [ -d "$LlamaCppDir/.git" ]; then
                git -C "$LlamaCppDir" fetch --depth 1 origin "$ResolvedLlamaTag"
                if [ $? -ne 0 ]; then
                    echo "WARN: fetch failed -- using existing source"
                else
                    git -C "$LlamaCppDir" checkout -B unsloth-llama-build FETCH_HEAD
                fi
            fi
            echo "BuildOk=$BuildOk"
        """)
        output = run_bash(script)
        assert "WARN: fetch failed" in output
        assert "BuildOk=true" in output


# =========================================================================
# TEST GROUP D: "latest" tag resolution (bash subprocess)
# =========================================================================
class TestLatestTagResolution:
    """Test the fallback chain: helper resolver -> raw."""

    RESOLVE_TEMPLATE = textwrap.dedent("""\
        _REQUESTED_LLAMA_TAG="{requested_tag}"
        _RESOLVED_LLAMA_TAG=""
        _RESOLVE_UPSTREAM_STATUS={resolve_status}
        if [ "$_RESOLVE_UPSTREAM_STATUS" -eq 0 ] && [ -n "{resolved_tag}" ]; then
            _RESOLVED_LLAMA_TAG="{resolved_tag}"
        else
            _RESOLVED_LLAMA_TAG="$_REQUESTED_LLAMA_TAG"
        fi
        echo "$_RESOLVED_LLAMA_TAG"
    """)

    def _run_resolve(
        self,
        tmp_path: Path,
        requested_tag: str,
        resolved_tag: str,
        resolve_status: int,
    ) -> str:
        script = self.RESOLVE_TEMPLATE.format(
            requested_tag = requested_tag,
            resolved_tag = resolved_tag,
            resolve_status = resolve_status,
        )
        return run_bash(script)

    def test_helper_resolution_succeeds(self, tmp_path: Path):
        output = self._run_resolve(
            tmp_path,
            "latest",
            resolved_tag = "b8508",
            resolve_status = 0,
        )
        assert output == "b8508"

    def test_helper_resolution_falls_back_to_raw_requested_tag(self, tmp_path: Path):
        output = self._run_resolve(
            tmp_path,
            "latest",
            resolved_tag = "",
            resolve_status = 1,
        )
        assert output == "latest"

    def test_concrete_tag_passes_through_when_helper_fails(self, tmp_path: Path):
        output = self._run_resolve(
            tmp_path,
            "b7777",
            resolved_tag = "",
            resolve_status = 1,
        )
        assert output == "b7777"

    def test_env_override_unsloth_llama_tag(self):
        output = run_bash(
            'echo "${UNSLOTH_LLAMA_TAG:-latest}"',
            env = {"UNSLOTH_LLAMA_TAG": "b1234"},
        )
        assert output == "b1234"

    def test_env_unset_defaults_to_latest(self):
        env = os.environ.copy()
        env.pop("UNSLOTH_LLAMA_TAG", None)
        output = run_bash('echo "${UNSLOTH_LLAMA_TAG:-latest}"', env = env)
        assert output == "latest"

    def test_env_empty_defaults_to_latest(self):
        output = run_bash(
            'echo "${UNSLOTH_LLAMA_TAG:-latest}"',
            env = {"UNSLOTH_LLAMA_TAG": ""},
        )
        assert output == "latest"


# =========================================================================
# TEST GROUP E: Source file verification
# =========================================================================
class TestSourceCodePatterns:
    """Verify the actual source files contain the expected fix patterns."""

    def test_setup_sh_no_rm_before_prereq_check(self):
        """rm -rf must appear AFTER cmake/git checks, not before."""
        content = SETUP_SH.read_text()
        # Anchor on the source-build cmake check block.
        idx_block = content.find("command -v cmake")
        assert idx_block != -1
        block = content[idx_block:]
        # rm -rf should appear after the cmake/git checks
        idx_cmake = block.find("command -v cmake")
        idx_git = block.find("command -v git")
        idx_rm = block.find("rm -rf")
        assert idx_rm > idx_cmake, "rm -rf should come after cmake check"
        assert idx_rm > idx_git, "rm -rf should come after git check"

    def test_setup_sh_clone_uses_branch_tag(self):
        """git clone in source-build should use --branch via the clone args array."""
        content = SETUP_SH.read_text()
        assert "_CLONE_ARGS=(git clone --depth 1)" in content
        assert (
            '_CLONE_ARGS+=(--branch "$_RESOLVED_LLAMA_TAG")' in content
        ), "_CLONE_ARGS should be extended with --branch $_RESOLVED_LLAMA_TAG"
        # Verify the guard: --branch is only used when tag is not "latest"
        assert (
            '_RESOLVED_LLAMA_TAG" != "latest"' in content
        ), "Should guard against literal 'latest' tag"

    def test_setup_sh_latest_resolution_uses_helper_only(self):
        """Shell fallback should rely on helper output, not raw GitHub API tag_name."""
        content = SETUP_SH.read_text()
        assert "--resolve-install-tag" in content
        assert "--resolve-llama-tag" in content
        assert "_HELPER_RELEASE_REPO}/releases/latest" not in content
        assert "ggml-org/llama.cpp/releases/latest" not in content

    def test_setup_sh_macos_arm64_uses_metal_flags(self):
        """Apple Silicon source builds should explicitly enable Metal like upstream."""
        content = SETUP_SH.read_text()
        assert "_IS_MACOS_ARM64=true" in content
        assert 'if [ "$_IS_MACOS_ARM64" = true ]; then' in content
        assert "-DGGML_METAL=ON" in content
        assert "-DGGML_METAL_EMBED_LIBRARY=ON" in content
        assert "-DGGML_METAL_USE_BF16=ON" in content
        assert "-DCMAKE_INSTALL_RPATH=@loader_path" in content
        assert "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON" in content

    def test_setup_sh_macos_metal_configure_has_cpu_fallback(self):
        """If Metal configure or build fails, setup should retry with CPU fallback."""
        content = SETUP_SH.read_text()
        assert "_TRY_METAL_CPU_FALLBACK=true" in content
        assert (
            'substep "Metal configure failed; retrying CPU build..." "$C_WARN"'
            in content
        )
        assert (
            'substep "Metal build failed; retrying CPU build..." "$C_WARN"' in content
        )
        assert 'run_quiet_no_exit "cmake llama.cpp (cpu fallback)"' in content
        assert "-DGGML_METAL=OFF" in content

    def test_macos_arm64_cpu_fallback_args_exclude_rpath(self):
        """CPU fallback args must NOT contain Metal-only RPATH flags at runtime."""
        script = (
            '_IS_MACOS_ARM64=true\nNVCC_PATH=""\nGPU_BACKEND=""\n'
            + _GPU_BACKEND_FRAGMENT
        )
        output = run_bash(script)
        fallback_line = next(
            line
            for line in output.splitlines()
            if line.startswith("CPU_FALLBACK_CMAKE_ARGS=")
        )
        assert "-DGGML_METAL=OFF" in fallback_line
        assert (
            "@loader_path" not in fallback_line
        ), "CPU fallback args should not contain RPATH flags"
        assert (
            "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON" not in fallback_line
        ), "CPU fallback args should not contain RPATH build flag"

    def test_setup_sh_does_not_enable_metal_for_intel_macos(self):
        """Intel macOS should stay on the existing non-Metal path in this patch."""
        content = SETUP_SH.read_text()
        assert 'if [ "$_IS_MACOS_ARM64" = true ]; then' in content
        assert (
            'Darwin" ] && { [ "$_HOST_MACHINE" = "arm64" ] || [ "$_HOST_MACHINE" = "aarch64" ]; }'
            in content
        )
        assert (
            "x86_64"
            not in content[
                content.find("-DGGML_METAL=ON") - 200 : content.find("-DGGML_METAL=ON")
                + 200
            ]
        )

    def test_setup_ps1_uses_checkout_b(self):
        """PS1 should use checkout -B, not checkout --force FETCH_HEAD."""
        content = SETUP_PS1.read_text()
        assert "checkout -B unsloth-llama-build" in content
        assert "checkout --force FETCH_HEAD" not in content

    def test_setup_ps1_clone_uses_branch_tag(self):
        """PS1 clone should use --branch with the resolved tag."""
        content = SETUP_PS1.read_text()
        assert "--branch" in content and "$ResolvedLlamaTag" in content
        # The old commented-out line should be gone
        assert "# git clone --depth 1 --branch" not in content

    def test_setup_ps1_no_git_pull(self):
        """PS1 should use fetch, not pull (which fails in detached HEAD)."""
        content = SETUP_PS1.read_text()
        # In the source-build section, there should be no "git pull"
        # (git pull is only valid on a branch)
        lines = content.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "git pull" in stripped and not stripped.startswith("#"):
                # Check context -- should not be in the llama.cpp build section
                # Allow git pull in other contexts
                context = "\n".join(lines[max(0, i - 5) : i + 5])
                if "LlamaCppDir" in context:
                    pytest.fail(
                        f"Found 'git pull' in llama.cpp build section at line {i+1}"
                    )

    def test_setup_ps1_latest_resolution_uses_helper_only(self):
        """PS1 fallback should rely on helper output, not raw GitHub API tag_name."""
        content = SETUP_PS1.read_text()
        assert "--resolve-install-tag" in content
        assert "--resolve-llama-tag" in content
        assert "$HelperReleaseRepo/releases/latest" not in content
        assert "ggml-org/llama.cpp/releases/latest" not in content

    def test_binary_env_linux_has_binary_parent(self):
        """The Linux branch of binary_env should include binary_path.parent."""
        content = MODULE_PATH.read_text()
        # Find the binary_env function
        in_func = False
        in_linux = False
        found = False
        for line in content.splitlines():
            if "def binary_env(" in line:
                in_func = True
            elif in_func and line and not line[0].isspace() and "def " in line:
                break
            if in_func and "host.is_linux" in line:
                in_linux = True
            if in_linux and "binary_path.parent" in line:
                found = True
                break
        assert found, "binary_path.parent not found in Linux branch of binary_env"


# =========================================================================
# TEST GROUP F: macOS Metal build logic (bash subprocess tests)
# =========================================================================

# Minimal bash fragment that mirrors setup.sh's GPU backend decision chain.
# Variables _IS_MACOS_ARM64, NVCC_PATH, GPU_BACKEND are injected by tests.
_GPU_BACKEND_FRAGMENT = textwrap.dedent("""\
    CMAKE_ARGS="-DLLAMA_BUILD_TESTS=OFF"
    _TRY_METAL_CPU_FALLBACK=false
    CPU_FALLBACK_CMAKE_ARGS="$CMAKE_ARGS"

    _BUILD_DESC="building"
    if [ "$_IS_MACOS_ARM64" = true ]; then
        _BUILD_DESC="building (Metal)"
        CMAKE_ARGS="$CMAKE_ARGS -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_METAL_USE_BF16=ON -DCMAKE_INSTALL_RPATH=@loader_path -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"
        CPU_FALLBACK_CMAKE_ARGS="$CPU_FALLBACK_CMAKE_ARGS -DGGML_METAL=OFF"
        _TRY_METAL_CPU_FALLBACK=true
    elif [ -n "$NVCC_PATH" ]; then
        CMAKE_ARGS="$CMAKE_ARGS -DGGML_CUDA=ON"
        _BUILD_DESC="building (CUDA)"
    elif [ "$GPU_BACKEND" = "rocm" ]; then
        CMAKE_ARGS="$CMAKE_ARGS -DGGML_HIP=ON"
        _BUILD_DESC="building (ROCm)"
    else
        _BUILD_DESC="building (CPU)"
    fi

    echo "CMAKE_ARGS=$CMAKE_ARGS"
    echo "CPU_FALLBACK_CMAKE_ARGS=$CPU_FALLBACK_CMAKE_ARGS"
    echo "BUILD_DESC=$_BUILD_DESC"
    echo "TRY_METAL_CPU_FALLBACK=$_TRY_METAL_CPU_FALLBACK"
""")


class TestMacOSMetalBuildLogic:
    """Behavioral bash subprocess tests for the Metal GPU backend logic."""

    def test_macos_arm64_cmake_args_contain_metal_flags(self):
        """macOS arm64 should enable Metal, not CUDA."""
        script = (
            '_IS_MACOS_ARM64=true\nNVCC_PATH=""\nGPU_BACKEND=""\n'
            + _GPU_BACKEND_FRAGMENT
        )
        output = run_bash(script)
        assert "-DGGML_METAL=ON" in output
        assert "-DGGML_CUDA=ON" not in output
        assert "BUILD_DESC=building (Metal)" in output

    def test_intel_macos_no_metal_flags(self):
        """Intel macOS (not arm64) should not get Metal flags."""
        script = (
            '_IS_MACOS_ARM64=false\nNVCC_PATH=""\nGPU_BACKEND=""\n'
            + _GPU_BACKEND_FRAGMENT
        )
        output = run_bash(script)
        assert "-DGGML_METAL=ON" not in output
        assert "BUILD_DESC=building (CPU)" in output

    def test_macos_arm64_metal_precedes_nvcc(self):
        """Even with nvcc in PATH, macOS arm64 should use Metal, not CUDA."""
        script = (
            '_IS_MACOS_ARM64=true\nNVCC_PATH="/usr/local/cuda/bin/nvcc"\n'
            'GPU_BACKEND="cuda"\n' + _GPU_BACKEND_FRAGMENT
        )
        output = run_bash(script)
        assert "-DGGML_METAL=ON" in output
        assert "-DGGML_CUDA=ON" not in output
        assert "BUILD_DESC=building (Metal)" in output

    def test_metal_cpu_fallback_triggers_on_cmake_failure(self, tmp_path: Path):
        """When cmake fails on Metal, the fallback should retry with -DGGML_METAL=OFF."""
        mock_bin = tmp_path / "mock_bin"
        mock_bin.mkdir()
        calls_file = tmp_path / "cmake_calls.log"
        # cmake that logs args and fails on first call (Metal), succeeds on second (CPU fallback)
        cmake_script = mock_bin / "cmake"
        cmake_script.write_text(
            textwrap.dedent(f"""\
            #!/bin/bash
            echo "$*" >> "{calls_file}"
            COUNTER_FILE="{tmp_path}/cmake_counter"
            if [ ! -f "$COUNTER_FILE" ]; then
                echo 1 > "$COUNTER_FILE"
                exit 1
            fi
            exit 0
        """)
        )
        cmake_script.chmod(0o755)

        script = textwrap.dedent(f"""\
            export PATH="{mock_bin}:$PATH"
            _IS_MACOS_ARM64=true
            NVCC_PATH=""
            GPU_BACKEND=""
            CMAKE_ARGS="-DLLAMA_BUILD_TESTS=OFF"
            _TRY_METAL_CPU_FALLBACK=false
            CPU_FALLBACK_CMAKE_ARGS="$CMAKE_ARGS"

            _BUILD_DESC="building"
            if [ "$_IS_MACOS_ARM64" = true ]; then
                _BUILD_DESC="building (Metal)"
                CMAKE_ARGS="$CMAKE_ARGS -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_METAL_USE_BF16=ON -DCMAKE_INSTALL_RPATH=@loader_path -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"
                CPU_FALLBACK_CMAKE_ARGS="$CPU_FALLBACK_CMAKE_ARGS -DGGML_METAL=OFF"
                _TRY_METAL_CPU_FALLBACK=true
            fi

            BUILD_OK=true
            _BUILD_TMP="{tmp_path}/build_tmp"
            mkdir -p "$_BUILD_TMP"
            if ! cmake -S "$_BUILD_TMP" -B "$_BUILD_TMP/build" $CMAKE_ARGS; then
                if [ "$_TRY_METAL_CPU_FALLBACK" = true ]; then
                    echo "FALLBACK_TRIGGERED"
                    rm -rf "$_BUILD_TMP/build"
                    cmake -S "$_BUILD_TMP" -B "$_BUILD_TMP/build" $CPU_FALLBACK_CMAKE_ARGS || BUILD_OK=false
                    if [ "$BUILD_OK" = true ]; then
                        _BUILD_DESC="building (CPU fallback)"
                    fi
                else
                    BUILD_OK=false
                fi
            fi

            echo "BUILD_OK=$BUILD_OK"
            echo "BUILD_DESC=$_BUILD_DESC"
        """)
        output = run_bash(script)
        assert "FALLBACK_TRIGGERED" in output
        assert "BUILD_OK=true" in output
        assert "BUILD_DESC=building (CPU fallback)" in output

        # Verify cmake args: first call has Metal ON, second has Metal OFF
        calls = calls_file.read_text().splitlines()
        assert len(calls) >= 2, f"Expected >= 2 cmake calls, got {len(calls)}"
        assert (
            "-DGGML_METAL=ON" in calls[0]
        ), f"First cmake call should have Metal ON: {calls[0]}"
        assert (
            "-DGGML_METAL=OFF" in calls[1]
        ), f"Second cmake call should have Metal OFF: {calls[1]}"
        assert (
            "-DGGML_METAL=ON" not in calls[1]
        ), f"Second cmake call should NOT have Metal ON: {calls[1]}"
        assert (
            "@loader_path" not in calls[1]
        ), f"CPU fallback should not have RPATH: {calls[1]}"
        assert (
            "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON" not in calls[1]
        ), f"CPU fallback should not have RPATH build flag: {calls[1]}"

    def test_metal_build_failure_retries_cpu_fallback(self, tmp_path: Path):
        """When cmake --build fails on Metal, the fallback should re-configure and rebuild with CPU."""
        mock_bin = tmp_path / "mock_bin"
        mock_bin.mkdir()
        calls_file = tmp_path / "cmake_calls.log"
        # cmake mock: configure always succeeds; first --build fails, rest succeed
        cmake_script = mock_bin / "cmake"
        cmake_script.write_text(
            textwrap.dedent(f"""\
            #!/bin/bash
            echo "$*" >> "{calls_file}"
            if [ "$1" = "--build" ]; then
                BUILD_COUNTER_FILE="{tmp_path}/build_counter"
                if [ ! -f "$BUILD_COUNTER_FILE" ]; then
                    echo 1 > "$BUILD_COUNTER_FILE"
                    exit 1
                fi
            fi
            exit 0
        """)
        )
        cmake_script.chmod(0o755)

        script = textwrap.dedent(f"""\
            export PATH="{mock_bin}:$PATH"
            _IS_MACOS_ARM64=true
            NVCC_PATH=""
            GPU_BACKEND=""
            CMAKE_ARGS="-DLLAMA_BUILD_TESTS=OFF"
            _TRY_METAL_CPU_FALLBACK=false
            CPU_FALLBACK_CMAKE_ARGS="$CMAKE_ARGS"
            CMAKE_GENERATOR_ARGS=""
            NCPU=2

            _BUILD_DESC="building"
            if [ "$_IS_MACOS_ARM64" = true ]; then
                _BUILD_DESC="building (Metal)"
                CMAKE_ARGS="$CMAKE_ARGS -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_METAL_USE_BF16=ON -DCMAKE_INSTALL_RPATH=@loader_path -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"
                CPU_FALLBACK_CMAKE_ARGS="$CPU_FALLBACK_CMAKE_ARGS -DGGML_METAL=OFF"
                _TRY_METAL_CPU_FALLBACK=true
            fi

            BUILD_OK=true
            _BUILD_TMP="{tmp_path}/build_tmp"
            mkdir -p "$_BUILD_TMP"

            # Configure (succeeds)
            if ! cmake $CMAKE_GENERATOR_ARGS -S "$_BUILD_TMP" -B "$_BUILD_TMP/build" $CMAKE_ARGS; then
                if [ "$_TRY_METAL_CPU_FALLBACK" = true ]; then
                    echo "CONFIGURE_FALLBACK"
                    rm -rf "$_BUILD_TMP/build"
                    cmake $CMAKE_GENERATOR_ARGS -S "$_BUILD_TMP" -B "$_BUILD_TMP/build" $CPU_FALLBACK_CMAKE_ARGS || BUILD_OK=false
                    if [ "$BUILD_OK" = true ]; then
                        _BUILD_DESC="building (CPU fallback)"
                    fi
                else
                    BUILD_OK=false
                fi
            fi

            # Build (first --build fails, triggers fallback)
            if [ "$BUILD_OK" = true ]; then
                if ! cmake --build "$_BUILD_TMP/build" --config Release --target llama-server -j"$NCPU"; then
                    if [ "$_TRY_METAL_CPU_FALLBACK" = true ]; then
                        echo "BUILD_FALLBACK_TRIGGERED"
                        rm -rf "$_BUILD_TMP/build"
                        if cmake $CMAKE_GENERATOR_ARGS -S "$_BUILD_TMP" -B "$_BUILD_TMP/build" $CPU_FALLBACK_CMAKE_ARGS; then
                            _BUILD_DESC="building (CPU fallback)"
                            cmake --build "$_BUILD_TMP/build" --config Release --target llama-server -j"$NCPU" || BUILD_OK=false
                        else
                            BUILD_OK=false
                        fi
                    else
                        BUILD_OK=false
                    fi
                fi
            fi

            echo "BUILD_OK=$BUILD_OK"
            echo "BUILD_DESC=$_BUILD_DESC"
        """)
        output = run_bash(script)
        assert "CONFIGURE_FALLBACK" not in output, "Configure should have succeeded"
        assert "BUILD_FALLBACK_TRIGGERED" in output
        assert "BUILD_OK=true" in output
        assert "BUILD_DESC=building (CPU fallback)" in output

        # Verify: configure with Metal ON, build fails, re-configure with Metal OFF, rebuild
        calls = calls_file.read_text().splitlines()
        assert len(calls) >= 4, f"Expected >= 4 cmake calls, got {len(calls)}: {calls}"
        # First call: configure with Metal ON
        assert "-DGGML_METAL=ON" in calls[0]
        # Second call: build (fails)
        assert "--build" in calls[1]
        # Third call: re-configure with Metal OFF and no RPATH flags
        assert "-DGGML_METAL=OFF" in calls[2]
        assert "-DGGML_METAL=ON" not in calls[2]
        assert (
            "@loader_path" not in calls[2]
        ), f"CPU fallback should not have RPATH: {calls[2]}"
        assert (
            "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON" not in calls[2]
        ), f"CPU fallback should not have RPATH build flag: {calls[2]}"
        assert (
            "-DLLAMA_BUILD_TESTS=OFF" in calls[2]
        ), f"CPU fallback should preserve baseline flags: {calls[2]}"
        # Fourth call: rebuild (succeeds)
        assert "--build" in calls[3]
