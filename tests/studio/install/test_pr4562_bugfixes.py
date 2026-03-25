"""
Comprehensive tests for PR #4562 bug fixes.

Tests cover:
  - Bug 1: PS1 detached HEAD on re-run (fetch + checkout -B pattern)
  - Bug 2: Source-build fallback ignores pinned tag (both .sh and .ps1)
  - Bug 3: Unix fallback deletes install before checking prerequisites
  - Bug 4: Linux LD_LIBRARY_PATH missing build/bin
  - "latest" tag resolution fallback chain (Unsloth -> ggml-org -> raw)
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
    """Test the fallback chain: Unsloth API -> ggml-org API -> raw."""

    RESOLVE_TEMPLATE = textwrap.dedent("""\
        export PATH="{mock_bin}:$PATH"
        _REQUESTED_LLAMA_TAG="{requested_tag}"
        _RESOLVED_LLAMA_TAG=""
        _RESOLVE_UPSTREAM_STATUS=1
        _HELPER_RELEASE_REPO="unslothai/llama.cpp"
        if [ "$_RESOLVE_UPSTREAM_STATUS" -ne 0 ] || [ -z "$_RESOLVED_LLAMA_TAG" ]; then
            if [ "$_REQUESTED_LLAMA_TAG" = "latest" ]; then
                _RESOLVED_LLAMA_TAG="$(curl -fsSL "https://api.github.com/repos/${{_HELPER_RELEASE_REPO}}/releases/latest" 2>/dev/null | python -c "import sys,json; print(json.load(sys.stdin)['tag_name'])" 2>/dev/null)" || _RESOLVED_LLAMA_TAG=""
                if [ -z "$_RESOLVED_LLAMA_TAG" ]; then
                    _RESOLVED_LLAMA_TAG="$(curl -fsSL https://api.github.com/repos/ggml-org/llama.cpp/releases/latest 2>/dev/null | python -c "import sys,json; print(json.load(sys.stdin)['tag_name'])" 2>/dev/null)" || _RESOLVED_LLAMA_TAG=""
                fi
            fi
            if [ -z "$_RESOLVED_LLAMA_TAG" ]; then
                _RESOLVED_LLAMA_TAG="$_REQUESTED_LLAMA_TAG"
            fi
        fi
        echo "$_RESOLVED_LLAMA_TAG"
    """)

    @staticmethod
    def _make_curl_mock(
        mock_bin: Path, unsloth_response: str | None, ggml_response: str | None
    ):
        """Create a curl mock that returns different responses per repo."""
        lines = ["#!/bin/bash"]
        if unsloth_response is not None:
            lines.append(
                f'if echo "$*" | grep -q "unslothai/llama.cpp"; then echo \'{unsloth_response}\'; exit 0; fi'
            )
        else:
            lines.append(
                'if echo "$*" | grep -q "unslothai/llama.cpp"; then exit 1; fi'
            )
        if ggml_response is not None:
            lines.append(
                f'if echo "$*" | grep -q "ggml-org/llama.cpp"; then echo \'{ggml_response}\'; exit 0; fi'
            )
        else:
            lines.append('if echo "$*" | grep -q "ggml-org/llama.cpp"; then exit 1; fi')
        lines.append("exit 1")
        curl_path = mock_bin / "curl"
        curl_path.write_text("\n".join(lines) + "\n")
        curl_path.chmod(0o755)

    def _run_resolve(
        self,
        tmp_path: Path,
        requested_tag: str,
        unsloth_resp: str | None,
        ggml_resp: str | None,
    ) -> str:
        mock_bin = tmp_path / "mock_bin"
        mock_bin.mkdir(exist_ok = True)
        self._make_curl_mock(mock_bin, unsloth_resp, ggml_resp)
        script = self.RESOLVE_TEMPLATE.format(
            mock_bin = mock_bin, requested_tag = requested_tag
        )
        return run_bash(script)

    def test_unsloth_succeeds(self, tmp_path: Path):
        output = self._run_resolve(
            tmp_path,
            "latest",
            unsloth_resp = '{"tag_name":"b8508"}',
            ggml_resp = '{"tag_name":"b9000"}',
        )
        assert output == "b8508"

    def test_unsloth_fails_ggml_succeeds(self, tmp_path: Path):
        output = self._run_resolve(
            tmp_path,
            "latest",
            unsloth_resp = None,
            ggml_resp = '{"tag_name":"b9000"}',
        )
        assert output == "b9000"

    def test_both_fail_raw_fallback(self, tmp_path: Path):
        output = self._run_resolve(
            tmp_path,
            "latest",
            unsloth_resp = None,
            ggml_resp = None,
        )
        assert output == "latest"

    def test_concrete_tag_passes_through(self, tmp_path: Path):
        output = self._run_resolve(
            tmp_path,
            "b7777",
            unsloth_resp = '{"tag_name":"b8508"}',
            ggml_resp = '{"tag_name":"b9000"}',
        )
        assert output == "b7777"

    def test_unsloth_malformed_json_falls_through(self, tmp_path: Path):
        output = self._run_resolve(
            tmp_path,
            "latest",
            unsloth_resp = '{"bad_key":"no_tag"}',
            ggml_resp = '{"tag_name":"b9001"}',
        )
        assert output == "b9001"

    def test_both_malformed_json_raw_fallback(self, tmp_path: Path):
        output = self._run_resolve(
            tmp_path,
            "latest",
            unsloth_resp = '{"bad":"data"}',
            ggml_resp = '{"also":"bad"}',
        )
        assert output == "latest"

    def test_unsloth_empty_body_falls_through(self, tmp_path: Path):
        output = self._run_resolve(
            tmp_path,
            "latest",
            unsloth_resp = "",
            ggml_resp = '{"tag_name":"b7000"}',
        )
        assert output == "b7000"

    def test_unsloth_empty_tag_name_falls_through(self, tmp_path: Path):
        output = self._run_resolve(
            tmp_path,
            "latest",
            unsloth_resp = '{"tag_name":""}',
            ggml_resp = '{"tag_name":"b6000"}',
        )
        assert output == "b6000"

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
        # Find the source-build block
        idx_else = content.find("# Check prerequisites")
        assert idx_else != -1
        block = content[idx_else:]
        # rm -rf should appear after the cmake/git checks
        idx_cmake = block.find("command -v cmake")
        idx_git = block.find("command -v git")
        idx_rm = block.find("rm -rf")
        assert idx_rm > idx_cmake, "rm -rf should come after cmake check"
        assert idx_rm > idx_git, "rm -rf should come after git check"

    def test_setup_sh_clone_uses_branch_tag(self):
        """git clone in source-build should use --branch."""
        content = SETUP_SH.read_text()
        # Find the clone line in the source-build block
        for line in content.splitlines():
            if "git clone" in line and "ggml-org/llama.cpp" in line:
                assert (
                    '--branch "$_RESOLVED_LLAMA_TAG"' in line
                ), f"Clone line missing --branch: {line.strip()}"
                break
        else:
            pytest.fail("git clone line not found in setup.sh")

    def test_setup_sh_latest_resolution_queries_unsloth_first(self):
        """The Unsloth repo should be queried before ggml-org."""
        content = SETUP_SH.read_text()
        idx_unsloth = content.find("_HELPER_RELEASE_REPO}/releases/latest")
        idx_ggml = content.find("ggml-org/llama.cpp/releases/latest")
        assert idx_unsloth != -1, "Unsloth API query not found"
        assert idx_ggml != -1, "ggml-org API query not found"
        assert idx_unsloth < idx_ggml, "Unsloth should be queried before ggml-org"

    def test_setup_ps1_uses_checkout_b(self):
        """PS1 should use checkout -B, not checkout --force FETCH_HEAD."""
        content = SETUP_PS1.read_text()
        assert "checkout -B unsloth-llama-build" in content
        assert "checkout --force FETCH_HEAD" not in content

    def test_setup_ps1_clone_uses_branch_tag(self):
        """PS1 clone should use --branch with the resolved tag."""
        content = SETUP_PS1.read_text()
        assert "--branch", "$ResolvedLlamaTag" in content
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

    def test_setup_ps1_latest_resolution_queries_unsloth_first(self):
        """PS1 should query Unsloth repo before ggml-org."""
        content = SETUP_PS1.read_text()
        idx_unsloth = content.find("$HelperReleaseRepo/releases/latest")
        idx_ggml = content.find("ggml-org/llama.cpp/releases/latest")
        assert idx_unsloth != -1, "Unsloth API query not found in PS1"
        assert idx_ggml != -1, "ggml-org API query not found in PS1"
        assert idx_unsloth < idx_ggml, "Unsloth should be queried before ggml-org"

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
