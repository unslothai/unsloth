"""
Tests for the current llama.cpp wrapper policy in setup.sh / setup.ps1.

Tests cover:
  - Bash subprocess: PR_FORCE promotion, user-override, zero/empty/invalid ignored
  - Bash subprocess: source remains pinned to ggml-org even if env source is set
  - Static source checks: mainline repo/source are hardcoded for now
  - PowerShell subprocess: PR_FORCE promotion and fixed-source parity

Run: pytest tests/studio/install/test_llama_pr_force_and_source.py -v
"""

import os
import shlex
import subprocess
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
SETUP_SH = PACKAGE_ROOT / "studio" / "setup.sh"
SETUP_PS1 = PACKAGE_ROOT / "studio" / "setup.ps1"

BASH = "/bin/bash"
PWSH = "/usr/bin/pwsh"
PWSH_AVAILABLE = os.path.isfile(PWSH) and os.access(PWSH, os.X_OK)
requires_pwsh = pytest.mark.skipif(not PWSH_AVAILABLE, reason = "pwsh not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_bash(
    script: str, *, timeout: int = 10, env: dict | None = None
) -> subprocess.CompletedProcess:
    """Run a bash script fragment and return the CompletedProcess."""
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    return subprocess.run(
        [BASH, "-c", script],
        capture_output = True,
        text = True,
        timeout = timeout,
        env = run_env,
    )


def run_pwsh(
    script: str, *, timeout: int = 10, env: dict | None = None
) -> subprocess.CompletedProcess:
    """Run a PowerShell script fragment and return the CompletedProcess."""
    run_env = os.environ.copy()
    run_env["NO_COLOR"] = "1"
    if env:
        run_env.update(env)
    return subprocess.run(
        [PWSH, "-NoProfile", "-Command", script],
        capture_output = True,
        text = True,
        timeout = timeout,
        env = run_env,
    )


# ---------------------------------------------------------------------------
# Shared bash stubs
# ---------------------------------------------------------------------------
BASH_STUBS = textwrap.dedent("""\
    step()    { echo "step:$1:$2"; }
    substep() { :; }
    verbose_substep() { :; }
    print_llama_error_log() { :; }
    C_ERR= C_WARN= C_OK= C_RST= C_TITLE= C_DIM=
""")

RUN_QUIET_STUB = textwrap.dedent("""\
    run_quiet_no_exit() { local _label="$1"; shift; "$@"; return $?; }
""")


def make_mock_git(tmp_path: Path, *, fail_on: str = "") -> tuple[Path, Path]:
    """Create a mock git binary that logs calls. Returns (mock_bin, log_file)."""
    mock_bin = tmp_path / "mock_bin"
    mock_bin.mkdir(exist_ok = True)
    log_file = tmp_path / "git_calls.log"

    if fail_on:
        script = (
            f'#!/bin/bash\necho "$*" >> {log_file}\n'
            f'_args=("$@")\n'
            f"_i=0\n"
            f'while [ "${{_args[$_i]:-}}" = "-C" ]; do _i=$((_i+2)); done\n'
            f'_subcmd="${{_args[$_i]:-}}"\n'
            f'if [ "$_subcmd" = "{fail_on}" ]; then exit 1; fi\n'
            f"exit 0\n"
        )
    else:
        script = f'#!/bin/bash\necho "$*" >> {log_file}\nexit 0\n'

    git_bin = mock_bin / "git"
    git_bin.write_text(script)
    git_bin.chmod(0o755)
    return mock_bin, log_file


# =========================================================================
# Bash fragment that exercises PR_FORCE and fixed _LLAMA_SOURCE resolution
# =========================================================================
def _bash_resolution_fragment(
    llama_pr: str = "",
    llama_pr_force: str = "",
    llama_source: str = "",
    default_pr_force: str = "",
    default_source: str = "https://github.com/ggml-org/llama.cpp",
) -> str:
    """Build the bash fragment that mirrors setup.sh resolution logic."""
    return BASH_STUBS + textwrap.dedent(f"""\
        _LLAMA_PR={shlex.quote(llama_pr) if llama_pr else '""'}
        _DEFAULT_LLAMA_PR_FORCE={shlex.quote(default_pr_force) if default_pr_force else '""'}
        _DEFAULT_LLAMA_SOURCE={shlex.quote(default_source)}

        _LLAMA_PR_FORCE={shlex.quote(llama_pr_force) if llama_pr_force else '"$_DEFAULT_LLAMA_PR_FORCE"'}
        export UNSLOTH_LLAMA_SOURCE={shlex.quote(llama_source) if llama_source else '""'}
        _LLAMA_SOURCE="$_DEFAULT_LLAMA_SOURCE"
        _LLAMA_SOURCE="${{_LLAMA_SOURCE%.git}}"

        _NEED_LLAMA_SOURCE_BUILD=false
        _SKIP_PREBUILT_INSTALL=false

        if [ "$_LLAMA_SOURCE" != "https://github.com/ggml-org/llama.cpp" ]; then
            step "llama.cpp" "custom source: $_LLAMA_SOURCE -- forcing source build"
            _NEED_LLAMA_SOURCE_BUILD=true
            _SKIP_PREBUILT_INSTALL=true
        fi

        if [ -z "$_LLAMA_PR" ] && [ -n "$_LLAMA_PR_FORCE" ] && \\
           [[ "$_LLAMA_PR_FORCE" =~ ^[0-9]+$ ]] && [ "$_LLAMA_PR_FORCE" -gt 0 ]; then
            _LLAMA_PR="$_LLAMA_PR_FORCE"
            step "llama.cpp" "baked-in PR_FORCE=$_LLAMA_PR_FORCE"
        fi

        echo "LLAMA_PR=$_LLAMA_PR"
        echo "LLAMA_SOURCE=$_LLAMA_SOURCE"
        echo "NEED_SOURCE=$_NEED_LLAMA_SOURCE_BUILD"
        echo "SKIP_PREBUILT=$_SKIP_PREBUILT_INSTALL"
    """)


# =========================================================================
# TEST GROUP A: Bash PR_FORCE promotion (subprocess)
# =========================================================================
class TestBashPrForcePromotion:
    """PR_FORCE promotes to _LLAMA_PR when user hasn't set one."""

    def test_baked_in_pr_force_promotes(self):
        script = _bash_resolution_fragment(default_pr_force = "12345")
        r = run_bash(script)
        assert r.returncode == 0
        assert "LLAMA_PR=12345" in r.stdout
        assert "baked-in PR_FORCE=12345" in r.stdout

    def test_env_pr_force_promotes(self):
        script = _bash_resolution_fragment(llama_pr_force = "999")
        r = run_bash(script)
        assert r.returncode == 0
        assert "LLAMA_PR=999" in r.stdout

    def test_user_pr_overrides_pr_force(self):
        """UNSLOTH_LLAMA_PR takes priority over PR_FORCE."""
        script = _bash_resolution_fragment(
            llama_pr = "100",
            llama_pr_force = "200",
        )
        r = run_bash(script)
        assert r.returncode == 0
        assert "LLAMA_PR=100" in r.stdout
        assert "baked-in PR_FORCE" not in r.stdout

    def test_user_pr_overrides_baked_in(self):
        script = _bash_resolution_fragment(
            llama_pr = "100",
            default_pr_force = "200",
        )
        r = run_bash(script)
        assert r.returncode == 0
        assert "LLAMA_PR=100" in r.stdout
        assert "baked-in PR_FORCE" not in r.stdout

    def test_pr_force_zero_ignored(self):
        script = _bash_resolution_fragment(llama_pr_force = "0")
        r = run_bash(script)
        assert r.returncode == 0
        assert "LLAMA_PR=" in r.stdout
        assert "baked-in PR_FORCE" not in r.stdout

    def test_pr_force_empty_ignored(self):
        script = _bash_resolution_fragment(default_pr_force = "")
        r = run_bash(script)
        assert r.returncode == 0
        assert "LLAMA_PR=" in r.stdout
        assert "baked-in PR_FORCE" not in r.stdout

    def test_pr_force_alpha_ignored(self):
        script = _bash_resolution_fragment(llama_pr_force = "abc")
        r = run_bash(script)
        assert r.returncode == 0
        assert "LLAMA_PR=" in r.stdout
        assert "baked-in PR_FORCE" not in r.stdout

    def test_pr_force_negative_ignored(self):
        script = _bash_resolution_fragment(llama_pr_force = "-5")
        r = run_bash(script)
        assert r.returncode == 0
        assert "LLAMA_PR=" in r.stdout

    def test_pr_force_decimal_ignored(self):
        script = _bash_resolution_fragment(llama_pr_force = "12.34")
        r = run_bash(script)
        assert r.returncode == 0
        assert "LLAMA_PR=" in r.stdout


# =========================================================================
# TEST GROUP B: Bash fixed mainline source (subprocess)
# =========================================================================
class TestBashFixedMainlineSource:
    """Source remains pinned to ggml-org while the temporary policy is active."""

    def test_default_source_no_force(self):
        script = _bash_resolution_fragment()
        r = run_bash(script)
        assert r.returncode == 0
        assert "NEED_SOURCE=false" in r.stdout
        assert "SKIP_PREBUILT=false" in r.stdout
        assert "custom source:" not in r.stdout

    def test_env_source_override_is_ignored(self):
        script = _bash_resolution_fragment(
            llama_source = "https://github.com/unslothai/llama.cpp.git",
        )
        r = run_bash(script)
        assert r.returncode == 0
        assert "LLAMA_SOURCE=https://github.com/ggml-org/llama.cpp" in r.stdout
        assert "NEED_SOURCE=false" in r.stdout
        assert "SKIP_PREBUILT=false" in r.stdout

    def test_baked_in_source_stays_mainline(self):
        script = _bash_resolution_fragment(
            default_source = "https://github.com/ggml-org/llama.cpp",
        )
        r = run_bash(script)
        assert r.returncode == 0
        assert "LLAMA_SOURCE=https://github.com/ggml-org/llama.cpp" in r.stdout


# =========================================================================
# TEST GROUP C: Bash clone URL parameterization (subprocess with mock git)
# =========================================================================
class TestBashCloneUrlParameterized:
    """Verify git clone uses _LLAMA_SOURCE instead of hardcoded URL."""

    @staticmethod
    def _clone_script(
        mock_bin: Path,
        build_tmp: str,
        llama_pr: str = "",
        llama_source: str = "https://github.com/ggml-org/llama.cpp",
        resolved_tag: str = "b8508",
    ) -> str:
        return RUN_QUIET_STUB + textwrap.dedent(f"""\
            export PATH="{mock_bin}:$PATH"
            _LLAMA_PR={shlex.quote(llama_pr) if llama_pr else '""'}
            _LLAMA_SOURCE={shlex.quote(llama_source)}
            _RESOLVED_LLAMA_TAG={shlex.quote(resolved_tag)}
            _BUILD_TMP={shlex.quote(build_tmp)}
            BUILD_OK=true

            if [ -n "$_LLAMA_PR" ]; then
                run_quiet_no_exit "clone llama.cpp" \\
                    git clone --depth 1 "${{_LLAMA_SOURCE}}.git" "$_BUILD_TMP" || BUILD_OK=false
            else
                _CLONE_ARGS=(git clone --depth 1)
                if [ "$_RESOLVED_LLAMA_TAG" != "latest" ] && [ -n "$_RESOLVED_LLAMA_TAG" ]; then
                    _CLONE_ARGS+=(--branch "$_RESOLVED_LLAMA_TAG")
                fi
                _CLONE_ARGS+=("${{_LLAMA_SOURCE}}.git" "$_BUILD_TMP")
                run_quiet_no_exit "clone llama.cpp" \\
                    "${{_CLONE_ARGS[@]}}" || BUILD_OK=false
            fi
            echo "BUILD_OK=$BUILD_OK"
        """)

    def test_pr_path_uses_custom_source(self, tmp_path: Path):
        mock_bin, log_file = make_mock_git(tmp_path)
        build_tmp = str(tmp_path / "build_tmp")
        script = self._clone_script(
            mock_bin,
            build_tmp,
            llama_pr = "123",
            llama_source = "https://github.com/unslothai/llama.cpp",
        )
        r = run_bash(script)
        assert r.returncode == 0
        log = log_file.read_text()
        assert "unslothai/llama.cpp.git" in log
        assert "ggml-org" not in log

    def test_non_pr_path_uses_custom_source(self, tmp_path: Path):
        mock_bin, log_file = make_mock_git(tmp_path)
        build_tmp = str(tmp_path / "build_tmp")
        script = self._clone_script(
            mock_bin,
            build_tmp,
            llama_source = "https://github.com/unslothai/llama.cpp",
        )
        r = run_bash(script)
        assert r.returncode == 0
        log = log_file.read_text()
        assert "unslothai/llama.cpp.git" in log
        assert "ggml-org" not in log

    def test_default_source_unchanged(self, tmp_path: Path):
        mock_bin, log_file = make_mock_git(tmp_path)
        build_tmp = str(tmp_path / "build_tmp")
        script = self._clone_script(mock_bin, build_tmp)
        r = run_bash(script)
        assert r.returncode == 0
        log = log_file.read_text()
        assert "ggml-org/llama.cpp.git" in log

    def test_latest_tag_omits_branch_flag(self, tmp_path: Path):
        """resolved_tag='latest' should not pass --branch to git clone."""
        mock_bin, log_file = make_mock_git(tmp_path)
        build_tmp = str(tmp_path / "build_tmp")
        script = self._clone_script(
            mock_bin,
            build_tmp,
            resolved_tag = "latest",
        )
        r = run_bash(script)
        assert r.returncode == 0
        log = log_file.read_text()
        assert "--branch" not in log
        assert "ggml-org/llama.cpp.git" in log

    def test_empty_tag_omits_branch_flag(self, tmp_path: Path):
        """resolved_tag='' (empty) should not pass --branch to git clone."""
        mock_bin, log_file = make_mock_git(tmp_path)
        build_tmp = str(tmp_path / "build_tmp")
        script = self._clone_script(
            mock_bin,
            build_tmp,
            resolved_tag = "",
        )
        r = run_bash(script)
        assert r.returncode == 0
        log = log_file.read_text()
        assert "--branch" not in log
        assert "ggml-org/llama.cpp.git" in log


# =========================================================================
# TEST GROUP D: Static source patterns -- setup.sh
# =========================================================================
class TestSourcePatternsSh:
    """Verify setup.sh keeps the temporary mainline-only llama.cpp policy."""

    @pytest.fixture(autouse = True)
    def _load_source(self):
        self.content = SETUP_SH.read_text()

    def test_has_default_pr_force(self):
        assert '_DEFAULT_LLAMA_PR_FORCE=""' in self.content

    def test_has_default_source(self):
        assert (
            '_DEFAULT_LLAMA_SOURCE="https://github.com/ggml-org/llama.cpp"'
            in self.content
        )

    def test_has_pr_force_env_read(self):
        assert "UNSLOTH_LLAMA_PR_FORCE" in self.content

    def test_source_env_override_removed(self):
        assert "UNSLOTH_LLAMA_SOURCE:-${_DEFAULT_LLAMA_SOURCE}" not in self.content
        assert '_LLAMA_SOURCE="${_DEFAULT_LLAMA_SOURCE}"' in self.content

    def test_release_repo_override_removed(self):
        assert "UNSLOTH_LLAMA_RELEASE_REPO:-unslothai/llama.cpp" not in self.content
        assert '_HELPER_RELEASE_REPO="ggml-org/llama.cpp"' in self.content

    def test_force_compile_skips_prebuilt_resolution_early(self):
        assert 'if [ "$_LLAMA_FORCE_COMPILE" = "1" ]; then' in self.content
        assert "_SKIP_PREBUILT_INSTALL=true" in self.content

    def test_force_compile_uses_requested_tag_without_helper(self):
        assert 'if [ "$_LLAMA_FORCE_COMPILE" = "1" ]; then' in self.content
        assert '_RESOLVED_LLAMA_TAG="$_REQUESTED_LLAMA_TAG"' in self.content

    def test_pr_force_resolution_block(self):
        assert '_LLAMA_PR="$_LLAMA_PR_FORCE"' in self.content

    def test_source_trailing_git_strip(self):
        assert "${_LLAMA_SOURCE%.git}" in self.content

    def test_clone_urls_parameterized_pr_path(self):
        """PR clone path uses ${_LLAMA_SOURCE}.git, not hardcoded URL."""
        pr_clone_idx = self.content.index(
            'if [ -n "$_LLAMA_PR" ]; then\n'
            '            run_quiet_no_exit "clone llama.cpp"'
        )
        else_idx = self.content.index("else\n", pr_clone_idx)
        pr_block = self.content[pr_clone_idx:else_idx]
        assert '"${_LLAMA_SOURCE}.git"' in pr_block
        assert "ggml-org/llama.cpp.git" not in pr_block

    def test_clone_urls_parameterized_tag_path(self):
        """Non-PR clone path uses the resolved source URL, not a hardcoded URL."""
        # Find the non-PR clone line (after _CLONE_ARGS)
        idx = self.content.index("_CLONE_ARGS=(git clone --depth 1)")
        block = self.content[idx : idx + 400]
        assert '"${_RESOLVED_SOURCE_URL}.git"' in block
        assert "ggml-org/llama.cpp.git" not in block

    def test_no_hardcoded_clone_urls(self):
        """No remaining hardcoded ggml-org clone URLs in clone commands."""
        lines = self.content.splitlines()
        for i, line in enumerate(lines, 1):
            if "git clone" in line and "ggml-org/llama.cpp.git" in line:
                pytest.fail(
                    f"Line {i} has hardcoded ggml-org clone URL: {line.strip()}"
                )


# =========================================================================
# TEST GROUP E: Static source patterns -- setup.ps1
# =========================================================================
class TestSourcePatternsPs1:
    """Verify setup.ps1 keeps the temporary mainline-only llama.cpp policy."""

    @pytest.fixture(autouse = True)
    def _load_source(self):
        self.content = SETUP_PS1.read_text()

    def test_has_default_pr_force(self):
        assert '$DefaultLlamaPrForce = ""' in self.content

    def test_has_default_source(self):
        assert (
            '$DefaultLlamaSource = "https://github.com/ggml-org/llama.cpp"'
            in self.content
        )

    def test_has_pr_force_env_read(self):
        assert "$env:UNSLOTH_LLAMA_PR_FORCE" in self.content

    def test_source_env_override_removed(self):
        assert "$LlamaSource = if ($env:UNSLOTH_LLAMA_SOURCE)" not in self.content
        assert "$LlamaSource = $DefaultLlamaSource" in self.content

    def test_release_repo_override_removed(self):
        assert (
            "$HelperReleaseRepo = if ($env:UNSLOTH_LLAMA_RELEASE_REPO)"
            not in self.content
        )
        assert '$HelperReleaseRepo = "ggml-org/llama.cpp"' in self.content

    def test_force_compile_skips_prebuilt_resolution_early(self):
        assert 'if ($env:UNSLOTH_LLAMA_FORCE_COMPILE -eq "1") {' in self.content
        assert "$SkipPrebuiltInstall = $true" in self.content

    def test_force_compile_uses_requested_tag_without_helper(self):
        assert 'if ($env:UNSLOTH_LLAMA_FORCE_COMPILE -eq "1") {' in self.content
        assert "$ResolvedLlamaTag = $RequestedLlamaTag" in self.content

    def test_pr_force_promotion_block(self):
        assert "$LlamaPr = $LlamaPrForce" in self.content

    def test_source_trailing_git_strip(self):
        assert ".EndsWith('.git')" in self.content

    def test_clone_urls_parameterized_pr_path(self):
        """PR clone path uses $LlamaSource.git, not hardcoded URL."""
        pr_idx = self.content.index(
            "if ($LlamaPr) {\n", self.content.index("Cloning llama.cpp")
        )
        else_idx = self.content.index("} else {", pr_idx)
        pr_block = self.content[pr_idx:else_idx]
        assert '"$LlamaSource.git"' in pr_block
        assert "ggml-org/llama.cpp.git" not in pr_block

    def test_clone_urls_parameterized_tag_path(self):
        """Non-PR clone path uses the resolved source URL, not a hardcoded URL."""
        clone_args_idx = self.content.index('$cloneArgs = @("clone"')
        block = self.content[clone_args_idx : clone_args_idx + 400]
        assert '"$ResolvedSourceUrl.git"' in block
        assert "ggml-org/llama.cpp.git" not in block

    def test_no_hardcoded_clone_urls(self):
        """No remaining hardcoded ggml-org clone URLs in clone commands."""
        lines = self.content.splitlines()
        for i, line in enumerate(lines, 1):
            if "git clone" in line and "ggml-org/llama.cpp.git" in line:
                pytest.fail(
                    f"Line {i} has hardcoded ggml-org clone URL: {line.strip()}"
                )


# =========================================================================
# TEST GROUP F: PowerShell PR_FORCE promotion (subprocess)
# =========================================================================
@requires_pwsh
class TestPwshPrForcePromotion:
    """PR_FORCE promotion and fixed-source logic via pwsh subprocess."""

    FRAGMENT_TEMPLATE = textwrap.dedent("""\
        function step($a, $b, $c) { Write-Output "step:$a`:$b" }

        $DefaultLlamaPrForce = "%%DEFAULT_PR_FORCE%%"
        $DefaultLlamaSource = "%%DEFAULT_SOURCE%%"

        $LlamaPr = if ($env:UNSLOTH_LLAMA_PR) { $env:UNSLOTH_LLAMA_PR.Trim() } else { "" }
        $LlamaPrForce = if ($env:UNSLOTH_LLAMA_PR_FORCE) { $env:UNSLOTH_LLAMA_PR_FORCE.Trim() } else { $DefaultLlamaPrForce }
        $LlamaSource = $DefaultLlamaSource
        if ($LlamaSource.EndsWith('.git')) { $LlamaSource = $LlamaSource.Substring(0, $LlamaSource.Length - 4) }

        $NeedLlamaSourceBuild = $false
        $SkipPrebuiltInstall = $false

        if ($LlamaSource -ne "https://github.com/ggml-org/llama.cpp") {
            step "llama.cpp" "custom source: $LlamaSource -- forcing source build" "Yellow"
            $NeedLlamaSourceBuild = $true
            $SkipPrebuiltInstall = $true
        }

        if (-not $LlamaPr -and $LlamaPrForce -and $LlamaPrForce -match '^\\d+$' -and [int]$LlamaPrForce -gt 0) {
            $LlamaPr = $LlamaPrForce
            step "llama.cpp" "baked-in PR_FORCE=$LlamaPrForce" "Yellow"
        }

        Write-Output "LLAMA_PR=$LlamaPr"
        Write-Output "LLAMA_SOURCE=$LlamaSource"
        Write-Output "NEED_SOURCE=$NeedLlamaSourceBuild"
        Write-Output "SKIP_PREBUILT=$SkipPrebuiltInstall"
    """)

    def _run(
        self,
        default_pr_force: str = "",
        default_source: str = "https://github.com/ggml-org/llama.cpp",
        env: dict | None = None,
    ) -> subprocess.CompletedProcess:
        script = self.FRAGMENT_TEMPLATE.replace(
            "%%DEFAULT_PR_FORCE%%",
            default_pr_force,
        ).replace(
            "%%DEFAULT_SOURCE%%",
            default_source,
        )
        run_env = {}
        # Ensure env vars are unset by default
        run_env["UNSLOTH_LLAMA_PR"] = ""
        run_env["UNSLOTH_LLAMA_PR_FORCE"] = ""
        if env:
            run_env.update(env)
        return run_pwsh(script, env = run_env)

    def test_baked_in_pr_force_promotes(self):
        r = self._run(default_pr_force = "12345")
        assert r.returncode == 0
        assert "LLAMA_PR=12345" in r.stdout
        assert "baked-in PR_FORCE=12345" in r.stdout

    def test_env_pr_force_promotes(self):
        r = self._run(env = {"UNSLOTH_LLAMA_PR_FORCE": "999"})
        assert r.returncode == 0
        assert "LLAMA_PR=999" in r.stdout

    def test_user_pr_overrides_pr_force(self):
        r = self._run(
            env = {
                "UNSLOTH_LLAMA_PR": "100",
                "UNSLOTH_LLAMA_PR_FORCE": "200",
            }
        )
        assert r.returncode == 0
        assert "LLAMA_PR=100" in r.stdout
        assert "baked-in PR_FORCE" not in r.stdout

    def test_pr_force_zero_ignored(self):
        r = self._run(env = {"UNSLOTH_LLAMA_PR_FORCE": "0"})
        assert r.returncode == 0
        assert "LLAMA_PR=" in r.stdout
        assert "baked-in PR_FORCE" not in r.stdout

    def test_pr_force_alpha_ignored(self):
        r = self._run(env = {"UNSLOTH_LLAMA_PR_FORCE": "abc"})
        assert r.returncode == 0
        assert "baked-in PR_FORCE" not in r.stdout

    def test_env_source_override_is_ignored(self):
        r = self._run(
            env = {
                "UNSLOTH_LLAMA_SOURCE": "https://github.com/unslothai/llama.cpp",
            }
        )
        assert r.returncode == 0
        assert "LLAMA_SOURCE=https://github.com/ggml-org/llama.cpp" in r.stdout
        assert "NEED_SOURCE=False" in r.stdout
        assert "SKIP_PREBUILT=False" in r.stdout

    def test_default_source_no_force(self):
        r = self._run()
        assert r.returncode == 0
        assert "NEED_SOURCE=False" in r.stdout
        assert "SKIP_PREBUILT=False" in r.stdout

    def test_trailing_git_override_is_ignored(self):
        r = self._run(
            env = {
                "UNSLOTH_LLAMA_SOURCE": "https://github.com/unslothai/llama.cpp.git",
            }
        )
        assert r.returncode == 0
        assert "LLAMA_SOURCE=https://github.com/ggml-org/llama.cpp" in r.stdout

    def test_baked_in_source_stays_mainline(self):
        r = self._run(default_source = "https://github.com/ggml-org/llama.cpp")
        assert r.returncode == 0
        assert "LLAMA_SOURCE=https://github.com/ggml-org/llama.cpp" in r.stdout
