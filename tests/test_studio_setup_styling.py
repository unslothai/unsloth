"""Tests for PR #4494 Studio setup log styling.

Validates terminal output styling across:
- studio/backend/startup_banner.py
- studio/install_python_stack.py
- studio/setup.sh  (bash subprocess tests)
- studio/setup.ps1 (structural/static analysis only -- no pwsh on Linux)
- unsloth_cli/commands/studio.py (CLI tests via typer CliRunner)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root is importable even without pip install -e .
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

SETUP_SH = _REPO / "studio" / "setup.sh"
SETUP_PS1 = _REPO / "studio" / "setup.ps1"

# ── Bash fragments inlined from setup.sh for subprocess tests ─────────

_BASH_COLOR_INIT = r"""
if [ -n "${NO_COLOR:-}" ]; then
    C_TITLE= C_DIM= C_OK= C_WARN= C_ERR= C_RST=
elif [ -t 1 ] || [ -n "${FORCE_COLOR:-}" ]; then
    C_TITLE=$'\033[38;5;150m'
    C_DIM=$'\033[38;5;245m'
    C_OK=$'\033[38;5;108m'
    C_WARN=$'\033[38;5;136m'
    C_ERR=$'\033[91m'
    C_RST=$'\033[0m'
else
    C_TITLE= C_DIM= C_OK= C_WARN= C_ERR= C_RST=
fi
"""

_BASH_HELPERS = r"""
step()    { printf "  ${C_DIM}%-15s${C_RST}${3:-$C_OK}%s${C_RST}\n" "$1" "$2"; }
substep() { printf "  ${C_DIM}%-15s%s${C_RST}\n" "" "$1"; }
"""

_BASH_RUN_QUIET = r"""
_run_quiet() {
    local on_fail=$1
    local label=$2
    shift 2
    local tmplog
    tmplog=$(mktemp) || {
        step "error" "Failed to create temporary file" "$C_ERR"
        [ "$on_fail" = "exit" ] && exit 1 || return 1
    }
    if "$@" >"$tmplog" 2>&1; then
        rm -f "$tmplog"
        return 0
    else
        local exit_code=$?
        step "error" "$label failed (exit code $exit_code)" "$C_ERR"
        cat "$tmplog" >&2
        rm -f "$tmplog"
        if [ "$on_fail" = "exit" ]; then
            exit "$exit_code"
        else
            return "$exit_code"
        fi
    fi
}
run_quiet() { _run_quiet exit "$@"; }
run_quiet_no_exit() { _run_quiet return "$@"; }
"""

_BASH_TRY_QUIET = r"""
try_quiet() {
    local label="$1"; shift
    local tmplog; tmplog=$(mktemp)
    if "$@" > "$tmplog" 2>&1; then
        rm -f "$tmplog"
        return 0
    else
        local exit_code=$?
        if [ "${UNSLOTH_VERBOSE:-0}" = "1" ]; then
            step "error" "$label failed (exit code $exit_code)" "$C_ERR"
            cat "$tmplog" >&2
        fi
        rm -f "$tmplog"
        return $exit_code
    fi
}
"""

# Template for REQUESTED_PYTHON_VERSION tests (uses .replace() placeholders)
_REQUESTED_PY_TPL = r"""
C_DIM= C_OK= C_WARN= C_ERR= C_RST= C_TITLE=
step()    { printf "  %-15s%s\n" "$1" "$2"; }
substep() { printf "  %-15s%s\n" "" "$1"; }
MIN_PY_MINOR=__MIN__
MAX_PY_MINOR=__MAX__
BEST_PY=""
BEST_MINOR=0
REQUESTED_PYTHON_VERSION="__REQ__"
if [ -n "${REQUESTED_PYTHON_VERSION:-}" ] && [ -x "$REQUESTED_PYTHON_VERSION" ]; then
    _req_ver=$("$REQUESTED_PYTHON_VERSION" --version 2>&1 | awk '{print $2}')
    _req_major=$(echo "$_req_ver" | cut -d. -f1)
    _req_minor=$(echo "$_req_ver" | cut -d. -f2)
    if [ "$_req_major" -eq 3 ] 2>/dev/null && \
       [ "$_req_minor" -ge "$MIN_PY_MINOR" ] 2>/dev/null && \
       [ "$_req_minor" -le "$MAX_PY_MINOR" ] 2>/dev/null; then
        BEST_PY="$REQUESTED_PYTHON_VERSION"
        substep "using requested Python: $BEST_PY"
    else
        substep "ignoring requested Python $REQUESTED_PYTHON_VERSION ($_req_ver) -- outside range"
    fi
fi
echo "BEST_PY=$BEST_PY"
"""


def _run_bash(
    script: str, env_extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a bash script fragment with a sanitised environment."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("NO_COLOR", "FORCE_COLOR", "UNSLOTH_VERBOSE")
    }
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove colour / verbosity env vars before every test."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("UNSLOTH_VERBOSE", raising=False)


# ═══════════════════════════════════════════════════════════════════════
# 1. startup_banner.py -- stdout_supports_color()  (9 tests)
# ═══════════════════════════════════════════════════════════════════════


class TestBannerSupportsColor:
    """NO_COLOR > FORCE_COLOR > isatty."""

    @staticmethod
    def _call():
        from studio.backend.startup_banner import stdout_supports_color

        return stdout_supports_color()

    def test_no_tty_no_env(self):
        with patch("sys.stdout.isatty", return_value=False):
            assert self._call() is False

    def test_tty_returns_true(self):
        with patch("sys.stdout.isatty", return_value=True):
            assert self._call() is True

    def test_no_color_disables_with_tty(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        with patch("sys.stdout.isatty", return_value=True):
            assert self._call() is False

    def test_force_color_enables_no_tty(self, monkeypatch):
        monkeypatch.setenv("FORCE_COLOR", "1")
        with patch("sys.stdout.isatty", return_value=False):
            assert self._call() is True

    def test_force_color_zero_truthy(self, monkeypatch):
        monkeypatch.setenv("FORCE_COLOR", "0")
        with patch("sys.stdout.isatty", return_value=False):
            assert self._call() is True

    def test_force_color_empty_falsy(self, monkeypatch):
        monkeypatch.setenv("FORCE_COLOR", "")
        with patch("sys.stdout.isatty", return_value=False):
            assert self._call() is False

    def test_no_color_empty_no_disable(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "")
        with patch("sys.stdout.isatty", return_value=True):
            assert self._call() is True

    def test_no_color_whitespace_no_disable(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "   ")
        with patch("sys.stdout.isatty", return_value=True):
            assert self._call() is True

    def test_no_color_overrides_force(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("FORCE_COLOR", "1")
        with patch("sys.stdout.isatty", return_value=True):
            assert self._call() is False


# ═══════════════════════════════════════════════════════════════════════
# 2. startup_banner.py -- print_port_in_use_notice()  (3 tests)
# ═══════════════════════════════════════════════════════════════════════


class TestPortInUseNotice:

    def test_contains_both_ports(self, capsys):
        from studio.backend.startup_banner import print_port_in_use_notice

        with patch(
            "studio.backend.startup_banner.stdout_supports_color", return_value=False
        ):
            print_port_in_use_notice(8888, 8889)
        out = capsys.readouterr().out
        assert "8888" in out
        assert "8889" in out

    def test_no_color_no_ansi(self, capsys):
        from studio.backend.startup_banner import print_port_in_use_notice

        with patch(
            "studio.backend.startup_banner.stdout_supports_color", return_value=False
        ):
            print_port_in_use_notice(8888, 8889)
        assert "\033[" not in capsys.readouterr().out

    def test_color_has_ansi(self, capsys):
        from studio.backend.startup_banner import print_port_in_use_notice

        with patch(
            "studio.backend.startup_banner.stdout_supports_color", return_value=True
        ):
            print_port_in_use_notice(8888, 8889)
        assert "\033[" in capsys.readouterr().out


# ═══════════════════════════════════════════════════════════════════════
# 3. startup_banner.py -- print_studio_access_banner()  (11 tests)
# ═══════════════════════════════════════════════════════════════════════


class TestStudioAccessBanner:

    PORT = 8888

    @staticmethod
    def _call(*, bind_host, display_host, port=8888, color=False):
        from studio.backend.startup_banner import print_studio_access_banner

        with patch(
            "studio.backend.startup_banner.stdout_supports_color",
            return_value=color,
        ):
            print_studio_access_banner(
                port=port, bind_host=bind_host, display_host=display_host
            )

    @pytest.mark.parametrize(
        "bind_host,display_host,expect_local,from_another,bound_addr,api_host",
        [
            ("0.0.0.0", "192.168.1.100", "127.0.0.1", True, False, "127.0.0.1"),
            ("0.0.0.0", "127.0.0.1", "127.0.0.1", False, False, "127.0.0.1"),
            ("0.0.0.0", "0.0.0.0", "127.0.0.1", False, False, "127.0.0.1"),
            ("::", "2001:db8::1", "[::1]", True, False, "[::1]"),
            ("::1", "::1", "[::1]", False, False, "[::1]"),
            ("127.0.0.1", "127.0.0.1", "127.0.0.1", False, False, "127.0.0.1"),
            ("localhost", "localhost", "127.0.0.1", False, False, "127.0.0.1"),
            ("192.168.1.5", "192.168.1.5", "127.0.0.1", False, True, "192.168.1.5"),
        ],
        ids=[
            "wildcard_v4_ext",
            "wildcard_v4_loop",
            "wildcard_v4_self",
            "wildcard_v6_ext",
            "loopback_v6",
            "loopback_v4",
            "localhost",
            "specific_ip",
        ],
    )
    def test_hosts(
        self,
        capsys,
        bind_host,
        display_host,
        expect_local,
        from_another,
        bound_addr,
        api_host,
    ):
        self._call(bind_host=bind_host, display_host=display_host, color=False)
        out = capsys.readouterr().out
        assert f"http://{expect_local}:{self.PORT}" in out
        assert ("From another device" in out) == from_another
        assert ("Bound address" in out) == bound_addr
        assert f"http://{api_host}:{self.PORT}/api" in out

    def test_color_on_has_ansi(self, capsys):
        self._call(bind_host="127.0.0.1", display_host="127.0.0.1", color=True)
        assert "\033[" in capsys.readouterr().out

    def test_color_off_no_ansi(self, capsys):
        self._call(bind_host="127.0.0.1", display_host="127.0.0.1", color=False)
        assert "\033[" not in capsys.readouterr().out

    def test_api_urls_present(self, capsys):
        self._call(bind_host="0.0.0.0", display_host="10.0.0.1", color=False)
        out = capsys.readouterr().out
        assert "/api" in out
        assert "/api/health" in out


# ═══════════════════════════════════════════════════════════════════════
# 4. install_python_stack.py -- Color detection  (5 tests)
# ═══════════════════════════════════════════════════════════════════════


class TestInstallColorDetect:

    @staticmethod
    def _mod():
        import studio.install_python_stack as ips

        return ips

    def test_no_tty(self):
        ips = self._mod()
        with patch("sys.stdout.isatty", return_value=False):
            assert ips._stdout_supports_color() is False

    def test_tty_linux(self, monkeypatch):
        ips = self._mod()
        monkeypatch.setattr(ips, "IS_WINDOWS", False)
        with patch("sys.stdout.isatty", return_value=True):
            assert ips._stdout_supports_color() is True

    def test_force_color(self, monkeypatch):
        monkeypatch.setenv("FORCE_COLOR", "1")
        assert self._mod()._stdout_supports_color() is True

    def test_no_color(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        with patch("sys.stdout.isatty", return_value=True):
            assert self._mod()._stdout_supports_color() is False

    def test_windows_ctypes_fail(self, monkeypatch):
        ips = self._mod()
        monkeypatch.setattr(ips, "IS_WINDOWS", True)
        with patch("sys.stdout.isatty", return_value=True):
            # ctypes.windll does not exist on Linux, so the except branch fires
            assert ips._stdout_supports_color() is False


# ═══════════════════════════════════════════════════════════════════════
# 5. install_python_stack.py -- _step() formatting  (6 tests)
# ═══════════════════════════════════════════════════════════════════════


class TestInstallStep:

    @staticmethod
    def _mod():
        import studio.install_python_stack as ips

        return ips

    def test_short_label_padding(self, monkeypatch, capsys):
        ips = self._mod()
        monkeypatch.setattr(ips, "_HAS_COLOR", False)
        ips._step("deps", "ok")
        out = capsys.readouterr().out
        # "deps" (4 chars) + 11 spaces padding to fill 15-col layout
        assert "  deps" + " " * 11 + "ok" in out

    def test_exact_15_label(self, monkeypatch, capsys):
        ips = self._mod()
        monkeypatch.setattr(ips, "_HAS_COLOR", False)
        label = "x" * 15
        ips._step(label, "v")
        out = capsys.readouterr().out
        assert label + "v" in out

    def test_long_label(self, monkeypatch, capsys):
        ips = self._mod()
        monkeypatch.setattr(ips, "_HAS_COLOR", False)
        label = "x" * 20
        ips._step(label, "v")
        out = capsys.readouterr().out
        # Label is clamped to _COL (15) chars; value appears right after
        assert label[:15] + "v" in out
        assert label not in out

    def test_default_green(self, monkeypatch, capsys):
        ips = self._mod()
        monkeypatch.setattr(ips, "_HAS_COLOR", True)
        ips._step("deps", "ok")
        assert "\033[38;5;108m" in capsys.readouterr().out

    def test_custom_color_fn(self, monkeypatch, capsys):
        ips = self._mod()
        monkeypatch.setattr(ips, "_HAS_COLOR", True)
        ips._step("err", "fail", color_fn=ips._red)
        assert "\033[91m" in capsys.readouterr().out

    def test_no_color_no_ansi(self, monkeypatch, capsys):
        ips = self._mod()
        monkeypatch.setattr(ips, "_HAS_COLOR", False)
        ips._step("deps", "ok")
        assert "\033[" not in capsys.readouterr().out


# ═══════════════════════════════════════════════════════════════════════
# 6. install_python_stack.py -- _progress()  (4 tests)
# ═══════════════════════════════════════════════════════════════════════


class TestInstallProgress:

    @staticmethod
    def _mod():
        import studio.install_python_stack as ips

        return ips

    def test_verbose_no_output(self, monkeypatch, capsys):
        ips = self._mod()
        monkeypatch.setattr(ips, "VERBOSE", True)
        monkeypatch.setattr(ips, "_STEP", 0)
        monkeypatch.setattr(ips, "_TOTAL", 5)
        ips._progress("lbl")
        assert capsys.readouterr().out == ""
        assert ips._STEP == 1

    def test_normal_has_cr(self, monkeypatch, capsys):
        ips = self._mod()
        monkeypatch.setattr(ips, "VERBOSE", False)
        monkeypatch.setattr(ips, "_STEP", 0)
        monkeypatch.setattr(ips, "_TOTAL", 5)
        monkeypatch.setattr(ips, "_HAS_COLOR", False)
        ips._progress("lbl")
        out = capsys.readouterr().out
        assert "\r" in out
        assert not out.endswith("\n")

    def test_final_step_newline(self, monkeypatch, capsys):
        ips = self._mod()
        monkeypatch.setattr(ips, "VERBOSE", False)
        monkeypatch.setattr(ips, "_STEP", 4)
        monkeypatch.setattr(ips, "_TOTAL", 5)
        monkeypatch.setattr(ips, "_HAS_COLOR", False)
        ips._progress("done")
        assert capsys.readouterr().out.endswith("\n")

    def test_bar_format(self, monkeypatch, capsys):
        ips = self._mod()
        monkeypatch.setattr(ips, "VERBOSE", False)
        monkeypatch.setattr(ips, "_STEP", 0)
        monkeypatch.setattr(ips, "_TOTAL", 10)
        monkeypatch.setattr(ips, "_HAS_COLOR", False)
        ips._progress("s1")
        out = capsys.readouterr().out
        assert re.search(r"\[=+-+\]", out)
        assert " 1/10" in out


# ═══════════════════════════════════════════════════════════════════════
# 7. setup.sh -- Color initialization  (4 bash tests)
# ═══════════════════════════════════════════════════════════════════════


class TestBashColorInit:

    _SCRIPT = _BASH_COLOR_INIT + '\necho "DIM=${C_DIM}END"'

    def test_no_env_empty(self):
        r = _run_bash(self._SCRIPT)
        assert r.returncode == 0
        assert "DIM=END" in r.stdout

    def test_force_color_sets(self):
        r = _run_bash(self._SCRIPT, {"FORCE_COLOR": "1"})
        assert r.returncode == 0
        # C_DIM is non-empty when FORCE_COLOR is set
        assert "DIM=END" not in r.stdout

    def test_no_color_clears(self):
        r = _run_bash(self._SCRIPT, {"NO_COLOR": "1"})
        assert r.returncode == 0
        assert "DIM=END" in r.stdout

    def test_no_color_beats_force(self):
        r = _run_bash(self._SCRIPT, {"NO_COLOR": "1", "FORCE_COLOR": "1"})
        assert r.returncode == 0
        assert "DIM=END" in r.stdout


# ═══════════════════════════════════════════════════════════════════════
# 8. setup.sh -- step() and substep()  (5 bash tests)
# ═══════════════════════════════════════════════════════════════════════


class TestBashStep:

    _PRE = _BASH_COLOR_INIT + _BASH_HELPERS

    def test_step_padding(self):
        r = _run_bash(self._PRE + '\nstep "deps" "installed"')
        assert r.returncode == 0
        assert "deps" in r.stdout and "installed" in r.stdout

    def test_step_15_label(self):
        r = _run_bash(self._PRE + '\nstep "aaaaaaaaaaaaaaa" "v"')
        assert "aaaaaaaaaaaaaaa" in r.stdout and "v" in r.stdout

    def test_step_long_label(self):
        r = _run_bash(self._PRE + '\nstep "aaaaaaaaaaaaaaaaaaaa" "v"')
        assert "aaaaaaaaaaaaaaaaaaaa" in r.stdout

    def test_step_color(self):
        r = _run_bash(self._PRE + '\nstep "deps" "ok"', {"FORCE_COLOR": "1"})
        assert "\033[" in r.stdout

    def test_substep(self):
        r = _run_bash(self._PRE + '\nsubstep "building..."')
        assert "building..." in r.stdout


# ═══════════════════════════════════════════════════════════════════════
# 9. setup.sh -- run_quiet / run_quiet_no_exit  (5 bash tests)
# ═══════════════════════════════════════════════════════════════════════


class TestBashRunQuiet:

    _PRE = _BASH_COLOR_INIT + _BASH_HELPERS + _BASH_RUN_QUIET

    def test_no_exit_success(self):
        r = _run_bash(self._PRE + '\nrun_quiet_no_exit "ok" true; echo "RC=$?"')
        assert "RC=0" in r.stdout

    def test_no_exit_failure_rc(self):
        r = _run_bash(
            self._PRE
            + "\nrun_quiet_no_exit \"cmd\" bash -c 'echo log_data; exit 42'"
            + '\necho "RC=$?"'
        )
        assert "RC=42" in r.stdout

    def test_no_exit_failure_shows_error(self):
        r = _run_bash(
            self._PRE
            + "\nrun_quiet_no_exit \"my cmd\" bash -c 'echo log_data; exit 1'"
            + "\ntrue"
        )
        assert "my cmd failed" in r.stdout
        assert "log_data" in r.stderr

    def test_run_quiet_failure_exits(self):
        script = self._PRE + """
(
    run_quiet "bad" false
    echo "SHOULD_NOT_REACH"
)
echo "OUTER=$?"
"""
        r = _run_bash(script)
        assert "SHOULD_NOT_REACH" not in r.stdout

    def test_run_quiet_success(self):
        r = _run_bash(self._PRE + '\nrun_quiet "ok" true; echo "RC=$?"')
        assert "RC=0" in r.stdout


# ═══════════════════════════════════════════════════════════════════════
# 10. setup.sh -- try_quiet  (5 bash tests)
# ═══════════════════════════════════════════════════════════════════════


class TestBashTryQuiet:

    _PRE = _BASH_COLOR_INIT + _BASH_HELPERS + _BASH_TRY_QUIET

    def test_success_rc0(self):
        r = _run_bash(self._PRE + '\ntry_quiet "ok" true; echo "RC=$?"')
        assert "RC=0" in r.stdout

    def test_fail_quiet_no_output(self):
        r = _run_bash(
            self._PRE
            + "\ntry_quiet \"cmd\" bash -c 'echo secret; exit 1'"
            + '\necho "RC=$?"'
        )
        assert "RC=1" in r.stdout
        assert "secret" not in r.stdout
        assert "error" not in r.stdout

    def test_fail_verbose_shows(self):
        r = _run_bash(
            self._PRE
            + "\ntry_quiet \"cmd\" bash -c 'echo visible; exit 1'"
            + '\necho "RC=$?"',
            {"UNSLOTH_VERBOSE": "1"},
        )
        assert "cmd failed" in r.stdout
        assert "visible" in r.stderr

    def test_preserves_exit_code(self):
        r = _run_bash(
            self._PRE
            + "\ntry_quiet \"cmd\" bash -c 'exit 42'"
            + '\necho "RC=$?"'
        )
        assert "RC=42" in r.stdout

    def test_success_verbose_quiet(self):
        r = _run_bash(
            self._PRE + '\ntry_quiet "cmd" true; echo "RC=$?"',
            {"UNSLOTH_VERBOSE": "1"},
        )
        assert "RC=0" in r.stdout
        assert "error" not in r.stdout


# ═══════════════════════════════════════════════════════════════════════
# 11. setup.sh -- REQUESTED_PYTHON_VERSION  (5 bash tests)
# ═══════════════════════════════════════════════════════════════════════


class TestBashRequestedPython:

    _PY_MINOR = sys.version_info.minor

    def _block(self, *, requested="", min_minor=None, max_minor=None):
        if min_minor is None:
            min_minor = self._PY_MINOR - 1
        if max_minor is None:
            max_minor = self._PY_MINOR + 1
        return (
            _REQUESTED_PY_TPL.replace("__MIN__", str(min_minor))
            .replace("__MAX__", str(max_minor))
            .replace("__REQ__", requested)
        )

    def test_unset_skipped(self):
        r = _run_bash(self._block(requested=""))
        assert r.stdout.strip().endswith("BEST_PY=")

    def test_valid_in_range(self):
        r = _run_bash(self._block(requested=sys.executable))
        assert f"BEST_PY={sys.executable}" in r.stdout
        assert "using requested Python" in r.stdout

    def test_nonexistent_skipped(self):
        r = _run_bash(self._block(requested="/nonexistent/python3.99"))
        assert r.stdout.strip().endswith("BEST_PY=")

    def test_not_executable_skipped(self):
        r = _run_bash(self._block(requested="/dev/null"))
        assert r.stdout.strip().endswith("BEST_PY=")

    def test_out_of_range(self):
        r = _run_bash(
            self._block(
                requested=sys.executable,
                min_minor=self._PY_MINOR + 10,
                max_minor=self._PY_MINOR + 11,
            )
        )
        assert r.stdout.strip().endswith("BEST_PY=")
        assert "ignoring" in r.stdout


# ═══════════════════════════════════════════════════════════════════════
# 12. setup.ps1 -- Structural checks  (9 tests)
# ═══════════════════════════════════════════════════════════════════════


class TestPS1Structural:
    """Static analysis of setup.ps1 -- no PowerShell runtime needed."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = SETUP_PS1.read_text(encoding="utf-8")

    def test_fn_step(self):
        assert re.search(r"^function step\s*\{", self.content, re.MULTILINE)

    def test_fn_substep(self):
        assert re.search(r"^function substep\s*\{", self.content, re.MULTILINE)

    def test_fn_get_studio_ansi(self):
        assert "function Get-StudioAnsi" in self.content

    def test_fn_enable_vt(self):
        assert "function Enable-StudioVirtualTerminal" in self.content

    def test_ansi_title_code(self):
        assert "38;5;150m" in self.content

    def test_ansi_dim_code(self):
        assert "38;5;245m" in self.content

    def test_ansi_ok_code(self):
        assert "38;5;108m" in self.content

    def test_label_padding(self):
        assert ".PadRight(15)" in self.content

    def test_env_vars_referenced(self):
        assert "NO_COLOR" in self.content
        assert "UNSLOTH_VERBOSE" in self.content


# ═══════════════════════════════════════════════════════════════════════
# 13. studio.py CLI -- setup --verbose  (4 tests)
# ═══════════════════════════════════════════════════════════════════════


class TestCLISetup:

    def test_verbose_sets_env(self):
        from typer.testing import CliRunner

        from unsloth_cli.commands.studio import studio_app

        runner = CliRunner()
        mock_result = MagicMock(returncode=0)
        with (
            patch(
                "unsloth_cli.commands.studio._find_setup_script",
                return_value=Path("/f/setup.sh"),
            ),
            patch(
                "unsloth_cli.commands.studio.subprocess.run",
                return_value=mock_result,
            ) as mock_run,
            patch(
                "unsloth_cli.commands.studio.platform.system",
                return_value="Linux",
            ),
        ):
            result = runner.invoke(studio_app, ["setup", "--verbose"])
        assert result.exit_code == 0
        env_arg = mock_run.call_args.kwargs.get("env")
        assert env_arg is not None
        assert env_arg["UNSLOTH_VERBOSE"] == "1"

    def test_no_verbose_env_none(self):
        from typer.testing import CliRunner

        from unsloth_cli.commands.studio import studio_app

        runner = CliRunner()
        mock_result = MagicMock(returncode=0)
        with (
            patch(
                "unsloth_cli.commands.studio._find_setup_script",
                return_value=Path("/f/setup.sh"),
            ),
            patch(
                "unsloth_cli.commands.studio.subprocess.run",
                return_value=mock_result,
            ) as mock_run,
            patch(
                "unsloth_cli.commands.studio.platform.system",
                return_value="Linux",
            ),
        ):
            result = runner.invoke(studio_app, ["setup"])
        assert result.exit_code == 0
        env_arg = mock_run.call_args.kwargs.get("env")
        assert env_arg is None

    def test_script_not_found(self):
        from typer.testing import CliRunner

        from unsloth_cli.commands.studio import studio_app

        runner = CliRunner()
        with patch(
            "unsloth_cli.commands.studio._find_setup_script",
            return_value=None,
        ):
            result = runner.invoke(studio_app, ["setup"])
        assert result.exit_code == 1
        assert "Could not find" in result.output

    def test_nonzero_rc_propagated(self):
        from typer.testing import CliRunner

        from unsloth_cli.commands.studio import studio_app

        runner = CliRunner()
        mock_result = MagicMock(returncode=42)
        with (
            patch(
                "unsloth_cli.commands.studio._find_setup_script",
                return_value=Path("/f/setup.sh"),
            ),
            patch(
                "unsloth_cli.commands.studio.subprocess.run",
                return_value=mock_result,
            ),
            patch(
                "unsloth_cli.commands.studio.platform.system",
                return_value="Linux",
            ),
        ):
            result = runner.invoke(studio_app, ["setup"])
        assert result.exit_code == 42
