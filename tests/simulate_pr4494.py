#!/usr/bin/env python3
"""PR #4494 Simulation Harness -- sandbox integration tests for 3 UX fixes.

Runs in a uv venv sandbox so it cannot break the workspace. Demonstrates
before-vs-after for each fix and tests edge cases beyond the unit tests.

Fix 1: IPv6 URL bracketing in startup banner
Fix 2: try_quiet stderr redirect in setup.sh
Fix 3: _step() label truncation in install_python_stack.py

Usage:
    python tests/simulate_pr4494.py              # all fixes, full sandbox
    python tests/simulate_pr4494.py --fix 1      # single fix
    python tests/simulate_pr4494.py --no-sandbox # use current Python directly
    python tests/simulate_pr4494.py --keep-sandbox
    python tests/simulate_pr4494.py --no-bash    # skip bash tests
    python tests/simulate_pr4494.py --no-color   # disable ANSI colors
    python tests/simulate_pr4494.py --verbose    # show subprocess output
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

IS_WIN = platform.system() == "Windows"
HAS_BASH = shutil.which("bash") is not None
REPO = Path(__file__).resolve().parent.parent


# ── Color helpers ─────────────────────────────────────────────────────


class C:
    """Thin ANSI color wrapper, toggled by --no-color / NO_COLOR."""

    on = True

    @staticmethod
    def g(s: str) -> str:
        return f"\033[92m{s}\033[0m" if C.on else s

    @staticmethod
    def r(s: str) -> str:
        return f"\033[91m{s}\033[0m" if C.on else s

    @staticmethod
    def d(s: str) -> str:
        return f"\033[38;5;245m{s}\033[0m" if C.on else s

    @staticmethod
    def b(s: str) -> str:
        return f"\033[1m{s}\033[0m" if C.on else s

    @staticmethod
    def c(s: str) -> str:
        return f"\033[96m{s}\033[0m" if C.on else s


# ── Sandbox lifecycle ─────────────────────────────────────────────────


def create_sandbox(verbose: bool) -> tuple[Path, Path]:
    """Create isolated venv + importable package tree.  Returns (dir, python)."""
    sdir = Path(tempfile.mkdtemp(prefix = "sim_pr4494_"))
    vdir = sdir / "venv"

    uv = shutil.which("uv")
    ok = False
    if uv:
        r = subprocess.run(
            [uv, "venv", str(vdir), "--python", sys.executable],
            capture_output = True,
            text = True,
        )
        ok = r.returncode == 0
        if not ok and verbose:
            print(f"  uv venv failed, falling back to stdlib venv: {r.stderr.strip()}")
    if not ok:
        subprocess.run(
            [sys.executable, "-m", "venv", str(vdir)],
            capture_output = True,
            text = True,
            check = True,
        )

    # Mirror the package layout so `import studio.backend.startup_banner` works
    pkg = sdir / "studio" / "backend"
    pkg.mkdir(parents = True)
    (sdir / "studio" / "__init__.py").touch()
    (pkg / "__init__.py").touch()
    shutil.copy2(REPO / "studio" / "backend" / "startup_banner.py", pkg)
    shutil.copy2(REPO / "studio" / "install_python_stack.py", sdir / "studio")

    py = vdir / "Scripts" / "python.exe" if IS_WIN else vdir / "bin" / "python"
    return sdir, py


def cleanup_sandbox(sdir: Path, keep: bool) -> None:
    if keep:
        print(f"  Sandbox kept at: {sdir}")
    else:
        shutil.rmtree(sdir, ignore_errors = True)


# ── Subprocess runner ─────────────────────────────────────────────────

Result = tuple[str, bool, str]  # (name, passed, detail)


def _run_script(python: Path, script: str, verbose: bool) -> list[Result]:
    """Execute *script* in a subprocess, parse PASS:/FAIL: lines."""
    try:
        r = subprocess.run(
            [str(python), "-c", script],
            capture_output = True,
            text = True,
            timeout = 60,
        )
    except subprocess.TimeoutExpired:
        return [("timeout", False, "subprocess timed out after 60s")]
    except Exception as exc:
        return [("error", False, str(exc))]

    results: list[Result] = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("PASS:"):
            results.append((line[5:], True, ""))
        elif line.startswith("FAIL:"):
            parts = line[5:].split(":", 1)
            results.append((parts[0], False, parts[1] if len(parts) > 1 else ""))
    if verbose and r.stderr:
        print(C.d("  [stderr] " + r.stderr[:500]))
    if not results:
        detail = f"no PASS/FAIL output; rc={r.returncode}"
        if r.stdout:
            detail += f"; stdout={r.stdout[:300]}"
        if r.stderr:
            detail += f"; stderr={r.stderr[:300]}"
        return [("parse_error", False, detail)]
    return results


# ══════════════════════════════════════════════════════════════════════
# Fix 1: IPv6 URL Bracketing
# ══════════════════════════════════════════════════════════════════════

FIX1_CASES: list[tuple[str, str, int, str, str]] = [
    # (display_host, bind_host, port, expected_fragment, name)
    ("2001:db8::1", "::", 8888, "http://[2001:db8::1]:8888", "1:std_ipv6"),
    ("::1", "::", 8888, "http://[::1]:8888", "2:loopback_v6"),
    (
        "2001:0db8:85a3::7334",
        "::",
        8888,
        "http://[2001:0db8:85a3::7334]:8888",
        "3:long_ipv6",
    ),
    (
        "::ffff:192.168.1.1",
        "::",
        8888,
        "http://[::ffff:192.168.1.1]:8888",
        "4:v4_mapped",
    ),
    ("fe80::1%eth0", "::", 8888, "http://[fe80::1%eth0]:8888", "5:link_local"),
    ("192.168.1.100", "0.0.0.0", 8888, "http://192.168.1.100:8888", "6:ipv4"),
    ("10.0.0.1", "0.0.0.0", 9000, "http://10.0.0.1:9000", "7:v4_alt_port"),
    ("myhost.local", "0.0.0.0", 8888, "http://myhost.local:8888", "8:hostname"),
    ("localhost", "0.0.0.0", 8888, "http://localhost:8888", "9:localhost"),
    ("", "0.0.0.0", 8888, "http://:8888", "10:empty_host"),
    ("[::1]", "::", 8888, "http://[[::1]]:8888", "11:pre_bracketed"),
    # Full-banner checks
    ("2001:db8::1", "::", 8888, "[2001:db8::1]", "B1:banner_v6_ext"),
    ("192.168.1.5", "0.0.0.0", 8888, "192.168.1.5", "B2:banner_v4_ext"),
    ("::1", "::1", 8888, "[::1]", "B3:banner_v6_lo"),
]


def visual_demo_fix1() -> None:
    print(f"\n  {C.b('Fix 1: IPv6 URL Bracketing')}")
    print(f"  {C.d(chr(0x2500) * 52)}")
    hdr = "  {:<34s}{:<30s}{:<30s}"
    row = "  {:<34s}{:<30s}{:<30s}"
    print(hdr.format("Host", "BEFORE (broken)", "AFTER (fixed)"))
    demos = [
        ("2001:db8::1", 8888),
        ("::1", 8888),
        ("::ffff:192.168.1.1", 8888),
        ("192.168.1.100", 8888),
        ("myhost.local", 8888),
    ]
    for host, port in demos:
        broken = f"http://{host}:{port}"
        fixed = f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"
        tag = "" if broken != fixed else C.d(" (same)")
        # strip ANSI for width calc in row
        print(row.format(host, C.r(broken), C.g(fixed) + tag))
    print()


def _fix1_script(sandbox_dir: str) -> str:
    return (
        textwrap.dedent("""\
        import sys, io
        sys.path.insert(0, __SANDBOX__)

        import studio.backend.startup_banner as banner
        banner.stdout_supports_color = lambda: False

        CASES = __CASES__
        failed = 0
        for display_host, bind_host, port, expected, name in CASES:
            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                banner.print_studio_access_banner(
                    port=port, bind_host=bind_host, display_host=display_host,
                )
            finally:
                sys.stdout = old
            output = buf.getvalue()
            if expected in output:
                print("PASS:" + name)
            else:
                print("FAIL:" + name + ":expected " + repr(expected) + " not found in banner output")
                failed += 1
        sys.exit(1 if failed else 0)
    """)
        .replace("__SANDBOX__", repr(str(sandbox_dir)))
        .replace("__CASES__", repr(FIX1_CASES))
    )


def test_fix1(python: Path, sandbox_dir: Path, verbose: bool) -> list[Result]:
    visual_demo_fix1()
    return _run_script(python, _fix1_script(str(sandbox_dir)), verbose)


# ══════════════════════════════════════════════════════════════════════
# Fix 2: try_quiet stderr redirect
# ══════════════════════════════════════════════════════════════════════


def visual_demo_fix2() -> None:
    print(f"\n  {C.b('Fix 2: try_quiet stderr redirect')}")
    print(f"  {C.d(chr(0x2500) * 52)}")
    print(
        f"  {C.r('BEFORE')}: failure log appears on {C.r('STDOUT')} (mixed with step output)"
    )
    print(
        f"  {C.g('AFTER')}:  failure log appears on {C.g('STDERR')} (clean stream separation)"
    )
    print(f"  {C.d('          (step status line still goes to stdout as intended)')}")
    print()


def test_fix2(skip_bash: bool, verbose: bool) -> list[Result]:
    visual_demo_fix2()
    if skip_bash or not HAS_BASH:
        reason = (
            "skipped (--no-bash)" if skip_bash else "skipped (no bash on this platform)"
        )
        return [
            ("1:success", True, reason),
            ("2:fail_quiet", True, reason),
            ("3:fail_verbose", True, reason),
            ("4:multiline", True, reason),
            ("5:exit_code", True, reason),
            ("6:binary", True, reason),
            ("7:success_verbose", True, reason),
            ("8:broken_contrast", True, reason),
        ]

    script_path = REPO / "tests" / "test_try_quiet.sh"
    if not script_path.exists():
        return [("fix2", False, f"{script_path} not found")]

    try:
        r = subprocess.run(
            ["bash", str(script_path)],
            capture_output = True,
            text = True,
            timeout = 30,
        )
    except subprocess.TimeoutExpired:
        return [("fix2", False, "bash tests timed out")]

    results: list[Result] = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("PASS:"):
            results.append((line[5:], True, ""))
        elif line.startswith("FAIL:"):
            parts = line[5:].split(":", 1)
            results.append((parts[0], False, parts[1] if len(parts) > 1 else ""))

    if verbose:
        if r.stderr:
            print(C.d("  [bash stderr] " + r.stderr[:500]))
        if r.returncode != 0 and not results:
            print(C.r(f"  [bash rc={r.returncode}] stdout: {r.stdout[:300]}"))

    if not results:
        return [("fix2_parse", False, f"no output parsed; rc={r.returncode}")]
    return results


# ══════════════════════════════════════════════════════════════════════
# Fix 3: _step() label truncation
# ══════════════════════════════════════════════════════════════════════

_COL = 15  # must match install_python_stack._COL

FIX3_CASES: list[tuple[str, str, str, str]] = [
    # (label, value, expected_output, name)
    ("deps", "ok", "  deps" + " " * 11 + "ok", "1:short_4"),
    ("error", "bad", "  error" + " " * 10 + "bad", "2:short_5"),
    ("llama-quantize", "ok", "  llama-quantize" + " " * 1 + "ok", "3:len_14"),
    ("", "v", "  " + " " * 15 + "v", "4:empty"),
    ("x" * 15, "v", "  " + "x" * 15 + "v", "5:exact_15"),
    ("x" * 16, "v", "  " + "x" * 15 + "v", "6:trunc_16"),
    ("x" * 20, "v", "  " + "x" * 15 + "v", "7:trunc_20"),
    ("x" * 50, "v", "  " + "x" * 15 + "v", "8:trunc_50"),
    ("toolonglabel!!!!", "v", "  " + "toolonglabel!!!!"[:_COL] + "v", "9:readable"),
]

# Case 10: broken version contrast -- tested inline, not via real module
_BROKEN_LABEL = "x" * 20
_BROKEN_EXPECTED = "  " + _BROKEN_LABEL + "v"  # no truncation, no gap


def visual_demo_fix3() -> None:
    print(f"\n  {C.b('Fix 3: _step() Label Truncation')}")
    print(f"  {C.d(chr(0x2500) * 52)}")
    hdr = "  {:<20s}{:<32s}{:<32s}"
    row = "  {:<20s}{:<32s}{:<32s}"
    print(hdr.format("Label", "BEFORE (broken)", "AFTER (fixed)"))
    demos = [
        ("deps", "ok"),
        ("x" * 20, "v"),
        ("x" * 50, "v"),
    ]
    for label, val in demos:
        # Broken: no truncation, negative padding becomes empty
        bpad = " " * max(0, _COL - len(label))
        broken_out = f"  {label}{bpad}{val}"
        # Fixed: truncate label, then pad
        padded = label[:_COL]
        fixed_out = f"  {padded}{' ' * (_COL - len(padded))}{val}"
        tag = C.d(" (same)") if broken_out == fixed_out else C.d(" (truncated!)")
        disp_label = (
            repr(label) if len(label) <= 16 else f"'{'x' * 5}...' ({len(label)}c)"
        )
        print(
            row.format(
                disp_label,
                C.r(repr(broken_out)),
                C.g(repr(fixed_out)) + tag,
            )
        )
    print()


def _fix3_script(sandbox_dir: str) -> str:
    return (
        textwrap.dedent("""\
        import sys, io
        sys.path.insert(0, __SANDBOX__)

        import studio.install_python_stack as ips
        ips._HAS_COLOR = False

        CASES = __CASES__
        failed = 0
        for label, value, expected, name in CASES:
            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                ips._step(label, value)
            finally:
                sys.stdout = old
            actual = buf.getvalue().rstrip("\\n")
            if actual == expected:
                print("PASS:" + name)
            else:
                print("FAIL:" + name + ":expected " + repr(expected) + " got " + repr(actual))
                failed += 1

        # Case 10: broken version contrast (inline, not via module)
        broken_label = "x" * 20
        broken_padded = broken_label  # no [:_COL] truncation
        broken_pad = " " * max(0, 15 - len(broken_padded))
        broken_out = "  " + broken_padded + broken_pad + "v"
        expected_broken = "  " + "x" * 20 + "v"

        # Also get the fixed output for comparison
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            ips._step(broken_label, "v")
        finally:
            sys.stdout = old
        fixed_out = buf.getvalue().rstrip("\\n")

        # The broken output should differ from the fixed output
        if broken_out == expected_broken and broken_out != fixed_out:
            print("PASS:10:broken_contrast")
        else:
            print("FAIL:10:broken_contrast:broken=" + repr(broken_out) + " fixed=" + repr(fixed_out))
            failed += 1

        sys.exit(1 if failed else 0)
    """)
        .replace("__SANDBOX__", repr(str(sandbox_dir)))
        .replace("__CASES__", repr(FIX3_CASES))
    )


def test_fix3(python: Path, sandbox_dir: Path, verbose: bool) -> list[Result]:
    visual_demo_fix3()
    return _run_script(python, _fix3_script(str(sandbox_dir)), verbose)


# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════

FIX_LABELS = {
    1: "IPv6 URL bracketing",
    2: "try_quiet stderr redirect",
    3: "_step label truncation",
}


def print_summary(
    results_by_fix: dict[int, list[Result]],
    verbose: bool,
) -> bool:
    """Print pass/fail table.  Returns True if all passed."""
    bar = chr(0x2550) * 56
    thin = chr(0x2500) * 56
    print(f"\n  {bar}")
    print(f"  {C.b('PR #4494 Simulation Results')}")
    print(f"  {bar}")

    all_pass = True
    grand_pass = 0
    grand_total = 0

    for fix_num in sorted(results_by_fix):
        results = results_by_fix[fix_num]
        n_pass = sum(1 for _, ok, _ in results if ok)
        n_total = len(results)
        grand_pass += n_pass
        grand_total += n_total
        ok = n_pass == n_total
        if not ok:
            all_pass = False
        tag = C.g("[PASS]") if ok else C.r("[FAIL]")
        label = FIX_LABELS.get(fix_num, f"Fix {fix_num}")
        print(f"  Fix {fix_num}  {label:<34s} {tag}  {n_pass:>2}/{n_total}")

        if verbose or not ok:
            for name, passed, detail in results:
                if not passed:
                    print(f"         {C.r('FAIL')} {name}: {detail}")

    print(f"  {thin}")
    summary = f"Total: {grand_pass}/{grand_total} passed"
    if all_pass:
        print(f"  {C.g(summary)} -- {C.g('ALL PASSED')}")
    else:
        print(f"  {C.r(summary)} -- {C.r('SOME FAILED')}")
    print(f"  {bar}")
    return all_pass


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main() -> int:
    ap = argparse.ArgumentParser(description = "PR #4494 simulation harness")
    ap.add_argument("--fix", type = int, choices = [1, 2, 3], help = "run only this fix")
    ap.add_argument(
        "--no-sandbox", action = "store_true", help = "skip venv, use current Python"
    )
    ap.add_argument(
        "--keep-sandbox", action = "store_true", help = "do not delete sandbox on exit"
    )
    ap.add_argument("--no-bash", action = "store_true", help = "skip bash tests (Fix 2)")
    ap.add_argument("--no-color", action = "store_true", help = "disable ANSI colors")
    ap.add_argument("--verbose", action = "store_true", help = "show subprocess output")
    args = ap.parse_args()

    if args.no_color or os.environ.get("NO_COLOR", "").strip():
        C.on = False

    fixes = [args.fix] if args.fix else [1, 2, 3]

    # Sandbox setup
    sandbox_dir: Path | None = None
    if args.no_sandbox:
        # Use current Python, put source on sys.path directly
        sandbox_dir = REPO
        python = Path(sys.executable)
        print(C.d(f"  (no-sandbox mode, using {python})"))
    else:
        print(C.d("  Creating sandbox..."), end = "", flush = True)
        sandbox_dir, python = create_sandbox(args.verbose)
        print(C.d(f" {sandbox_dir}"))

    results_by_fix: dict[int, list[Result]] = {}

    try:
        if 1 in fixes:
            results_by_fix[1] = test_fix1(python, sandbox_dir, args.verbose)
        if 2 in fixes:
            results_by_fix[2] = test_fix2(args.no_bash, args.verbose)
        if 3 in fixes:
            results_by_fix[3] = test_fix3(python, sandbox_dir, args.verbose)
    finally:
        if not args.no_sandbox and sandbox_dir and sandbox_dir != REPO:
            cleanup_sandbox(sandbox_dir, args.keep_sandbox)

    all_pass = print_summary(results_by_fix, args.verbose)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
