"""Regression tests for the offline `scripts/lockfile_supply_chain_audit.py`."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "lockfile_supply_chain_audit.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

sys.path.insert(0, str(REPO_ROOT))
from scripts import lockfile_supply_chain_audit as lsa  # noqa: E402


def _run_auditor(
    *,
    root: Path,
    npm_lockfiles: list[Path] | None = None,
    cargo_lockfiles: list[Path] | None = None,
    strict: bool = False,
    timeout: int = 30,
) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SCRIPT), "--root", str(root)]
    if strict:
        cmd.append("--strict")
    for p in npm_lockfiles or []:
        cmd.extend(["--npm-lockfile", str(p)])
    for p in cargo_lockfiles or []:
        cmd.extend(["--cargo-lockfile", str(p)])
    return subprocess.run(
        cmd,
        capture_output = True,
        text = True,
        timeout = timeout,
    )


# ---------------------------------------------------------------------------
# npm lockfile audit.
# ---------------------------------------------------------------------------


def test_malicious_lockfile_exits_1(tmp_path):
    """Non-registry URL + IOC substring + missing integrity hash -> auditor exits 1."""
    fixture = FIXTURES / "malicious_lockfile.json"
    assert fixture.is_file()
    proc = _run_auditor(root = tmp_path, npm_lockfiles = [fixture])
    assert proc.returncode == 1, (
        f"expected exit 1, got {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    combined = proc.stdout + proc.stderr
    assert "non-registry-resolved-url" in combined
    assert "missing-integrity-hash" in combined
    assert "known-ioc-string" in combined
    # IOC literal built at runtime so CodeQL's
    # py/incomplete-url-substring-sanitization rule doesn't false-positive on
    # the source-literal + `in` (the operand is the scanner's own output).
    _ioc_host = "filev2." + "getsession.org"
    assert _ioc_host in combined


def test_clean_lockfile_exits_0(tmp_path):
    fixture = FIXTURES / "clean_lockfile.json"
    proc = _run_auditor(root = tmp_path, npm_lockfiles = [fixture])
    assert proc.returncode == 0, (
        f"expected exit 0, got {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "0 findings" in proc.stdout


def test_audit_npm_lockfile_direct_call_findings():
    """In-process audit_npm_lockfile() returns the same findings as the subprocess."""
    findings = lsa.audit_npm_lockfile(FIXTURES / "malicious_lockfile.json")
    kinds = {f.kind for f in findings}
    assert "non-registry-resolved-url" in kinds
    assert "missing-integrity-hash" in kinds
    assert "known-ioc-string" in kinds


# ---------------------------------------------------------------------------
# IOC string table -- gated on Fork 1's NPM_IOC_STRINGS additions.
# ---------------------------------------------------------------------------


_MAY12_IOCS = (
    "git-tanstack.com",
    "transformers.pyz",
    "/tmp/transformers.pyz",
    "With Love TeamPCP",
)


def test_npm_ioc_strings_contains_may11_baseline():
    """May-11 wave IOCs must remain in NPM_IOC_STRINGS (baseline)."""
    iocs = set(lsa.NPM_IOC_STRINGS)
    for needle in (
        "router_init.js",
        "tanstack_runner.js",
        "router_runtime.js",
        "filev2.getsession.org",
    ):
        assert needle in iocs, f"baseline IOC {needle!r} disappeared"


@pytest.mark.skipif(
    not all(s in lsa.NPM_IOC_STRINGS for s in _MAY12_IOCS),
    reason = "Fork 1 (May-12 IOC additions) not merged yet",
)
def test_npm_ioc_strings_contains_may12_additions():
    iocs = set(lsa.NPM_IOC_STRINGS)
    for needle in _MAY12_IOCS:
        assert needle in iocs


@pytest.mark.skipif(
    not hasattr(lsa, "BLOCKED_NPM_VERSIONS"),
    reason = "Fork 1 (BLOCKED_NPM_VERSIONS in auditor) not merged yet",
)
def test_lockfile_auditor_blocked_versions_match_scanner():
    """Auditor's BLOCKED_NPM_VERSIONS must mirror the scanner's table verbatim."""
    from scripts import scan_npm_packages as snp
    assert (
        lsa.BLOCKED_NPM_VERSIONS == snp.BLOCKED_NPM_VERSIONS
    ), "auditor and scanner BLOCKED_NPM_VERSIONS tables drifted"


# ---------------------------------------------------------------------------
# Cargo.lock audit.
# ---------------------------------------------------------------------------


_MALICIOUS_CARGO_LOCK = """\
version = 3

[[package]]
name = "fix-path-env"
version = "0.0.1"
source = "git+https://example.com/foo#deadbeef"

[[package]]
name = "honest-crate"
version = "1.0.0"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "0000000000000000000000000000000000000000000000000000000000000000"
"""


def test_malicious_cargo_lockfile_refused(tmp_path):
    """git+https:// Cargo source trips non-registry-cargo-source; --strict makes it blocking."""
    lockfile = tmp_path / "Cargo.lock"
    lockfile.write_text(_MALICIOUS_CARGO_LOCK)
    proc = _run_auditor(
        root = tmp_path,
        npm_lockfiles = [FIXTURES / "clean_lockfile.json"],
        cargo_lockfiles = [lockfile],
        strict = True,
    )
    assert proc.returncode == 1
    combined = proc.stdout + proc.stderr
    assert "non-registry-cargo-source" in combined
    assert "git+https://example.com" in combined


def test_malicious_cargo_lockfile_default_mode_advisory(tmp_path):
    """Default mode emits non-registry-cargo-source as advisory ::warning:: but exits 0."""
    lockfile = tmp_path / "Cargo.lock"
    lockfile.write_text(_MALICIOUS_CARGO_LOCK)
    proc = _run_auditor(
        root = tmp_path,
        npm_lockfiles = [FIXTURES / "clean_lockfile.json"],
        cargo_lockfiles = [lockfile],
    )
    assert proc.returncode == 0, (
        f"expected exit 0 (advisory), got {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    combined = proc.stdout + proc.stderr
    assert "non-registry-cargo-source" in combined
    assert "advisory finding" in combined


def test_audit_cargo_lockfile_direct_call(tmp_path):
    lockfile = tmp_path / "Cargo.lock"
    lockfile.write_text(_MALICIOUS_CARGO_LOCK)
    findings = lsa.audit_cargo_lockfile(lockfile)
    kinds = {f.kind for f in findings}
    assert "non-registry-cargo-source" in kinds


# ---------------------------------------------------------------------------
# GitHub Actions annotation escape: ::warning:: / ::error:: messages
# are truncated at the first newline unless escaped, so the multi-line
# Finding must be collapsed via the spec'd %0A / %0D / %25 encoding.
# ---------------------------------------------------------------------------


def test_gha_escape_collapses_finding_to_one_line():
    """_gha_escape() encodes \\n/\\r/% so GHA annotations aren't truncated; % must escape first."""
    assert lsa._gha_escape("a\nb\nc") == "a%0Ab%0Ac"
    assert lsa._gha_escape("a\rb") == "a%0Db"
    assert lsa._gha_escape("100%") == "100%25"
    # Order regression: `%` must escape before `\n`, else escapes double-encode.
    assert lsa._gha_escape("a%b\nc") == "a%25b%0Ac"

    f = lsa.Finding(
        path = "/x/lock.json",
        package = "node_modules/foo",
        kind = "missing-integrity-hash",
        detail = "bad stuff",
    )
    escaped = lsa._gha_escape(str(f))
    assert "\n" not in escaped
    assert "%0A" in escaped
    assert "missing-integrity-hash" in escaped
    assert "node_modules/foo" in escaped
    assert "bad stuff" in escaped


def test_advisory_finding_emitted_as_single_line_annotation(tmp_path):
    """Advisory ::warning:: must be one physical line (%0A-escaped). Regression for PR #5604."""
    lockfile = tmp_path / "Cargo.lock"
    lockfile.write_text(_MALICIOUS_CARGO_LOCK)
    proc = _run_auditor(
        root = tmp_path,
        npm_lockfiles = [FIXTURES / "clean_lockfile.json"],
        cargo_lockfiles = [lockfile],
    )
    warning_lines = [
        line for line in proc.stderr.splitlines() if line.startswith("::warning::")
    ]
    assert warning_lines, (
        "expected at least one ::warning:: annotation; " f"stderr was:\n{proc.stderr}"
    )
    for line in warning_lines:
        # One physical line: kind/package/detail joined via %0A, not split.
        assert "%0A" in line, (
            f"::warning:: line has no %0A escape; multi-line text "
            f"would be truncated by GH Actions:\n{line}"
        )
        assert "non-registry-cargo-source" in line
        assert "package:" in line
        assert "detail:" in line


# ---------------------------------------------------------------------------
# SF4: skip env var requires a justification value.
# ---------------------------------------------------------------------------


def test_skip_env_var_with_short_value_rejected(tmp_path):
    """SF4: a short/boolean UNSLOTH_LOCKFILE_AUDIT_SKIP is rejected; a real justification is honored."""
    fixture = FIXTURES / "clean_lockfile.json"

    # Case 1 -- "1" rejected, audit RUNS.
    env_bad = {**os.environ, "UNSLOTH_LOCKFILE_AUDIT_SKIP": "1"}
    proc_bad = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--npm-lockfile",
            str(fixture),
        ],
        capture_output = True,
        text = True,
        timeout = 30,
        env = env_bad,
    )
    combined_bad = proc_bad.stdout + proc_bad.stderr
    assert "::warning::" in combined_bad, combined_bad
    assert "REQUIRES a justification" in combined_bad, combined_bad
    # Per-file banner proves the audit ran.
    assert "[lockfile-audit] npm:" in combined_bad, combined_bad
    # Clean fixture -> exit 0, but the audit was performed.
    assert proc_bad.returncode == 0, (
        f"expected rc 0 on clean fixture, got {proc_bad.returncode}\n"
        f"--- stdout ---\n{proc_bad.stdout}\n"
        f"--- stderr ---\n{proc_bad.stderr}"
    )

    # Case 2 -- a real-looking justification accepted, audit skipped.
    env_ok = {**os.environ, "UNSLOTH_LOCKFILE_AUDIT_SKIP": "ticket-5397"}
    proc_ok = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--npm-lockfile",
            str(fixture),
        ],
        capture_output = True,
        text = True,
        timeout = 30,
        env = env_ok,
    )
    combined_ok = proc_ok.stdout + proc_ok.stderr
    assert proc_ok.returncode == 0
    assert "::warning::" in combined_ok
    assert "skipped" in combined_ok.lower()
    assert "ticket-5397" in combined_ok
    # Skip path: no "npm:" banner means the audit body never ran.
    assert "[lockfile-audit] npm:" not in combined_ok, combined_ok

    # Case 3 -- the booleanish tokens are ALL rejected.
    for bad_val in ("true", "yes", "on", "0", ""):
        env_b = {**os.environ, "UNSLOTH_LOCKFILE_AUDIT_SKIP": bad_val}
        p = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--root",
                str(tmp_path),
                "--npm-lockfile",
                str(fixture),
            ],
            capture_output = True,
            text = True,
            timeout = 30,
            env = env_b,
        )
        c = p.stdout + p.stderr
        assert (
            "::warning::" in c and "REQUIRES" in c
        ), f"value {bad_val!r} should have been rejected; got:\n{c}"
        assert "[lockfile-audit] npm:" in c, (
            f"value {bad_val!r} should have fallen through to run audit; " f"got:\n{c}"
        )
