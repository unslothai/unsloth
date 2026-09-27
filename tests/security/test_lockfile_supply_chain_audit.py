"""Regression tests for the offline `scripts/lockfile_supply_chain_audit.py`."""

from __future__ import annotations

import json
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
    # IOC literal built at runtime so CodeQL's py/incomplete-url-substring-sanitization rule doesn't false-positive on
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


def test_malicious_cargo_lockfile_default_mode_blocks(tmp_path):
    """Default mode refuses a non-registry cargo source; provenance blocks without --strict."""
    lockfile = tmp_path / "Cargo.lock"
    lockfile.write_text(_MALICIOUS_CARGO_LOCK)
    proc = _run_auditor(
        root = tmp_path,
        npm_lockfiles = [FIXTURES / "clean_lockfile.json"],
        cargo_lockfiles = [lockfile],
    )
    assert proc.returncode == 1, (
        f"expected exit 1 (blocking), got {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    combined = proc.stdout + proc.stderr
    assert "non-registry-cargo-source" in combined
    assert "blocking finding" in combined


def test_provenance_kinds_block_by_default():
    """The four provenance/integrity kinds must stay in BLOCKING_KINDS."""
    for kind in (
        "non-registry-resolved-url",
        "missing-integrity-hash",
        "non-registry-cargo-source",
        "missing-cargo-checksum",
    ):
        assert kind in lsa.BLOCKING_KINDS, f"{kind} must block in default mode"


def test_non_registry_npm_tarball_blocks_by_default(tmp_path):
    """A file: tarball with a valid-looking integrity must not pass the default audit,
    or `npm ci` runs its lifecycle scripts on the runner."""
    lockfile = tmp_path / "package-lock.json"
    lockfile.write_text(
        json.dumps(
            {
                "name": "victim",
                "version": "1.0.0",
                "lockfileVersion": 3,
                "packages": {
                    "": {"name": "victim", "version": "1.0.0"},
                    "node_modules/evil-pkg": {
                        "version": "1.0.0",
                        "resolved": "file:evil-pkg-1.0.0.tgz",
                        "integrity": "sha512-" + "A" * 86 + "==",
                    },
                },
            }
        )
    )
    proc = _run_auditor(root = tmp_path, npm_lockfiles = [lockfile])
    assert proc.returncode == 1, (
        f"expected exit 1 (blocking), got {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "non-registry-resolved-url" in proc.stdout + proc.stderr


def test_audit_cargo_lockfile_direct_call(tmp_path):
    lockfile = tmp_path / "Cargo.lock"
    lockfile.write_text(_MALICIOUS_CARGO_LOCK)
    findings = lsa.audit_cargo_lockfile(lockfile)
    kinds = {f.kind for f in findings}
    assert "non-registry-cargo-source" in kinds


# ::warning::


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


def test_blocking_finding_emitted_as_single_line_annotation(tmp_path):
    """The ::error:: annotation must be one physical line (%0A-escaped)."""
    lockfile = tmp_path / "Cargo.lock"
    lockfile.write_text(_MALICIOUS_CARGO_LOCK)
    proc = _run_auditor(
        root = tmp_path,
        npm_lockfiles = [FIXTURES / "clean_lockfile.json"],
        cargo_lockfiles = [lockfile],
    )
    error_lines = [line for line in proc.stderr.splitlines() if line.startswith("::error::")]
    assert error_lines, f"expected at least one ::error:: annotation; stderr was:\n{proc.stderr}"
    for line in error_lines:
        # One physical line: kind/package/detail joined via %0A, not split.
        assert "%0A" in line, (
            f"::error:: line has no %0A escape; multi-line text "
            f"would be truncated by GH Actions:\n{line}"
        )
        assert "non-registry-cargo-source" in line
        assert "package:" in line
        assert "detail:" in line


def test_advisory_finding_emitted_as_single_line_annotation(tmp_path):
    """The advisory ::warning:: path stays %0A-escaped too. `missing-resolved-url` is
    incompleteness, not a fetchable source, so it stays advisory."""
    lockfile = tmp_path / "package-lock.json"
    lockfile.write_text(
        json.dumps(
            {
                "name": "victim",
                "version": "1.0.0",
                "lockfileVersion": 3,
                "packages": {
                    "": {"name": "victim", "version": "1.0.0"},
                    "node_modules/incomplete-pkg": {"version": "1.0.0"},
                },
            }
        )
    )
    proc = _run_auditor(root = tmp_path, npm_lockfiles = [lockfile])
    assert proc.returncode == 0, (
        f"expected exit 0 (advisory), got {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    warning_lines = [line for line in proc.stderr.splitlines() if line.startswith("::warning::")]
    assert warning_lines, f"expected a ::warning:: annotation; stderr was:\n{proc.stderr}"
    for line in warning_lines:
        assert "%0A" in line
        assert "missing-resolved-url" in line


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


# ---------------------------------------------------------------------------
# Followup regression tests for #5604:
#   - unsupported lockfile versions must block in default mode (v1 downgrade would otherwise pass with rc=0 because
#     the structural walk only runs on v2/v3)
#   - the ``UNSLOTH_LOCKFILE_AUDIT_SKIP`` warning must be routed through ``_gha_escape()`` so an attacker-controlled
#     value cannot inject a second workflow-command line via embedded ``\n::error::...``
#   - the audit script must be invoked BEFORE ``npm install`` in any workflow that consumes the audited lockfiles
# ---------------------------------------------------------------------------


def test_unsupported_lockfile_version_blocks_default(tmp_path):
    """A v1 lockfile (or any non-v2/v3 version) means the structural
    dependency walk never runs, so ``blocked-known-malicious`` /
    ``known-ioc-string`` findings cannot be produced. Treating that as
    advisory lets an attacker downgrade a checked-in lockfile to v1
    and silently exit CI with rc=0. Default mode must refuse.
    """
    p = tmp_path / "package-lock.json"
    p.write_text(
        "{\n"
        '  "name": "test",\n'
        '  "version": "1.0.0",\n'
        '  "lockfileVersion": 1,\n'
        '  "dependencies": {"react": {"version": "18.2.0"}}\n'
        "}\n"
    )
    proc = _run_auditor(root = tmp_path, npm_lockfiles = [p])
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 1, (
        f"v1 lockfile must block default mode (was advisory pre-followup); "
        f"rc={proc.returncode}\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )
    assert "unsupported-lockfile-version" in combined, combined


def test_blocking_kinds_contains_unsupported_lockfile_version():
    """Direct module-level assertion: if anyone moves
    ``unsupported-lockfile-version`` back out of BLOCKING_KINDS this
    test trips immediately, before they re-introduce the downgrade
    bypass."""
    assert "unsupported-lockfile-version" in lsa.BLOCKING_KINDS


def test_skip_env_warning_escapes_workflow_command_injection(tmp_path):
    """An attacker controlling ``UNSLOTH_LOCKFILE_AUDIT_SKIP`` could
    embed a literal ``\\n::error::...`` and split the warning into a
    second workflow-command annotation. Both branches interpolate the
    value and so both must route it through ``_gha_escape()``: the
    accepted branch echoes the stripped value, the rejected branch
    echoes the PRE-strip raw one.

    Which branch a value takes is decided AFTER stripping, so a payload
    has to be chosen for the branch it is meant to exercise. Anything
    long enough to carry a whole ``::error::`` lands on the accepted
    side; the rejected side is reachable only under the 5-char floor or
    on a booleanish token, which is why branch B below looks so small.
    """
    fixture = FIXTURES / "clean_lockfile.json"

    def _physical_lines_starting_with_double_colon(stderr: str) -> list[str]:
        # GH Actions parses workflow commands per physical line.
        # Only lines that START with `::` after any leading whitespace count as a new annotation.
        # Any such line BEYOND the first warning is an injected command.
        return [ln for ln in stderr.splitlines() if ln.lstrip().startswith("::")]

    injected_bad = "%inject\n::error::bad"  # contains %, \n, and ::
    env_a = {**os.environ, "UNSLOTH_LOCKFILE_AUDIT_SKIP": injected_bad}
    proc_a = subprocess.run(
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
        env = env_a,
    )
    # The stripped value is "%inject\n::error::bad" (len 21) and is not a booleanish token -> accepted-skip path; rc 0,
    # audit skipped.
    assert proc_a.returncode == 0
    assert "%0A" in proc_a.stderr and "%25" in proc_a.stderr, (
        "skip value containing \\n and %% must be %0A / %25 escaped; "
        f"stderr was:\n{proc_a.stderr}"
    )
    cmd_lines_a = _physical_lines_starting_with_double_colon(proc_a.stderr)
    assert len(cmd_lines_a) == 1 and cmd_lines_a[0].startswith("::warning::"), (
        "exactly one ::-prefixed physical line expected (the audit's own "
        f"::warning::); injection split the message into: {cmd_lines_a}"
    )

    # Branch B
    # test_skip_env_var_with_short_value_rejected covers this branch with a plain "1", which carries no control
    # character and so cannot tell escaped from unescaped.
    injected_short = "1\n%"
    env_b = {**os.environ, "UNSLOTH_LOCKFILE_AUDIT_SKIP": injected_short}
    proc_b = subprocess.run(
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
    combined_b = proc_b.stdout + proc_b.stderr
    assert "REQUIRES a justification" in combined_b, combined_b
    # Per-file banner: a rejected value must fall through to the audit, not skip it. Clean fixture, so rc 0 with the
    # audit performed.
    assert "[lockfile-audit] npm:" in combined_b, combined_b
    assert proc_b.returncode == 0, (
        f"rejected skip must still run the audit and pass a clean fixture; "
        f"rc={proc_b.returncode}\n--- stdout ---\n{proc_b.stdout}\n"
        f"--- stderr ---\n{proc_b.stderr}"
    )
    assert "%0A" in proc_b.stderr and "%25" in proc_b.stderr, (
        "the RAW skip value's \\n and %% must be %0A / %25 escaped; "
        f"stderr was:\n{proc_b.stderr}"
    )
    warnings_b = [
        ln
        for ln in _physical_lines_starting_with_double_colon(proc_b.stderr)
        if ln.startswith("::warning::")
    ]
    assert len(warnings_b) == 1 and "Proceeding with audit" in warnings_b[0], (
        "the rejected-skip warning must stay on ONE physical line; an "
        f"unescaped newline split it: {proc_b.stderr!r}"
    )


def test_audit_runs_before_npm_install_in_consumer_workflows():
    """Any GH Actions workflow that consumes one of the audited
    lockfiles via ``npm install`` / ``npm ci`` must run the
    lockfile_supply_chain_audit step BEFORE that install, otherwise a
    compromised lockfile's lifecycle scripts execute before the audit
    can refuse the run.

    Parsed as YAML and checked per JOB, not per file. Reading the raw
    text instead sees neither half of the guarantee: an install written
    as a ``run: |`` block scalar is invisible, and a file-wide "first
    audit offset" lets one job's audit vouch for another job's install.
    Together those left this test green with the whole Windows audit
    step of studio-tauri-smoke.yml deleted.
    """
    import re

    import yaml

    # The Linux jobs call `python3`; the Windows and macOS jobs pin an interpreter with setup-python and call `python`.
    # Same audit.
    audit_re = re.compile(r"\bpython3?\s+scripts/lockfile_supply_chain_audit\.py\b")
    install_re = re.compile(r"\bnpm\s+(?:install|ci)\b")

    workflows_dir = REPO_ROOT / ".github" / "workflows"
    for wf_name in ("studio-tauri-smoke.yml", "release-desktop.yml"):
        wf = workflows_dir / wf_name
        assert wf.is_file(), f"missing workflow: {wf}"
        doc = yaml.safe_load(wf.read_text(encoding = "utf-8"))
        checked = 0
        for job_id, job in (doc.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            # (step index, offset within that step) of the job's first audit, so an audit and an install sharing one
            # step are still ordered against each other.
            audited_at = None
            for index, step in enumerate(job.get("steps") or []):
                run = step.get("run") if isinstance(step, dict) else None
                if not isinstance(run, str):
                    continue
                audit = audit_re.search(run)
                if audit is not None and audited_at is None:
                    audited_at = (index, audit.start())
                install = install_re.search(run)
                if install is None:
                    continue
                checked += 1
                assert audited_at is not None and audited_at < (index, install.start()), (
                    f"{wf_name}: job {job_id!r} reaches ``npm install`` / "
                    f"``npm ci`` in step {index} with no lockfile audit before "
                    f"it in that job; a compromised lockfile's lifecycle "
                    f"scripts would execute before the audit can refuse it"
                )
        assert checked, (
            f"{wf_name}: found no ``npm install`` / ``npm ci`` step at all, so "
            f"this guard passed vacuously -- the workflow or the pattern drifted"
        )
