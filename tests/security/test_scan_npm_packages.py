"""Regression tests for `scripts/scan_npm_packages.py`.

These tests must run fully offline. The `network_blocker` fixture in
conftest.py refuses any non-loopback socket connect from the test
process; scanner subprocesses are invoked against fixtures that never
trigger an HTTP fetch.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "scan_npm_packages.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Import the module so we can introspect the IOC tables directly.
sys.path.insert(0, str(REPO_ROOT))
from scripts import scan_npm_packages as snp  # noqa: E402


# ---------------------------------------------------------------------------
# Subprocess helpers.
# ---------------------------------------------------------------------------


def _run_scanner(lockfile: Path, *, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--lockfile", str(lockfile)],
        capture_output = True,
        text = True,
        timeout = timeout,
    )


# ---------------------------------------------------------------------------
# Lockfile pass: structural-only fixtures (no network).
# ---------------------------------------------------------------------------


def test_malicious_lockfile_exits_1():
    """Structural IOCs alone must fail the scanner.

    `structural_only_lockfile.json` contains: (a) a non-registry
    `resolved` URL (filev2.getsession.org), (b) an entry missing
    its `integrity` field. Both are caught in `parse_lockfile()`
    before any tarball download attempt -- so the test is fully
    offline.
    """
    fixture = FIXTURES / "structural_only_lockfile.json"
    assert fixture.is_file(), fixture
    proc = _run_scanner(fixture)
    assert proc.returncode == 1, (
        f"expected exit 1, got {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    combined = proc.stdout + proc.stderr
    # The scanner aggregates structural findings into the summary
    # rather than printing each one individually. Assert on the
    # count + the FAIL banner instead.
    assert "2 structural finding(s)" in combined
    assert "FAIL" in combined
    # And confirm `parse_lockfile()` actually surfaces the right
    # `pattern` codes via the in-process API.
    entries, struct = snp.parse_lockfile(fixture)
    patterns = {f.pattern for f in struct}
    assert {"non-registry-resolved-url", "missing-integrity-hash"} <= patterns


def test_clean_lockfile_exits_0():
    """The clean fixture only contains entries that `parse_lockfile()`
    skips entirely (workspace root + workspace `link` symlink +
    nested fold-in), so the scanner exits 0 with no network access.
    """
    fixture = FIXTURES / "clean_lockfile.json"
    assert fixture.is_file(), fixture
    proc = _run_scanner(fixture)
    assert proc.returncode == 0, (
        f"expected exit 0, got {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "0 finding(s)" in proc.stdout
    assert "0 hard error(s)" in proc.stdout


# ---------------------------------------------------------------------------
# BLOCKED_NPM_VERSIONS table -- gated on Fork 1.
# ---------------------------------------------------------------------------


_BLOCKED_AVAILABLE = hasattr(snp, "BLOCKED_NPM_VERSIONS")


@pytest.mark.skipif(
    not _BLOCKED_AVAILABLE,
    reason = "Fork 1 (BLOCKED_NPM_VERSIONS constant) not merged yet",
)
def test_blocked_npm_versions_complete():
    table = snp.BLOCKED_NPM_VERSIONS
    tanstack_keys = [k for k in table if k.startswith("@tanstack/")]
    assert len(tanstack_keys) == 42, (
        f"expected 42 @tanstack/* entries, got {len(tanstack_keys)}: "
        f"{sorted(tanstack_keys)}"
    )
    assert "@opensearch-project/opensearch" in table
    assert table["@opensearch-project/opensearch"] == {
        "3.5.3",
        "3.6.2",
        "3.7.0",
        "3.8.0",
    }
    squawk = [k for k in table if k.startswith("@squawk/")]
    assert len(squawk) >= 22, (
        f"expected at least 22 @squawk/* entries (full safedep.io enumeration), "
        f"got {len(squawk)}: {sorted(squawk)}"
    )
    # @squawk/mcp must cover the full malicious range 0.9.1 .. 0.9.5
    # (safedep.io enumeration; we initially had only 0.9.5).
    assert {"0.9.1", "0.9.2", "0.9.3", "0.9.4", "0.9.5"} <= table["@squawk/mcp"]

    uipath = [k for k in table if k.startswith("@uipath/")]
    assert len(uipath) >= 64, (
        f"expected at least 64 @uipath/* entries (Aikido enumeration), "
        f"got {len(uipath)}: {sorted(uipath)}"
    )
    # Anchor a known entry: the rpa-tool 0.9.5 version is in the published list.
    assert "0.9.5" in table["@uipath/rpa-tool"]

    # Aikido (May-12 wave): @mistralai/* npm scope (separate from PyPI mistralai).
    assert table["@mistralai/mistralai"] == {"2.2.2", "2.2.3", "2.2.4"}
    assert table["@mistralai/mistralai-gcp"] == {"1.7.1", "1.7.2", "1.7.3"}
    assert table["@mistralai/mistralai-azure"] == {"1.7.1", "1.7.2", "1.7.3"}

    # Aikido: @tallyui/* (10 packages x 3 versions).
    tallyui = [k for k in table if k.startswith("@tallyui/")]
    assert len(tallyui) == 10, f"expected 10 @tallyui/*, got {sorted(tallyui)}"

    # Aikido: @beproduct/nestjs-auth covers the 0.1.2 .. 0.1.19 range (18 versions).
    assert table["@beproduct/nestjs-auth"] == {f"0.1.{i}" for i in range(2, 20)}

    # Aikido: unscoped infostealer packages (10 total).
    for unscoped in (
        "safe-action",
        "ts-dna",
        "cross-stitch",
        "cmux-agent-mcp",
        "agentwork-cli",
        "git-branch-selector",
        "wot-api",
        "git-git-git",
        "nextmove-mcp",
        "ml-toolkit-ts",
    ):
        assert unscoped in table, f"missing unscoped malicious pkg: {unscoped}"

    # Aikido: payload SHA-256 hashes wired into KNOWN_IOC_STRINGS.
    ioc = snp.KNOWN_IOC_STRINGS
    assert "ab4fcadaec49c03278063dd269ea5eef82d24f2124a8e15d7b90f2fa8601266c" in ioc
    assert "2ec78d556d696e208927cc503d48e4b5eb56b31abc2870c2ed2e98d6be27fc96" in ioc
    assert "bun run tanstack_runner.js" in ioc


@pytest.mark.skipif(
    not _BLOCKED_AVAILABLE,
    reason = "Fork 1 (BLOCKED_NPM_VERSIONS pre-fetch hook) not merged yet",
)
def test_blocked_npm_versions_short_circuits_download():
    """With Fork 1's pre-fetch hook, the malicious tanstack entry
    must produce a `blocked-known-malicious` finding without ever
    calling out to the npm registry. The full malicious fixture
    contains the tanstack entry; the test asserts exit 1 and that
    the new finding pattern appears in scanner output.
    """
    fixture = FIXTURES / "malicious_lockfile.json"
    proc = _run_scanner(fixture, timeout = 10)
    assert proc.returncode == 1
    combined = proc.stdout + proc.stderr
    assert "blocked-known-malicious" in combined or "BLOCKED_NPM_VERSIONS" in combined


# ---------------------------------------------------------------------------
# KNOWN_IOC_STRINGS coverage -- every IOC must trip the scanner.
# ---------------------------------------------------------------------------


def _extract_pkg_with_ioc(ioc: str, tmp_path: Path) -> Path:
    """Build a one-file npm package extract tree embedding `ioc` in
    `package.json`. Returns the extract root.
    """
    pkg_json = {
        "name": "ioc-fixture",
        "version": "0.0.1",
        "description": f"contains literal: {ioc}",
    }
    root = tmp_path / f"pkg_{abs(hash(ioc)) % 10**8}"
    (root / "package").mkdir(parents = True)
    (root / "package" / "package.json").write_text(
        json.dumps(pkg_json),
        encoding = "utf-8",
    )
    return root


def test_every_known_ioc_string_caught(tmp_path):
    """For every entry in `KNOWN_IOC_STRINGS`, embed the IOC in a
    one-file package tree and confirm `scan_extracted_tree()`
    surfaces it. Guards against silent regex / table drift.
    """
    iocs = snp.KNOWN_IOC_STRINGS
    assert iocs, "KNOWN_IOC_STRINGS unexpectedly empty"

    pkg = snp.PackageEntry(
        name = "ioc-fixture",
        version = "0.0.1",
        resolved = "https://registry.npmjs.org/ioc-fixture/-/ioc-fixture-0.0.1.tgz",
        integrity = "sha512-stub",
        lockfile_key = "node_modules/ioc-fixture",
    )

    for ioc in iocs:
        root = _extract_pkg_with_ioc(ioc, tmp_path)
        findings = snp.scan_extracted_tree(pkg = pkg, root = root)
        hit = any(ioc in f.evidence or ioc in f.detail for f in findings)
        assert hit, (
            f"KNOWN_IOC_STRINGS[{ioc!r}] not detected by scan_extracted_tree; "
            f"findings = {[str(f) for f in findings]}"
        )


# ---------------------------------------------------------------------------
# Sanity: lockfile parse pass surfaces the structural findings we expect.
# ---------------------------------------------------------------------------


def test_parse_lockfile_structural_findings():
    """`parse_lockfile()` returns (entries, structural_findings). The
    structural-only fixture should produce 2 structural findings and
    0 entries (because both bad entries are `continue`d).
    """
    entries, struct = snp.parse_lockfile(FIXTURES / "structural_only_lockfile.json")
    assert entries == []
    patterns = {f.pattern for f in struct}
    assert "non-registry-resolved-url" in patterns
    assert "missing-integrity-hash" in patterns
