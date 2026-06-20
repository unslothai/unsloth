"""Regression tests for scripts/scan_npm_packages.py. Run fully offline (network_blocker fixture)."""

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

# Import the module to introspect IOC tables directly.
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
    """Structural IOCs alone (non-registry resolved URL + missing integrity) fail the scanner offline."""
    fixture = FIXTURES / "structural_only_lockfile.json"
    assert fixture.is_file(), fixture
    proc = _run_scanner(fixture)
    assert proc.returncode == 1, (
        f"expected exit 1, got {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    combined = proc.stdout + proc.stderr
    # Scanner aggregates structural findings into the summary; assert on count + FAIL banner.
    assert "2 structural finding(s)" in combined
    assert "FAIL" in combined
    # Confirm parse_lockfile() surfaces the right pattern codes via the in-process API.
    entries, struct = snp.parse_lockfile(fixture)
    patterns = {f.pattern for f in struct}
    assert {"non-registry-resolved-url", "missing-integrity-hash"} <= patterns


def test_clean_lockfile_exits_0():
    """Clean fixture has only entries parse_lockfile() skips, so the scanner exits 0 offline."""
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
        f"expected 42 @tanstack/* entries, got {len(tanstack_keys)}: " f"{sorted(tanstack_keys)}"
    )
    assert "@opensearch-project/opensearch" in table
    assert table["@opensearch-project/opensearch"] == {"3.5.3", "3.6.2", "3.7.0", "3.8.0"}
    squawk = [k for k in table if k.startswith("@squawk/")]
    assert len(squawk) >= 22, (
        f"expected at least 22 @squawk/* entries (full safedep.io enumeration), "
        f"got {len(squawk)}: {sorted(squawk)}"
    )
    # @squawk/mcp must cover the full malicious range 0.9.1..0.9.5 (safedep.io enumeration).
    assert {"0.9.1", "0.9.2", "0.9.3", "0.9.4", "0.9.5"} <= table["@squawk/mcp"]

    uipath = [k for k in table if k.startswith("@uipath/")]
    assert len(uipath) >= 64, (
        f"expected at least 64 @uipath/* entries (Aikido enumeration), "
        f"got {len(uipath)}: {sorted(uipath)}"
    )
    # Anchor a known published entry.
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
    """Pre-fetch hook flags the malicious tanstack entry (exit 1) without hitting the npm registry."""
    fixture = FIXTURES / "malicious_lockfile.json"
    proc = _run_scanner(fixture, timeout = 10)
    assert proc.returncode == 1
    combined = proc.stdout + proc.stderr
    assert "blocked-known-malicious" in combined or "BLOCKED_NPM_VERSIONS" in combined


# ---------------------------------------------------------------------------
# KNOWN_IOC_STRINGS coverage -- every IOC must trip the scanner.
# ---------------------------------------------------------------------------


def _extract_pkg_with_ioc(ioc: str, tmp_path: Path) -> Path:
    """Build a one-file npm package extract tree embedding `ioc` in package.json; return its root."""
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
    """Each KNOWN_IOC_STRINGS entry must be surfaced by scan_extracted_tree(); guards table drift."""
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
    """Structural-only fixture yields 2 structural findings and 0 entries."""
    entries, struct = snp.parse_lockfile(FIXTURES / "structural_only_lockfile.json")
    assert entries == []
    patterns = {f.pattern for f in struct}
    assert "non-registry-resolved-url" in patterns
    assert "missing-integrity-hash" in patterns


# ---------------------------------------------------------------------------
# Code-only scanning (_strip_js_noncode): blank comments WITHOUT touching
# strings/regex/code, preserve geometry, fail open on lexer confusion.
# ---------------------------------------------------------------------------


def _strip(src):
    out = snp._strip_js_noncode(src)
    assert len(out) == len(src), "geometry (length) must be preserved"
    assert out.count("\n") == src.count("\n"), "newline count must be preserved"
    return out


def test_strip_blanks_line_and_block_comments():
    out = _strip("var x = 1; // eval(atob('p'))\n/* subprocess */ run();")
    assert "var x = 1;" in out and "run();" in out
    assert "eval(atob" not in out
    assert "subprocess" not in out


def test_strip_keeps_url_in_string_and_template():
    src = 'const a = "http://example.com/x";\nconst b = `http://${h}//y`; go();'
    out = _strip(src)
    assert out == src  # nothing is a comment -> byte-identical
    assert "http://example.com/x" in out and "//y" in out


def test_strip_regex_with_escaped_slashes_keeps_trailing_code():
    # A naive "// = comment" stripper would eat `evil()`; the lexer must not.
    src = r"const re = /https?:\/\//g; evil();"
    out = _strip(src)
    assert out == src
    assert "evil();" in out


def test_strip_preserves_assigned_base64_payload():
    # npm droppers hide payloads in assigned string literals -- never blank them.
    src = 'var B = "QWxhZGRpbjpvcGVuc2VzYW1l"; new Function(atob(B))();'
    out = _strip(src)
    assert out == src
    assert "QWxhZGRpbjpvcGVuc2VzYW1l" in out


def test_strip_fails_open_on_unterminated_block_comment():
    src = "code(); /* never closed"
    assert snp._strip_js_noncode(src) == src  # fail open: unchanged, still fully scanned


def test_strip_only_applies_to_js_family():
    # A `//`-containing JSON/YAML string must be left intact (JS lexer must not apply).
    PKG = snp.PackageEntry(
        name = "x",
        version = "1.0.0",
        resolved = "https://registry.npmjs.org/x/-/x-1.0.0.tgz",
        integrity = "sha512-z",
        lockfile_key = "node_modules/x",
    )
    # scan_text_blob strips for .js but not for .json.
    yaml_like = 'url: "http://h"  # a yaml comment, not JS\n'
    # Verify the stripper is gated on suffix (JS lexer not applied to non-JS suffixes).
    assert "".endswith(snp._JS_FAMILY_SUFFIXES) is False
    assert ".js" in snp._JS_FAMILY_SUFFIXES and ".json" not in snp._JS_FAMILY_SUFFIXES


# ---------------------------------------------------------------------------
# Detection survives stripping; comment-only IOC is suppressed.
# ---------------------------------------------------------------------------


_PKG = snp.PackageEntry(
    name = "x",
    version = "1.0.0",
    resolved = "https://registry.npmjs.org/x/-/x-1.0.0.tgz",
    integrity = "sha512-z",
    lockfile_key = "node_modules/x",
)
_BLOB = "QWxhZGRpbg" * 240  # ~2.4 KiB base64-ish


def test_real_payload_still_flags_after_stripping():
    # Obfuscated blob behind Function(), wrapped in comments that get blanked.
    src = f'/* header */ var f = new Function("{_BLOB}"); f(); // tail\n'
    pats = {f.pattern for f in snp.scan_text_blob(_PKG, "m.js", src)}
    assert "obfuscated-blob" in pats
    # eval-with-string + atob shape, comment between the two halves.
    src2 = "(0,eval)(/* x */ atob('ZG8='));"
    pats2 = {f.pattern for f in snp.scan_text_blob(_PKG, "m.js", src2)}
    assert "js-fetch-eval" in pats2


def test_payload_entirely_in_comment_is_suppressed():
    src = f'/* var f = new Function("{_BLOB}"); */ var ok = 1;'
    js = snp.scan_text_blob(_PKG, "m.js", src)
    assert js == []  # blanked -> clean
    # Control: same bytes as non-JS (unstripped) WOULD flag.
    txt = snp.scan_text_blob(_PKG, "m.txt", src)
    assert any(f.pattern == "obfuscated-blob" for f in txt)


def test_ioc_in_assigned_string_survives_stripping():
    # A real C2 host lives in a string literal, not a comment -> still caught.
    src = 'var c = "filev2.getsession.org"; // doc note\n'
    pats = {f.pattern for f in snp.scan_text_blob(_PKG, "m.js", src)}
    assert "known-ioc-string" in pats


# ---------------------------------------------------------------------------
# Baseline allowlist -- suppress reviewed findings, fail on new kinds.
# ---------------------------------------------------------------------------


def _finding(
    pkg,
    fn,
    pattern,
    sev = snp.HIGH,
):
    return snp.Finding(severity = sev, package = pkg, filename = fn, pattern = pattern)


def test_norm_pkg_name_strips_version_keeps_scope():
    assert snp._norm_pkg_name("@scope/pkg@1.2.3") == "@scope/pkg"
    assert snp._norm_pkg_name("pkg@1.2.3") == "pkg"
    assert snp._norm_pkg_name("@scope/pkg") == "@scope/pkg"
    assert snp._norm_pkg_name("<root>") == "<root>"


def test_baseline_key_is_version_stable():
    # Same in-package path across a version bump -> identical key. npm tarballs
    # root every file at ``package/``, so the path is stable; only the version in
    # the display name changes.
    a = _finding("left-pad@1.0.0", "package/index.js", "obfuscated-blob")
    b = _finding("left-pad@9.9.9", "package/index.js", "obfuscated-blob")
    assert snp._finding_key(a) == snp._finding_key(b)


def test_baseline_key_distinguishes_same_basename_diff_dir():
    # Package-relative keying: the same basename in a different directory is a
    # DIFFERENT key, so a new dist/ vs src/ file is not silently suppressed.
    a = _finding("pkg@1.0.0", "package/dist/index.js", "obfuscated-blob")
    b = _finding("pkg@1.0.0", "package/src/index.js", "obfuscated-blob")
    assert snp._finding_key(a) != snp._finding_key(b)


def test_baseline_suppresses_listed_but_not_new_pattern(tmp_path):
    bl = tmp_path / "bl.json"
    bl.write_text(
        json.dumps(
            {
                "version": snp._BASELINE_SCHEMA_VERSION,
                "entries": [
                    {
                        "package": "aws-sdk",
                        "file": "package/metadata.js",
                        "pattern": "cred-surface-host (outbound)",
                        "severity": "HIGH",
                    }
                ],
            }
        ),
        encoding = "utf-8",
    )
    baseline = snp._load_baseline(str(bl))

    listed = _finding("aws-sdk@2.0.0", "package/metadata.js", "cred-surface-host (outbound)")
    # A NEW kind of finding in the SAME file is a different pattern -> not suppressed.
    new_kind = _finding("aws-sdk@2.0.0", "package/metadata.js", "obfuscated-blob")
    active, suppressed = snp._partition_baseline([listed, new_kind], baseline)
    assert listed in suppressed
    assert new_kind in active


def test_write_then_load_baseline_roundtrip(tmp_path):
    bl = tmp_path / "out.json"
    findings = [
        _finding("evil@1.0.0", "package/a.js", "obfuscated-blob", snp.CRITICAL),
        _finding("evil@1.0.0", "package/a.js", "obfuscated-blob", snp.CRITICAL),  # dup
        _finding("noise@1.0.0", "package/b.js", "js-env-token", snp.MEDIUM),  # below thresh
    ]
    n = snp._write_baseline(str(bl), findings, snp._SEVERITY_RANK[snp.HIGH])
    assert n == 1  # dedup + MEDIUM excluded
    keys = snp._load_baseline(str(bl))
    assert (snp._norm_pkg_name("evil@1.0.0"), "a.js", "obfuscated-blob") in keys
    # MEDIUM below HIGH threshold -> not written.
    assert all(k[2] != "js-env-token" for k in keys)


def test_legacy_schema_baseline_is_ignored(tmp_path):
    # A pre-v2 baseline stored basenames; its keys are ambiguous under
    # package-relative matching, so a populated legacy file is ignored (fail
    # closed) rather than silently suppressing a different same-named file.
    bl = tmp_path / "legacy.json"
    bl.write_text(
        json.dumps(
            {
                "version": 1,
                "entries": [
                    {"package": "aws-sdk", "file": "index.js", "pattern": "obfuscated-blob"}
                ],
            }
        ),
        encoding = "utf-8",
    )
    assert snp._load_baseline(str(bl)) == set()


def test_committed_baseline_is_empty_and_valid():
    # Shipped baseline must parse and (by design) suppress nothing: the live corpus is clean.
    path = REPO_ROOT / "scripts" / "scan_npm_packages_baseline.json"
    assert path.is_file()
    doc = json.loads(path.read_text(encoding = "utf-8"))
    assert doc.get("entries") == []
    assert snp._load_baseline(str(path)) == set()
