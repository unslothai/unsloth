# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Write-restricted token launcher for Limited mode on Windows.

The first half runs everywhere and pins the launcher's contract (flags, restricting
SIDs, probe evaluation, manifest reconciliation, how os_sandbox selects it and how it
falls back). The second half runs only on Windows against a real restricted token.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from core.inference import os_sandbox
from core.inference import tool_isolation
from core.inference import tools as inference_tools
from core.inference import windows_lpac
from core.inference import windows_restricted_token as token_launcher


SOURCE = Path(token_launcher.__file__).read_text(encoding = "utf-8")


class _FakeLedgerFiles:
    """The ``CreateFileW``/``CloseHandle`` pair the windows_lpac ledger locks with.

    Only the property the launcher depends on is modelled: an exclusive open
    (``dwShareMode`` 0) of a path that is already open fails, and closing the
    handle releases it. ``busy`` is the other Studio process this one cannot see,
    holding the lock for a key this one must therefore not be given.
    """

    def __init__(self, busy = ()) -> None:
        self.busy = {key + windows_lpac._LEDGER_LOCK_SUFFIX for key in busy}
        self.opened: dict[int, str] = {}
        self.attempts: list[tuple[str, int]] = []
        self._next = 9000

    def hold(self, *keys: str) -> None:
        """Let another Studio process take these keys, from now on."""
        self.busy |= {key + windows_lpac._LEDGER_LOCK_SUFFIX for key in keys}

    def create_file(self, path, _access, share, _attributes, disposition, _flags, _template):
        self.attempts.append((os.path.basename(path), share))
        if share == 0 and (
            path in self.opened.values() or os.path.basename(path) in self.busy
        ):
            return windows_lpac._INVALID_HANDLE_VALUE
        self._next += 1
        self.opened[self._next] = path
        return self._next

    def close(self, handle) -> None:
        self.opened.pop(getattr(handle, "value", handle), None)

    def locked(self, key: str) -> bool:
        """Whether the lock file for one key was ever opened with no sharing."""
        return (key + windows_lpac._LEDGER_LOCK_SUFFIX, 0) in self.attempts


@pytest.fixture(autouse = True)
def ledger_state(tmp_path_factory, monkeypatch):
    """A windows_lpac ledger root of this test's own, and a wait it never sits out.

    Every DACL edit the launcher makes now runs inside that ledger, whose lock
    file lives beside the LPAC manifests. A test host with no LOCALAPPDATA cannot
    place one, and a launcher that cannot place its lock refuses every revoke, so
    without this fixture every cleanup test would be a test of the refusal path.
    """
    root = tmp_path_factory.mktemp("lpac-ledger")
    monkeypatch.setattr(windows_lpac, "_manifest_root", lambda: str(root))
    # Handles and depths are process-lived, so each test starts from its own.
    monkeypatch.setattr(windows_lpac, "_LEDGER_MUTEXES", {})
    monkeypatch.setattr(windows_lpac, "_LEDGER_FILE_LOCKS", {})
    monkeypatch.setattr(windows_lpac, "_LEDGER_DEPTH", {})
    monkeypatch.setattr(windows_lpac, "_LEDGER_REFUSED", {})
    monkeypatch.setattr(windows_lpac, "_LEDGER_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(token_launcher, "_DEFERRED_CLEANUPS", [])
    return root


@pytest.fixture
def isolated_capability_cache():
    before = dict(os_sandbox._capability_cache)
    os_sandbox._capability_cache.clear()
    try:
        yield
    finally:
        os_sandbox._capability_cache.clear()
        os_sandbox._capability_cache.update(before)


_DENIED = "PermissionError: [Errno 13] Permission denied"


def _good_findings(sid_text: str) -> dict:
    """What the probe child reports on a host that matches the documented model."""
    return {
        "restricted": True,
        "restricted_sids": [sid_text, "S-1-5-5-0-1234", "S-1-1-0"],
        "privileges": 1,
        "in_job": True,
        "user_sid": "S-1-5-21-1-2-3-1001",
        "integrity_sid": "S-1-16-8192",
        "interpreter_readable": True,
        "secret_exists": True,
        "secret_readable": True,
        "secret_writable": _DENIED,
        "sibling_writable": _DENIED,
        # The directory granted the token's own user SID and never the launch
        # SID: readable, because the first access check allows it, and refused a
        # write, because only the restricting SIDs decide that.
        "user_only_readable": True,
        "user_only_writable": _DENIED,
        "workdir_readable": True,
        "workdir_writable": True,
        "temp_writable": True,
        "temp_is_private": True,
        "devnull": True,
        # The named pipe is a disclosure, the anonymous one a requirement: the
        # first takes a descriptor the named pipe filesystem supplies, the
        # second takes the token's default DACL, which this launcher edits.
        "pipe": True,
        "anonymous_pipe": True,
    }


def _profile_denied_findings(sid_text: str) -> dict:
    """The same host with an administrator's ACLs: the fence holds, reads do not.

    Staging round 16 on windows-2022 and windows-latest. The child starts, every
    write outside the workdir is refused, and a file in the user profile cannot
    be read at all because it is reachable only through a group this token
    disabled.
    """
    return {
        **_good_findings(sid_text),
        "secret_exists": False,
        "secret_readable": _DENIED,
        "secret_writable": _DENIED,
    }


# ── contract pins (every platform) ───────────────────────────────────────────


def test_public_api_profile_and_token_flags_are_pinned():
    assert token_launcher.__all__ == ["WindowsRestrictedTokenBackend"]
    backend = token_launcher.WindowsRestrictedTokenBackend()
    assert backend.identity == "windows-restricted-token"
    assert backend.profile_id == "windows-restricted-token-write-isolation-v1"
    # The pre-probe default is the more disclosing of the two profile codes, and
    # it names the pipe restriction as well: a launch must never claim more
    # confinement, or more capability, than the probe observed.
    assert backend.limitations == (
        "user_profile_readable",
        "network_unrestricted",
        "everyone_writable_objects_writable",
        "named_pipes_denied",
    )
    assert token_launcher._LIMITATIONS_PROFILE_UNREADABLE == (
        "user_profile_unreadable",
        "network_unrestricted",
        "everyone_writable_objects_writable",
    )
    assert token_launcher._disclosed_limitations(
        profile_readable = True, named_pipes = True
    ) == token_launcher._LIMITATIONS
    assert token_launcher._disclosed_limitations(
        profile_readable = True, named_pipes = False
    ) == (*token_launcher._LIMITATIONS, token_launcher._LIMITATION_NAMED_PIPES_DENIED)
    assert token_launcher._disclosed_limitations(profile_readable = False, named_pipes = True) == (
        token_launcher._LIMITATIONS_PROFILE_UNREADABLE
    )
    # DISABLE_MAX_PRIVILEGE | LUA_TOKEN | WRITE_RESTRICTED, the Codex / DeepSeek flag set.
    assert token_launcher._RESTRICTED_TOKEN_FLAGS == 0x1 | 0x4 | 0x8
    assert token_launcher._RESTRICTED_TOKEN_FLAGS & 0x2 == 0  # never SANDBOX_INERT
    # The token is a copy of Studio's own primary token; no privilege is needed to assign it.
    assert "OpenProcessToken(" in SOURCE
    assert "CreateProcessAsUserW(" in SOURCE
    assert "subprocess.Popen(" not in SOURCE
    assert "_PROC_THREAD_ATTRIBUTE_JOB_LIST" in SOURCE
    assert "Limited processes may not break away from their Job Object" in SOURCE
    # Chromium's broker flags, and a named desktop: a console allocated for the
    # child, or a desktop left to the default, is a start-up failure the payload
    # never survives to explain.
    assert token_launcher._DETACHED_PROCESS == 0x00000008
    assert "_CREATE_NO_WINDOW" not in SOURCE
    assert "StartupInfo.lpDesktop" in SOURCE
    assert '_ADMINISTRATORS_SID = "S-1-5-32-544"' in SOURCE
    # The restricting set, the default DACL, the job attachment order and the
    # failure paths are pinned behaviourally below, against a fake _api().


def test_no_failure_this_module_raises_mentions_lpac():
    # Limited mode is not an AppContainer; messages reused from the shared
    # helpers are restated before they reach a user or a log.
    for node in ast.walk(ast.parse(SOURCE)):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", "") not in {"SandboxUnavailableError", "OSError"}:
            continue
        for text in ast.walk(node):
            if isinstance(text, ast.Constant) and isinstance(text.value, str):
                assert "LPAC" not in text.value, ast.dump(node)


def test_shared_validation_failures_are_restated_for_limited_mode():
    assert token_launcher._limited_wording(
        "the LPAC workdir contains a reparse point: C:\\w\\link"
    ) == "the Limited mode workdir contains a reparse point: C:\\w\\link"
    assert token_launcher._limited_wording("LPAC requires a non-root directory on a local drive") == (
        "Limited mode requires a non-root directory on a local drive"
    )
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "Limited mode workdir") as caught:
        with token_launcher._limited_mode_wording():
            raise os_sandbox.SandboxUnavailableError("the LPAC workdir root is a reparse point")
    assert "LPAC" not in str(caught.value)
    # A message that never mentioned LPAC is re-raised untouched, transient flag included.
    original = os_sandbox.SandboxUnavailableError("a Limited mode private temp path is outside its root")
    with pytest.raises(os_sandbox.SandboxUnavailableError) as unchanged:
        with token_launcher._limited_mode_wording():
            raise original
    assert unchanged.value is original


def test_random_launch_sid_is_a_fresh_domain_sid():
    seen = {token_launcher._random_domain_sid_text() for _ in range(64)}
    assert len(seen) == 64
    for text in seen:
        assert token_launcher._is_launch_sid_text(text)
        parts = text.split("-")
        assert parts[:4] == ["S", "1", "5", "21"]
        assert all(0 <= int(part) < 2**32 for part in parts[4:])
    for bad in ("S-1-5-32-544", "S-1-15-2-1", "S-1-5-21-1-2-3", "S-1-5-21-a-b-c-d", 5, None):
        assert not token_launcher._is_launch_sid_text(bad)


def test_probe_evaluation_requires_every_observation():
    sid = "S-1-5-21-1-2-3-4"
    passed = token_launcher._evaluate_probe(_good_findings(sid), sid_text = sid)
    assert passed.available is True
    assert passed.failures == ()
    assert passed.limitations == token_launcher._LIMITATIONS
    empty = token_launcher._evaluate_probe({}, sid_text = sid)
    assert empty.available is False
    assert empty.held == ()
    # Each requirement is a separate failure, so a report of nothing names them all
    # rather than the first one alphabetically.
    assert "the token is not restricted" in empty.failures
    assert "the child is not inside its Job Object" in empty.failures
    flips = {
        "restricted": (False, "not restricted"),
        "privileges": (3, "kept privileges"),
        "in_job": (False, "not inside its Job Object"),
        "interpreter_readable": (_DENIED, "interpreter could not be read"),
        "secret_writable": (True, "outside the workdir was writable"),
        "sibling_writable": (True, "another launch's temp"),
        "user_only_writable": (True, "granted to the token's user but not to the launch SID"),
        "workdir_readable": (_DENIED, "workdir was not readable"),
        "workdir_writable": (_DENIED, "workdir was not writable"),
        "temp_writable": (_DENIED, "private temp was not writable"),
        "temp_is_private": (False, "TEMP was not redirected"),
        "devnull": ("PermissionError", "NUL device was unavailable"),
        "anonymous_pipe": ("PermissionError", "anonymous pipes were unavailable"),
    }
    for key, (value, fragment) in flips.items():
        findings = _good_findings(sid)
        findings[key] = value
        verdict = token_launcher._evaluate_probe(findings, sid_text = sid)
        assert verdict.available is False, key
        assert any(fragment in failure for failure in verdict.failures), (key, verdict.failures)
        # Only that one requirement failed; the rest are still reported as held.
        assert len(verdict.failures) == 1, (key, verdict.failures)
        assert len(verdict.held) == 14, (key, verdict.held)
        assert fragment in verdict.reason()
        assert "what held" in verdict.reason()

    # The named pipe is the one observation that is disclosed rather than
    # required: multiprocessing.Queue, Pool, Manager and a DataLoader with
    # workers stop working, but the interpreter, torch and single-process
    # training do not, so refusing the launch would cost more than it protects.
    denied = _good_findings(sid)
    denied["pipe"] = "PermissionError: [WinError 5] Access is denied."
    verdict = token_launcher._evaluate_probe(denied, sid_text = sid)
    assert verdict.available is True
    assert token_launcher._LIMITATION_NAMED_PIPES_DENIED in verdict.limitations
    assert "named pipes were refused" in verdict.reason()
    assert "num_workers above zero" in verdict.reason()
    without_launch_sid = _good_findings(sid)
    without_launch_sid["restricted_sids"] = ["S-1-1-0"]
    assert token_launcher._evaluate_probe(without_launch_sid, sid_text = sid).failures == (
        "the launch SID is not a restricting SID",
    )
    without_everyone = _good_findings(sid)
    without_everyone["restricted_sids"] = [sid]
    assert token_launcher._evaluate_probe(without_everyone, sid_text = sid).failures == (
        "Everyone is not a restricting SID",
    )


def test_an_unreadable_user_profile_is_disclosed_rather_than_failed_closed():
    """A child that cannot read the user's documents is confined more tightly
    than Limited mode documents, not less, so it is not a probe failure.

    Whether reads survive the token is a property of the account: on an
    administrator's, a file reachable only through BUILTIN\\Administrators is
    refused on the first access check, because that group is deny-only here. The
    record says which of the two the launch actually got.
    """
    sid = "S-1-5-21-1-2-3-4"
    verdict = token_launcher._evaluate_probe(_profile_denied_findings(sid), sid_text = sid)

    assert verdict.available is True
    assert verdict.failures == ()
    assert verdict.limitations == token_launcher._LIMITATIONS_PROFILE_UNREADABLE
    assert "user_profile_unreadable" in verdict.limitations
    assert "user_profile_readable" not in verdict.limitations
    reason = verdict.reason()
    assert "passed" in reason
    assert "were not readable" in reason
    # The reason carries the observation that produced it, and the principal the
    # denied access check was decided against.
    assert _DENIED in reason
    assert "S-1-5-21-1-2-3-1001" in reason
    # The write fence is still the workdir against its sibling, both of which the
    # child reached, so it is evidence even here.
    assert "the workdir is writable" in verdict.held
    assert "another launch's temp directory refused a write" in verdict.held

    readable = token_launcher._evaluate_probe(_good_findings(sid), sid_text = sid)
    assert readable.limitations == token_launcher._LIMITATIONS
    assert "stayed readable" in readable.reason()
    assert "user_profile_unreadable" not in readable.limitations


def test_the_write_fence_is_proved_where_the_first_access_check_cannot_explain_it():
    """base/granted-user carries the token's user SID and never the launch SID.

    That is the only place the fence can be demonstrated as the fence: its reads
    pass the first access check, so a refused write there can only have come from
    the second, which is what WRITE_RESTRICTED is. Without it, a host whose DACLs
    refuse the child everything looks exactly like a host whose restricting SIDs
    are doing the work, and the ACE this launcher now adds for the user SID would
    have no evidence against it.
    """
    sid = "S-1-5-21-1-2-3-4"
    verdict = token_launcher._evaluate_probe(_good_findings(sid), sid_text = sid)
    assert verdict.available is True
    assert "a directory the launch SID does not restrict refused a write" in verdict.held
    assert "stayed readable, so its refused write was the restricting SIDs" in verdict.reason()

    # A write that got through there is the fence failing, not a disclosure.
    writable = {**_good_findings(sid), "user_only_writable": True}
    broken = token_launcher._evaluate_probe(writable, sid_text = sid)
    assert broken.available is False
    assert broken.failures == (
        "a directory granted to the token's user but not to the launch SID was writable",
    )

    # And a refusal that the DACL alone explains is still accepted, but it is
    # named as the weaker evidence it is rather than counted as proof.
    unreadable = {**_good_findings(sid), "user_only_readable": _DENIED}
    weak = token_launcher._evaluate_probe(unreadable, sid_text = sid)
    assert weak.available is True
    assert "not by itself evidence about the restricting SIDs" in weak.reason()
    assert _DENIED in weak.reason()


def test_a_broken_sandbox_still_fails_closed_whatever_the_profile_reads():
    """The disclosure never rescues a launch that would make Limited mode a lie."""
    sid = "S-1-5-21-1-2-3-4"
    for key, value in (
        ("interpreter_readable", _DENIED),
        ("secret_writable", True),
        ("sibling_writable", True),
        ("user_only_writable", True),
        ("in_job", False),
        ("workdir_writable", _DENIED),
        ("restricted", False),
    ):
        for findings in (_good_findings(sid), _profile_denied_findings(sid)):
            findings[key] = value
            verdict = token_launcher._evaluate_probe(findings, sid_text = sid)
            assert verdict.available is False, (key, findings["secret_readable"])
            assert verdict.reason().startswith("the restricted-token live probe failed: ")


def test_the_verdict_reports_every_failure_at_once():
    """The old probe returned the first surprise and stopped, which is how a host
    reported an unreadable profile without ever saying whether its workdir worked.
    """
    sid = "S-1-5-21-1-2-3-4"
    findings = _profile_denied_findings(sid)
    findings["workdir_writable"] = _DENIED
    findings["in_job"] = False
    findings["devnull"] = "PermissionError: NUL"

    verdict = token_launcher._evaluate_probe(findings, sid_text = sid)

    assert len(verdict.failures) == 3
    reason = verdict.reason()
    for fragment in ("workdir was not writable", "not inside its Job Object", "NUL device"):
        assert fragment in reason, fragment
    # And what did hold is still named, so the failure is diagnosable.
    assert "the token is restricted" in verdict.held
    assert "what held: " in reason


def test_a_verdict_takes_host_side_notes_without_losing_its_own():
    verdict = token_launcher._ProbeVerdict(held = ("a",), notes = ("first",))
    assert verdict.noting("") is verdict
    assert verdict.noting("second").notes == ("first", "second")
    assert verdict.noting("second").held == ("a",)
    assert verdict.notes == ("first",)  # frozen: the original is untouched


def test_the_probe_payload_reports_values_rather_than_bare_booleans():
    """The child reports what happened, and the host decides what it meant.

    A bare False cannot separate "the sandbox refused this" from "the path was
    not there", and staging round 16 turned on exactly that distinction.
    """
    payload = token_launcher._PROBE_PAYLOAD
    ast.parse(payload)  # the child is never syntax-checked on a Windows host first
    for fragment in (
        "def describe(exc)",
        '"interpreter_readable"',
        '"workdir_readable"',
        '"secret_exists"',
        '"user_sid"',
        '"integrity_sid"',
        '"user_only_readable"',
        '"user_only_writable"',
        "input.txt",
    ):
        assert fragment in payload, fragment
    assert "return False" not in payload
    assert payload.count("describe(exc)") >= 4
    # argv: the secret, the sibling, the user-SID directory, then the private
    # temp the host substitutes in once the launch identity exists.
    assert "secret, sibling, user_only = sys.argv[1], sys.argv[2], sys.argv[3]" in payload
    assert "os.path.normcase(sys.argv[4])" in payload


def _probing_backend(monkeypatch, verdict_for):
    """A backend whose live probe is replaced, on a host pretending to be Windows."""
    monkeypatch.setattr(token_launcher, "_is_windows", lambda: True)
    monkeypatch.setattr(windows_lpac, "_api", lambda: SimpleNamespace())
    monkeypatch.setattr(
        token_launcher.WindowsRestrictedTokenBackend,
        "reconcile_stale_manifests",
        lambda self: None,
    )
    monkeypatch.setattr(
        token_launcher.WindowsRestrictedTokenBackend, "_live_probe", lambda self: verdict_for()
    )
    return token_launcher.WindowsRestrictedTokenBackend()


def test_probe_publishes_the_disclosure_it_observed(monkeypatch):
    """os_sandbox builds the execution record from backend.limitations, so the
    probe has to replace the class default with what it actually saw.

    Claiming user_profile_readable on a host where the profile was unreadable, or
    the reverse, would be a false statement in the record and in the dropdown.
    """
    sid = "S-1-5-21-1-2-3-4"
    for findings, expected in (
        (_good_findings(sid), token_launcher._LIMITATIONS),
        (_profile_denied_findings(sid), token_launcher._LIMITATIONS_PROFILE_UNREADABLE),
    ):
        backend = _probing_backend(
            monkeypatch, lambda findings = findings: token_launcher._evaluate_probe(
                findings, sid_text = sid
            )
        )
        capability = backend.probe()
        assert capability.available is True, capability.reason
        assert capability.profile_id == token_launcher._PROFILE_ID
        assert capability.protection_state == "preview"
        assert capability.limitations == expected
        assert backend.limitations == expected
        assert backend.probe() is capability  # cached until forced


def test_a_failed_probe_leaves_the_disclosure_at_its_cautious_default(monkeypatch):
    sid = "S-1-5-21-1-2-3-4"
    broken = _good_findings(sid)
    broken["sibling_writable"] = True
    backend = _probing_backend(
        monkeypatch, lambda: token_launcher._evaluate_probe(broken, sid_text = sid)
    )

    capability = backend.probe()

    assert capability.available is False
    assert "another launch's temp directory was writable" in capability.reason
    assert capability.limitations == ()
    assert backend.limitations == token_launcher._LIMITATIONS_UNPROBED


def test_a_probe_that_cannot_run_at_all_is_reported_as_itself(monkeypatch):
    def explode(self):
        raise RuntimeError("ctypes exploded")

    backend = _probing_backend(monkeypatch, lambda: None)
    monkeypatch.setattr(token_launcher.WindowsRestrictedTokenBackend, "_live_probe", explode)

    capability = backend.probe()

    assert capability.available is False
    assert capability.reason == (
        "the restricted-token live probe could not run: ctypes exploded"
    )


def _dacl_api(kinds, *, result: int = 0):
    """A fake ``_api()`` whose DACL walk hands back one ACE per entry in ``kinds``."""
    aces = [windows_lpac._ACE_HEADER(kind, 0, 20) for kind in kinds]
    acl = windows_lpac._ACL(2, 0, 64, len(kinds), 0)
    names = iter(["S-1-5-18", "S-1-5-32-544", "S-1-5-21-1-2-3-1001", "S-1-1-0"])
    freed: list[str] = []

    def get_named(path, kind, information, owner, group, dacl, sacl, descriptor):
        if result == 0:
            dacl._obj.value = ctypes.addressof(acl)
            descriptor._obj.value = 4242
        return result

    def get_ace(pointer, index, out):
        out._obj.value = ctypes.addressof(aces[index])
        return 1

    def convert(sid, out):
        out._obj.value = next(names)
        return 1

    api = SimpleNamespace(
        advapi32 = SimpleNamespace(
            GetNamedSecurityInfoW = get_named, GetAce = get_ace, ConvertSidToStringSidW = convert
        ),
        kernel32 = SimpleNamespace(LocalFree = lambda value: freed.append(getattr(value, "value", value))),
    )
    api.freed = freed
    api.held = (aces, acl)  # kept alive for the duration of the walk
    return api


def test_a_denied_read_names_the_principals_the_file_does_grant(monkeypatch):
    """The next question after "the read was refused" is "refused against what".

    A file reachable only through BUILTIN\\Administrators explains the refusal by
    itself, because that group is deny-only in this token.
    """
    api = _dacl_api((0, 0))
    monkeypatch.setattr(windows_lpac, "_api", lambda: api)

    assert token_launcher._dacl_trustees("C:\\probe\\secret.txt") == ("S-1-5-18", "S-1-5-32-544")
    assert 4242 in api.freed  # the descriptor is released even on the happy path
    assert token_launcher._denied_read_note("C:\\probe\\secret.txt").startswith("that file grants ")

    # An object ACE puts its SID somewhere else, so a diagnostic skips it rather
    # than reading the wrong bytes as a SID.
    monkeypatch.setattr(windows_lpac, "_api", lambda: _dacl_api((5, 0)))
    assert token_launcher._dacl_trustees("C:\\probe\\secret.txt") == ("S-1-5-18",)

    # A descriptor that cannot be read is no diagnosis, never an exception.
    monkeypatch.setattr(windows_lpac, "_api", lambda: _dacl_api((0,), result = 5))
    assert token_launcher._dacl_trustees("C:\\probe\\secret.txt") == ()
    assert token_launcher._denied_read_note("C:\\probe\\secret.txt") == ""


def test_a_denied_root_is_named_with_the_principals_it_does_grant(monkeypatch):
    """Round 18 reported three refusals and nothing about the DACLs behind them.

    The child only sees "Permission denied". Whether the two ACEs this launcher
    adds actually reached the directory is readable from the host side alone, and
    it is what separates "the ACE was not enough" from "the ACE was never made".
    """
    monkeypatch.setattr(token_launcher, "_dacl_trustees", lambda path: ("S-1-5-18", "S-1-5-32-544"))
    findings = {"workdir_readable": True, "workdir_writable": _DENIED, "temp_writable": True}
    roots = (
        ("the workdir", "C:\\w", ("workdir_readable", "workdir_writable")),
        ("the private temp", "C:\\t", ("temp_writable",)),
    )

    # Only the roots the child could not reach; a root that worked is not noise.
    assert token_launcher._denied_root_notes(findings, roots) == (
        "the workdir grants S-1-5-18, S-1-5-32-544",
    )

    def explode(path):
        raise OSError(5, "GetNamedSecurityInfoW")

    monkeypatch.setattr(token_launcher, "_dacl_trustees", explode)
    assert token_launcher._denied_root_notes(findings, roots) == (
        "the workdir grants no DACL this host would read",
    )


def test_the_probes_user_sid_directory_is_best_effort():
    """A host that will not let the evidence directory be granted still gets a launcher.

    The refused write there is then explained by neither access check in
    particular, so the verdict says that rather than counting it as proof.
    """
    assert "os.mkdir(user_only)" in SOURCE
    assert 'granted-user"' in SOURCE
    assert "carries neither of this launch's SIDs" in SOURCE
    # And the grant is undone before the probe tree is removed, not only with it.
    assert "_revoke_sid(user_only" in SOURCE


def test_the_denied_read_note_never_replaces_the_diagnosis(monkeypatch):
    def explode():
        raise OSError(5, "GetNamedSecurityInfoW")

    monkeypatch.setattr(windows_lpac, "_api", explode)
    assert token_launcher._denied_read_note("C:\\probe\\secret.txt") == ""


@pytest.mark.skipif(sys.platform == "win32", reason = "the launcher is Windows-only")
def test_launcher_is_unavailable_off_windows(tmp_path):
    backend = token_launcher.WindowsRestrictedTokenBackend()
    capability = backend.probe()
    assert capability.available is False
    assert capability.qualified is False
    assert "require Windows" in capability.reason
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "require Windows"):
        backend.prepare(os_sandbox.ToolLaunchPlan(argv = ("x",), workdir = str(tmp_path), env = {}))
    assert os_sandbox._limited_isolation_backend() is None


def test_launch_environment_redirects_temp_and_keeps_system_root(monkeypatch):
    identity = SimpleNamespace(private_temp = r"C:\Users\u\AppData\Local\Unsloth\Studio\limited-temp\a")
    monkeypatch.setenv("SystemRoot", r"C:\WINDOWS")
    env = token_launcher._launch_environment(
        {"PATH": "p", "Temp": r"C:\old", "TMP": r"C:\old", "PYTHONIOENCODING": "utf-8"}, identity
    )
    assert env["TEMP"] == env["TMP"] == identity.private_temp
    assert "Temp" not in env
    assert env["PATH"] == "p"
    assert env["SystemRoot"] == r"C:\WINDOWS"
    kept = token_launcher._launch_environment({"SYSTEMROOT": r"D:\W"}, identity)
    assert kept["SYSTEMROOT"] == r"D:\W" and "SystemRoot" not in kept


def _manifest(root: Path, sid: str, **overrides) -> Path:
    payload = {
        "version": 1,
        "kind": "restricted-token",
        "sid": sid,
        "workdir": str(root / "work"),
        "private_temp": str(root / "temp" / ("a" * 24)),
        "granted_roots": [str(root / "work"), str(root / "temp" / ("a" * 24))],
        "owner_pid": 4242,
        "owner_created": 7,
    }
    payload.update(overrides)
    path = root / (token_launcher._MANIFEST_PREFIX + sid + ".json")
    path.write_text(json.dumps(payload), encoding = "utf-8")
    return path


def _ledger_kernel32(ledger: _FakeLedgerFiles | None = None, **extra) -> SimpleNamespace:
    """A minimal ``kernel32`` that can still open and release the ledger lock file.

    A host that answers no ``CreateFileW`` has no ledger, and a launcher with no
    ledger refuses every revoke, so a fake meant to exercise a revoke has to
    answer it.
    """
    ledger = ledger or _FakeLedgerFiles()
    return SimpleNamespace(
        CreateFileW = ledger.create_file, CloseHandle = ledger.close, **extra
    )


def _unmapped_sid_api(freed: list, *, resolves: bool = False) -> SimpleNamespace:
    """A fake ``_api()`` whose LookupAccountSidW resolves no SID unless asked to."""

    def lookup(system, sid, name, name_length, domain, domain_length, use):
        return 1 if resolves else 0

    return SimpleNamespace(
        kernel32 = _ledger_kernel32(LocalFree = freed.append),
        advapi32 = SimpleNamespace(LookupAccountSidW = lookup),
    )


def test_manifest_parser_accepts_only_this_launchers_records(tmp_path):
    sid = "S-1-5-21-1-2-3-4"
    good = _manifest(tmp_path, sid)
    assert token_launcher._parse_manifest(good)["sid"] == sid
    assert token_launcher._parse_manifest(_manifest(tmp_path, sid, version = 2)) is None
    assert token_launcher._parse_manifest(_manifest(tmp_path, sid, kind = "lpac")) is None
    assert token_launcher._parse_manifest(_manifest(tmp_path, "S-1-15-2-1-2-3-4-5")) is None
    assert token_launcher._parse_manifest(_manifest(tmp_path, sid, granted_roots = ["rel"])) is None
    assert token_launcher._parse_manifest(_manifest(tmp_path, sid, owner_pid = "1")) is None
    # granted_roots drives an ACL revoke, so it must be exactly the two roots
    # _create_identity grants: the manifest's own workdir and private temp.
    workdir = str(tmp_path / "work")
    private_temp = str(tmp_path / "temp" / ("a" * 24))
    for planted in (
        [workdir, private_temp, str(tmp_path / "documents")],
        [workdir],
        [str(tmp_path / "documents"), private_temp],
        [workdir, workdir],
        [],
    ):
        assert token_launcher._parse_manifest(
            _manifest(tmp_path, sid, granted_roots = planted)
        ) is None, planted
    assert token_launcher._parse_manifest(
        _manifest(tmp_path, sid, granted_roots = [private_temp, workdir])
    )["granted_roots"] == [private_temp, workdir]
    foreign = tmp_path / "unsloth.studio.deadbeef.json"  # an AppContainer manifest
    foreign.write_text(json.dumps({"version": 1, "moniker": "unsloth.studio.deadbeef"}))
    assert token_launcher._parse_manifest(foreign) is None
    broken = tmp_path / (token_launcher._MANIFEST_PREFIX + sid + "x.json")
    broken.write_text("{not json")
    assert token_launcher._parse_manifest(broken) is None


def test_reconcile_cleans_only_manifests_whose_owner_is_gone(tmp_path, monkeypatch):
    manifests = tmp_path / "manifests"
    temp_root = tmp_path / "temp"
    manifests.mkdir()
    temp_root.mkdir()
    (tmp_path / "work").mkdir()
    dead_sid = "S-1-5-21-1-1-1-1"
    live_sid = "S-1-5-21-2-2-2-2"
    dead_temp = temp_root / ("a" * 24)
    live_temp = temp_root / ("b" * 24)
    dead_temp.mkdir()
    live_temp.mkdir()
    (dead_temp / "scratch.txt").write_text("x")
    dead = _manifest(manifests, dead_sid, workdir = str(tmp_path / "work"), private_temp = str(dead_temp),
                     granted_roots = [str(tmp_path / "work"), str(dead_temp)], owner_pid = 4242)
    live = _manifest(manifests, live_sid, workdir = str(tmp_path / "work"), private_temp = str(live_temp),
                     granted_roots = [str(tmp_path / "work"), str(live_temp)], owner_pid = 5151)
    # A manifest whose file name does not match its SID is evidence, never acted on.
    mismatched = manifests / (token_launcher._MANIFEST_PREFIX + "S-1-5-21-3-3-3-3.json")
    mismatched.write_text(dead.read_text(encoding = "utf-8"), encoding = "utf-8")

    revoked: list[tuple[str, str]] = []
    freed: list[int] = []
    monkeypatch.setattr(token_launcher, "_manifest_root", lambda: str(manifests))
    monkeypatch.setattr(token_launcher, "_temp_root", lambda: str(temp_root))
    monkeypatch.setattr(token_launcher, "_sid_from_text", lambda text: ctypes.c_void_p(hash(text) & 0xFFFF or 1))
    monkeypatch.setattr(
        windows_lpac, "_process_identity", lambda pid = None: (5151, 7) if pid == 5151 else None
    )
    monkeypatch.setattr(windows_lpac, "_revoke_sid", lambda path, sid, **kw: revoked.append((path, sid.value)))
    monkeypatch.setattr(windows_lpac, "_api", lambda: _unmapped_sid_api(freed))
    monkeypatch.setattr(token_launcher, "_last_error", lambda: token_launcher._ERROR_NONE_MAPPED)

    token_launcher.WindowsRestrictedTokenBackend().reconcile_stale_manifests()

    assert not dead.exists()
    assert not dead_temp.exists()
    assert live.exists() and live_temp.exists()
    assert mismatched.exists()
    assert [path for path, _ in revoked] == [str(dead_temp), str(tmp_path / "work")]
    assert len(freed) == 1


def test_identity_cleanup_is_idempotent_and_reports_every_failure(tmp_path, monkeypatch):
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    private = temp_root / ("c" * 24)
    private.mkdir()
    manifest = tmp_path / "m.json"
    manifest.write_text("{}")
    monkeypatch.setattr(token_launcher, "_temp_root", lambda: str(temp_root))
    calls: list[str] = []

    def revoke(path, sid, **kw):
        calls.append(path)
        if path.endswith("work"):
            raise OSError(5, "denied")

    monkeypatch.setattr(windows_lpac, "_revoke_sid", revoke)
    freed: list = []
    monkeypatch.setattr(
        windows_lpac, "_api", lambda: SimpleNamespace(kernel32 = _ledger_kernel32(LocalFree = freed.append))
    )
    monkeypatch.setattr(token_launcher, "_sid_from_text", lambda text: ctypes.c_void_p(2))
    identity = token_launcher._LaunchIdentity(
        ctypes.c_void_p(1), "S-1-5-21-9-9-9-9", str(tmp_path / "work"), str(private), str(manifest),
        (str(tmp_path / "work"), str(private)), 1, 1,
    )
    with pytest.raises(OSError, match = "ACL .*work"):
        identity.cleanup()
    # The temp went even though a revoke failed, and the manifest stayed as evidence.
    assert not private.exists()
    assert manifest.exists()
    assert identity.cleaned is False
    # The SID allocation is released even by the failing path, and the retry
    # converts the recorded SID text again rather than reusing freed memory.
    assert [item.value for item in freed] == [1]
    calls.clear()
    monkeypatch.setattr(windows_lpac, "_revoke_sid", lambda path, sid, **kw: calls.append(path))
    identity.cleanup()
    assert identity.cleaned is True
    assert not manifest.exists()
    identity.cleanup()  # idempotent
    assert calls == [str(private), str(tmp_path / "work")]
    assert [item.value for item in freed] == [1, 2]


# ── Win32 call behaviour (every platform, against a fake _api()) ─────────────


class _WinApiRecorder:
    """A windows_lpac._api() stand-in that records the ordered Win32 calls."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.closed: list[int] = []
        self.freed: list[int] = []
        self.create_token_result = 1
        self.create_process_result = 1
        self.resume_result = 0
        self.job_list_supported = True
        self.active_processes = [0]
        self.last_error = token_launcher._ERROR_INSUFFICIENT_BUFFER
        # The window station and desktop this process is connected to, and the
        # error a host that refuses their DACL edit would report.
        self.user_object_names = {101: "WinSta0", 102: "Default"}
        self.user_object_error = 0
        self.user_object_dacl = 4096  # 0 stands for a NULL DACL: allows everyone
        self.acl_error = 0
        self.desktop = None
        # The ledger every DACL edit is now taken under. A recorder that could
        # not answer CreateFileW would be a host with no lock at all, on which
        # every revoke is refused rather than performed.
        self.ledger = _FakeLedgerFiles()
        self.kernel32 = SimpleNamespace(
            CreateFileW = self.ledger.create_file,
            GetCurrentProcess = lambda: 11,
            GetCurrentThreadId = lambda: 7,
            CloseHandle = self._close,
            LocalFree = self._free,
            InitializeProcThreadAttributeList = self._initialize_attributes,
            UpdateProcThreadAttribute = self._update_attribute,
            DeleteProcThreadAttributeList = lambda attributes: self._note(
                "DeleteProcThreadAttributeList"
            ),
            AssignProcessToJobObject = lambda job, process: self._note("AssignProcessToJobObject"),
            ResumeThread = self._resume,
            TerminateProcess = self._terminate,
            QueryInformationJobObject = self._query_job,
        )
        self.advapi32 = SimpleNamespace(
            OpenProcessToken = self._open_process_token,
            CreateRestrictedToken = self._create_restricted_token,
            CreateProcessAsUserW = self._create_process,
            GetSecurityDescriptorDacl = self._security_descriptor_dacl,
            SetEntriesInAclW = self._set_entries_in_acl,
            InitializeSecurityDescriptor = lambda descriptor, revision: self._note(
                "InitializeSecurityDescriptor"
            ),
            SetSecurityDescriptorDacl = lambda descriptor, present, acl, defaulted: self._note(
                "SetSecurityDescriptorDacl"
            ),
        )
        self.user32 = SimpleNamespace(
            GetProcessWindowStation = lambda: 101,
            GetThreadDesktop = self._thread_desktop,
            GetUserObjectInformationW = self._user_object_information,
            GetUserObjectSecurity = self._get_user_object_security,
            SetUserObjectSecurity = self._set_user_object_security,
        )

    def names(self) -> list[str]:
        return [call[0] for call in self.calls]

    def call(self, name: str) -> tuple:
        return next(call for call in self.calls if call[0] == name)

    def _note(self, name: str, *args) -> int:
        self.calls.append((name, *args))
        return 1

    @staticmethod
    def _value(handle):
        return getattr(handle, "value", handle)

    def _close(self, handle) -> int:
        self.ledger.close(handle)
        self.closed.append(self._value(handle))
        return self._note("CloseHandle", self._value(handle))

    def _free(self, sid) -> int:
        self.freed.append(self._value(sid))
        self._note("LocalFree", self._value(sid))
        return 0

    def _open_process_token(self, process, access, out) -> int:
        out._obj.value = 4321
        return self._note("OpenProcessToken", access)

    def _create_restricted_token(
        self, source, flags, disable_count, disable, privilege_count, privileges,
        restrict_count, restrict, out,
    ) -> int:
        self.calls.append((
            "CreateRestrictedToken",
            flags,
            disable_count,
            None if disable is None else len(disable),
            [] if disable is None else [(entry.Sid, entry.Attributes) for entry in disable],
            privilege_count,
            privileges,
            restrict_count,
            len(restrict),
            [(entry.Sid, entry.Attributes) for entry in restrict],
        ))
        if not self.create_token_result:
            return 0
        out._obj.value = 9999
        return 1

    def _initialize_attributes(self, attributes, count, flags, size) -> int:
        self._note("InitializeProcThreadAttributeList", count)
        if attributes is None:
            size._obj.value = 64
            return 0
        return 1

    def _update_attribute(self, attributes, flags, attribute, value, size, a, b) -> int:
        self._note("UpdateProcThreadAttribute", attribute)
        if attribute == windows_lpac._PROC_THREAD_ATTRIBUTE_JOB_LIST and not self.job_list_supported:
            self.last_error = token_launcher._ERROR_NOT_SUPPORTED
            return 0
        return 1

    def _create_process(
        self, token, application, command_line, process_attributes, thread_attributes,
        inherit, flags, environment, workdir, startup, process_information,
    ) -> int:
        self.desktop = startup.contents.lpDesktop
        self.calls.append(("CreateProcessAsUserW", self._value(token), flags, workdir))
        if not self.create_process_result:
            return 0
        info = process_information._obj
        info.hProcess = 1234
        info.hThread = 5678
        info.dwProcessId = 4242
        return 1

    def _thread_desktop(self, thread) -> int:
        self._note("GetThreadDesktop", thread)
        return 102

    def _user_object_information(self, handle, kind, buffer, length, needed) -> int:
        self._note("GetUserObjectInformationW", self._value(handle))
        buffer.value = self.user_object_names[self._value(handle)]
        return 1

    def _get_user_object_security(self, handle, information, buffer, length, needed) -> int:
        self._note("GetUserObjectSecurity", self._value(handle))
        if buffer is None:
            needed._obj.value = 64
            self.last_error = token_launcher._ERROR_INSUFFICIENT_BUFFER
            return 0
        if self.user_object_error:
            self.last_error = self.user_object_error
            return 0
        return 1

    def _set_user_object_security(self, handle, information, descriptor) -> int:
        if self.user_object_error:
            self.last_error = self.user_object_error
            return 0
        return self._note("SetUserObjectSecurity", self._value(handle))

    def _security_descriptor_dacl(self, descriptor, present, dacl, defaulted) -> int:
        present._obj.value = 1
        dacl._obj.value = self.user_object_dacl
        return self._note("GetSecurityDescriptorDacl")

    def _set_entries_in_acl(self, count, entries, old, out) -> int:
        entry = entries._obj
        self.calls.append((
            "SetEntriesInAclW",
            entry.grfAccessPermissions,
            entry.grfAccessMode,
            entry.grfInheritance,
            self._value(old),
        ))
        if self.acl_error:
            return self.acl_error
        out._obj.value = 8192
        return 0

    def _resume(self, thread) -> int:
        self._note("ResumeThread")
        return self.resume_result

    def _terminate(self, process, code) -> int:
        return self._note("TerminateProcess", self._value(process))

    def _query_job(self, handle, kind, info, length, returned) -> int:
        self._note("QueryInformationJobObject", kind)
        remaining = self.active_processes.pop(0) if len(self.active_processes) > 1 else (
            self.active_processes[0]
        )
        info._obj.ActiveProcesses = remaining
        return 1


class _FakeJob:
    """The _WindowsJob surface _spawn_restricted and the drain wait use."""

    def __init__(self, recorder: _WinApiRecorder, handle: int = 55) -> None:
        self._handle = handle
        self._recorder = recorder

    def terminate(self) -> bool:
        self._recorder._note("job.terminate")
        return True

    def close(self) -> None:
        self._recorder._note("job.close")
        self._handle = None


_STDIO_PLAN = {
    "stdout": subprocess.PIPE,
    "stderr": subprocess.STDOUT,
    "stdin": subprocess.DEVNULL,
    "text": True,
    "encoding": "utf-8",
    "errors": "replace",
    "close_fds": True,
}


def _launch_identity(tmp_path) -> object:
    return token_launcher._LaunchIdentity(
        ctypes.c_void_p(77), "S-1-5-21-1-1-1-1", str(tmp_path), str(tmp_path / "temp"),
        str(tmp_path / "m.json"), (str(tmp_path),), 1, 1,
    )


def _prepared(tmp_path) -> os_sandbox.PreparedSandboxLaunch:
    return os_sandbox.PreparedSandboxLaunch(
        argv = (sys.executable, "-c", "pass"),
        workdir = str(tmp_path),
        env = {"PATH": "p", "TEMP": str(tmp_path)},
        preexec_fn = None,
        backend = token_launcher._BACKEND_IDENTITY,
    )


def test_create_restricted_token_pins_the_flags_and_the_restricting_set(monkeypatch):
    recorder = _WinApiRecorder()
    sids: dict[str, int] = {}

    def sid_from_text(text):
        sids.setdefault(text, len(sids) + 1)
        return ctypes.c_void_p(sids[text])

    groups = {
        token_launcher._TOKEN_LOGON_SID: ["S-1-5-5-0-9999"],
        token_launcher._TOKEN_GROUPS: ["S-1-5-32-544", "S-1-5-21-1-2-3-513"],
    }
    dacl: list[tuple[int, int]] = []
    monkeypatch.setattr(windows_lpac, "_api", lambda: recorder)
    monkeypatch.setattr(token_launcher, "_sid_from_text", sid_from_text)
    monkeypatch.setattr(token_launcher, "_token_group_sids", lambda api, token, kind: groups[kind])
    monkeypatch.setattr(
        token_launcher, "_set_default_dacl",
        lambda api, token, sid: dacl.append((recorder._value(token), sid.value)),
    )
    identity = SimpleNamespace(sid = ctypes.c_void_p(77), sid_string = "S-1-5-21-1-1-1-1")

    handle = token_launcher._create_restricted_token(identity)

    assert handle.value == 9999
    (
        _, flags, disable_count, disable_length, disable_entries,
        privilege_count, privileges, restrict_count, restrict_length, restrict_entries,
    ) = recorder.call("CreateRestrictedToken")
    assert flags == token_launcher._RESTRICTED_TOKEN_FLAGS == 0x1 | 0x4 | 0x8
    assert privilege_count == 0 and privileges is None
    # Counts must match the arrays they describe: they are adjacent DWORDs.
    assert disable_count == disable_length == 1
    assert disable_entries == [(sids["S-1-5-32-544"], 0)]
    assert restrict_count == restrict_length == 3
    # The launch SID first, then the logon SID, then Everyone; Attributes zero.
    assert [entry[0] for entry in restrict_entries] == [
        sids["S-1-5-21-1-1-1-1"], sids["S-1-5-5-0-9999"], sids["S-1-1-0"]
    ]
    assert [entry[1] for entry in restrict_entries] == [0, 0, 0]
    # The default DACL is widened to the launch SID, on the restricted token.
    assert dacl == [(9999, 77)]
    # Every converted SID is released and the source token handle is closed.
    assert sorted(recorder.freed) == sorted(sids.values())
    assert 4321 in recorder.closed


def test_create_restricted_token_declines_when_the_logon_sid_is_unavailable(monkeypatch):
    recorder = _WinApiRecorder()
    monkeypatch.setattr(windows_lpac, "_api", lambda: recorder)
    monkeypatch.setattr(token_launcher, "_sid_from_text", lambda text: ctypes.c_void_p(1))
    monkeypatch.setattr(token_launcher, "_set_default_dacl", lambda api, token, sid: None)
    identity = SimpleNamespace(sid = ctypes.c_void_p(77), sid_string = "S-1-5-21-1-1-1-1")

    # An empty logon SID list would silently drop it from the restricting set.
    monkeypatch.setattr(token_launcher, "_token_group_sids", lambda api, token, kind: [])
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "logon SID"):
        token_launcher._create_restricted_token(identity)

    # A failed query is not swallowed either; both fall back to the process guard.
    def failing(api, token, kind):
        raise OSError(5, "GetTokenInformation(28)")

    monkeypatch.setattr(token_launcher, "_token_group_sids", failing)
    with pytest.raises(OSError):
        token_launcher._create_restricted_token(identity)
    assert "CreateRestrictedToken" not in recorder.names()
    assert recorder.closed == [4321, 4321]  # the source token is released both times


def test_administrators_lookup_failure_stays_a_swallow(monkeypatch):
    recorder = _WinApiRecorder()
    monkeypatch.setattr(windows_lpac, "_api", lambda: recorder)
    monkeypatch.setattr(token_launcher, "_sid_from_text", lambda text: ctypes.c_void_p(1))
    monkeypatch.setattr(token_launcher, "_set_default_dacl", lambda api, token, sid: None)

    def groups(api, token, kind):
        if kind == token_launcher._TOKEN_GROUPS:
            raise OSError(5, "GetTokenInformation(2)")
        return ["S-1-5-5-0-9999"]

    monkeypatch.setattr(token_launcher, "_token_group_sids", groups)
    identity = SimpleNamespace(sid = ctypes.c_void_p(77), sid_string = "S-1-5-21-1-1-1-1")
    assert token_launcher._create_restricted_token(identity).value == 9999
    # LUA_TOKEN already covers the Administrators gap, so the token is still built.
    assert recorder.call("CreateRestrictedToken")[2] == 0


def _spawn_with_fake_api(tmp_path, monkeypatch, recorder, job):
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(get_osfhandle = lambda fd: fd))
    monkeypatch.setattr(windows_lpac, "_api", lambda: recorder)
    monkeypatch.setattr(token_launcher, "_last_error", lambda: recorder.last_error)
    resources = token_launcher._LaunchResources(token = wintypes.HANDLE(4711), job = job)
    prepared = _prepared(tmp_path)
    return prepared, resources


def test_spawn_restricted_attaches_the_job_before_it_resumes_the_child(tmp_path, monkeypatch):
    recorder = _WinApiRecorder()
    recorder.active_processes = [1, 0]
    job = _FakeJob(recorder)
    prepared, resources = _spawn_with_fake_api(tmp_path, monkeypatch, recorder, job)

    process = token_launcher._spawn_restricted(
        prepared, dict(_STDIO_PLAN), _launch_identity(tmp_path), resources
    )

    names = recorder.names()
    created = names.index("CreateProcessAsUserW")
    # The job is attached at creation through PROC_THREAD_ATTRIBUTE_JOB_LIST, so
    # no instruction ever runs outside it, and the resume comes last.
    assert ("UpdateProcThreadAttribute", windows_lpac._PROC_THREAD_ATTRIBUTE_JOB_LIST) in recorder.calls
    assert names.index("UpdateProcThreadAttribute") < created < names.index("ResumeThread")
    assert "AssignProcessToJobObject" not in names
    token_value, flags, workdir = recorder.call("CreateProcessAsUserW")[1:]
    assert token_value == 4711
    assert flags & windows_lpac._CREATE_SUSPENDED
    assert workdir == prepared.workdir
    assert process.pid == 4242
    # The token handle is released once the child holds its own reference, and
    # the job belongs to the process from here.
    assert 4711 in recorder.closed
    assert resources.token is None and resources.job is None

    prepared.cleanup()
    assert prepared.cleanup_diagnostics == []
    names = recorder.names()
    # Kill the job, wait for it to actually drain, only then release the handles.
    assert names.index("job.terminate") < names.index("QueryInformationJobObject")
    assert names.index("QueryInformationJobObject") < names.index("job.close")
    assert names.count("QueryInformationJobObject") == 2  # one process left, then none


def test_spawn_restricted_falls_back_to_assigning_the_job_before_the_resume(tmp_path, monkeypatch):
    recorder = _WinApiRecorder()
    recorder.job_list_supported = False
    prepared, resources = _spawn_with_fake_api(tmp_path, monkeypatch, recorder, _FakeJob(recorder))

    token_launcher._spawn_restricted(
        prepared, dict(_STDIO_PLAN), _launch_identity(tmp_path), resources
    )

    names = recorder.names()
    assert names.index("CreateProcessAsUserW") < names.index("AssignProcessToJobObject")
    assert names.index("AssignProcessToJobObject") < names.index("ResumeThread")
    prepared.cleanup()
    assert prepared.cleanup_diagnostics == []


def test_spawn_restricted_terminates_the_suspended_child_when_the_resume_fails(tmp_path, monkeypatch):
    recorder = _WinApiRecorder()
    recorder.resume_result = 0xFFFFFFFF
    job = _FakeJob(recorder)
    prepared, resources = _spawn_with_fake_api(tmp_path, monkeypatch, recorder, job)
    monkeypatch.setattr(windows_lpac, "_winerror", lambda prefix, code = None: OSError(5, prefix))

    with pytest.raises(OSError, match = "ResumeThread"):
        token_launcher._spawn_restricted(
            prepared, dict(_STDIO_PLAN), _launch_identity(tmp_path), resources
        )

    # Nothing ever ran: the still-suspended child is terminated and every handle
    # this launch owns is released.
    assert ("TerminateProcess", 1234) in recorder.calls
    assert 5678 in recorder.closed and 1234 in recorder.closed
    assert 4711 in recorder.closed  # the token was consumed by CreateProcessAsUserW
    assert prepared.cleanup_callbacks == []
    assert resources.job is job  # still the launch's, released by prepare's cleanup
    resources.close()
    assert recorder.names().count("job.close") == 1


def test_spawn_restricted_refuses_a_launch_whose_resources_are_gone(tmp_path, monkeypatch):
    recorder = _WinApiRecorder()
    prepared, resources = _spawn_with_fake_api(tmp_path, monkeypatch, recorder, _FakeJob(recorder))
    resources.close()
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "already released"):
        token_launcher._spawn_restricted(
            prepared, dict(_STDIO_PLAN), _launch_identity(tmp_path), resources
        )


# The two SIDs a launch puts on its roots, and the fake pointers they convert
# to: 3 is the per-launch random SID (the second access check), 4 is the token's
# own user SID (the first). Every grant and revoke below is recorded as a
# (path, pointer) pair so the two can never be confused for one another.
_TOKEN_USER_SID = "S-1-5-21-1-2-3-1001"
_LAUNCH_SID_VALUE = 3
_USER_SID_VALUE = 4


def _prepare_environment(
    tmp_path, monkeypatch, recorder, *, already_granted = (), already_named = None
):
    """Everything prepare() touches on a host, replaced by recorders.

    ``already_granted`` names the roots whose DACL provably already grants the
    token's user SID modify, so no ACE is needed there. ``already_named`` names
    the roots that carry any ACE for that account at all, which is the weaker
    condition cleanup keys off; it defaults to ``already_granted``.
    """
    manifests = tmp_path / "manifests"
    temp_root = tmp_path / "temp"
    work = tmp_path / "work"
    for path in (manifests, temp_root, work):
        path.mkdir()
    granted: list[tuple[str, int]] = []
    revoked: list[tuple[str, int]] = []
    granted_already = {str(path) for path in already_granted}
    named_already = (
        granted_already if already_named is None else {str(path) for path in already_named}
    )
    monkeypatch.setattr(token_launcher, "_is_windows", lambda: True)
    monkeypatch.setattr(token_launcher, "_last_error", lambda: recorder.last_error)
    monkeypatch.setattr(token_launcher, "_manifest_root", lambda: str(manifests))
    monkeypatch.setattr(token_launcher, "_temp_root", lambda: str(temp_root))
    monkeypatch.setattr(
        token_launcher, "_sid_from_text",
        lambda text: ctypes.c_void_p(
            _USER_SID_VALUE if text == _TOKEN_USER_SID else _LAUNCH_SID_VALUE
        ),
    )
    monkeypatch.setattr(
        token_launcher, "_token_user_sid_text", lambda api, token: _TOKEN_USER_SID
    )
    monkeypatch.setattr(token_launcher, "_dacl_names_sid", lambda path, sid: path in named_already)
    monkeypatch.setattr(
        windows_lpac, "_existing_access",
        lambda path, sids, required: path in granted_already,
    )
    monkeypatch.setattr(windows_lpac, "_api", lambda: recorder)
    monkeypatch.setattr(windows_lpac, "_validate_workdir", lambda path: str(work))
    monkeypatch.setattr(windows_lpac, "_canonical_inner_argv", lambda argv, env: tuple(argv))
    monkeypatch.setattr(windows_lpac, "_process_identity", lambda pid = None: (os.getpid(), 5))
    monkeypatch.setattr(
        windows_lpac, "_grant_modify", lambda path, sid: granted.append((path, sid.value))
    )
    monkeypatch.setattr(
        windows_lpac, "_revoke_sid", lambda path, sid, **kw: revoked.append((path, sid.value))
    )
    return SimpleNamespace(
        manifests = manifests, temp_root = temp_root, work = work,
        granted = granted, revoked = revoked,
    )


def test_prepare_builds_the_token_and_the_job_before_it_returns(tmp_path, monkeypatch):
    recorder = _WinApiRecorder()
    host = _prepare_environment(tmp_path, monkeypatch, recorder)
    job = _FakeJob(recorder)
    monkeypatch.setattr(
        token_launcher, "_create_restricted_token", lambda identity: wintypes.HANDLE(4711)
    )
    monkeypatch.setattr(windows_lpac, "_job_object_with_limits", lambda: job)

    prepared = token_launcher.WindowsRestrictedTokenBackend().prepare(
        os_sandbox.ToolLaunchPlan(argv = ("x",), workdir = str(host.work), env = {})
    )

    identity = prepared.spawn_callback._launch_identity
    # Both SIDs, on both roots: the launch SID for the second access check and
    # the token's own user SID for the first. One without the other is a root
    # the child cannot use.
    assert host.granted == [
        (str(host.work), _LAUNCH_SID_VALUE),
        (identity.private_temp, _LAUNCH_SID_VALUE),
        (str(host.work), _USER_SID_VALUE),
        (identity.private_temp, _USER_SID_VALUE),
    ]
    assert Path(identity.manifest_path).exists()
    prepared.cleanup()
    assert prepared.cleanup_diagnostics == []
    # LIFO: the job dies and drains, then the ACEs and the private temp go.
    names = recorder.names()
    assert names.index("job.close") < recorder.calls.index(("LocalFree", 3))
    assert 4711 in recorder.closed
    assert host.revoked == [
        (identity.private_temp, _LAUNCH_SID_VALUE),
        (identity.private_temp, _USER_SID_VALUE),
        (str(host.work), _LAUNCH_SID_VALUE),
        (str(host.work), _USER_SID_VALUE),
    ]
    assert not Path(identity.manifest_path).exists()
    assert list(host.temp_root.iterdir()) == []


def test_prepare_declines_and_cleans_up_when_the_token_cannot_be_created(tmp_path, monkeypatch):
    recorder = _WinApiRecorder()
    host = _prepare_environment(tmp_path, monkeypatch, recorder)
    jobs: list[object] = []

    def refuse(identity):
        raise OSError(5, "CreateRestrictedToken denied")

    monkeypatch.setattr(token_launcher, "_create_restricted_token", refuse)
    monkeypatch.setattr(
        windows_lpac, "_job_object_with_limits", lambda: jobs.append(_FakeJob(recorder))
    )

    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "restricted token and job"):
        token_launcher.WindowsRestrictedTokenBackend().prepare(
            os_sandbox.ToolLaunchPlan(argv = ("x",), workdir = str(host.work), env = {})
        )

    # A token failure is a fallback, not a failed tool call, and it leaves no
    # grant, manifest, private temp or handle behind.
    # The grants were applied before the token failed, and all of them were undone.
    # The user SID is never reached: it is read off a token that does not exist.
    assert [sid for _path, sid in host.granted] == [_LAUNCH_SID_VALUE] * 2
    assert host.granted[0][0] == str(host.work)
    assert os.path.dirname(host.granted[1][0]) == str(host.temp_root)
    assert host.revoked == list(reversed(host.granted))
    assert list(host.manifests.iterdir()) == []
    assert list(host.temp_root.iterdir()) == []
    assert jobs == []  # the job is never created once the token failed
    # The launch SID is released once, after the ACLs built for it (8192 is the
    # fake ACL every window station and desktop edit allocates and frees).
    assert recorder.freed.count(3) == 1 and recorder.freed[-1] == 3


def test_the_launch_sid_is_granted_the_window_station_and_desktop_and_loses_them(
    tmp_path, monkeypatch
):
    """CreateProcessAsUser: "the DACLs for the window station and desktop must
    grant access to the user or the logon session represented by the hToken
    parameter". The launch SID is generated per launch, so it is on no DACL on
    the host until this grant, and under WRITE_RESTRICTED every write the child
    makes against those objects is checked against the restricting SIDs alone.
    """
    recorder = _WinApiRecorder()
    host = _prepare_environment(tmp_path, monkeypatch, recorder)
    monkeypatch.setattr(
        token_launcher, "_create_restricted_token", lambda identity: wintypes.HANDLE(4711)
    )
    monkeypatch.setattr(windows_lpac, "_job_object_with_limits", lambda: _FakeJob(recorder))

    prepared = token_launcher.WindowsRestrictedTokenBackend().prepare(
        os_sandbox.ToolLaunchPlan(argv = ("x",), workdir = str(host.work), env = {})
    )
    identity = prepared.spawn_callback._launch_identity

    assert identity.user_objects == ("window station", "desktop")
    assert identity.user_object_reason == ""
    # The child is given this process's own window station and desktop by name,
    # so the objects it connects to are the ones just granted.
    assert identity.desktop == "WinSta0\\Default"
    edits = [call for call in recorder.calls if call[0] == "SetEntriesInAclW"]
    assert [call[1:4] for call in edits] == [
        (
            token_launcher._WINSTA_GRANT,
            windows_lpac._GRANT_ACCESS,
            token_launcher._NO_PROPAGATE_INHERIT_ACE,
        ),
        (token_launcher._DESKTOP_GRANT, windows_lpac._GRANT_ACCESS, 0),
    ]
    # Never WRITE_DAC, WRITE_OWNER or DELETE: a sandboxed child that could
    # rewrite the window station DACL could grant itself anything.
    assert not (token_launcher._WINSTA_GRANT | token_launcher._DESKTOP_GRANT) & 0x000D0000
    assert [call[1] for call in recorder.calls if call[0] == "SetUserObjectSecurity"] == [101, 102]

    prepared.cleanup()
    assert prepared.cleanup_diagnostics == []
    revokes = [call for call in recorder.calls if call[0] == "SetEntriesInAclW"][2:]
    assert [call[2] for call in revokes] == [windows_lpac._REVOKE_ACCESS] * 2


def test_a_host_that_refuses_the_user_object_dacl_records_it_and_launches_anyway(
    tmp_path, monkeypatch
):
    """A refusal leaves the launch exactly where it was before this grant existed.

    The child then depends on the logon SID and Everyone ACEs the session
    happens to carry, which is what every host did until now; the reason is
    recorded so the live probe's failure text names it instead of guessing.
    """
    recorder = _WinApiRecorder()
    recorder.user_object_error = 5
    host = _prepare_environment(tmp_path, monkeypatch, recorder)
    monkeypatch.setattr(windows_lpac, "_winerror", lambda prefix, code = None: OSError(5, prefix))
    monkeypatch.setattr(
        token_launcher, "_create_restricted_token", lambda identity: wintypes.HANDLE(4711)
    )
    monkeypatch.setattr(windows_lpac, "_job_object_with_limits", lambda: _FakeJob(recorder))

    prepared = token_launcher.WindowsRestrictedTokenBackend().prepare(
        os_sandbox.ToolLaunchPlan(argv = ("x",), workdir = str(host.work), env = {})
    )
    identity = prepared.spawn_callback._launch_identity

    assert identity.user_objects == ()
    assert "window station" in identity.user_object_reason
    assert "SetUserObjectSecurity" not in recorder.names()
    # Nothing was granted, so cleanup has nothing to revoke and says nothing.
    prepared.cleanup()
    assert prepared.cleanup_diagnostics == []


def test_a_user_object_with_no_dacl_is_left_exactly_as_it_was(tmp_path, monkeypatch):
    """A NULL DACL allows everyone everything, so there is nothing to add.

    Writing a DACL that holds only this launch's ACE would take the interactive
    window station away from the user's own session, which is a far worse
    outcome than a launch that starts without the grant.
    """
    recorder = _WinApiRecorder()
    recorder.user_object_dacl = 0
    host = _prepare_environment(tmp_path, monkeypatch, recorder)
    monkeypatch.setattr(
        token_launcher, "_create_restricted_token", lambda identity: wintypes.HANDLE(4711)
    )
    monkeypatch.setattr(windows_lpac, "_job_object_with_limits", lambda: _FakeJob(recorder))

    prepared = token_launcher.WindowsRestrictedTokenBackend().prepare(
        os_sandbox.ToolLaunchPlan(argv = ("x",), workdir = str(host.work), env = {})
    )
    identity = prepared.spawn_callback._launch_identity

    assert identity.user_objects == ()
    assert identity.user_object_reason == "the desktop has no DACL, so it already grants every SID"
    assert "SetEntriesInAclW" not in recorder.names()
    assert "SetUserObjectSecurity" not in recorder.names()
    prepared.cleanup()
    assert prepared.cleanup_diagnostics == []


def _mutex_recorder(recorder: _WinApiRecorder, monkeypatch) -> None:
    """Give a recorder a host that answers CreateMutexW, and a fresh handle cache.

    The mutex is the ledger's same-session fast path and never the lock, so a
    recorder without this is a host that falls straight through to the lock file.
    """
    monkeypatch.setattr(windows_lpac, "_LEDGER_MUTEXES", {})
    recorder.kernel32.CreateMutexW = lambda _attributes, _owned, name: (
        recorder._note("CreateMutexW", name) and 900
    )
    recorder.kernel32.WaitForSingleObject = lambda handle, _ms: (
        recorder._note("WaitForSingleObject", handle) and windows_lpac._WAIT_OBJECT_0
    )
    recorder.kernel32.ReleaseMutex = lambda handle: recorder._note("ReleaseMutex", handle)


def test_two_launches_never_interleave_one_user_object_dacl_edit(monkeypatch):
    """The window station and desktop edits are a read-modify-write on objects
    this process neither owns nor has to itself.

    Unsynchronised, two launches each read the DACL before either writes, and
    the second write puts back a DACL that never carried the first launch's ACE:
    a child that is already running loses the window station it was given. A
    launch racing a cleanup resurrects a revoked ACE the same way, and that one
    outlives the launch, on the user's interactive window station.
    """
    recorder = _WinApiRecorder()
    monkeypatch.setattr(windows_lpac, "_api", lambda: recorder)
    monkeypatch.setattr(token_launcher, "_last_error", lambda: recorder.last_error)
    _mutex_recorder(recorder, monkeypatch)
    order: list[str] = []
    read_dacl = recorder.user32.GetUserObjectSecurity
    write_dacl = recorder.user32.SetUserObjectSecurity

    def slow_read(handle, information, buffer, length, needed):
        order.append(threading.current_thread().name)
        if buffer is not None:
            # The window the unsynchronised version loses: the descriptor is in
            # hand and the write has not happened yet.
            time.sleep(0.02)
        return read_dacl(handle, information, buffer, length, needed)

    def watched_write(handle, information, descriptor):
        order.append(threading.current_thread().name)
        return write_dacl(handle, information, descriptor)

    recorder.user32.GetUserObjectSecurity = slow_read
    recorder.user32.SetUserObjectSecurity = watched_write

    def edit() -> None:
        token_launcher._edit_user_object_dacl(
            recorder,
            101,
            ctypes.c_void_p(3),
            mode = windows_lpac._GRANT_ACCESS,
            access = token_launcher._WINSTA_GRANT,
            inheritance = token_launcher._NO_PROPAGATE_INHERIT_ACE,
        )

    threads = [threading.Thread(target = edit, name = name) for name in ("first", "second")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout = 10)
    assert not any(thread.is_alive() for thread in threads)

    # Two contiguous runs, one per launch: neither read the DACL the other was
    # about to replace.
    runs = [name for index, name in enumerate(order) if index == 0 or order[index - 1] != name]
    assert len(order) == 6, order
    assert runs in (["first", "second"], ["second", "first"]), order
    # And the same edit is ordered against a second Studio process, which the
    # in-process lock cannot reach. The session mutex is the fast path; the lock
    # file opened with no sharing is what actually says so.
    key = token_launcher._user_object_key(recorder, 101)
    assert ("CreateMutexW", windows_lpac._LEDGER_MUTEX_PREFIX + key) in recorder.calls
    assert recorder.ledger.locked(key)
    names = recorder.names()
    for index, name in enumerate(names):
        if name != "SetUserObjectSecurity":
            continue
        before = names[:index]
        assert before.count("WaitForSingleObject") == before.count("ReleaseMutex") + 1, names
    assert names.count("WaitForSingleObject") == names.count("ReleaseMutex") == 2


def test_the_desktop_grant_is_the_loader_pair_and_not_every_desktop_right():
    """A sandboxed child gets what user32 needs to initialise, and nothing else.

    Chromium names DESKTOP_WRITEOBJECTS | DESKTOP_READOBJECTS as the "Access
    required for UI thread to initialize (when user32.dll loads without win32k
    lockdown)", the system connects a process to its desktop with MAXIMUM_ALLOWED
    rather than a fixed mask, and under WRITE_RESTRICTED this ACE is consulted
    only for the write rights. So the rest of DESKTOP_ALL_ACCESS bought nothing
    and granted a great deal.
    """
    assert token_launcher._DESKTOP_GRANT == 0x0001 | 0x0080 | 0x00020000
    dangerous = {
        "DESKTOP_CREATEWINDOW": 0x0002,
        "DESKTOP_CREATEMENU": 0x0004,
        "DESKTOP_HOOKCONTROL": 0x0008,
        "DESKTOP_JOURNALRECORD": 0x0010,
        "DESKTOP_JOURNALPLAYBACK": 0x0020,
        "DESKTOP_SWITCHDESKTOP": 0x0100,
    }
    for name, right in dangerous.items():
        assert not token_launcher._DESKTOP_GRANT & right, name
    # Still never WRITE_DAC, WRITE_OWNER or DELETE, on either object.
    assert not (token_launcher._WINSTA_GRANT | token_launcher._DESKTOP_GRANT) & 0x000D0000
    assert not hasattr(token_launcher, "_DESKTOP_ALL_ACCESS")


def test_the_window_station_grant_is_only_what_the_second_check_can_decide():
    """The ACE names the launch SID, so most of WINSTA_ALL_ACCESS was inert.

    A random S-1-5-21 value is in TokenRestrictedSids and in none of the token's
    groups, so the first access check never consults this ACE; under
    WRITE_RESTRICTED the second consults it only for write access. The
    interactive window station's generic mapping puts exactly three object
    rights on the write side, so the other six could never decide anything
    whatever was written here.
    """
    write_side = {
        "WINSTA_ACCESSCLIPBOARD": 0x0004,
        "WINSTA_CREATEDESKTOP": 0x0008,
        "WINSTA_WRITEATTRIBUTES": 0x0010,
    }
    read_or_execute_side = {
        "WINSTA_ENUMDESKTOPS": 0x0001,
        "WINSTA_READATTRIBUTES": 0x0002,
        "WINSTA_ACCESSGLOBALATOMS": 0x0020,
        "WINSTA_EXITWINDOWS": 0x0040,
        "WINSTA_ENUMERATE": 0x0100,
        "WINSTA_READSCREEN": 0x0200,
    }
    for name, right in read_or_execute_side.items():
        assert not token_launcher._WINSTA_GRANT & right, name
    # The clipboard is the one live right that reaches the user rather than the
    # child: this is their real interactive window station. Chromium's own
    # WinSta0 grant omits it and its sandbox document names clipboard denial as
    # a property of a sandboxed target.
    assert not token_launcher._WINSTA_GRANT & write_side["WINSTA_ACCESSCLIPBOARD"]
    assert token_launcher._WINSTA_GRANT & write_side["WINSTA_CREATEDESKTOP"]
    assert token_launcher._WINSTA_GRANT & write_side["WINSTA_WRITEATTRIBUTES"]
    assert token_launcher._WINSTA_GRANT == 0x0008 | 0x0010 | 0x00020000
    # Still never WRITE_DAC, WRITE_OWNER or DELETE, on either user object.
    assert not (token_launcher._WINSTA_GRANT | token_launcher._DESKTOP_GRANT) & 0x000D0000
    # WINSTA_ALL_ACCESS is gone the way _DESKTOP_ALL_ACCESS went: naming it
    # invites granting it.
    assert not hasattr(token_launcher, "_WINSTA_ALL_ACCESS")


def _dead_owner_manifest(manifest: Path, text: str, **overrides) -> None:
    payload = {**json.loads(text), "owner_pid": 4242, "owner_created": 7, **overrides}
    manifest.write_text(json.dumps(payload), encoding = "utf-8")


def test_the_crash_manifest_records_and_revokes_the_user_object_grants(tmp_path, monkeypatch):
    """A crashed launch's window station and desktop ACEs are reconcilable.

    The manifest recorded the filesystem grants and not these, so a Studio that
    died between the grant and the revoke left the launch SID on the user's own
    interactive objects with nothing able to find it again. The record is
    write-ahead, so it names what the launch planned rather than what it
    managed; revoking an ACE that is not there is a no-op.
    """
    recorder = _WinApiRecorder()
    host = _prepare_environment(tmp_path, monkeypatch, recorder)
    monkeypatch.setattr(
        token_launcher, "_create_restricted_token", lambda identity: wintypes.HANDLE(4711)
    )
    monkeypatch.setattr(windows_lpac, "_job_object_with_limits", lambda: _FakeJob(recorder))
    recorder.advapi32.LookupAccountSidW = lambda *_arguments: 0

    prepared = token_launcher.WindowsRestrictedTokenBackend().prepare(
        os_sandbox.ToolLaunchPlan(argv = ("x",), workdir = str(host.work), env = {})
    )
    identity = prepared.spawn_callback._launch_identity
    manifest = Path(identity.manifest_path)
    recorded = manifest.read_text(encoding = "utf-8")
    payload = json.loads(recorded)
    assert payload["user_objects"] == ["window station", "desktop"]
    assert payload["desktop"] == "WinSta0\\Default"
    prepared.cleanup()
    assert prepared.cleanup_diagnostics == []

    # What the crash left: the same record, with an owner that is gone.
    monkeypatch.setattr(windows_lpac, "_process_identity", lambda pid = None: None)
    recorder.last_error = token_launcher._ERROR_NONE_MAPPED
    _dead_owner_manifest(manifest, recorded)
    recorder.calls.clear()

    token_launcher.WindowsRestrictedTokenBackend().reconcile_stale_manifests()

    revokes = [
        call for call in recorder.calls
        if call[0] == "SetEntriesInAclW" and call[2] == windows_lpac._REVOKE_ACCESS
    ]
    assert len(revokes) == 2
    assert [call[1] for call in recorder.calls if call[0] == "SetUserObjectSecurity"] == [101, 102]
    assert not manifest.exists()

    # A record from another session names objects this process is not connected
    # to. Its SID names no account and is never reused, so it is left for the
    # Studio that is on that desktop rather than revoked from the wrong one.
    _dead_owner_manifest(manifest, recorded, desktop = "OtherStation\\Other")
    recorder.last_error = token_launcher._ERROR_NONE_MAPPED
    recorder.calls.clear()

    token_launcher.WindowsRestrictedTokenBackend().reconcile_stale_manifests()

    assert "SetUserObjectSecurity" not in recorder.names()
    assert not manifest.exists()


def test_a_manifest_may_only_name_the_user_objects_this_launcher_grants(tmp_path):
    sid = "S-1-5-21-1-2-3-4"
    assert token_launcher._parse_manifest(_manifest(tmp_path, sid))["user_objects"] == []
    assert token_launcher._parse_manifest(
        _manifest(tmp_path, sid, user_objects = ["desktop"], desktop = "WinSta0\\Default")
    )["desktop"] == "WinSta0\\Default"
    # The reconciler drives a DACL edit on this process's own objects from these
    # two, so nothing else may appear in either.
    for planted in (["clipboard"], ["window station", "screen"], "desktop", [1]):
        assert token_launcher._parse_manifest(
            _manifest(tmp_path, sid, user_objects = planted)
        ) is None, planted
    assert token_launcher._parse_manifest(_manifest(tmp_path, sid, desktop = ["a"])) is None


# ── the first access check (staging round 18) ────────────────────────────────


def test_the_user_sid_is_read_off_the_token_that_was_built(monkeypatch):
    """TOKEN_USER is what the first access check is decided against, so it is queried.

    Never the launcher's own process token and never a well-known SID: the SID
    that has to appear on the workdir is whatever the token that will run the
    child actually carries.
    """
    entry = token_launcher._SID_AND_ATTRIBUTES(0xBEEF, 0)
    buffer = ctypes.create_string_buffer(
        ctypes.string_at(ctypes.byref(entry), ctypes.sizeof(entry)), ctypes.sizeof(entry)
    )
    queried: list[tuple] = []
    monkeypatch.setattr(
        token_launcher, "_token_information",
        lambda api, token, kind: queried.append((token, kind)) or buffer,
    )
    monkeypatch.setattr(windows_lpac, "_sid_string", lambda api, sid: f"S-1-5-21-{sid.value:x}")

    assert token_launcher._token_user_sid_text(SimpleNamespace(), 4711) == "S-1-5-21-beef"
    # TokenUser is 1, and the handle is the one passed in.
    assert queried == [(4711, token_launcher._TOKEN_USER)]
    assert token_launcher._TOKEN_USER == 1

    empty = token_launcher._SID_AND_ATTRIBUTES(0, 0)
    monkeypatch.setattr(
        token_launcher, "_token_information",
        lambda api, token, kind: ctypes.create_string_buffer(
            ctypes.string_at(ctypes.byref(empty), ctypes.sizeof(empty)), ctypes.sizeof(empty)
        ),
    )
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "user SID"):
        token_launcher._token_user_sid_text(SimpleNamespace(), 4711)


def test_the_user_sid_ace_is_added_after_the_token_and_recorded_before_it(tmp_path, monkeypatch):
    """Staging round 18: the child could not read or write the roots it was granted.

    A restricted token is judged twice and both checks must allow. The per-launch
    SID lives in TokenRestrictedSids alone, so the first check never sees it, and
    an ACE naming only it leaves both the reads and the writes of the workdir to
    whatever the directory inherited. Each root therefore gets a second ACE for
    the token's own user SID, which only exists once the token does, and which is
    recorded in the write-ahead manifest before it is applied.
    """
    recorder = _WinApiRecorder()
    host = _prepare_environment(tmp_path, monkeypatch, recorder)
    order: list[str] = []
    monkeypatch.setattr(
        token_launcher, "_create_restricted_token",
        lambda identity: order.append("token") or wintypes.HANDLE(4711),
    )
    monkeypatch.setattr(windows_lpac, "_job_object_with_limits", lambda: _FakeJob(recorder))
    recorded: list[dict] = []
    write_manifest = token_launcher._write_manifest
    grant_modify = windows_lpac._grant_modify

    def record(identity):
        write_manifest(identity)
        recorded.append(json.loads(Path(identity.manifest_path).read_text(encoding = "utf-8")))
        order.append("manifest")

    monkeypatch.setattr(token_launcher, "_write_manifest", record)
    monkeypatch.setattr(
        windows_lpac, "_grant_modify",
        lambda path, sid: grant_modify(path, sid) or order.append(f"grant {sid.value}"),
    )

    prepared = token_launcher.WindowsRestrictedTokenBackend().prepare(
        os_sandbox.ToolLaunchPlan(argv = ("x",), workdir = str(host.work), env = {})
    )
    identity = prepared.spawn_callback._launch_identity

    assert order == [
        "manifest", f"grant {_LAUNCH_SID_VALUE}", f"grant {_LAUNCH_SID_VALUE}",
        "token",
        "manifest", f"grant {_USER_SID_VALUE}", f"grant {_USER_SID_VALUE}",
    ]
    # The first record cannot name a SID that did not exist yet; the second is
    # written before the ACE it describes, so a crash between the two is
    # reconcilable and never the other way round.
    assert recorded[0]["user_sid"] == "" and recorded[0]["user_sid_roots"] == []
    assert recorded[1]["user_sid"] == _TOKEN_USER_SID
    assert recorded[1]["user_sid_roots"] == [str(host.work), identity.private_temp]
    assert identity.user_sid_string == _TOKEN_USER_SID
    assert identity.user_sid_roots == (str(host.work), identity.private_temp)

    prepared.cleanup()
    assert prepared.cleanup_diagnostics == []
    # Every ACE this launch added, and both SIDs on both roots.
    assert host.revoked == [
        (identity.private_temp, _LAUNCH_SID_VALUE),
        (identity.private_temp, _USER_SID_VALUE),
        (str(host.work), _LAUNCH_SID_VALUE),
        (str(host.work), _USER_SID_VALUE),
    ]
    assert not Path(identity.manifest_path).exists()


def test_a_root_that_already_names_the_user_sid_keeps_its_dacl(tmp_path, monkeypatch):
    """An ACE that was there before the launch is neither added nor taken away.

    SetEntriesInAclW(REVOKE_ACCESS) drops every ACE for a trustee, an inherited
    one included, so a cleanup that revoked the user's own SID unconditionally
    would hand back a session workdir the user no longer reaches. The roots that
    already grant it are left byte for byte as they were.
    """
    recorder = _WinApiRecorder()
    host = _prepare_environment(
        tmp_path, monkeypatch, recorder, already_granted = (tmp_path / "work",)
    )
    monkeypatch.setattr(
        token_launcher, "_create_restricted_token", lambda identity: wintypes.HANDLE(4711)
    )
    monkeypatch.setattr(windows_lpac, "_job_object_with_limits", lambda: _FakeJob(recorder))

    prepared = token_launcher.WindowsRestrictedTokenBackend().prepare(
        os_sandbox.ToolLaunchPlan(argv = ("x",), workdir = str(host.work), env = {})
    )
    identity = prepared.spawn_callback._launch_identity

    assert identity.user_sid_roots == (identity.private_temp,)
    assert (str(host.work), _USER_SID_VALUE) not in host.granted
    assert (identity.private_temp, _USER_SID_VALUE) in host.granted
    payload = json.loads(Path(identity.manifest_path).read_text(encoding = "utf-8"))
    assert payload["user_sid_roots"] == [identity.private_temp]

    prepared.cleanup()
    assert prepared.cleanup_diagnostics == []
    assert (str(host.work), _USER_SID_VALUE) not in host.revoked
    assert (identity.private_temp, _USER_SID_VALUE) in host.revoked
    # The launch SID is still revoked from both: it names nobody and this launch
    # is the only thing that ever put it anywhere.
    assert (str(host.work), _LAUNCH_SID_VALUE) in host.revoked


def test_a_root_named_but_not_granted_is_widened_and_never_revoked(tmp_path, monkeypatch):
    """The one case where "needs an ACE" and "may lose one" disagree.

    A root that already carries an ACE for this account, but not one that covers
    what the child needs, gets the grant (the child has to be able to work there)
    and keeps it (a revoke would take the account's own ACE with it). That widens
    nothing the account did not already hold on a directory of its own.
    """
    recorder = _WinApiRecorder()
    host = _prepare_environment(
        tmp_path, monkeypatch, recorder, already_granted = (), already_named = (tmp_path / "work",)
    )
    monkeypatch.setattr(
        token_launcher, "_create_restricted_token", lambda identity: wintypes.HANDLE(4711)
    )
    monkeypatch.setattr(windows_lpac, "_job_object_with_limits", lambda: _FakeJob(recorder))

    prepared = token_launcher.WindowsRestrictedTokenBackend().prepare(
        os_sandbox.ToolLaunchPlan(argv = ("x",), workdir = str(host.work), env = {})
    )
    identity = prepared.spawn_callback._launch_identity

    assert (str(host.work), _USER_SID_VALUE) in host.granted
    assert identity.user_sid_roots == (identity.private_temp,)
    prepared.cleanup()
    assert (str(host.work), _USER_SID_VALUE) not in host.revoked


def test_the_grant_plan_answers_needed_and_revocable_separately(monkeypatch):
    """_existing_access decides the grant, _dacl_names_sid decides the revoke.

    The grant question has to be the access check's own, so a deny ACE, an
    inherit-only ACE or a read-only ACE all still get the grant. The revoke
    question has to be the weaker one, because SetEntriesInAclW(REVOKE_ACCESS)
    takes every ACE for the trustee, and this trustee is a real account.
    """
    masks: list[int] = []
    monkeypatch.setattr(
        windows_lpac, "_existing_access",
        lambda path, sids, required: masks.append(required) or "modify" in path,
    )
    # Anything the access check accepts is necessarily on the DACL, so the
    # weaker question is true wherever the stronger one is.
    monkeypatch.setattr(
        token_launcher, "_dacl_names_sid", lambda path, sid: "named" in path or "modify" in path
    )
    sid = ctypes.c_void_p(4)

    # Nothing there: granted, and cleanup takes it back.
    assert token_launcher._root_grant_plan("C:\\fresh", sid) == (True, True)
    # Already modify: left exactly as it was, and never revoked.
    assert token_launcher._root_grant_plan("C:\\modify-named", sid) == (False, False)
    # Named but not sufficient (read-only, inherit-only, or denied): granted,
    # and never taken back.
    assert token_launcher._root_grant_plan("C:\\named", sid) == (True, False)

    # The rights asked about are exactly the ones _grant_modify leaves behind,
    # in the specific form a stored ACE carries them: Explorer's "Modify".
    assert set(masks) == {token_launcher._MODIFY_ACCESS}
    assert token_launcher._MODIFY_ACCESS == 0x1301BF
    assert not token_launcher._MODIFY_ACCESS & 0x00040000  # never WRITE_DAC
    assert not token_launcher._MODIFY_ACCESS & 0x00080000  # never WRITE_OWNER


def test_dacl_names_sid_asks_the_path_and_releases_its_descriptor(monkeypatch):
    seen: list[tuple] = []
    freed: list[int] = []

    def get_named(path, kind, information, owner, group, dacl, sacl, descriptor):
        seen.append((path, kind, information))
        dacl._obj.value = 8192
        descriptor._obj.value = 4242
        return 0

    api = SimpleNamespace(
        advapi32 = SimpleNamespace(GetNamedSecurityInfoW = get_named),
        kernel32 = SimpleNamespace(LocalFree = lambda value: freed.append(value.value)),
    )
    monkeypatch.setattr(windows_lpac, "_api", lambda: api)
    monkeypatch.setattr(
        windows_lpac, "_acl_contains_sid", lambda acl, sid: (acl.value, sid.value) == (8192, 4)
    )

    assert token_launcher._dacl_names_sid("C:\\w", ctypes.c_void_p(4)) is True
    assert token_launcher._dacl_names_sid("C:\\w", ctypes.c_void_p(9)) is False
    assert seen[0] == (
        "C:\\w", windows_lpac._SE_FILE_OBJECT, token_launcher._DACL_SECURITY_INFORMATION
    )
    assert freed == [4242, 4242]

    # A descriptor that cannot be read is a failure, never a silent "no ACE":
    # answering False there would add an ACE the cleanup then cannot account for.
    monkeypatch.setattr(
        api.advapi32, "GetNamedSecurityInfoW",
        lambda *arguments: 5,
    )
    monkeypatch.setattr(windows_lpac, "_winerror", lambda prefix, code = None: OSError(5, prefix))
    with pytest.raises(OSError, match = "GetNamedSecurityInfoW"):
        token_launcher._dacl_names_sid("C:\\w", ctypes.c_void_p(4))


def test_a_manifest_may_only_point_a_user_sid_revoke_at_its_own_roots(tmp_path):
    """The recorded user SID names a real account, so where it may be revoked is bounded.

    granted_roots is already pinned to this launch's own workdir and private
    temp; the user-SID roots are a subset of those, so a planted manifest cannot
    aim a revoke of somebody's account at a directory of theirs.
    """
    sid = "S-1-5-21-1-2-3-4"
    workdir = str(tmp_path / "work")
    private_temp = str(tmp_path / "temp" / ("a" * 24))
    # Absent in a record written before this grant existed: reconciled for its
    # launch SID exactly as it was.
    older = token_launcher._parse_manifest(_manifest(tmp_path, sid))
    assert older["user_sid"] == "" and older["user_sid_roots"] == []
    good = token_launcher._parse_manifest(
        _manifest(tmp_path, sid, user_sid = "S-1-5-21-9-9-9-1001", user_sid_roots = [workdir])
    )
    assert good["user_sid_roots"] == [workdir]
    for planted in (
        [str(tmp_path / "documents")],
        [workdir, str(tmp_path / "documents")],
        ["relative"],
        [1],
        "workdir",
    ):
        assert token_launcher._parse_manifest(
            _manifest(tmp_path, sid, user_sid = "S-1-5-21-9-9-9-1001", user_sid_roots = planted)
        ) is None, planted
    # Roots without a SID to revoke, or a SID that is not one, name nothing.
    assert token_launcher._parse_manifest(
        _manifest(tmp_path, sid, user_sid = "", user_sid_roots = [workdir])
    ) is None
    assert token_launcher._parse_manifest(
        _manifest(tmp_path, sid, user_sid = "runneradmin", user_sid_roots = [private_temp])
    ) is None
    planted_sid = _manifest(tmp_path, sid, user_sid = ["S-1-5-18"])
    assert token_launcher._parse_manifest(planted_sid) is None


def test_reconcile_revokes_the_user_sid_only_for_this_studios_own_account(tmp_path, monkeypatch):
    """A crashed launch's user-SID ACEs are reconcilable, and only by the account they name.

    The launch SID is safe to revoke out of a record because it names nobody; a
    real account's SID is not, so the record is acted on only where it matches
    the account this Studio is running as, which is the only one whose ACEs this
    launcher can have made.
    """
    recorder = _WinApiRecorder()
    host = _prepare_environment(tmp_path, monkeypatch, recorder)
    monkeypatch.setattr(
        token_launcher, "_create_restricted_token", lambda identity: wintypes.HANDLE(4711)
    )
    monkeypatch.setattr(windows_lpac, "_job_object_with_limits", lambda: _FakeJob(recorder))
    recorder.advapi32.LookupAccountSidW = lambda *_arguments: 0

    prepared = token_launcher.WindowsRestrictedTokenBackend().prepare(
        os_sandbox.ToolLaunchPlan(argv = ("x",), workdir = str(host.work), env = {})
    )
    identity = prepared.spawn_callback._launch_identity
    manifest = Path(identity.manifest_path)
    recorded = manifest.read_text(encoding = "utf-8")
    prepared.cleanup()
    host.revoked.clear()

    monkeypatch.setattr(windows_lpac, "_process_identity", lambda pid = None: None)
    recorder.last_error = token_launcher._ERROR_NONE_MAPPED
    _dead_owner_manifest(manifest, recorded)

    token_launcher.WindowsRestrictedTokenBackend().reconcile_stale_manifests()

    assert host.revoked == [
        (identity.private_temp, _LAUNCH_SID_VALUE),
        (identity.private_temp, _USER_SID_VALUE),
        (str(host.work), _LAUNCH_SID_VALUE),
        (str(host.work), _USER_SID_VALUE),
    ]
    assert not manifest.exists()

    # The same record under another account: its launch SID still goes, its
    # user SID is left for the Studio that could have created it.
    host.revoked.clear()
    monkeypatch.setattr(token_launcher, "_process_user_sid_text", lambda: "S-1-5-21-7-7-7-500")
    recorder.last_error = token_launcher._ERROR_NONE_MAPPED
    _dead_owner_manifest(manifest, recorded)

    token_launcher.WindowsRestrictedTokenBackend().reconcile_stale_manifests()

    assert host.revoked == [
        (identity.private_temp, _LAUNCH_SID_VALUE),
        (str(host.work), _LAUNCH_SID_VALUE),
    ]
    assert not manifest.exists()


def test_two_launches_never_interleave_one_root_dacl_edit(monkeypatch):
    """The workdir DACL is the same read-modify-write the window station was.

    Unsynchronised, two Limited launches in one chat, a Python tool and a
    Terminal tool, each read the workdir DACL before either writes, and the
    second write puts back a DACL that never carried the first launch's ACE: a
    child that is already running is refused the workdir it was given. A launch
    racing a cleanup resurrects a revoked ACE the same way. Two ACEs per root
    now, so twice as many edits race.
    """
    recorder = _WinApiRecorder()
    monkeypatch.setattr(windows_lpac, "_api", lambda: recorder)
    _mutex_recorder(recorder, monkeypatch)
    order: list[str] = []
    root = os.path.join(os.sep + "Work", "session")

    def edit() -> None:
        with token_launcher._root_acl_edit(root):
            # The window an unsynchronised read-modify-write loses: the DACL is
            # in hand and the write has not happened yet.
            order.append(threading.current_thread().name)
            time.sleep(0.02)
            order.append(threading.current_thread().name)

    threads = [threading.Thread(target = edit, name = name) for name in ("first", "second")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout = 10)
    assert not any(thread.is_alive() for thread in threads)

    runs = [name for index, name in enumerate(order) if index == 0 or order[index - 1] != name]
    assert len(order) == 4, order
    assert runs in (["first", "second"], ["second", "first"]), order
    # And ordered against a second Studio process, which the in-process lock
    # cannot reach, under a key derived from the root itself. The session mutex
    # is only the fast path in front of it; the lock file is the lock.
    key = token_launcher._root_acl_key(root)
    assert ("CreateMutexW", windows_lpac._LEDGER_MUTEX_PREFIX + key) in recorder.calls
    assert recorder.names().count("WaitForSingleObject") == 2
    assert recorder.names().count("ReleaseMutex") == 2
    assert recorder.ledger.attempts.count((key + windows_lpac._LEDGER_LOCK_SUFFIX, 0)) == 2
    # A ledger key becomes a file name beside the manifests, so it is a digest.
    assert key.startswith(token_launcher._ROOT_ACL_LEDGER_PREFIX)
    assert "\\" not in key and "/" not in key and len(key) < 64
    # Per root: a launch editing its private temp is not held up by a workdir.
    assert key != token_launcher._root_acl_key(os.path.join(os.sep + "Work", "other"))
    assert key == token_launcher._root_acl_key(root)


@pytest.mark.skipif(sys.platform != "win32", reason = "normcase folds case only on Windows")
def test_the_root_acl_mutex_name_folds_case_like_the_filesystem():
    """Two Studios have to derive one name from one directory, spelled either way."""
    assert token_launcher._root_acl_key("C:\\Work\\Session") == (
        token_launcher._root_acl_key("c:\\work\\session")
    )


def test_every_root_dacl_edit_happens_under_that_roots_lock(tmp_path, monkeypatch):
    """Not just the new ACE: the launch SID's grants and both revokes as well."""
    recorder = _WinApiRecorder()
    host = _prepare_environment(tmp_path, monkeypatch, recorder)
    monkeypatch.setattr(
        token_launcher, "_create_restricted_token", lambda identity: wintypes.HANDLE(4711)
    )
    monkeypatch.setattr(windows_lpac, "_job_object_with_limits", lambda: _FakeJob(recorder))
    stack: list[str] = []
    edits: list[tuple[str, str | None]] = []
    real_edit = token_launcher._root_acl_edit
    grant = windows_lpac._grant_modify
    revoke = windows_lpac._revoke_sid

    @contextmanager
    def watched(path, **kwargs):
        stack.append((path, kwargs.get("destructive", False)))
        try:
            with real_edit(path, **kwargs):
                yield
        finally:
            stack.pop()

    monkeypatch.setattr(token_launcher, "_root_acl_edit", watched)
    monkeypatch.setattr(
        windows_lpac, "_grant_modify",
        lambda path, sid: edits.append((path, stack[-1] if stack else None)) or grant(path, sid),
    )
    monkeypatch.setattr(
        windows_lpac, "_revoke_sid",
        lambda path, sid, **kw: (
            edits.append((path, stack[-1] if stack else None)) or revoke(path, sid, **kw)
        ),
    )

    prepared = token_launcher.WindowsRestrictedTokenBackend().prepare(
        os_sandbox.ToolLaunchPlan(argv = ("x",), workdir = str(host.work), env = {})
    )
    prepared.cleanup()
    assert prepared.cleanup_diagnostics == []

    # Two SIDs on two roots, granted and revoked: eight edits, each one holding
    # the ledger for the root it is editing.
    assert len(edits) == 8, edits
    assert all(held == path for path, (held, _destructive) in edits), edits
    # And every revoke declared itself destructive, which is what makes a busy
    # ledger skip it rather than run it unsynchronised. The four grants did not:
    # a grant takes nothing away, so a busy ledger must not fail a tool call.
    assert [destructive for _path, (_held, destructive) in edits] == [False] * 4 + [True] * 4


def _launch_under_a_busy_ledger(tmp_path, monkeypatch):
    """One prepared launch whose workdir key another Studio process holds."""
    recorder = _WinApiRecorder()
    host = _prepare_environment(tmp_path, monkeypatch, recorder)
    monkeypatch.setattr(
        token_launcher, "_create_restricted_token", lambda identity: wintypes.HANDLE(4711)
    )
    monkeypatch.setattr(windows_lpac, "_job_object_with_limits", lambda: _FakeJob(recorder))
    recorder.ledger.hold(token_launcher._root_acl_key(str(host.work)))
    prepared = token_launcher.WindowsRestrictedTokenBackend().prepare(
        os_sandbox.ToolLaunchPlan(argv = ("x",), workdir = str(host.work), env = {})
    )
    return recorder, host, prepared


def test_a_busy_ledger_skips_the_root_revoke_and_keeps_the_whole_record(tmp_path, monkeypatch):
    """A revoke without the ledger is the damage the ledger exists to prevent.

    The session mutex this used to hold proceeds when it is not acquired, so a
    second Studio holding the workdir was no obstacle at all: the revoke ran
    against a DACL that process was in the middle of rewriting, and the two SIDs
    it takes are not equally cheap to get wrong. The launch SID names one launch,
    but the token's own user SID is the account, shared by every concurrent
    launch of it, and a revoke of that one lands on a sibling launch's running
    child. Neither may run unsynchronised; both are left for reconciliation.

    The grant is the other half. It takes nothing away, so it proceeds under the
    same busy ledger with a diagnostic rather than failing the tool call.
    """
    recorder, host, prepared = _launch_under_a_busy_ledger(tmp_path, monkeypatch)
    identity = prepared.spawn_callback._launch_identity

    # The launch was not refused: all four grants were made, on the root whose
    # ledger is busy as much as on the one whose is free.
    assert host.granted == [
        (str(host.work), _LAUNCH_SID_VALUE),
        (identity.private_temp, _LAUNCH_SID_VALUE),
        (str(host.work), _USER_SID_VALUE),
        (identity.private_temp, _USER_SID_VALUE),
    ]
    assert identity.user_sid_roots == (str(host.work), identity.private_temp)

    prepared.cleanup()

    # The private temp's ledger was free, so its ACEs went. The workdir's was
    # not, so neither of its ACEs was touched - including the account's, which
    # another launch may be relying on this very second.
    assert host.revoked == [
        (identity.private_temp, _LAUNCH_SID_VALUE),
        (identity.private_temp, _USER_SID_VALUE),
    ]
    # And the record survives whole, because the manifest is what reconciliation
    # reads: the ACEs it names are still on the workdir.
    assert Path(identity.manifest_path).exists()
    assert identity.cleaned is False
    assert any("could not be taken" in text for text in prepared.cleanup_diagnostics), (
        prepared.cleanup_diagnostics
    )
    assert token_launcher._DEFERRED_CLEANUPS == [identity]


def test_a_refused_cleanup_is_retried_once_the_other_studio_lets_go(tmp_path, monkeypatch):
    """A skipped revoke has to be recoverable, and inside this process too.

    reconcile_stale_manifests deliberately leaves a manifest whose owning process
    is still alive alone, and while Studio runs it is that process, so the record
    is reachable only through the queue the refusal put it on. Without that queue
    a launch SID would sit on the user's own workdir until Studio exited.
    """
    recorder, host, prepared = _launch_under_a_busy_ledger(tmp_path, monkeypatch)
    identity = prepared.spawn_callback._launch_identity
    prepared.cleanup()
    assert host.revoked == [
        (identity.private_temp, _LAUNCH_SID_VALUE),
        (identity.private_temp, _USER_SID_VALUE),
    ]

    recorder.ledger.busy.clear()  # the other Studio finished
    token_launcher.WindowsRestrictedTokenBackend().reconcile_stale_manifests()

    assert (str(host.work), _LAUNCH_SID_VALUE) in host.revoked
    assert (str(host.work), _USER_SID_VALUE) in host.revoked
    assert not Path(identity.manifest_path).exists()
    assert identity.cleaned is True
    assert token_launcher._DEFERRED_CLEANUPS == []


def test_a_busy_ledger_skips_the_user_object_revoke_and_leaves_it_recorded(tmp_path, monkeypatch):
    """The one ACE that outlives its launch on the user's real window station.

    The window station and the desktop are separate keys, so a ledger that
    refuses one says nothing about the other and both are attempted. What must
    not happen is the revoke running while another launch is granting: this is
    the user's own interactive session object, and the loser of that race is a
    resurrected ACE nobody is left to remove.
    """
    recorder = _WinApiRecorder()
    host = _prepare_environment(tmp_path, monkeypatch, recorder)
    monkeypatch.setattr(
        token_launcher, "_create_restricted_token", lambda identity: wintypes.HANDLE(4711)
    )
    monkeypatch.setattr(windows_lpac, "_job_object_with_limits", lambda: _FakeJob(recorder))
    prepared = token_launcher.WindowsRestrictedTokenBackend().prepare(
        os_sandbox.ToolLaunchPlan(argv = ("x",), workdir = str(host.work), env = {})
    )
    identity = prepared.spawn_callback._launch_identity
    assert identity.user_objects == ("window station", "desktop")
    writes = recorder.names().count("SetUserObjectSecurity")

    recorder.ledger.hold(token_launcher._user_object_key(recorder, 101))  # the window station
    prepared.cleanup()

    # The desktop's key was free and its ACE went; the window station's was not.
    assert recorder.names().count("SetUserObjectSecurity") == writes + 1
    assert identity.user_objects == ("window station", "desktop")
    assert Path(identity.manifest_path).exists()
    assert any("window station" in text for text in prepared.cleanup_diagnostics), (
        prepared.cleanup_diagnostics
    )
    assert token_launcher._DEFERRED_CLEANUPS == [identity]


def test_a_non_interactive_window_station_still_names_a_ledger_lock_file(ledger_state):
    """``Service-0x0-3e7$`` is a window station name and not a profile name.

    The ledger turns a key into a file beside the manifests and accepts only what
    CreateAppContainerProfile accepts, so a key carrying a dollar sign is not a
    slower lock, it is no lock: the open is refused, and under the fail-closed
    rule every revoke on that host would be deferred for as long as it ran. The
    variable half is a digest for exactly that reason.
    """
    recorder = _WinApiRecorder()
    recorder.user_object_names = {101: "Service-0x0-3e7$", 102: "Default"}
    key = token_launcher._user_object_key(recorder, 101)
    path = windows_lpac._ledger_lock_path(key)
    assert os.path.dirname(path) == str(ledger_state)
    assert os.path.basename(path) == key + windows_lpac._LEDGER_LOCK_SUFFIX
    # Still per object, and still the same key from either spelling of one name.
    assert key != token_launcher._user_object_key(recorder, 102)
    recorder.user_object_names[101] = "service-0x0-3e7$"
    assert key == token_launcher._user_object_key(recorder, 101)
    # The same holds of a root, whose path carries separators and has no length
    # bound at all.
    long_root = os.path.join(os.sep + "Work", "x" * 300)
    windows_lpac._ledger_lock_path(token_launcher._root_acl_key(long_root))


def test_the_two_ledgers_are_only_ever_taken_in_one_order(monkeypatch):
    """Both launchers can be live in one Studio, so the order has to be stated.

    This module takes its own process-local lock first, then goes through
    windows_lpac._installation_ledger, which takes _SHARED_GRANTS_LOCK and only
    then the session mutex and the lock file. windows_lpac takes the last three
    and never the first, so neither launcher can wait on the other backwards.
    """
    recorder = _WinApiRecorder()
    monkeypatch.setattr(windows_lpac, "_api", lambda: recorder)
    root = os.path.join(os.sep + "Work", "ordered")
    inside = threading.Event()

    def edit() -> None:
        with token_launcher._root_acl_edit(root):
            inside.set()

    # Holding what windows_lpac orders its own grants with is enough to hold a
    # Limited DACL edit out, which is what makes the two launchers one order.
    with windows_lpac._SHARED_GRANTS_LOCK:
        waiting = threading.Thread(target = edit)
        waiting.start()
        assert not inside.wait(0.5)
    waiting.join(timeout = 10)
    assert inside.is_set()

    # And it is taken second: a thread stopped on this module's own lock has not
    # taken windows_lpac's, so no cycle exists between them.
    inside.clear()
    with token_launcher._ROOT_ACL_LOCK:
        blocked = threading.Thread(target = edit)
        blocked.start()
        assert not inside.wait(0.2)
        assert windows_lpac._SHARED_GRANTS_LOCK.acquire(timeout = 1.0)
        windows_lpac._SHARED_GRANTS_LOCK.release()
    blocked.join(timeout = 10)
    assert inside.is_set()


def test_a_second_live_launch_keeps_the_shared_user_sid_ace(tmp_path, monkeypatch):
    """The launch SID names one launch; the account SID names every launch.

    So the per-launch revoke that is right for the first is wrong for the
    second: it would pull the workdir out from under a child that is still
    running. The write-ahead manifests are the register of who holds what, so
    that is what the revoke asks.
    """
    recorder = _WinApiRecorder()
    host = _prepare_environment(tmp_path, monkeypatch, recorder)
    monkeypatch.setattr(
        token_launcher, "_create_restricted_token", lambda identity: wintypes.HANDLE(4711)
    )
    monkeypatch.setattr(windows_lpac, "_job_object_with_limits", lambda: _FakeJob(recorder))
    backend = token_launcher.WindowsRestrictedTokenBackend()
    plan = os_sandbox.ToolLaunchPlan(argv = ("x",), workdir = str(host.work), env = {})

    first = backend.prepare(plan)
    second = backend.prepare(plan)
    first_temp = first.spawn_callback._launch_identity.private_temp
    host.revoked.clear()

    first.cleanup()
    assert first.cleanup_diagnostics == []
    # This launch's own SID always goes, and so does the account SID on the temp
    # directory no other launch shares.
    assert (str(host.work), _LAUNCH_SID_VALUE) in host.revoked
    assert (first_temp, _USER_SID_VALUE) in host.revoked
    # The shared one on the shared workdir stays while the sibling is alive.
    assert (str(host.work), _USER_SID_VALUE) not in host.revoked

    host.revoked.clear()
    second.cleanup()
    assert second.cleanup_diagnostics == []
    assert (str(host.work), _USER_SID_VALUE) in host.revoked


def test_a_dead_launchs_record_is_litter_and_never_a_claim(tmp_path, monkeypatch):
    recorder = _WinApiRecorder()
    host = _prepare_environment(tmp_path, monkeypatch, recorder)
    monkeypatch.setattr(
        token_launcher, "_create_restricted_token", lambda identity: wintypes.HANDLE(4711)
    )
    monkeypatch.setattr(windows_lpac, "_job_object_with_limits", lambda: _FakeJob(recorder))
    backend = token_launcher.WindowsRestrictedTokenBackend()
    plan = os_sandbox.ToolLaunchPlan(argv = ("x",), workdir = str(host.work), env = {})
    first = backend.prepare(plan)
    second = backend.prepare(plan)
    second_manifest = Path(second.spawn_callback._launch_identity.manifest_path)
    _dead_owner_manifest(second_manifest, second_manifest.read_text(encoding = "utf-8"))
    host.revoked.clear()

    first.cleanup()

    # The sibling's owner is gone, so its record holds nothing; the ACE goes and
    # reconcile_stale_manifests deals with what it left behind.
    assert (str(host.work), _USER_SID_VALUE) in host.revoked


def test_the_process_user_sid_guard_refuses_rather_than_raises(monkeypatch):
    recorder = _WinApiRecorder()
    monkeypatch.setattr(windows_lpac, "_api", lambda: recorder)
    monkeypatch.setattr(
        token_launcher, "_token_user_sid_text", lambda api, token: "S-1-5-21-1-1-1-5"
    )
    assert token_launcher._process_user_sid_text() == "S-1-5-21-1-1-1-5"
    assert 4321 in recorder.closed  # the token handle is released either way

    def explode(api, token):
        raise OSError(5, "GetTokenInformation(1)")

    monkeypatch.setattr(token_launcher, "_token_user_sid_text", explode)
    assert token_launcher._process_user_sid_text() == ""
    recorder.advapi32.OpenProcessToken = lambda process, access, out: 0
    assert token_launcher._process_user_sid_text() == ""


def test_spawn_restricted_detaches_the_child_and_names_its_desktop(tmp_path, monkeypatch):
    """Chromium's broker creates every sandboxed target with exactly
    CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT | DETACHED_PROCESS. A detached
    child has no console allocated for it while it is starting under the
    restricted token, and every stdio handle here is redirected already.
    """
    recorder = _WinApiRecorder()
    prepared, resources = _spawn_with_fake_api(tmp_path, monkeypatch, recorder, _FakeJob(recorder))
    identity = _launch_identity(tmp_path)
    identity.desktop = "WinSta0\\Default"

    token_launcher._spawn_restricted(prepared, dict(_STDIO_PLAN), identity, resources)

    flags = recorder.call("CreateProcessAsUserW")[2]
    assert flags & token_launcher._DETACHED_PROCESS
    assert flags & windows_lpac._CREATE_SUSPENDED
    assert flags & windows_lpac._CREATE_UNICODE_ENVIRONMENT
    assert not flags & token_launcher._CREATE_NEW_CONSOLE
    # Never NULL: a child whose desktop is unspecified is put on a noninteractive
    # window station, which a restricted token cannot connect to.
    assert recorder.desktop == "WinSta0\\Default"

    # A caller's own console cannot be combined with DETACHED_PROCESS, so it is
    # refused here rather than failing the creation call.
    other = _WinApiRecorder()
    prepared, resources = _spawn_with_fake_api(tmp_path, monkeypatch, other, _FakeJob(other))
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "own console"):
        token_launcher._spawn_restricted(
            prepared,
            dict(_STDIO_PLAN, creationflags = token_launcher._CREATE_NEW_CONSOLE),
            identity,
            resources,
        )


def test_a_child_that_dies_in_process_start_up_is_named_by_its_status(tmp_path, monkeypatch):
    """The probe reason names the step, not just a negative number."""
    identity = _launch_identity(tmp_path)
    identity.desktop = "WinSta0\\Default"
    identity.user_objects = ("window station", "desktop")
    monkeypatch.setattr(
        token_launcher, "_control_child", lambda argv, workdir, env, desktop: "CONTROL SAYS"
    )

    reason = token_launcher._probe_start_failure(-1073741502, "", identity, {}, (sys.executable,))

    assert "STATUS_DLL_INIT_FAILED (0xc0000142)" in reason
    assert "before running" in reason
    assert "WinSta0\\Default" in reason
    assert "window station, desktop" in reason
    assert reason.endswith("CONTROL SAYS")

    # A refused DACL edit is part of the same sentence, and "neither" is said
    # rather than left blank.
    identity.user_objects = ()
    identity.user_object_reason = "the launch SID could not be granted the window station DACL: 5"
    reason = token_launcher._probe_start_failure(-1073741502, "", identity, {}, (sys.executable,))
    assert "the launch SID is on the DACL of neither" in reason
    assert "could not be granted the window station DACL: 5" in reason

    # A payload that ran and failed is reported as itself, with its output.
    assert token_launcher._probe_start_failure(1, "boom", identity, {}, ()) == (
        "the probe child exited with 1: boom"
    )


def test_a_job_that_provably_did_not_drain_is_a_cleanup_diagnostic(monkeypatch):
    """Cleanup is LIFO, so this callback runs before the ACL revoke and the temp
    removal; a child that outlived its job must not make those two silent.

    Both launchers share the one implementation, so the wait is patched where it
    lives rather than on the Limited-mode alias.
    """
    recorder = _WinApiRecorder()
    monkeypatch.setattr(windows_lpac, "_api", lambda: recorder)
    process = SimpleNamespace(close = lambda: recorder._note("process.close"))
    assert token_launcher._close_after_drain is windows_lpac._close_after_drain

    monkeypatch.setattr(windows_lpac, "_wait_for_job_drain", lambda job: False)
    with pytest.raises(OSError, match = "still held a process"):
        token_launcher._close_after_drain(process, _FakeJob(recorder))
    # The handles are released even so: keeping them would leak the job and the
    # process for the rest of Studio's life.
    names = recorder.names()
    assert names.index("job.terminate") < names.index("process.close")

    # A host that cannot be asked is not evidence of a live child.
    monkeypatch.setattr(windows_lpac, "_wait_for_job_drain", lambda job: None)
    token_launcher._close_after_drain(process, _FakeJob(recorder))
    monkeypatch.setattr(windows_lpac, "_wait_for_job_drain", lambda job: True)
    token_launcher._close_after_drain(process, _FakeJob(recorder))


def test_wait_for_job_drain_polls_until_the_job_is_empty(monkeypatch):
    recorder = _WinApiRecorder()
    recorder.active_processes = [2, 1, 0]
    monkeypatch.setattr(windows_lpac, "_api", lambda: recorder)
    assert token_launcher._wait_for_job_drain(_FakeJob(recorder)) is True
    assert recorder.names().count("QueryInformationJobObject") == 3
    # A job that never drains gives up instead of blocking cleanup forever.
    recorder.active_processes = [3]
    started = time.perf_counter()
    assert token_launcher._wait_for_job_drain(_FakeJob(recorder), timeout = 0.05) is False
    assert time.perf_counter() - started < 2
    # A host without QueryInformationJobObject is reported, not spun on, and is
    # told apart from a job that provably still holds a process: only the latter
    # is evidence that cleanup is about to run against a live child.
    del recorder.kernel32.QueryInformationJobObject
    assert token_launcher._wait_for_job_drain(_FakeJob(recorder)) is None
    # A job whose handle is already closed has nothing left to wait for.
    assert token_launcher._wait_for_job_drain(SimpleNamespace(_handle = None)) is True


def test_read_only_files_are_cleared_and_the_removal_retried(tmp_path, monkeypatch):
    handler = token_launcher._rmtree_error_handler()
    assert list(handler) == ["onexc" if sys.version_info >= (3, 12) else "onerror"]
    victim = tmp_path / "readonly.txt"
    victim.write_text("x", encoding = "utf-8")
    chmods: list[tuple[str, int]] = []
    retried: list[str] = []
    monkeypatch.setattr(os, "chmod", lambda path, mode: chmods.append((str(path), mode)))

    token_launcher._force_removable(retried.append, str(victim), PermissionError(13, "denied"))
    assert chmods == [(str(victim), stat.S_IWRITE)]
    assert retried == [str(victim)]
    # The pre-3.12 onerror shape reaches the same handler.
    token_launcher._rmtree_onerror(
        retried.append, str(victim), (PermissionError, PermissionError(13, "denied"), None)
    )
    assert retried == [str(victim), str(victim)]
    # A path that vanished under the walk is not a failure.
    token_launcher._force_removable(retried.append, str(tmp_path / "gone"), FileNotFoundError())
    assert len(retried) == 2 and len(chmods) == 2


def test_private_temp_removal_retries_a_sharing_violation_then_gives_up(tmp_path, monkeypatch):
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    private = temp_root / ("d" * 24)
    private.mkdir()
    (private / "scratch.txt").write_text("x", encoding = "utf-8")
    monkeypatch.setattr(token_launcher, "_temp_root", lambda: str(temp_root))
    sleeps: list[float] = []
    monkeypatch.setattr(token_launcher.time, "sleep", sleeps.append)
    attempts: list[str] = []
    real_rmtree = shutil.rmtree

    def flaky(path, **kwargs):
        attempts.append(path)
        if len(attempts) < 3:
            raise PermissionError(13, "the file is in use by another process")
        return real_rmtree(path, **kwargs)

    monkeypatch.setattr(token_launcher.shutil, "rmtree", flaky)
    token_launcher._remove_private_temp(str(private))
    assert len(attempts) == 3 and not private.exists()
    assert sleeps == [0.05, 0.1]

    private.mkdir()
    attempts.clear()
    monkeypatch.setattr(
        token_launcher.shutil, "rmtree",
        lambda path, **kwargs: attempts.append(path) or (_ for _ in ()).throw(
            PermissionError(13, "still in use")
        ),
    )
    with pytest.raises(PermissionError):
        token_launcher._remove_private_temp(str(private))
    assert len(attempts) == token_launcher._TEMP_REMOVAL_ATTEMPTS


def test_a_junction_in_the_private_temp_is_unlinked_instead_of_followed(tmp_path, monkeypatch):
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    private = temp_root / ("e" * 24)
    (private / "nested").mkdir(parents = True)
    (private / "nested" / "scratch.txt").write_text("x", encoding = "utf-8")
    junction = private / "junction"
    junction.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding = "utf-8")
    monkeypatch.setattr(token_launcher, "_temp_root", lambda: str(temp_root))
    monkeypatch.setattr(
        token_launcher, "_is_reparse_point", lambda path: str(path) == str(junction)
    )
    scanned: list[str] = []
    removed: list[str] = []
    real_scandir = os.scandir
    real_rmdir = os.rmdir

    def scandir(path = "."):
        scanned.append(str(path))
        return real_scandir(path)

    def rmdir(path, **kwargs):
        removed.append(str(path))
        return real_rmdir(path, **kwargs)

    monkeypatch.setattr(os, "scandir", scandir)
    monkeypatch.setattr(os, "rmdir", rmdir)

    token_launcher._remove_private_temp(str(private))

    assert not private.exists()
    assert (outside / "keep.txt").exists()
    # The reparse point is removed as a link (RemoveDirectoryW), never walked.
    assert str(junction) in removed
    assert str(junction) not in scanned


def test_reconcile_refuses_a_manifest_whose_sid_names_a_real_account(tmp_path, monkeypatch):
    manifests = tmp_path / "manifests"
    temp_root = tmp_path / "temp"
    manifests.mkdir()
    temp_root.mkdir()
    (tmp_path / "work").mkdir()
    private = temp_root / ("a" * 24)
    private.mkdir()
    planted = _manifest(
        manifests, "S-1-5-21-1004336348-1177238915-682003330-1001",
        workdir = str(tmp_path / "work"), private_temp = str(private),
        granted_roots = [str(tmp_path / "work"), str(private)], owner_pid = 4242,
    )
    revoked: list[str] = []
    freed: list = []
    monkeypatch.setattr(token_launcher, "_manifest_root", lambda: str(manifests))
    monkeypatch.setattr(token_launcher, "_temp_root", lambda: str(temp_root))
    monkeypatch.setattr(token_launcher, "_sid_from_text", lambda text: ctypes.c_void_p(9))
    monkeypatch.setattr(windows_lpac, "_process_identity", lambda pid = None: None)
    monkeypatch.setattr(windows_lpac, "_revoke_sid", lambda path, sid, **kw: revoked.append(path))
    monkeypatch.setattr(windows_lpac, "_api", lambda: _unmapped_sid_api(freed, resolves = True))

    token_launcher.WindowsRestrictedTokenBackend().reconcile_stale_manifests()

    # A planted manifest must not turn into an ACL revoke on a real account's tree.
    assert planted.exists() and private.exists()
    assert revoked == []
    assert [item.value for item in freed] == [9]


def test_reconcile_removes_orphaned_temporary_manifests(tmp_path, monkeypatch):
    manifests = tmp_path / "manifests"
    temp_root = tmp_path / "temp"
    manifests.mkdir()
    temp_root.mkdir()
    prefix = token_launcher._MANIFEST_PREFIX
    dead = manifests / (prefix + "S-1-5-21-1-1-1-1.json.tmp")
    dead.write_text(json.dumps({"owner_pid": 4242, "owner_created": 7}), encoding = "utf-8")
    live = manifests / (prefix + "S-1-5-21-2-2-2-2.json.tmp")
    live.write_text(json.dumps({"owner_pid": 5151, "owner_created": 7}), encoding = "utf-8")
    partial = manifests / (prefix + "S-1-5-21-3-3-3-3.json.tmp")
    partial.write_text('{"owner_pid": 51', encoding = "utf-8")
    stale = manifests / (prefix + "S-1-5-21-4-4-4-4.json.tmp")
    stale.write_text('{"owner_pid": 51', encoding = "utf-8")
    old = time.time() - token_launcher._ORPHAN_TEMPORARY_MANIFEST_SECONDS - 60
    os.utime(stale, (old, old))
    foreign = manifests / "unrelated.json.tmp"
    foreign.write_text("{}", encoding = "utf-8")
    monkeypatch.setattr(token_launcher, "_manifest_root", lambda: str(manifests))
    monkeypatch.setattr(token_launcher, "_temp_root", lambda: str(temp_root))
    monkeypatch.setattr(
        windows_lpac, "_process_identity", lambda pid = None: (5151, 7) if pid == 5151 else None
    )

    token_launcher.WindowsRestrictedTokenBackend().reconcile_stale_manifests()

    assert not dead.exists()  # its writer is gone, so os.replace will never run
    assert not stale.exists()  # unreadable, but older than any write can be
    assert live.exists() and partial.exists() and foreign.exists()


# ── os_sandbox selection and fallback (every platform, via a fake launcher) ──


class _FakeLauncher:
    identity = "windows-restricted-token"
    profile_id = "windows-restricted-token-write-isolation-v1"
    limitations = token_launcher._LIMITATIONS

    def __init__(self, *, available: bool = True, decline: Exception | None = None):
        self.available = available
        self.decline = decline
        self.probe_calls: list[bool] = []
        self.prepared: list[os_sandbox.ToolLaunchPlan] = []

    def probe(self, *, force: bool = False):
        self.probe_calls.append(force)
        if self.available:
            return os_sandbox.SandboxCapability(
                self.identity, True, "fake token probe passed", available = True,
                protection_state = "preview", profile_id = self.profile_id,
                limitations = self.limitations,
            )
        return os_sandbox.SandboxCapability(
            self.identity, False, "fake token probe failed", available = False
        )

    def prepare(self, spec):
        self.prepared.append(spec)
        if self.decline is not None:
            raise self.decline
        return os_sandbox.PreparedSandboxLaunch(
            argv = spec.argv, workdir = spec.workdir, env = {**spec.env, "TEMP": "private"},
            preexec_fn = None, backend = self.identity, timeout_seconds = spec.timeout_seconds,
            close_fds = spec.close_fds, terminate_descendants = spec.terminate_descendants,
            spawn_callback = lambda prepared, kwargs: None,
        )


def _limited_plan(tmp_path, monkeypatch, launcher):
    store = tool_isolation.LimitedGrantStore(ttl_seconds = 60, max_entries = 4)
    monkeypatch.setattr(tool_isolation, "_LIMITED_GRANTS", store)
    monkeypatch.setattr(os_sandbox, "_platform_backend", lambda: None)
    monkeypatch.setattr(os_sandbox, "_environment_fingerprint", lambda _backend: "env-token")
    monkeypatch.setattr(os_sandbox, "_limited_isolation_backend", lambda: launcher)
    capability = os_sandbox.capability_snapshot()
    grant = tool_isolation.issue_limited_grant(
        current_subject = "actor", tool_ui_session_id = "page",
        probe_generation = capability.probe_generation,
    )
    return os_sandbox.ToolLaunchPlan(
        argv = (sys.executable, "-c", "pass"), workdir = str(tmp_path), env = {"PATH": "p"},
        requested_mode = "limited", current_subject = "actor", tool_ui_session_id = "page",
        limited_grant = grant.token,
    )


def test_limited_mode_runs_under_the_token_launcher_once_it_qualified(
    tmp_path, monkeypatch, isolated_capability_cache
):
    launcher = _FakeLauncher()
    plan = _limited_plan(tmp_path, monkeypatch, launcher)
    prepared = os_sandbox.prepare_tool_launch(plan)
    record = prepared.execution_record
    assert prepared.backend == "windows-restricted-token"
    assert prepared.spawn_callback is not None
    assert prepared.env["TEMP"] == "private"
    assert record.effective_mode == "limited"
    assert record.os_isolation is False
    assert record.backend == "windows-restricted-token"
    assert record.profile_id == "windows-restricted-token-write-isolation-v1"
    assert record.limitations == token_launcher._LIMITATIONS
    # The token leaves the host network reachable, which the record must not
    # contradict: "deny" would read as no network path out of the sandbox.
    assert record.network_policy == "unrestricted"
    assert set(os_sandbox._LIMITED_SAFEGUARDS) <= set(record.retained_safeguards)
    assert {"write_restricted_token", "job_object"} <= set(record.retained_safeguards)
    assert record.probe_generation
    assert launcher.prepared[0].workdir == os.path.realpath(str(tmp_path))
    assert launcher.prepared[0].requested_mode == "limited"


def test_limited_mode_falls_back_to_the_process_guard_when_the_launcher_declines(
    tmp_path, monkeypatch, isolated_capability_cache
):
    launcher = _FakeLauncher(decline = os_sandbox.SandboxUnavailableError("workdir too large"))
    plan = _limited_plan(tmp_path, monkeypatch, launcher)
    prepared = os_sandbox.prepare_tool_launch(plan)
    record = prepared.execution_record
    assert prepared.backend == "process-guard"
    assert prepared.spawn_callback is None
    assert record.backend == "process-guard"
    assert record.profile_id == "limited-software-safeguards-v1"
    assert record.retained_safeguards == os_sandbox._LIMITED_SAFEGUARDS
    assert "restricted_token_unavailable" in record.limitations
    assert record.network_policy == "unrestricted"
    assert record.os_isolation is False
    # An OS-level refusal (an ACL API error) falls back the same way; anything else propagates.
    launcher.decline = OSError(5, "denied")
    assert "restricted_token_unavailable" in os_sandbox.prepare_tool_launch(plan).execution_record.limitations
    launcher.decline = RuntimeError("bug")
    with pytest.raises(RuntimeError):
        os_sandbox.prepare_tool_launch(plan)


def test_limited_mode_ignores_a_launcher_that_did_not_qualify(
    tmp_path, monkeypatch, isolated_capability_cache
):
    launcher = _FakeLauncher(available = False)
    plan = _limited_plan(tmp_path, monkeypatch, launcher)
    prepared = os_sandbox.prepare_tool_launch(plan)
    record = prepared.execution_record
    assert record.backend == "process-guard"
    assert "restricted_token_unavailable" not in record.limitations
    assert launcher.prepared == []
    if sys.platform != "linux":
        assert "detached_descendant_cleanup_unverified" in record.limitations


def test_limited_grant_is_not_bound_to_the_launcher_probe(tmp_path, monkeypatch, isolated_capability_cache):
    # The token launcher qualifying (or not) must not invalidate grants: it is
    # deliberately outside probe_generation, so a grant issued before the launcher
    # probe still authorises the launch that now runs under the token.
    plan = _limited_plan(tmp_path, monkeypatch, _FakeLauncher(available = False))
    before = os_sandbox.capability_snapshot()
    os_sandbox._capability_cache.clear()
    monkeypatch.setattr(os_sandbox, "_limited_isolation_backend", lambda: _FakeLauncher())
    after = os_sandbox.capability_snapshot()
    assert before.probe_generation == after.probe_generation
    assert before.limited_backend == "process-guard"
    assert after.limited_backend == "windows-restricted-token"
    prepared = os_sandbox.prepare_tool_launch(plan)
    assert prepared.execution_record.backend == "windows-restricted-token"


def test_capability_snapshot_reports_what_limited_runs_under(monkeypatch, isolated_capability_cache):
    os_backend = SimpleNamespace(
        identity = "fake-os",
        profile_id = "fake-os-v1",
        probe = lambda: os_sandbox.SandboxCapability("fake-os", False, "no bwrap", available = False),
    )
    monkeypatch.setattr(os_sandbox, "_platform_backend", lambda: os_backend)
    monkeypatch.setattr(os_sandbox, "_environment_fingerprint", lambda _backend: "env-cap")
    launcher = _FakeLauncher()
    monkeypatch.setattr(os_sandbox, "_limited_isolation_backend", lambda: launcher)

    first = os_sandbox.capability_snapshot()
    assert first.available is False
    assert first.limited_backend == "windows-restricted-token"
    assert first.limited_profile_id == "windows-restricted-token-write-isolation-v1"
    assert first.limited_limitations == token_launcher._LIMITATIONS
    assert "passed" in first.limited_reason
    assert launcher.probe_calls == [False]
    assert os_sandbox.capability_snapshot() is first  # cached with its limited fields
    forced = os_sandbox.capability_snapshot(force = True)
    assert launcher.probe_calls == [False, True]
    assert forced.limited_backend == "windows-restricted-token"

    os_sandbox._capability_cache.clear()
    monkeypatch.setattr(os_sandbox, "_limited_isolation_backend", lambda: _FakeLauncher(available = False))
    unavailable = os_sandbox.capability_snapshot()
    assert unavailable.limited_backend == "process-guard"
    assert unavailable.limited_profile_id == "limited-software-safeguards-v1"
    assert unavailable.limited_limitations == ()
    assert unavailable.limited_reason == "fake token probe failed"

    os_sandbox._capability_cache.clear()
    monkeypatch.setattr(os_sandbox, "_limited_isolation_backend", lambda: None)
    plain = os_sandbox.capability_snapshot()
    assert plain.limited_backend == "process-guard"
    assert plain.limited_reason == ""


def test_capability_snapshot_survives_a_crashing_launcher_probe(monkeypatch, isolated_capability_cache):
    monkeypatch.setattr(os_sandbox, "_platform_backend", lambda: None)
    monkeypatch.setattr(os_sandbox, "_environment_fingerprint", lambda _backend: "env-crash")

    class _Crashing(_FakeLauncher):
        def probe(self, *, force: bool = False):
            raise RuntimeError("ctypes exploded")

    monkeypatch.setattr(os_sandbox, "_limited_isolation_backend", lambda: _Crashing())
    unsupported = os_sandbox.capability_snapshot()
    assert unsupported.limited_backend == "process-guard"
    assert "RuntimeError: ctypes exploded" in unsupported.limited_reason
    os_backend = SimpleNamespace(
        identity = "fake-os", profile_id = "fake-os-v1",
        probe = lambda: os_sandbox.SandboxCapability("fake-os", False, "no", available = False),
    )
    monkeypatch.setattr(os_sandbox, "_platform_backend", lambda: os_backend)
    os_sandbox._capability_cache.clear()
    capability = os_sandbox.capability_snapshot()
    assert capability.limited_backend == "process-guard"
    assert "RuntimeError: ctypes exploded" in capability.limited_reason


def test_tools_treats_both_job_owning_backends_alike():
    assert inference_tools._WINDOWS_JOB_OWNING_BACKENDS == {"windows-lpac", "windows-restricted-token"}
    source = Path(inference_tools.__file__).read_text(encoding = "utf-8")
    assert source.count("or prepared_launch.backend in _WINDOWS_JOB_OWNING_BACKENDS") == 2
    assert 'prepared_launch.backend == "windows-lpac"' not in source


def test_windows_lpac_exposes_the_shared_job_factory():
    lpac_source = Path(windows_lpac.__file__).read_text(encoding = "utf-8")
    assert "def _job_object_with_limits()" in lpac_source
    assert "_PROC_THREAD_ATTRIBUTE_JOB_LIST = 0x0002000D" in lpac_source
    for name in ("OpenProcessToken", "CreateRestrictedToken", "SetTokenInformation",
                 "CreateProcessAsUserW", "IsTokenRestricted", "IsProcessInJob"):
        assert f"{name}.argtypes" in lpac_source, name
    assert windows_lpac.__all__ == ["WindowsLpacBackend", "WindowsLpacProcess"]


# ── live Windows tests ───────────────────────────────────────────────────────


@pytest.fixture(scope = "module")
def live_token_backend():
    if sys.platform != "win32":
        pytest.skip("native restricted-token tests run only on Windows")
    backend = os_sandbox._limited_isolation_backend()
    assert isinstance(backend, token_launcher.WindowsRestrictedTokenBackend)
    started = time.perf_counter()
    capability = backend.probe(force = True)
    print(f"restricted-token probe: {time.perf_counter() - started:.2f}s {capability.reason}")
    assert capability.available is True, capability.reason
    assert capability.profile_id == backend.profile_id
    # Which of the two profile disclosures this host earns depends on the account
    # the suite runs as, so the fixture pins the pair rather than one of them.
    assert capability.limitations in (
        token_launcher._LIMITATIONS,
        token_launcher._LIMITATIONS_PROFILE_UNREADABLE,
    ), capability.limitations
    assert backend.limitations == capability.limitations
    return backend


def _run_live(backend, workdir: Path, code: str, *argv: str, timeout: int = 60) -> tuple[str, int, float]:
    started = time.perf_counter()
    prepared = backend.prepare(
        os_sandbox.ToolLaunchPlan(
            argv = (sys.executable, "-I", "-S", "-c", code, *argv),
            workdir = str(workdir),
            env = {
                "PATH": os.pathsep.join(
                    (os.path.dirname(sys.executable), os.path.join(os.environ["SystemRoot"], "System32"))
                ),
                "PYTHONIOENCODING": "utf-8",
            },
        )
    )
    identity = prepared.spawn_callback._launch_identity
    # The window station and desktop DACLs are a documented precondition of
    # CreateProcessAsUser; a host that refuses the edit says so here rather than
    # in a start-up failure the child never survives to explain.
    assert identity.user_objects == ("window station", "desktop"), identity.user_object_reason
    assert identity.desktop and "\\" in identity.desktop
    # The SID the first access check is decided against, read off the token that
    # was built. Which roots needed an ACE for it depends on what the workdir
    # already granted, so only the subset relation is pinned.
    assert identity.user_sid_string.startswith("S-1-")
    assert set(identity.user_sid_roots) <= set(identity.granted_roots)
    process = None
    output = ""
    try:
        process = os_sandbox.spawn_prepared_launch(
            prepared, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, stdin = subprocess.DEVNULL,
            text = True, encoding = "utf-8", errors = "replace", cwd = prepared.workdir,
            env = prepared.env, close_fds = True,
        )
        process.wait(timeout = timeout)
        output = process.stdout.read()
        returncode = process.returncode
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout = 10)
        prepared.cleanup()
        assert prepared.cleanup_diagnostics == []
        assert not Path(identity.private_temp).exists()
        assert not Path(identity.manifest_path).exists()
        acl = subprocess.run(["icacls", str(workdir)], capture_output = True, text = True).stdout
        assert identity.sid_string not in acl
    return output, returncode, time.perf_counter() - started


def test_live_token_child_writes_only_the_workdir_and_private_temp(live_token_backend, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    secret_root = Path(token_launcher._private_root("limited-probe-tests"))
    secret = secret_root / f"secret-{os.getpid()}.txt"
    secret.write_text("secret", encoding = "utf-8")
    try:
        output, returncode, elapsed = _run_live(
            live_token_backend, work,
            "import json, os, sys\n"
            "def w(p):\n"
            "    try:\n"
            "        open(p, 'a').write('x'); return True\n"
            "    except OSError as e:\n"
            "        return type(e).__name__\n"
            "def r(p):\n"
            "    try:\n"
            "        return open(p).read()\n"
            "    except OSError as e:\n"
            "        return type(e).__name__\n"
            "print(json.dumps({'secret_read': r(sys.argv[1]), 'secret_write': w(sys.argv[1]),"
            " 'work': w('out.txt'), 'temp': w(os.path.join(os.environ['TEMP'], 't.txt')),"
            " 'user_temp': w(os.path.join(sys.argv[2], 'u.txt')), 'exe': sys.executable}))",
            str(secret), str(secret_root),
        )
        assert returncode == 0, output
        report = json.loads(output.strip().splitlines()[-1])
        # Reads usually keep the user's access, which the record discloses. On an
        # administrator's account a file reachable only through BUILTIN\Administrators
        # is refused instead, because that group is deny-only in this token; both are
        # accepted here and the disclosure is what has to match, not the outcome.
        assert report["secret_read"] in ("secret", "PermissionError"), report
        readable = report["secret_read"] == "secret"
        assert live_token_backend.limitations == token_launcher._disclosed_limitations(
            profile_readable = readable, named_pipes = True
        )
        assert report["secret_write"] == "PermissionError"
        assert report["user_temp"] == "PermissionError"
        assert report["work"] is True and (work / "out.txt").exists()
        assert report["temp"] is True
        assert os.path.normcase(report["exe"]) == os.path.normcase(sys.executable)
        print(f"restricted-token launch round trip: {elapsed:.2f}s")
        assert elapsed < 30
    finally:
        secret.unlink(missing_ok = True)


def test_live_token_child_keeps_nul_pipes_and_multiprocessing(live_token_backend, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    output, returncode, _ = _run_live(
        live_token_backend, work,
        "import os\n"
        "open(os.devnull, 'r+b').close()\n"
        "import multiprocessing.connection as c\n"
        "a, b = c.Pipe(); a.send('ping'); assert b.recv() == 'ping'\n"
        "import subprocess, sys\n"
        "print(subprocess.run([sys.executable, '-I', '-S', '-c', 'print(\"grandchild ok\")'],"
        " capture_output=True, text=True).stdout.strip())\n"
        "print('ok')",
    )
    assert returncode == 0, output
    assert "grandchild ok" in output and output.strip().endswith("ok")


def test_live_token_is_restricted_lua_and_privilege_stripped(live_token_backend, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    output, returncode, _ = _run_live(
        live_token_backend, work,
        token_launcher._PROBE_PAYLOAD,
        str(work / "missing-secret"), str(work), str(work / "missing-user-only"), "",
    )
    assert returncode == 0, output
    findings = json.loads(output.strip().splitlines()[-1])
    assert findings["restricted"] is True
    assert findings["privileges"] <= 1
    assert findings["in_job"] is True
    assert "S-1-1-0" in findings["restricted_sids"]
    assert any(token_launcher._is_launch_sid_text(s) for s in findings["restricted_sids"])
    assert findings["devnull"] is True and findings["pipe"] is True
    assert findings["interpreter_readable"] is True
    # The principal the first access check is decided against, which is what makes
    # an unreadable user profile diagnosable rather than merely surprising.
    assert findings["user_sid"].startswith("S-1-")
    # argv[1] is a file that was never created, so the read reports the error it
    # got instead of a bare False.
    assert findings["secret_exists"] is False
    assert isinstance(findings["secret_readable"], str)
    # Same for argv[3] here: this test drives the payload by hand rather than
    # through _live_probe, so no user-SID directory exists to read or write.
    assert findings["user_only_readable"] is not True
    assert findings["user_only_writable"] is not True


def test_live_detached_grandchild_dies_with_the_job(live_token_backend, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    output, returncode, _ = _run_live(
        live_token_backend, work,
        "import subprocess, sys\n"
        "p = subprocess.Popen([sys.executable, '-I', '-S', '-c', 'import time; time.sleep(300)'],"
        " creationflags=0x8 | 0x200)\n"  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        "print(p.pid)",
    )
    assert returncode == 0, output
    grandchild = int(output.strip().splitlines()[-1])
    deadline = time.time() + 10
    while time.time() < deadline and windows_lpac._process_identity(grandchild) is not None:
        time.sleep(0.2)
    assert windows_lpac._process_identity(grandchild) is None


def test_live_breakaway_is_refused_before_any_process_exists(live_token_backend, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    prepared = live_token_backend.prepare(
        os_sandbox.ToolLaunchPlan(argv = (sys.executable, "-c", "pass"), workdir = str(work), env = {})
    )
    try:
        with pytest.raises(os_sandbox.SandboxUnavailableError, match = "break away"):
            prepared.spawn_callback(
                prepared,
                {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL,
                 "creationflags": 0x01000000},
            )
        with pytest.raises(os_sandbox.SandboxUnavailableError, match = "stdio plan"):
            prepared.spawn_callback(prepared, {"stdout": subprocess.PIPE})
    finally:
        prepared.cleanup()
    assert prepared.cleanup_diagnostics == []


def test_live_limited_tool_call_records_the_token_launcher(live_token_backend, tmp_path, monkeypatch):
    workdir = tmp_path / "limited-work"
    workdir.mkdir()
    monkeypatch.setattr(inference_tools, "_get_workdir", lambda _session: str(workdir))
    monkeypatch.setattr(os_sandbox, "_platform_backend", lambda: None)
    before = dict(os_sandbox._capability_cache)
    os_sandbox._capability_cache.clear()
    try:
        capability = os_sandbox.capability_snapshot(force = True)
        assert capability.available is False
        assert capability.limited_backend == "windows-restricted-token"
        grant = tool_isolation.issue_limited_grant(
            current_subject = "test:limited-user", tool_ui_session_id = "test-page-token",
            probe_generation = capability.probe_generation,
        )
        records = []
        result = inference_tools._python_exec(
            "import os\nopen('made-here.txt', 'w').write('x')\nprint(os.environ['TEMP'])",
            timeout = 60, tool_execution_mode = "limited", current_subject = "test:limited-user",
            tool_ui_session_id = "test-page-token", limited_grant = grant.token,
            launch_record_callback = records.append,
        )
    finally:
        os_sandbox._capability_cache.clear()
        os_sandbox._capability_cache.update(before)
    assert (workdir / "made-here.txt").exists(), result
    assert "limited-temp" in result
    assert len(records) == 1
    assert records[0].backend == "windows-restricted-token"
    assert records[0].effective_mode == "limited"
    assert "write_restricted_token" in records[0].retained_safeguards
    # The record repeats what the probe observed, never the class default.
    assert records[0].limitations == live_token_backend.limitations
    assert records[0].limitations in (
        token_launcher._LIMITATIONS,
        token_launcher._LIMITATIONS_PROFILE_UNREADABLE,
    )
    assert not Path(result.strip().splitlines()[-1]).exists()  # private temp removed after the call


def test_token_information_class_constants_are_integers():
    # Staging round 4: a ctypes Structure named after the TOKEN_INFORMATION_CLASS
    # constant shadowed it, and SetTokenInformation received the type object as its
    # class argument ("argument 2: TypeError ... cannot be interpreted as an integer"),
    # which made every restricted-token probe fail and Limited fall back silently.
    for name in ("_TOKEN_DEFAULT_DACL", "_TOKEN_LOGON_SID", "_TOKEN_GROUPS_CLASS", "_TOKEN_PRIVILEGES_CLASS"):
        value = getattr(token_launcher, name, None)
        if value is not None:
            assert isinstance(value, int), name
    assert token_launcher._TOKEN_DEFAULT_DACL == 6
    assert issubclass(token_launcher._TOKEN_DEFAULT_DACL_INFO, ctypes.Structure)


def test_set_default_dacl_passes_the_integer_class_and_frees_the_acl(monkeypatch):
    calls: list[tuple] = []
    # GetTokenInformation hands back a raw buffer; a zeroed one means no default DACL.
    info = ctypes.create_string_buffer(ctypes.sizeof(token_launcher._TOKEN_DEFAULT_DACL_INFO))
    monkeypatch.setattr(
        token_launcher, "_token_information", lambda api, token, kind: calls.append(("query", kind)) or info
    )

    def set_entries(count, entries, old_acl, new_acl_ref):
        calls.append(("SetEntriesInAclW", int(count), old_acl))
        new_acl_ref._obj.value = 0x5150
        return 0

    def set_token_information(token, klass, info_ref, size):
        calls.append(("SetTokenInformation", token, klass, int(size)))
        return 1

    api = SimpleNamespace(
        advapi32 = SimpleNamespace(
            SetEntriesInAclW = set_entries, SetTokenInformation = set_token_information
        ),
        kernel32 = SimpleNamespace(LocalFree = lambda handle: calls.append(("LocalFree", handle.value if hasattr(handle, "value") else handle))),
    )
    token_launcher._set_default_dacl(api, 77, ctypes.c_void_p(0x1234))
    assert calls[0] == ("query", 6)
    assert calls[1][:2] == ("SetEntriesInAclW", 1)
    set_call = next(call for call in calls if call[0] == "SetTokenInformation")
    assert set_call[1] == 77
    assert isinstance(set_call[2], int) and set_call[2] == 6
    assert set_call[3] == ctypes.sizeof(token_launcher._TOKEN_DEFAULT_DACL_INFO)
    assert ("LocalFree", 0x5150) in calls
