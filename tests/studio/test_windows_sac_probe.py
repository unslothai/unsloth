# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Guards for scripts/windows_sac_probe, the Smart App Control evidence probe.

The PowerShell half cannot run here; its guards read the source. The Python
scenario is driven against a fake Studio.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_DIR = REPO_ROOT / "scripts" / "windows_sac_probe"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "windows-llama-signature-audit.yml"


def _load_scenario():
    spec = importlib.util.spec_from_file_location(
        "studio_scenario", PROBE_DIR / "studio_scenario.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_the_repo_variant_shorthand_is_split_into_the_two_load_fields():
    """/api/inference/load has no shorthand parser: a colon left in model_path
    reaches the hub as an invalid repository id, so the default probe never
    reached the llama.cpp load it exists to exercise."""
    s = _load_scenario()
    assert s.split_model_ref("unsloth/Qwen3.5-2B-MTP-GGUF:UD-Q4_K_XL") == (
        "unsloth/Qwen3.5-2B-MTP-GGUF",
        "UD-Q4_K_XL",
    )
    assert s.split_model_ref("unsloth/Qwen3.5-2B-GGUF") == ("unsloth/Qwen3.5-2B-GGUF", None)
    assert s.split_model_ref(r"C:\models\x.gguf") == (r"C:\models\x.gguf", None)
    assert s.split_model_ref("C:/models/x.gguf") == ("C:/models/x.gguf", None)


def test_the_status_poller_does_not_shadow_thread_stop(monkeypatch):
    """Thread.join() calls its own internal _stop(); an Event assigned over it
    raised out of the finally block before the results were written."""
    s = _load_scenario()
    monkeypatch.setattr(s, "_request", lambda *a, **k: (200, {}))
    poller = s.StatusPoller("http://127.0.0.1:1", "t", interval = 0.01)
    assert not isinstance(getattr(poller, "_stop", None), threading.Event)
    poller.start()
    poller.stop()
    poller.join(timeout = 5)  # raised TypeError before the rename
    assert not poller.is_alive()
    assert poller.in_flight_ms() is None


def test_a_never_opened_studio_is_not_rotated_to_a_published_password(
    tmp_path, monkeypatch, capsys
):
    """The rotation is permanent and revert does not undo it; a default here
    left every probed machine on a known credential, and printing the value
    put it into the evidence zip."""
    s = _load_scenario()
    (tmp_path / "auth").mkdir()
    (tmp_path / "auth" / ".bootstrap_password").write_text("boot-secret", encoding = "utf-8")
    posted: list[tuple[str, dict]] = []

    def fake(
        base_url,
        method,
        path,
        payload = None,
        token = None,
        timeout = 900,
    ):
        posted.append((path, payload or {}))
        return 200, {"access_token": "tok"}

    monkeypatch.setattr(s, "_request", fake)
    with pytest.raises(SystemExit, match = "password"):
        s.authenticate("http://x", tmp_path, None)
    assert posted == [], "nothing may be rotated without an operator password"

    s.authenticate("http://x", tmp_path, "operators-choice")
    change = [p for p in posted if p[0] == "/api/auth/change-password"]
    assert change and change[0][1] == {
        "current_password": "boot-secret",
        "new_password": "operators-choice",
    }
    assert "operators-choice" not in capsys.readouterr().out
    source = (PROBE_DIR / "studio_scenario.py").read_text(encoding = "utf-8")
    assert "unsloth-sac-probe" not in source


def test_the_scenario_loads_with_the_variant_field_and_unloads_by_model_path(tmp_path, monkeypatch):
    """UnloadRequest.model_path is required: an empty body is a 422 that
    leaves the runtime resident for the next matrix cell to reuse."""
    s = _load_scenario()
    calls: list[tuple[str, str, dict]] = []

    def fake(
        base_url,
        method,
        path,
        payload = None,
        token = None,
        timeout = 900,
    ):
        calls.append((method, path, payload or {}))
        if path == "/api/liveness":
            return 200, {}
        if path == "/api/auth/login":
            return 200, {"access_token": "tok"}
        if path == "/v1/chat/completions":
            return 200, {"choices": [{"message": {"content": "hi", "tool_calls": []}}]}
        # The padded routes answer with a real payload; {} is a truncated pad.
        return 200, {"status": "done"}

    def fake_stream(
        base_url,
        path,
        payload,
        token = None,
        timeout = 900,
    ):
        calls.append(("STREAM", path, payload or {}))
        return (
            200,
            [
                {"type": "tool_start", "tool_name": "web_search"},
                {"type": "tool_end", "tool_name": "web_search", "result": "Reykjavik: 7 September"},
                {"choices": [{"delta": {"content": "found it"}}]},
            ],
            None,
        )

    monkeypatch.setattr(s, "_request", fake)
    monkeypatch.setattr(s, "_stream_events", fake_stream)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "studio_scenario.py",
            "--model",
            "unsloth/Qwen3.5-2B-MTP-GGUF:UD-Q4_K_XL",
            "--out",
            str(tmp_path),
            "--port",
            "1",
            "--password",
            "pw",
            "--home",
            str(tmp_path),
            "--poll-seconds",
            "0.05",
        ],
    )
    assert s.main() == 0
    load = next(c for c in calls if c[1] == "/api/inference/load")
    assert load[2] == {"model_path": "unsloth/Qwen3.5-2B-MTP-GGUF", "gguf_variant": "UD-Q4_K_XL"}
    unloads = [c for c in calls if c[1] == "/api/inference/unload"]
    assert len(unloads) == 2, "one eviction before the load, one unload after"
    assert all(u[2] == {"model_path": "unsloth/Qwen3.5-2B-MTP-GGUF"} for u in unloads)
    chat = next(c for c in calls if c[1] == "/v1/chat/completions")
    assert chat[2]["model"] == "unsloth/Qwen3.5-2B-MTP-GGUF"
    streamed = [c for c in calls if c[0] == "STREAM"]
    assert len(streamed) == 2 and all(c[2]["stream"] is True for c in streamed)
    assert all(c[2]["tool_choice"]["function"]["name"] == "web_search" for c in streamed)
    results = json.loads((tmp_path / "scenario-results.json").read_text(encoding = "utf-8"))
    assert results["steps"]["unload"]["ok"] is True
    assert "in_flight_ms" in results["status_poll"] and "stalls_ms" in results["status_poll"]
    assert results["gguf_variant"] == "UD-Q4_K_XL"


def test_the_poller_abandons_a_read_at_the_frontend_timeout_and_measures_the_stall(monkeypatch):
    """The frontend ticks every 5 s and abandons a status read after 10 s, so a
    stall is a stream of abandoned reads; one 300 s request neither reproduces
    that load nor records the 75 to 80 s stall this exists to measure."""
    s = _load_scenario()
    assert s.STATUS_READ_TIMEOUT_S == 10.0 and s.STATUS_INTERVAL_S == 5.0
    source = (PROBE_DIR / "studio_scenario.py").read_text(encoding = "utf-8")
    status_call = source[source.index('"/api/inference/status"') :]
    assert "timeout = self.read_timeout" in status_call[: status_call.index(")")]
    assert "poller.join(timeout = STATUS_READ_TIMEOUT_S + 15)" in source

    def stalled(
        base_url,
        method,
        path,
        payload = None,
        token = None,
        timeout = 900,
    ):
        # What _request returns once urllib gives up at `timeout`.
        threading.Event().wait(timeout)
        return 0, "timed out"

    monkeypatch.setattr(s, "_request", stalled)
    poller = s.StatusPoller("http://127.0.0.1:1", "t", interval = 0.02, read_timeout = 0.05)
    poller.start()
    threading.Event().wait(0.4)
    poller.stop()
    poller.join(timeout = 5)
    assert len(poller.polls) >= 4, "abandoned reads must be followed at once by the next"
    assert all(timed_out for _, _, timed_out in poller.polls)
    stalls = poller.stalls_ms()
    assert len(stalls) == 1 and stalls[0] >= 200, stalls
    # Mixed history: two runs of abandoned reads separated by a good read.
    poller.polls = [(0.0, 50.0, True), (0.06, 50.0, True), (0.2, 5.0, False), (0.3, 50.0, True)]
    assert [round(x) for x in poller.stalls_ms()] == [110, 50]


def test_an_empty_bootstrap_file_means_rotated(tmp_path, monkeypatch):
    """Studio truncates .bootstrap_password on Windows when it cannot delete
    it, so an existence check picked the empty string over the operator's
    password and every login failed on exactly the locked-down machines."""
    s = _load_scenario()
    (tmp_path / "auth").mkdir()
    (tmp_path / "auth" / ".bootstrap_password").write_text("", encoding = "utf-8")
    posted: list[tuple[str, dict]] = []

    def fake(
        base_url,
        method,
        path,
        payload = None,
        token = None,
        timeout = 900,
    ):
        posted.append((path, payload or {}))
        return 200, {"access_token": "tok"}

    monkeypatch.setattr(s, "_request", fake)
    s.authenticate("http://x", tmp_path, "operators-choice")
    assert posted[0] == ("/api/auth/login", {"username": "unsloth", "password": "operators-choice"})
    assert not any(p[0] == "/api/auth/change-password" for p in posted)


def test_a_model_already_resident_is_evicted_first_and_never_counts_as_loaded(
    tmp_path, monkeypatch
):
    """/load answers already_loaded for a resident model and starts nothing, so
    no PE is loaded inside the evidence window and the absence of events would
    read as an allow."""
    s = _load_scenario()
    calls: list[tuple[str, str]] = []

    def fake(
        base_url,
        method,
        path,
        payload = None,
        token = None,
        timeout = 900,
    ):
        calls.append((method, path))
        if path == "/api/liveness":
            return 200, {}
        if path == "/api/auth/login":
            return 200, {"access_token": "tok"}
        if path == "/api/inference/load":
            return 200, {"status": "already_loaded", "model": "x"}
        return 200, {"status": "done"}

    monkeypatch.setattr(s, "_request", fake)
    monkeypatch.setattr(s, "_stream_events", lambda *a, **k: (200, [], None))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "studio_scenario.py",
            "--model",
            "m",
            "--out",
            str(tmp_path),
            "--port",
            "1",
            "--password",
            "pw",
            "--home",
            str(tmp_path),
            "--poll-seconds",
            "0.05",
        ],
    )
    assert s.main() == 1
    paths = [c[1] for c in calls]
    assert paths.index("/api/inference/unload") < paths.index("/api/inference/load")
    results = json.loads((tmp_path / "scenario-results.json").read_text(encoding = "utf-8"))
    assert results["steps"]["load"]["ok"] is False
    assert "already resident" in results["steps"]["load"]["error"]
    assert results["evicted_before_load"]["status"] == 200


def test_a_tool_end_carrying_a_refusal_or_error_is_not_an_execution(monkeypatch):
    """The loop emits tool_end for a declined call, a lost runtime and an
    interrupted call too, with the reason in `result`."""
    s = _load_scenario()
    assert s.tool_end_failure("") is not None
    assert s.tool_end_failure(None) is not None
    assert s.tool_end_failure(s.TOOL_REJECTED_MESSAGE) == "declined before running"
    assert s.tool_end_failure(
        "Error: lost connection to llama-server before the tool call completed."
    )
    assert s.tool_end_failure('{"results": [{"title": "Burj Khalifa"}]}') is None
    monkeypatch.setattr(
        s,
        "_stream_events",
        lambda *a, **k: (
            200,
            [
                {"type": "tool_start", "tool_name": "web_search"},
                {"type": "tool_end", "tool_name": "web_search", "result": s.TOOL_REJECTED_MESSAGE},
            ],
            None,
        ),
    )
    turn = s.chat("http://x", "t", "m", "look it up", tools = True)
    assert (
        turn["ok"] is False
        and turn["tools_run"] == []
        and turn["tools_failed"][0]["why"] == "declined before running"
    )


def test_the_powershell_probe_collects_honestly_and_never_installs_from_run():
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    # run restarts an installed Studio and never installs one; prepare may.
    assert "Initialize-Studio $dir $false" in ps1 and "Initialize-Studio $dir $true" in ps1
    assert "function Initialize-Studio([string] $dir, [bool] $allowInstall)" in ps1
    # No policy list is a failed setup, not a warning.
    assert (
        "CiTool listed no policies" in ps1
        and "could not be verified as active; read the 3076 count" not in ps1
    )
    # Only "nothing matched" is an empty window; any other query failure is recorded and fatal.
    assert "NoMatchingEventsFound*" in ps1 and "events-collection-error.txt" in ps1
    assert 'Write-Warning "no CodeIntegrity events in the window' not in ps1
    # Each Defender preference is restored on its own.
    assert "foreach ($r in $restores)" in ps1 and ps1.count("Set-MpPreference @params") == 1


def test_the_new_guard_runs_in_the_unfiltered_lint_job():
    lint = (REPO_ROOT / ".github" / "workflows" / "workflow-trigger-lint.yml").read_text(
        encoding = "utf-8"
    )
    assert "tests/studio/test_windows_sac_probe.py" in lint


def test_the_powershell_probe_restores_what_prepare_changed_and_unmounts_efi():
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    # EFI mounted only by the probe is unmounted by the probe, on both stages.
    assert ps1.count("Dismount-Efi $mounted") == 2 and "mountvol.exe S: /D" in ps1
    # Native exit codes are read rather than trusted.
    assert "Invoke-Native 'CiTool.exe' @('-r')" in ps1 and "Invoke-Native 'mountvol.exe'" in ps1
    # The policy is verified active, not merely listed.
    assert "Test-PolicyActive $NOISG_GUID" in ps1
    # A pre-existing policy with the same GUID is saved and restored, not deleted.
    assert "AuditPolicyPreexisting" in ps1 and "preexisting-policy.cip" in ps1
    # The CodeIntegrity log settings are recorded and restored.
    assert (
        "CiLogMaxSize" in ps1 and "CiLogEnabled" in ps1 and "/ms:$($baseline.CiLogMaxSize)" in ps1
    )
    # Zero events serialise as `[]`, not as an empty file.
    assert (
        "ConvertTo-Json -InputObject $shaped" in ps1
        and "ConvertTo-Json -InputObject @($inventory)" in ps1
    )
    # The runtime and the logs follow Studio's own overrides.
    assert "UNSLOTH_LLAMA_CPP_PATH" in ps1 and "UNSLOTH_STUDIO_HOME" in ps1
    assert "Join-Path $env:USERPROFILE '.unsloth\\studio\\logs'" not in ps1


def test_the_signature_audit_covers_every_windows_family():
    """ROCm and the arm64 CPU bundle are built and packaged apart from the x64
    ones; a set without them passes under enforce while those users get
    unsigned PEs."""
    body = WORKFLOW.read_text(encoding = "utf-8")
    for pattern in (
        "*windows-x64-cpu.zip",
        "*windows-arm64-cpu.zip",
        "*windows-x64-vulkan.zip",
        "*windows-x64-cuda12-legacy.zip",
        "*windows-x64-rocm-gfx110X.zip",
    ):
        assert pattern in body, pattern


def test_the_readme_does_not_claim_the_release_tag_pins_a_run():
    body = (PROBE_DIR / "README.md").read_text(encoding = "utf-8")
    assert "read\nby the installer only" in body or "read by the installer only" in body
    assert "re-run the Studio installer" in body
    assert "$env:UNSLOTH_STUDIO_PASSWORD" in body


def test_a_padded_load_reply_is_read_for_its_deferred_error():
    """/api/inference/load commits a 200 after 15 seconds and reports a later
    failure only in the body, so a download followed by the CodeIntegrity
    refusal this probe exists to catch was recorded as a successful load."""
    s = _load_scenario()
    assert s.padded_route_failure(200, {"status": "loaded"}) is None
    assert s.padded_route_failure(500, {"detail": "boom"}) == "{'detail': 'boom'}"
    deferred = s.padded_route_failure(
        200, {"_deferred_error": {"status_code": 500, "detail": "llama-server was blocked"}}
    )
    assert deferred and "llama-server was blocked" in deferred
    assert s.padded_route_failure(200, {}) is not None, "a truncated padded body is not a load"
    assert s.padded_route_failure(200, "") is not None
    source = (PROBE_DIR / "studio_scenario.py").read_text(encoding = "utf-8")
    assert source.count("padded_route_failure(status, body)") == 2, "both padded routes read it"


def test_a_tool_turn_is_ok_only_when_a_tool_actually_ran(monkeypatch):
    """The non-streaming route drains the tool loop and returns the final text
    alone, so a model answering from memory passed as a tool turn."""
    s = _load_scenario()
    monkeypatch.setattr(
        s,
        "_stream_events",
        lambda *a, **k: (200, [{"choices": [{"delta": {"content": "Burj Khalifa"}}]}], None),
    )
    memory = s.chat("http://x", "t", "m", "look it up", tools = True)
    assert memory["ok"] is False and memory["tool_calls"] == 0 and "no tool_end" in memory["error"]
    monkeypatch.setattr(
        s,
        "_stream_events",
        lambda *a, **k: (
            200,
            [
                {"type": "tool_start", "tool_name": "web_search"},
                {"type": "tool_end", "tool_name": "web_search", "result": "Burj Khalifa, 828 m"},
            ],
            None,
        ),
    )
    ran = s.chat("http://x", "t", "m", "look it up", tools = True)
    assert ran["ok"] is True and ran["tools_run"] == ["web_search"]


def test_the_powershell_probe_handles_retries_skips_and_occupied_drives():
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    # -SkipStudio means signature only: no Studio start or install in the window.
    assert (
        "if (-not $SkipStudio -and -not (Test-StudioResponding $Port)) { Initialize-Studio $dir $false }"
        in ps1
    )
    # collect refuses to invent an event window.
    assert "(Get-Date).AddHours(-2)" not in ps1
    assert "no window-start.txt under" in ps1
    # An occupied S: that is not the EFI system partition is an error, not "already mounted".
    assert "'S:\\EFI\\Microsoft\\Boot'" in ps1
    # A clean machine's detections serialise as [].
    assert "ConvertTo-Json -InputObject $detections" in ps1
    # winget upgrade --all is opt-in, since revert cannot undo it.
    assert "if ($UpgradePackages) {" in ps1 and "if (-not $SkipUpdates) {" in ps1
    winget = ps1.index("winget upgrade --all --accept")
    assert ps1.rfind("if ($UpgradePackages) {", 0, winget) > ps1.rfind(
        "if (-not $SkipUpdates) {", 0, winget
    )
    # A rerun keeps the first baseline and does not treat its own policy as pre-existing.
    assert "$baseline = Get-Content -LiteralPath $baselinePath -Raw | ConvertFrom-Json" in ps1
    assert "(Test-Path -LiteralPath $NOISG_DEST) -and -not $baseline.AuditPolicyApplied" in ps1
    # Rollback state is on disk before the refresh that can fail.
    applied = ps1.index("$baseline.AuditPolicyApplied = $true")
    refresh = ps1.index("Invoke-Native 'CiTool.exe' @('-r')")
    persist = ps1.index("Set-Content -LiteralPath $baselinePath", applied)
    assert applied < persist < refresh
    # The interpreter locator shares Get-StudioHome's precedence (STUDIO_HOME alias included).
    locator = ps1[
        ps1.index("function Get-StudioPython") : ps1.index("function Test-StudioResponding")
    ]
    assert "Get-StudioHome" in locator and "$env:UNSLOTH_STUDIO_HOME" not in locator


def test_the_signature_audit_fails_on_any_missing_bundle():
    """A native exit code is not terminating in pwsh and a later download
    replaces $LASTEXITCODE, so a family whose asset was missing was audited as
    absent and passed under enforce."""
    body = WORKFLOW.read_text(encoding = "utf-8")
    download = body.index("gh release download $tag")
    loop_end = body.index("Get-ChildItem bundles | Format-Table")
    block = body[download:loop_end]
    assert "if ($LASTEXITCODE -ne 0) { throw" in block
    assert "Get-ChildItem bundles -Filter $pattern" in block


class _FakeStream:
    """What urlopen hands back: an iterable of SSE lines with a status."""

    def __init__(self, lines):
        self.status = 200
        self._lines = [line.encode() for line in lines]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._lines)


def _stream(monkeypatch, s, lines):
    monkeypatch.setattr(s.urllib.request, "urlopen", lambda req, timeout = 0: _FakeStream(lines))
    return s._stream_events("http://x", "/v1/chat/completions", {"stream": True}, "t")


def test_a_stream_that_errors_after_the_tool_ran_is_not_a_success(monkeypatch):
    """A failure after the status line went out arrives in band as an error
    frame with the 200 kept, and the stream ends without [DONE]; the tool_end
    before it must not carry the turn."""
    s = _load_scenario()
    tool_end = 'data: {"type": "tool_end", "tool_name": "web_search", "result": "found"}\n'
    status, events, error = _stream(
        monkeypatch,
        s,
        [tool_end, 'data: {"error": {"message": "llama-server exited", "type": "server_error"}}\n'],
    )
    assert status == 200 and error and "llama-server exited" in error
    status, events, error = _stream(monkeypatch, s, [tool_end])
    assert error and "without [DONE]" in error
    status, events, error = _stream(monkeypatch, s, [tool_end, "data: [DONE]\n"])
    assert error is None and len(events) == 1
    monkeypatch.setattr(
        s,
        "_stream_events",
        lambda *a, **k: (
            200,
            [{"type": "tool_end", "tool_name": "web_search", "result": "found"}],
            "stream error: llama-server exited",
        ),
    )
    turn = s.chat("http://x", "t", "m", "look it up", tools = True)
    assert turn["ok"] is False and turn["tools_run"] == ["web_search"] and "exited" in turn["error"]


def test_the_powershell_probe_fails_closed_on_the_log_and_finishes_the_policy_refresh():
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    # A channel that cannot be enabled fails prepare; there is no evidence without it.
    assert "could not configure the CodeIntegrity log" not in ps1
    assert "is still disabled after wevtutil sl" in ps1
    # revert refreshes the policy set in every branch, absent file included, and
    # clears AuditPolicyApplied only after the refresh succeeded.
    block = ps1[
        ps1.index("Write-Section 'Remove audit policy'") : ps1.index(
            "Write-Section 'Restore CodeIntegrity log'"
        )
    ]
    assert block.count("Invoke-Native 'CiTool.exe' @('-r')") == 1
    assert block.index("audit policy file already absent") < block.index(
        "Invoke-Native 'CiTool.exe' @('-r')"
    )
    assert block.index("Invoke-Native 'CiTool.exe' @('-r')") < block.index(
        "$baseline.AuditPolicyApplied = $false"
    )
    assert block.index("Dismount-Efi $mounted") < block.index(
        "$baseline.AuditPolicyApplied = $false"
    )
    # Defender detections are the probe window's only.
    assert "Where-Object { $_.InitialDetectionTime -ge $start }" in ps1


def test_a_tool_end_the_loop_closed_without_running_is_not_an_execution():
    """studio_tool_loop.py closes a truncated, cancelled, disabled or
    budget-exhausted call with a non-empty result that is neither the refusal
    nor an Error: string."""
    s = _load_scenario()
    for result in (
        "Unsloth did not execute this tool call because the provider stopped mid-call at its output limit.",
        "Unsloth stopped this tool call before it returned, so there is no result. The tool may have already done part of its work.",
        "Unsloth did not execute this tool call because the tool is disabled.",
        "Unsloth did not run this call because an identical one had already completed.",
    ):
        assert s.tool_end_failure(result) is not None, result
    assert s.tool_end_failure("Unsloth Studio docs: https://docs.unsloth.ai") is None


def test_studio_logs_reach_the_evidence_only_through_the_backend_redactor(tmp_path):
    """The zip is attached to an issue. A PowerShell port of the redaction
    rules drifted from the canonical suite within one review round, so the
    copy runs Studio's own utils.log_redaction under the managed interpreter."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    assert "Copy-Item -LiteralPath $studioLogs" not in ps1 and "Redact-Secrets" not in ps1
    assert (
        "redact_logs.py" in ps1
        and "$python = Get-StudioPython" in ps1[ps1.index("function Invoke-Collect") :]
    )
    assert "no managed interpreter to run the redactor" in ps1
    # The helper, driven against this checkout's backend on the canonical cases.
    src = tmp_path / "logs"
    src.mkdir()
    lines = [
        "Downloading with token hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345",
        "OPENAI_API_KEY=opaquevalue123456",
        'password="correct horse battery staple"',
        'llama-server --api-key "abcdef ghijklmnop" --port 8080',
        "Authorization: Basic dXNlcm5hbWU6c3VwZXJzZWNyZXQ=",
        "Cookie: unsloth_session=8f3c9d1ab77e4f0a9c2b3d4e",
        "n_tokens = 4096",
    ]
    (src / "studio.log").write_text("\n".join(lines), encoding = "utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(PROBE_DIR / "redact_logs.py"),
            str(src),
            str(tmp_path / "out"),
            "--backend",
            str(REPO_ROOT / "studio" / "backend"),
        ],
        capture_output = True,
        text = True,
        timeout = 120,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    out = (tmp_path / "out" / "studio.log").read_text(encoding = "utf-8")
    for secret in (
        "hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345",
        "opaquevalue123456",
        "horse battery staple",
        "abcdef ghijklmnop",
        "dXNlcm5hbWU6c3VwZXJzZWNyZXQ=",
        "8f3c9d1ab77e4f0a9c2b3d4e",
    ):
        assert secret not in out, secret
    assert "n_tokens = 4096" in out


def test_the_powershell_probe_rejects_empty_inventories_and_keeps_reverting():
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    assert "no PE files found under $llamaDir" in ps1
    revert = ps1[ps1.index("function Invoke-Revert") :]
    assert "$policyError = $_" in revert
    assert revert.index("$policyError = $_") < revert.index(
        "Write-Section 'Restore CodeIntegrity log'"
    )
    assert revert.index("Write-Section 'Restore Defender preferences'") < revert.index(
        "if ($null -ne $policyError) {"
    )
    assert "the audit policy is still applied" in revert


def test_rollback_state_is_persisted_before_the_efi_partition_changes_and_stays_out_of_the_zip():
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    block = ps1[
        ps1.index("Write-Section 'Audit policy'") : ps1.index(
            "Write-Section 'Policy state after applying'"
        )
    ]
    persist = block.index("$baseline.AuditPolicyApplied = $true")
    assert block.index("Copy-Item -LiteralPath $NOISG_DEST -Destination $ROLLBACK_POLICY") < persist
    assert (
        persist
        < block.index("Set-Content -LiteralPath $baselinePath", persist)
        < block.index("Copy-Item -LiteralPath $AuditPolicy -Destination $NOISG_DEST")
    )
    assert "Join-Path (Join-Path $dir 'rollback') 'preexisting-policy.cip'" in ps1
    assert "Compress-Archive -Path (Join-Path $dir '*')" not in ps1
    assert "if ($rel -like 'rollback\\*') { continue }" in ps1
    assert "$saved = $ROLLBACK_POLICY" in ps1[ps1.index("function Invoke-Revert") :]


def test_the_inventory_is_of_the_runtime_studio_resolved(tmp_path, monkeypatch):
    """A folder selected in Studio's settings, or LLAMA_SERVER_PATH, wins over
    the managed default in Studio; an inventory of the default would then hash
    a different build than the scenario drove."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    llama = ps1[ps1.index("function Get-LlamaDir") : ps1.index("function Resolve-LlamaDir")]
    assert llama.index("$env:LLAMA_SERVER_PATH") < llama.index("$env:UNSLOTH_LLAMA_CPP_PATH")
    resolve = ps1[ps1.index("function Resolve-LlamaDir") : ps1.index("function Invoke-Native")]
    assert "runtime-selection.json" in resolve and "$sel.resolved_binary" in resolve
    run = ps1[ps1.index("function Invoke-Run") : ps1.index("function Invoke-Collect")]
    assert run.index("& python @scenarioArgs") < run.index("$llamaDir = Resolve-LlamaDir $dir")
    s = _load_scenario()
    seen: list[str] = []

    def fake(
        base_url,
        method,
        path,
        payload = None,
        token = None,
        timeout = 900,
    ):
        seen.append(path)
        if path == "/api/auth/login":
            return 200, {"access_token": "tok"}
        if path == "/api/settings/llama-cpp-path":
            return 200, {
                "path": "D:\\llama",
                "source": "studio",
                "resolved_binary": "D:\\llama\\llama-server.exe",
            }
        return 200, {"status": "done"}

    monkeypatch.setattr(s, "_request", fake)
    monkeypatch.setattr(s, "_stream_events", lambda *a, **k: (200, [], None))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "studio_scenario.py",
            "--model",
            "m",
            "--out",
            str(tmp_path),
            "--port",
            "1",
            "--password",
            "pw",
            "--home",
            str(tmp_path),
            "--poll-seconds",
            "0.05",
        ],
    )
    s.main()
    sel = json.loads((tmp_path / "runtime-selection.json").read_text(encoding = "utf-8"))
    assert sel["resolved_binary"].endswith("llama-server.exe") and sel["source"] == "studio"
    assert seen.index("/api/settings/llama-cpp-path") < seen.index("/api/inference/load")
