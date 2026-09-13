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
            # The identity marker the real route publishes: discover_port
            # refuses a stranger answering this path, since the next thing it
            # does with the port is post the Studio password to it.
            return 200, {"status": "alive", "service": "Unsloth UI Backend"}
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
            # The identity marker the real route publishes: discover_port
            # refuses a stranger answering this path, since the next thing it
            # does with the port is post the Studio password to it.
            return 200, {"status": "alive", "service": "Unsloth UI Backend"}
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
    # Each Defender preference is raised, and restored, on its own.
    assert "foreach ($r in $restores)" in ps1 and ps1.count("Set-MpPreference @params") == 2


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
    """Every published Windows asset, enumerated from the release.

    This used to name five literal patterns, which sampled one CUDA profile out
    of seven and one ROCm out of seven. Each profile is packaged separately, so
    the audit could pass while the bundle a modern NVIDIA machine actually
    selects shipped unsigned. Enumerating means a profile added later is covered
    without anyone editing the workflow, so the guard is that nothing narrows it
    back to a literal list.
    """
    body = WORKFLOW.read_text(encoding = "utf-8")
    assert "gh release view $tag --repo unslothai/llama.cpp --json assets" in body
    assert "Where-Object { $_ -like '*windows*' -and $_ -like '*.zip' }" in body
    assert 'throw "no Windows assets on $tag"' in body
    # A hard coded profile would silently shrink the set again.
    for sampled in (
        "*windows-x64-cuda12-legacy.zip",
        "*windows-x64-rocm-gfx110X.zip",
        "*windows-arm64-cpu.zip",
    ):
        assert sampled not in body, sampled
    # Every enumerated asset has to end up inventoried.
    assert "enumerated $($assets.Count) Windows asset(s) but inventoried $audited" in body


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
    download = body.index("gh release download $env:AUDIT_TAG")
    loop_end = body.index("$audited = @($rows | Group-Object Bundle).Count")
    block = body[download:loop_end]
    # Checked per asset, inside the loop, not once after it.
    assert 'Write-Host "::error::gh release download exited $LASTEXITCODE for $asset"' in block
    assert 'Write-Host "::error::$asset did not download"' in block
    # And the bundle has to actually be on disk afterwards.
    assert "Get-Item (Join-Path 'bundles' $asset)" in block


def test_the_signature_audit_streams_one_bundle_at_a_time():
    """Seventeen Windows assets are several GB; the hosted runner cannot hold
    them all at once, which is why the audit used to sample instead."""
    body = WORKFLOW.read_text(encoding = "utf-8")
    inventory = body.index("- name: Inventory Authenticode signatures")
    block = body[inventory:]
    assert "Remove-Item $zip.FullName -Force" in block, "the bundle is never deleted"
    assert block.index("Remove-Item $zip.FullName -Force") < block.index(
        "$audited = @($rows | Group-Object Bundle).Count"
    ), "the delete must be inside the per-asset loop"


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
        and "$python = Resolve-StudioPythonFor $dir"
        in ps1[ps1.index("function Invoke-Collect") :]
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
    assert "no PE files found under ${llamaDir}:" in ps1
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
    assert "if ($rel -like 'rollback\\*' -or $rel -like 'raw-logs\\*') { continue }" in ps1
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
        if path == "/api/liveness":
            return 200, {"status": "alive", "service": "Unsloth UI Backend"}
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


def test_prepare_restarts_a_running_studio_and_only_prepare_may():
    """Studio's startup is where the venv's native modules load; a process that
    was already up loaded them before the window and the policy existed."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    assert "function Stop-Studio" in ps1
    init = ps1[ps1.index("function Initialize-Studio") : ps1.index("function Save-Baseline")]
    answering = init.index("if (Test-StudioResponding $Port) {")
    assert init.index("if (-not $allowInstall) {", answering) < init.index(
        "Stop-Studio $Port", answering
    )
    stop = ps1[ps1.index("function Stop-Studio") : ps1.index("function Initialize-Studio")]
    assert "ParentProcessId = $owner" in stop, "children (llama-server, workers) are stopped first"


def test_redirected_studio_output_and_the_scenario_console_are_redacted_too():
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    assert "$startLog = Join-Path (Join-Path $dir 'raw-logs') 'studio-start.log'" in ps1
    assert "Start-Studio $python $Port $startLog" in ps1
    assert "$log = Join-Path (Join-Path $dir 'raw-logs') 'studio-scenario.log'" in ps1
    assert "if ($rel -like 'rollback\\*' -or $rel -like 'raw-logs\\*') { continue }" in ps1
    collect = ps1[ps1.index("function Invoke-Collect") : ps1.index("function Invoke-Revert")]
    assert (
        "$rawLogs = Join-Path $dir 'raw-logs'" in collect
        and "foreach ($source in $sources)" in collect
    )


def test_sample_submission_is_opt_in():
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    assert "[switch] $SendSamples" in ps1
    raise_block = ps1[
        ps1.index("Write-Section 'Raise security settings'") : ps1.index(
            "Write-Section 'CodeIntegrity log'"
        )
    ]
    assert raise_block.index("if ($SendSamples) {") < raise_block.index(
        "$wanted['SubmitSamplesConsent'] = 'SendAllSamples'"
    )
    assert raise_block.count("SubmitSamplesConsent") == 1


def test_collect_judges_the_load_step_not_the_exit_code():
    """A failed search, chat or unload after a working load is still a valid
    load-time measurement; calling it a null result throws the evidence away."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    collect = ps1[ps1.index("function Invoke-Collect") : ps1.index("function Invoke-Revert")]
    assert "$loadOk = [bool]$results.steps.load.ok" in collect
    assert "if ($loadOk) {" in collect and "later scenario step(s) failed" in collect
    assert "did NOT load a model" in collect


def test_the_venv_inventory_comes_from_the_running_interpreter_and_cannot_be_empty():
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    assert "function Resolve-VenvDir" in ps1
    assert "$venvDir = Resolve-VenvDir" in ps1
    assert "no PE files found under $venvDir" in ps1
    # The event scoping in collect must resolve the venv the same way the
    # inventory does, or a custom-home venv is counted as somebody else's - and
    # it reads run's recorded answer rather than resolving it again, since the
    # override may only ever have been set in the shell that ran run.
    collect = ps1[ps1.index("function Invoke-Collect") : ps1.index("function Invoke-Revert")]
    assert "$venvTail = " in collect and "Get-ScopeTail (Resolve-VenvDir $dir)" in collect
    assert "$VENV_DIR -replace" not in collect


def test_the_readme_clones_a_durable_ref():
    body = (PROBE_DIR / "README.md").read_text(encoding = "utf-8")
    assert "--branch windows-sac-probe" not in body
    assert "git clone --depth 1 https://github.com/unslothai/unsloth" in body
    assert "-SendSamples" in body


def test_the_app_control_audit_keeps_its_positive_control():
    """No 3076 events must never be reportable as a pass on its own.

    A signature-only policy that failed to load produces exactly the same empty
    event set as a bundle Windows is happy with, so the job builds an unsigned
    binary, runs it, and requires that it be flagged. The verdict step is gated
    on that control having fired, which is the whole reason the result means
    anything; a change that drops either half turns the job into a green tick
    that proves nothing.
    """
    import yaml

    body = WORKFLOW.read_text(encoding = "utf-8")
    parsed = yaml.safe_load(body)
    job = parsed["jobs"]["code-integrity"]
    steps = {s.get("name", ""): s for s in job["steps"]}

    control = next(n for n in steps if n.startswith("Positive control"))
    assert "control_fired=true" in steps[control]["run"]
    assert "control_fired=false" in steps[control]["run"]

    # The gate itself. Without it a runner where the policy never loaded reports
    # a clean audit.
    assert steps["Verdict"]["if"] == "always() && steps.control.outputs.control_fired == 'true'"

    # Applying a policy is not the same as it being active, and the job must
    # refuse rather than assume.
    apply_step = next(steps[n] for n in steps if n.startswith("Apply the policy"))
    assert "not in the active policy set after refresh" in apply_step["run"]

    # Audit, never enforcement: an enforcing policy could brick the runner.
    assert "SmartAppControlAuditNoISG" in body
    assert "--remove-policy" in body


def test_the_inventory_root_is_the_selected_runtime_not_the_binary_directory():
    """<root>\\build\\bin\\Release is a supported layout
    (llama_cpp_path_settings.llama_server_candidates), so taking the parent of
    resolved_binary inventoried Release\\ alone and scoped every sibling PE
    under the selected root out into 'other'."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    resolve = ps1[ps1.index("function Resolve-LlamaDir") : ps1.index("$PE_EXT = ")]
    assert resolve.index("Test-Path -LiteralPath $sel.path -PathType Container") < resolve.index(
        "return (Split-Path -Parent $sel.resolved_binary)"
    ), "the selected root wins; the binary's parent is only the direct-binary fallback"
    assert "return $sel.path" in resolve


def test_event_scoping_matches_path_tails_literally():
    """-like reads [ and ] as pattern syntax, and a directory called [llama] is
    legal and supported (tests/test_installer_system32_guard.py), so an
    unescaped tail matched none of its events and dropped its 3076/3077 records
    out of the Unsloth headline."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    collect = ps1[ps1.index("function Invoke-Collect") : ps1.index("function Invoke-Revert")]
    for name in ("$tail = ", "$venvTail = "):
        line = collect[collect.index(name) : collect.index(name) + 200]
        assert "Get-ScopeTail" in line, name
    assert "WildcardPattern]::Escape($trimmed) + '\\'" in ps1
    workflow = WORKFLOW.read_text(encoding = "utf-8")
    verdict = workflow[workflow.index("$dir = $env:RUNTIME_DIR") :]
    assert "WildcardPattern]::Escape" in verdict
    assert "-replace '^[A-Za-z]:', ''" in verdict, "device-form paths are matched on the tail"
    assert '$_.Message -like "*$dir*"' not in workflow


def test_installer_trees_under_a_custom_studio_home_are_recorded_for_revert():
    """setup.ps1 resolves $NodeParent from UNSLOTH_STUDIO_HOME/STUDIO_HOME and
    $UnslothHome from the parent of the managed llama.cpp dir, so hard coding
    %USERPROFILE%\\.unsloth left the custom-home trees administrator-owned with
    no ACL repair."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    init = ps1[ps1.index("function Initialize-Studio") : ps1.index("function Save-Baseline")]
    assert "$installRoots = @($unslothHome, $override, (Split-Path -Parent (Get-LlamaDir)))" in init
    for tree in ("'node'", "'whisper.cpp'", "'.cache'"):
        assert f"(Join-Path $_ {tree})" in init
        assert f"(Join-Path $unslothHome {tree})" not in init


def test_trees_the_installer_created_are_recorded_even_when_the_install_fails():
    """A run that created node\\ or .cache\\ and then died before the managed
    interpreter existed still left administrator-owned trees; returning early
    left StudioInstalledByProbe false and revert skipped their ACL repair."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    init = ps1[ps1.index("function Initialize-Studio") : ps1.index("function Save-Baseline")]
    assert init.index("$created = @($absentBefore") < init.index(
        "Write-Warning 'Studio still not found after the installer ran."
    )
    assert init.index("$b.StudioInstallRoots = @(") < init.index(
        "Write-Warning 'Studio still not found after the installer ran."
    )


def test_prepare_fails_when_a_running_studio_cannot_be_restarted():
    """The existing process loaded its venv native modules before the policy and
    the window existed, and run deliberately does not restart a responsive
    Studio, so a silent return let prepare report completion for a cell whose
    startup loads can never be observed."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    init = ps1[ps1.index("function Initialize-Studio") : ps1.index("function Save-Baseline")]
    assert "if (-not (Stop-Studio $Port)) { return }" not in init
    stop = init[init.index("if (-not (Stop-Studio $Port))") :]
    assert stop[: stop.index("}")].count("throw") == 1


def test_each_defender_preference_is_applied_on_its_own_and_deviations_recorded():
    """One shared try meant the first policy-controlled setting threw and every
    later Set-MpPreference was never called, so the cell ran without the
    documented cloud-block and PUA baseline with nothing in the evidence."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    raise_block = ps1[
        ps1.index("Write-Section 'Raise security settings'") : ps1.index(
            "Write-Section 'CodeIntegrity log'"
        )
    ]
    assert "foreach ($name in $wanted.Keys)" in raise_block
    for name in ("DisableRealtimeMonitoring", "MAPSReporting", "CloudBlockLevel", "PUAProtection"):
        assert name in raise_block
    # Written when a setting did not take, and cleared on the success path, so a
    # retry of the label cannot leave a marker contradicting its own evidence.
    assert "$mpErrorPath = Join-Path $dir 'defender-preference-errors.txt'" in raise_block
    assert "Remove-Item -LiteralPath $mpErrorPath" in raise_block
    stale = ps1[ps1.index("foreach ($stale in @(") :]
    assert "events-collection-error.txt" in stale[: stale.index(")) {")]


def test_an_efi_dismount_failure_is_never_reported_as_a_completed_stage():
    """The next stage's Mount-Efi finds the EFI tree already there, returns
    $false and so never retries the unmount, leaving S: exposed for good."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    dismount = ps1[ps1.index("function Dismount-Efi") : ps1.index("function Test-PolicyActive")]
    assert "$script:EfiStillMounted = $true" in dismount
    prepare = ps1[
        ps1.index("function Invoke-Prepare") : ps1.index("function Get-SignatureInventory")
    ]
    assert "if ($script:EfiStillMounted) {" in prepare
    revert = ps1[ps1.index("function Invoke-Revert") :]
    assert "-or $script:EfiStillMounted" in revert


def test_revert_exits_nonzero_when_a_studio_tree_acl_repair_failed():
    """The ACL loop sat outside the aggregate failure accounting, so revert
    printed 'revert complete' and exited zero while the user's own Studio tree
    was still unreadable."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    revert = ps1[ps1.index("function Invoke-Revert") :]
    assert revert.count("$aclFailures++") == 2, "both the nonzero icacls and the catch count"
    assert "$aclFailures -gt 0" in revert
    assert revert.index("$aclFailures = $rejected.Count") < revert.index("$aclFailures -gt 0")


def test_a_partial_inventory_says_so_in_the_evidence():
    """Get-ChildItem discarded the error for a subtree it could not read and
    the caller only rejects a completely empty result, so a partial inventory
    was described as every PE under the tree."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    assert "-ErrorAction SilentlyContinue -ErrorVariable enumErrors" in ps1
    assert "$script:InventoryErrors += " in ps1
    run = ps1[ps1.index("function Invoke-Run") : ps1.index("function Invoke-Collect")]
    assert "inventory-enumeration-errors.txt" in run
    assert "$script:InventoryErrors.Count -gt 0" in run


def test_a_partial_collection_is_marked_inside_the_zip():
    """A console warning does not travel with the archive, so a zip missing the
    raw evtx or a core artifact read as a complete evidence package."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    collect = ps1[ps1.index("function Invoke-Collect") : ps1.index("function Invoke-Revert")]
    # evtx export, staging, redaction failure, no-interpreter, an unverified
    # empty window, one that no policy could have filled, one Smart App Control
    # was never shown to be refusing code in, and a scenario that loaded
    # nothing: every path that would let the archive be read as more than it is
    # records itself.
    assert collect.count("$collectionProblems += ") == 8
    redact = collect[collect.index("$redactor = Join-Path") :]
    assert redact.index('$collectionProblems += "log redaction failed') < redact.index(
        "Remove-Item -LiteralPath (Join-Path $dir 'studio-logs')"
    ), "recorded before the partial redacted output is deleted"
    assert "no managed interpreter to run the redactor, so studio-logs" in collect
    # Written into the STAGED tree, after the copy loop, or it never reaches the zip.
    warn = collect.index("Set-Content -LiteralPath (Join-Path $stage 'collection-warnings.txt')")
    assert collect.index("foreach ($item in Get-ChildItem -LiteralPath $dir -Recurse -File)") < warn
    assert warn < collect.index("Compress-Archive -Path (Join-Path $stage '*')")
    # And a retry that succeeded clears the failed attempt's marker.
    assert collect.index(
        "Remove-Item -LiteralPath (Join-Path $dir 'events-collection-error.txt')"
    ) > collect.index("could not read $CI_LOG, so the event window was not collected")


def test_the_app_control_verdict_cannot_pass_on_an_unread_channel():
    """SilentlyContinue turned a failed query into an empty array and the job
    reported that no runtime binary would be refused."""
    workflow = WORKFLOW.read_text(encoding = "utf-8")
    verdict = workflow[workflow.index("      - name: Verdict\n        if: always()") :]
    verdict = verdict[: verdict.index("      - name: Export the CodeIntegrity events")]
    assert "-ErrorAction Stop" in verdict
    assert "NoMatchingEventsFound" in verdict
    assert "::error::could not read the CodeIntegrity channel" in verdict


def test_the_code_integrity_artifact_carries_the_events_it_is_named_for():
    workflow = WORKFLOW.read_text(encoding = "utf-8")
    assert "- name: Export the CodeIntegrity events" in workflow
    upload = workflow[workflow.index("name: code-integrity-events") :]
    assert "code-integrity-events.json" in upload
    assert "CodeIntegrity-Operational.evtx" in upload
    # Unconditional: a control that did not fire is when the records matter most.
    export = workflow[workflow.index("- name: Export the CodeIntegrity events") :]
    assert export[: export.index("run: |")].count("if: always()") == 1
    assert "control_fired" not in export[: export.index("run: |")]


def test_the_audit_channel_resize_is_verified_not_assumed():
    """A channel that was already enabled reads back enabled even when the
    resize failed, and the 1 MB default can wrap between the positive control
    and the verdict."""
    workflow = WORKFLOW.read_text(encoding = "utf-8")
    step = workflow[workflow.index("- name: Require a runner that can host a policy") :]
    step = step[: step.index("- name: Fetch the Smart App Control audit policies")]
    assert 'if ($LASTEXITCODE -ne 0) { throw "wevtutil sl $log exited' in step
    assert "$maxSize -lt 67108864" in step


def test_acl_repair_reaches_a_custom_home_outside_the_user_profile():
    """An UNSLOTH_STUDIO_HOME on another volume (D:\\Unsloth) is exactly where
    the elevated installer leaves administrator-owned trees, and anchoring
    containment on %USERPROFILE% discarded every one of them."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    revert = ps1[ps1.index("function Invoke-Revert") :]
    assert "$profileRoot" not in revert
    assert "$allowedRoots = @(" in revert
    roots = revert[revert.index("$allowedRoots = @(") :]
    roots = roots[: roots.index("$recorded = @()")]
    for source in ("$env:USERPROFILE", "$override", "(Get-StudioHome)", "(Get-LlamaDir)"):
        assert source in roots
    # Still two gates: prepare recorded it, and it resolves under a live root.
    assert (
        "if ($baseline.StudioInstalledByProbe) { $recorded = @($baseline.StudioInstallRoots) }"
        in revert
    )
    assert "$full.StartsWith($_ + '\\', [StringComparison]::OrdinalIgnoreCase)" in revert


def test_prepare_proves_the_audit_policy_actually_evaluates_loads():
    """Being in CiTool's policy list is not evidence of evaluation: a policy
    that loaded and evaluates nothing produces a window with no 3076 in it,
    which reads exactly like a clean allow. The CI job runs the same control."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    fn = ps1[
        ps1.index("function Test-AuditPolicyEvaluating") : ps1.index("function Invoke-Prepare")
    ]
    assert "-OutputType ConsoleApplication" in fn
    # An enforcing machine refuses the control outright: that 3077 is stronger
    # evidence of evaluation than the 3076 an audit-only machine produces, and
    # under an installed audit policy either answers the question.
    assert "function Test-AuditPolicyEvaluating([int[]] $AcceptIds = @(3076, 3077)) {" in fn
    assert "$AcceptIds -contains $_.Id" in fn
    # Polled, not slept once, and never staged into the evidence.
    assert "foreach ($attempt in 1..10)" in fn
    assert "Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue" in fn
    assert '$dir = Join-Path $WorkDir ".control-$Label"' in fn

    prepare = ps1[
        ps1.index("function Invoke-Prepare") : ps1.index("function Get-SignatureInventory")
    ]
    assert prepare.index("-not (Test-PolicyActive $NOISG_GUID)") < prepare.index(
        "$controlFired = Test-AuditPolicyEvaluating"
    ), "the control runs only once the policy is verified to be in the active set"
    # $null (no control could be built) is not a failure; $false is.
    assert "if ($false -eq $controlFired) {" in prepare
    assert "$baseline.AuditPolicyControlFired = $controlFired" in prepare
    assert "AuditPolicyControlFired = $null" in ps1

    # And collect refuses to read an unverified empty window as an allow.
    collect = ps1[ps1.index("function Invoke-Collect") : ps1.index("function Invoke-Revert")]
    assert "$true -ne $b.AuditPolicyControlFired" in collect
    assert "NULL result, not an allow" in collect


def test_only_a_3076_or_3077_counts_as_a_verdict_for_the_null_result_warning():
    """3033 and 3090-3092 are context. Keying the warning on the total scoped
    count meant one scoped allow-and-origin record presented an unverified
    window, holding no verdict at all, as an ordinary result."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    collect = ps1[ps1.index("function Invoke-Collect") : ps1.index("function Invoke-Revert")]
    assert "$verdicts = $blocks + $audits" in collect
    assert "if ($verdicts -eq 0 -and (Test-Path -LiteralPath $baselineForControl))" in collect
    assert "if ($ours.Count -eq 0 -and (Test-Path -LiteralPath $baselineForControl))" not in collect


def test_a_completed_revert_spends_its_baseline():
    """revert leaves baseline.json behind, so a later prepare on the same label
    reused the first run's snapshot and a second revert would write those stale
    Defender values back over whatever the machine legitimately carries now."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    assert "RevertCompletedAt       = $null" in ps1
    prepare = ps1[
        ps1.index("function Invoke-Prepare") : ps1.index("function Get-SignatureInventory")
    ]
    assert "if ($previous -and -not $previous.RevertCompletedAt) {" in prepare
    assert "$baseline = Save-Baseline $dir" in prepare

    revert = ps1[ps1.index("function Invoke-Revert") :]
    assert "-NotePropertyName RevertCompletedAt" in revert
    # Past every failure check: a partial revert still needs its baseline.
    assert revert.index("revert did not fully restore this machine") < revert.index(
        "-NotePropertyName RevertCompletedAt"
    )


def test_the_app_control_verdict_polls_before_reporting_an_allow():
    """3076 delivery is asynchronous - the positive control polls for exactly
    that reason - and the exercise step ends the moment llama-server exits, so
    one query could run before the record landed and report a clean allow."""
    workflow = WORKFLOW.read_text(encoding = "utf-8")
    verdict = workflow[workflow.index("      - name: Verdict\n        if: always()") :]
    verdict = verdict[: verdict.index("      - name: Export the CodeIntegrity events")]
    assert "foreach ($attempt in 1..10) {" in verdict
    assert "if ($ours.Count -gt 0) { break }" in verdict
    assert "Start-Sleep -Seconds 3" in verdict
    # The scoping still happens inside the loop, or polling proves nothing.
    assert verdict.index("foreach ($attempt in 1..10) {") < verdict.index(
        '$_.Message -like "*$tail*"'
    )


def test_the_scenario_password_is_cleared_even_when_the_run_is_interrupted():
    """Ctrl+C stops the pipeline without entering the normal path or the catch,
    and the script runs in the operator's own session, so the secret stayed in
    that console and was inherited by everything started from it afterwards."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    run = ps1[ps1.index("function Invoke-Run") : ps1.index("function Invoke-Collect")]
    clear = "Remove-Item Env:\\SAC_PROBE_STUDIO_PASSWORD -ErrorAction SilentlyContinue"
    assert run.count(clear) == 1
    assert run.index("} finally {") < run.index(clear)


def test_a_custom_studio_home_is_normalized_the_way_studio_normalizes_it(monkeypatch):
    """UNSLOTH_STUDIO_HOME=~\\my-studio is supported (storage_roots.studio_root
    calls expanduser().resolve()), but the raw string is a cwd-relative
    directory named '~', so auth/.bootstrap_password was read from the wrong
    place and a never-opened Studio logged in with the wrong password."""
    s = _load_scenario()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", "  ~/my-studio  ")
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    assert s.resolve_studio_home(s.default_studio_home()) == Path.home() / "my-studio"

    # Whitespace alone is unset, exactly as Studio treats it.
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", "   ")
    assert s.default_studio_home() == str(Path.home() / ".unsloth" / "studio")

    # The PowerShell half resolves the same override the same way, or the venv
    # is inventoried in one place and the event scoping matches no event at all.
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    assert "function Resolve-ConfiguredPath" in ps1
    assert "function Get-StudioHomeOverride" in ps1
    assert "$override = if ($env:UNSLOTH_STUDIO_HOME)" not in ps1
    home = ps1[ps1.index("function Get-StudioHome {") : ps1.index("function Get-LlamaDir")]
    assert "$override = Get-StudioHomeOverride" in home


def test_a_prefix_colliding_sibling_is_not_scoped_as_the_selected_runtime():
    """The channel is machine-wide and the matrix runs two builds side by side,
    so ...\\llama.cpp-b10830 sitting beside the selected ...\\llama.cpp had its
    3076/3077 counted as a verdict on the build the scenario actually drove."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    fn = ps1[ps1.index("function Get-ScopeTail") : ps1.index("function Get-EventDataMap")]
    assert "TrimEnd('\\', '/')" in fn
    assert "WildcardPattern]::Escape($trimmed) + '\\'" in fn
    workflow = WORKFLOW.read_text(encoding = "utf-8")
    verdict = workflow[workflow.index("$dir = $env:RUNTIME_DIR") :]
    assert "($dir -replace '^[A-Za-z]:', '').TrimEnd('\\')) + '\\'" in verdict


def test_a_window_with_no_audit_policy_is_a_null_result_unless_sac_enforces():
    """-AuditPolicy is optional. Without it, an off machine evaluates nothing
    and the policy Smart App Control runs in evaluation mode does not log audit
    events to this channel, so only genuine enforcement can produce a verdict
    and every other empty window is inconclusive rather than an allow."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    collect = ps1[ps1.index("function Invoke-Collect") : ps1.index("function Invoke-Revert")]
    assert "$sacMode = [string]$sacNow.Mode" in collect
    assert "$sacAtPrepare = [string]$b.Sac.Mode" in collect
    branch = "} elseif ($sacMode -ne 'enforcement' -or $sacAtPrepare -ne 'enforcement') {"
    assert branch in collect
    guard = collect[collect.index(branch) + len(branch) :]
    guard = guard[: guard.index("} elseif ")]
    # In the zip, not only on a console the reader never sees.
    assert "$collectionProblems +=" in guard
    assert "NULL result, not an allow" in guard


def test_an_empty_window_with_no_observed_load_never_reads_as_an_allow():
    """A scenario that stopped at login loaded no PE under the policy, so the
    positive control alone must not earn the clean-allow line - and the caveat
    used to be worked out after the zip was written, so the archive said
    nothing about it."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    collect = ps1[ps1.index("function Invoke-Collect") : ps1.index("function Invoke-Revert")]
    assert "$scenarioProblem = $null" in collect
    assert "if ($scenarioProblem) { $collectionProblems += $scenarioProblem }" in collect
    assert "} elseif (-not $loadOk) {" in collect
    # Computed before the staging loop, or it cannot reach the archive.
    assert collect.index(
        "if ($scenarioProblem) { $collectionProblems += $scenarioProblem }"
    ) < collect.index("$stage = Join-Path $WorkDir")
    # ... and before the branch that would otherwise report an allow.
    assert collect.index("$scenarioProblem = $null") < collect.index(
        "$verdicts = $blocks + $audits"
    )


def test_a_studio_that_never_answered_fails_prepare():
    """Start-Studio's result was discarded, so prepare printed 'prepare
    complete' over a Studio that never started and whose venv modules
    therefore never loaded inside the window."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    init = ps1[ps1.index("function Initialize-Studio") : ps1.index("function Save-Baseline")]
    assert "if (-not (Start-Studio $python $Port $startLog)) {" in init
    assert "| Out-Null\n}" not in init.split("Start-Studio $python $Port $startLog")[-1]
    # prepare fails; run only warns, because its inventories are still evidence.
    assert "if ($allowInstall) {" in init.split("Start-Studio $python $Port $startLog")[1]
    assert 'throw "Studio did not answer on port $Port within 5 minutes' in init


def test_only_studio_is_treated_as_studio_on_the_port():
    """prepare force-stops whatever owns the port, and 8888 is a busy default,
    so a status-only check could kill an unrelated server."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    fn = ps1[ps1.index("function Test-StudioResponding") : ps1.index("function Get-EventDataMap")]
    assert "$body.service -eq 'Unsloth UI Backend'" in fn
    assert "return $r.StatusCode -eq 200" not in fn
    # The field the check reads is the one Studio actually publishes.
    liveness = (REPO_ROOT / "studio" / "backend" / "main.py").read_text(encoding = "utf-8")
    route = liveness[liveness.index('@app.get("/api/liveness")') :][:1200]
    assert '"service": "Unsloth UI Backend"' in route


def test_the_defender_baseline_is_read_back_not_assumed():
    """A tamper-protected or policy-managed preference is ignored rather than
    refused: Set-MpPreference returns without error and the value never
    changes, so the cell was graded under a baseline it never adopted."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    prepare = ps1[
        ps1.index("function Invoke-Prepare") : ps1.index("function Get-SignatureInventory")
    ]
    assert "$applied = Get-MpPreference" in prepare
    assert "but reads back as $actual" in prepare
    # Into the same file the throwing failures use, so one artifact carries
    # every deviation, and before that file is written.
    assert prepare.index("$applied = Get-MpPreference") < prepare.index(
        "$mpErrorPath = Join-Path $dir 'defender-preference-errors.txt'"
    )
    assert '$mpFailed += "${name}: set to $expected but reads back as $actual' in prepare


def test_a_numeric_defender_readback_is_compared_not_waved_through():
    """Get-MpPreference returns CIM numbers on some builds, so MAPSReporting 0
    against a requested Advanced has to compare unequal: that is exactly the
    machine where the request was ignored."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    fn = ps1[
        ps1.index("function Test-MpPreferenceMatch") : ps1.index("function Get-MpPreferenceType")
    ]
    assert "[Enum]::Parse($type, [string]$expected, $true)" in fn
    assert "[Enum]::Parse($type, [string]$actual, $true)" in fn
    # Undecidable is its own answer, never a silent match.
    assert "return $null\n}" in fn
    # From the cmdlet's own metadata, never a hand-written name-to-code table.
    assert "$cmd.Parameters[$name]" in ps1
    prepare = ps1[
        ps1.index("function Invoke-Prepare") : ps1.index("function Get-SignatureInventory")
    ]
    assert (
        "$same = Test-MpPreferenceMatch $actual $expected (Get-MpPreferenceType $name)" in prepare
    )
    assert "} elseif ($null -eq $same) {" in prepare


def test_revert_reads_the_defender_values_back_before_spending_the_baseline():
    """An ignored restore leaves $failed at zero, so revert reported success
    and stamped RevertCompletedAt over a machine still carrying the raised
    settings."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    revert = ps1[ps1.index("function Invoke-Revert") :]
    assert "$restored = Get-MpPreference" in revert
    assert (
        "$same = Test-MpPreferenceMatch $restored.($r.Name) $r.Value (Get-MpPreferenceType $r.Name)"
        in revert
    )
    assert revert.index("$restored = Get-MpPreference") < revert.index("$restoreFailures = $failed")
    assert revert.index("$restored = Get-MpPreference") < revert.index(
        "-NotePropertyName RevertCompletedAt"
    )


def test_install_roots_survive_a_prepare_retry():
    """A retry finds the first attempt's trees already present, so replacing
    the recorded list dropped them and revert left them administrator-owned."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    init = ps1[ps1.index("function Initialize-Studio") : ps1.index("function Save-Baseline")]
    assert "$b.StudioInstallRoots = @(@($b.StudioInstallRoots) + $created |" in init
    assert "$b.StudioInstallRoots = $created" not in init


def test_an_installer_that_produced_no_interpreter_fails_prepare():
    """prepare printed 'prepare complete' with no Studio to start, and the
    operator only found out when run rejected the empty venv."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    init = ps1[ps1.index("function Initialize-Studio") : ps1.index("function Save-Baseline")]
    tail = init[init.index("if (-not $python) {\n            # prepare cannot go on") :]
    assert "if ($allowInstall) {" in tail
    assert "throw 'the installer ran but produced no managed interpreter" in tail
    # run still warns and returns: its inventories are evidence either way.
    assert "Write-Warning 'Studio still not found after the installer ran." in tail


def test_the_ci_verdict_needs_a_runtime_that_was_actually_extracted():
    """The verdict runs on always(), so a download or expand failure left a
    clean signature-only verdict in the job summary for a runtime that was
    never loaded."""
    workflow = WORKFLOW.read_text(encoding = "utf-8")
    verdict = workflow[workflow.index("      - name: Verdict\n        if: always()") :]
    verdict = verdict[: verdict.index("      - name: Export the CodeIntegrity events")]
    assert "if (-not $dir) {" in verdict
    assert "::error::the shipped runtime was never extracted" in verdict
    assert verdict.index("if (-not $dir) {") < verdict.index("foreach ($attempt in 1..10) {")


def test_the_scenario_will_not_hand_its_password_to_an_unidentified_server():
    """discover_port picks the port the operator's Studio password is posted
    to, and these defaults are shared (8888 is Jupyter's), so a catch-all 200
    used to be enough to receive the credential."""
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
        seen.append(base_url)
        if base_url.endswith(":8888"):
            return 200, {"status": "alive"}  # a stranger answering the path
        if base_url.endswith(":8890"):
            return 200, {"status": "alive", "service": "Unsloth UI Backend"}
        return 0, "refused"

    import pytest as _pytest

    monkey = _pytest.MonkeyPatch()
    try:
        monkey.setattr(s, "_request", fake)
        assert s.discover_port(None) == 8890
        with _pytest.raises(SystemExit):
            s.discover_port(8888)
    finally:
        monkey.undo()


def test_an_efi_mount_this_probe_owns_is_retried_by_the_next_stage():
    """$script:EfiStillMounted dies with the process, so the next stage found
    an EFI-backed S:, called it pre-existing and never retried the unmount
    while still reporting success."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    assert "$EFI_OWNED_MARKER = Join-Path $WorkDir '.efi-mounted-by-probe'" in ps1
    mount = ps1[ps1.index("function Mount-Efi") : ps1.index("$script:EfiStillMounted = $false")]
    assert "if (Test-Path -LiteralPath $EFI_OWNED_MARKER) {" in mount
    # Claimed before the pre-existing verdict, and written when we mount.
    assert mount.index("Test-Path -LiteralPath $EFI_OWNED_MARKER") < mount.index("return $false")
    assert "Set-Content -LiteralPath $EFI_OWNED_MARKER" in mount
    dismount = ps1[ps1.index("function Dismount-Efi") : ps1.index("function Test-PolicyActive")]
    assert "Remove-Item -LiteralPath $EFI_OWNED_MARKER" in dismount


def test_a_lost_preexisting_policy_backup_is_a_revert_failure():
    """Falling through to the removal branch deleted an administrator's policy
    and then reported the rollback complete."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    revert = ps1[ps1.index("function Invoke-Revert") :]
    assert (
        "if ($baseline.AuditPolicyPreexisting -and -not (Test-Path -LiteralPath $saved)) {"
        in revert
    )
    guard = revert[revert.index("$baseline.AuditPolicyPreexisting -and -not") :]
    assert guard.index('throw "the baseline says a policy with $NOISG_GUID') < guard.index(
        "Remove-Item -LiteralPath $NOISG_DEST"
    )


def test_prepare_requires_the_log_capacity_it_asked_for():
    """A channel that was already enabled reads back enabled even when the
    resize was refused, and the 1 MB default wraps while every venv load is
    audited: the dropped records read as a clean window."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    prepare = ps1[
        ps1.index("function Invoke-Prepare") : ps1.index("function Get-SignatureInventory")
    ]
    assert "$ciNow.MaxSize -lt 67108864" in prepare
    assert prepare.index("$ciNow.MaxSize -lt 67108864") < prepare.index(
        "Initialize-Studio $dir $true"
    )


def test_run_revalidates_the_policy_and_the_control_after_a_reboot():
    """Only the file on the EFI partition survives a reboot. A policy that did
    not load on the new boot left collect reading the pre-reboot control as
    proof that this window was audited."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    run = ps1[ps1.index("function Invoke-Run") : ps1.index("function Invoke-Collect")]
    assert "if (-not (Test-PolicyActive $NOISG_GUID)) {" in run
    assert "LastBootUpTime" in run
    assert "$controlFired = Test-AuditPolicyEvaluating" in run
    assert "$runBaseline.AuditPolicyControlFired = $controlFired" in run
    # Before anything is measured, not after.
    assert run.index("Test-PolicyActive $NOISG_GUID") < run.index(
        "Write-Section 'Venv signature inventory'"
    )


def test_a_failed_dismount_is_reclaimed_by_the_next_revert():
    """The policy block clears AuditPolicyApplied even when the dismount that
    follows the refresh failed, so the next revert skipped the block, never
    reached Mount-Efi, and stamped the rollback complete with S: still up."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    assert "function Clear-EfiOwnership {" in ps1
    reclaim = ps1[ps1.index("function Clear-EfiOwnership") : ps1.index("function Test-PolicyActive")]
    # Reclaims only; a revert must never mount the partition itself.
    assert "mountvol.exe" not in reclaim
    assert "Dismount-Efi $true" in reclaim
    # And only when S: is still the partition we left there.
    assert "Test-Path -LiteralPath 'S:\\EFI\\Microsoft\\Boot'" in reclaim
    revert = ps1[ps1.index("function Invoke-Revert") :]
    assert "Clear-EfiOwnership" in revert
    # Outside the policy block, and before it: the block is skipped entirely
    # once AuditPolicyApplied is false, which is the state that needs the retry.
    assert revert.index("Clear-EfiOwnership") < revert.index("if ($baseline.AuditPolicyApplied) {")


def test_a_recorded_tree_revert_refuses_to_touch_is_a_rollback_failure():
    """A UNSLOTH_STUDIO_HOME set for the prepare shell only is gone after a
    reboot, so the trees prepare recorded fall outside the roots revert derives.
    Warning and filtering them out left $aclFailures at zero and revert reported
    a machine it had not restored."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    revert = ps1[ps1.index("function Invoke-Revert") :]
    assert "$rejected += $full" in revert
    assert "$aclFailures = $rejected.Count" in revert
    # A recorded tree that no longer exists needs no repair and is not a failure.
    assert "if (-not (Test-Path -LiteralPath $path)) { continue }" in revert
    # ... and "this run did not install Studio" is false when trees were refused.
    assert "if ($trees.Count -eq 0 -and $rejected.Count -eq 0) {" in revert
    assert revert.index("$aclFailures = $rejected.Count") < revert.index("foreach ($tree in $trees)")


def test_an_allow_needs_live_enforcement_and_a_control_on_the_boot_that_measured():
    """Without -AuditPolicy the allow verdict rested on Sac.Mode as prepare read
    it. The probe supports a reboot between the stages, Windows settles the mode
    on the boot path, and the registry value can disagree with the policy set
    that loaded, so that reading says nothing about the window it graded."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    collect = ps1[ps1.index("function Invoke-Collect") : ps1.index("function Invoke-Revert")]
    # Graded on the live state, and on both ends of the window.
    assert "$sacNow = Get-SacState" in collect
    assert "$sacMode = [string]$sacNow.Mode" in collect
    assert "$sacAtPrepare = [string]$b.Sac.Mode" in collect
    assert collect.index("$sacNow = Get-SacState") < collect.index("$sacMode = [string]$sacNow.Mode")
    # Enforcement in the registry is an intent; the refusal is the observation.
    assert "} elseif ($true -ne $b.SacControlFired) {" in collect
    allow = collect.index("so this window is a real allow")
    assert collect.index("} elseif ($true -ne $b.SacControlFired) {") < allow
    # The zip shows the reader the same state the verdict used.
    assert "$sacNow | ConvertTo-Json -Depth 6 |" in collect
    assert "Get-SacState | ConvertTo-Json" not in collect

    ps1_head = ps1[: ps1.index("function Invoke-Run")]
    assert "SacControlFired         = $null" in ps1_head
    prepare = ps1[ps1.index("function Invoke-Prepare") : ps1.index("function Get-SignatureInventory")]
    assert "$sacBefore = Get-SacState" in prepare
    assert "$sacFired = Test-AuditPolicyEvaluating" in prepare
    # Before the window opens, or the control's own 3077 lands in the evidence.
    assert prepare.index("$sacFired = Test-AuditPolicyEvaluating") < prepare.index(
        "'window-start.txt'"
    )
    run = ps1[ps1.index("function Invoke-Run") : ps1.index("function Invoke-Collect")]
    assert "} elseif ([string]$runBaseline.Sac.Mode -eq 'enforcement') {" in run
    assert "if ($sacNow.Mode -ne 'enforcement') {" in run
    assert "($true -ne $runBaseline.SacControlFired)" in run
    assert "-NotePropertyName SacControlFired" in run


def test_the_readme_pins_the_runtime_before_the_window_opens():
    """prepare opens the event window and then installs or restarts Studio, so
    installing the pinned release afterwards put loads from two releases in one
    window, over the one managed directory collect scopes events by."""
    body = (PROBE_DIR / "README.md").read_text(encoding = "utf-8")
    pin = body[body.index("To pin a specific runtime for a cell") : body.index("## Reading the output")]
    assert "before `prepare`" in pin
    assert "-Stage prepare -Label custom-b10715-sac-on" in pin
    # The pinned prepare comes first in the block the operator copies.
    assert pin.index("-Stage prepare -Label custom-b10715-sac-on") < pin.index(
        "-Stage run     -Label custom-b10715-sac-on"
    )
    assert "restart Studio, then:" not in pin
    # And the upstream row carries the same ordering.
    assert "again\nbefore `prepare`" in pin


def test_a_spent_baseline_is_never_written_back_over_the_machine():
    """revert reloaded a baseline carrying RevertCompletedAt and reapplied every
    Defender and channel value in it, silently undoing anything changed since
    the revert that spent it - which for Defender means lowering protections
    nobody asked to lower. prepare already refuses to reuse one."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    revert = ps1[ps1.index("function Invoke-Revert") :]
    assert "if ($baseline.RevertCompletedAt) {" in revert
    guard = revert.index("if ($baseline.RevertCompletedAt) {")
    # Nothing is restored past the guard, but the EFI reclaim still runs before
    # it: an unstamped baseline is an incomplete revert and is still retried.
    assert revert.index("Clear-EfiOwnership") < guard
    assert guard < revert.index("Write-Section 'Restore CodeIntegrity log'")
    assert guard < revert.index("if ($baseline.AuditPolicyApplied) {")


def test_the_sac_positive_control_counts_only_an_enforced_refusal():
    """The audit-policy path wants a 3076 or a 3077; the Smart App Control path
    is testing that unsigned code is REFUSED, and only a 3077 shows that.
    Evaluation mode logs nothing to this channel, so a 3076 there was written by
    some other audit policy and says nothing about enforcement."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    assert "function Test-AuditPolicyEvaluating([int[]] $AcceptIds = @(3076, 3077)) {" in ps1
    fn = ps1[ps1.index("function Test-AuditPolicyEvaluating") : ps1.index("function Invoke-Prepare")]
    assert "$AcceptIds -contains $_.Id" in fn
    assert "$_.Id -eq 3076 -or $_.Id -eq 3077" not in fn
    # Both Smart App Control call sites pin 3077; the audit-policy ones do not.
    assert ps1.count("Test-AuditPolicyEvaluating -AcceptIds @(3077)") == 2
    prepare = ps1[ps1.index("function Invoke-Prepare") : ps1.index("function Get-SignatureInventory")]
    assert "$controlFired = Test-AuditPolicyEvaluating\n" in prepare
    assert "$sacFired = Test-AuditPolicyEvaluating -AcceptIds @(3077)" in prepare
    run = ps1[ps1.index("function Invoke-Run") : ps1.index("function Invoke-Collect")]
    assert "$controlFired = Test-AuditPolicyEvaluating\n" in run
    assert "$sacFired = Test-AuditPolicyEvaluating -AcceptIds @(3077)" in run
    # And neither refusal message still offers a 3076 as proof of enforcement.
    for msg in (prepare, run):
        assert "without raising a 3076 or 3077, so nothing is refusing" not in msg
    assert ps1.count("without being refused with a 3077") == 2


def test_only_the_verified_loopback_listener_is_ever_stopped():
    """Test-StudioResponding verifies http://127.0.0.1:$port, but -LocalPort on
    its own returns every listener holding that port on any local address, and
    this function force-stops each owner and its children. A process on ::1 or a
    LAN address would have been killed as an unverified stranger."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    stop = ps1[ps1.index("function Stop-Studio") : ps1.index("function Initialize-Studio")]
    assert "$_.LocalAddress -eq '127.0.0.1' -or $_.LocalAddress -eq '0.0.0.0'" in stop
    # Filtered before any owner is collected, let alone stopped.
    assert stop.index("LocalAddress") < stop.index("Stop-Process")
    # No owner that can serve the verified endpoint means refuse, not kill all.
    assert stop.index("LocalAddress") < stop.index("if (-not $owners) {")
    # The endpoint the liveness check actually verified.
    assert "http://127.0.0.1:$port/api/liveness" in ps1


def test_the_managed_venv_is_recorded_when_the_studio_home_already_exists():
    """The elevated installer creates <home>\\unsloth_studio, but a Studio home
    that already exists is not in $absentBefore and the venv was in no candidate
    list at all. With the other candidates present too, nothing was recorded,
    StudioInstalledByProbe stayed false, and revert printed 'nothing to repair'
    and stamped a completed rollback over the 673-PE tree Studio cannot start
    without."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    init = ps1[ps1.index("function Initialize-Studio") : ps1.index("function Save-Baseline")]
    cands = init[init.index("$candidates = @(") : init.index("$absentBefore = @(")]
    assert "(Join-Path (Get-StudioHome) 'unsloth_studio')" in cands
    # And revert must accept it: it resolves under a root derived from the live
    # environment, so the containment gate does not reject it as somebody else's.
    revert = ps1[ps1.index("function Invoke-Revert") :]
    roots = revert[revert.index("$allowedRoots = @(") : revert.index("$recorded = @()")]
    assert "(Get-StudioHome)" in roots


def test_run_refuses_a_label_whose_revert_already_completed():
    """revert leaves window-start.txt behind, so the window still on disk is the
    one the reverted run measured. A run after it would overwrite the scenario
    results and both inventories with a current runtime while collect kept
    exporting the pre-revert window, on a machine that is no longer prepared."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    run = ps1[ps1.index("function Invoke-Run") : ps1.index("function Invoke-Collect")]
    assert "if ($runBaseline.RevertCompletedAt) {" in run
    # Before anything is re-verified, measured or overwritten.
    guard = run.index("if ($runBaseline.RevertCompletedAt) {")
    assert guard < run.index("if ($runBaseline.AuditPolicyApplied) {")
    assert guard < run.index("Write-Section 'Venv signature inventory'")
    # All three stages now refuse a spent baseline rather than two of them.
    assert "if ($baseline.RevertCompletedAt) {" in ps1[ps1.index("function Invoke-Revert") :]


def test_collect_scopes_the_venv_run_measured_not_the_one_this_shell_resolves():
    """Both resolutions go through UNSLOTH_STUDIO_HOME, and a reboot between the
    stages is supported, so a collect from a shell without that override scoped
    the window against the legacy default while run had inventoried a custom
    venv. The real tree's events were then filed as somebody else's - and the
    one enforced 3077 this probe has ever seen was inside that tree."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    assert "function Resolve-VenvDir([string] $dir) {" in ps1
    fn = ps1[ps1.index("function Resolve-VenvDir") : ps1.index("function Test-StudioResponding")]
    assert "$recorded = Join-Path $dir 'venv-selection.txt'" in fn
    # The recorded answer wins, and only when a run directory was passed.
    assert fn.index("$recorded = Join-Path") < fn.index("$studioPython = Get-StudioPython")
    run = ps1[ps1.index("function Invoke-Run") : ps1.index("function Invoke-Collect")]
    assert "$venvDir | Set-Content -LiteralPath (Join-Path $dir 'venv-selection.txt')" in run
    # Written from the value the inventory actually used.
    assert run.index("$venvDir = Resolve-VenvDir") < run.index("'venv-selection.txt'")
    # A reopened window must not inherit the previous one's answer.
    prepare = ps1[ps1.index("function Invoke-Prepare") : ps1.index("function Get-SignatureInventory")]
    stale = prepare[prepare.index("foreach ($stale in @(") :]
    assert "'venv-selection.txt'" in stale[: stale.index(")) {")]


def test_the_prepare_timestamp_is_read_back_culture_invariantly():
    """ConvertFrom-Json turns CapturedAt into a DateTime, PowerShell stringifies
    that with the invariant culture (MM/dd/yyyy) to bind the string overload,
    and [datetime]::Parse reads the CURRENT one. On a dd/MM machine every
    prepare after the 12th of a month threw here and left $preparedAt null, so
    the after-a-reboot revalidation was skipped and collect graded the window on
    a control that fired on a different boot."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    run = ps1[ps1.index("function Invoke-Run") : ps1.index("function Invoke-Collect")]
    assert "[datetime]::Parse($runBaseline.CapturedAt)" not in run
    # The audit-policy cell, the Smart App Control cell, and the window-start
    # freshness guard, all three of them.
    assert run.count("[datetime]::Parse([string]$runBaseline.CapturedAt,") == 3
    # Four reads now: the two CapturedAt reboot checks, and the freshness guard,
    # which parses window-start.txt and CapturedAt as a pair.
    assert run.count("[cultureinfo]::InvariantCulture") == 4
    # Still gating the control, not a parse nobody reads.
    assert run.count("$bootedAt -and $preparedAt -and $bootedAt -gt $preparedAt") == 2


def test_the_streamed_turns_opt_into_the_tool_control_frames(monkeypatch):
    """/v1/chat/completions emits a clean OpenAI stream for external clients:
    tool_start and tool_end carry no `choices`, so they are suppressed unless
    the caller sends X-Unsloth-Events, which the Studio frontend does. Without
    it chat() saw an empty `finished` list on every real Studio, reported 'no
    tool_end event: the turn executed no tool', and the scenario exited nonzero
    with the tool behaviour it exists to measure never observed."""
    backend = (REPO_ROOT / "studio" / "backend" / "routes" / "inference.py").read_text(
        encoding = "utf-8"
    )
    # The gate this opts into, so the test fails if the backend renames it.
    assert 'UI_STREAM_EVENTS_HEADER = "X-Unsloth-Events"' in backend
    frontend = (
        REPO_ROOT / "studio" / "frontend" / "src" / "features" / "chat" / "api" / "chat-api.ts"
    ).read_text(encoding = "utf-8")
    assert '"X-Unsloth-Events": "1"' in frontend

    s = _load_scenario()
    seen: dict[str, str] = {}

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __iter__(self):
            return iter([b'data: {"type": "tool_end", "tool_name": "web_search", "result": "x"}\n',
                         b"data: [DONE]\n"])

    def fake_urlopen(req, timeout = None):
        seen.update(req.headers)
        return _Resp()

    monkeypatch.setattr(s.urllib.request, "urlopen", fake_urlopen)
    status, events, error = s._stream_events("http://x", "/v1/chat/completions", {}, token = "t")
    # urllib title-cases header keys.
    assert seen.get("X-unsloth-events") == "1", seen
    assert status == 200 and error is None
    assert any(e.get("type") == "tool_end" for e in events)


def test_collect_reads_the_studio_logs_of_the_install_run_measured():
    """Both the log root and the redactor interpreter went through
    Get-StudioHome, so a collect from a shell without a custom
    UNSLOTH_STUDIO_HOME looked under the legacy default: Studio's backend logs
    were absent from the zip with nothing recording why, and no managed
    interpreter was found to redact even the probe's own raw logs."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    assert "function Resolve-StudioHomeFor([string] $dir) {" in ps1
    assert "function Resolve-StudioPythonFor([string] $dir) {" in ps1
    helpers = ps1[ps1.index("function Resolve-StudioHomeFor") : ps1.index("function Test-StudioResponding")]
    # Both derive from the venv run recorded, and both still fall back to the
    # live search for an evidence directory that predates venv-selection.txt.
    assert helpers.count("Resolve-VenvDir $dir") == 2
    assert "return (Get-StudioHome)" in helpers and "return (Get-StudioPython)" in helpers
    collect = ps1[ps1.index("function Invoke-Collect") : ps1.index("function Invoke-Revert")]
    assert "$studioLogs = Join-Path (Resolve-StudioHomeFor $dir) 'logs'" in collect
    assert "$python = Resolve-StudioPythonFor $dir" in collect
    assert "Join-Path (Get-StudioHome) 'logs'" not in collect


def test_run_refuses_an_event_window_older_than_its_own_baseline():
    """prepare captures a fresh baseline for a reverted label and only reopens
    the window at the very end, so a throw in between - a CodeIntegrity channel
    it cannot raise, a missing -AuditPolicy, a policy that will not load -
    leaves an unspent baseline beside the previous run's window-start.txt. The
    spent-baseline guard passes on that, and collect then exported the reverted
    run's events against the inventories taken now."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    run = ps1[ps1.index("function Invoke-Run") : ps1.index("function Invoke-Collect")]
    assert "$runStartPath = Join-Path $dir 'window-start.txt'" in run
    assert "$runStart -and $runPrepared -and $runStart -lt $runPrepared" in run
    # Before anything is measured, and after the spent-baseline guard.
    assert run.index("$runStartPath = Join-Path") < run.index("Write-Section 'Venv signature inventory'")
    assert run.index("if ($runBaseline.RevertCompletedAt) {") < run.index("$runStartPath = Join-Path")
    # collect must NOT gain the same guard: an operator who never collected the
    # previous cycle can still collect it after a prepare failed.
    collect = ps1[ps1.index("function Invoke-Collect") : ps1.index("function Invoke-Revert")]
    assert "$runPrepared" not in collect
    # And the window still opens last, after the positive control, so the
    # control's own event stays outside the window it measures.
    prepare = ps1[ps1.index("function Invoke-Prepare") : ps1.index("function Get-SignatureInventory")]
    assert prepare.rindex("Test-AuditPolicyEvaluating") < prepare.index("'window-start.txt'")


def test_the_unattended_tool_turns_never_wait_on_an_approval_nobody_reads():
    """Opting into the control frames also opens the confirm gate: an unset
    permission_mode is read as 'auto', and under auto web_search prompts as soon
    as the model supplies a url. The approval is waited on for an hour against
    this script's 900s read timeout, so the turn would hang and fail as a
    transport error. 'off' disables the gate only; the sandbox stays on."""
    scenario = (PROBE_DIR / "studio_scenario.py").read_text(encoding = "utf-8")
    assert 'payload["permission_mode"] = "off"' in scenario
    # Only on the tool turns, and beside the selection they gate.
    tools = scenario[scenario.index("    if tools:") : scenario.index("        text = \"\"")]
    assert 'payload["permission_mode"] = "off"' in tools
    assert 'payload["permission_mode"] = "full"' not in scenario
    # The sandbox stays on: neither of the two ways to drop it is set.
    assert 'payload["bypass_permissions"]' not in scenario
    # The backend meanings this relies on.
    models = (REPO_ROOT / "studio" / "backend" / "models" / "inference.py").read_text(
        encoding = "utf-8"
    )
    assert '_KNOWN_PERMISSION_MODES = ("ask", "auto", "off", "full")' in models
    backend = (REPO_ROOT / "studio" / "backend" / "routes" / "inference.py").read_text(
        encoding = "utf-8"
    )
    assert 'if mode in ("off", "full"):' in backend


def test_a_runtime_at_the_root_of_a_volume_is_refused_not_matched_against_everything():
    r"""UNSLOTH_LLAMA_CPP_PATH=D:\ is returned verbatim by Get-LlamaDir, and a
    llama-server.exe at a volume root resolves to the same place. Stripping the
    drive letter and the trailing separator then leaves nothing, so the tail was
    a lone separator, which every path in this machine-wide channel contains:
    the Git Bash msys-2.0.dll events this scoping exists to keep in 'other' were
    counted as Unsloth 3076s and the cell reported blocks it never saw."""
    ps1 = (PROBE_DIR / "sac-probe.ps1").read_text(encoding = "utf-8")
    fn = ps1[ps1.index("function Get-ScopeTail") : ps1.index("function Get-EventDataMap")]
    assert "if (-not $trimmed) {" in fn
    # Refused before the pattern is built, not warned about afterwards.
    assert fn.index("if (-not $trimmed) {") < fn.index("WildcardPattern]::Escape($trimmed)")
    assert "throw" in fn[fn.index("if (-not $trimmed) {") : fn.index("return (")]
    # The ordinary shapes are untouched: the drive letter still goes (device
    # paths carry none) and the tail still ends at a separator.
    assert "($root -replace '^[A-Za-z]:', '').TrimEnd('\\', '/')" in fn
    assert "WildcardPattern]::Escape($trimmed) + '\\'" in fn
    # Both scoping call sites go through it, so neither tree can be graded on a
    # volume-wide match.
    collect = ps1[ps1.index("function Invoke-Collect") : ps1.index("function Invoke-Revert")]
    assert "$tail = Get-ScopeTail (Resolve-LlamaDir $dir)" in collect
    assert "$venvTail = Get-ScopeTail (Resolve-VenvDir $dir)" in collect
