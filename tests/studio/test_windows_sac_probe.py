# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Guards for scripts/windows_sac_probe, the Smart App Control evidence probe."""

from __future__ import annotations

import functools
import importlib.util
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from unsloth_pwsh_runner import run_pwsh

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_DIR = REPO_ROOT / "scripts" / "windows_sac_probe"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "windows-llama-signature-audit.yml"
SCENARIO = PROBE_DIR / "studio_scenario.py"
PS1 = PROBE_DIR / "sac-probe.ps1"


@functools.cache
def _text(path):
    return Path(path).read_text(encoding = "utf-8")


def _cut(text, start, end = None):
    return text[text.index(start) :] if end is None else text[text.index(start) : text.index(end)]


def _ps1(start = None, end = None):
    return _text(PS1) if start is None else _cut(_text(PS1), start, end)


def _prepare():
    return _ps1("function Invoke-Prepare", "function Get-SignatureInventory")


def _run():
    return _ps1("function Invoke-Run", "function Invoke-Collect")


def _collect():
    return _ps1("function Invoke-Collect", "function Invoke-Revert")


def _revert():
    return _ps1("function Invoke-Revert")


def _init():
    return _ps1("function Initialize-Studio", "function Save-Baseline")


def _verdict():
    verdict = _cut(_text(WORKFLOW), "      - name: Verdict\n        if: always()")
    return verdict[: verdict.index("      - name: Export the CodeIntegrity events")]


def _has(text, *needles):
    missing = [n for n in needles if n not in text]
    assert not missing, missing


def _lacks(text, *needles):
    present = [n for n in needles if n in text]
    assert not present, present


def _before(text, first, *later):
    for other in later:
        assert text.index(first) < text.index(other), (first, other)


def _job():
    import yaml

    return yaml.safe_load(_text(WORKFLOW))["jobs"]["code-integrity"]


def _step(key, value):
    return next(s for s in _job()["steps"] if s.get(key) == value)["run"]


def _run_ps(script, *args):
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("pwsh is required")
    return run_pwsh(
        [pwsh, "-NoProfile", "-File", str(script), *map(str, args)],
        capture_output = True,
        text = True,
        timeout = 120,
    )


def _assert_ok(proc):
    assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-1500:]


def _drive(tmp_path, names, body, strict = True, **params):
    """Run body after defining the named sac-probe.ps1 functions from their AST."""
    head = "param([string]$Src" + "".join(f",[string]${k}" for k in params) + ")\n"
    head += "$a=[System.Management.Automation.Language.Parser]::ParseFile($Src,[ref]$null,[ref]$null)\n"
    if strict:
        head += "$ErrorActionPreference = 'Stop'\n"
    head += f"$want = '{','.join(names)}'.Split(',')\n"
    head += "foreach($f in $a.FindAll({$args[0] -is [System.Management.Automation.Language.FunctionDefinitionAst]},$true)){\n"
    head += "  if($want -contains $f.Name){ Invoke-Expression $f.Extent.Text } }\n"
    if strict:
        head += "$CI_LOG = 'Microsoft-Windows-CodeIntegrity/Operational'\nfunction Start-Sleep { }\n"
    driver = tmp_path / "drive.ps1"
    driver.write_text(head + body, encoding = "utf-8")
    args = ["-Src", PS1]
    for key, value in params.items():
        args += [f"-{key}", value]
    _assert_ok(_run_ps(driver, *args))


def _load_scenario():
    spec = importlib.util.spec_from_file_location("studio_scenario", SCENARIO)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def s():
    return _load_scenario()


TOOL_START = {"type": "tool_start", "tool_name": "web_search"}


def _tool_end(result):
    return {"type": "tool_end", "tool_name": "web_search", "result": result}


def _events(monkeypatch, s, events, error = None):
    monkeypatch.setattr(s, "_stream_events", lambda *a, **k: (200, events, error))


def _chat(s):
    return s.chat("http://x", "t", "m", "look it up", tools = True)


def _bootstrap(s, tmp_path, monkeypatch, password):
    (tmp_path / "auth").mkdir()
    (tmp_path / "auth" / ".bootstrap_password").write_text(password, encoding = "utf-8")
    posted: list[tuple[str, dict]] = []

    def fake(base_url, method, path, payload = None, token = None, timeout = 900):
        posted.append((path, payload or {}))
        return 200, {"access_token": "tok"}

    monkeypatch.setattr(s, "_request", fake)
    return posted


def _main(s, monkeypatch, tmp_path, model = "m", replies = None, events = ()):
    """Run the scenario against a fake Studio; returns (exit code, calls, results)."""
    calls: list[tuple[str, str, dict]] = []
    answers = {
        "/api/liveness": (200, {"status": "alive", "service": "Unsloth UI Backend"}),
        "/api/auth/login": (200, {"access_token": "tok"}),
        **(replies or {}),
    }

    def fake(base_url, method, path, payload = None, token = None, timeout = 900):
        calls.append((method, path, payload or {}))
        return answers.get(path, (200, {"status": "done"}))

    def fake_stream(base_url, path, payload, token = None, timeout = 900):
        calls.append(("STREAM", path, payload or {}))
        return 200, list(events), None

    monkeypatch.setattr(s, "_request", fake)
    monkeypatch.setattr(s, "_stream_events", fake_stream)
    argv = ["--model", model, "--out", tmp_path, "--port", "1", "--password", "pw"]
    argv += ["--home", tmp_path, "--poll-seconds", "0.05"]
    monkeypatch.setattr(sys, "argv", ["studio_scenario.py", *map(str, argv)])
    rc = s.main()
    results = tmp_path / "scenario-results.json"
    return rc, calls, json.loads(_text(results)) if results.exists() else None


@pytest.mark.parametrize(
    "ref, expected",
    [
        ("unsloth/Qwen3.5-2B-MTP-GGUF:UD-Q4_K_XL", ("unsloth/Qwen3.5-2B-MTP-GGUF", "UD-Q4_K_XL")),
        ("unsloth/Qwen3.5-2B-GGUF", ("unsloth/Qwen3.5-2B-GGUF", None)),
        (r"C:\models\x.gguf", (r"C:\models\x.gguf", None)),
        ("C:/models/x.gguf", ("C:/models/x.gguf", None)),
    ],
)
def test_the_repo_variant_shorthand_is_split_into_the_two_load_fields(s, ref, expected):
    """/api/inference/load has no shorthand parser"""
    assert s.split_model_ref(ref) == expected


def test_the_status_poller_does_not_shadow_thread_stop(s, monkeypatch):
    """Thread.join() calls its own internal _stop()"""
    monkeypatch.setattr(s, "_request", lambda *a, **k: (200, {}))
    poller = s.StatusPoller("http://127.0.0.1:1", "t", interval = 0.01)
    assert not isinstance(getattr(poller, "_stop", None), threading.Event)
    poller.start()
    poller.stop()
    poller.join(timeout = 5)
    assert not poller.is_alive()
    assert poller.in_flight_ms() is None


def test_a_never_opened_studio_is_not_rotated_to_a_published_password(
    s, tmp_path, monkeypatch, capsys
):
    """The rotation is permanent and revert does not undo it"""
    posted = _bootstrap(s, tmp_path, monkeypatch, "boot-secret")
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
    assert "unsloth-sac-probe" not in _text(SCENARIO)


def test_the_scenario_loads_with_the_variant_field_and_unloads_by_model_path(
    s, tmp_path, monkeypatch
):
    """UnloadRequest.model_path is required"""
    events = [
        TOOL_START,
        _tool_end("Reykjavik: 7 September"),
        {"choices": [{"delta": {"content": "found it"}}]},
    ]
    reply = {"choices": [{"message": {"content": "hi", "tool_calls": []}}]}
    rc, calls, results = _main(
        s,
        monkeypatch,
        tmp_path,
        "unsloth/Qwen3.5-2B-MTP-GGUF:UD-Q4_K_XL",
        {"/v1/chat/completions": (200, reply)},
        events,
    )
    assert rc == 0
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
    assert results["steps"]["unload"]["ok"] is True
    assert "in_flight_ms" in results["status_poll"] and "stalls_ms" in results["status_poll"]
    assert results["gguf_variant"] == "UD-Q4_K_XL"


def test_the_poller_abandons_a_read_at_the_frontend_timeout_and_measures_the_stall(s, monkeypatch):
    """The frontend ticks every 5 s and abandons a status read after 10 s, so a stall is a stream of abandoned reads"""
    assert s.STATUS_READ_TIMEOUT_S == 10.0 and s.STATUS_INTERVAL_S == 5.0
    source = _text(SCENARIO)
    status_call = source[source.index('"/api/inference/status"') :]
    assert "timeout = self.read_timeout" in status_call[: status_call.index(")")]
    assert "poller.join(timeout = STATUS_READ_TIMEOUT_S + 15)" in source

    def stalled(base_url, method, path, payload = None, token = None, timeout = 900):
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
    poller.polls = [(0.0, 50.0, True), (0.06, 50.0, True), (0.2, 5.0, False), (0.3, 50.0, True)]
    assert [round(x) for x in poller.stalls_ms()] == [205, 50]


def test_an_empty_bootstrap_file_means_rotated(s, tmp_path, monkeypatch):
    """Studio truncates .bootstrap_password on Windows when it cannot delete it, so an existence check picked the empty string over the operator's password."""
    posted = _bootstrap(s, tmp_path, monkeypatch, "")
    s.authenticate("http://x", tmp_path, "operators-choice")
    assert posted[0] == ("/api/auth/login", {"username": "unsloth", "password": "operators-choice"})
    assert not any(p[0] == "/api/auth/change-password" for p in posted)


def test_a_model_already_resident_is_evicted_first_and_never_counts_as_loaded(
    s, tmp_path, monkeypatch
):
    """/load answers already_loaded for a resident model and starts nothing, so no PE is loaded inside the evidence window."""
    already = {"/api/inference/load": (200, {"status": "already_loaded", "model": "x"})}
    rc, calls, results = _main(s, monkeypatch, tmp_path, replies = already)
    assert rc == 1
    paths = [c[1] for c in calls]
    assert paths.index("/api/inference/unload") < paths.index("/api/inference/load")
    assert results["steps"]["load"]["ok"] is False
    assert "already resident" in results["steps"]["load"]["error"]
    assert results["evicted_before_load"]["status"] == 200


def test_a_tool_end_carrying_a_refusal_or_error_is_not_an_execution(s, monkeypatch):
    """The loop emits tool_end for a declined call, a lost runtime and an interrupted call too, with the reason in `result`."""
    assert s.tool_end_failure("") is not None
    assert s.tool_end_failure(None) is not None
    assert s.tool_end_failure(s.TOOL_REJECTED_MESSAGE) == "declined before running"
    assert s.tool_end_failure(
        "Error: lost connection to llama-server before the tool call completed."
    )
    assert s.tool_end_failure('{"results": [{"title": "Burj Khalifa"}]}') is None
    _events(monkeypatch, s, [TOOL_START, _tool_end(s.TOOL_REJECTED_MESSAGE)])
    turn = _chat(s)
    assert turn["ok"] is False and turn["tools_run"] == []
    assert turn["tools_failed"][0]["why"] == "declined before running"


def test_the_powershell_probe_collects_honestly_and_never_installs_from_run():
    ps1 = _ps1()
    _has(
        ps1,
        "Initialize-Studio $dir $false",
        "Initialize-Studio $dir $true",
        "function Initialize-Studio([string] $dir, [bool] $allowInstall)",
        "CiTool listed no policies",
        "NoMatchingEventsFound*",
        "events-collection-error.txt",
        "foreach ($r in $restores)",
    )
    _lacks(
        ps1,
        "could not be verified as active; read the 3076 count",
        'Write-Warning "no CodeIntegrity events in the window',
    )
    assert ps1.count("Set-MpPreference @params") == 2


def test_the_new_guard_runs_in_the_unfiltered_lint_job():
    lint = _text(REPO_ROOT / ".github" / "workflows" / "workflow-trigger-lint.yml")
    assert "tests/studio/test_windows_sac_probe.py" in lint


def test_the_powershell_probe_restores_what_prepare_changed_and_unmounts_efi():
    ps1 = _ps1()
    # Both stages and prepare's rollback unmount an EFI only the probe mounted.
    assert ps1.count("Dismount-Efi $mounted") == 3
    _has(
        ps1,
        "mountvol.exe S: /D",
        "Invoke-Native 'CiTool.exe' @('-r')",
        "Invoke-Native 'mountvol.exe'",
        "Test-PolicyActive $NOISG_GUID",
        "AuditPolicyPreexisting",
        "preexisting-policy.cip",
        "CiLogMaxSize",
        "CiLogEnabled",
        "/ms:$($baseline.CiLogMaxSize)",
        "ConvertTo-Json -InputObject $shaped",
        "ConvertTo-Json -InputObject @($inventory)",
        "UNSLOTH_LLAMA_CPP_PATH",
        "UNSLOTH_STUDIO_HOME",
    )
    assert "Join-Path $env:USERPROFILE '.unsloth\\studio\\logs'" not in ps1


def test_the_signature_audit_covers_every_windows_family():
    body = _text(WORKFLOW)
    _has(
        body,
        "gh release view $tag --repo unslothai/llama.cpp --json assets",
        "Where-Object { $_ -like '*windows*' -and $_ -like '*.zip' }",
        'throw "no Windows assets on $tag"',
        "enumerated $($assets.Count) Windows asset(s) but inventoried $audited",
    )
    # A hard coded profile would silently shrink the set again.
    _lacks(
        body,
        "*windows-x64-cuda12-legacy.zip",
        "*windows-x64-rocm-gfx110X.zip",
        "*windows-arm64-cpu.zip",
    )


def test_the_readme_does_not_claim_the_release_tag_pins_a_run():
    body = _text(PROBE_DIR / "README.md")
    assert "read\nby the installer only" in body or "read by the installer only" in body
    _has(body, "re-run the Studio installer", "$env:UNSLOTH_STUDIO_PASSWORD")


def test_a_padded_load_reply_is_read_for_its_deferred_error(s):
    """/api/inference/load commits a 200 after 15 seconds and reports a later failure only in the body."""
    assert s.padded_route_failure(200, {"status": "loaded"}) is None
    assert s.padded_route_failure(500, {"detail": "boom"}) == "{'detail': 'boom'}"
    deferred = s.padded_route_failure(
        200, {"_deferred_error": {"status_code": 500, "detail": "llama-server was blocked"}}
    )
    assert deferred and "llama-server was blocked" in deferred
    assert s.padded_route_failure(200, {}) is not None, "a truncated padded body is not a load"
    assert s.padded_route_failure(200, "") is not None
    assert _text(SCENARIO).count("padded_route_failure(status, body)") == 2, "both padded routes"


def test_a_tool_turn_is_ok_only_when_a_tool_actually_ran(s, monkeypatch):
    """The non-streaming route drains the tool loop and returns the final text alone, so a model answering from memory passed as a tool turn."""
    _events(monkeypatch, s, [{"choices": [{"delta": {"content": "Burj Khalifa"}}]}])
    memory = _chat(s)
    assert memory["ok"] is False and memory["tool_calls"] == 0 and "no tool_end" in memory["error"]
    _events(monkeypatch, s, [TOOL_START, _tool_end("Burj Khalifa, 828 m")])
    ran = _chat(s)
    assert ran["ok"] is True and ran["tools_run"] == ["web_search"]


def test_the_powershell_probe_handles_retries_skips_and_occupied_drives():
    ps1 = _ps1()
    _has(
        ps1,
        "if (-not $SkipStudio -and -not (Test-StudioResponding $Port)) { Initialize-Studio $dir $false }",
        "no window-start.txt under",
        "'S:\\EFI\\Microsoft\\Boot'",
        "ConvertTo-Json -InputObject $detections",
        "if ($UpgradePackages) {",
        "if (-not $SkipUpdates) {",
        "$baseline = Get-Content -LiteralPath $baselinePath -Raw | ConvertFrom-Json",
        "(Test-Path -LiteralPath $NOISG_DEST) -and -not $baseline.AuditPolicyApplied",
    )
    assert "(Get-Date).AddHours(-2)" not in ps1
    # winget upgrade --all is opt-in, since revert cannot undo it.
    winget = ps1.index("winget upgrade --all --accept")
    assert ps1.rfind("if ($UpgradePackages) {", 0, winget) > ps1.rfind(
        "if (-not $SkipUpdates) {", 0, winget
    )
    applied = ps1.index("$baseline.AuditPolicyApplied = $true")
    persist = ps1.index("Save-ProbeBaseline $baseline $baselinePath", applied)
    assert applied < persist < ps1.index("Invoke-Native 'CiTool.exe' @('-r')")
    locator = _ps1("function Get-StudioPython", "function Test-StudioResponding")
    assert "Get-StudioHome" in locator and "$env:UNSLOTH_STUDIO_HOME" not in locator


def test_the_signature_audit_fails_on_any_missing_bundle():
    """A native exit code is not terminating in pwsh and a later download replaces $LASTEXITCODE."""
    block = _cut(
        _text(WORKFLOW),
        "gh release download $env:AUDIT_TAG",
        "$audited = @($rows | Group-Object Bundle).Count",
    )
    _has(
        block,
        'Write-Host "::error::gh release download exited $LASTEXITCODE for $asset"',
        'Write-Host "::error::$asset did not download"',
        "Get-Item (Join-Path 'bundles' $asset)",
    )


def test_the_signature_audit_streams_one_bundle_at_a_time():
    """Seventeen Windows assets are several GB"""
    block = _cut(_text(WORKFLOW), "- name: Inventory Authenticode signatures")
    assert "Remove-Item $zip.FullName -Force" in block, "the bundle is never deleted"
    _before(
        block, "Remove-Item $zip.FullName -Force", "$audited = @($rows | Group-Object Bundle).Count"
    )


class _FakeStream:
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


def test_a_stream_that_errors_after_the_tool_ran_is_not_a_success(s, monkeypatch):
    """A failure after the status line went out arrives in band as an error frame with the 200 kept, and the stream ends without [DONE]"""
    tool_end = 'data: {"type": "tool_end", "tool_name": "web_search", "result": "found"}\n'
    error_frame = 'data: {"error": {"message": "llama-server exited", "type": "server_error"}}\n'
    status, events, error = _stream(monkeypatch, s, [tool_end, error_frame])
    assert status == 200 and error and "llama-server exited" in error
    status, events, error = _stream(monkeypatch, s, [tool_end])
    assert error and "without [DONE]" in error
    status, events, error = _stream(monkeypatch, s, [tool_end, "data: [DONE]\n"])
    assert error is None and len(events) == 1
    _events(monkeypatch, s, [_tool_end("found")], "stream error: llama-server exited")
    turn = _chat(s)
    assert turn["ok"] is False and turn["tools_run"] == ["web_search"] and "exited" in turn["error"]


def test_the_powershell_probe_fails_closed_on_the_log_and_finishes_the_policy_refresh():
    ps1 = _ps1()
    assert "could not configure the CodeIntegrity log" not in ps1
    assert "is still disabled after wevtutil sl" in ps1
    block = _ps1("Write-Section 'Remove audit policy'", "Write-Section 'Restore CodeIntegrity log'")
    refresh = "Invoke-Native 'CiTool.exe' @('-r')"
    assert block.count(refresh) == 1
    _before(block, "audit policy file already absent", refresh)
    _before(block, refresh, "$baseline.AuditPolicyApplied = $false")
    _before(block, "Dismount-Efi $mounted", "$baseline.AuditPolicyApplied = $false")
    assert "Where-Object { $_.InitialDetectionTime -ge $start }" in ps1


def test_a_tool_end_the_loop_closed_without_running_is_not_an_execution(s):
    """studio_tool_loop.py closes a truncated, cancelled, disabled or budget-exhausted call with a non-empty result that is neither the refusal nor an Error"""
    for result in (
        "Unsloth did not execute this tool call because the provider stopped mid-call at its output limit.",
        "Unsloth stopped this tool call before it returned, so there is no result. The tool may have already done part of its work.",
        "Unsloth did not execute this tool call because the tool is disabled.",
        "Unsloth did not run this call because an identical one had already completed.",
    ):
        assert s.tool_end_failure(result) is not None, result
    assert s.tool_end_failure("Unsloth Studio docs: https://docs.unsloth.ai") is None


def test_studio_logs_reach_the_evidence_only_through_the_backend_redactor(tmp_path):
    """The zip is attached to an issue."""
    ps1 = _ps1()
    _lacks(ps1, "Copy-Item -LiteralPath $studioLogs", "Redact-Secrets")
    _has(ps1, "redact_logs.py", "no managed interpreter to run the redactor")
    assert "$python = Resolve-StudioPythonFor $dir" in _ps1("function Invoke-Collect")
    src = tmp_path / "logs"
    src.mkdir()
    secrets = (
        "hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345",
        "opaquevalue123456",
        "horse battery staple",
        "abcdef ghijklmnop",
        "dXNlcm5hbWU6c3VwZXJzZWNyZXQ=",
        "8f3c9d1ab77e4f0a9c2b3d4e",
    )
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
    redactor = [sys.executable, str(PROBE_DIR / "redact_logs.py"), str(src), str(tmp_path / "out")]
    redactor += ["--backend", str(REPO_ROOT / "studio" / "backend")]
    proc = subprocess.run(redactor, capture_output = True, text = True, timeout = 120)
    assert proc.returncode == 0, proc.stderr[-1500:]
    out = (tmp_path / "out" / "studio.log").read_text(encoding = "utf-8")
    _lacks(out, *secrets)
    assert "n_tokens = 4096" in out


def test_the_powershell_probe_rejects_empty_inventories_and_keeps_reverting():
    assert "no PE files found under ${llamaDir}:" in _ps1()
    revert = _revert()
    _has(revert, "$policyError = $_", "the audit policy is still applied")
    _before(revert, "$policyError = $_", "Write-Section 'Restore CodeIntegrity log'")
    _before(revert, "Write-Section 'Restore Defender preferences'", "if ($null -ne $policyError) {")


def test_rollback_state_is_persisted_before_the_efi_partition_changes_and_stays_out_of_the_zip():
    block = _ps1("Write-Section 'Audit policy'", "Write-Section 'Policy state after applying'")
    persist = block.index("$baseline.AuditPolicyApplied = $true")
    assert block.index("Copy-Item -LiteralPath $NOISG_DEST -Destination $ROLLBACK_POLICY") < persist
    assert (
        persist
        < block.index("Save-ProbeBaseline $baseline $baselinePath", persist)
        < block.index("Copy-Item -LiteralPath $AuditPolicy -Destination $NOISG_DEST")
    )
    ps1 = _ps1()
    _has(
        ps1,
        "Join-Path (Join-Path $dir 'rollback') 'preexisting-policy.cip'",
        "if ($rel -like 'rollback\\*' -or $rel -like 'raw-logs\\*') { continue }",
    )
    assert "Compress-Archive -Path (Join-Path $dir '*')" not in ps1
    assert "$saved = $ROLLBACK_POLICY" in _revert()


def test_the_inventory_is_of_the_runtime_studio_resolved(s, tmp_path, monkeypatch):
    """A folder selected in Studio's settings, or LLAMA_SERVER_PATH, wins over the managed default in Studio"""
    llama = _ps1("function Get-LlamaDir", "function Resolve-LlamaDir")
    _before(llama, "$env:LLAMA_SERVER_PATH", "$env:UNSLOTH_LLAMA_CPP_PATH")
    resolve = _ps1("function Resolve-LlamaDir", "function Invoke-Native")
    _has(resolve, "runtime-selection.json", "$sel.resolved_binary")
    _before(_run(), "& python @scenarioArgs", "$llamaDir = Resolve-LlamaDir $dir")
    selection = {
        "path": "D:\\llama",
        "source": "studio",
        "resolved_binary": "D:\\llama\\llama-server.exe",
    }
    _, calls, _ = _main(
        s, monkeypatch, tmp_path, replies = {"/api/settings/llama-cpp-path": (200, selection)}
    )
    sel = json.loads(_text(tmp_path / "runtime-selection.json"))
    assert sel["resolved_binary"].endswith("llama-server.exe") and sel["source"] == "studio"
    seen = [c[1] for c in calls]
    assert seen.index("/api/settings/llama-cpp-path") < seen.index("/api/inference/load")


def test_prepare_restarts_a_running_studio_and_only_prepare_may():
    """Studio's startup is where the venv's native modules load"""
    assert "function Stop-Studio" in _ps1()
    init = _init()
    answering = init.index("if (Test-StudioResponding $Port) {")
    assert init.index("if (-not $allowInstall) {", answering) < init.index(
        "Stop-Studio $Port", answering
    )
    stop = _ps1("function Stop-Studio", "function Initialize-Studio")
    assert "ParentProcessId = $owner" in stop, "children (llama-server, workers) are stopped first"


def test_redirected_studio_output_and_the_scenario_console_are_redacted_too():
    _has(
        _ps1(),
        "$startLog = Join-Path (Join-Path $dir 'raw-logs') 'studio-start.log'",
        "Start-Studio $python $Port $startLog",
        "$log = Join-Path (Join-Path $dir 'raw-logs') 'studio-scenario.log'",
        "if ($rel -like 'rollback\\*' -or $rel -like 'raw-logs\\*') { continue }",
    )
    _has(_collect(), "$rawLogs = Join-Path $dir 'raw-logs'", "foreach ($source in $sources)")


def test_sample_submission_is_opt_in():
    assert "[switch] $SendSamples" in _ps1()
    block = _ps1("Write-Section 'Raise security settings'", "Write-Section 'CodeIntegrity log'")
    _before(block, "if ($SendSamples) {", "$wanted['SubmitSamplesConsent'] = 'SendAllSamples'")
    assert block.count("SubmitSamplesConsent") == 1


def test_collect_judges_the_load_step_not_the_exit_code():
    """A failed search, chat or unload after a working load is still a valid load-time measurement"""
    _has(
        _collect(),
        "$loadOk = [bool]$results.steps.load.ok",
        "if ($loadOk) {",
        "later scenario step(s) failed",
        "did NOT load a model",
    )


def test_the_venv_inventory_comes_from_the_running_interpreter_and_cannot_be_empty():
    _has(
        _ps1(),
        "function Resolve-VenvDir",
        "$venvDir = Resolve-VenvDir",
        "no PE files found under $venvDir",
    )
    # collect reads run's recorded venv instead of resolving it again in this shell.
    collect = _collect()
    _has(collect, "$venvTail = ", "Get-ScopeTail (Resolve-VenvDir $dir)")
    assert "$VENV_DIR -replace" not in collect


def test_the_readme_clones_a_durable_ref():
    body = _text(PROBE_DIR / "README.md")
    assert "--branch windows-sac-probe" not in body
    _has(body, "git clone --depth 1 https://github.com/unslothai/unsloth", "-SendSamples")


def test_the_app_control_audit_keeps_its_positive_control():
    """No 3076 events must never be reportable as a pass on its own."""
    steps = {s.get("name", ""): s for s in _job()["steps"]}
    control = next(n for n in steps if n.startswith("Positive control"))
    _has(steps[control]["run"], "control_fired=true", "control_fired=false")
    assert steps["Verdict"]["if"] == "always() && steps.control.outputs.control_fired == 'true'"
    apply_step = next(steps[n] for n in steps if n.startswith("Apply the policy"))
    assert "not in the active policy set after refresh" in apply_step["run"]
    # Audit, never enforcement: an enforcing policy could brick the runner.
    _has(_text(WORKFLOW), "SmartAppControlAuditNoISG", "--remove-policy")


_LISTED = '{"Policies":[{"PolicyID":"{AAAA}","FriendlyName":"X AuditNoISG","IsEnforced":true}]}'
_UNLISTED = '{"Policies":[]}'


def _removal_step(tmp_path, name, remove_exit = 0, refresh_exit = None, list_exit = 0, listing = ""):
    stub = "function CiTool.exe {\n"
    stub += f"  if ($args -contains '--remove-policy') {{ $global:LASTEXITCODE = {remove_exit}; return 'remove' }}\n"
    if refresh_exit is not None:
        stub += f"  if ($args -contains '--refresh') {{ $global:LASTEXITCODE = {refresh_exit}; return 'refresh' }}\n"
    stub += f"  if ($args -contains '-json') {{ $global:LASTEXITCODE = {list_exit}; return '{listing}' }}\n"
    stub += "  $global:LASTEXITCODE = 0; return 'Friendly Name: none'\n}\n$env:SAC_POLICY_ID = '{AAAA}'\n"
    script = tmp_path / f"remove_{name}.ps1"
    script.write_text(stub + _step("name", "Remove the policy"), encoding = "utf-8")
    return _run_ps(script)


@pytest.mark.parametrize(
    "remove_exit, listing, code, marker",
    [
        (5, _LISTED, 1, "::error::"),
        (0, _LISTED, 0, "::warning::"),
        (0, _UNLISTED, 0, "removed and no longer active"),
    ],
    ids = ["failed", "lingering", "clean"],
)
def test_the_policy_removal_step_fails_when_citool_could_not_remove_it(
    tmp_path, remove_exit, listing, code, marker
):
    """The refresh and the listing after --remove-policy overwrote its exit code, so a failed removal reported a clean one."""
    proc = _removal_step(tmp_path, "r", remove_exit = remove_exit, listing = listing)
    assert proc.returncode == code, proc.stdout + proc.stderr
    assert marker in proc.stdout


def test_the_inventory_root_is_the_selected_runtime_not_the_binary_directory():
    r"""<root>\build\bin\Release is a supported layout, so the parent of resolved_binary inventoried Release\ alone."""
    resolve = _ps1("function Resolve-LlamaDir", "$PE_EXT = ")
    assert resolve.index("Test-Path -LiteralPath $sel.path -PathType Container") < resolve.index(
        "return (Split-Path -Parent $sel.resolved_binary)"
    ), "the selected root wins; the binary's parent is only the direct-binary fallback"
    assert "return $sel.path" in resolve


def test_event_scoping_matches_path_tails_literally():
    """-like reads [ and ] as pattern syntax, and a directory called [llama] is legal and supported."""
    collect = _collect()
    for name in ("$tail = ", "$venvTail = "):
        assert "Get-ScopeTail" in collect[collect.index(name) : collect.index(name) + 200], name
    assert "WildcardPattern]::Escape($trimmed) + '\\'" in _ps1()
    workflow = _text(WORKFLOW)
    verdict = _cut(workflow, "$dir = $env:RUNTIME_DIR")
    assert "WildcardPattern]::Escape" in verdict
    assert "-replace '^[A-Za-z]:', ''" in verdict, "device-form paths are matched on the tail"
    assert '$_.Message -like "*$dir*"' not in workflow


def test_installer_trees_under_a_custom_studio_home_are_recorded_for_revert():
    """setup.ps1 resolves $NodeParent from UNSLOTH_STUDIO_HOME/STUDIO_HOME and $UnslothHome from the managed llama.cpp dir's parent."""
    init = _init()
    assert "$installRoots = @($unslothHome, $override, (Split-Path -Parent (Get-LlamaDir)))" in init
    for tree in ("'node'", "'whisper.cpp'", "'.cache'"):
        assert f"(Join-Path $_ {tree})" in init
        assert f"(Join-Path $unslothHome {tree})" not in init


def test_trees_the_installer_created_are_recorded_even_when_the_install_fails():
    init = _init()
    warning = "Write-Warning 'Studio still not found after the installer ran."
    _before(init, "$created = @($absentBefore", warning)
    _before(init, "$b.StudioInstallRoots = @(", warning)


def test_prepare_fails_when_a_running_studio_cannot_be_restarted():
    """The existing process loaded its venv native modules before the policy and the window existed."""
    init = _init()
    assert "if (-not (Stop-Studio $Port)) { return }" not in init
    stop = _cut(init, "if (-not (Stop-Studio $Port))")
    assert stop[: stop.index("}")].count("throw") == 1


def test_each_defender_preference_is_applied_on_its_own_and_deviations_recorded():
    """One shared try meant the first policy-controlled setting threw and every later Set-MpPreference was never called."""
    block = _ps1("Write-Section 'Raise security settings'", "Write-Section 'CodeIntegrity log'")
    _has(
        block,
        "foreach ($name in $wanted.Keys)",
        "DisableRealtimeMonitoring",
        "MAPSReporting",
        "CloudBlockLevel",
        "PUAProtection",
        "$mpErrorPath = Join-Path $dir 'defender-preference-errors.txt'",
        "Remove-Item -LiteralPath $mpErrorPath",
    )
    stale = _ps1("foreach ($stale in @(")
    assert "events-collection-error.txt" in stale[: stale.index(")) {")]


def test_an_efi_dismount_failure_is_never_reported_as_a_completed_stage():
    """The next stage's Mount-Efi finds the EFI tree already there and so never retries the unmount."""
    dismount = _ps1("function Dismount-Efi", "function Test-PolicyActive")
    assert "$script:EfiStillMounted = $true" in dismount
    assert "if ($script:EfiStillMounted) {" in _prepare()
    assert "-or $script:EfiStillMounted" in _revert()


def test_revert_exits_nonzero_when_a_studio_tree_acl_repair_failed():
    """The ACL loop sat outside the aggregate failure accounting."""
    revert = _revert()
    assert revert.count("$aclFailures++") == 2, "both the nonzero icacls and the catch count"
    assert "$aclFailures -gt 0" in revert
    _before(revert, "$aclFailures = $rejected.Count", "$aclFailures -gt 0")


def test_a_partial_inventory_says_so_in_the_evidence():
    """Get-ChildItem discarded the error for a subtree it could not read."""
    _has(
        _ps1(),
        "-ErrorAction SilentlyContinue -ErrorVariable enumErrors",
        "$script:InventoryErrors += ",
    )
    _has(_run(), "inventory-enumeration-errors.txt", "$script:InventoryErrors.Count -gt 0")


def test_a_partial_collection_is_marked_inside_the_zip():
    """A console warning does not travel with the archive."""
    collect = _collect()
    # evtx export, staging, redaction, no interpreter, unverified empty window, no policy,
    # no SAC refusal shown, nothing loaded, and audits from another policy
    assert collect.count("$collectionProblems += ") == 9
    redact = _cut(collect, "$redactor = Join-Path")
    assert redact.index('$collectionProblems += "log redaction failed') < redact.index(
        "Remove-Item -LiteralPath (Join-Path $dir 'studio-logs')"
    ), "recorded before the partial redacted output is deleted"
    assert "no managed interpreter to run the redactor, so studio-logs" in collect
    warn = "Set-Content -LiteralPath (Join-Path $stage 'collection-warnings.txt')"
    _before(collect, "foreach ($item in Get-ChildItem -LiteralPath $dir -Recurse -File)", warn)
    _before(collect, warn, "Compress-Archive -Path (Join-Path $stage '*')")
    _before(
        collect,
        "could not read $CI_LOG, so the event window was not collected",
        "Remove-Item -LiteralPath (Join-Path $dir 'events-collection-error.txt')",
    )


def test_the_app_control_verdict_cannot_pass_on_an_unread_channel():
    """SilentlyContinue turned a failed query into an empty array."""
    _has(
        _verdict(),
        "-ErrorAction Stop",
        "NoMatchingEventsFound",
        "::error::could not read the CodeIntegrity channel",
    )


def test_the_code_integrity_artifact_carries_the_events_it_is_named_for():
    workflow = _text(WORKFLOW)
    assert "- name: Export the CodeIntegrity events" in workflow
    upload = _cut(workflow, "name: code-integrity-events")
    _has(upload, "code-integrity-events.json", "CodeIntegrity-Operational.evtx")
    export = _cut(workflow, "- name: Export the CodeIntegrity events")
    head = export[: export.index("run: |")]
    assert head.count("if: always()") == 1 and "control_fired" not in head


def test_the_audit_channel_resize_is_verified_not_assumed():
    """A channel that was already enabled reads back enabled even when the resize failed."""
    step = _cut(_text(WORKFLOW), "- name: Require a runner that can host a policy")
    step = step[: step.index("- name: Fetch the Smart App Control audit policies")]
    _has(step, 'if ($LASTEXITCODE -ne 0) { throw "wevtutil sl $log exited', "$maxSize -lt 67108864")


def test_acl_repair_reaches_a_custom_home_outside_the_user_profile():
    r"""An UNSLOTH_STUDIO_HOME on another volume (D:\Unsloth) is exactly where the elevated installer leaves administrator-owned trees."""
    revert = _revert()
    assert "$profileRoot" not in revert
    assert "$allowedRoots = @(" in revert
    roots = _cut(revert, "$allowedRoots = @(")
    roots = roots[: roots.index("$recorded = @()")]
    _has(roots, "$env:USERPROFILE", "$override", "(Get-StudioHome)", "(Get-LlamaDir)")
    _has(
        revert,
        "if ($baseline.StudioInstalledByProbe) { $recorded = @($baseline.StudioInstallRoots) }",
        "$full.StartsWith($_ + '\\', [StringComparison]::OrdinalIgnoreCase)",
    )


_CONTROL_SIGNATURE = "function Test-AuditPolicyEvaluating([int[]] $AcceptIds = @(3076, 3077), [string] $FromPolicy = $null) {"


def test_prepare_proves_the_audit_policy_actually_evaluates_loads():
    """Being in CiTool's policy list is not evidence of evaluation"""
    fn = _ps1("function Test-AuditPolicyEvaluating", "function Invoke-Prepare")
    _has(
        fn,
        "-OutputType ConsoleApplication",
        _CONTROL_SIGNATURE,
        "$AcceptIds -contains $_.Id",
        "foreach ($attempt in 1..10)",
        "Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue",
        '$dir = Join-Path $WorkDir ".control-$Label"',
    )
    prepare = _prepare()
    assert prepare.index("-not (Test-PolicyActive $NOISG_GUID)") < prepare.index(
        "$controlFired = Test-AuditPolicyEvaluating"
    ), "the control runs only once the policy is verified to be in the active set"
    _has(prepare, "if ($false -eq $controlFired) {", "$baseline.AuditPolicyControlFired = $controlFired")
    assert "AuditPolicyControlFired = $null" in _ps1()
    _has(_collect(), "$true -ne $b.AuditPolicyControlFired", "NULL result, not an allow")


def test_only_a_3076_or_3077_counts_as_a_verdict_for_the_null_result_warning():
    """3033 and 3090-3092 are context."""
    collect = _collect()
    _has(
        collect,
        "$verdicts = $blocks + $audits",
        "if ($verdicts -eq 0 -and (Test-Path -LiteralPath $baselineForControl))",
    )
    assert "if ($ours.Count -eq 0 -and (Test-Path -LiteralPath $baselineForControl))" not in collect


def test_a_completed_revert_spends_its_baseline():
    """A later prepare on the same label reused the first run's snapshot."""
    assert "RevertCompletedAt       = $null" in _ps1()
    _has(
        _prepare(),
        "if ($previous -and -not $previous.RevertCompletedAt) {",
        "$baseline = Save-Baseline $dir",
    )
    revert = _revert()
    assert "-NotePropertyName RevertCompletedAt" in revert
    _before(revert, "revert did not fully restore this machine", "-NotePropertyName RevertCompletedAt")


def test_the_app_control_verdict_polls_before_reporting_an_allow():
    """3076 delivery is asynchronous and the exercise step ends the moment llama-server exits."""
    verdict = _verdict()
    _has(
        verdict,
        "foreach ($attempt in 1..10) {",
        "if ($ours.Count -gt 0 -and $ours.Count -eq $seen) { break }",
        "Start-Sleep -Seconds 3",
    )
    _before(verdict, "foreach ($attempt in 1..10) {", '$tail -and $subject -like "*$tail*"')


def test_the_scenario_password_is_cleared_even_when_the_run_is_interrupted():
    """Ctrl+C skips both the normal path and the catch, and the console is the operator's own."""
    run = _run()
    clear = "Remove-Item Env:\\SAC_PROBE_STUDIO_PASSWORD -ErrorAction SilentlyContinue"
    assert run.count(clear) == 1
    _before(run, "} finally {", clear)


def test_a_custom_studio_home_is_normalized_the_way_studio_normalizes_it(s, monkeypatch):
    r"""UNSLOTH_STUDIO_HOME=~\my-studio is supported (storage_roots.studio_root calls expanduser().resolve())."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", "  ~/my-studio  ")
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    assert s.resolve_studio_home(s.default_studio_home()) == Path.home() / "my-studio"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", "   ")
    assert s.default_studio_home() == str(Path.home() / ".unsloth" / "studio")
    ps1 = _ps1()
    _has(ps1, "function Resolve-ConfiguredPath", "function Get-StudioHomeOverride")
    assert "$override = if ($env:UNSLOTH_STUDIO_HOME)" not in ps1
    home = _ps1("function Get-StudioHome {", "function Get-LlamaDir")
    assert "$override = Get-StudioHomeOverride" in home


def test_a_prefix_colliding_sibling_is_not_scoped_as_the_selected_runtime():
    r"""...\llama.cpp-b10830 beside the selected ...\llama.cpp had its 3076/3077 counted as a verdict on the selected build."""
    fn = _ps1("function Get-ScopeTail", "function Get-EventDataMap")
    _has(fn, "TrimEnd('\\', '/')", "WildcardPattern]::Escape($trimmed) + '\\'")
    verdict = _cut(_text(WORKFLOW), "$dir = $env:RUNTIME_DIR")
    assert "($dir -replace '^[A-Za-z]:', '').TrimEnd('\\')) + '\\'" in verdict


def test_a_window_with_no_audit_policy_is_a_null_result_unless_sac_enforces():
    """-AuditPolicy is optional."""
    collect = _collect()
    branch = "} elseif ($sacMode -ne 'enforcement' -or $sacAtPrepare -ne 'enforcement') {"
    _has(collect, "$sacMode = [string]$sacNow.Mode", "$sacAtPrepare = [string]$b.Sac.Mode", branch)
    guard = collect[collect.index(branch) + len(branch) :]
    guard = guard[: guard.index("} elseif ")]
    _has(guard, "$collectionProblems +=", "NULL result, not an allow")


def test_an_empty_window_with_no_observed_load_never_reads_as_an_allow():
    """A scenario that stopped at login loaded no PE under the policy, so the positive control alone must not earn the clean-allow line."""
    collect = _collect()
    marked = "if ($scenarioProblem) { $collectionProblems += $scenarioProblem }"
    _has(collect, "$scenarioProblem = $null", marked, "} elseif (-not $loadOk) {")
    _before(collect, marked, "$stage = Join-Path $WorkDir")
    _before(collect, "$scenarioProblem = $null", "$verdicts = $blocks + $audits")


def test_a_studio_that_never_answered_fails_prepare():
    """Start-Studio's result was discarded, so prepare printed 'prepare complete' over a Studio that never started."""
    init = _init()
    start = "Start-Studio $python $Port $startLog"
    assert "if (-not (Start-Studio $python $Port $startLog)) {" in init
    assert "| Out-Null\n}" not in init.split(start)[-1]
    # prepare fails; run only warns, because its inventories are still evidence.
    assert "if ($allowInstall) {" in init.split(start)[1]
    assert 'throw "Studio did not answer on port $Port within 5 minutes' in init


def test_only_studio_is_treated_as_studio_on_the_port():
    """prepare force-stops whatever owns the port, and 8888 is a busy default."""
    fn = _ps1("function Test-StudioResponding", "function Get-EventDataMap")
    assert "$body.service -eq 'Unsloth UI Backend'" in fn
    assert "return $r.StatusCode -eq 200" not in fn
    main = _text(REPO_ROOT / "studio" / "backend" / "main.py")
    assert '"service": "Unsloth UI Backend"' in _cut(main, '@app.get("/api/liveness")')[:1200]


def test_the_defender_baseline_is_read_back_not_assumed():
    """A tamper-protected or policy-managed preference is ignored rather than refused"""
    prepare = _prepare()
    _has(
        prepare,
        "$applied = Get-MpPreference",
        "but reads back as $actual",
        '$mpFailed += "${name}: set to $expected but reads back as $actual',
    )
    _before(
        prepare,
        "$applied = Get-MpPreference",
        "$mpErrorPath = Join-Path $dir 'defender-preference-errors.txt'",
    )


def test_a_numeric_defender_readback_is_compared_not_waved_through():
    """Get-MpPreference returns CIM numbers on some builds."""
    fn = _ps1("function Test-MpPreferenceMatch", "function Get-MpPreferenceType")
    _has(
        fn,
        "[Enum]::Parse($type, [string]$expected, $true)",
        "[Enum]::Parse($type, [string]$actual, $true)",
        "return $null\n}",
    )
    assert "$cmd.Parameters[$name]" in _ps1()
    _has(
        _prepare(),
        "$same = Test-MpPreferenceMatch $actual $expected (Get-MpPreferenceType $name)",
        "} elseif ($null -eq $same) {",
    )


def test_revert_reads_the_defender_values_back_before_spending_the_baseline():
    """An ignored restore leaves $failed at zero."""
    revert = _revert()
    assert (
        "$same = Test-MpPreferenceMatch $restored.($r.Name) $r.Value (Get-MpPreferenceType $r.Name)"
        in revert
    )
    _before(
        revert,
        "$restored = Get-MpPreference",
        "$restoreFailures = $failed",
        "-NotePropertyName RevertCompletedAt",
    )


def test_install_roots_survive_a_prepare_retry():
    """A retry finds the first attempt's trees already present."""
    init = _init()
    assert "$b.StudioInstallRoots = @(@($b.StudioInstallRoots) + $created |" in init
    assert "$b.StudioInstallRoots = $created" not in init


def test_an_installer_that_produced_no_interpreter_fails_prepare():
    tail = _cut(_init(), "if (-not $python) {\n            # prepare cannot go on")
    _has(
        tail,
        "if ($allowInstall) {",
        "throw 'the installer ran but produced no managed interpreter",
        "Write-Warning 'Studio still not found after the installer ran.",
    )


def test_the_ci_verdict_needs_a_runtime_that_was_actually_extracted():
    """The verdict runs on always(), so a download or expand failure left a clean verdict."""
    verdict = _verdict()
    _has(verdict, "if (-not $dir) {", "::error::the shipped runtime was never extracted")
    _before(verdict, "if (-not $dir) {", "foreach ($attempt in 1..10) {")


def test_the_scenario_will_not_hand_its_password_to_an_unidentified_server(s, monkeypatch):
    """discover_port picks the port the operator's password is posted to, and 8888 is Jupyter's."""

    def fake(base_url, method, path, payload = None, token = None, timeout = 900):
        if base_url.endswith(":8888"):
            return 200, {"status": "alive"}
        if base_url.endswith(":8890"):
            return 200, {"status": "alive", "service": "Unsloth UI Backend"}
        return 0, "refused"

    monkeypatch.setattr(s, "_request", fake)
    assert s.discover_port(None) == 8890
    with pytest.raises(SystemExit):
        s.discover_port(8888)


def test_an_efi_mount_this_probe_owns_is_retried_by_the_next_stage():
    """$script:EfiStillMounted dies with the process."""
    assert "$EFI_OWNED_MARKER = Join-Path $WorkDir '.efi-mounted-by-probe'" in _ps1()
    mount = _ps1("function Mount-Efi", "$script:EfiStillMounted = $false")
    _has(mount, "if (Test-Path -LiteralPath $EFI_OWNED_MARKER) {", "Set-Content -LiteralPath $EFI_OWNED_MARKER")
    _before(mount, "Test-Path -LiteralPath $EFI_OWNED_MARKER", "return $false")
    dismount = _ps1("function Dismount-Efi", "function Test-PolicyActive")
    assert "Remove-Item -LiteralPath $EFI_OWNED_MARKER" in dismount


def test_a_lost_preexisting_policy_backup_is_a_revert_failure():
    """Falling through to the removal branch deleted an administrator's policy."""
    revert = _revert()
    assert (
        "if ($baseline.AuditPolicyPreexisting -and -not (Test-Path -LiteralPath $saved)) {"
        in revert
    )
    guard = _cut(revert, "$baseline.AuditPolicyPreexisting -and -not")
    _before(guard, 'throw "the baseline says a policy with $NOISG_GUID', "Remove-Item -LiteralPath $NOISG_DEST")


def test_prepare_requires_the_log_capacity_it_asked_for():
    """The 1 MB default wraps while every venv load is audited"""
    prepare = _prepare()
    assert "$ciNow.MaxSize -lt 67108864" in prepare
    _before(prepare, "$ciNow.MaxSize -lt 67108864", "Initialize-Studio $dir $true")


def test_run_revalidates_the_policy_and_the_control_after_a_reboot():
    """Only the file on the EFI partition survives a reboot."""
    run = _run()
    _has(
        run,
        "if (-not (Test-PolicyActive $NOISG_GUID)) {",
        "LastBootUpTime",
        "$controlFired = Test-AuditPolicyEvaluating",
        "$runBaseline.AuditPolicyControlFired = $controlFired",
    )
    _before(run, "Test-PolicyActive $NOISG_GUID", "Write-Section 'Venv signature inventory'")


def test_a_failed_dismount_is_reclaimed_by_the_next_revert():
    """The policy block clears AuditPolicyApplied even when the dismount after the refresh failed."""
    assert "function Clear-EfiOwnership {" in _ps1()
    reclaim = _ps1("function Clear-EfiOwnership", "function Test-PolicyActive")
    # Reclaims only a drive still pointing at the partition we left; never mounts.
    assert "mountvol.exe" not in reclaim
    _has(reclaim, "Dismount-Efi $true", "Test-Path -LiteralPath 'S:\\EFI\\Microsoft\\Boot'")
    revert = _revert()
    assert "Clear-EfiOwnership" in revert
    _before(revert, "Clear-EfiOwnership", "if ($baseline.AuditPolicyApplied) {")


def test_a_recorded_tree_revert_refuses_to_touch_is_a_rollback_failure():
    """A UNSLOTH_STUDIO_HOME set for the prepare shell only is gone after a reboot."""
    revert = _revert()
    _has(
        revert,
        "$rejected += $full",
        "$aclFailures = $rejected.Count",
        "if (-not (Test-Path -LiteralPath $path)) { continue }",
        "if ($trees.Count -eq 0 -and $rejected.Count -eq 0) {",
    )
    _before(revert, "$aclFailures = $rejected.Count", "foreach ($tree in $trees)")


def test_an_allow_needs_live_enforcement_and_a_control_on_the_boot_that_measured():
    """Without -AuditPolicy the allow verdict rested on Sac.Mode as prepare read it."""
    collect = _collect()
    fired = "} elseif ($true -ne $b.SacControlFired) {"
    _has(
        collect,
        "$sacMode = [string]$sacNow.Mode",
        "$sacAtPrepare = [string]$b.Sac.Mode",
        fired,
        "$sacNow | ConvertTo-Json -Depth 6 |",
    )
    _before(collect, "$sacNow = Get-SacState", "$sacMode = [string]$sacNow.Mode")
    _before(collect, fired, "so this window is a real allow")
    assert "Get-SacState | ConvertTo-Json" not in collect
    assert "SacControlFired         = $null" in _ps1()[: _ps1().index("function Invoke-Run")]
    prepare = _prepare()
    _has(prepare, "$sacBefore = Get-SacState", "$sacFired = Test-AuditPolicyEvaluating")
    _before(prepare, "$sacFired = Test-AuditPolicyEvaluating", "'window-start.txt'")
    _has(
        _run(),
        "} elseif ([string]$runBaseline.Sac.Mode -eq 'enforcement') {",
        "if ($sacNow.Mode -ne 'enforcement') {",
        "($true -ne $runBaseline.SacControlFired)",
        "-NotePropertyName SacControlFired",
    )


def test_the_readme_pins_the_runtime_before_the_window_opens():
    """Installing the pinned release after prepare put loads from two releases in one window."""
    pin = _cut(
        _text(PROBE_DIR / "README.md"),
        "To pin a specific runtime for a cell",
        "## Reading the output",
    )
    _has(pin, "before `prepare`", "-Stage prepare -Label custom-b10715-sac-on", "again\nbefore `prepare`")
    _before(
        pin,
        "-Stage prepare -Label custom-b10715-sac-on",
        "-Stage run     -Label custom-b10715-sac-on",
    )
    assert "restart Studio, then:" not in pin


def test_a_spent_baseline_is_never_written_back_over_the_machine():
    """revert reapplied a spent baseline, silently undoing anything changed since, including lowering Defender protections."""
    revert = _revert()
    guard = "if ($baseline.RevertCompletedAt) {"
    assert guard in revert
    _before(revert, "Clear-EfiOwnership", guard)
    _before(
        revert,
        guard,
        "Write-Section 'Restore CodeIntegrity log'",
        "if ($baseline.AuditPolicyApplied) {",
    )


def test_the_sac_positive_control_counts_only_an_enforced_refusal():
    """The audit-policy path wants a 3076 or a 3077"""
    ps1 = _ps1()
    assert _CONTROL_SIGNATURE in ps1
    fn = _ps1("function Test-AuditPolicyEvaluating", "function Invoke-Prepare")
    assert "$AcceptIds -contains $_.Id" in fn
    assert "$_.Id -eq 3076 -or $_.Id -eq 3077" not in fn
    # Both Smart App Control call sites pin 3077; the audit-policy ones do not.
    assert ps1.count("Test-AuditPolicyEvaluating -AcceptIds @(3077)") == 2
    for stage in (_prepare(), _run()):
        _has(
            stage,
            "$controlFired = Test-AuditPolicyEvaluating -FromPolicy $NOISG_GUID\n",
            "$sacFired = Test-AuditPolicyEvaluating -AcceptIds @(3077)",
        )
        assert "without raising a 3076 or 3077, so nothing is refusing" not in stage
    assert ps1.count("without being refused with a 3077") == 2


def test_only_the_verified_loopback_listener_is_ever_stopped():
    """-LocalPort on its own returns every listener holding that port on any local address."""
    stop = _ps1("function Stop-Studio", "function Initialize-Studio")
    assert "$_.LocalAddress -eq '127.0.0.1' -or $_.LocalAddress -eq '0.0.0.0'" in stop
    _before(stop, "LocalAddress", "Stop-Process", "if (-not $owners) {")
    assert "http://127.0.0.1:$port/api/liveness" in _ps1()


def test_the_managed_venv_is_recorded_when_the_studio_home_already_exists():
    r"""The elevated installer creates <home>\unsloth_studio, which was in no candidate list."""
    cands = _cut(_init(), "$candidates = @(", "$absentBefore = @(")
    assert "(Join-Path (Get-StudioHome) 'unsloth_studio')" in cands
    assert "(Get-StudioHome)" in _cut(_revert(), "$allowedRoots = @(", "$recorded = @()")


def test_run_refuses_a_label_whose_revert_already_completed():
    """revert leaves window-start.txt behind."""
    run = _run()
    guard = "if ($runBaseline.RevertCompletedAt) {"
    assert guard in run
    _before(run, guard, "if ($runBaseline.AuditPolicyApplied) {", "Write-Section 'Venv signature inventory'")
    assert "if ($baseline.RevertCompletedAt) {" in _revert()


def test_collect_scopes_the_venv_run_measured_not_the_one_this_shell_resolves():
    """A collect from a shell without the UNSLOTH_STUDIO_HOME override scoped against the legacy default."""
    assert "function Resolve-VenvDir([string] $dir) {" in _ps1()
    fn = _ps1("function Resolve-VenvDir", "function Test-StudioResponding")
    assert "$recorded = Join-Path $dir 'venv-selection.txt'" in fn
    _before(fn, "$recorded = Join-Path", "$studioPython = Get-StudioPython")
    run = _run()
    assert "$venvDir | Set-Content -LiteralPath (Join-Path $dir 'venv-selection.txt')" in run
    _before(run, "$venvDir = Resolve-VenvDir", "'venv-selection.txt'")
    stale = _cut(_prepare(), "foreach ($stale in @(")
    assert "'venv-selection.txt'" in stale[: stale.index(")) {")]


def test_the_prepare_timestamp_is_read_back_culture_invariantly():
    """[datetime]::Parse reads the CURRENT culture, not the invariant one PowerShell stringified with."""
    run = _run()
    assert "[datetime]::Parse($runBaseline.CapturedAt)" not in run
    assert run.count("[datetime]::Parse([string]$runBaseline.CapturedAt,") == 3
    assert run.count("[cultureinfo]::InvariantCulture") == 4
    assert run.count("$bootedAt -and $preparedAt -and $bootedAt -gt $preparedAt") == 2


def test_the_streamed_turns_opt_into_the_tool_control_frames(s, monkeypatch):
    """/v1/chat/completions emits a clean OpenAI stream for external clients"""
    backend = _text(REPO_ROOT / "studio" / "backend" / "routes" / "inference.py")
    assert 'UI_STREAM_EVENTS_HEADER = "X-Unsloth-Events"' in backend
    chat_api = REPO_ROOT / "studio" / "frontend" / "src" / "features" / "chat" / "api" / "chat-api.ts"
    assert '"X-Unsloth-Events": "1"' in _text(chat_api)
    seen: dict[str, str] = {}
    lines = [
        'data: {"type": "tool_end", "tool_name": "web_search", "result": "x"}\n',
        "data: [DONE]\n",
    ]

    def fake_urlopen(req, timeout = None):
        seen.update(req.headers)
        return _FakeStream(lines)

    monkeypatch.setattr(s.urllib.request, "urlopen", fake_urlopen)
    status, events, error = s._stream_events("http://x", "/v1/chat/completions", {}, token = "t")
    assert seen.get("X-unsloth-events") == "1", seen
    assert status == 200 and error is None
    assert any(e.get("type") == "tool_end" for e in events)


def test_collect_reads_the_studio_logs_of_the_install_run_measured():
    """Both the log root and the redactor interpreter went through Get-StudioHome."""
    _has(
        _ps1(),
        "function Resolve-StudioHomeFor([string] $dir) {",
        "function Resolve-StudioPythonFor([string] $dir) {",
    )
    helpers = _ps1("function Resolve-StudioHomeFor", "function Test-StudioResponding")
    assert helpers.count("Resolve-VenvDir $dir") == 2
    _has(helpers, "return (Get-StudioHome)", "return (Get-StudioPython)")
    collect = _collect()
    _has(
        collect,
        "$studioLogs = Join-Path (Resolve-StudioHomeFor $dir) 'logs'",
        "$python = Resolve-StudioPythonFor $dir",
    )
    assert "Join-Path (Get-StudioHome) 'logs'" not in collect


def test_run_refuses_an_event_window_older_than_its_own_baseline():
    """A prepare that throws between the fresh baseline and the reopened window leaves an unspent baseline beside the old window-start.txt."""
    run = _run()
    _has(
        run,
        "$runStartPath = Join-Path $dir 'window-start.txt'",
        "$runStart -and $runPrepared -and $runStart -lt $runPrepared",
    )
    _before(run, "$runStartPath = Join-Path", "Write-Section 'Venv signature inventory'")
    _before(run, "if ($runBaseline.RevertCompletedAt) {", "$runStartPath = Join-Path")
    assert "$runPrepared" not in _collect(), "collect must NOT gain the same guard"
    prepare = _prepare()
    assert prepare.rindex("Test-AuditPolicyEvaluating") < prepare.index("'window-start.txt'")


def test_the_unattended_tool_turns_never_wait_on_an_approval_nobody_reads():
    """Opting into the control frames also opens the confirm gate"""
    scenario = _text(SCENARIO)
    tools = _cut(scenario, "    if tools:", '        text = ""')
    assert 'payload["permission_mode"] = "off"' in tools
    _lacks(scenario, 'payload["permission_mode"] = "full"', 'payload["bypass_permissions"]')
    models = _text(REPO_ROOT / "studio" / "backend" / "models" / "inference.py")
    assert '_KNOWN_PERMISSION_MODES = ("ask", "auto", "off", "full")' in models
    backend = _text(REPO_ROOT / "studio" / "backend" / "routes" / "inference.py")
    assert 'if mode in ("off", "full"):' in backend


def test_a_runtime_at_the_root_of_a_volume_is_refused_not_matched_against_everything():
    r"""UNSLOTH_LLAMA_CPP_PATH=D:\ left a lone-separator tail that every path in the machine-wide channel contains."""
    fn = _ps1("function Get-ScopeTail", "function Get-EventDataMap")
    _before(fn, "if (-not $trimmed) {", "WildcardPattern]::Escape($trimmed)")
    assert "throw" in _cut(fn, "if (-not $trimmed) {", "return (")
    _has(
        fn,
        "($root -replace '^[A-Za-z]:', '').TrimEnd('\\', '/')",
        "WildcardPattern]::Escape($trimmed) + '\\'",
    )
    _has(
        _collect(),
        "$tail = Get-ScopeTail (Resolve-LlamaDir $dir)",
        "$venvTail = Get-ScopeTail (Resolve-VenvDir $dir)",
    )


def test_revert_stops_only_the_elevated_studio_this_probe_started(tmp_path):
    """The Studio that prepare/run starts holds an administrator token."""
    _has(_revert(), "Stop-ProbeStudio $dir", "$studioStillRunning -or")
    fake = tmp_path / "fakepy"
    fake.write_text("#!/bin/sh\nexec sleep 300\n")
    fake.chmod(0o755)
    run = tmp_path / "run"
    run.mkdir()
    body = r"""
function Get-RunDir { $Run }
function Test-StudioResponding($p){ $true }
function Start-Sleep { }
function Get-CimInstance { }
function Start-Process { param($FilePath,$ArgumentList,$RedirectStandardOutput,$RedirectStandardError,$WindowStyle,[switch]$PassThru)
  Microsoft.PowerShell.Management\Start-Process -FilePath $FilePath -ArgumentList $ArgumentList `
    -RedirectStandardOutput $RedirectStandardOutput -RedirectStandardError $RedirectStandardError -PassThru:$PassThru }
Start-Studio $Py 8888 "$Run/log" | Out-Null
$r = Get-Content "$Run/probe-studio.json" -Raw | ConvertFrom-Json
if (-not (Stop-ProbeStudio $Run)) { exit 11 }
if (Get-Process -Id $r.Id -ErrorAction SilentlyContinue) { exit 12 }
Start-Studio $Py 8888 "$Run/log" | Out-Null
$r = Get-Content "$Run/probe-studio.json" -Raw | ConvertFrom-Json
@{ Id = $r.Id; StartTicks = 1 } | ConvertTo-Json | Set-Content "$Run/probe-studio.json"
[void](Stop-ProbeStudio $Run)
$alive = [bool](Get-Process -Id $r.Id -ErrorAction SilentlyContinue)
Stop-Process -Id $r.Id -Force -ErrorAction SilentlyContinue
if (-not $alive) { exit 13 }
exit 0
"""
    names = ["Start-Studio", "Stop-ProcessTree", "Stop-ProbeStudio"]
    _drive(tmp_path, names, body, strict = False, Run = run, Py = fake)


def test_run_refuses_a_label_that_was_never_prepared():
    """Get-RunDir creates the directory, so a new or mistyped -Label reached run empty."""
    run = _run()
    guard = "-not (Test-Path -LiteralPath (Join-Path $dir 'baseline.json'))"
    assert "-not (Test-Path -LiteralPath (Join-Path $dir 'window-start.txt'))" in run
    _before(
        run,
        guard,
        "throw \"label '$Label' has no baseline.json",
        "Initialize-Studio $dir $false",
        "Write-Section 'Venv signature inventory'",
    )


def test_a_second_studio_launch_stops_the_first_probe_studio():
    """probe-studio.json holds one record."""
    fn = _ps1("function Start-Studio", "function Stop-Studio")
    stop = "if (-not (Stop-ProbeStudio (Get-RunDir))) {"
    _before(fn, stop, "$proc = Start-Process", "'probe-studio.json'")
    assert "throw" in _cut(fn, stop, "$proc = Start-Process")


def test_a_reused_label_never_archives_an_earlier_upgrade_transcript():
    """collect stages the whole run directory, and the stale sweep runs after the winget block."""
    _before(
        _prepare(),
        "Remove-Item -LiteralPath (Join-Path $dir 'winget-upgrade.log')",
        "if ($UpgradePackages) {",
        "Tee-Object -FilePath (Join-Path $dir 'winget-upgrade.log')",
    )


def test_events_are_scoped_by_the_evaluated_file_not_the_requesting_process():
    """A 3076/3077 message names both the requesting process and the file."""
    shaped = _cut(_collect(), "$shaped = @($events", "$scopeByActivity = @{}")
    _has(
        shaped,
        "foreach ($field in @('File Name', 'FileNameBuffer'))",
        'if ($subject -like "*$tail*")',
        'elseif ($subject -like "*$venvTail*")',
        "$subject = $msg",
        "EventData   = $data",
    )
    assert "$msg -like" not in shaped


def test_the_app_control_verdict_scopes_by_the_evaluated_file(tmp_path):
    """Matching the whole 3076 message counted an in-runtime llama-server loading an outside file as a runtime file blocked."""
    verdict = _verdict()
    start = verdict.index("            $ours = @($events | Where-Object {")
    end = verdict.index("            })\n", start) + len("            })\n")
    snippet = verdict[start:end]
    assert "$_.Message -like" not in snippet
    _has(
        snippet,
        "'File Name'",
        "'FileNameBuffer'",
        "([xml] $_.ToXml()).Event.EventData.Data",
        '$tail -and $subject -like "*$tail*"',
    )
    driver = tmp_path / "drive.ps1"
    driver.write_text(
        r"""
param([string]$Snippet)
$fromPolicy = { $true }
function New-Ev([string]$msg, [hashtable]$data) {
  $x = '<Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event"><EventData>'
  foreach ($k in $data.Keys) { $x += "<Data Name='$k'>$([Security.SecurityElement]::Escape($data[$k]))</Data>" }
  $x += '</EventData></Event>'
  $o = [pscustomobject]@{ Message = $msg; Tag = '' }
  $o | Add-Member -MemberType ScriptMethod -Name ToXml -Value ([scriptblock]::Create("'" + $x.Replace("'", "''") + "'"))
  $o
}
$tail = [Management.Automation.WildcardPattern]::Escape('\a\rt') + '\'
$rt = '\Device\HarddiskVolume3\a\rt\'
$out = '\Device\HarddiskVolume3\elsewhere\x.dll'
$e1 = New-Ev "Process $($rt)llama-server.exe attempted to load $out" @{ 'File Name' = $out; 'Process Name' = "$($rt)llama-server.exe" }
$e1.Tag = 'outside'
$e2 = New-Ev "Process C:\w\pwsh.exe attempted to load $($rt)ggml.dll" @{ 'File Name' = "$($rt)ggml.dll"; 'Process Name' = 'C:\w\pwsh.exe' }
$e2.Tag = 'inside'
$e3 = New-Ev "Process $($rt)llama-server.exe attempted to load $out" @{ 'FileNameBuffer' = $out }
$e3.Tag = 'outside-buffer'
$e4 = New-Ev "no data, loaded $($rt)cuda.dll" @{ }
$e4.Tag = 'fallback'
$events = @($e1, $e2, $e3, $e4)
. ([scriptblock]::Create($Snippet))
($ours | ForEach-Object Tag) -join ','
""",
        encoding = "utf-8",
    )
    proc = _run_ps(driver, "-Snippet", snippet)
    _assert_ok(proc)
    assert proc.stdout.strip() == "inside,fallback", proc.stdout + proc.stderr


def test_a_studio_whose_record_cannot_be_written_is_stopped_and_fails(tmp_path):
    """An elevated Studio with no probe-studio.json is one revert can neither stop nor report."""
    body = r"""
$global:stopped = @()
function Get-RunDir { $Run }
function Stop-ProbeStudio($d){ $true }
function Test-StudioResponding($p){ $true }
function Start-Sleep { }
function Stop-ProcessTree([int]$id){ $global:stopped += $id }
function Start-Process { [pscustomobject]@{ Id = 4242; StartTime = [datetime]::UtcNow } }
function Set-Content { throw 'disk full' }
$threw = $false
try { Start-Studio 'py' 8888 "$Run/log" | Out-Null } catch { $threw = $true; Write-Host $_ }
if (-not $threw) { exit 21 }
if ($global:stopped -notcontains 4242) { exit 22 }
exit 0
"""
    _drive(tmp_path, ["Start-Studio"], body, strict = False, Run = tmp_path)


def test_the_audit_policy_control_needs_an_event_from_the_audit_policy_itself(tmp_path):
    """On a Smart App Control enforcing machine the unsigned control is refused by Smart App Control's own policy with a 3077 whether or not the audit policy evaluates anything."""
    for stage in (_prepare(), _run()):
        _has(
            stage,
            "$controlFired = Test-AuditPolicyEvaluating -FromPolicy $NOISG_GUID",
            "$sacFired = Test-AuditPolicyEvaluating -AcceptIds @(3077)",
        )
    body = r"""
$WorkDir = $Work; $Label = 't'
$NOISG = '{5283AC0F-FFF1-49AE-ADA1-8A933130CAD6}'
function Add-Type { param($TypeDefinition,$OutputAssembly,$OutputType)
  Set-Content -LiteralPath $OutputAssembly -Value "#!/bin/sh`ntrue"; chmod +x $OutputAssembly }
function New-Ev([int]$id, [hashtable]$data) {
  $x = '<Event><EventData>' + (($data.GetEnumerator() | ForEach-Object { "<Data Name='$($_.Key)'>$($_.Value)</Data>" }) -join '') + '</EventData></Event>'
  $e = [pscustomobject]@{ Id = $id; Message = 'a process attempted to load C:\x\unsigned-control.exe' }
  $e | Add-Member -MemberType ScriptMethod -Name ToXml -Value ({ $x }.GetNewClosure())
  $e }
$global:evs = @(New-Ev 3077 @{ 'File Name' = 'C:\x\unsigned-control.exe'; PolicyGUID = '{0283AC0F-FFF1-49AE-ADA1-8A933130CAD6}'; PolicyNameBuffer = 'VerifiedAndReputableDesktop' })
function Get-WinEvent { $global:evs }
if ($true -eq (Test-AuditPolicyEvaluating -FromPolicy $NOISG)) { exit 31 }
if ($true -ne (Test-AuditPolicyEvaluating -AcceptIds @(3077))) { exit 32 }
$global:evs = @(New-Ev 3076 @{ 'File Name' = 'C:\x\unsigned-control.exe'; PolicyGUID = $NOISG.ToLower() })
if ($true -ne (Test-AuditPolicyEvaluating -FromPolicy $NOISG)) { exit 33 }
$global:evs = @(New-Ev 3076 @{ 'File Name' = 'C:\x\unsigned-control.exe'; PolicyNameBuffer = 'VerifiedAndReputableDesktopEvaluationAuditNoISG' })
if ($true -ne (Test-AuditPolicyEvaluating -FromPolicy $NOISG)) { exit 34 }
exit 0
"""
    names = [
        "Get-EventDataMap",
        "Test-EventFromPolicy",
        "Test-EventDataFromPolicy",
        "Test-AuditPolicyEvaluating",
    ]
    _drive(tmp_path, names, body, Work = tmp_path)


def test_collect_waits_for_code_integrity_delivery_to_settle(tmp_path):
    """The channel is written asynchronously, so a single early snapshot could grade zero events."""
    collect = _collect()
    assert "$events = @(Read-SettledCiEvents $start $CI_EVENT_IDS)" in collect
    assert "Get-WinEvent -FilterHashtable" not in collect[: collect.index("$shaped = @($events")]
    body = r"""
$global:reads = 0
$global:script = @('none', 1, 3, 3, 3)
function Get-WinEvent { param($FilterHashtable, $ErrorAction)
  $n = $global:script[[math]::Min($global:reads, $global:script.Count - 1)]; $global:reads++
  if ($n -eq 'none') { Write-Error -ErrorId 'NoMatchingEventsFound' -Message 'No events were found' -ErrorAction Stop }
  if ($n -eq 'denied') { Write-Error -ErrorId 'AccessDenied' -Message 'denied' -ErrorAction Stop }
  foreach ($i in 1..$n) { [pscustomobject]@{ Id = 3076 } }
  [pscustomobject]@{ Id = 3000 } }
$got = @(Read-SettledCiEvents ([datetime]::UtcNow) @(3076, 3077))
if ($got.Count -ne 3) { Write-Host "got $($got.Count) after $global:reads reads"; exit 41 }
# Two equal empty reads 3 s apart are not settled: polling runs at least the 30 s floor.
$global:slept = 0
function Start-Sleep { param([int]$Seconds) $global:slept += $Seconds }
$global:reads = 0; $global:script = @('none', 'none', 'none', 'none', 'none', 1)
$got = @(Read-SettledCiEvents ([datetime]::UtcNow) @(3076, 3077))
if ($got.Count -ne 1) { Write-Host "late record missed: got $($got.Count) after $global:reads reads"; exit 44 }
if ($global:slept -lt 30) { Write-Host "settled after only $global:slept s"; exit 45 }
$global:reads = 0; $global:slept = 0; $global:script = @(1..100)
[void](Read-SettledCiEvents ([datetime]::UtcNow) @(3076))
if ($global:reads -gt 25 -or $global:slept -gt 75) { exit 46 }
$global:reads = 0; $global:script = @(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
$got = @(Read-SettledCiEvents ([datetime]::UtcNow) @(3076) -Attempts 5)
if ($global:reads -ne 5 -or $got.Count -ne 5) { exit 42 }
$global:reads = 0; $global:script = @('denied')
$threw = $false
try { [void](Read-SettledCiEvents ([datetime]::UtcNow) @(3076)) } catch { $threw = $true }
if (-not $threw) { exit 43 }
exit 0
"""
    _drive(tmp_path, ["Read-SettledCiEvents"], body, Work = tmp_path)


def test_the_ci_verdict_needs_a_completed_exercise_before_an_allow():
    """A --version that exited non-zero without raising a 3076 used to reach the clean verdict."""
    exercise = _cut(_text(WORKFLOW), "      - name: Load the shipped runtime under the policy")
    exercise = exercise[: exercise.index("      - name: Verdict")]
    assert '"EXERCISE_EXIT=$code"' in exercise
    _before(exercise, "$code = $LASTEXITCODE", '"EXERCISE_EXIT=$code"')
    verdict = _verdict()
    gate = "if ($ours.Count -eq 0 -and $env:EXERCISE_EXIT -ne '0') {"
    _before(verdict, "foreach ($attempt in 1..10) {", gate)
    _before(
        verdict,
        gate,
        "No binary in the shipped runtime would be refused",
        '$summary = "## App Control audit (signature-only)',
    )
    assert "exit 1" in _cut(verdict, gate, '$summary = "## App Control audit')


def test_collect_counts_only_the_installed_policys_audit_events(tmp_path):
    """Another audit-mode policy already on the machine logs 3076s too."""
    collect = _collect()
    _has(
        collect,
        "Test-EventDataFromPolicy $e.EventData $NOISG_GUID",
        "$audits = @($ours | Where-Object { & $isOurAudit $_ }).Count",
        '$collectionProblems += "$otherPolicyAudits audit event(s)',
    )
    assert "$audits = @($ours | Where-Object { $_.Id -eq 3076 }).Count" not in collect
    start = collect.index("    $isOurAudit = {")
    predicate = collect[start : collect.index("\n    }\n", start) + 6]
    body = r"""
$NOISG = '{5283AC0F-FFF1-49AE-ADA1-8A933130CAD6}'
$ours = [ordered]@{ 'File Name' = 'C:\s\llama.cpp\ggml.dll'; PolicyGUID = $NOISG.ToLower() }
$byName = [ordered]@{ 'File Name' = 'C:\s\llama.cpp\ggml.dll'; PolicyNameBuffer = 'VerifiedAndReputableDesktopEvaluationAuditNoISG' }
$other = [ordered]@{ 'File Name' = 'C:\s\llama.cpp\ggml.dll'; PolicyGUID = '{11111111-2222-3333-4444-555555555555}'; PolicyNameBuffer = 'ContosoAllowList' }
if ($true -ne (Test-EventDataFromPolicy $ours $NOISG)) { exit 51 }
if ($true -ne (Test-EventDataFromPolicy $byName $NOISG)) { exit 52 }
if ($true -eq (Test-EventDataFromPolicy $other $NOISG)) { exit 53 }
if ($true -eq (Test-EventDataFromPolicy $null $NOISG)) { exit 54 }
"""
    body += predicate + r"""
$NOISG_GUID = $NOISG
$evOurs = [pscustomobject]@{ Id = 3076; EventData = $ours }
$evOther = [pscustomobject]@{ Id = 3076; EventData = $other }
$auditApplied = $true
if ($true -ne (& $isOurAudit $evOurs)) { exit 55 }
if ($true -eq (& $isOurAudit $evOther)) { exit 56 }
$auditApplied = $false
if ($true -eq (& $isOurAudit $evOurs)) { exit 57 }
if ($true -eq (& $isOurAudit $evOther)) { exit 58 }
exit 0
"""
    _drive(tmp_path, ["Test-EventDataFromPolicy"], body, Work = tmp_path)


def test_a_baseline_write_that_dies_keeps_the_previous_snapshot(tmp_path):
    """revert restores from baseline.json, so a rewrite that fails part way must leave the last good snapshot."""
    ps1 = _ps1()
    rest = ps1.replace(_ps1("function Save-ProbeBaseline", "function Test-EventFromPolicy"), "")
    assert "ConvertTo-Json -Depth 6 | Set-Content" not in rest
    assert rest.count("Save-ProbeBaseline ") >= 9
    body = r"""
$path = Join-Path $Work 'baseline.json'
Save-ProbeBaseline ([pscustomobject]@{ Label = 'first'; AuditPolicyApplied = $true }) $path
if ((Get-Content -LiteralPath $path -Raw | ConvertFrom-Json).Label -ne 'first') { exit 61 }
Save-ProbeBaseline ([pscustomobject]@{ Label = 'second'; AuditPolicyApplied = $true }) $path
if ((Get-Content -LiteralPath $path -Raw | ConvertFrom-Json).Label -ne 'second') { exit 62 }
if (Test-Path -LiteralPath "$path.tmp") { exit 63 }
function Set-Content {
  param([string]$LiteralPath, [string]$Encoding, [Parameter(ValueFromPipeline = $true)]$InputObject)
  process {
    [System.IO.File]::WriteAllText($LiteralPath, '{"Lab')
    throw 'disk full'
  }
}
try { Save-ProbeBaseline ([pscustomobject]@{ Label = 'third' }) $path; exit 64 } catch { }
$after = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
if ($after.Label -ne 'second' -or $true -ne $after.AuditPolicyApplied) { exit 65 }
exit 0
"""
    _drive(tmp_path, ["Save-ProbeBaseline"], body, Work = tmp_path)


def test_collect_reads_the_studio_home_run_recorded(tmp_path):
    """With a custom home that has no interpreter of its own, the venv is the legacy one while logs go under the configured home."""
    run = _run()
    assert "Get-StudioHome | Set-Content -LiteralPath (Join-Path $dir 'studio-home.txt')" in run
    assert "'venv-selection.txt', 'studio-home.txt'," in _ps1()
    body = r"""
$legacyHome = Join-Path $Work 'legacy-home'
$script:legacyVenv = Join-Path $legacyHome 'unsloth_studio'
function Resolve-VenvDir([string] $dir) { return $script:legacyVenv }
function Get-StudioHome { return 'C:\live' }
$d = Join-Path $Work 'label'
New-Item -ItemType Directory -Force -Path $d | Out-Null
if ((Resolve-StudioHomeFor $d) -ne $legacyHome) { exit 71 }
Set-Content -LiteralPath (Join-Path $d 'studio-home.txt') -Value 'D:\custom-home' -Encoding UTF8
if ((Resolve-StudioHomeFor $d) -ne 'D:\custom-home') { exit 72 }
exit 0
"""
    _drive(tmp_path, ["Resolve-StudioHomeFor"], body, Work = tmp_path)


def test_a_stall_is_measured_through_the_read_that_ended_it(s):
    """Seven abandoned 10 s reads and a reply 6 s into the next one is a 76 s stall, past the watchdog's ~75 s budget."""
    poller = s.StatusPoller("http://127.0.0.1:1", "t")
    poller.polls = [(10.0 * i, 10_000.0, True) for i in range(7)] + [(70.0, 6_000.0, False)]
    assert [round(x) for x in poller.stalls_ms()] == [76_000]


def test_the_tool_error_prefixes_mirror_the_backend_set(s):
    """The loop reports a failed tool with any of the backend's error prefixes, not only "Error:"."""
    import ast

    parser = REPO_ROOT / "studio" / "backend" / "core" / "inference" / "tool_call_parser.py"
    backend = next(
        ast.literal_eval(node.value)
        for node in ast.parse(_text(parser)).body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "TOOL_ERROR_PREFIXES" for t in node.targets)
    )
    assert tuple(s.TOOL_ERROR_PREFIXES) == tuple(backend)
    for result in (
        *(prefix + " something went wrong" for prefix in backend),
        "Search failed: HTTP 429",
        "Execution error: NameError",
        "Blocked: the sandbox refused this path",
        "Failed to fetch https://example.com",
    ):
        assert s.tool_end_failure(result) is not None, result


def test_an_expired_session_is_renewed_with_the_refresh_token(s, tmp_path, monkeypatch):
    """Studio's access token lasts 60 minutes and a scenario can outlive it"""
    live = {"access": "a1", "refresh": "r1"}
    refreshes: list[str] = []

    def once(base_url, method, path, payload, token, timeout):
        if path == "/api/auth/login":
            return 200, {"access_token": "a1", "refresh_token": "r1"}
        if path == "/api/auth/refresh":
            refreshes.append(payload["refresh_token"])
            if payload["refresh_token"] != live["refresh"]:
                return 401, "Invalid or expired refresh token"
            n = len(refreshes) + 1
            live.update(access = f"a{n}", refresh = f"r{n}")
            return 200, {"access_token": f"a{n}", "refresh_token": f"r{n}"}
        return (200, {"ok": token}) if token == live["access"] else (401, "expired")

    def status(token):
        return s._request("http://x", "GET", "/api/inference/status", token = token)

    monkeypatch.setattr(s, "_request_once", once)
    creds = s.authenticate("http://x", tmp_path, "pw")
    assert isinstance(creds, s.Credentials) and creds.refresh == "r1"
    assert status(creds) == (200, {"ok": "a1"})
    live["access"] = "expired-by-server"
    assert status(creds) == (200, {"ok": "a2"})
    assert refreshes == ["r1"] and creds.refresh == "r2"
    assert creds.renew("http://x", "a1") is True and refreshes == ["r1"]
    streamed: list[str] = []

    def stream_once(base_url, path, payload, token, timeout):
        streamed.append(token)
        return (200, [], None) if token == live["access"] else (401, [], "expired")

    monkeypatch.setattr(s, "_stream_events_once", stream_once)
    live["access"] = "expired-again"
    assert s._stream_events("http://x", "/v1/chat/completions", {}, creds)[0] == 200
    assert streamed == ["a2", "a3"] and refreshes == ["r1", "r2"]
    assert status("stale")[0] == 401


def test_a_transient_refresh_failure_keeps_the_refresh_token(s, monkeypatch):
    """A transport error or a 5xx from /api/auth/refresh may never have reached Studio's store, so the single-use token is kept."""
    answers = [
        (0, "connection reset"),
        (503, "busy"),
        (200, {"access_token": "a2", "refresh_token": "r2"}),
    ]

    def once(base_url, method, path, payload, token, timeout):
        assert path == "/api/auth/refresh" and payload["refresh_token"] == "r1"
        return answers.pop(0)

    monkeypatch.setattr(s, "_request_once", once)
    creds = s.Credentials("a1", "r1")
    assert creds.renew("http://x", "a1") is False and creds.refresh == "r1"
    assert creds.renew("http://x", "a1") is False and creds.refresh == "r1"
    assert creds.renew("http://x", "a1") is True and (creds.access, creds.refresh) == ("a2", "r2")
    monkeypatch.setattr(s, "_request_once", lambda *a: (401, "Invalid or expired refresh token"))
    assert creds.renew("http://x", "a2") is False and creds.refresh is None


def test_the_workflow_attributes_its_3076_events_to_the_installed_policy(tmp_path):
    """Another audit-mode policy on the runner logs 3076s as well."""
    body = _text(WORKFLOW)
    marker = "          $fromPolicy = {\n"
    assert body.count(marker) == 2
    _has(
        body,
        "$fired = @($named | Where-Object { & $fromPolicy $_ })",
        '$tail -and $subject -like "*$tail*" -and (& $fromPolicy $_)',
    )
    start = body.index("          $policyBare = ")
    end = body.index("\n          }\n", body.index(marker)) + len("\n          }\n")
    block = "\n".join(line[10:] for line in body[start:end].splitlines())
    script = tmp_path / "attr.ps1"
    script.write_text(
        "$env:SAC_POLICY_ID = '{5283AC0F-FFF1-49AE-ADA1-8A933130CAD6}'\n"
        + block
        + r"""
function Ev([string]$xml) { $o = [pscustomobject]@{ X = $xml }; $o | Add-Member ScriptMethod ToXml { $this.X }; $o }
$ns = 'http://schemas.microsoft.com/win/2004/08/events/event'
$ours = Ev "<Event xmlns='$ns'><EventData><Data Name='File Name'>C:\x\unsigned-control.exe</Data><Data Name='PolicyGUID'>{5283ac0f-fff1-49ae-ada1-8a933130cad6}</Data></EventData></Event>"
$byName = Ev "<Event xmlns='$ns'><EventData><Data Name='PolicyNameBuffer'>VerifiedAndReputableDesktopEvaluationAuditNoISG</Data></EventData></Event>"
$other = Ev "<Event xmlns='$ns'><EventData><Data Name='File Name'>C:\x\unsigned-control.exe</Data><Data Name='PolicyGUID'>{11111111-2222-3333-4444-555555555555}</Data><Data Name='PolicyNameBuffer'>ContosoAllowList</Data></EventData></Event>"
$broken = Ev "not xml"
if ($true -ne (& $fromPolicy $ours)) { exit 81 }
if ($true -ne (& $fromPolicy $byName)) { exit 82 }
if ($true -eq (& $fromPolicy $other)) { exit 83 }
if ($true -eq (& $fromPolicy $broken)) { exit 84 }
exit 0
""",
        encoding = "utf-8",
    )
    _assert_ok(_run_ps(script))


_PROBE_MOCKS = r"""
$WorkDir = $Work; $SkipUpdates = $true; $UpgradePackages = $false; $SendSamples = $false
$NOISG_GUID = '{AAAA}'; $NOISG_DEST = Join-Path $Work 'efi/active/test.cip'
$AuditPolicy = Join-Path $Work 'source.bin'; 'probe policy' | Set-Content -LiteralPath $AuditPolicy
$script:EfiStillMounted = $false
function Assert-Elevated { }
function Write-Section { }
function Clear-EfiOwnership { }
function Mount-Efi { $true }
function Dismount-Efi { }
function Invoke-Native { }
function Set-MpPreference { }
function Get-MpPreference { [pscustomobject]@{ DisableRealtimeMonitoring = $false } }
function Get-MpPreferenceType { [string] }
function Test-MpPreferenceMatch { $true }
function Get-CiLogSettings { [pscustomobject]@{ Enabled = $true; MaxSize = 67108864 } }
function Initialize-Studio { }
function Stop-ProbeStudio { $true }
function Get-StudioHomeOverride { $Work }
function Get-StudioHome { $Work }
function Get-LlamaDir { Join-Path $Work 'runtime' }
function Save-Baseline([string] $dir) {
  $b = [pscustomobject]@{ CapturedAt = (Get-Date).ToString('o'); Sac = [pscustomobject]@{ Mode = 'off'; RegistryState = 0; Policies = @() }
    AuditPolicyApplied = $false; AuditPolicyPreexisting = $false; AuditPolicyControlFired = $null; RevertCompletedAt = $null }
  Save-ProbeBaseline $b (Join-Path $dir 'baseline.json'); $b }
function Read-Baseline([string] $label) { Get-Content -LiteralPath (Join-Path (Join-Path $Work $label) 'baseline.json') -Raw | ConvertFrom-Json }
"""


def _drive_stages(tmp_path, body):
    names = [
        "Get-RunDir",
        "Get-RollbackPolicyPath",
        "Save-ProbeBaseline",
        "Test-PolicyActive",
        "Get-UnrevertedLabel",
        "Invoke-Prepare",
        "Invoke-Revert",
    ]
    _drive(tmp_path, names, _PROBE_MOCKS + body, Work = tmp_path)


def test_prepare_refuses_while_another_label_is_unreverted(tmp_path):
    """prepare B while A was unreverted saved A's policy as B's pre-existing one, so reverting A then B put the probe policy back."""
    _drive_stages(
        tmp_path,
        r"""
function Get-RollbackPolicyPath { throw 'PAST-GUARD' }
New-Item -ItemType Directory -Force -Path (Join-Path $Work 'A') | Out-Null
Save-ProbeBaseline ([pscustomobject]@{ AuditPolicyApplied = $true; RevertCompletedAt = $null }) (Join-Path $Work 'A/baseline.json')
$Label = 'B'
try { Invoke-Prepare; exit 71 } catch { if ("$_" -notlike "*label 'A' has not been reverted*revert -Label A*") { Write-Host "$_"; exit 72 } }
# A retry of the same label, and a label whose revert completed, are not refused.
$Label = 'A'
try { Invoke-Prepare; exit 73 } catch { if ("$_" -ne 'PAST-GUARD') { Write-Host "$_"; exit 74 } }
Save-ProbeBaseline ([pscustomobject]@{ AuditPolicyApplied = $false; RevertCompletedAt = 'done' }) (Join-Path $Work 'A/baseline.json')
$Label = 'B'
try { Invoke-Prepare; exit 75 } catch { if ("$_" -ne 'PAST-GUARD') { Write-Host "$_"; exit 76 } }
exit 0
""",
    )


def test_revert_keeps_the_baseline_pending_while_the_policy_is_still_active(tmp_path):
    """Before Windows 11 24H2 a removed policy stays active until a restart, and revert spent the baseline anyway."""
    _drive_stages(
        tmp_path,
        r"""
$global:listed = $true
function Get-SacState { [pscustomobject]@{ Policies = @(if ($global:listed) { [pscustomobject]@{ PolicyID = '{aaaa}'; FriendlyName = 'AuditNoISG' } }) } }
$Label = 'r'
New-Item -ItemType Directory -Force -Path (Split-Path $NOISG_DEST) | Out-Null
'probe policy' | Set-Content -LiteralPath $NOISG_DEST
New-Item -ItemType Directory -Force -Path (Join-Path $Work 'r') | Out-Null
Save-ProbeBaseline ([pscustomobject]@{ AuditPolicyApplied = $true; AuditPolicyPreexisting = $false; RevertCompletedAt = $null }) (Join-Path $Work 'r/baseline.json')
try { Invoke-Revert; exit 51 } catch { if ("$_" -notlike '*still active*restart*revert -Label r again*') { Write-Host "$_"; exit 52 } }
$b = Read-Baseline 'r'
if ($b.RevertCompletedAt -or $true -ne $b.AuditPolicyApplied) { exit 53 }
# After the restart the policy is gone and the same baseline completes.
$global:listed = $false
Invoke-Revert
$b = Read-Baseline 'r'
if (-not $b.RevertCompletedAt -or $false -ne $b.AuditPolicyApplied) { exit 54 }
exit 0
""",
    )


def test_a_prepare_that_fails_after_applying_the_policy_rolls_it_back(tmp_path):
    """A failed positive control threw with the probe policy still installed."""
    _drive_stages(
        tmp_path,
        r"""
function Get-SacState { [pscustomobject]@{ Policies = @([pscustomobject]@{ PolicyID = 'other' }; if (Test-Path -LiteralPath $NOISG_DEST) { [pscustomobject]@{ PolicyID = '{aaaa}'; FriendlyName = 'AuditNoISG' } }) } }
function Test-AuditPolicyEvaluating { $false }
$Label = 'fresh'
try { Invoke-Prepare; exit 41 } catch { if ("$_" -notlike '*raised no 3076 or 3077*') { Write-Host "$_"; exit 42 } }
if (Test-Path -LiteralPath $NOISG_DEST) { exit 43 }
if ($false -ne (Read-Baseline 'fresh').AuditPolicyApplied) { exit 44 }
Save-ProbeBaseline ([pscustomobject]@{ RevertCompletedAt = 'done' }) (Join-Path $Work 'fresh/baseline.json')
# An administrator's policy under the same GUID is put back, not deleted.
New-Item -ItemType Directory -Force -Path (Split-Path $NOISG_DEST) | Out-Null
'admin policy' | Set-Content -LiteralPath $NOISG_DEST
$Label = 'admin'
try { Invoke-Prepare; exit 45 } catch { }
if ((Get-Content -LiteralPath $NOISG_DEST -Raw).Trim() -ne 'admin policy') { exit 46 }
Save-ProbeBaseline ([pscustomobject]@{ RevertCompletedAt = 'done' }) (Join-Path $Work 'admin/baseline.json')
Remove-Item -LiteralPath $NOISG_DEST
# A prepare that succeeds keeps the policy for the stages after it.
function Test-AuditPolicyEvaluating { $true }
$Label = 'ok'
Invoke-Prepare
if (-not (Test-Path -LiteralPath $NOISG_DEST) -or $true -ne (Read-Baseline 'ok').AuditPolicyApplied) { exit 47 }
exit 0
""",
    )


@pytest.mark.parametrize(
    "refresh_exit, list_exit, listing",
    [(5, 0, _UNLISTED), (0, 7, _UNLISTED), (0, 0, "INVALID JSON")],
    ids = ["refresh", "listing", "malformed"],
)
def test_the_policy_removal_step_never_reports_an_unverified_removal(
    tmp_path, refresh_exit, list_exit, listing
):
    """A failed refresh or policy listing, or a listing that is not JSON, read as an empty list."""
    proc = _removal_step(tmp_path, "u", 0, refresh_exit, list_exit, listing)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    _has(proc.stdout, "::warning::", "could not be verified")
    assert "removed and no longer active" not in proc.stdout


def test_the_audit_loads_every_shipped_entry_point():
    body = _step("id", "exercise")
    assert "& $server.FullName --version" in body
    assert "& $quantize.FullName --help" in body, "llama-quantize loads its own impl DLL"


def test_a_quantize_that_failed_to_load_gives_no_verdict():
    _has(_text(WORKFLOW), 'Add-Content -Path $env:GITHUB_ENV -Value "QUANTIZE_EXIT=$LASTEXITCODE"')
    _has(_verdict(), "[int]$env:QUANTIZE_EXIT -lt 0", "llama-quantize failed to load")


def test_the_event_poll_waits_for_the_count_to_settle():
    verdict = _verdict()
    _has(verdict, "$ours.Count -gt 0 -and $ours.Count -eq $seen) { break }", "$seen = $ours.Count")
    _lacks(verdict, "if ($ours.Count -gt 0) { break }")
