# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Guards for scripts/windows_sac_probe, the Smart App Control evidence probe.

The PowerShell half cannot run here; its guards read the source. The Python
scenario is driven against a fake Studio.
"""

from __future__ import annotations

import importlib.util
import json
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
        return 200, {}

    monkeypatch.setattr(s, "_request", fake)
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
    unload = next(c for c in calls if c[1] == "/api/inference/unload")
    assert unload[2] == {"model_path": "unsloth/Qwen3.5-2B-MTP-GGUF"}
    chat = next(c for c in calls if c[1] == "/v1/chat/completions")
    assert chat[2]["model"] == "unsloth/Qwen3.5-2B-MTP-GGUF"
    results = json.loads((tmp_path / "scenario-results.json").read_text(encoding = "utf-8"))
    assert results["steps"]["unload"]["ok"] is True
    assert "in_flight_ms" in results["status_poll"]
    assert results["gguf_variant"] == "UD-Q4_K_XL"


def test_a_status_poll_still_in_flight_at_the_end_is_counted(monkeypatch):
    """The 75 to 80 second stall this exists to catch is exactly the poll that
    a 30 second join abandoned, so its duration was absent from the summary."""
    s = _load_scenario()
    source = (PROBE_DIR / "studio_scenario.py").read_text(encoding = "utf-8")
    assert "poller.join(timeout = 330)" in source
    release = threading.Event()

    def slow(*a, **k):
        release.wait(5)
        return 200, {}

    monkeypatch.setattr(s, "_request", slow)
    poller = s.StatusPoller("http://127.0.0.1:1", "t", interval = 0.01)
    poller.start()
    deadline = threading.Event()
    deadline.wait(0.2)
    assert poller.in_flight_ms() is not None and poller.in_flight_ms() > 100
    poller.stop()
    release.set()
    poller.join(timeout = 5)


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
