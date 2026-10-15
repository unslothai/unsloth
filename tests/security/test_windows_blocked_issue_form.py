# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Windows-blocked issue form must keep its product, detection and AMSI fields required."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
import yaml

# The shared runner, not a direct subprocess: xdist workers share one pwsh startup cache,
# so a startup crash would show up as this test failing.
# tests/studio/test_pwsh_calls_use_the_shared_runner.py enforces this.
from unsloth_pwsh_runner import run_pwsh


REPO = Path(__file__).resolve().parents[2]
FORM = REPO / ".github" / "ISSUE_TEMPLATE" / "install-blocked-windows.yml"

# Fields a false-positive report cannot be made without, plus the blocking-mechanism error id.
REQUIRED_FIELD_IDS = ("av-product", "detection-name", "error-text", "probe")


def _form() -> dict:
    return yaml.safe_load(FORM.read_text(encoding = "utf-8"))


def _fields() -> dict:
    return {item["id"]: item for item in _form()["body"] if isinstance(item, dict) and "id" in item}


def _snippet() -> str:
    """The PowerShell the form asks the reporter to run."""
    for item in _form()["body"]:
        description = (item.get("attributes") or {}).get("description") or ""
        match = re.search(r"```powershell\n(.*?)```", description, re.S)
        if match:
            return match.group(1)
    raise AssertionError(
        "the form contains no ```powershell block with newlines in it. The overwhelmingly likely "
        "cause is that its `description:` uses a YAML folded scalar (`>`) instead of a literal one "
        "(`|`): folding joins every line, so the fence and the code end up on one line and the "
        "script the reporter pastes is unusable. That exact mistake shipped in the first draft."
    )


def test_the_form_exists_and_is_valid_yaml() -> None:
    assert FORM.is_file(), f"missing {FORM.relative_to(REPO)}"
    form = _form()
    assert form.get("name"), "an issue form needs a name or GitHub will not offer it"
    assert isinstance(form.get("body"), list) and form["body"], "the form has no body"


@pytest.mark.parametrize("field_id", REQUIRED_FIELD_IDS)
def test_the_fields_a_vendor_needs_stay_required(field_id: str) -> None:
    fields = _fields()
    assert field_id in fields, (
        f"the form no longer has a {field_id!r} field. Every report of this class before the form "
        f"existed named no product and no detection name, and none of them could be submitted."
    )
    validations = fields[field_id].get("validations") or {}
    assert validations.get("required") is True, (
        f"{field_id!r} is no longer required. A markdown template could not enforce this, which is "
        f"why this is a form; making the field optional gives that up for nothing."
    )


def test_the_collection_script_survived_the_yaml() -> None:
    """A folded scalar joins lines and would ship a one-line, unusable script."""
    snippet = _snippet()
    assert snippet.count("\n") > 20, (
        "the collection script has collapsed to almost nothing. Its `description` must use a "
        "literal block scalar (`|`), not a folded one (`>`): folding joins the lines and destroys "
        "every newline in the code block."
    )
    for marker in (
        "SecurityCenter2",
        "AMSI\\Providers",
        "InprocServer32",
        "Get-MpComputerStatus",
        "AMRunningMode",
    ):
        assert marker in snippet, f"the collection script no longer gathers {marker}"


def test_the_collection_script_parses() -> None:
    """Parsed, not eyeballed. It is pasted verbatim by people who are already stuck."""
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("pwsh is unavailable")

    snippet = _snippet()
    probe = (
        "$ErrorActionPreference = 'Stop'; $errors = $null; $tokens = $null; "
        "$null = [System.Management.Automation.Language.Parser]::ParseInput("
        "[System.IO.File]::ReadAllText($env:UNSLOTH_SNIPPET_PATH), [ref]$tokens, [ref]$errors); "
        "if ($errors.Count) { $errors | ForEach-Object { $_.Message }; exit 1 }; "
        'Write-Output "OK $($tokens.Count)"'
    )
    # Passed via a file and env var so the snippet's own quoting cannot change parsing.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "snippet.ps1"
        path.write_text(snippet, encoding = "utf-8")
        result = run_pwsh(
            [pwsh, "-NoProfile", "-NonInteractive", "-Command", probe],
            capture_output = True,
            text = True,
            timeout = 120,
            env = {**__import__("os").environ, "UNSLOTH_SNIPPET_PATH": str(path)},
        )

    assert (
        result.returncode == 0
    ), f"the collection script in {FORM.name} does not parse:\n{result.stdout}\n{result.stderr}"


def test_the_collection_script_only_reads() -> None:
    """The collection script must only read: no downloads, no execution of fetched code, no writes."""
    snippet = _snippet()
    for banned, why in (
        ("Invoke-WebRequest", "a diagnostic must not download"),
        ("Invoke-RestMethod", "a diagnostic must not download"),
        ("Invoke-Expression", "a diagnostic must not evaluate strings"),
        ("iex", "a diagnostic must not evaluate strings"),
        ("Add-Type", "that compiles C# through csc.exe, which is the shape being removed"),
        ("FromBase64String", "encode-then-run is the shape being removed"),
        ("Set-MpPreference", "a diagnostic must never change the scanner it is reporting on"),
        ("Add-MpPreference", "a diagnostic must never change the scanner it is reporting on"),
        ("Remove-Item", "a diagnostic must not delete"),
        ("Set-Content", "a diagnostic must not write"),
        ("Start-Process", "a diagnostic must not launch anything"),
    ):
        assert banned not in snippet, f"the collection script uses {banned}: {why}"


def test_the_labels_the_form_declares_are_real() -> None:
    """GitHub drops unknown labels silently, so each label the form declares must be in the known list."""
    KNOWN_REPOSITORY_LABELS = {"bug", "windows", "antivirus-false-positive"}
    declared = set(_form().get("labels") or [])
    assert declared, "the form declares no labels, so nothing routes it to triage"
    unknown = declared - KNOWN_REPOSITORY_LABELS
    assert not unknown, (
        f"the form declares {sorted(unknown)}, which are not in the known-label list. Create them "
        f"on the repository with `gh label create` and add them here; GitHub will otherwise drop "
        f"them without a word and the issue will arrive unlabelled."
    )


def test_every_line_that_can_carry_a_path_is_redacted() -> None:
    """Probe output in a public issue must be redacted at print time, not left to the reporter."""
    snippet = _snippet()
    assert "function Hide-Personal" in snippet, (
        "the collection script no longer defines the redaction helper, so the profile directory and "
        "account name reach a public issue verbatim"
    )
    for carrier, why in (
        ("pathToSignedProductExe", "the antivirus product's own path"),
        ("$clsid   $dll", "the AMSI provider DLL path"),
        ("$($_.TimeCreated)", "the Defender event path and process name"),
    ):
        line = next((ln for ln in snippet.splitlines() if carrier in ln), None)
        assert line is not None, f"the collection script no longer prints {why} ({carrier})"
        assert (
            "Hide-Personal" in line
        ), f"the line printing {why} is no longer redacted: {line.strip()}"


def test_the_form_does_not_promise_more_privacy_than_it_delivers() -> None:
    """The form must not claim the script touches no personal data, since it prints detection paths."""
    description = _fields()["probe"]["attributes"]["description"]
    assert "touches no personal data" not in description, (
        "the form claims again that the collection script touches no personal data, but it prints "
        "detection paths from the last two hours"
    )
    assert (
        "Read the output before you paste it" in description
    ), "the form no longer tells the reporter to read the output before publishing it"


def test_the_defender_event_fields_survive_a_real_message(tmp_path: Path) -> None:
    """Defender indents its detail lines, so extraction must match indented Field: value lines."""
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("pwsh is unavailable")

    snippet = _snippet()
    match = re.search(r"\$_ -match '([^']+)'", snippet)
    assert match, "the collection script no longer filters the Defender event message lines"
    pattern = match.group(1)

    script = tmp_path / "fields.ps1"
    script.write_text(
        "\n".join(
            [
                '$msg = @"',
                "Windows Defender Antivirus has detected malware or other potentially unwanted software.",
                " Name: HackTool:Win64/Mimikatz.A",
                " ID: 2147747903",
                " Severity: High",
                " Path: C:\\Users\\jsmith\\AppData\\Local\\Temp\\m64.exe",
                " Detection Source: Real-Time Protection",
                " Process Name: C:\\Windows\\System32\\cmd.exe",
                " Action: Allowed",
                '"@',
                f"$sel = (($msg -split \"`r?`n\") | Where-Object {{ $_ -match '{pattern}' }}) -join ' | '",
                'Write-Output "SELECTED:$sel"',
            ]
        ),
        encoding = "utf-8",
    )
    done = run_pwsh(
        [pwsh, "-NoProfile", "-NonInteractive", "-File", str(script)],
        capture_output = True,
        text = True,
        timeout = 120,
    )
    selected = done.stdout
    for field in ("Name:", "Path:", "Process Name:", "Detection Source:", "Action:"):
        assert field in selected, (
            f"the filter dropped {field!r} from a Defender message in the documented indented shape, "
            f"so the report would carry a timestamp and an event id and nothing else. Selected: "
            f"{selected.strip()!r}"
        )


def test_a_localised_defender_message_still_reports_its_details(tmp_path: Path) -> None:
    """Localised Defender labels miss, so a count-based fallback prints the whole redacted message."""
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("pwsh is unavailable")

    snippet = _snippet()
    assert "localised" in snippet, "the collection script no longer has a localisation fallback"

    german = [
        "Windows Defender Antivirus hat Malware gefunden.",
        " Name: HackTool:Win64/Mimikatz.A",
        " Schweregrad: Hoch",
        " Pfad: C:\\Users\\jsmith\\AppData\\Local\\Temp\\m64.exe",
    ]
    match = re.search(r"\$_ -match '([^']+)'", snippet)
    assert match
    script = tmp_path / "loc.ps1"
    script.write_text(
        "\n".join(
            ["$lines = @(" + ", ".join(f"'{line}'" for line in german) + ")"]
            + [
                f"$matched = @($lines | Where-Object {{ $_ -match '{match.group(1)}' }})",
                "if ($matched.Count -ge 2) { $fields = $matched -join ' | ' }",
                "else { $fields = '(localised) ' + (($lines | Where-Object { $_.Trim() }) -join ' | ') }",
                'Write-Output "OUT:$fields"',
            ]
        ),
        encoding = "utf-8",
    )
    done = run_pwsh(
        [pwsh, "-NoProfile", "-NonInteractive", "-File", str(script)],
        capture_output = True,
        text = True,
        timeout = 120,
    )
    assert (
        "Schweregrad" in done.stdout and "Pfad" in done.stdout
    ), f"a German Defender message lost its severity and path: {done.stdout.strip()!r}"


def test_the_screenshot_field_is_not_a_rendered_textarea() -> None:
    """A render: textarea wraps attachments in a code fence, so screenshots need an unrendered field."""
    fields = _fields()
    assert "screenshot" in fields, (
        "there is no screenshot field, so a reporter whose failure was a dialog has nowhere to put "
        "it except a rendered textarea, where it will not display"
    )
    assert "render" not in fields["screenshot"]["attributes"], (
        "the screenshot field is rendered, so an attachment dropped into it becomes literal text "
        "inside a code block"
    )
    error_description = fields["error-text"]["attributes"]["description"]
    assert (
        "a screenshot is fine" not in error_description
    ), "the rendered error field still invites a screenshot it cannot display"


def test_the_required_error_field_warns_about_its_own_paths() -> None:
    """The error-text field can carry user-profile paths, so it must warn that the issue is public."""
    description = _fields()["error-text"]["attributes"]["description"]
    assert "public" in description.lower(), (
        "the error field does not mention that the issue is public, though it is required and "
        "routinely contains C:\\Users\\<name>"
    )


def test_every_field_that_asks_for_a_path_says_the_issue_is_public() -> None:
    """Every field that asks for a path, including file-path, must say the issue is public."""
    fields = _fields()
    for name in ("error-text", "probe", "file-path"):
        attributes = fields[name]["attributes"]
        text = f"{attributes.get('label', '')}\n{attributes.get('description', '')}"
        assert (
            "public" in text.lower()
        ), f"the {name} field asks for a filesystem path without saying the issue is public"
    # The probe is exempt: Hide-Personal already redacts it before anything is written.
    for name in ("error-text", "file-path"):
        description = fields[name]["attributes"].get("description", "")
        assert (
            "<me>" in description
        ), f"the {name} field warns that the issue is public but never shows what to write instead"


def _hide_personal_source() -> str:
    """Just the Hide-Personal function, lifted out of the shipped snippet."""
    snippet = _snippet()
    start = snippet.index("function Hide-Personal")
    depth = 0
    for i in range(start, len(snippet)):
        if snippet[i] == "{":
            depth += 1
        elif snippet[i] == "}":
            depth -= 1
            if depth == 0:
                return snippet[start : i + 1]
    raise AssertionError("Hide-Personal is not brace-balanced in the shipped snippet")


@pytest.mark.parametrize(
    ("username", "line", "expected"),
    [
        # The two corruptions a bare substring replace causes; both rewrite the evidence.
        (
            "win",
            "Windows Defender Antivirus 4.18.24090.11",
            "Windows Defender Antivirus 4.18.24090.11",
        ),
        ("cat", "Trojan:Script/Wacatac.B!ml", "Trojan:Script/Wacatac.B!ml"),
        ("alice", r"C:\Users\alice\Downloads\x.ps1", r"C:\Users\<user>\Downloads\x.ps1"),
        ("alice", r"CORP\alice", r"CORP\<user>"),
        ("al", r"C:\Users\alice\x.ps1", r"C:\Users\alice\x.ps1"),
    ],
)
def test_redaction_only_fires_on_a_real_account_component(
    tmp_path: Path, username: str, line: str, expected: str
) -> None:
    """Account-name redaction must match whole path components, so a user named win spares Windows."""
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("pwsh is unavailable")

    script = tmp_path / "redact.ps1"
    script.write_text(
        _hide_personal_source()
        + "\n"
        + "$env:USERPROFILE = 'C:\\Users\\__no_such_profile__'\n"
        + f"$env:USERNAME = '{username}'\n"
        + "Write-Output (Hide-Personal $env:UNSLOTH_LINE)\n",
        encoding = "utf-8",
    )
    import os as _os

    done = run_pwsh(
        [pwsh, "-NoProfile", "-NonInteractive", "-File", str(script)],
        capture_output = True,
        text = True,
        timeout = 120,
        env = {**_os.environ, "UNSLOTH_LINE": line},
    )
    assert done.returncode == 0, f"{done.stdout}\n{done.stderr}"
    assert (
        done.stdout.strip() == expected
    ), f"account {username!r} turned {line!r} into {done.stdout.strip()!r}"


def test_the_probe_does_not_change_the_callers_error_preference() -> None:
    """Probe body runs in a script block so $ErrorActionPreference cannot leak into the user's session."""
    snippet = _snippet().strip()
    assert snippet.startswith("& {"), (
        "the probe no longer runs inside a script block, so the preference it sets leaks into the "
        "session the reporter pasted it into"
    )
    assert snippet.endswith("}"), "the script block is not closed"
    assert snippet.index("$ErrorActionPreference") > snippet.index(
        "& {"
    ), "the preference is set outside the block that was supposed to contain it"
