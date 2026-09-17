# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Windows-blocked issue form has to keep asking for the three fields a vendor needs.

#8523, #6326, #6588, #6648, #10540 and #10805 all reported a security product blocking Unsloth and
named no product, no detection name and no AMSI provider, so not one of them could be submitted to a
vendor or proven fixed. The form exists to make those fields required; if a later edit relaxes them,
or breaks the collection script that produces them, we are back to unactionable reports and nothing
would otherwise fail.

The collection script is checked by parsing it. A diagnostic that does not run is worse than none: it
costs the user a round trip and produces an error message about our form instead of about their
antivirus. That is not hypothetical here -- the first version of this form put the script in a YAML
folded scalar (`>`), which joins the lines and silently destroyed every newline in it.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
import yaml

# The shared runner, not a direct subprocess call: a direct one shares a single
# $XDG_CACHE_HOME/powershell startup cache with every other xdist worker, and an interpreter that
# dies at startup then renders as this test failing rather than as the crash it was.
# tests/studio/test_pwsh_calls_use_the_shared_runner.py enforces this.
from unsloth_pwsh_runner import run_pwsh


REPO = Path(__file__).resolve().parents[2]
FORM = REPO / ".github" / "ISSUE_TEMPLATE" / "install-blocked-windows.yml"

# The three things a false-positive submission cannot be made without, plus the error id that says
# which mechanism blocked it.
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
    # Each section the triage actually reads. Losing one silently narrows what a report can answer.
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
    # Through a file and an environment variable, so nothing about the snippet's own quoting can
    # change how it is handed to the parser.
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
    """It runs on a stranger's machine while they are already having a bad day.

    No downloads, no execution of anything fetched, and nothing that writes. It also must not carry
    the shapes the installers are being cleaned of, or the diagnostic gets blocked by the same
    product that blocked the installer.
    """
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
    """GitHub drops an unknown label silently rather than erroring.

    All three of these were absent when the form was first written, so every issue filed through it
    would have arrived with `bug` only -- losing exactly the two triage labels the form exists to
    attach, with nothing anywhere reporting the loss.

    Checked against a checked-in list rather than the GitHub API, because a test that needs the
    network is a test that gets skipped. The list is the contract: if someone adds a label here they
    must create it on the repository too.
    """
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
    """The form is `required: true` and the issue is public, so this output gets published.

    Three of the sections print file paths, and the Defender one prints `Path` and `Process Name`
    for every detection in the last two hours, related to Unsloth or not. On a normal machine those
    paths start with the profile directory and therefore carry the account name, and the file name
    itself can be anything the person happened to have open. Redaction is applied at the point of
    printing rather than trusted to the reporter, because the reporter is someone whose install is
    already broken and who is pasting a block they did not write.
    """
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
    """The old wording said it touches no personal data, which was not true of the event section.

    An assurance that overstates is worse than none: it is read by exactly the people least placed
    to check it, and it discourages the one thing that does work, which is reading the output first.
    """
    description = _fields()["probe"]["attributes"]["description"]
    assert "touches no personal data" not in description, (
        "the form claims again that the collection script touches no personal data, but it prints "
        "detection paths from the last two hours"
    )
    assert (
        "Read the output before you paste it" in description
    ), "the form no longer tells the reporter to read the output before publishing it"


def test_the_defender_event_fields_survive_a_real_message(tmp_path: Path) -> None:
    """Defender indents its detail lines, so an anchor on the field name matches none of them.

    The rendered message for 1116 and its siblings is a header line followed by indented
    `Field: value` lines (Microsoft's own 1117 reference lists the field set, and real 1116 output
    shows the same). Anchoring on `^Name:` therefore matched nothing at all, and the section that
    exists to carry the detection name, its path and the process that triggered it emitted only a
    timestamp and an event id. That is a silent loss of the single most useful field in the report.

    Driven through the snippet's own filter against a message in the documented shape, rather than
    asserting the regex text, so a future rewrite of the extraction is judged on what it extracts.
    """
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
    """Defender localises the field labels, and a partial match is worse than none.

    On a non-English Windows the English labels miss, so the entry would be a timestamp and an
    event id. Some labels coincide across languages -- German renders `Name:` identically -- so
    testing for an empty result is not enough: one match out of six looks like a successful
    extraction. The fallback is therefore count-based, and prints the whole message, redacted,
    when too few fields resolve.
    """
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
    """A `render:` textarea wraps everything in a code fence, attachments included.

    So an image dragged into one appears as literal markup rather than as a picture. The error
    field keeps `render: text`, because it carries pasted console output where the monospace block
    is worth having; screenshots get their own unrendered field instead.
    """
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
    """The probe output is not the only required field that publishes a path.

    A PowerShell error quotes the script's location, so on the documented cloned-checkout entry
    point the required error text carries the reporter's user name, and they are asked to paste it
    verbatim before they ever reach the probe field's privacy note.
    """
    description = _fields()["error-text"]["attributes"]["description"]
    assert "public" in description.lower(), (
        "the error field does not mention that the issue is public, though it is required and "
        "routinely contains C:\\Users\\<name>"
    )


def test_every_field_that_asks_for_a_path_says_the_issue_is_public() -> None:
    """The privacy wording covered two fields and missed the one that asks for a path outright.

    `error-text` and `probe` both warn that the issue is public and invite redaction, but
    `file-path` asked for a full path with nothing but a `C:\\Users\\...` shaped placeholder to go
    on. A reporter who follows the placeholder publishes their user name, and the protections added
    everywhere else make that omission read as deliberate rather than missed.
    """
    fields = _fields()
    for name in ("error-text", "probe", "file-path"):
        attributes = fields[name]["attributes"]
        text = f"{attributes.get('label', '')}\n{attributes.get('description', '')}"
        assert (
            "public" in text.lower()
        ), f"the {name} field asks for a filesystem path without saying the issue is public"
    # The probe is exempt from this second one: `Hide-Personal` substitutes the profile and user
    # name before anything is written, so there is nothing for the reporter to rewrite by hand.
    # These two are pasted unaided, and a warning with no worked replacement tends to be answered by
    # dropping the field rather than by editing it.
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
        # The two corruptions a bare substring replace causes. Both rewrite the exact evidence the
        # form exists to collect, and both look like a clean report to the person pasting it.
        (
            "win",
            "Windows Defender Antivirus 4.18.24090.11",
            "Windows Defender Antivirus 4.18.24090.11",
        ),
        ("cat", "Trojan:Script/Wacatac.B!ml", "Trojan:Script/Wacatac.B!ml"),
        # Still redacted where it is genuinely the account.
        ("alice", r"C:\Users\alice\Downloads\x.ps1", r"C:\Users\<user>\Downloads\x.ps1"),
        ("alice", r"CORP\alice", r"CORP\<user>"),
        # A longer account name that merely starts with the same letters must not be half-eaten.
        ("al", r"C:\Users\alice\x.ps1", r"C:\Users\alice\x.ps1"),
    ],
)
def test_redaction_only_fires_on_a_real_account_component(
    tmp_path: Path, username: str, line: str, expected: str
) -> None:
    """Driven through the shipped function, not by reading its regex.

    The first version replaced every case-insensitive occurrence of the account name anywhere in
    the line, so an account called `win` rewrote `Windows Defender` and one called `cat` rewrote
    `Wacatac`. Those are the product and detection names the whole form is built to capture, and the
    reporter cannot tell it happened.
    """
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
    """The form promises the probe changes nothing, and then set a preference at the prompt.

    `$ErrorActionPreference = 'Continue'` pasted into an interactive session persists for the rest
    of that session, so a reporter who runs with `Stop` silently loses it. Running the body inside
    a script block scopes the assignment to the block.
    """
    snippet = _snippet().strip()
    assert snippet.startswith("& {"), (
        "the probe no longer runs inside a script block, so the preference it sets leaks into the "
        "session the reporter pasted it into"
    )
    assert snippet.endswith("}"), "the script block is not closed"
    assert snippet.index("$ErrorActionPreference") > snippet.index(
        "& {"
    ), "the preference is set outside the block that was supposed to contain it"
