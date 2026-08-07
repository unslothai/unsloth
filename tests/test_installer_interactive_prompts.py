# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Refuse new interactive yes/no prompts in the installer and setup scripts.

#7016 added `Open Unsloth Studio in your default browser after launch?
[Y/n]` to the interactive install and had to be reverted in #8040. Extra
questions stall a piped install and persist an answer nobody can find
again, so the only preference setup may ask about is whether to start
Studio when it finishes.

Allowlist, not a ban: every prompt in the tree is listed in
`APPROVED_PROMPTS` with its reason, so a new one fails with instructions
instead of landing quietly. Two passes catch it -- literal `[Y/n]`
markers, and interactive read sites (`read` off /dev/tty or `-p`,
`Read-Host`) for a marker built from a variable, as #7016's was. Text
only, so it runs on every platform in the parity matrix.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Everything a user can run to install, update or remove Unsloth. `unsloth
# studio update` shells out to studio/setup.sh, so it counts too.
SCANNED_SCRIPTS = (
    "install.sh",
    "install.ps1",
    "studio/setup.sh",
    "studio/setup.ps1",
    "scripts/uninstall.sh",
    "scripts/uninstall.ps1",
)

# Every question these scripts may ask, keyed by (script, normalised
# question) because line numbers move and wording does not. Do NOT add an
# entry just to turn a build green: decide the prompt is wanted first.
APPROVED_PROMPTS: dict[tuple[str, str], str] = {
    ("install.sh", "start unsloth studio now?"): (
        "The one sanctioned preference prompt: launch Studio after install."
    ),
    ("install.ps1", "start unsloth studio now?"): (
        "Windows half of the sanctioned launch prompt above."
    ),
    ("install.sh", "accept?"): (
        "Consent before `sudo apt-get install` of missing system packages, "
        "not a preference. Declining prints the command to run by hand."
    ),
    ("studio/setup.sh", "accept?"): (
        "Same sudo consent, for the packages llama.cpp needs to build. "
        "Declining skips the build rather than aborting."
    ),
}

_MARKER = re.compile(
    r"\[\s*[yn]\s*/\s*[yn]\s*\]|\(\s*[yn]\s*/\s*[yn]\s*\)|\byes\s*/\s*no\b", re.IGNORECASE
)

# Anything that blocks waiting on a human.
_POSIX_READ = re.compile(
    r"(?:^|[\s;&|(])read\s+(?![a-zA-Z_]+=)[^\n]*?(?:<\s*/dev/tty|(?:^|\s)-p(?:\s|\"|'))"
)
_PWSH_READ = re.compile(r"Read-Host|PromptForChoice|ReadKey\s*\(", re.IGNORECASE)

_QUOTED = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"' r"|'([^']*)'")

# Regex/glob syntax: a `sed` pattern next to a prompt also contains a `?`.
_REGEXY = re.compile(r"\\|\(\?|\^|\$\(|\[0|\{[0-9]|\.\*|\|")

# Stripped so a question built from a format string keys the same as a literal.
_FORMAT_NOISE = re.compile(
    r"%[-#0 +]*\d*(?:\.\d+)*[sdfxbq%]|\\[nrte]|\$\{?[A-Za-z_][A-Za-z0-9_]*\}?"
)


def _is_comment(line: str) -> bool:
    """Whole-line `#` comment. Inline ones are left alone: telling a comment from
    a `#` inside a string needs a parser, and a false positive only costs an
    unnecessary allowlist entry."""
    return line.lstrip().startswith("#")


def _quoted_strings(line: str) -> list[str]:
    return [group for match in _QUOTED.finditer(line) for group in match.groups() if group]


def normalise_question(text: str) -> str:
    """Reduce a prompt string to a stable allowlist key: drop the marker, format
    conversions, variable references and edge punctuation, then lowercase."""
    text = _MARKER.sub(" ", text)
    text = _FORMAT_NOISE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" -:>*_=").lower()


def _looks_like_question(text: str) -> bool:
    """Prose with a question mark, not a `sed`/`-match` pattern that has one."""
    return "?" in text and not _REGEXY.search(text)


def _nearby_question(lines: list[str], index: int, *, direction: int) -> str:
    """Nearest readable question around `lines[index]`, or "". Walk back (-1) from
    a read site to the `printf` that drew it, forward (+1) from a bare `[Y/n]`
    hint assigned into a variable and interpolated below (the #7016 shape)."""
    candidates = list(_quoted_strings(lines[index]))
    for step in range(1, 9):
        neighbour = index + direction * step
        if not 0 <= neighbour < len(lines):
            break
        line = lines[neighbour]
        if _is_comment(line) or not line.strip():
            continue
        candidates.extend(_quoted_strings(line))

    for text in candidates:
        if _looks_like_question(text):
            normalised = normalise_question(text)
            if normalised:
                return normalised
    return ""


def _question_for_read(lines: list[str], index: int) -> str:
    """Allowlist key for a read site, falling back to the read line itself so an
    unlabelled prompt is still allowlisted explicitly rather than ignored."""
    return _nearby_question(lines, index, direction = -1) or (
        f"<unlabelled read: {normalise_question(lines[index])}>"
    )


def find_prompts(script: str, source: str) -> list[tuple[str, int, str]]:
    """Return (script, line_number, normalised question) for every prompt site."""
    lines = source.splitlines()
    powershell = script.endswith(".ps1")
    found: dict[tuple[str, str], tuple[str, int, str]] = {}

    for index, line in enumerate(lines):
        if _is_comment(line):
            continue
        line_number = index + 1

        for text in _quoted_strings(line):
            if not _MARKER.search(text):
                continue
            # A bare `[Y/n]` is a hint variable; its question is printed below.
            question = (
                normalise_question(text)
                or _nearby_question(lines, index, direction = 1)
                or f"<yes/no marker with no question: {text.strip()}>"
            )
            found.setdefault((script, question), (script, line_number, question))

        reads = _PWSH_READ if powershell else _POSIX_READ
        if reads.search(line):
            question = _question_for_read(lines, index)
            found.setdefault((script, question), (script, line_number, question))

    return sorted(found.values(), key = lambda item: item[1])


def _failure_message(script: str, line_number: int, question: str) -> str:
    return (
        f"\n"
        f"A new interactive yes/no prompt was added to {script} (line {line_number}):\n"
        f"\n"
        f"    {question!r}\n"
        f"\n"
        f"Installers and setup scripts must not grow new questions. The only\n"
        f"preference the setup is allowed to ask about is 'Start Unsloth Studio\n"
        f"now?'; the remaining approved prompts are sudo consent before we\n"
        f"elevate. A prompt added here stalls `curl ... | sh` installs and\n"
        f"persists an answer the user cannot easily find again. This is what\n"
        f"#7016 did and why it was reverted in #8040.\n"
        f"\n"
        f"VERIFY THAT THIS PROMPT IS SUPPOSED TO BE HERE.\n"
        f"\n"
        f"  - If it is not: remove it. Take the setting as a flag or an\n"
        f"    environment variable with a non-interactive default instead.\n"
        f"  - If it genuinely is: add\n"
        f"\n"
        f'        ({script!r}, {question!r}): "why this prompt is needed",\n'
        f"\n"
        f"    to APPROVED_PROMPTS in tests/test_installer_interactive_prompts.py\n"
        f"    in the same PR, so the decision shows up in the diff.\n"
    )


@pytest.mark.parametrize("script", SCANNED_SCRIPTS)
def test_no_unapproved_interactive_prompts(script: str):
    path = REPO_ROOT / script
    assert path.is_file(), f"{script} is missing -- update SCANNED_SCRIPTS if it moved"

    for found_script, line_number, question in find_prompts(
        script, path.read_text(encoding = "utf-8")
    ):
        if (found_script, question) not in APPROVED_PROMPTS:
            pytest.fail(_failure_message(found_script, line_number, question), pytrace = False)


def test_approved_prompts_all_still_exist():
    """Delete a prompt, delete its entry: a stale one waves through a future prompt
    that happens to reuse the wording."""
    live = set()
    for script in SCANNED_SCRIPTS:
        source = (REPO_ROOT / script).read_text(encoding = "utf-8")
        live.update(
            (found_script, question) for found_script, _, question in find_prompts(script, source)
        )

    stale = sorted(key for key in APPROVED_PROMPTS if key not in live)
    assert not stale, (
        f"APPROVED_PROMPTS lists prompts that no longer exist: {stale}. "
        f"Remove the entries from tests/test_installer_interactive_prompts.py."
    )


def test_every_installer_script_is_scanned():
    """A prompt in a script nobody scans is the same regression, one step removed."""
    patterns = (
        "install*.sh",
        "install*.ps1",
        "studio/setup*.sh",
        "studio/setup*.ps1",
        "scripts/uninstall*.sh",
        "scripts/uninstall*.ps1",
    )
    on_disk = {
        path.relative_to(REPO_ROOT).as_posix()
        for pattern in patterns
        for path in REPO_ROOT.glob(pattern)
    }
    unscanned = sorted(on_disk - set(SCANNED_SCRIPTS))
    assert not unscanned, (
        f"installer/setup scripts not covered by the prompt guard: {unscanned}. "
        f"Add them to SCANNED_SCRIPTS in tests/test_installer_interactive_prompts.py "
        f"and to the paths filter in .github/workflows/cross-platform-parity-ci.yml."
    )


def test_approved_prompts_are_documented():
    for key, reason in APPROVED_PROMPTS.items():
        assert reason.strip(), f"APPROVED_PROMPTS[{key}] needs a reason, not an empty string"


# Detector self-tests: a scan that silently stops matching passes everything.


def test_detects_literal_marker_prompt():
    source = 'printf "  Enable telemetry? [Y/n] "\nread -r _reply </dev/tty || _reply="n"\n'
    assert find_prompts("install.sh", source) == [("install.sh", 1, "enable telemetry?")]


def test_detects_prompt_whose_marker_comes_from_a_variable():
    """The #7016 shape: only the read site gives it away."""
    source = (
        '_browser_hint="[Y/n]"\n'
        'printf "  Open Unsloth Studio in your default browser after launch? %s " "$_browser_hint"\n'
        'read -r _browser_reply </dev/tty || _browser_reply=""\n'
    )
    questions = [question for _, _, question in find_prompts("install.sh", source)]
    assert "open unsloth studio in your default browser after launch?" in questions


def test_detects_powershell_prompt_whose_marker_comes_from_a_variable():
    source = (
        "$_browserHint = if ($_existingPref -eq '0') { '[y/N]' } else { '[Y/n]' }\n"
        '$_browserReply = Read-Host "  Open Unsloth Studio in your default browser after launch? $_browserHint"\n'
    )
    assert [question for _, _, question in find_prompts("install.ps1", source)] == [
        "open unsloth studio in your default browser after launch?",
    ]


def test_detects_read_dash_p():
    source = 'read -p "Keep existing config? [y/N] " _reply\n'
    assert [question for _, _, question in find_prompts("install.sh", source)] == [
        "keep existing config?",
    ]


def test_detects_powershell_read_host():
    source = '$reply = Read-Host "  Install desktop shortcuts? [Y/n]"\n'
    assert [question for _, _, question in find_prompts("install.ps1", source)] == [
        "install desktop shortcuts?",
    ]


def test_detects_unlabelled_read():
    source = 'read -r _reply </dev/tty || _reply="n"\n'
    questions = [question for _, _, question in find_prompts("install.sh", source)]
    assert len(questions) == 1 and questions[0].startswith("<unlabelled read:")


def test_ignores_comments_and_non_interactive_reads():
    source = (
        "# Ask the user [Y/n] before doing anything.\n"
        'read -r _line < "$config_file"\n'
        'while IFS= read -r _entry; do :; done < "$manifest"\n'
        "read _major _minor <<EOF\n1 2\nEOF\n"
    )
    assert find_prompts("install.sh", source) == []


def test_reports_the_prompt_in_the_failure_message():
    message = _failure_message("install.sh", 42, "enable telemetry?")
    assert "enable telemetry?" in message
    assert "APPROVED_PROMPTS" in message
    assert "#8040" in message
