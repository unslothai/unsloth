# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Refuse new interactive yes/no prompts in the installer and setup scripts.

#7016 added `Open Unsloth Studio in your default browser after launch?
[Y/n]` and had to be reverted in #8040: extra questions stall a piped
install and persist an answer nobody can find again. The only preference
setup may ask about is whether to start Studio when it finishes.

Allowlist, not a ban: every prompt in the tree is listed in
`APPROVED_PROMPTS` with its reason, so a new one fails with instructions
instead of landing quietly. Two passes catch it: literal `[Y/n]` markers,
and interactive read sites (shell `read` and `select`, `Read-Host` and
the console reads, `input()`, `set /p`) for a marker built from a
variable, as #7016's was. Text only, so it runs on every platform in the
parity matrix.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# What a user runs to install, update or remove Unsloth. Everything these launch
# in turn is scanned too (`unsloth studio update` runs setup.sh, which builds
# whisper.cpp; install.sh fetches and runs the WSL bootstrap), since a question
# down there stalls the same install.
ENTRY_POINTS = (
    "install.sh",
    "install.ps1",
    "studio/setup.sh",
    "studio/setup.ps1",
    "studio/setup.bat",
)

SCANNED_SCRIPTS = ENTRY_POINTS + (
    "scripts/build_whisper_cpp.sh",
    "scripts/install_gemma4_mlx.sh",
    "scripts/install_qwen3_6_mlx.sh",
    "scripts/install_rocm_wsl_strixhalo.sh",
    "scripts/uninstall.sh",
    "scripts/uninstall.ps1",
    "studio/install_llama_prebuilt.py",
    "studio/install_node_prebuilt.py",
    "studio/install_python_stack.py",
    "studio/install_whisper_prebuilt.py",
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

# Anything that blocks waiting on a human. `-p` is matched as an option word, not
# anchored on whitespace: it may be bundled (`read -rp`) or follow `read` directly.
# The scan stops at `;|&`, since a `mkdir -p` later on the line is another command.
_POSIX_READ = re.compile(
    r"(?:^|[\s;&|(])read\s+(?![a-zA-Z_]+=)"
    r"(?:-[a-zA-Z]*p(?=[\s\"']|$)|[^\n;|&]*?(?:<\s*/dev/tty|\s-[a-zA-Z]*p(?=[\s\"']|$)))"
)

# An unredirected `read` takes the terminal it inherited, so it blocks too. Loop
# and pipeline reads are fed by the `done < ...` or the pipe: data, not questions.
# Options then variable names to end of line, in command position, so the word
# `read` in a heredoc of prose is not a prompt.
_BARE_READ = re.compile(
    r"(?:^|[;&(])\s*(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*"
    r"read(?:\s+(?:-[a-zA-Z0-9]+|\d+))*(?:\s+[A-Za-z_][A-Za-z0-9_]*)+"
    r"(?:\s*(?:\|\||&&).*)?\s*$"
)
_LOOP = re.compile(r"\b(?:while|until|for)\b")

# `select reply in Yes No` is the other builtin that blocks for an answer; its
# question is the PS3 assignment above it, which the nearby scan already reads.
_SELECT = re.compile(r"(?:^|[;&(])\s*select\s+[A-Za-z_][A-Za-z0-9_]*\s+in\s")

_PWSH_READ = re.compile(
    r"Read-Host|PromptForChoice|(?:Console\]::(?:In\.)?|UI\.)Read(?:Line|Key)?\b|ReadKey\s*\(",
    re.IGNORECASE,
)

# The Python helpers setup runs, and the batch launcher.
_PY_READ = re.compile(r"(?<![.\w])input\s*\(|getpass\.getpass\s*\(|confirm\s*\(")
_BAT_READ = re.compile(r"\bset\s+/p\b|\bchoice\b", re.IGNORECASE)

# A helper filename, with or without its `scripts/` prefix: sibling invocations
# such as "$SCRIPT_DIR/build_deps.sh" carry no prefix. Resolved against the repo
# before it counts, so a name that is not a real script is ignored.
_HELPER_REF = re.compile(r"(?<![$\w])((?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.(?:sh|ps1|py|bat))")

_QUOTED = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"' r"|'([^']*)'")

# `<# ... #>` opened and closed on one line.
_INLINE_BLOCK = re.compile(r"<#.*?#>", re.DOTALL)

# Regex/glob syntax: a `sed` pattern next to a prompt also contains a `?`.
_REGEXY = re.compile(r"\\|\(\?|\^|\$\(|\[0|\{[0-9]|\.\*|\|")

_ESCAPE = re.compile(r"\\[nrte]")

# A conversion or variable becomes a placeholder rather than vanishing, so wording
# spliced into the middle of an approved question cannot normalise back onto it.
_SUBSTITUTION = re.compile(r"%[-#0 +]*\d*(?:\.\d+)*[sdfxbq%]|\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")
_TRAILING_SUBSTITUTION = re.compile(r"(?:\s*<var>)+$")


def _is_comment(line: str) -> bool:
    """Whole-line `#` comment. Inline ones need a parser to tell from a `#` inside
    a string, and a false positive only costs an allowlist entry."""
    return line.lstrip().startswith("#")


def _is_interactive_read(line: str, script: str) -> bool:
    """A read that waits on a person: a prompt option, /dev/tty, `select`, `input()`,
    `set /p`, or plain inherited stdin. Redirected and loop reads consume a file, not
    a person. Quoted text is blanked first, so a message naming `Read-Host` is not
    one."""
    code = _QUOTED.sub('""', line)
    if script.endswith(".ps1"):
        return bool(_PWSH_READ.search(code))
    if script.endswith(".py"):
        return bool(_PY_READ.search(code))
    if script.endswith(".bat"):
        return bool(_BAT_READ.search(code))
    if _POSIX_READ.search(code) or _SELECT.search(code):
        return True
    # `||` is a fallback, not a pipeline: `read -r reply || reply=n` still blocks.
    piped = "|" in code.replace("||", "")
    return "<" not in code and not piped and bool(_BARE_READ.search(code))


def blank_comments(source: str, script: str) -> str:
    """Blank comments, keeping the line count so numbers still line up. `.ps1` files
    carry `<# ... #>` blocks, where a documented Read-Host example would otherwise
    fail CI over a prompt that cannot run. `.bat` comments are REM or `::`."""
    lines = []
    in_block = False
    powershell = script.endswith(".ps1")
    batch = script.endswith(".bat")
    for line in source.splitlines():
        if powershell and in_block:
            # Code can follow the terminator on the same line.
            lines.append(line.split("#>", 1)[1] if "#>" in line else "")
            in_block = "#>" not in line
            continue
        if batch and re.match(r"\s*(?:REM\b|::)", line, re.IGNORECASE):
            lines.append("")
            continue
        if powershell:
            line = _INLINE_BLOCK.sub("", line)
            if "<#" in line:
                lines.append(line.split("<#", 1)[0])
                in_block = True
                continue
        lines.append("" if _is_comment(line) else line)
    return "\n".join(lines)


def _quoted_strings(line: str) -> list[str]:
    return [group for match in _QUOTED.finditer(line) for group in match.groups() if group]


def normalise_question(text: str) -> str:
    """Reduce a prompt string to a stable allowlist key: drop the marker and edge
    punctuation, placeholder the substitutions, then lowercase."""
    text = _MARKER.sub(" ", text)
    text = _ESCAPE.sub(" ", text)
    text = _SUBSTITUTION.sub("<var>", text)
    text = re.sub(r"\s+", " ", text).strip()
    # A trailing one is the `[Y/n]` hint the marker pass already read.
    text = _TRAILING_SUBSTITUTION.sub("", text)
    return text.strip(" -:>*_=").lower()


def _looks_like_question(text: str) -> bool:
    """Prose with a question mark, not a `sed`/`-match` pattern that has one."""
    return "?" in text and not _REGEXY.search(text)


def _nearby_questions(lines: list[str], index: int, *, direction: int) -> list[str]:
    """Readable questions around `lines[index]`, nearest first. Walk back (-1) from a
    read site to the `printf` that drew it, forward (+1) from a bare `[Y/n]` hint
    assigned into a variable and interpolated below (the #7016 shape)."""
    candidates = list(_quoted_strings(lines[index]))
    for step in range(1, 9):
        neighbour = index + direction * step
        if not 0 <= neighbour < len(lines):
            break
        line = lines[neighbour]
        if not line.strip():
            continue
        candidates.extend(_quoted_strings(line))

    questions = []
    for text in candidates:
        if not _looks_like_question(text):
            continue
        normalised = normalise_question(text)
        if normalised and normalised not in questions:
            questions.append(normalised)
    return questions


def _questions_for_read(lines: list[str], index: int) -> list[str]:
    """Allowlist keys for a read site. Every question in reach, not just the nearest:
    one read can serve a branch each, and validating only the closest lets the other
    branch through. Falls back to the read line so an unlabelled prompt still has to
    be allowlisted rather than ignored."""
    return _nearby_questions(lines, index, direction = -1) or [
        f"<unlabelled read: {normalise_question(lines[index])}>"
    ]


def find_prompts(script: str, source: str) -> list[tuple[str, int, str]]:
    """Return (script, line_number, normalised question) for every prompt site."""
    lines = blank_comments(source, script).splitlines()
    found: dict[tuple[str, str], tuple[str, int, str]] = {}

    for index, line in enumerate(lines):
        line_number = index + 1

        for text in _quoted_strings(line):
            if not _MARKER.search(text):
                continue
            # A bare `[Y/n]` is a hint variable; its question is printed below.
            forward = _nearby_questions(lines, index, direction = 1)
            question = (
                normalise_question(text)
                or (forward[0] if forward else "")
                or f"<yes/no marker with no question: {text.strip()}>"
            )
            found.setdefault((script, question), (script, line_number, question))

        if _is_interactive_read(line, script):
            for question in _questions_for_read(lines, index):
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
        "studio/setup*.bat",
        "scripts/install*.sh",
        "scripts/install*.ps1",
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


def test_helpers_the_installers_invoke_are_scanned():
    """Naming the helpers by hand only ever covers the ones we thought of, so read
    them back out instead: whatever the installers reach, the guard scans. Every
    scanned script is a source, not just the entry points, so a helper that grows
    a helper of its own is caught as soon as the first one is listed here. Python
    helpers are scanned but not read back: a shell script naming a file runs it, a
    Python file naming one imports it, and library code is not an installer."""
    referenced = set()
    for script in SCANNED_SCRIPTS:
        if script.endswith(".py"):
            continue
        path = REPO_ROOT / script
        source = blank_comments(path.read_text(encoding = "utf-8"), script)
        for match in _HELPER_REF.finditer(source):
            # Below the script that names it, below the repo, or in scripts/, by
            # full path and by name so a URL still resolves to the local copy.
            # What resolves nowhere is a filename in a message, not an invocation.
            reference, name = match.group(1), match.group(1).rsplit("/", 1)[-1]
            for candidate in (
                path.parent / reference,
                REPO_ROOT / reference,
                path.parent / name,
                REPO_ROOT / "scripts" / name,
            ):
                if candidate.is_file():
                    referenced.add(candidate.resolve().relative_to(REPO_ROOT).as_posix())

    unscanned = sorted(referenced - set(SCANNED_SCRIPTS))
    assert not unscanned, (
        f"helpers invoked by the installers but not scanned: {unscanned}. "
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


@pytest.mark.parametrize(
    "read_line",
    (
        'read -p "  Build llama.cpp with CUDA support? " _reply',
        'read -rp "  Build llama.cpp with CUDA support? " _reply',
        'read -rsp "  Build llama.cpp with CUDA support? " _reply',
        'read -r -p "  Build llama.cpp with CUDA support? " _reply',
        'read -p"  Build llama.cpp with CUDA support? " _reply',
    ),
)
def test_detects_read_dash_p_without_a_marker(read_line: str):
    """`-p` carries the prompt itself, so there is no marker and only the read pass
    stands between these and a stalled install. Bundling is the usual bash
    spelling, so it must not hinge on a space before `-p`."""
    assert [question for _, _, question in find_prompts("studio/setup.sh", read_line + "\n")] == [
        "build llama.cpp with cuda support?",
    ]


def test_ignores_dash_p_belonging_to_a_later_command():
    """`mkdir -p` after a non-interactive read is not a prompt: matching it fails CI
    on an ordinary installer edit."""
    source = (
        'read -r _line < "$config_file"; mkdir -p "$_dest"\n'
        'while IFS= read -r _root; do mkdir -p "$_root"; done < "$manifest"\n'
        'read -r _v < "$_pci_vendor" && install -p "$_v" "$_dest"\n'
    )
    assert find_prompts("install.sh", source) == []


def test_detects_powershell_read_host():
    source = '$reply = Read-Host "  Install desktop shortcuts? [Y/n]"\n'
    assert [question for _, _, question in find_prompts("install.ps1", source)] == [
        "install desktop shortcuts?",
    ]


def test_detects_powershell_console_readline():
    """Read-Host is the usual spelling, but the console reads block just as hard and
    can carry their question in a variable, out of the marker pass's reach."""
    source = (
        'Write-Host "  Install desktop shortcuts? $_hint" -NoNewline\n'
        "$reply = [Console]::ReadLine()\n"
    )
    assert [question for _, _, question in find_prompts("install.ps1", source)] == [
        "install desktop shortcuts?",
    ]


@pytest.mark.parametrize("read_line", ("read -r _reply", "read _reply", "  read -r _reply"))
def test_detects_a_bare_read_from_inherited_stdin(read_line: str):
    """The commonest prompt of all: a question printed with no marker, answered by a
    read with no redirection. It takes the terminal the installer inherited."""
    source = f'printf "  Continue with installation? "\n{read_line}\n'
    assert [question for _, _, question in find_prompts("install.sh", source)] == [
        "continue with installation?",
    ]


def test_detects_a_bare_read_with_a_fallback():
    """`|| reply=n` is the house style for an EOF default, and a fallback is not a
    pipeline: the read still takes the terminal."""
    source = 'printf "  Continue with installation? "\nread -r _reply || _reply=n\n'
    assert [question for _, _, question in find_prompts("install.sh", source)] == [
        "continue with installation?",
    ]


@pytest.mark.parametrize(
    "read_line", ("read -r -n 1 _reply", "read -r -t 10 _reply", "read -rn1 _reply")
)
def test_detects_a_bare_read_with_option_arguments(read_line: str):
    """A one-character or timed confirmation is still a confirmation."""
    source = f'printf "  Continue with installation? "\n{read_line}\n'
    assert [question for _, _, question in find_prompts("install.sh", source)] == [
        "continue with installation?",
    ]


def test_detects_an_assignment_prefixed_read():
    """`IFS= read -r reply` is one command, not an assignment."""
    source = 'printf "  Continue with installation? "\nIFS= read -r _reply\n'
    assert [question for _, _, question in find_prompts("install.sh", source)] == [
        "continue with installation?",
    ]


def test_detects_a_python_helper_prompt():
    """setup.sh runs these with `python <helper>.py`, so `input()` blocks the update."""
    source = 'reply = input("Enable telemetry? [Y/n] ")\n'
    assert [question for _, _, question in find_prompts("studio/install_python_stack.py", source)][
        0
    ] == "enable telemetry?"


def test_detects_a_batch_prompt():
    source = 'set /p _reply="Enable telemetry? [Y/n] "\n'
    assert [question for _, _, question in find_prompts("studio/setup.bat", source)] == [
        "enable telemetry?",
    ]


def test_keeps_code_after_a_block_comment_terminator():
    source = '<#\n.SYNOPSIS\n#> $reply = Read-Host "Enable telemetry? [Y/n]"\n'
    assert [question for _, _, question in find_prompts("install.ps1", source)] == [
        "enable telemetry?",
    ]


def test_detects_powershell_host_ui_readline():
    source = 'Write-Host "  Continue? " -NoNewline\n$reply = $Host.UI.ReadLine()\n'
    assert [question for _, _, question in find_prompts("install.ps1", source)] == ["continue?"]


def test_helper_reference_keeps_its_subdirectory():
    """A nested helper resolves by its path, a URL by its name."""
    assert _HELPER_REF.findall('sh "$SCRIPT_DIR/helpers/build_deps.sh"') == [
        "helpers/build_deps.sh",
    ]


def test_detects_a_select_loop():
    """The other builtin that blocks for an answer. PS3 holds the question."""
    source = 'PS3="  Continue with installation? "\nselect _reply in Yes No; do break; done\n'
    assert [question for _, _, question in find_prompts("install.sh", source)] == [
        "continue with installation?",
    ]


def test_interpolation_cannot_rewrite_an_approved_question():
    """Splicing new wording into an approved prompt is a new question. A substitution
    at the end is only the marker hint, so that one still normalises away."""
    assert normalise_question("Start Unsloth Studio %s now? [Y/n]") == (
        "start unsloth studio <var> now?"
    )
    assert normalise_question("  Start Unsloth Studio now? %s ") == "start unsloth studio now?"


def test_ignores_an_input_api_named_in_a_message():
    source = 'Write-Host "Read-Host is unavailable on this terminal"\n'
    assert find_prompts("install.ps1", source) == []


def test_ignores_a_single_line_powershell_block_comment():
    source = '<# Read-Host "Enable telemetry? [Y/n]" #>\nWrite-Host "done"\n'
    assert find_prompts("install.ps1", source) == []


def test_detects_powershell_console_read():
    source = 'Write-Host "  Continue? " -NoNewline\n$key = [Console]::Read()\n'
    assert [question for _, _, question in find_prompts("install.ps1", source)] == ["continue?"]


def test_validates_every_question_a_read_can_ask():
    """One read serving a branch each: approving the nearest question would wave the
    other one through under an allowlisted key."""
    source = (
        'if [ "$_telemetry" = ask ]; then\n'
        '    printf "  Enable telemetry? [Y/n] "\n'
        "else\n"
        '    printf "  Start Unsloth Studio now? [Y/n] "\n'
        "fi\n"
        'read -r _reply </dev/tty || _reply="n"\n'
    )
    questions = {question for _, _, question in find_prompts("install.sh", source)}
    assert "enable telemetry?" in questions


def test_ignores_a_powershell_block_comment():
    """Comment-based help documents what the script does, sometimes by example. A
    prompt that cannot run must not fail CI: studio/setup.ps1 opens with such a block."""
    source = (
        "<#\n"
        ".SYNOPSIS\n"
        '    Historically this asked Read-Host "Enable telemetry? [Y/n]" here.\n'
        "#>\n"
        'Write-Host "done"\n'
    )
    assert find_prompts("install.ps1", source) == []


def test_helper_reference_matches_a_sibling_path():
    """Helpers invoke each other by their own directory, with no scripts/ prefix."""
    assert _HELPER_REF.findall('run_quiet "$SCRIPT_DIR/build_deps.sh" --yes') == ["build_deps.sh"]


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
