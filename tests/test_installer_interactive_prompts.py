# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Refuse new interactive yes/no prompts in the installer and setup scripts.

#7016 added `Open Unsloth Studio in your default browser after launch?
[Y/n]` and had to be reverted in #8040: extra questions stall a piped
install and persist an answer nobody can find again. The only preference
setup may ask about is whether to start Unsloth when it finishes.

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

# What a user runs to install, update or remove Unsloth.
# Everything these launch in turn is scanned too (`unsloth studio update` runs setup.sh, which builds whisper.cpp;
# install.sh fetches and runs the WSL bootstrap), since a question down there stalls the same install.
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
    "studio/install_manifest.py",
    "studio/install_node_prebuilt.py",
    "studio/install_python_stack.py",
    "studio/install_sd_cpp_prebuilt.py",
    "studio/install_whisper_prebuilt.py",
    # install_python_stack runs this one with sys.executable.
    "studio/backend/requirements/single-env/patch_metadata.py",
)

# Every question these scripts may ask, keyed by (script, normalised question) because line numbers move and wording
# does not. Do NOT add an entry just to turn a build green: decide the prompt is wanted first.
APPROVED_PROMPTS: dict[tuple[str, str], str] = {
    ("install.sh", "start unsloth studio now?"): (
        "The one sanctioned preference prompt: launch Unsloth after install."
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

# Anything that blocks waiting on a human. `-p` is matched as an option word, not anchored on whitespace: it may be
# bundled (`read -rp`) or follow `read` directly. The scan stops at `;|&`, since a `mkdir -p` later on the line is
# another command.
_POSIX_READ = re.compile(
    r"(?:^|[\s;&|(])read\s+(?![a-zA-Z_]+=)"
    r"(?:-[a-zA-Z]*p(?=[\s\"']|$)|[^\n;|&]*?(?:<\s*/dev/tty|\s-[a-zA-Z]*p(?=[\s\"']|$)))"
)

# An unredirected `read` takes the terminal it inherited, so it blocks too. Loop and pipeline reads are fed by the
# `done < ...` or the pipe: data, not questions. Options then variable names to end of line, in command position, so
# the word `read` in a heredoc of prose is not a prompt.
_BARE_READ = re.compile(
    r"(?:^|[;&(]|\b(?:if|elif|then|else)\b)\s*!?\s*(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*"
    # No variable name at all is valid: the answer lands in $REPLY.
    r"read(?:\s+(?:-[a-zA-Z0-9]+|[\d.]+|\"\"|''))*(?:\s+[A-Za-z_][A-Za-z0-9_]*)*"
    r"(?:\s*(?:;|\|\||&&).*)?\s*(?:#.*)?$"
)
_LOOP = re.compile(r"\b(?:while|until|for)\b")

# `select reply in Yes No` is the other builtin that blocks for an answer; its question is the PS3 assignment above it,
# which the nearby scan already reads.
_SELECT = re.compile(r"(?:^|[;&(])\s*select\s+[A-Za-z_][A-Za-z0-9_]*\s+in\s")

_PWSH_READ = re.compile(
    r"Read-Host|PromptForChoice|(?:Console\]::(?:In\.)?|UI\.)Read(?:Line|Key)?\b|ReadKey\s*\(",
    re.IGNORECASE,
)

# The Python helpers setup runs, and the batch launcher.
_PY_READ = re.compile(
    r"(?<![.\w])(?:input|getpass)\s*\(|getpass\.getpass\s*\("
    r"|click\.confirm\s*\(|sys\.stdin(?:\.buffer)?\.read(?:line)?\s*\("
)
# In command position: starting the line or a `&`-joined command, after `do`, or as the body of a single-line `if`.
# Echoing the word `choice` is not a prompt. `pause` asks nothing but waits for a keypress, which stalls setup just
# the same.
_BAT_READ = re.compile(
    r"(?:^\s*|[&(]\s*|\bdo\s+)@?\s*(?:set\s+/p\b|choice\b|pause\b)"
    r"|^\s*@?\s*(?:if|else)\b.*?\s(?:set\s+/p\b|choice\b|pause\b)",
    re.IGNORECASE,
)

# A helper filename, with or without its `scripts/` prefix: sibling invocations such as "$SCRIPT_DIR/build_deps.sh"
# carry no prefix. Resolved against the repo before it counts, so a name that is not a real script is ignored.
_HELPER_REF = re.compile(r"(?<![$\w])((?:[A-Za-z0-9_.-]+[\\/])*[A-Za-z0-9_.-]+\.(?:sh|ps1|py|bat))")

_QUOTED = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"' r"|'([^']*)'")

# An f-string field that calls something.
_FIELD = re.compile(r"\{[^}]*\(")

# `<# ... #>` opened and closed on one line.
_INLINE_BLOCK = re.compile(r"<#.*?#>", re.DOTALL)

# `@' ... '@` is literal, unlike the expandable `@" ... "@`. The uninstaller prints its help from one and install.ps1
# embeds source in the other.
_HERESTRING_OPEN = re.compile(r"@'\s*$")
_HERESTRING_CLOSE = re.compile(r"^\s*'@")

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


def _blank_strings(line: str, script: str = "") -> str:
    """Blank quoted text so a message naming an input API is not one. A string that
    still executes is kept: a `$(...)` in a double-quoted shell or PowerShell string,
    a field in an f-string. Single quotes and a missing f prefix are literal."""

    def keep(match: re.Match) -> str:
        text = match.group(0)
        if script.endswith(".py"):
            # Past the quotes, so a triple-quoted f-string keeps its prefix.
            prefix = line[: match.start()].rstrip("\"'")[-2:].lower()
            expands = "f" in prefix and _FIELD.search(text)
        else:
            expands = text.startswith('"') and "$(" in text
        return text if expands else '""'

    return _QUOTED.sub(keep, line)


def _is_interactive_read(
    line: str,
    script: str,
    *,
    loop_input: bool = False,
) -> bool:
    """A read that waits on a person: a prompt option, /dev/tty, `select`, `input()`,
    `set /p`, or plain inherited stdin. Redirected and loop reads consume a file, not
    a person. Quoted text is blanked first, so a message naming `Read-Host` is not
    one."""
    code = _blank_strings(line, script)
    if script.endswith(".ps1"):
        return bool(_PWSH_READ.search(code))
    if script.endswith(".py"):
        return bool(_PY_READ.search(code))
    if script.endswith(".bat"):
        # `set /p version=<VERSION.txt` reads the file, not the user, but the redirection belongs to that command alone.
        return any("<" not in part and _BAT_READ.search(part) for part in code.split("&"))
    if _POSIX_READ.search(code) or _SELECT.search(code):
        return True
    # Inside a file-fed loop only a bare read consumes the file: an explicit /dev/tty or -p read above overrode it and
    # still waits on the terminal.
    if loop_input:
        return False
    # Per command: in `read -r reply; echo done | tee log` the pipe is the echo's. `||` is a fallback, not a pipeline,
    # and leaves the read on the terminal.
    for command in code.split(";"):
        piped = "|" in command.replace("||", "")
        if "<" not in command and not piped and _BARE_READ.search(command):
            return True
    return False


# `<<EOF`, not the `<<<` here-string, and not inside a quoted string: install.sh prints a shell-profile marker
# containing `# <<< Unsloth ... <<<`.
_HEREDOC = re.compile(r"<<(?!<)-?\s*[\"']?([A-Za-z_][A-Za-z0-9_]*)[\"']?")
_INTERPRETER = re.compile(r"\b(?:python[0-9.]*|node|perl|ruby|osascript)\b[^<]*<<")


def _blank_heredocs(lines: list[str]) -> list[str]:
    """Blank heredoc bodies, keeping the line count. A `read -r reply` shown in a
    help text is documentation: scripts/uninstall.sh already prints one."""
    out = list(lines)
    terminator = ""
    start = 0
    for index, line in enumerate(lines):
        if terminator:
            if line.strip() == terminator:
                out[start:index] = [""] * (index - start)
                terminator = ""
            continue
        # Openers come from code: not from a string, not from an inline comment.
        code = _blank_strings(line).split("#")[0]
        match = _HEREDOC.search(code)
        # `python - <<PY` runs its body. Blanking that would hide real code, which is the one thing worse than
        # scanning it as shell.
        if match and not _INTERPRETER.search(code):
            terminator, start = match.group(1), index + 1
    # An unterminated opener was not one. Blanking to EOF would silently blind the scan for the rest of the file,
    # prompts included.
    return out


def _redirected_loop_bodies(lines: list[str]) -> set[int]:
    """Indices inside a `do ... done < file` block. The redirection feeds every read
    in the body, but it sits on the `done`, so the read line alone looks interactive."""
    inside: set[int] = set()
    opened: list[int] = []
    for index, line in enumerate(lines):
        if re.search(r"(?:^|[;&])\s*do\b|\bdo\s*$", line):
            opened.append(index)
        if re.match(r"\s*done\b", line):
            if not opened:
                continue
            start = opened.pop()
            # Only stdin feeds them: `done | tee` pipes the output away, and `done 3<config` opens a spare descriptor.
            if re.search(r"(?<![1-9])<", line):
                inside.update(range(start, index))
    return inside


def _outside_docstring(line: str, delimiter: str) -> tuple[str, str]:
    """Strip triple-quoted regions from a Python line, returning what is left and the
    delimiter still open. A docstring showing an `input()` example documents the
    script, it does not prompt."""
    while True:
        if delimiter:
            end = line.find(delimiter)
            if end == -1:
                return "", delimiter
            line, delimiter = line[end + 3 :], ""
            continue
        opener = min((i for i in (line.find('"""'), line.find("'''")) if i != -1), default = -1)
        if opener == -1:
            return line, ""
        # An f-string field executes, so this one is code, not documentation.
        if "f" in line[max(0, opener - 2) : opener].lower():
            return line, ""
        head, delimiter = line[:opener], line[opener : opener + 3]
        rest, delimiter = _outside_docstring(line[opener + 3 :], delimiter)
        return head + rest, delimiter


def blank_comments(source: str, script: str) -> str:
    """Blank comments, keeping the line count so numbers still line up. `.ps1` files
    carry `<# ... #>` blocks, where a documented Read-Host example would otherwise
    fail CI over a prompt that cannot run. `.bat` comments are REM or `::`."""
    lines = []
    in_block = False
    in_docstring = ""
    powershell = script.endswith(".ps1")
    batch = script.endswith(".bat")
    python = script.endswith(".py")
    opened = -1
    in_herestring = False
    for index, line in enumerate(source.splitlines()):
        if powershell and in_block:
            # Code can follow the terminator on the same line.
            lines.append(line.split("#>", 1)[1] if "#>" in line else "")
            in_block = "#>" not in line
            continue
        # A comment naming a delimiter is not one: reading it as an opener would blank the rest of the file and take
        # every prompt in it out of the scan.
        if _is_comment(line) or (batch and re.match(r"\s*(?:REM\b|::)", line, re.IGNORECASE)):
            lines.append("")
            continue
        if python:
            line, in_docstring = _outside_docstring(line, in_docstring)
            if in_docstring and opened < 0:
                opened = index
            elif not in_docstring:
                opened = -1
        if powershell and in_herestring:
            lines.append("")
            in_herestring = not _HERESTRING_CLOSE.match(line)
            continue
        if powershell:
            if _HERESTRING_OPEN.search(_blank_strings(line)):
                in_herestring = True
            line = _INLINE_BLOCK.sub("", line)
            if "<#" in line:
                lines.append(line.split("<#", 1)[0])
                in_block, opened = True, index
                continue
        lines.append(line)
    if in_block or in_docstring:
        # Unterminated, so it was never a comment or a docstring. Put the lines back.
        original = source.splitlines()
        lines[opened + 1 :] = original[opened + 1 :]
    return "\n".join(lines)


def _quoted_strings(line: str) -> list[str]:
    """Outer strings first, then anything quoted inside an executable region."""
    found = []
    for match in _QUOTED.finditer(line):
        for group in match.groups():
            if not group:
                continue
            found.append(group)
            if "$(" in group or _FIELD.search(group):
                found.extend(_quoted_strings(group))
    return found


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
    if not script.endswith((".ps1", ".py", ".bat")):
        lines = _blank_heredocs(lines)
    redirected = _redirected_loop_bodies(lines)
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

        if _is_interactive_read(line, script, loop_input = index in redirected):
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
        "setup*.sh",
        "setup*.ps1",
        "setup*.bat",
        "install*.bat",
        "install*.py",
        "studio/install_*.py",
        "scripts/install*.sh",
        "scripts/install*.ps1",
        "uninstall*.sh",
        "uninstall*.ps1",
        "uninstall*.bat",
        "uninstall*.py",
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
    a helper of its own is caught as soon as the first one is listed here."""
    referenced = set()
    for script in SCANNED_SCRIPTS:
        path = REPO_ROOT / script
        source = blank_comments(path.read_text(encoding = "utf-8"), script)
        for match in _HELPER_REF.finditer(source):
            # Below the script that names it, below the repo, or in scripts/, by full path and by name so a URL still
            # resolves to the local copy. What resolves nowhere is a filename in a message, not an invocation.
            reference = match.group(1).replace("\\", "/")
            name = reference.rsplit("/", 1)[-1]
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


def test_the_workflow_runs_for_every_scanned_script():
    """A scanned script whose path filter is missing is only checked on the PR that
    registers it. Read the filters back rather than keeping two lists in step by
    hand. Parsed with a regex, not yaml: the parity runner installs pytest only."""
    workflow = (REPO_ROOT / ".github/workflows/cross-platform-parity-ci.yml").read_text(
        encoding = "utf-8"
    )
    blocks, collecting = [], None
    for line in workflow.splitlines():
        entry = re.match(r"^\s*-\s*'([^']+)'\s*$", line)
        if collecting is not None and entry:
            collecting.append(entry.group(1))
            continue
        # A comment or a blank line inside the list does not end it. Treating one as the end dropped every filter
        # after it, so this guard passed while reading nothing.
        if collecting is not None and (not line.strip() or line.strip().startswith("#")):
            continue
        if collecting:
            blocks.append(collecting)
        collecting = [] if line.strip() == "paths:" else None
    if collecting:
        blocks.append(collecting)
    assert blocks, "no paths: filters found in cross-platform-parity-ci.yml"

    def matcher(pattern: str) -> re.Pattern:
        # GitHub's `*` stops at a path separator; `**` does not.
        body = (
            re.escape(pattern)
            .replace(r"\*\*", "\x00")
            .replace(r"\*", "[^/]*")
            .replace("\x00", ".*")
        )
        return re.compile(body + "$")

    def covered(script: str, patterns: list[str]) -> bool:
        # GitHub applies `!` exclusions in order, so the last match decides.
        included = False
        for pattern in patterns:
            negated = pattern.startswith("!")
            if matcher(pattern.lstrip("!")).match(script):
                included = not negated
        return included

    # Each event on its own: a filter present only on push leaves pull requests bare.
    for patterns in blocks:
        uncovered = sorted(script for script in SCANNED_SCRIPTS if not covered(script, patterns))
        assert not uncovered, (
            f"scanned scripts with no path filter in cross-platform-parity-ci.yml: {uncovered}. "
            f"A later PR touching only one of these would not run this guard."
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


def test_a_comment_naming_a_delimiter_opens_nothing():
    """The heredoc lesson applied to the other two languages: reading a delimiter out
    of a comment blanks the rest of the file and takes every prompt in it with it."""
    python_source = '# opening delimiter is """\nprint("  Enable telemetry? ")\n_reply = input()\n'
    assert [
        question for _, _, question in find_prompts("studio/install_python_stack.py", python_source)
    ] == ["enable telemetry?"]
    pwsh_source = '# block comments begin with <#\n$reply = Read-Host "  Enable telemetry? "\n'
    assert [question for _, _, question in find_prompts("install.ps1", pwsh_source)] == [
        "enable telemetry?",
    ]


def test_an_unterminated_docstring_blanks_nothing():
    source = '_text = """open\nprint("  Enable telemetry? ")\n_reply = input()\n'
    questions = [
        question for _, _, question in find_prompts("studio/install_python_stack.py", source)
    ]
    assert "enable telemetry?" in questions


def test_detects_a_read_with_a_trailing_comment():
    source = 'printf "  Enable telemetry? "\nread -r _reply # use the inherited terminal\n'
    assert [question for _, _, question in find_prompts("install.sh", source)] == [
        "enable telemetry?",
    ]


def test_an_inline_comment_does_not_open_a_heredoc():
    source = ': # example uses <<EOF\nprintf "  Enable telemetry? "\nread -r _reply\n'
    questions = [question for _, _, question in find_prompts("install.sh", source)]
    assert "enable telemetry?" in questions


def test_an_unterminated_heredoc_opener_blanks_nothing():
    """Blanking to end of file would silently blind the scan for the rest of it."""
    source = 'cat <<EOF\nstuff\nprintf "  Enable telemetry? "\nread -r _reply\n'
    questions = [question for _, _, question in find_prompts("install.sh", source)]
    assert "enable telemetry?" in questions


def test_a_pipe_on_a_later_command_does_not_feed_the_read():
    source = 'printf "  Enable telemetry? "\nread -r _reply; echo done | tee install.log\n'
    questions = [question for _, _, question in find_prompts("install.sh", source)]
    assert "enable telemetry?" in questions


def test_a_spare_descriptor_on_the_done_is_not_stdin():
    source = (
        'while true; do\n    printf "  Enable telemetry? "\n    read -r _reply\ndone 3<config\n'
    )
    questions = [question for _, _, question in find_prompts("install.sh", source)]
    assert "enable telemetry?" in questions


def test_ignores_code_shaped_text_in_a_literal_string():
    """Single quotes do not expand in PowerShell, and a plain Python string has no
    fields. Keeping those would fail CI on a diagnostic message."""
    assert find_prompts("install.ps1", "Write-Host 'Example: $(Read-Host \"Q?\")'\n") == []
    assert (
        find_prompts("studio/install_python_stack.py", "print(\"Example: {input('Q?')}\")\n") == []
    )


def test_ignores_a_batch_read_from_a_file():
    assert find_prompts("studio/setup.bat", "set /p version=<VERSION.txt\n") == []


def test_detects_a_read_followed_by_another_command():
    source = 'printf "  Continue with installation? "\nread -r _reply; echo done\n'
    assert [question for _, _, question in find_prompts("install.sh", source)] == [
        "continue with installation?",
    ]


def test_keeps_a_loop_read_when_only_the_output_is_piped():
    """`done | tee` consumes the loop's stdout. Its reads still take the terminal."""
    source = 'while true; do\n    printf "  Enable telemetry? "\n    read -r _reply\ndone | tee install.log\n'
    questions = [question for _, _, question in find_prompts("install.sh", source)]
    assert "enable telemetry?" in questions


def test_ignores_a_command_documented_in_a_heredoc():
    source = "_usage() {\n    cat <<EOF\nExample: read -r reply\nEOF\n}\n"
    assert find_prompts("scripts/uninstall.sh", source) == []


def test_a_here_string_does_not_open_a_heredoc():
    """`<<<` is a here-string, and install.sh prints one inside a profile marker.
    Reading it as a heredoc blanked the rest of the file, prompts included."""
    source = (
        "printf '# <<< Unsloth marker <<<\\n'\nprintf \"  Enable telemetry? \"\nread -r _reply\n"
    )
    questions = [question for _, _, question in find_prompts("install.sh", source)]
    assert "enable telemetry?" in questions


def test_detects_a_prompt_inside_a_triple_quoted_f_string():
    source = 'print(f"""Answer: {input(\'Enable telemetry? \')}""")\n'
    questions = [
        question for _, _, question in find_prompts("studio/install_python_stack.py", source)
    ]
    assert "enable telemetry?" in questions


def test_detects_a_directly_imported_getpass():
    source = 'print("  Continue with installation? ")\n_reply = getpass()\n'
    assert [
        question for _, _, question in find_prompts("studio/install_python_stack.py", source)
    ] == ["continue with installation?"]


def test_detects_a_batch_pause():
    """It asks nothing, but an unattended install still stops dead on it."""
    source = "echo   Review the notes above.\npause\n"
    assert len(find_prompts("studio/setup.bat", source)) == 1


def test_detects_a_batch_prompt_in_a_conditional():
    source = 'if exist config choice /M "Enable telemetry?"\n'
    assert [question for _, _, question in find_prompts("studio/setup.bat", source)] == [
        "enable telemetry?",
    ]


def test_detects_a_negated_read():
    """`if ! cmd` is house style in these scripts; a read is no different."""
    source = 'printf "  Enable telemetry? "\nif ! read -r _reply; then _reply=n; fi\n'
    assert [question for _, _, question in find_prompts("install.sh", source)] == [
        "enable telemetry?",
    ]


def test_an_interpreter_heredoc_is_not_blanked():
    """`python - <<PY` runs its body. Hiding real code is worse than reading it as
    shell, which still leaves the marker pass looking at it."""
    source = 'python - <<PY\nprint("Enable telemetry? [Y/n]")\nPY\n'
    questions = [question for _, _, question in find_prompts("install.sh", source)]
    assert "enable telemetry?" in questions


def test_detects_a_read_used_as_a_condition():
    source = 'printf "  Continue with installation? "\nif read -r _reply; then :; fi\n'
    assert [question for _, _, question in find_prompts("install.sh", source)] == [
        "continue with installation?",
    ]


def test_keeps_an_explicit_terminal_read_inside_a_fed_loop():
    """`</dev/tty` overrides the loop's stdin, so only a bare read consumes the file."""
    source = (
        'while true; do\n    printf "  Enable telemetry? "\n'
        "    read -r _reply </dev/tty\ndone < manifest\n"
    )
    questions = [question for _, _, question in find_prompts("install.sh", source)]
    assert "enable telemetry?" in questions


def test_a_batch_redirection_belongs_to_its_own_command():
    source = 'choice /M "Enable telemetry?" & set /p version=<VERSION.txt\n'
    questions = [question for _, _, question in find_prompts("studio/setup.bat", source)]
    assert "enable telemetry?" in questions


def test_ignores_a_powershell_literal_here_string():
    """`@' ... '@` cannot expand, and the uninstaller prints its help from one."""
    source = "Write-Host @'\nRead-Host \"Enable telemetry? [Y/n]\"\n'@\n"
    assert find_prompts("scripts/uninstall.ps1", source) == []


def test_ignores_a_read_in_a_loop_redirected_at_the_done():
    """The redirection feeds the read, it just sits three lines below it."""
    source = 'while true; do\n    read -r _line || break\ndone < "$manifest"\n'
    assert find_prompts("install.sh", source) == []


def test_detects_a_python_stdin_read():
    source = 'print("  Continue with installation? ")\n_reply = sys.stdin.readline()\n'
    assert [
        question for _, _, question in find_prompts("studio/install_python_stack.py", source)
    ] == [
        "continue with installation?",
    ]


def test_detects_a_prompt_inside_an_f_string():
    """An f-string field executes, so blanking the string cannot blank it."""
    source = "print(f\"Answer: {input('Enable telemetry? ')}\")\n"
    questions = [
        question for _, _, question in find_prompts("studio/install_python_stack.py", source)
    ]
    assert "enable telemetry?" in questions


def test_ignores_a_python_function_named_confirm():
    source = "def confirm(value):\n    return value\n"
    assert find_prompts("studio/install_python_stack.py", source) == []


def test_detects_a_batch_prompt_with_echo_suppressed():
    source = '@choice /M "Enable telemetry?"\n'
    assert [question for _, _, question in find_prompts("studio/setup.bat", source)] == [
        "enable telemetry?",
    ]


def test_detects_a_read_with_no_variable():
    """`read -r` on its own is valid: the answer lands in $REPLY."""
    source = 'printf "  Continue with installation? "\nread -r\n'
    assert [question for _, _, question in find_prompts("install.sh", source)] == [
        "continue with installation?",
    ]


def test_detects_a_prompt_inside_an_expandable_string():
    """A `$(...)` subexpression executes, so blanking quoted text cannot blank it."""
    source = "Write-Host \"Response: $(Read-Host 'Enable telemetry?')\"\n"
    assert [question for _, _, question in find_prompts("install.ps1", source)] == [
        "enable telemetry?",
    ]


def test_ignores_a_batch_input_command_named_in_a_message():
    source = 'echo choice /M "Enable telemetry?" is not supported here\n'
    assert find_prompts("studio/setup.bat", source) == []


def test_ignores_a_python_docstring_example():
    source = '"""Doc.\n\nreply = input("Enable telemetry? [Y/n]")\n"""\nvalue = 1\n'
    assert find_prompts("studio/install_python_stack.py", source) == []


def test_helper_reference_keeps_a_windows_subdirectory():
    assert _HELPER_REF.findall(r'& "$PSScriptRoot\helpers\setup_extra.ps1"') == [
        r"helpers\setup_extra.ps1",
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
