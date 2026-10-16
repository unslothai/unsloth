# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Guards SDKs whose keyword arguments our probes call, since a major may remove them."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
ANTHROPIC_PROBES = (
    REPO / ".github" / "scripts" / "studio_smoke" / "multi_turn_chat.py",
    WORKFLOWS / "studio-inference-smoke.yml",
    WORKFLOWS / "studio-mac-ui-smoke.yml",
    WORKFLOWS / "studio-windows-inference-smoke.yml",
)

# Each with the first major it must NOT reach: `<999` contains a `<` and admits everything.
GUARDED = {
    # v1 probes pass sampling parameters through extra_body.
    "anthropic": (2,),
    # 3.3.1 resolves today and the probes pass, so the bound sits above it.
    "openai": (4,),
    "playwright": (2,),
}

# The specifier is optional: a bare `pip install openai` must be seen too.
_REQUIREMENT = re.compile(r"""^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?(?P<spec>.*)$""")

# Any pip spelling, behind any path, quoted or not (e.g. "$STUDIO_VENV/bin/pip").
_PIP_INSTALL = re.compile(
    r"""(?:^|[\s"'/\\])pip[0-9]*(?:\.[0-9]+)*(?:\.exe)?["']?\s+install(?:\s|$)"""
)

# The whole version token: `<4.post1` admits 4.0.
_UPPER = re.compile(r"(?P<op><=|<)(?P<version>[^,\s'\"]+)")
_NUMERIC = re.compile(r"^[0-9][0-9.]*$")
_EXACT = re.compile(r"===?(?P<version>[0-9][0-9.]*)")
_COMPATIBLE = re.compile(r"~=(?P<version>[0-9][0-9.]*)")


def _strip_inline_comment(line: str) -> str:
    """Trailing comments are stripped, but only where a hash starts a word, so URL fragments survive."""
    quote = ""
    for index, char in enumerate(line):
        if quote:
            if char == quote:
                quote = ""
        elif char in "'\"":
            quote = char
        elif char == "#" and (index == 0 or line[index - 1].isspace()):
            return line[:index]
    return line


def _install_commands_in(text: str) -> list[tuple[int, str]]:
    """Joins backslash continuations, so a guarded SDK on a later line of a pip install is seen."""
    commands = []
    pending: list[str] = []
    start = 0
    for number, line in enumerate(text.splitlines(), 1):
        line = _strip_inline_comment(line)
        if not line.strip():
            continue
        if not pending:
            start = number
        stripped = line.rstrip()
        # Backslash for sh, backtick for PowerShell.
        if stripped.endswith("\\") or stripped.endswith("`"):
            pending.append(stripped[:-1])
            continue
        pending.append(stripped)
        joined = " ".join(pending)
        pending = []
        if _PIP_INSTALL.search(joined):
            commands.append((start, joined))
    if pending:
        joined = " ".join(pending)
        if _PIP_INSTALL.search(joined):
            commands.append((start, joined))
    return commands


def _workflow_files(root: Path = WORKFLOWS) -> list[Path]:
    """Both extensions: GitHub accepts .yaml, and scanning only .yml would leave one
    silently unchecked while the .yml pins kept the anti-vacuity test satisfied."""
    return sorted(list(root.glob("*.yml")) + list(root.glob("*.yaml")))


def _install_lines() -> list[tuple[Path, int, str]]:
    found = []
    for path in _workflow_files():
        for number, command in _install_commands_in(path.read_text(encoding = "utf-8")):
            found.append((path, number, command))
    return found


def _requirements_in(command: str, package: str) -> list[str]:
    """Tokenized so a name inside a URL or -r path is not an install; an empty string is a bare install."""
    try:
        tokens = shlex.split(command, posix = True)
    except ValueError:
        tokens = command.split()
    found = []
    for token in tokens:
        if token.startswith("-") or "/" in token or "\\" in token:
            continue
        match = _REQUIREMENT.match(token)
        if not match:
            continue
        if match.group("name").lower().replace("_", "-") != package:
            continue
        found.append(match.group("spec"))
    return found


def _specs_for(package: str) -> list[tuple[Path, int, str]]:
    hits = []
    for path, number, command in _install_lines():
        for spec in _requirements_in(command, package):
            hits.append((path, number, spec))
    return hits


def _anthropic_create_blocks() -> list[tuple[Path, str]]:
    blocks = []
    for path in ANTHROPIC_PROBES:
        lines = path.read_text(encoding = "utf-8").splitlines()
        for index, line in enumerate(lines):
            if ".messages.create(" not in line:
                continue
            indent = len(line) - len(line.lstrip())
            block = [line]
            for continuation in lines[index + 1 :]:
                block.append(continuation)
                if (
                    continuation.strip() == ")"
                    and len(continuation) - len(continuation.lstrip()) == indent
                ):
                    break
            else:
                raise AssertionError(f"unterminated messages.create call in {path}")
            blocks.append((path, "\n".join(block)))
    return blocks


def _release(raw: str) -> tuple[int, ...]:
    """Keeps every written component: ~=3.0 means <4 but ~=3.0.0 means <3.1, so zeros matter."""
    return tuple(int(part) for part in raw.strip(".").split(".") if part.isdigit())


def _version(raw: str) -> tuple[int, ...]:
    """Drops trailing zeros so <4.0 and <4 compare equal; the tuples differed for the same release."""
    parts = list(_release(raw))
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def _excludes(spec: str, major: tuple[int, ...]) -> bool:
    """Only a spec that cannot resolve to the major counts; <=2 admits 2.0 itself, so it does not."""
    if not spec.strip():
        return False
    for match in _EXACT.finditer(spec):
        pinned = _version(match.group("version"))
        if pinned and pinned < major:
            return True
    for match in _COMPATIBLE.finditer(spec):
        release = _release(match.group("version"))
        # ~=X.Y means >=X.Y,<X+1; ~=X.Y.Z means >=X.Y.Z,<X.Y+1.
        if len(release) == 2:
            implied = (release[0] + 1,)
        elif len(release) > 2:
            implied = (release[0], release[1] + 1)
        else:
            continue
        if implied <= major:
            return True
    for match in _UPPER.finditer(spec):
        raw = match.group("version")
        # Fail closed on anything that is not a plain release.
        if not _NUMERIC.match(raw):
            continue
        bound = _version(raw)
        if not bound:
            continue
        if bound <= major if match.group("op") == "<" else bound < major:
            return True
    return False


@pytest.mark.parametrize("package", GUARDED)
def test_the_sdk_is_pinned_below_the_next_major(package: str) -> None:
    major = GUARDED[package]
    wanted = ".".join(str(part) for part in major)
    admits = [
        (path, number, spec)
        for path, number, spec in _specs_for(package)
        if not _excludes(spec, major)
    ]
    assert not admits, (
        f"{package} is installed in a way that admits {wanted} at "
        + ", ".join(f"{p.name}:{n} ({s})" for p, n, s in admits)
        + f". A new {package} major would reach CI with no change on our side; that is how"
        f" anthropic 1.0.0 broke 75 jobs. Pin below {wanted}, or move the boundary in this"
        " file in the same commit once CI has run against that major."
    )


def test_anthropic_sampling_parameters_use_extra_body() -> None:
    blocks = _anthropic_create_blocks()
    assert len(blocks) == 4, blocks
    for path, block in blocks:
        assert not re.search(r"\btemperature\s*=", block), path
        assert re.search(r'extra_body\s*=\s*\{[^\n]*["\']temperature["\']', block), path


def test_anthropic_smokes_install_v1() -> None:
    not_v1 = [
        (path, number, spec)
        for path, number, spec in _specs_for("anthropic")
        if not re.search(r"(?:^|,)>=1(?:\.\d+)*(?:,|$)", spec)
    ]
    assert not not_v1, not_v1


def test_a_bound_above_the_next_major_is_not_accepted() -> None:
    """Presence of an upper bound is not enough; a bound like <999 that admits the next major must fail."""
    assert not _excludes(">=1.50,<999", GUARDED["openai"])
    assert not _excludes(">=1.50,<5", GUARDED["openai"])
    assert not _excludes(">=1.50", GUARDED["openai"])
    assert not _excludes(">=1.50,<=4", GUARDED["openai"])
    assert _excludes(">=1.50,<4", GUARDED["openai"])
    assert _excludes(">=1.55,<1.58", GUARDED["playwright"])
    assert _excludes(">=1.45,<2", GUARDED["playwright"])


def test_a_pin_that_cannot_drift_is_accepted() -> None:
    """Judged by what the requirement can resolve to, not by operator: ==3.0.0 and ~=1.4 are accepted."""
    assert _excludes("==3.0.0", GUARDED["openai"])
    assert _excludes("===3.0.0", GUARDED["openai"])
    assert not _excludes("==4.1.0", GUARDED["openai"])
    assert _excludes("~=1.4", GUARDED["playwright"])
    assert _excludes("~=1.4.5", GUARDED["playwright"])


def test_a_bare_or_extras_install_is_not_invisible() -> None:
    """A bare or extras install still counts as an install, so an unbounded package is reported."""
    assert _requirements_in("pip install openai", "openai") == [""]
    assert _requirements_in("pip install 'openai[datalib]>=1.50'", "openai") == [">=1.50"]
    assert not _excludes("", GUARDED["openai"]), "a bare install constrains nothing"
    assert _requirements_in("pip install -r reqs/openai.txt", "openai") == []
    assert (
        _requirements_in(
            "pip install --index-url https://example.test/openai/simple pytest", "openai"
        )
        == []
    )


def test_a_pin_on_a_continuation_line_is_still_seen() -> None:
    """Pins on backslash-continued pip install lines must be seen; a line scan only reads the first."""
    text = (
        "      - name: Install\n"
        "        run: |\n"
        "          pip install 'pytest>=8' \\\n"
        "            'openai>=1.50' \\\n"
        "            'anthropic>=0.40,<1'\n"
    )
    commands = _install_commands_in(text)
    assert len(commands) == 1, commands
    number, command = commands[0]
    assert number == 3, "the command should be reported at the line it starts on"
    assert "openai>=1.50" in command and "anthropic>=0.40,<1" in command


def test_the_guard_is_reading_real_pins() -> None:
    """An empty scan would make every assertion above pass silently."""
    for package in GUARDED:
        assert _specs_for(package), (
            f"no `pip install` line in .github/workflows pins {package} any more; either"
            " the probes moved or the regex stopped matching, and the guard above is now"
            " vacuous"
        )


def test_a_commented_out_pin_is_not_mistaken_for_an_install() -> None:
    lines = _install_lines()
    assert lines, "no pip install lines found at all"
    assert all(not line.lstrip().startswith("#") for _, _, line in lines)


def test_a_yaml_workflow_is_scanned_too(tmp_path) -> None:
    """GitHub accepts .yaml. Scanning one extension leaves the other unchecked."""
    (tmp_path / "a.yml").write_text("x", encoding = "utf-8")
    (tmp_path / "b.yaml").write_text("x", encoding = "utf-8")
    assert [p.name for p in _workflow_files(tmp_path)] == ["a.yml", "b.yaml"]


def test_pip_is_recognized_beyond_the_bare_command() -> None:
    """Matches pip3 and a quoted venv pip path too, not only the literal pip install text."""
    for command in (
        "pip install openai",
        "pip3 install openai",
        '"$STUDIO_VENV/bin/pip" install openai',
        "python -m pip install openai",
    ):
        assert _PIP_INSTALL.search(command), command
    assert not _PIP_INSTALL.search("npm install openai")
    assert not _PIP_INSTALL.search("pip download openai")


def test_equivalent_bounds_compare_equal() -> None:
    """`<4.0` and `<4` are the same boundary; only tuple length differed."""
    assert _version("4.0") == _version("4") == (4,)
    assert _version("1.58") == (1, 58)
    assert _excludes(">=1.50,<4.0", GUARDED["openai"])
    assert _excludes(">=1.50,<4.0.0", GUARDED["openai"])
    assert not _excludes(">=1.50,<4.0.1", GUARDED["openai"])


def test_a_bound_with_a_suffix_is_not_read_as_its_digits() -> None:
    """A bound with a suffix, like <4.post1, admits 4.0 and must fail closed, not be read as <4."""
    assert not _excludes(">=1.50,<4.post1", GUARDED["openai"])
    assert not _excludes(">=1.50,<4+local", GUARDED["openai"])
    assert _excludes(">=1.50,<4", GUARDED["openai"])


def test_a_compatible_pin_keeps_its_written_precision() -> None:
    """~=3.0 means <4 but ~=3.0.0 means <3.1, so trailing zeros must not be normalized away."""
    assert _excludes("~=3.0", GUARDED["openai"])
    assert _excludes("~=3.0.0", GUARDED["openai"])
    assert _release("3.0.0") == (3, 0, 0)
    assert _version("3.0.0") == (3,)


def test_a_trailing_comment_is_not_an_install() -> None:
    """Trailing comments must not count as installs in either direction, so commented pins are ignored."""
    assert _install_commands_in("      - run: echo ok  # pip install 'openai<4'\n") == []
    assert _install_commands_in("          pip install 'openai>=1.50,<4'  # below 4\n")
    assert _strip_inline_comment("pip install 'git+https://x#egg=y'") == (
        "pip install 'git+https://x#egg=y'"
    )
    assert _strip_inline_comment("pip install git+https://x#egg=y") == (
        "pip install git+https://x#egg=y"
    )


def test_a_windows_pip_executable_is_recognized() -> None:
    """`pip.exe` matched neither the digits-only suffix nor the `/`-only separator."""
    for command in (
        "pip.exe install openai",
        '"$VENV\\Scripts\\pip.exe" install openai',
        "pip3.12.exe install openai",
    ):
        assert _PIP_INSTALL.search(command), command
    assert not _PIP_INSTALL.search("npm install openai")
    assert not _PIP_INSTALL.search("pip download openai")


def test_a_powershell_continuation_is_joined() -> None:
    """PowerShell continues lines with a backtick, not a backslash, so pwsh installs must be joined too."""
    text = (
        "        shell: pwsh\n"
        "        run: |\n"
        "          python -m pip install `\n"
        "            openai\n"
    )
    commands = _install_commands_in(text)
    assert len(commands) == 1, commands
    assert "openai" in commands[0][1]
    assert _requirements_in(commands[0][1], "openai") == [""]
