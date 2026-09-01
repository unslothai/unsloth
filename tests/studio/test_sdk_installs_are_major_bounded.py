# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SDKs the probes call by name must not float across a major.

On 2026-08-20 `anthropic` 1.0.0 was published. The inference smoke workflows installed
`anthropic>=0.40` with no upper bound, pip resolved the new major, and v1 had removed
`temperature`, `top_p` and `top_k` as accepted arguments on `messages.create()`. Every
probe that pins `temperature = 0.0` for determinism started raising:

    TypeError: Messages.create() got an unexpected keyword argument 'temperature'

75 job failures in about four hours, across every PR that ran those workflows, none of
them caused by the PR they were reported against. #9432 pinned `<1` to stop it.

This guard is deliberately NOT "every `>=` needs an upper bound". Most pins here are
libraries whose majors do not reach our code, and asserting on all of them would be noise
that gets suppressed. It covers the packages whose *call surface* our own probe scripts
use directly with keyword arguments, which is exactly the surface a major is allowed to
remove. Adding to this set needs that property; removing from it needs a reason.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"

# Packages our probe code calls directly by keyword, each with the first major it must NOT reach.
# Asserting the boundary rather than "some upper bound exists" is what makes this mean anything: `openai>=1.50,<999`
# contains a `<` and admits every major it is meant to keep out.
GUARDED = {
    # v1 removed temperature, top_p and top_k from messages.create().
    "anthropic": (1,),
    # 3.3.1 resolves today and the probes pass, so the bound sits above it, not at <2.
    "openai": (4,),
    "playwright": (2,),
}

# One `pip install` argument:
_REQUIREMENT = re.compile(r"""^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?(?P<spec>.*)$""")

# pip, pip3, pip3.12, pip.exe, and any of those behind a path with either separator, quoted or not.
# mlx-ci.yml:439 uses "$STUDIO_VENV/bin/pip", and the Windows workflows can use pip.exe;
_PIP_INSTALL = re.compile(
    r"""(?:^|[\s"'/\\])pip[0-9]*(?:\.[0-9]+)*(?:\.exe)?["']?\s+install(?:\s|$)"""
)

# The whole version token, suffix included.
# Capturing only the numeric prefix turned `<4.post1` into `<4`, and such a bound is LOOSER than its digits: `<4.post1`
# admits 4.0.
_UPPER = re.compile(r"(?P<op><=|<)(?P<version>[^,\s'\"]+)")
_NUMERIC = re.compile(r"^[0-9][0-9.]*$")
_EXACT = re.compile(r"===?(?P<version>[0-9][0-9.]*)")
_COMPATIBLE = re.compile(r"~=(?P<version>[0-9][0-9.]*)")


def _strip_inline_comment(line: str) -> str:
    """Drop a trailing shell comment, respecting quotes.

    Only whole-line comments were dropped before, so `run: echo ok  # pip install
    'openai<4'` was scanned as a real install: commented text could satisfy the bound
    and anti-vacuity checks, and merely naming a bare guarded package after a `#` could
    fail CI. A `#` only opens a comment at the start of a word, so `git+https://x#egg=y`
    survives.
    """
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
    """Every `pip install` COMMAND, continuations joined, comment lines dropped.

    `pip install \\` over several lines is the house style at 19 sites here, and a
    line-at-a-time scan sees only the first, so a guarded SDK below it would be
    unchecked while the anti-vacuity test stayed satisfied by the single-line pins.

    Comments go because #9432 put comments naming `anthropic` beside the pins, and one
    must not be able to satisfy or trip this.
    """
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
    """Every requirement for `package` in one pip command, as its raw specifier.

    "" means a bare `pip install openai`: a requirement with no constraint, which is
    not the same as the package being absent. Tokenized so a name inside a URL or a
    `-r` path is not mistaken for an install of it.
    """
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


def _release(raw: str) -> tuple[int, ...]:
    """Release segment as ints, every component kept.

    `~=` semantics depend on how many components were written -- `~=3.0` implies <4
    and `~=3.0.0` implies <3.1 -- so it needs this rather than the normalized form.
    """
    return tuple(int(part) for part in raw.strip(".").split(".") if part.isdigit())


def _version(raw: str) -> tuple[int, ...]:
    """Release with trailing zeros dropped, so 4.0 and 4 compare equal.

    Without this, `<4.0` was rejected against a boundary of `(4,)`: the tuples differ
    even though the two bounds are the same release.
    """
    parts = list(_release(raw))
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def _excludes(spec: str, major: tuple[int, ...]) -> bool:
    """Does this requirement keep `major` out?

    - no specifier at all resolves whatever is current, so it excludes nothing
    - `<2` and `<1.58` both exclude 2.x; `<=2` does not, since it admits 2.0 itself
    - `==3.0.0` cannot drift anywhere, so an exact pin below the major is safe
    - `~=1.4` means `>=1.4,<2`, so it carries an upper bound of its own

    Compared as integer tuples rather than through `packaging`, which the
    workflow-trigger-lint job does not install.
    """
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


def test_a_bound_above_the_next_major_is_not_accepted() -> None:
    """`<` alone is not the property; keeping the major out is.

    Checking only for the presence of an upper bound let `openai>=1.50,<999` through,
    which admits every major the pin exists to exclude.
    """
    assert not _excludes(">=1.50,<999", GUARDED["openai"])
    assert not _excludes(">=1.50,<5", GUARDED["openai"])
    assert not _excludes(">=1.50", GUARDED["openai"])
    # <=4 admits 4.0 itself, so it does not exclude the 4 series.
    assert not _excludes(">=1.50,<=4", GUARDED["openai"])
    assert _excludes(">=1.50,<4", GUARDED["openai"])
    # A narrower window than the boundary is stricter, and still fine.
    assert _excludes(">=1.55,<1.58", GUARDED["playwright"])
    assert _excludes(">=1.45,<2", GUARDED["playwright"])


def test_a_pin_that_cannot_drift_is_accepted() -> None:
    """An exact or compatible-release pin already excludes the major.

    Rejecting them would push people to add a redundant `<N` beside an `==`, so the
    rule is what the requirement can RESOLVE to, not which operator was typed.
    """
    assert _excludes("==3.0.0", GUARDED["openai"])
    assert _excludes("===3.0.0", GUARDED["openai"])
    assert not _excludes("==4.1.0", GUARDED["openai"])
    # ~=1.4 is >=1.4,<2, so it keeps 2.x out.
    assert _excludes("~=1.4", GUARDED["playwright"])
    # ~=1.4.5 is >=1.4.5,<1.5, narrower still.
    assert _excludes("~=1.4.5", GUARDED["playwright"])


def test_a_bare_or_extras_install_is_not_invisible() -> None:
    """`pip install openai` resolves whatever major is current.

    Requiring a version specifier matched nothing here, so the package was neither
    bounded nor reported. Extras are the same shape.
    """
    assert _requirements_in("pip install openai", "openai") == [""]
    assert _requirements_in("pip install 'openai[datalib]>=1.50'", "openai") == [">=1.50"]
    assert not _excludes("", GUARDED["openai"]), "a bare install constrains nothing"
    # A name inside a URL or a requirements path is not an install of that package.
    assert _requirements_in("pip install -r reqs/openai.txt", "openai") == []
    assert (
        _requirements_in(
            "pip install --index-url https://example.test/openai/simple pytest", "openai"
        )
        == []
    )


def test_a_pin_on_a_continuation_line_is_still_seen() -> None:
    """`pip install \\` over several lines is the house style at 19 sites here.

    A line-at-a-time scan sees only the first line, so a guarded SDK added below it would
    be invisible while the anti-vacuity test stayed satisfied by the single-line pins.
    """
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
    """`"$STUDIO_VENV/bin/pip" install` is already used at mlx-ci.yml:439.

    Matching the literal text `pip install` missed it and `pip3` alike, so a guarded
    SDK installed either way was never inspected.
    """
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
    # <4.0.1 still admits 4.0, so it does not exclude the 4 series.
    assert not _excludes(">=1.50,<4.0.1", GUARDED["openai"])


def test_a_bound_with_a_suffix_is_not_read_as_its_digits() -> None:
    """`<4.post1` admits 4.0, so it must not be read as `<4`.

    The regex used to capture only the numeric prefix, which quietly turned a looser
    bound into a passing one. Anything that is not a plain release now fails closed.
    """
    assert not _excludes(">=1.50,<4.post1", GUARDED["openai"])
    assert not _excludes(">=1.50,<4+local", GUARDED["openai"])
    assert _excludes(">=1.50,<4", GUARDED["openai"])


def test_a_compatible_pin_keeps_its_written_precision() -> None:
    """`~=3.0` implies <4 and `~=3.0.0` implies <3.1, so the component count matters.

    Normalizing trailing zeros before this branch collapsed both to `(3,)` and made the
    guard reject two valid pins.
    """
    assert _excludes("~=3.0", GUARDED["openai"])
    assert _excludes("~=3.0.0", GUARDED["openai"])
    assert _release("3.0.0") == (3, 0, 0)
    assert _version("3.0.0") == (3,)


def test_a_trailing_comment_is_not_an_install() -> None:
    """Only whole-line comments were dropped, which cut both ways.

    Commented text could satisfy the bound and anti-vacuity checks after the real
    installs were gone, and merely naming a bare guarded package after a `#` could fail
    CI. A `#` only opens a comment at the start of a word, so a URL fragment survives.
    """
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
    """PowerShell continues with a backtick, not a backslash.

    Nothing here uses the form yet, but 92 pwsh steps live in these workflows and 9 sit
    around a pip install, so it is one long install list away.
    """
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
