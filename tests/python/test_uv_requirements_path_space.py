"""Every uv `-r` in install.ps1 goes through the space-safety helper (issue #11012).

uv splits `-r`, `-c` and `--overrides` on whitespace and offers no quoting escape, so a path with a
space arrives as two bogus requirement files. `--overrides` was fixed in #10765 by making the file
install.ps1 writes space-free, but `-r $NoTorchReq` resolves under `$RepoRoot` or `$VenvDir`, both
chosen by the user, and was still passed through verbatim.

A static check rather than a behavioural one: the failure only reproduces on Windows with a spaced
install root, and that lane is not where a future edit would be caught. Asserting on the call shape
means the regression is caught wherever the test suite runs.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"

HELPER = "Get-UvSafeRequirementsPath"


def _code_lines(text: str) -> list[str]:
    """Drop whole-line comments so a commented example never satisfies an assertion."""
    return [line for line in text.splitlines() if not line.lstrip().startswith("#")]


def test_the_helper_is_defined():
    lines = _code_lines(INSTALL_PS1.read_text(encoding = "utf-8"))
    assert any(re.match(rf"\s*function\s+{HELPER}\b", line) for line in lines), (
        f"{HELPER} is missing from install.ps1"
    )


def test_every_uv_requirements_flag_uses_a_sanitised_path():
    lines = _code_lines(INSTALL_PS1.read_text(encoding = "utf-8"))

    # Variables the helper's result reaches, rather than a naming convention: first those assigned
    # from the helper itself, then those assigned from the .Path of one of those. Keying on a name
    # like "*Safe" would pass for a variable that merely looked sanitised.
    from_helper = set()
    for line in lines:
        hit = re.search(rf"(\$[A-Za-z_][A-Za-z0-9_]*)\s*=\s*{HELPER}\b", line)
        if hit:
            from_helper.add(hit.group(1))
    sanitised = set()
    for line in lines:
        hit = re.search(r"(\$[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\$[A-Za-z_][A-Za-z0-9_]*)\.Path\b", line)
        if hit and hit.group(2) in from_helper:
            sanitised.add(hit.group(1))
    sanitised |= {f"{name}.Path" for name in from_helper}

    # `-r <arg>` where the argument is a PowerShell variable. A literal path in the repo cannot
    # carry a user-chosen space, so only variables are of interest here.
    uses = [(n, line) for n, line in enumerate(lines, 1) if re.search(r"\s-r\s+\$", line)]
    assert uses, "no uv -r call sites found; this test is no longer measuring anything"

    unsafe = []
    for number, line in uses:
        arg = re.search(r"\s-r\s+(\$[A-Za-z_][A-Za-z0-9_:]*(?:\.Path)?)", line)
        if arg and arg.group(1) not in sanitised:
            unsafe.append(f"line {number}: {line.strip()}")
    assert not unsafe, (
        "uv -r is handed a path that never passed through "
        f"{HELPER}, so an install root containing a space truncates it:\n" + "\n".join(unsafe)
    )


def test_a_copy_is_removed_but_the_users_own_file_is_not():
    """The helper may return a copy; only a copy may be deleted."""
    text = INSTALL_PS1.read_text(encoding = "utf-8")
    body = text[text.index(f"function {HELPER}"):]
    body = body[:body.index("\n    function ")]
    assert "Temporary = $true" in body, "the helper never reports that it made a copy"
    assert "Temporary = $false" in body, "the helper never reports a pass-through path"

    for line in _code_lines(text):
        if "Remove-Item" in line and "NoTorchReqArg" in line:
            break
    else:
        raise AssertionError("a temporary copy is created but never removed")

    guarded = re.search(
        r"if \(\$NoTorchReqSafe\.Temporary\) \{\s*\n\s*Remove-Item", text
    )
    assert guarded, (
        "the Remove-Item is not guarded on .Temporary, so a pass-through path "
        "would delete the user's own requirements file"
    )
