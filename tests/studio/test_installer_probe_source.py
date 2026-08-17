# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The payload probe must survive the quoting of the script that carries it.

setup.sh stores its probe in a DOUBLE-quoted shell string, so the shell rewrites the
source before python ever sees it: a doubled backslash collapses to one, which turned
`.replace('/', '\\\\')` into an unterminated literal and killed the probe outright. The
failure is invisible from the file: reading the text back gives valid Python, and the
runtime symptom is only an empty answer, which every caller treats as inconclusive.

So this compiles what the SHELL produces, not what the file contains, and bans the
construct at the root. setup.ps1 carries the same program in a single-quoted here-string
where no rewriting happens; it is compiled here too, and the two are held equivalent.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SETUP_SH = REPO / "studio" / "setup.sh"
SETUP_PS1 = REPO / "studio" / "setup.ps1"

SH_ASSIGN = re.compile(r'(_PKG_PROBE_PY="\n(.*?)\n")\n', re.S)
PS_HERESTRING = re.compile(r"\$_pkgProbeCode = @'\n(.*?)\n'@", re.S)


def _sh_probe() -> tuple[str, str]:
    m = SH_ASSIGN.search(SETUP_SH.read_text(encoding = "utf-8"))
    assert m, "_PKG_PROBE_PY is gone from setup.sh"
    return m.group(1), m.group(2)


def _bash() -> str:
    # A bare 'bash' on a Windows PATH is System32's WSL launcher, which exits 1
    # when no distribution is installed; the bash that runs shell scripts here is
    # the one Git ships.
    if sys.platform != "win32":
        return "bash"
    for env in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(env)
        if root and (cand := Path(root) / "Git" / "bin" / "bash.exe").exists():
            return str(cand)
    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        return found
    pytest.skip("no usable bash on this Windows runner")


def _shell_expanded(assignment: str, tmp_path: Path) -> str:
    script = tmp_path / "assign.sh"
    out = tmp_path / "probe.py"
    # as_posix + quotes: a raw C:\... path pasted into bash loses its backslashes
    # outside quotes, and Git-Bash on Windows accepts the forward-slash spelling.
    script.write_text(
        "#!/usr/bin/env bash\n"
        + assignment
        + f'\nprintf "%s" "$_PKG_PROBE_PY" > "{out.as_posix()}"\n',
        encoding = "utf-8",
    )
    subprocess.run([_bash(), str(script)], check = True)
    return out.read_text(encoding = "utf-8")


def test_shell_expanded_probe_is_valid_python(tmp_path: Path) -> None:
    assignment, raw = _sh_probe()
    expanded = _shell_expanded(assignment, tmp_path)
    # argv[1] is the package name the caller passes; any name compiles the same.
    compile(expanded.replace("sys.argv[1]", "'unsloth'"), "<shell-expanded probe>", "exec")
    assert expanded.strip() == raw.strip(), (
        "the shell rewrote the probe source. Whatever construct did that has to go: "
        "the probe runs what the SHELL produces, not what this file shows."
    )


def test_probe_avoids_the_escapes_the_shell_rewrites() -> None:
    """Inside double quotes bash rewrites a backslash only before \\ $ ` " or a newline.

    A lone `\\.` in a regex therefore survives and is fine; `\\\\` is what collapsed and
    broke the source. chr(92) is how to spell a backslash the shell cannot touch.
    """
    _, raw = _sh_probe()
    offenders = [
        f"{number}: {line.strip()}"
        for number, line in enumerate(raw.splitlines(), start = 1)
        if re.search(r'\\[\\$`"]', line)
    ]
    assert not offenders, (
        "these escapes are rewritten by the shell that carries the probe; "
        f"use chr(92): {offenders[:5]}"
    )
    quoted = [
        f"{number}: {line.strip()}"
        for number, line in enumerate(raw.splitlines(), start = 1)
        if '"' in line or "`" in line
    ]
    assert not quoted, (
        "a double quote ENDS the shell string carrying the probe and a backtick runs a "
        f"command inside it, comments included; use plain single quotes: {quoted[:5]}"
    )


def test_powershell_probe_compiles_and_matches() -> None:
    m = PS_HERESTRING.search(SETUP_PS1.read_text(encoding = "utf-8"))
    assert m, "$_pkgProbeCode is gone from setup.ps1"
    ps_probe = m.group(1)
    compile(ps_probe, "<powershell probe>", "exec")
    # A here-string cannot carry a double quote out of setup.ps1's -c wrapping either.
    assert '"' not in ps_probe, "the PowerShell probe must not contain a double quote"

    _, sh_probe = _sh_probe()

    def _norm(text: str) -> str:
        # the only intended difference: where the package name comes from
        text = text.replace("sys.argv[1]", "PKG").replace("_pkg", "PKG")
        return re.sub(r"^(import |from |PKG = os\.environ).*\n", "", text, flags = re.M)

    assert _norm(sh_probe) == _norm(ps_probe), "the two probes have drifted apart"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
