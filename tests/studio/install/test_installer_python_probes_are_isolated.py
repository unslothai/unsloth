# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""python -c probes that import studio run with -I, since cwd on sys.path can shadow the wheel."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_INSTALLERS = ("install.sh", "install.ps1", "studio/setup.sh", "studio/setup.ps1")

# `-c` and its quoted program, with the flags written before it on the same logical line.
_DASH_C = re.compile(r"(?P<flags>(?:[ \t]+-[A-Za-z]+)*)[ \t]+-c[ \t]+(?P<q>['\"])")
_IMPORTS_STUDIO = re.compile(
    r"\bimport studio\b|\bfrom studio\b|files\(\s*['\"]studio['\"]\s*\)|import [\w ,]*\bstudio\b"
)
# The word before the flags names the interpreter: a python path or a variable holding one.
_INTERPRETER = re.compile(r"(?i)(python[\w.]*\"?|\$\w*py\w*\"?|\$\{?\w*py\w*\}?\"?)$")


def _probes(text: str) -> list[tuple[str, str]]:
    """(flags, program) for every `<python> [flags] -c '<program>'` in an installer."""
    joined = re.sub(
        r"[\\`]\r?\n[ \t]*", " ", text
    )  # bash `\` and PowerShell backtick continuations
    found = []
    for match in _DASH_C.finditer(joined):
        before = joined[: match.start()].rsplit("\n", 1)[-1].rstrip()
        if not _INTERPRETER.search(before):
            continue
        quote = match.group("q")
        end = joined.find(quote, match.end())
        if end == -1:
            continue
        found.append((match.group("flags"), joined[match.end() : end]))
    return found


def _offenders(text: str) -> list[str]:
    return [
        program.strip().splitlines()[0][:80]
        for flags, program in _probes(text)
        if _IMPORTS_STUDIO.search(program) and not re.search(r"(^|\s)-I\b", flags)
    ]


@pytest.mark.parametrize("rel", _INSTALLERS)
def test_every_studio_probe_ignores_the_callers_directory(rel):
    path = _ROOT / rel
    if not path.is_file():
        pytest.skip(f"{rel} is not in this tree")
    offenders = _offenders(path.read_text(encoding = "utf-8"))
    assert not offenders, (
        f"{rel} imports `studio` through `python -c` without -I, so a run from an unsloth "
        f"checkout resolves the checkout's studio package instead of the venv's: {offenders}"
    )


def test_the_scan_finds_the_probes_it_guards():
    """The sweep must actually see install.sh's setup.sh lookup, or it would pass by finding
    nothing at all."""
    programs = [p for _, p in _probes((_ROOT / "install.sh").read_text(encoding = "utf-8"))]
    assert any("files('studio') / 'setup.sh'" in p for p in programs)
    assert any("installed_version_probe" in p for p in programs)


@pytest.mark.parametrize(
    "snippet, bad",
    [
        (
            'SETUP_SH=$("$VENV_DIR/bin/python" -c "\nimport importlib.resources\nprint(importlib.resources.files(\'studio\'))\n")',
            True,
        ),
        (
            'SETUP_SH=$("$VENV_DIR/bin/python" -I -c "\nprint(importlib.resources.files(\'studio\'))\n")',
            False,
        ),
        ("v=$(\"$_VENV_PY\" -c '\nfrom studio.install_manifest import x\n')", True),
        ('$x = (& $VenvPython -c "import pathlib, studio; print(studio.__file__)")', True),
        ('$x = (& $VenvPython -I -c "import pathlib, studio; print(studio.__file__)")', False),
        (
            '"$VENV_DIR/bin/python" -I \\\n    -c "import importlib.resources as r; print(r.files(\'studio\'))"',
            False,
        ),
        ('"$PY" -c "import torch; print(torch.__version__)"', False),
    ],
)
def test_the_scan_tells_an_isolated_probe_from_a_bare_one(snippet, bad):
    assert bool(_offenders(snippet)) is bad
