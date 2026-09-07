# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The pinned Diffusers release has to survive a fresh install.sh, not just an update.

MiniMax-H3 and MiniMax Music 3 need Diffusers 0.40.0 or newer, and Unsloth refuses
to load them otherwise. The pin originally lived in
studio/backend/requirements/base.txt, which did not reach fresh install.sh installs at
the time. base.txt now reaches those installs as an independent shared phase, but it
still runs too early to hold this pin safely.

These tests pin the shape that fixes it: exactly one file names diffusers, and the step
that installs it sits outside every skip.
"""

from __future__ import annotations

import ast
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
REQ_ROOT = REPO_ROOT / "studio" / "backend" / "requirements"
PIN_FILE = REQ_ROOT / "diffusers-pin.txt"

# The shape install_python_stack._filter_requirements writes: a dot, the source stem,
# "-filtered-", then tempfile's random suffix. NamedTemporaryFile's suffixes are
# [A-Za-z0-9_]{8}, so this cannot swallow a checked-in file that merely starts with a dot.
_GENERATED_FILTER = re.compile(r"\.[\w.-]+-filtered-\w{8}\.txt")
STACK = REPO_ROOT / "studio" / "install_python_stack.py"
INSTALL_SH = REPO_ROOT / "install.sh"


def _requirements(path: pathlib.Path) -> list[str]:
    """Requirement lines only: comments and flag lines dropped."""
    out = []
    for line in path.read_text(encoding = "utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if text and not text.startswith("-"):
            out.append(text)
    return out


def test_the_pin_file_exists_and_names_the_first_supported_release():
    assert PIN_FILE.is_file(), f"{PIN_FILE} is missing"
    lines = _requirements(PIN_FILE)
    modern = [line for line in lines if 'python_version >= "3.10"' in line]
    assert modern == ['diffusers==0.40.0 ; python_version >= "3.10"'], modern
    assert "://" not in modern[0], "the released dependency must not require a source build"
    assert 'python_version >= "3.10"' in modern[0], (
        "diffusers dropped Python 3.9 in 0.38, so the release needs a >= 3.10 marker or "
        "the resolver has no candidate at all on a 3.9 host"
    )


def test_only_the_pin_file_names_diffusers():
    """One source of truth. A second entry anywhere is how a release creeps back in:
    whichever step runs last wins, and the step order is not obvious from any one file."""
    offenders = {}
    for path in sorted(REQ_ROOT.rglob("*.txt")):
        if path == PIN_FILE:
            continue
        # install_python_stack._filter_requirements writes `.{stem}-filtered-XXXX.txt` BESIDE the source on purpose, so
        # relative -r/-c includes still resolve, and it does not delete it.
        # Matched by that exact shape rather than by "starts with a dot": a checked-in hidden file such as
        # .constraints.txt is a real requirements file and a real place the pin could be overridden from, so it stays in
        # the scan.
        if _GENERATED_FILTER.fullmatch(path.name):
            continue
        named = [line for line in _requirements(path) if line.lower().startswith("diffusers")]
        if named:
            offenders[str(path.relative_to(REPO_ROOT))] = named
    assert not offenders, (
        f"diffusers is requirement-listed outside diffusers-pin.txt: {offenders}. "
        f"Move it into the pin file so the dedicated late step remains authoritative."
    )


def test_the_pin_step_is_not_gated_by_skip_base_or_no_torch():
    """The pin must sit at function top level so it reaches every install path."""
    tree = ast.parse(STACK.read_text(encoding = "utf-8"))

    def _installs_pin(node: ast.AST) -> bool:
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            if getattr(call.func, "id", None) != "pip_install":
                continue
            for kw in call.keywords:
                if kw.arg == "req" and "diffusers-pin.txt" in ast.dump(kw.value):
                    return True
        return False

    found = False
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef):
            continue
        for stmt in func.body:  # top level of the function only, no if/else nesting
            if _installs_pin(stmt):
                found = True
    assert found, (
        "no unconditional pip_install of diffusers-pin.txt found at the top level of any "
        "function in install_python_stack.py. Nested under an `if`, the pin can miss an "
        "install path."
    )


def test_the_pin_step_runs_after_every_other_requirements_install():
    """Ordering matters: a later `uv pip install -r ...` can re-resolve diffusers back to a
    release. Keeping the pin last means nothing is left that could walk it forward."""
    source = STACK.read_text(encoding = "utf-8")
    pin_at = source.index("diffusers-pin.txt")
    later = [
        name
        for name in (
            "extras.txt",
            "extras-no-deps.txt",
            "studio.txt",
            "base.txt",
            "no-torch-runtime.txt",
            "data-designer-deps.txt",
            "data-designer.txt",
        )
        if source.rfind(name) > pin_at
    ]
    assert not later, f"these requirements files are installed after the diffusers pin: {later}"


def test_install_sh_still_delegates_the_core_package_skip():
    """The handoff flag skips core packages while allowing other base entries through."""
    assert 'SKIP_STUDIO_BASE="$_SKIP_BASE"' in INSTALL_SH.read_text(encoding = "utf-8")
    assert "_SKIP_BASE=1" in INSTALL_SH.read_text(encoding = "utf-8")
