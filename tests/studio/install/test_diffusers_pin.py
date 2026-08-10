# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The pinned Diffusers revision has to survive a fresh install.sh, not just an update.

MiniMax-H3 needs a Diffusers revision newer than any published release, and Studio
refuses to load it otherwise. The pin originally lived in
studio/backend/requirements/base.txt, which looks like the obvious home and is
completely dead on the install path that matters: install.sh installs unsloth itself
(which drags a diffusers RELEASE in from PyPI as a transitive dependency) and then runs
install_python_stack.py with SKIP_STUDIO_BASE=1, where the base-packages step is a bare
`pass`. A clean install therefore ended up on the release, every time, with no error.

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


def test_the_pin_file_exists_and_names_an_exact_revision():
    assert PIN_FILE.is_file(), f"{PIN_FILE} is missing"
    lines = _requirements(PIN_FILE)
    urls = [line for line in lines if "://" in line]
    assert len(urls) == 1, f"expected exactly one pinned URL, got {urls}"
    # A branch or tag would move under us; only a 40-char commit sha is reproducible.
    assert re.search(
        r"/archive/[0-9a-f]{40}\.zip", urls[0]
    ), f"the diffusers pin must name a full commit sha, not a moving ref: {urls[0]}"
    assert 'python_version >= "3.10"' in urls[0], (
        "diffusers dropped Python 3.9 in 0.38, so the archive needs a >= 3.10 marker or "
        "the resolver has no candidate at all on a 3.9 host"
    )


def test_only_the_pin_file_names_diffusers():
    """One source of truth. A second entry anywhere is how a release creeps back in:
    whichever step runs last wins, and the step order is not obvious from any one file."""
    offenders = {}
    for path in sorted(REQ_ROOT.rglob("*.txt")):
        if path == PIN_FILE:
            continue
        named = [line for line in _requirements(path) if line.lower().startswith("diffusers")]
        if named:
            offenders[str(path.relative_to(REPO_ROOT))] = named
    assert not offenders, (
        f"diffusers is requirement-listed outside diffusers-pin.txt: {offenders}. "
        f"Move it into the pin file; base.txt in particular is skipped entirely by install.sh."
    )


def test_the_pin_step_is_not_gated_by_skip_base_or_no_torch():
    """The whole bug in one assertion. base.txt's step lives under `if skip_base: pass`,
    so it never runs on install.sh; the pin's step has to sit at function top level,
    outside every conditional, or it inherits the same hole."""
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
        "function in install_python_stack.py. Nested under an `if`, the pin repeats the "
        "base.txt bug: applied on `unsloth studio update`, skipped on a fresh install.sh."
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


def test_install_sh_still_skips_the_base_step():
    """Guards the premise. If install.sh ever stops setting SKIP_STUDIO_BASE=1 this test
    fails loudly and the comments above (and the pin's separate file) can be revisited,
    rather than quietly describing an installer that no longer behaves that way."""
    assert 'SKIP_STUDIO_BASE="$_SKIP_BASE"' in INSTALL_SH.read_text(encoding = "utf-8")
    assert "_SKIP_BASE=1" in INSTALL_SH.read_text(encoding = "utf-8")
    stack = STACK.read_text(encoding = "utf-8")
    assert "if skip_base:\n        pass" in stack
