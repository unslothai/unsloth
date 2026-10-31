# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A --password login must not fall back to the bootstrap password, or the flag proves nothing."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = ROOT / "tests" / "kaggle" / "studio_gpu" / "run_studio_gpu.py"
SRC = PAYLOAD.read_text(encoding = "utf-8")


def test_the_password_reaches_the_studio_command():
    assert 'cmd += ["--password", self.args.studio_password]' in SRC


def test_login_uses_the_password_that_was_passed():
    assert "self.studio.login(self.args.studio_password)" in SRC


def test_there_is_no_fallback_to_the_bootstrap_password():
    """The whole assertion. If --password were ignored, Studio would seed a
    bootstrap password instead and the login would fail; a fallback would then
    quietly succeed and report a pass for a flag that did nothing."""
    tree = ast.parse(SRC)
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "authenticate"
    )
    branch = next(
        node
        for node in func.body
        if isinstance(node, ast.If) and "studio_password" in ast.dump(node.test)
    )
    # The branch must end in a Return: `if not failures: return` still falls through on failure.
    assert isinstance(branch.body[-1], ast.Return), (
        "the --password branch must END in an unconditional return; a "
        "conditional one falls through to the bootstrap path on failure and "
        "turns a flag that did nothing into a pass"
    )
    branch_src = ast.get_source_segment(SRC, branch) or ""
    assert "remember_bootstrap" not in branch_src, "the --password branch reads the bootstrap"


def test_the_generated_password_is_registered_as_a_secret_before_use():
    """Registered in __init__, which runs before the server starts and before
    anything writes a log. Registering it later would scrub the logs written
    after that point and not the banner."""
    tree = ast.parse(SRC)
    init = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    body = ast.get_source_segment(SRC, init) or ""
    assert "self.secrets.add(self.args.studio_password)" in body
    assert "secrets_module.token_urlsafe" in body, "auto must mint a fresh value per run"


def test_no_constant_password_is_committed():
    """A constant in a repo is a credential whether or not it is reachable."""
    tree = ast.parse(SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            names = [a.value for a in node.args if isinstance(a, ast.Constant)]
            if "--studio-password" in names:
                for kw in node.keywords:
                    if kw.arg == "default":
                        assert (
                            kw.value.value == ""
                        ), f"a default password is committed: {kw.value.value!r}"
                break
    else:
        raise AssertionError("--studio-password is not declared at all")


def test_the_default_keeps_the_previous_bootstrap_behaviour():
    """Empty means "use the bootstrap", so a caller that does not opt in is
    unaffected and the existing Studio leg does not change shape."""
    assert '"--studio-password",\n        default = "",' in SRC
