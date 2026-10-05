# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Starting Studio headless with --password, and the way that check goes vacuous.

`unsloth studio --password X` sets the INITIAL admin password when none is set
yet, which is exactly the state a scripted server is in. Without it the only way
in is the bootstrap password Studio seeds into a file and prints to its own log,
and a deployment that has to read a log to log in is not one anybody scripts
twice.

**The vacuity to avoid is a fallback.** If logging in with the passed password
failed and the payload then tried the bootstrap, the run would go green while
proving `--password` does nothing at all. There is deliberately no fallback, and
that is asserted from the source here.

**The second risk is a leaked credential.** The password reaches a process
command line, Studio's startup banner, `studio.log`, and the evidence bundle
that travels home. It is generated per run and registered as a secret before
anything can log it, so every log leaving the machine is scrubbed.
"""

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
    # Find the `if self.args.studio_password:` branch and assert it RETURNS
    # rather than falling through into the bootstrap path below it.
    branch = next(
        node
        for node in func.body
        if isinstance(node, ast.If) and "studio_password" in ast.dump(node.test)
    )
    # The LAST statement of the branch, unconditionally, and not merely "a
    # Return somewhere inside". `if not failures: return ...` contains a Return
    # and still falls through on the failure path -- which is the single case
    # this guard exists for. That mutation survived the first version of this
    # assertion, so the rule is now about the branch always returning.
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
