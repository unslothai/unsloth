# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Every Studio tab's backing endpoint, checked against the routes that exist.

The rest of this payload touches inference, training and export. A tab whose
router raised on import is invisible to all of it: the route is simply never
mounted, and the tab renders an error the moment a user opens it.

This is a smoke and says so. What makes it more than a list of strings is the
rule below that every path in `TAB_ENDPOINTS` resolves to a route actually
declared in `studio/backend/routes`. A typo'd path 404s exactly like a missing
router, so without that rule the payload could go red for the wrong reason --
or, worse, a path could be quietly "fixed" to something that always answers.

One endpoint choice is load-bearing: `/api/data-recipe/jobs/current` 404s by
design when no job is running, so using it would be red on correct behaviour.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = ROOT / "tests" / "kaggle" / "studio_gpu" / "run_studio_gpu.py"
SRC = PAYLOAD.read_text(encoding = "utf-8")
ROUTES = ROOT / "studio" / "backend" / "routes"
MAIN = ROOT / "studio" / "backend" / "main.py"


def _cls() -> ast.ClassDef:
    return next(n for n in ast.walk(ast.parse(SRC)) if isinstance(n, ast.ClassDef))


def _endpoints() -> list[tuple[str, str]]:
    for node in _cls().body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "TAB_ENDPOINTS" for t in node.targets
        ):
            return [
                (e.elts[0].value, e.elts[1].value)
                for e in node.value.elts
                if isinstance(e, ast.Tuple)
            ]
    raise AssertionError("TAB_ENDPOINTS is gone")


def _func(name: str) -> ast.FunctionDef:
    for node in _cls().body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no method named {name!r}")


def _body(name: str = "assert_tabs") -> str:
    return ast.get_source_segment(SRC, _func(name)) or ""


def _mounted_prefixes() -> set[str]:
    return set(
        re.findall(r'include_router\([^)]*prefix\s*=\s*"([^"]+)"', MAIN.read_text(encoding = "utf-8"))
    )


def _declared_get_paths() -> set[str]:
    paths: set[str] = set()
    for path in ROUTES.rglob("*.py"):
        paths.update(re.findall(r'@\w*router\.get\(\s*"([^"]+)"', path.read_text(encoding = "utf-8")))
    return paths


def test_the_assertion_exists_and_is_driven_from_the_run():
    assert _body()
    assert "self.assert_tabs()" in _body("execute")


def test_every_endpoint_resolves_to_a_route_that_exists():
    """The rule that turns a list of strings into a check. A typo'd path 404s
    exactly like a missing router, so the payload would go red naming a tab
    that is perfectly fine."""
    prefixes = _mounted_prefixes()
    declared = _declared_get_paths()
    unresolved = []
    for name, path in _endpoints():
        for prefix in prefixes:
            if path.startswith(prefix) and (path[len(prefix) :] or "/") in declared:
                break
        else:
            unresolved.append(f"{name}: {path}")
    assert unresolved == [], f"these paths match no declared GET route: {unresolved}"


def test_the_four_tabs_the_directive_names_are_all_covered():
    names = {name for name, _ in _endpoints()}
    for wanted in ("data_designer", "image_creation", "video_creation", "image_training"):
        assert any(n.startswith(wanted) for n in names), f"{wanted} is not covered"


def test_it_does_not_use_an_endpoint_that_404s_by_design():
    """`/jobs/current` raises 404 when no job is running. Using it would make
    this red on correct behaviour, which is how a check gets switched off."""
    assert all(path != "/api/data-recipe/jobs/current" for _, path in _endpoints())


def test_a_404_is_distinguished_from_other_failures():
    """A 500 is a live route with a broken handler; a 404 is a route that does
    not exist. They send a reader to different places."""
    func = _func("assert_tabs")
    tests = [ast.unparse(n.test) for n in ast.walk(func) if isinstance(n, ast.If)]
    assert any(t == "code == 404" for t in tests)
    assert any("code >= 400" in t for t in tests)


def test_a_200_that_is_not_a_json_object_fails():
    """Most of these are declared with a response_model, so a bare string or a
    null means something is standing in for the real handler."""
    func = _func("assert_tabs")
    assert any(
        "isinstance(body, (dict, list))" in ast.unparse(n.test)
        for n in ast.walk(func)
        if isinstance(n, ast.If)
    )


def test_an_endpoint_that_raises_is_a_failure_rather_than_a_skip():
    func = _func("assert_tabs")
    handlers = [n for n in ast.walk(func) if isinstance(n, ast.ExceptHandler)]
    assert handlers, "a transport error must not end the loop silently"
    joined = "\n".join(ast.unparse(h) for h in handlers)
    assert "failures.append" in joined


def test_it_runs_before_the_long_training_phase():
    """These need only a logged-in session. Behind a 20-minute training run, a
    training failure hides whether the tabs exist at all."""
    body = _body("execute")
    assert body.index("self.assert_tabs()") < body.index("self.assert_training()")
