# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The selectors `image_upload` counts with are the ones the composer actually renders.

`image_upload` decides it worked by counting attachment elements before and after the file is set.
A selector that matches nothing counts zero both times, which is indistinguishable from an upload
that silently did nothing, so the action can fail forever while pointing at the wrong file. That is
not hypothetical: the original selector (`.aui-composer-attachment`, plus a `data-slot` the app has
never set) matched no element in any build, and it went unnoticed because the action only mounts
once a model is selected and `--allow-not-run image_upload` excused every run until then.

So the class names are read out of the shipped source rather than retyped here, and checked against
the frontend file that renders them. A rename on either side breaks this test instead of quietly
turning the assertion into one that cannot pass.
"""

import ast
import re
from pathlib import Path

import pytest

_SCENE = Path(__file__).resolve().parents[1]
_ACTIONS = _SCENE / "actions.py"
_ATTACHMENT_TSX = (
    _SCENE.parents[3] / "studio" / "frontend" / "src" / "components" / "assistant-ui" / "attachment.tsx"
)

#: A class selector and nothing else: the coupling check below can only speak about bare class
#: names, so a selector that grew a descendant combinator or an attribute must not slip past it
#: wearing the same constant name.
_BARE_CLASS = re.compile(r"^\.([A-Za-z][\w-]*)$")


def _shipped_constant(name: str) -> str:
    """Return the string assigned to `name` at module level in actions.py.

    Read from source, not imported: importing the scene package drags in Playwright and the whole
    action registry, and the point here is to check the literal that ships.
    """
    tree = ast.parse(_ACTIONS.read_text(encoding = "utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                assert isinstance(node.value, ast.Constant), f"{name} is no longer a plain literal"
                assert isinstance(node.value.value, str), f"{name} is not a string"
                return node.value.value
    raise AssertionError(f"{name} is not assigned at module level in {_ACTIONS}")


def _class_tokens(text: str) -> set[str]:
    """Every whitespace-delimited token in the file, which is how a className is written."""
    return set(re.findall(r"[A-Za-z][\w-]*", text))


@pytest.mark.parametrize(
    "constant",
    ["_COMPOSER_ATTACHMENTS_CONTAINER", "_COMPOSER_ATTACHMENT_TILE"],
)
def test_the_counted_class_is_one_the_frontend_renders(constant):
    selector = _shipped_constant(constant)
    match = _BARE_CLASS.match(selector)
    assert match is not None, (
        f"{constant} is {selector!r}, which is not a bare class selector, so this test can no "
        "longer tell whether the frontend renders it. Either keep it a bare class or extend the "
        "coupling check to cover the new shape."
    )
    css_class = match.group(1)
    assert _ATTACHMENT_TSX.is_file(), f"{_ATTACHMENT_TSX} moved; update this test with it"
    tokens = _class_tokens(_ATTACHMENT_TSX.read_text(encoding = "utf-8"))
    assert css_class in tokens, (
        f"{constant} counts .{css_class}, but {_ATTACHMENT_TSX.name} never renders that class. "
        "image_upload would count zero before and zero after and report the upload as broken, "
        "whatever the composer actually did."
    )


def test_the_counting_query_is_built_from_both_constants():
    """The constants being right does not help if the query does not use them."""
    container = _shipped_constant("_COMPOSER_ATTACHMENTS_CONTAINER")
    tile = _shipped_constant("_COMPOSER_ATTACHMENT_TILE")
    query = _shipped_constant_expression("_COUNT_COMPOSER_ATTACHMENTS_JS")
    assert f"{container} {tile}" in query, (
        "the attachment count no longer scopes the tile to the composer's container, so it would "
        f"also count attachments on sent messages. Built query: {query!r}"
    )


def test_the_dead_selector_is_not_reintroduced():
    """The exact string that could never match, pinned so it cannot come back by copy-paste."""
    source = _ACTIONS.read_text(encoding = "utf-8")
    for dead in (".aui-composer-attachment,", 'data-slot="composer-attachment"'):
        assert dead not in source, (
            f"{dead!r} is back in {_ACTIONS.name}. The frontend has never rendered it; counting it "
            "makes image_upload fail regardless of whether the upload worked."
        )


def _shipped_constant_expression(name: str) -> str:
    """Evaluate the f-string/concat that builds `name`, using the other shipped constants.

    `ast.literal_eval` cannot fold an f-string, so the module's string constants are bound and the
    expression is evaluated against them and nothing else.
    """
    tree = ast.parse(_ACTIONS.read_text(encoding = "utf-8"))
    bound: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            value = eval(  # noqa: S307 - a literal from this repo's own source, bound to strings only
                compile(ast.Expression(node.value), str(_ACTIONS), "eval"),
                {"__builtins__": {}},
                dict(bound),
            )
        except Exception:  # noqa: BLE001 - anything non-constant is simply not a selector constant
            continue
        if isinstance(value, str):
            bound[target.id] = value
    assert name in bound, f"{name} is not a module-level string expression in {_ACTIONS}"
    return bound[name]
