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
_FRONTEND = _SCENE.parents[3] / "studio" / "frontend" / "src"
_ATTACHMENT_TSX = _FRONTEND / "components" / "assistant-ui" / "attachment.tsx"
_SHARED_COMPOSER_TSX = _FRONTEND / "features" / "chat" / "shared-composer.tsx"

#: Which file renders which composer's attachments. The pairs are positional in actions.py -- the
#: first container goes with the first tile -- so they are checked as pairs here too.
_RENDERED_BY = (_ATTACHMENT_TSX, _SHARED_COMPOSER_TSX)

#: A class selector and nothing else: the coupling check below can only speak about bare class
#: names, so a selector that grew a descendant combinator or an attribute must not slip past it
#: wearing the same constant name.
_BARE_CLASS = re.compile(r"^\.([A-Za-z][\w-]*)$")
#: `[data-thing]`, the handle shape used where a class would be a utility class and drift.
_BARE_ATTRIBUTE = re.compile(r"^\[([A-Za-z][\w-]*)\]$")


def _shipped_constant(name: str) -> tuple[str, ...]:
    """Return the tuple of selectors assigned to `name` at module level in actions.py.

    Read from source, not imported: importing the scene package drags in Playwright and the whole
    action registry, and the point here is to check the literal that ships.
    """
    tree = ast.parse(_ACTIONS.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                value = ast.literal_eval(node.value)
                assert isinstance(value, tuple), f"{name} is no longer a literal tuple"
                assert all(isinstance(item, str) for item in value), f"{name} holds non-strings"
                return value
    raise AssertionError(f"{name} is not assigned at module level in {_ACTIONS}")


def _tokens(text: str) -> set[str]:
    """Every identifier-shaped token in the file: how both a className and an attribute are written."""
    return set(re.findall(r"[A-Za-z][\w-]*", text))


@pytest.mark.parametrize("index", range(len(_RENDERED_BY)))
@pytest.mark.parametrize(
    "constant", ["_COMPOSER_ATTACHMENT_CONTAINERS", "_COMPOSER_ATTACHMENT_TILES"]
)
def test_each_counted_handle_is_one_its_composer_renders(constant, index):
    selectors = _shipped_constant(constant)
    assert len(selectors) == len(_RENDERED_BY), (
        f"{constant} lists {len(selectors)} selectors but {len(_RENDERED_BY)} files are checked. "
        "A composer was added or removed without updating this test, so one of them is now "
        "counted by nobody."
    )
    selector = selectors[index]
    source = _RENDERED_BY[index]
    assert source.is_file(), f"{source} moved; update this test with it"
    text = source.read_text(encoding="utf-8")

    klass = _BARE_CLASS.match(selector)
    attribute = _BARE_ATTRIBUTE.match(selector)
    assert klass or attribute, (
        f"{constant}[{index}] is {selector!r}, which is neither a bare class nor a bare attribute, "
        "so this test can no longer tell whether the frontend renders it. Either keep it simple or "
        "extend the coverage check to the new shape."
    )
    handle = (klass or attribute).group(1)
    assert handle in _tokens(text), (
        f"{constant}[{index}] counts {selector}, but {source.name} never renders {handle}. "
        "image_upload would count zero before and zero after on that composer and report the "
        "upload as broken, whatever it actually did."
    )


def _component_body(text: str, name: str) -> str:
    """The source of a top-level `function name(...)`, up to the closing brace in column 0."""
    start = text.index(f"function {name}(")
    rest = text[start:]
    end = rest.index("\n}\n")
    return rest[:end]


def test_the_compare_composer_tags_its_image_thumb_and_not_only_its_audio_chip():
    """A file-wide check cannot see one of two identical handles disappear.

    The compare composer tags both its image thumb and its audio chip with the same attribute, so
    dropping it from the thumb -- the one change that would silently stop image_upload counting on
    the compare screen -- leaves the token in the file and keeps the coverage check above green.
    Verified: that edit passes the coverage check and fails only this one. So the look is scoped to
    the thumb's own component.
    """
    attribute = _shipped_constant("_COMPOSER_ATTACHMENT_TILES")[1]
    match = _BARE_ATTRIBUTE.match(attribute)
    assert (
        match is not None
    ), f"the compare composer's tile selector {attribute!r} is not an attribute"
    handle = match.group(1)
    body = _component_body(_SHARED_COMPOSER_TSX.read_text(encoding="utf-8"), "PendingImageThumb")
    assert handle in _tokens(body), (
        f"PendingImageThumb no longer carries {handle}, so an image attached on the compare screen "
        "is counted by nothing and image_upload reports a working upload as broken."
    )


def test_the_counting_query_scopes_every_tile_to_its_own_container():
    """The constants being right does not help if the query does not use them.

    Each tile must be scoped to the container of the same composer. An unscoped tile would also
    count the attachments on already-sent messages, and a cross-paired one would count nothing.
    """
    containers = _shipped_constant("_COMPOSER_ATTACHMENT_CONTAINERS")
    tiles = _shipped_constant("_COMPOSER_ATTACHMENT_TILES")
    query = _shipped_constant_expression("_COUNT_COMPOSER_ATTACHMENTS_JS")
    for container, tile in zip(containers, tiles):
        assert (
            f"{container} {tile}" in query
        ), f"the attachment count does not scope {tile} to {container}. Built query: {query!r}"


def test_every_composer_container_is_probed_on_failure():
    """The diagnostic that separates a stale selector from a failed upload must cover both."""
    containers = _shipped_constant("_COMPOSER_ATTACHMENT_CONTAINERS")
    probe = _shipped_constant_expression("_COUNT_COMPOSER_ATTACHMENT_CONTAINERS_JS")
    for container in containers:
        assert container in probe, (
            f"{container} is not probed, so a failure on that composer cannot say whether the "
            f"markup moved. Built probe: {probe!r}"
        )


def test_the_dead_selector_is_not_reintroduced():
    """The exact string that could never match, pinned so it cannot come back by copy-paste."""
    source = _ACTIONS.read_text(encoding="utf-8")
    for dead in (".aui-composer-attachment,", 'data-slot="composer-attachment"'):
        assert dead not in source, (
            f"{dead!r} is back in {_ACTIONS.name}. The frontend has never rendered it; counting it "
            "makes image_upload fail regardless of whether the upload worked."
        )


def _shipped_constant_expression(name: str) -> str:
    """Evaluate the expression that builds `name`, using the other shipped constants.

    `ast.literal_eval` cannot fold an f-string or a join, so the module's own constants are bound in
    order and the expression is evaluated against them plus `zip`, and nothing else.
    """
    tree = ast.parse(_ACTIONS.read_text(encoding="utf-8"))
    bound: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            value = eval(  # noqa: S307 - this repo's own source, bound to its own constants
                compile(ast.Expression(node.value), str(_ACTIONS), "eval"),
                {"__builtins__": {"zip": zip}},
                dict(bound),
            )
        except Exception:  # noqa: BLE001 - anything that is not a selector constant is skipped
            continue
        if isinstance(value, (str, tuple)):
            bound[target.id] = value
    assert name in bound, f"{name} is not a module-level expression in {_ACTIONS}"
    built = bound[name]
    assert isinstance(built, str), f"{name} is {type(built).__name__}, expected a string"
    return built
