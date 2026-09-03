# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The node harnesses must follow the helpers their slices call (#9490 broke this).

``sanitizeAssistantReplayText`` gained a call to ``stripSearchImageTokens``, which lives in
another module, and every harness that sliced it died with ``ReferenceError`` -- seventeen
failures on main, on every open PR, in a test file nothing had touched. Naming that one helper
in the prelude would have fixed the day and not the next one, so ``_ts_deps`` resolves
references instead. These cases pin the resolver, because a harness generator that quietly
stops following dependencies fails as a ``ReferenceError`` in an unrelated suite.
"""

from __future__ import annotations

import re
import textwrap

import pytest

from _node_harness import WORKDIR, source_path
from _ts_deps import _Module, _balanced, _blank_noise, resolve_dependencies

ADAPTER = source_path("studio/frontend/src/features/chat/api/chat-adapter.ts")
SEARCH_IMAGES = source_path("studio/frontend/src/features/chat/search-images/search-images.ts")


def _resolved(harness: str) -> str:
    return resolve_dependencies(harness, (ADAPTER,))


def test_an_imported_helper_is_followed_out_of_its_own_module():
    """The exact break: a slice calls across a module boundary and must carry what it calls."""
    resolved = _resolved("function replay(text) { return sanitizeAssistantReplayText(text); }\n")
    assert "function stripSearchImageTokens(" in resolved
    assert (
        "function findCodeBlockRegions(" in resolved
    ), "the helper's own dependency has to come too, or resolution stops one hop short"


def test_the_helper_is_carried_verbatim():
    """Sliced, not restated: a paraphrase would pass while testing a different function."""
    body = SEARCH_IMAGES.read_text(encoding = "utf-8")
    start = body.index("export function stripSearchImageTokens(")
    end = body.index("\n}", start) + 2
    verbatim = body[start:end].removeprefix("export ")
    assert verbatim in _resolved(
        "function replay(text) { return sanitizeAssistantReplayText(text); }\n"
    )


def test_a_fixture_the_harness_already_defines_is_never_replaced():
    """The preludes stub what they mean to control, and resolution must not undo that."""
    harness = textwrap.dedent(
        """
        function stripSearchImageTokens(text) { return "STUB"; }
        function replay(text) { return sanitizeAssistantReplayText(text); }
        """
    )
    resolved = _resolved(harness)
    assert resolved.count("function stripSearchImageTokens(") == 1
    assert "STUB" in resolved


def test_an_unresolvable_name_is_left_alone_rather_than_guessed_at():
    """No worse than the state before: an unknown name stays unknown, nothing is invented."""
    resolved = _resolved("function f() { return someNameNoModuleExports(); }\n")
    assert "someNameNoModuleExports" not in resolved.replace(
        "function f() { return someNameNoModuleExports(); }", ""
    )


def test_declarations_land_above_the_ones_that_read_them():
    """A `const` read before its declaration is a TDZ crash, not `undefined`."""
    resolved = _resolved("function replay(text) { return sanitizeAssistantReplayText(text); }\n")
    assert resolved.index("function findCodeBlockRegions(") < resolved.index(
        "function stripSearchImageTokens("
    )


def test_braces_inside_strings_and_regexes_do_not_end_a_declaration():
    """``stripSearchImageTokens`` carries ``[0-9a-f]{12}``; counting that as structure truncates it."""
    blanked = _blank_noise('const a = /x{1,2}/g;\nconst b = "}{";\n// }\nconst c = 1;\n')
    assert blanked.count("{") == 0 and blanked.count("}") == 0
    assert len(blanked) == len('const a = /x{1,2}/g;\nconst b = "}{";\n// }\nconst c = 1;\n')


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("function f() {\n  return `a${b}c`;\n}\n", id = "template_hole"),
        pytest.param("function f() {\n  return `${`${x}`}`;\n}\n", id = "nested_template"),
        pytest.param("function f() {\n  return /^F\\d{1,2}$/.test(k);\n}\n", id = "returned_regex"),
        pytest.param("function f() {\n  return /a'b/.test(k);\n}\n", id = "quote_in_regex"),
        pytest.param("const r = x / y / z;\nconst s = 1;\n", id = "division"),
        pytest.param('const c = { "}": 1 };\n', id = "brace_key"),
    ],
)
def test_the_scanner_leaves_every_shape_balanced(source):
    """An unbalanced scan silently truncates or over-slices, and neither says so."""
    blanked = _blank_noise(source)
    assert len(blanked) == len(source)
    assert _balanced(blanked), blanked


def test_the_whole_frontend_scans_balanced():
    """The measure that catches a scanner bug wholesale rather than one shape at a time."""
    root = source_path("studio/frontend/src")
    unbalanced = [
        path
        for path in sorted(root.rglob("*.ts"))
        if not _balanced(_blank_noise(path.read_text(encoding = "utf-8")))
    ]
    assert unbalanced == []


def test_a_declaration_is_refused_rather_than_sliced_into_its_neighbour():
    """Over-slicing carries an inner ``export`` and a half statement: node will not parse it."""
    module = _Module(source_path("studio/frontend/src/lib/resolved-precision.ts"))
    for name in module.declarations:
        text = module.declaration_text(name)
        assert text is None or _balanced(_blank_noise(text)), name
        assert text is None or "\nexport " not in text, name


def test_a_pulled_const_that_needs_a_fixture_is_dropped_not_emitted(tmp_path):
    """The fixtures sit below this block, so reading one from it is a TDZ crash."""
    module = tmp_path / "src" / "m.ts"
    module.parent.mkdir(parents = True)
    module.write_text('export const BASE = "real";\nexport const HEADERS = [BASE, "x"];\n')
    resolved = resolve_dependencies(
        'const BASE = "STUB";\nconst used = HEADERS;\n', (module,), root = tmp_path / "src"
    )
    assert "const HEADERS" not in resolved.replace("const used = HEADERS;", "")


def test_a_destructured_fixture_is_not_declared_twice(tmp_path):
    """``const { only, ...rest } = ...`` is a binding a declaration regex does not see."""
    module = tmp_path / "src" / "m.ts"
    module.parent.mkdir(parents = True)
    module.write_text("export function only() {\n  return 1;\n}\nexport const pair = { only };\n")
    resolved = resolve_dependencies(
        "const { only } = fixtures;\nconst used = only();\n", (module,), root = tmp_path / "src"
    )
    assert "function only(" not in resolved


def test_every_harness_test_asks_for_resolution():
    """A suite that slices source without passing ``sources`` is one refactor from #9490 again."""
    for path in sorted((WORKDIR / "tests" / "studio").glob("test_*.py")):
        text = path.read_text(encoding = "utf-8")
        for call in re.findall(r"run_harness\((?:[^()]|\([^()]*\))*\)", text):
            if "_harness_source()" not in call:
                continue  # No slices to follow: this one builds its script inline.
            assert "sources =" in call, f"{path.name}: {call}"
