"""Tests for scripts/enforce_kwargs_spacing.py rewrite rules (AST-preserving, idempotent)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_SCRIPTS = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from enforce_kwargs_spacing import (  # noqa: E402
    collapse_short_asserts,
    enforce_spacing,
    merge_adjacent_string_literals,
    normalize_def_trailing_comma,
    remove_blank_after_short_import,
)


# (name, source) pairs where the blank after the import block MUST be removed.
_MUST_CHANGE = {
    "try_except_import": (
        "def f():\n"
        "    try:\n"
        "        import torch\n"
        "\n"
        "        return torch.inference_mode\n"
        "    except Exception:\n"
        "        from contextlib import nullcontext\n"
        "\n"
        "        return nullcontext\n"
    ),
    "if_from_import": (
        "def g():\n"
        "    if cond:\n"
        "        from . import locators\n"
        "\n"
        "        regions = locators.regions()\n"
    ),
    "multiple_consecutive_imports": (
        "def f():\n    import a\n    import b\n\n    return a, b\n"
    ),
    "type_checking_block": (
        "def f():\n"
        "    if TYPE_CHECKING:\n"
        "        import x\n"
        "\n"
        "        y = x\n"
        "        return y\n"
    ),
    "with_block": (
        "def f():\n    with ctx():\n        import a\n\n        return a.run()\n"
    ),
}

# Sources that MUST be left byte-for-byte unchanged.
_MUST_NOT_CHANGE = {
    "module_level": 'import os\n\nVALUE = os.environ.get("V")\n',
    "large_suite": (
        "def f():\n"
        "    import a\n"
        "\n"
        "    x = a.load()\n"
        "    y = transform(x)\n"
        "    return y\n"
    ),
    "comment_between": (
        "def f():\n    import a\n\n    # keep separated\n    return a.value\n"
    ),
    "import_is_last_stmt": "def f():\n    if cond:\n        import a\n\n",
    "no_blank_already": "def f():\n    import a\n    return a\n",
}


@pytest.mark.parametrize("name", sorted(_MUST_CHANGE))
def test_blank_removed_for_small_import_block(name):
    src = _MUST_CHANGE[name]
    out, changed = remove_blank_after_short_import(src)
    assert changed is True
    assert out != src
    # Import and following statement now adjacent.
    assert "\n\n" not in out or out.count("\n\n") < src.count("\n\n")
    assert ast.dump(ast.parse(out)) == ast.dump(ast.parse(src))
    out2, changed2 = remove_blank_after_short_import(out)
    assert out2 == out and changed2 is False


@pytest.mark.parametrize("name", sorted(_MUST_NOT_CHANGE))
def test_blank_preserved_when_not_applicable(name):
    src = _MUST_NOT_CHANGE[name]
    out, changed = remove_blank_after_short_import(src)
    assert changed is False
    assert out == src


def test_exact_output_try_block():
    src = (
        "def f():\n"
        "    try:\n"
        "        import torch\n"
        "\n"
        "        return torch.inference_mode\n"
        "    except Exception:\n"
        "        from contextlib import nullcontext\n"
        "\n"
        "        return nullcontext\n"
    )
    expected = (
        "def f():\n"
        "    try:\n"
        "        import torch\n"
        "        return torch.inference_mode\n"
        "    except Exception:\n"
        "        from contextlib import nullcontext\n"
        "        return nullcontext\n"
    )
    out, changed = remove_blank_after_short_import(src)
    assert changed is True
    assert out == expected


def test_exact_output_multiple_consecutive_imports():
    # Only the blank after the LAST import in a run is dropped; both imports kept.
    src = "def f():\n    import a\n    import b\n\n    return a, b\n"
    expected = "def f():\n    import a\n    import b\n    return a, b\n"
    out, changed = remove_blank_after_short_import(src)
    assert changed is True
    assert out == expected


def test_multiple_blank_lines_in_gap_all_removed():
    src = "def f():\n    import a\n\n\n    return a\n"
    expected = "def f():\n    import a\n    return a\n"
    out, changed = remove_blank_after_short_import(src)
    assert changed is True
    assert out == expected
    out2, changed2 = remove_blank_after_short_import(out)
    assert out2 == out and changed2 is False


def test_multiline_import_internal_blank_preserved():
    # A blank inside a parenthesized import is part of the import, not the gap.
    src = (
        "def g():\n"
        "    from mod import (\n"
        "        a,\n"
        "\n"
        "        b,\n"
        "    )\n"
        "\n"
        "    return a, b\n"
    )
    expected = (
        "def g():\n"
        "    from mod import (\n"
        "        a,\n"
        "\n"
        "        b,\n"
        "    )\n"
        "    return a, b\n"
    )
    out, changed = remove_blank_after_short_import(src)
    assert changed is True
    assert out == expected
    assert ast.dump(ast.parse(out)) == ast.dump(ast.parse(src))
    out2, changed2 = remove_blank_after_short_import(out)
    assert out2 == out and changed2 is False


def test_syntax_error_is_left_alone():
    src = "def f(:\n    import a\n\n    return a\n"
    out, changed = remove_blank_after_short_import(src)
    assert changed is False
    assert out == src


def test_enforce_spacing_pads_kwargs():
    src = "f(a=1, b = 2)\n"
    out, changed = enforce_spacing(src)
    assert changed is True
    assert "a = 1" in out and "b = 2" in out


def test_enforce_spacing_noop_when_already_spaced():
    src = "f(a = 1, b = 2)\n"
    out, changed = enforce_spacing(src)
    assert changed is False
    assert out == src


# ── Rule D: def one-per-line iff >= 3 params AND a default ──────────────────
# add comma -> force one-per-line; strip comma -> stay collapsible.

# Comma must be ADDED: >= 3 params, has a default, no trailing comma yet.
_DEF_ADD = {
    "three_with_default": "def f(a, b, c=1):\n    return a\n",
    "four_with_default": "def f(a, b, c, d=1):\n    return a\n",
    "kwonly_default": "def f(a, b, *, c=1):\n    return a\n",  # 3 real params, kw default
    "continuation_default": "def f(\n    a, b, c=1\n):\n    return a\n",
    "starred_with_default": "def f(a, b, *args, c=1):\n    return a\n",  # 4 params
}

# Comma must be STRIPPED: NOT (>=3 params and default), but a trailing comma exists.
_DEF_STRIP = {
    "three_no_default_multiline": "def f(\n    a,\n    b,\n    c,\n):\n    return a\n",
    "four_no_default_multiline": "def f(\n    a,\n    b,\n    c,\n    d,\n):\n    return a\n",
    "two_with_default": "def f(\n    a,\n    b=1,\n):\n    return a\n",  # < 3 params -> one line
    "single_arg": "def f(\n    a,\n):\n    return a\n",
}

# Left byte-for-byte unchanged.
_DEF_NOCHANGE = {
    "three_no_default_oneline": "def f(a, b, c):\n    return a\n",
    "two_with_default_oneline": "def f(a, b=1):\n    return a\n",  # < 3 -> one line, no comma
    "noparams": "def f():\n    return 1\n",
    "call_site": "x = foo(\n    a,\n    b,\n    c,\n    d,\n)\n",
    "nested_default_call": "def f(a=g(1, 2,)):\n    return a\n",  # 1 param, no def comma
    "three_default_already_comma": "def f(\n    a,\n    b,\n    c=1,\n):\n    return a\n",
}


@pytest.mark.parametrize("name", sorted(_DEF_ADD))
def test_def_comma_added(name):
    src = _DEF_ADD[name]
    out, changed = normalize_def_trailing_comma(src)
    assert changed is True
    assert ast.dump(ast.parse(out)) == ast.dump(ast.parse(src))
    assert out.count(",") == src.count(",") + 1
    out2, changed2 = normalize_def_trailing_comma(out)
    assert out2 == out and changed2 is False


@pytest.mark.parametrize("name", sorted(_DEF_STRIP))
def test_def_comma_stripped(name):
    src = _DEF_STRIP[name]
    out, changed = normalize_def_trailing_comma(src)
    assert changed is True
    assert ast.dump(ast.parse(out)) == ast.dump(ast.parse(src))
    assert out.count(",") == src.count(",") - 1
    out2, changed2 = normalize_def_trailing_comma(out)
    assert out2 == out and changed2 is False


@pytest.mark.parametrize("name", sorted(_DEF_NOCHANGE))
def test_def_comma_unchanged(name):
    src = _DEF_NOCHANGE[name]
    out, changed = normalize_def_trailing_comma(src)
    assert changed is False
    assert out == src


def test_def_comma_exact_output_strip_and_add():
    # >= 3 params + default -> add comma (force one-per-line)
    assert normalize_def_trailing_comma("def f(a, b, c=1):\n    return a\n")[0] == (
        "def f(a, b, c=1,):\n    return a\n"
    )
    # 3 params, no default -> strip comma (collapsible)
    assert (
        normalize_def_trailing_comma(
            "def f(\n    a,\n    b,\n    c,\n):\n    return a\n"
        )[0]
        == "def f(\n    a,\n    b,\n    c\n):\n    return a\n"
    )


# ── Rule C: merge adjacent same-line string literals ───────────────────────


@pytest.mark.parametrize(
    "src,expected",
    [
        ('x = "ab" "cd"\n', 'x = "abcd"\n'),
        ('d = "newly-" "added dep."\n', 'd = "newly-added dep."\n'),
        ('m = ("a. " "b.")\n', 'm = ("a. b.")\n'),
        ('x = r"a\\n" r"b"\n', 'x = r"a\\nb"\n'),
        ('x = "a\\"q" "b"\n', 'x = "a\\"qb"\n'),
        # f + plain folds into one f-string (plain braces escaped).
        ('x = f"a" "b"\n', 'x = f"ab"\n'),
        (
            'd = (f"{pkg}@{ver} is on the " "BLOCKED list")\n',
            'd = (f"{pkg}@{ver} is on the BLOCKED list")\n',
        ),
        ('x = f"a{z}" "{lit}"\n', 'x = f"a{z}{{lit}}"\n'),
        ('m = "plain " f"then {y}"\n', 'm = f"plain then {y}"\n'),  # plain + f
    ],
)
def test_merge_adjacent_strings(src, expected):
    out, changed = merge_adjacent_string_literals(src)
    assert changed is True
    assert out == expected
    assert ast.dump(ast.parse(out)) == ast.dump(ast.parse(src))
    out2, changed2 = merge_adjacent_string_literals(out)
    assert out2 == out and changed2 is False


@pytest.mark.parametrize(
    "src",
    [
        'x = "ab"\n',  # single literal
        "x = \"ab\" 'cd'\n",  # mixed quote style
        'x = b"a" b"b"\n',  # bytes: left side-by-side by request
        'x = rb"a" rb"b"\n',  # raw-bytes: also left alone
        'm = f"a {x} " f"after {y}"\n',  # pure f + f: left side-by-side
        'x = rf"a{z}" "b"\n',  # raw f-string: brace/backslash too subtle -> skip
        'x = f"a{z}" "\\N{BULLET}"\n',  # named escape: AST guard rejects the fold
        'x = (\n    "a"\n    "b"\n)\n',  # different lines, not merged
    ],
)
def test_merge_adjacent_strings_skips(src):
    out, changed = merge_adjacent_string_literals(src)
    assert changed is False
    assert out == src


def test_fstring_fold_skipped_when_statement_would_not_collapse():
    # Folding a long f + plain assert message can't fit on one line, so leave it.
    src = (
        "def f():\n"
        "    assert some_condition_holds_here, (\n"
        '        f"a fairly detailed message about {value} explaining " "why this failed badly"\n'
        "    )\n"
    )
    out, changed = merge_adjacent_string_literals(src)
    assert changed is False
    assert out == src


def test_fstring_fold_applied_when_statement_collapses():
    # A multi-line f + plain that fits on one line after folding is folded.
    src = (
        "def f():\n    raise ValueError(\n"
        '        f"bad {x}: " "try again"\n'
        "    )\n"
    )
    out, changed = merge_adjacent_string_literals(src)
    assert changed is True
    assert 'f"bad {x}: try again"' in out
    assert ast.dump(ast.parse(out)) == ast.dump(ast.parse(src))


def test_fstring_fold_applied_inside_large_multiline_call():
    # The fit guard only restricts asserts; an f + plain arg in a big call folds.
    src = (
        "findings.append(\n"
        "    Finding(\n"
        "        path=str(path),\n"
        "        package=key,\n"
        '        detail=(f"{name}@{ver} is on the " "BLOCKED list"),\n'
        "    )\n"
        ")\n"
    )
    out, changed = merge_adjacent_string_literals(src)
    assert changed is True
    assert 'detail=(f"{name}@{ver} is on the BLOCKED list")' in out
    assert ast.dump(ast.parse(out)) == ast.dump(ast.parse(src))


# ── collapse_short_asserts: strip the magic comma holding a short assert open ──
# Strips the trailing comma so ruff joins the assert onto one line; AST unchanged.


@pytest.mark.parametrize(
    "name,src",
    [
        (
            "dict_eq",
            'def t():\n    assert got == {\n        "a": 1,\n        "b": 2,\n    }\n',
        ),
        (
            "list_eq",
            'def t():\n    assert xs == [\n        "a",\n        "b",\n        "c",\n    ]\n',
        ),
        (
            "membership",
            'def t():\n    assert {\n        "type": "x",\n        "name": "y",\n    } in tools\n',
        ),
        (
            "tuple_message",
            "def t():\n    assert cond, (\n        base,\n        headers,\n    )\n",
        ),
        (
            "call_args",
            "def t():\n    assert eq(\n        a,\n        b,\n    )\n",
        ),
    ],
)
def test_collapse_short_assert_strips_trailing_comma(name, src):
    out, changed = collapse_short_asserts(src)
    assert changed is True
    # Magic trailing comma is gone, so ruff joins it on the next pass.
    assert out.count(",") == src.count(",") - 1
    assert ast.dump(ast.parse(out)) == ast.dump(ast.parse(src))
    out2, changed2 = collapse_short_asserts(out)
    assert out2 == out and changed2 is False


@pytest.mark.parametrize(
    "name,src",
    [
        # one-element tuple message: stripping (only,) -> (only) changes meaning.
        ("one_tuple_message", "def t():\n    assert cond, (\n        only,\n    )\n"),
        # a comment inside keeps ruff multi-line, so collapsing would oscillate.
        (
            "comment_inside",
            'def t():\n    assert x == {\n        "a": 1,  # keep\n        "b": 2,\n    }\n',
        ),
        # genuinely long: would not fit on one line, leave expanded.
        (
            "too_long",
            "def t():\n    assert some_really_long_left_operand_name_here == {\n"
            '        "alpha": 11111111,\n        "beta": 22222222,\n'
            '        "gamma": 33333333,\n        "delta": 44444444,\n    }\n',
        ),
        # already one line: nothing to do.
        ("one_line", 'def t():\n    assert got == {"a": 1, "b": 2}\n'),
    ],
)
def test_collapse_short_assert_left_alone(name, src):
    out, changed = collapse_short_asserts(src)
    assert changed is False
    assert out == src
