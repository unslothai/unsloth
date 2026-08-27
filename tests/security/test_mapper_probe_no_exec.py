# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`_get_new_mapper` used to `exec` an HTTPS response body.

When `from_pretrained` is handed a model name the installed `mapper.py` does not know,
`loader_utils` fetches mapper.py from raw.githubusercontent.com and used to run
`exec(response.text, namespace)`. That is arbitrary code execution inside an ordinary
model load, gated only on transport integrity: nothing checked that the body was the
file it claimed to be, and `exec` into a fresh dict still gets full builtins.

Only one thing in that file is data. The probe now `ast.literal_eval`s the
`__INT_TO_FLOAT_MAPPER` dict literal out of the fetched text and derives the five
tables with `mapper.build_mappers`, the installed version's own code. A hostile body
can therefore change what the probe *reports*, which was always true, but can no
longer run.

CPU-only and network-free: `requests.get` is stubbed, and `tests/security/conftest.py`
blocks non-loopback sockets anyway.
"""

import builtins
import pathlib
import re
import types
from contextlib import contextmanager

import pytest

from unsloth.models import loader_utils
from unsloth.models.mapper import build_mappers


REAL_MAPPER = open(
    __import__("pathlib").Path(loader_utils.__file__).with_name("mapper.py"),
    encoding = "utf-8",
).read()


class _Response:
    """The streaming half of `requests.Response`, which is all the probe uses.

    The probe reads in chunks and stops at its byte cap while reading, because
    `requests.get` would otherwise buffer and decode the whole body before any length
    check could run. It also follows redirects itself, since `requests` drains each
    intermediate body inside `get`, so a status and headers are needed as well.
    A fake offering only `.text` would let either of those regress silently.
    """

    def __init__(
        self,
        text,
        status_code = 200,
        headers = None,
    ):
        self.encoding = "utf-8"
        self.status_code = status_code
        self.headers = headers or {}
        self._body = text.encode("utf-8")
        self._read = False

    def iter_content(self, chunk_size = 1):
        yield self._body

    @property
    def raw(self):
        """What the probe actually reads through.

        It calls `read1`, which returns whatever ONE socket read produced, so a
        deadline check sits between every read rather than only between whole chunks.
        """
        return self

    def read1(self, amount = -1):
        if self._read:
            return b""
        self._read = True
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        return False


@contextmanager
def _serving(body, monkeypatch):
    """Makes the probe's `requests.get` return `body`."""
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Response(body))
    if not hasattr(requests, "compat"):
        monkeypatch.setattr(requests, "compat", types.SimpleNamespace(urljoin = lambda b, u: u))
    yield


@pytest.fixture
def no_dynamic_execution(monkeypatch):
    """Records any exec/eval reached during the probe.

    It RECORDS rather than raises. `_get_new_mapper` catches every exception, so an
    AssertionError raised here is swallowed and the probe returns its five empty
    tables - which then satisfies a `len(result) == 5` assertion just as well as the
    fixed implementation does. A tripwire the caller can eat is not a tripwire, so the
    test asserts on this list instead.
    """
    calls = []

    def _forbidden(name):
        def _record(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"_get_new_mapper called {name}()")

        return _record

    monkeypatch.setattr(builtins, "exec", _forbidden("exec"))
    monkeypatch.setattr(builtins, "eval", _forbidden("eval"))
    return calls


# --- the payload cannot run --------------------------------------------------

MARKER = "/tmp/unsloth_mapper_probe_marker"

PAYLOADS = [
    f"import os; os.system('touch {MARKER}')\n__INT_TO_FLOAT_MAPPER = {{}}\n",
    f"__INT_TO_FLOAT_MAPPER = {{}}\nimport os\nos.system('touch {MARKER}')\n",
    f"open({MARKER!r}, 'w').close()\n__INT_TO_FLOAT_MAPPER = {{}}\n",
    "__INT_TO_FLOAT_MAPPER = __import__('os').environ\n",
    "class X:\n    def __init__(self): __import__('os').getpid()\n__INT_TO_FLOAT_MAPPER = X()\n",
]


@pytest.mark.parametrize("payload", PAYLOADS)
def test_payload_in_the_response_is_never_executed(payload, monkeypatch, tmp_path):
    import os

    if os.path.exists(MARKER):
        os.remove(MARKER)

    with _serving(payload, monkeypatch):
        result = loader_utils._get_new_mapper()

    assert not os.path.exists(MARKER), "the fetched body executed"
    # Whatever it returns, it must be the five-table shape and carry nothing useful.
    assert len(result) == 5
    assert all(isinstance(table, dict) for table in result)


def test_probe_does_not_call_exec_or_eval(no_dynamic_execution, monkeypatch):
    """Stronger than the marker: the builtins are not reached at all.

    The assertion is on the recorded calls. Asserting only the returned shape would
    also hold for the pre-change implementation, since the AssertionError the fixture
    raises is caught by the probe's own bare except.
    """
    with _serving(REAL_MAPPER, monkeypatch):
        result = loader_utils._get_new_mapper()
    assert no_dynamic_execution == [], f"the probe reached {no_dynamic_execution}"
    # And it still did the work, rather than falling into the except and returning
    # empties, which is the other way this test could pass for the wrong reason.
    assert len(result) == 5
    assert all(result[:3]), "the probe returned nothing, so it proved nothing"


def test_a_body_that_is_not_python_is_survivable(monkeypatch):
    for body in ("", "<html>404</html>", "\x00\x01\x02", "def "):
        with _serving(body, monkeypatch):
            assert loader_utils._get_new_mapper() == ({}, {}, {}, {}, {})


def test_a_body_without_the_source_table_returns_nothing(monkeypatch):
    with _serving("SOMETHING_ELSE = {'a': 1}\n", monkeypatch):
        assert loader_utils._get_new_mapper() == ({}, {}, {}, {}, {})


def test_a_deeply_nested_body_is_survivable(monkeypatch):
    """literal_eval evaluates literals only, but it is not DoS-safe.

    ast.parse builds the whole tree before any literal-only check runs, so nesting past
    the compiler's recursion limit raises RecursionError, which is not a ValueError.
    The probe's bare except catches it; this pins that it stays caught.
    """
    body = "__INT_TO_FLOAT_MAPPER = " + "[" * 20000 + "]" * 20000
    with _serving(body, monkeypatch):
        assert loader_utils._get_new_mapper() == ({}, {}, {}, {}, {})


def _byte_cap():
    """The cap as the probe actually spells it, so lowering it cannot desync this."""
    source = pathlib.Path(loader_utils.__file__).read_text(encoding = "utf-8")
    match = re.search(r"^\s*byte_cap = ([0-9_]+)$", source, re.MULTILINE)
    assert match is not None, "the mapper probe no longer has a byte cap"
    return int(match.group(1).replace("_", ""))


def test_an_oversized_body_is_not_parsed_at_all(monkeypatch):
    """The size cap, which bounds parse cost rather than correctness.

    `requests`' timeout is per-read, not total, so a body can be arbitrarily large. The
    real mapper.py is around 50KB.
    """
    cap = _byte_cap()
    body = "__INT_TO_FLOAT_MAPPER = {'a' : ('b',)}\n" + ("# padding\n" * (cap // 10 + 1))
    assert len(body) > cap
    with _serving(body, monkeypatch):
        assert loader_utils._get_new_mapper() == ({}, {}, {}, {}, {})


def test_the_cap_bounds_parse_cost_not_just_download_size():
    """Short statements are the dense case: each one is several AST nodes.

    A body made of them stays well inside a generous cap while parsing to millions of
    nodes, which is why the cap is set from what the parse can afford rather than from
    what the network can afford.
    """
    cap = _byte_cap()
    assert cap <= 2_000_000, (
        f"a {cap} byte cap admits roughly {cap // 4} short statements, which is a "
        f"memory blow-up in ast.parse rather than a bounded read"
    )


def test_the_real_mapper_is_far_below_the_cap():
    assert len(REAL_MAPPER) < _byte_cap() / 2


# --- the probe still works ---------------------------------------------------


def test_real_mapper_body_reproduces_the_installed_tables(monkeypatch):
    """Serving the installed mapper.py back must reproduce the installed tables."""
    with _serving(REAL_MAPPER, monkeypatch):
        result = loader_utils._get_new_mapper()

    from unsloth.models import mapper

    expected = (
        mapper.INT_TO_FLOAT_MAPPER,
        mapper.FLOAT_TO_INT_MAPPER,
        mapper.MAP_TO_UNSLOTH_16bit,
        mapper.FLOAT_TO_FP8_BLOCK_MAPPER,
        mapper.FLOAT_TO_FP8_ROW_MAPPER,
    )
    assert result == expected
    assert len(result[0]) > 100, "the probe returned an empty table"


def test_a_newer_table_is_picked_up(monkeypatch):
    """The probe's actual job: report a mapping the installed tables do not have."""
    body = (
        "__INT_TO_FLOAT_MAPPER = {\n"
        "    'unsloth/some-new-model-bnb-4bit': ('unsloth/some-new-model',),\n"
        "}\n"
    )
    with _serving(body, monkeypatch):
        int_to_float, float_to_int, _, _, _ = loader_utils._get_new_mapper()

    assert int_to_float["unsloth/some-new-model-bnb-4bit"] == "unsloth/some-new-model"
    assert float_to_int["unsloth/some-new-model"] == "unsloth/some-new-model-bnb-4bit"


def test_an_annotated_source_table_is_still_read(monkeypatch):
    """A type annotation upstream must not turn the probe off.

    This function exists to understand NEWER mapper.py revisions, so a harmless
    `__INT_TO_FLOAT_MAPPER: dict = {...}` refactor on main would otherwise return five
    empty tables for every newly mapped model, with nothing looking broken.
    """
    plain = REAL_MAPPER
    annotated = plain.replace(
        "__INT_TO_FLOAT_MAPPER = ",
        "__INT_TO_FLOAT_MAPPER: dict = ",
        1,
    )
    assert annotated != plain, "the table is no longer spelled as a plain assignment"

    with _serving(plain, monkeypatch):
        expected = loader_utils._get_new_mapper()
    with _serving(annotated, monkeypatch):
        got = loader_utils._get_new_mapper()

    assert any(expected), "the unannotated form produced nothing, so this proves nothing"
    assert got == expected


def test_a_mutation_that_never_runs_is_not_read(monkeypatch):
    """The probe reads what the fetched module INSTALLS, not what it contains.

    `ast.walk` reaches inside function bodies and into branches that never execute. A
    mapping fabricated from one of those is worse than a missing mapping: the caller
    treats a hit as "this model is mapped in a newer unsloth" and raises the
    upgrade-only NotImplementedError for a name the newer file never installs.
    """
    dead = REAL_MAPPER + (
        "\n"
        "if False:\n"
        "    _add_with_lower(MAP_TO_UNSLOTH_16bit, 'vendor/never', 'unsloth/never')\n"
        "    FLOAT_TO_FP8_ROW_MAPPER['vendor/never-fp8'] = 'unsloth/never-fp8-row'\n"
        "\n"
        "def _helper():\n"
        "    FLOAT_TO_FP8_ROW_MAPPER['vendor/inside-a-def'] = 'unsloth/inside-a-def'\n"
    )

    with _serving(REAL_MAPPER, monkeypatch):
        expected = loader_utils._get_new_mapper()
    with _serving(dead, monkeypatch):
        got = loader_utils._get_new_mapper()

    assert any(expected), "the real mapper produced nothing, so this proves nothing"
    assert got == expected, "a mutation that never executes was read as installed"

    for table in got:
        assert not any("never" in str(key) for key in table)
        assert not any("inside-a-def" in str(key) for key in table)


def test_a_deferred_lambda_body_is_not_read(monkeypatch):
    """A `lambda` is an expression, so it seeded the expression walk.

    The walk already refuses to descend INTO a lambda it meets as a child, but the
    lambda attached to an assignment was the seed itself, and its body was read as if
    it ran at import. Same argument as the `def` above: nothing in it executes.
    """
    dead = REAL_MAPPER + (
        "\n"
        "unused = lambda: FLOAT_TO_FP8_ROW_MAPPER.update("
        "{'vendor/in-a-lambda': 'unsloth/in-a-lambda'})\n"
        "also_unused = [lambda: FLOAT_TO_FP8_ROW_MAPPER.update("
        "{'vendor/in-a-listed-lambda': 'unsloth/in-a-listed-lambda'})]\n"
    )

    with _serving(REAL_MAPPER, monkeypatch):
        expected = loader_utils._get_new_mapper()
    with _serving(dead, monkeypatch):
        got = loader_utils._get_new_mapper()

    assert any(expected), "the real mapper produced nothing, so this proves nothing"
    assert got == expected, "a lambda body was read as installed"
    for table in got:
        assert not any("lambda" in str(key) for key in table)


def test_an_alias_added_inside_the_builder_is_read(monkeypatch):
    """`build_mappers` is called at import, so its body runs.

    The aliases that cannot be derived from the source table live in there now, and
    the INSTALLED builder cannot know one a newer mapper.py adds - so it would be
    missing exactly the models main had just added.
    """
    lines = REAL_MAPPER.splitlines(True)
    for index, line in enumerate(lines):
        if line.startswith("def build_mappers("):
            insert = index + 1
            break
    else:
        raise AssertionError("build_mappers is gone from the shipped mapper")
    added = "".join(
        lines[:insert]
        + ['    _add_with_lower(MAP_TO_UNSLOTH_16bit, "vendor/new", "unsloth/new")\n']
        + lines[insert:]
    )
    with _serving(added, monkeypatch):
        tables = loader_utils._get_new_mapper()
    assert any(
        table.get("vendor/new") == "unsloth/new" for table in tables if isinstance(table, dict)
    ), "an alias added inside the builder was dropped"


def test_a_table_cleared_after_the_builder_stays_cleared(monkeypatch):
    """The builder's aliases are installed where the call runs, not afterwards.

    A fetched mapper that calls `build_mappers(...)` and then rebinds a table leaves
    that table empty; reapplying the builder's aliases at the end reported support the
    newer file does not have.
    """
    lines = REAL_MAPPER.splitlines(True)
    insert = next(i for i, line in enumerate(lines) if line.startswith("def build_mappers(")) + 1
    added = (
        "".join(
            lines[:insert]
            + ['    _add_with_lower(MAP_TO_UNSLOTH_16bit, "vendor/late", "unsloth/late")\n']
            + lines[insert:]
        )
        + "\nMAP_TO_UNSLOTH_16bit = {}\n"
    )
    with _serving(added, monkeypatch):
        tables = loader_utils._get_new_mapper()
    for table in tables:
        assert not any("vendor/late" in str(key) for key in table)


def test_a_builder_that_is_never_called_is_not_read(monkeypatch):
    """A `def` that nothing calls runs nothing, which is the rule everywhere here."""
    uncalled = REAL_MAPPER.replace(
        "= build_mappers(__INT_TO_FLOAT_MAPPER)", "= ({}, {}, {}, {}, {})"
    )
    # The `def` line names it too, so the CALL is what has to be gone.
    assert "= build_mappers(" not in uncalled
    lines = uncalled.splitlines(True)
    insert = next(i for i, line in enumerate(lines) if line.startswith("def build_mappers(")) + 1
    uncalled = "".join(
        lines[:insert]
        + ['    _add_with_lower(MAP_TO_UNSLOTH_16bit, "vendor/dead", "unsloth/dead")\n']
        + lines[insert:]
    )
    with _serving(uncalled, monkeypatch):
        tables = loader_utils._get_new_mapper()
    for table in tables:
        assert not any("vendor/dead" in str(key) for key in table)


def test_a_branch_that_does_run_is_still_read(monkeypatch):
    """The exclusion is about DEAD code, not about conditionals in general."""
    live = REAL_MAPPER + (
        "\n"
        "if True:\n"
        "    FLOAT_TO_FP8_ROW_MAPPER['vendor/live-fp8'] = 'unsloth/live-fp8-row'\n"
    )
    with _serving(live, monkeypatch):
        tables = loader_utils._get_new_mapper()
    assert any(
        "vendor/live-fp8" in table for table in tables if isinstance(table, dict)
    ), "a live branch's mutation was dropped"


def test_a_builder_called_only_in_dead_code_adds_nothing(monkeypatch):
    """`if False: build_mappers(...)` installs nothing, so neither does the probe.

    The call was located by a raw walk, which sees unreachable and deferred code, and
    the builder's aliases were then applied by the fallback at the end even though the
    execution-order walk had correctly left the call out. That fabricated support the
    fetched mapper does not have, and a lookup for such a name raises the upgrade-only
    `NotImplementedError`.
    """
    dead = REAL_MAPPER.replace("= build_mappers(__INT_TO_FLOAT_MAPPER)", "= ({}, {}, {}, {}, {})")
    assert "= build_mappers(" not in dead
    lines = dead.splitlines(True)
    insert = next(i for i, line in enumerate(lines) if line.startswith("def build_mappers(")) + 1
    dead = "".join(
        lines[:insert]
        + ['    _add_with_lower(MAP_TO_UNSLOTH_16bit, "vendor/fabricated", "unsloth/fabricated")\n']
        + lines[insert:]
    )
    dead += "\nif False:\n    build_mappers(__INT_TO_FLOAT_MAPPER)\n"
    with _serving(dead, monkeypatch):
        tables = loader_utils._get_new_mapper()
    for table in tables:
        assert not any("vendor/fabricated" in str(key) for key in table)


def test_a_mutation_below_an_unconditional_continue_is_not_installed(monkeypatch):
    """`continue` ends the suite exactly as `break` does."""
    unreachable = REAL_MAPPER + (
        "\n"
        "while True:\n"
        "    continue\n"
        "    FLOAT_TO_FP8_ROW_MAPPER['vendor/dead-continue'] = 'unsloth/dead-row'\n"
    )
    with _serving(unreachable, monkeypatch):
        tables = loader_utils._get_new_mapper()
    for table in tables:
        assert not any("vendor/dead-continue" in str(key) for key in table)


def test_a_helper_call_spelled_with_keywords_is_read(monkeypatch):
    """`_add_with_lower(mapper = ..., key = ..., value = ...)` adds the same alias.

    Requiring three positional arguments dropped it, so the probe answered "no newer
    support" for a model the newer mapper really does map.
    """
    keyworded = REAL_MAPPER + (
        "\n"
        '_add_with_lower(mapper = MAP_TO_UNSLOTH_16bit, key = "vendor/kw", '
        'value = "unsloth/kw")\n'
    )
    with _serving(keyworded, monkeypatch):
        tables = loader_utils._get_new_mapper()
    assert any(
        "vendor/kw" in table for table in tables if isinstance(table, dict)
    ), "a keyword-spelled helper call was dropped"


def test_a_helper_call_that_binds_one_parameter_twice_is_skipped(monkeypatch):
    """It raises rather than adding anything, so the probe adds nothing either."""
    doubled = REAL_MAPPER + (
        "\n"
        '_add_with_lower(MAP_TO_UNSLOTH_16bit, "vendor/twice", mapper = MAP_TO_UNSLOTH_16bit)\n'
    )
    with _serving(doubled, monkeypatch):
        tables = loader_utils._get_new_mapper()
    for table in tables:
        assert not any("vendor/twice" in str(key) for key in table)


def test_a_builder_call_inside_a_lambda_does_not_run_the_builder(monkeypatch):
    """A deferred call to `build_mappers` is not a call.

    `_executed_nodes` yields the parent statement as well as the expressions that run,
    and the builder search walked the parent - descending back into exactly the
    deferred children the yield had excluded. `unused = lambda:
    build_mappers(__INT_TO_FLOAT_MAPPER)` therefore applied the fetched builder's
    aliases, fabricating support that importing that mapper never installs.
    """
    lines = REAL_MAPPER.splitlines(True)
    insert = next(i for i, line in enumerate(lines) if line.startswith("def build_mappers(")) + 1
    deferred = (
        "".join(
            lines[:insert]
            + ['    _add_with_lower(MAP_TO_UNSLOTH_16bit, "vendor/deferred", "unsloth/deferred")\n']
            + lines[insert:]
        ).replace("= build_mappers(__INT_TO_FLOAT_MAPPER)", "= ({}, {}, {}, {}, {})")
        + "\nunused = lambda: build_mappers(__INT_TO_FLOAT_MAPPER)\n"
    )
    assert "= build_mappers(__INT_TO_FLOAT_MAPPER)" not in deferred.replace(
        "unused = lambda: build_mappers(__INT_TO_FLOAT_MAPPER)", ""
    )
    with _serving(deferred, monkeypatch):
        tables = loader_utils._get_new_mapper()
    for table in tables:
        assert not any("vendor/deferred" in str(key) for key in table)


def test_the_source_table_is_read_where_the_builder_runs(monkeypatch):
    """A source table rebound AFTER the builder ran is not what the module exports.

    Taking the last binding in the file reversed the module's own state: a mapper that
    builds its exports from one table and then assigns a different
    `__INT_TO_FLOAT_MAPPER` was reported as supporting the names in the second one, and
    querying those raised the upgrade-only NotImplementedError.
    """
    rebound = REAL_MAPPER + (
        '\n__INT_TO_FLOAT_MAPPER = {"vendor/rebound-bnb-4bit": ("vendor/rebound",)}\n'
    )

    with _serving(REAL_MAPPER, monkeypatch):
        expected = loader_utils._get_new_mapper()
    with _serving(rebound, monkeypatch):
        got = loader_utils._get_new_mapper()

    assert any(expected), "the real mapper produced nothing, so this proves nothing"
    assert got == expected, "a rebind after the builder call was read as the source"
    for table in got:
        assert not any("rebound" in str(key) for key in table)


def test_a_source_table_rebound_before_the_builder_is_the_one_read(monkeypatch):
    """The other half: the last binding BEFORE the call still wins.

    `__INT_TO_FLOAT_MAPPER = {}` followed by the real table is the shape this rule was
    written for, and it has to keep reading the real one.
    """
    lines = REAL_MAPPER.splitlines(True)
    start = next(
        i for i, line in enumerate(lines) if line.startswith("__INT_TO_FLOAT_MAPPER")
    )
    emptied = "".join(lines[:start] + ["__INT_TO_FLOAT_MAPPER = {}\n"] + lines[start:])

    with _serving(REAL_MAPPER, monkeypatch):
        expected = loader_utils._get_new_mapper()
    with _serving(emptied, monkeypatch):
        got = loader_utils._get_new_mapper()

    assert any(expected)
    assert got == expected, "an earlier empty binding was read instead of the real one"


def test_a_source_table_extended_before_the_builder_is_read(monkeypatch):
    """`__INT_TO_FLOAT_MAPPER.update({...})` is an ordinary way to add entries.

    Reading only the last literal assignment dropped the update, so the entries never
    reached `build_mappers` and a model added that way got no upgrade notice.
    """
    lines = REAL_MAPPER.splitlines(True)
    insert = next(i for i, line in enumerate(lines) if line.startswith("def build_mappers("))
    extended = "".join(
        lines[:insert]
        + [
            '__INT_TO_FLOAT_MAPPER.update('
            '{"vendor/added-bnb-4bit": ("vendor/added",)})\n',
            '__INT_TO_FLOAT_MAPPER["vendor/keyed-bnb-4bit"] = ("vendor/keyed",)\n',
        ]
        + lines[insert:]
    )
    with _serving(extended, monkeypatch):
        tables = loader_utils._get_new_mapper()
    joined = " ".join(str(sorted(table)) for table in tables)
    assert "vendor/added" in joined, "an update before the builder was dropped"
    assert "vendor/keyed" in joined, "a subscript assignment before the builder was dropped"


def test_a_source_mutation_after_the_builder_is_not_read(monkeypatch):
    """The other direction: the builder was handed what existed when it ran."""
    late = REAL_MAPPER + (
        '\n__INT_TO_FLOAT_MAPPER.update({"vendor/late-bnb-4bit": ("vendor/late",)})\n'
    )
    with _serving(REAL_MAPPER, monkeypatch):
        expected = loader_utils._get_new_mapper()
    with _serving(late, monkeypatch):
        got = loader_utils._get_new_mapper()
    assert any(expected)
    assert got == expected, "a mutation after the builder call was read as the source"


def test_a_direct_entry_survives_an_empty_source_table(monkeypatch):
    """A row-only FP8 entry cannot be expressed through the source table at all.

    Returning as soon as the source table was empty left every direct mutation unread,
    so importing the fetched mapper would support the model while the probe reported
    nothing.
    """
    emptied = REAL_MAPPER.replace(
        "= build_mappers(__INT_TO_FLOAT_MAPPER)", "= build_mappers({})"
    ) + (
        '\nFLOAT_TO_FP8_ROW_MAPPER["vendor/only-fp8"] = "unsloth/only-fp8-row"\n'
    )
    with _serving(emptied, monkeypatch):
        tables = loader_utils._get_new_mapper()
    assert any(
        table.get("vendor/only-fp8") == "unsloth/only-fp8-row"
        for table in tables if isinstance(table, dict)
    ), "a direct entry was dropped because the source table was empty"


def test_a_body_that_installs_nothing_still_reports_nothing(monkeypatch):
    """And the guard the early return used to give is kept, decided from the result."""
    for body in ("", "x = 1\n", "__INT_TO_FLOAT_MAPPER = {}\n"):
        with _serving(body, monkeypatch):
            assert loader_utils._get_new_mapper() == ({}, {}, {}, {}, {}), body


def test_a_source_entry_deleted_before_the_builder_is_gone(monkeypatch):
    """`del __INT_TO_FLOAT_MAPPER[...]` removes it from what the builder is handed.

    Replaying only the additions left the probe reporting support for a model the
    newer mapper had just dropped, which raises the upgrade-only NotImplementedError
    for a name upgrading would not support either.
    """
    lines = REAL_MAPPER.splitlines(True)
    insert = next(i for i, line in enumerate(lines) if line.startswith("def build_mappers("))
    with _serving(REAL_MAPPER, monkeypatch):
        expected = loader_utils._get_new_mapper()
    victim = next(iter(expected[0]))

    removed = "".join(
        lines[:insert]
        + [f'del __INT_TO_FLOAT_MAPPER[{victim!r}]\n']
        + lines[insert:]
    )
    with _serving(removed, monkeypatch):
        tables = loader_utils._get_new_mapper()

    assert victim in expected[0], "the fixture picked a name the probe never reported"
    assert victim not in tables[0], "a deleted source entry was still reported"


def test_deleting_the_whole_source_table_leaves_nothing_to_read(monkeypatch):
    """`del __INT_TO_FLOAT_MAPPER` unbinds it, so the builder is handed nothing."""
    lines = REAL_MAPPER.splitlines(True)
    insert = next(i for i, line in enumerate(lines) if line.startswith("def build_mappers("))
    removed = "".join(
        lines[:insert] + ["del __INT_TO_FLOAT_MAPPER\n"] + lines[insert:]
    )
    with _serving(removed, monkeypatch):
        assert loader_utils._get_new_mapper() == ({}, {}, {}, {}, {})
