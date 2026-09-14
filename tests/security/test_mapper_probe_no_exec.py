# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`_get_new_mapper` used to `exec` an HTTPS response body: arbitrary code execution
inside an ordinary model load, gated only on transport integrity. It now
`literal_eval`s the `__INT_TO_FLOAT_MAPPER` dict and derives the five tables with the
INSTALLED `build_mappers`, so a hostile body can still change what the probe reports,
which was always true, but can no longer run."""

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
    """The streaming half of `requests.Response`: a fake offering only `.text` would
    let the byte cap and the by-hand redirect following regress silently."""

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
        """`read1` returns what ONE socket read produced, so the deadline is checked
        between reads rather than only between whole chunks."""
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
    """Records any exec/eval reached during the probe. RECORDS rather than raises,
    since the probe's bare except would swallow an AssertionError."""
    calls = []

    def _forbidden(name):
        def _record(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"_get_new_mapper called {name}()")

        return _record

    monkeypatch.setattr(builtins, "exec", _forbidden("exec"))
    monkeypatch.setattr(builtins, "eval", _forbidden("eval"))
    return calls


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
    assert len(result) == 5
    assert all(isinstance(table, dict) for table in result)


def test_probe_does_not_call_exec_or_eval(no_dynamic_execution, monkeypatch):
    """Stronger than the marker: asserting only the returned shape would also hold
    for the old implementation, whose bare except ate the fixture's AssertionError."""
    with _serving(REAL_MAPPER, monkeypatch):
        result = loader_utils._get_new_mapper()
    assert no_dynamic_execution == [], f"the probe reached {no_dynamic_execution}"
    # And it did the WORK, rather than falling into the except and returning empties.
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
    """`ast.parse` builds the whole tree first, so deep nesting raises RecursionError,
    which is not a ValueError. This pins that the bare except keeps catching it."""
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
    """The size cap: `requests`' timeout is per-read, so a body can be any size."""
    cap = _byte_cap()
    body = "__INT_TO_FLOAT_MAPPER = {'a' : ('b',)}\n" + ("# padding\n" * (cap // 10 + 1))
    assert len(body) > cap
    with _serving(body, monkeypatch):
        assert loader_utils._get_new_mapper() == ({}, {}, {}, {}, {})


def test_the_cap_bounds_parse_cost_not_just_download_size():
    """Short statements are the dense case: several AST nodes each, so the cap is set
    from what the PARSE can afford rather than what the network can."""
    cap = _byte_cap()
    assert cap <= 2_000_000, (
        f"a {cap} byte cap admits roughly {cap // 4} short statements, which is a "
        f"memory blow-up in ast.parse rather than a bounded read"
    )


def test_the_real_mapper_is_far_below_the_cap():
    assert len(REAL_MAPPER) < _byte_cap() / 2


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
    """A harmless `__INT_TO_FLOAT_MAPPER: dict = {...}` upstream would otherwise
    return five empty tables with nothing looking broken."""
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
    """The probe reads what the fetched module INSTALLS, not what it contains: a
    fabricated mapping raises the upgrade notice for a name that does not exist."""
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
    """A `lambda` is an expression, so the one attached to an assignment SEEDED the
    walk even though the walk refuses to descend into one it meets as a child."""
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
    """`build_mappers` is called at import, so its body runs. The aliases that cannot
    be derived from the source table live in there, and the installed builder cannot
    know one a newer mapper.py adds."""
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
    """Installed where the call RUNS: a rebind written after it leaves the table
    empty, and reapplying the aliases at the end reported support that is not there."""
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
    """`if False: build_mappers(...)` installs nothing, so neither does the probe. The call was located by a raw walk, which sees unreachable and deferred code, and the builder's aliases were then applied by the fallback at the end even though the execution-order walk had correctly left the call out. That fabricated support the fetched mapper does not have, and a lookup for such a name raises the upgrade-only `NotImplementedError`."""
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
    """`_add_with_lower(mapper = ..., key = ..., value = ...)` adds the same alias. Requiring three positional arguments dropped it, so the probe answered "no newer support" for a model the newer mapper really does map."""
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
    """A deferred call to `build_mappers` is not a call: walking the parent statement
    descended back into the deferred children the yield had excluded."""
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
    """A source table rebound AFTER the builder ran is not what the module exports."""
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
    """The other half: the last binding BEFORE the call still wins. `__INT_TO_FLOAT_MAPPER = {}` followed by the real table is the shape this rule was written for, and it has to keep reading the real one."""
    lines = REAL_MAPPER.splitlines(True)
    start = next(i for i, line in enumerate(lines) if line.startswith("__INT_TO_FLOAT_MAPPER"))
    emptied = "".join(lines[:start] + ["__INT_TO_FLOAT_MAPPER = {}\n"] + lines[start:])

    with _serving(REAL_MAPPER, monkeypatch):
        expected = loader_utils._get_new_mapper()
    with _serving(emptied, monkeypatch):
        got = loader_utils._get_new_mapper()

    assert any(expected)
    assert got == expected, "an earlier empty binding was read instead of the real one"


def test_a_source_table_extended_before_the_builder_is_read(monkeypatch):
    """`.update({...})` is an ordinary way to extend the table before it is handed
    over, and the entries have to reach `build_mappers`."""
    lines = REAL_MAPPER.splitlines(True)
    insert = next(i for i, line in enumerate(lines) if line.startswith("def build_mappers("))
    extended = "".join(
        lines[:insert]
        + [
            "__INT_TO_FLOAT_MAPPER.update(" '{"vendor/added-bnb-4bit": ("vendor/added",)})\n',
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
    """A row-only FP8 entry cannot be expressed through the source table at all, so
    returning early on an empty one left every direct mutation unread."""
    emptied = REAL_MAPPER.replace(
        "= build_mappers(__INT_TO_FLOAT_MAPPER)", "= build_mappers({})"
    ) + ('\nFLOAT_TO_FP8_ROW_MAPPER["vendor/only-fp8"] = "unsloth/only-fp8-row"\n')
    with _serving(emptied, monkeypatch):
        tables = loader_utils._get_new_mapper()
    assert any(
        table.get("vendor/only-fp8") == "unsloth/only-fp8-row"
        for table in tables
        if isinstance(table, dict)
    ), "a direct entry was dropped because the source table was empty"


def test_a_body_that_installs_nothing_still_reports_nothing(monkeypatch):
    """And the guard the early return used to give is kept, decided from the result."""
    for body in ("", "x = 1\n", "__INT_TO_FLOAT_MAPPER = {}\n"):
        with _serving(body, monkeypatch):
            assert loader_utils._get_new_mapper() == ({}, {}, {}, {}, {}), body


def test_a_source_entry_deleted_before_the_builder_is_gone(monkeypatch):
    """`del __INT_TO_FLOAT_MAPPER[...]` removes it from what the builder is handed,
    so replaying only the additions reported a model the newer mapper had dropped."""
    lines = REAL_MAPPER.splitlines(True)
    insert = next(i for i, line in enumerate(lines) if line.startswith("def build_mappers("))
    with _serving(REAL_MAPPER, monkeypatch):
        expected = loader_utils._get_new_mapper()
    victim = next(iter(expected[0]))

    removed = "".join(
        lines[:insert] + [f"del __INT_TO_FLOAT_MAPPER[{victim!r}]\n"] + lines[insert:]
    )
    with _serving(removed, monkeypatch):
        tables = loader_utils._get_new_mapper()

    assert victim in expected[0], "the fixture picked a name the probe never reported"
    assert victim not in tables[0], "a deleted source entry was still reported"


def test_deleting_the_whole_source_table_leaves_nothing_to_read(monkeypatch):
    """`del __INT_TO_FLOAT_MAPPER` unbinds it, so the builder is handed nothing."""
    lines = REAL_MAPPER.splitlines(True)
    insert = next(i for i, line in enumerate(lines) if line.startswith("def build_mappers("))
    removed = "".join(lines[:insert] + ["del __INT_TO_FLOAT_MAPPER\n"] + lines[insert:])
    with _serving(removed, monkeypatch):
        assert loader_utils._get_new_mapper() == ({}, {}, {}, {}, {})


def test_an_exported_entry_deleted_by_the_fetched_mapper_is_gone(monkeypatch):
    """`del FLOAT_TO_INT_MAPPER["vendor/base"]` removes the alias on import."""
    with _serving(REAL_MAPPER, monkeypatch):
        expected = loader_utils._get_new_mapper()
    victim = next(iter(expected[1]))
    removed = REAL_MAPPER + f"\ndel FLOAT_TO_INT_MAPPER[{victim!r}]\n"
    with _serving(removed, monkeypatch):
        tables = loader_utils._get_new_mapper()
    assert victim in expected[1], "the fixture picked a name the probe never reported"
    assert victim not in tables[1], "a deleted exported entry was still reported"


def test_an_addition_below_a_return_in_the_builder_is_not_read(monkeypatch):
    """Nothing after an unconditional `return` in that suite runs."""
    lines = REAL_MAPPER.splitlines(True)
    insert = next(i for i, line in enumerate(lines) if line.startswith("def build_mappers(")) + 1
    dead = "".join(
        lines[:insert]
        + [
            "    if True:\n        return ({}, {}, {}, {}, {})\n",
            '    _add_with_lower(MAP_TO_UNSLOTH_16bit, "vendor/dead", "unsloth/dead")\n',
        ]
        + lines[insert:]
    )
    with _serving(dead, monkeypatch):
        tables = loader_utils._get_new_mapper()
    for table in tables:
        assert not any("vendor/dead" in str(key) for key in table)


def test_the_builder_call_that_binds_the_exports_is_the_one_read(monkeypatch):
    """A validation call over a dummy table before the real one is not the source."""
    lines = REAL_MAPPER.splitlines(True)
    insert = next(
        i
        for i, line in enumerate(lines)
        if line.rstrip().endswith("= build_mappers(__INT_TO_FLOAT_MAPPER)")
    )
    while insert > 0 and not lines[insert - 1].rstrip().endswith((")", ":", "}", "]")):
        insert -= 1
    dummied = "".join(
        lines[:insert]
        + [
            'DUMMY_SOURCE = {"vendor/dummy-bnb-4bit": ("vendor/dummy",)}\n',
            "_checked = build_mappers(DUMMY_SOURCE)\n",
        ]
        + lines[insert:]
    )
    with _serving(REAL_MAPPER, monkeypatch):
        expected = loader_utils._get_new_mapper()
    with _serving(dummied, monkeypatch):
        tables = loader_utils._get_new_mapper()
    assert any(expected)
    for table in tables:
        assert not any(
            "vendor/dummy" in str(key) for key in table
        ), "the dummy call was read as the one that populates the exports"
    assert tables == expected


def test_the_last_export_producing_call_is_the_one_read(monkeypatch):
    """Rebuilding the exports twice leaves the SECOND result installed. Reading the first invented support the second build removed and dropped what it added, either way through the upgrade-only NotImplementedError."""
    lines = REAL_MAPPER.splitlines(True)
    insert = (
        next(
            i
            for i, line in enumerate(lines)
            if line.rstrip().endswith("= build_mappers(__INT_TO_FLOAT_MAPPER)")
        )
        + 1
    )
    rebuilt = "".join(
        lines[:insert]
        + [
            'SECOND_SOURCE = {"vendor/second-bnb-4bit": ("vendor/second",)}\n',
            "(\n"
            "    INT_TO_FLOAT_MAPPER,\n"
            "    FLOAT_TO_INT_MAPPER,\n"
            "    MAP_TO_UNSLOTH_16bit,\n"
            "    FLOAT_TO_FP8_BLOCK_MAPPER,\n"
            "    FLOAT_TO_FP8_ROW_MAPPER,\n"
            ") = build_mappers(SECOND_SOURCE)\n",
        ]
        + lines[insert:]
    )
    with _serving(rebuilt, monkeypatch):
        tables = loader_utils._get_new_mapper()
    assert any(
        "vendor/second" in str(key) for table in tables for key in table
    ), "the second build was ignored"


def test_an_addition_after_a_returning_loop_is_not_read(monkeypatch):
    """`while True: return` ends the function, so nothing written below it runs."""
    lines = REAL_MAPPER.splitlines(True)
    insert = next(i for i, line in enumerate(lines) if line.startswith("def build_mappers(")) + 1
    dead = "".join(
        lines[:insert]
        + [
            "    while True:\n        return ({}, {}, {}, {}, {})\n",
            '    _add_with_lower(MAP_TO_UNSLOTH_16bit, "vendor/after-loop", "unsloth/after-loop")\n',
        ]
        + lines[insert:]
    )
    with _serving(dead, monkeypatch):
        tables = loader_utils._get_new_mapper()
    for table in tables:
        assert not any("vendor/after-loop" in str(key) for key in table)


def test_an_addition_after_a_breaking_loop_is_still_read(monkeypatch):
    """The other direction: `break` leaves the LOOP, and the rest of the body runs."""
    lines = REAL_MAPPER.splitlines(True)
    insert = next(i for i, line in enumerate(lines) if line.startswith("def build_mappers(")) + 1
    live = "".join(
        lines[:insert]
        + [
            "    while True:\n        break\n",
            '    _add_with_lower(MAP_TO_UNSLOTH_16bit, "vendor/after-break", "unsloth/after-break")\n',
        ]
        + lines[insert:]
    )
    with _serving(live, monkeypatch):
        tables = loader_utils._get_new_mapper()
    assert any(
        table.get("vendor/after-break") == "unsloth/after-break"
        for table in tables
        if isinstance(table, dict)
    ), "an addition after a loop that only breaks was dropped"


def test_a_mutation_between_two_builder_calls_is_replayed(monkeypatch):
    """Up to the SELECTED call: a validation call above a mutation of the source
    table made the probe stop early and build the exports from stale data."""
    lines = REAL_MAPPER.splitlines(True)
    insert = next(
        i
        for i, line in enumerate(lines)
        if line.rstrip().endswith("= build_mappers(__INT_TO_FLOAT_MAPPER)")
    )
    while insert > 0 and not lines[insert - 1].rstrip().endswith((")", ":", "}", "]")):
        insert -= 1
    staged = "".join(
        lines[:insert]
        + [
            "_checked = build_mappers(__INT_TO_FLOAT_MAPPER)\n",
            '__INT_TO_FLOAT_MAPPER.update({"vendor/between-bnb-4bit": ("vendor/between",)})\n',
        ]
        + lines[insert:]
    )
    with _serving(staged, monkeypatch):
        tables = loader_utils._get_new_mapper()
    assert any(
        "vendor/between" in str(key) for table in tables for key in table
    ), "a mutation between the validation call and the real one was dropped"


def test_builder_aliases_land_at_the_export_producing_call(monkeypatch):
    """Where the exports are BUILT: applying them at the first builder call put them
    above a rebind written between the two, which wiped them."""
    lines = REAL_MAPPER.splitlines(True)
    body = next(i for i, line in enumerate(lines) if line.startswith("def build_mappers(")) + 1
    added = (
        lines[:body]
        + ['    _add_with_lower(MAP_TO_UNSLOTH_16bit, "vendor/late-alias", "unsloth/late-alias")\n']
        + lines[body:]
    )
    insert = next(
        i
        for i, line in enumerate(added)
        if line.rstrip().endswith("= build_mappers(__INT_TO_FLOAT_MAPPER)")
    )
    while insert > 0 and not added[insert - 1].rstrip().endswith((")", ":", "}", "]")):
        insert -= 1
    staged = "".join(
        added[:insert]
        + [
            "_checked = build_mappers(__INT_TO_FLOAT_MAPPER)\n",
            "MAP_TO_UNSLOTH_16bit = {}\n",
        ]
        + added[insert:]
    )
    with _serving(staged, monkeypatch):
        tables = loader_utils._get_new_mapper()
    assert any(
        table.get("vendor/late-alias") == "unsloth/late-alias"
        for table in tables
        if isinstance(table, dict)
    ), "the builder's aliases were applied before a rebind that wiped them"


_EXPORT_BUILD = (
    "(\n"
    "    INT_TO_FLOAT_MAPPER,\n"
    "    FLOAT_TO_INT_MAPPER,\n"
    "    MAP_TO_UNSLOTH_16bit,\n"
    "    FLOAT_TO_FP8_BLOCK_MAPPER,\n"
    "    FLOAT_TO_FP8_ROW_MAPPER,\n"
    ") = build_mappers(__INT_TO_FLOAT_MAPPER)\n"
)


def _above_the_builder(lines):
    """The index of the line that opens the export-producing assignment."""
    insert = next(
        i
        for i, line in enumerate(lines)
        if line.rstrip().endswith("= build_mappers(__INT_TO_FLOAT_MAPPER)")
    )
    while insert > 0 and not lines[insert - 1].rstrip().endswith((")", ":", "}", "]")):
        insert -= 1
    return insert


def test_a_rebuild_drops_a_mutation_made_before_it(monkeypatch):
    """The second build REPLACES the tables, so an entry added between the two goes."""
    lines = REAL_MAPPER.splitlines(True)
    insert = _above_the_builder(lines)
    staged = "".join(
        lines[:insert]
        + [
            _EXPORT_BUILD,
            'FLOAT_TO_FP8_ROW_MAPPER["vendor/stale"] = "unsloth/stale-row"\n',
        ]
        + lines[insert:]
    )
    with _serving(staged, monkeypatch):
        tables = loader_utils._get_new_mapper()
    assert not any(
        "vendor/stale" in table for table in tables if isinstance(table, dict)
    ), "a mutation the later rebuild replaced was still reported as installed"


def test_a_mutation_above_a_class_shadow_still_counts(monkeypatch):
    """The shadow starts where the class binds the name, not at the suite's first line."""
    lines = REAL_MAPPER.splitlines(True)
    staged = "".join(
        lines
        + [
            "class _Late:\n",
            '    FLOAT_TO_FP8_ROW_MAPPER["vendor/live"] = "unsloth/live-row"\n',
            "    FLOAT_TO_FP8_ROW_MAPPER = {}\n",
        ]
    )
    with _serving(staged, monkeypatch):
        tables = loader_utils._get_new_mapper()
    assert any(
        table.get("vendor/live") == "unsloth/live-row"
        for table in tables
        if isinstance(table, dict)
    ), "an entry installed before the class bound the name was dropped"


# How mapper.py looks on any release older than `build_mappers`: the five exports are
# initialised empty and filled by a module-scope loop rather than by a builder call.
_NO_BUILDER_MAPPER = (
    '__INT_TO_FLOAT_MAPPER = {"vendor/x-bnb-4bit": ("vendor/x",)}\n'
    "INT_TO_FLOAT_MAPPER  = {}\n"
    "FLOAT_TO_INT_MAPPER  = {}\n"
    "MAP_TO_UNSLOTH_16bit = {}\n"
    "FLOAT_TO_FP8_BLOCK_MAPPER = {}\n"
    "FLOAT_TO_FP8_ROW_MAPPER   = {}\n"
    "for key, values in __INT_TO_FLOAT_MAPPER.items():\n"
    "    INT_TO_FLOAT_MAPPER[key] = values[0]\n"
)


def test_a_mapper_without_a_builder_still_reports_its_models(monkeypatch):
    """The empty initialisers above a fill loop are not the module clearing the
    tables: reading them as clears emptied all five and the upgrade notice died."""
    with _serving(_NO_BUILDER_MAPPER, monkeypatch):
        tables = loader_utils._get_new_mapper()
    assert any(table for table in tables), "the probe reported no models at all"
    assert any(
        "vendor/x" in str(key) or "vendor/x" in str(value)
        for table in tables
        if isinstance(table, dict)
        for key, value in table.items()
    ), "the source table's entry did not reach the exported tables"


def test_a_nonempty_rebind_still_replaces_the_table(monkeypatch):
    """Only the EMPTY literal is an initialiser; a real one still replaces."""
    staged = _NO_BUILDER_MAPPER + 'INT_TO_FLOAT_MAPPER = {"vendor/only": "unsloth/only"}\n'
    with _serving(staged, monkeypatch):
        tables = loader_utils._get_new_mapper()
    assert tables[0] == {"vendor/only": "unsloth/only"}, tables[0]


def test_the_upgrade_notice_fires_for_a_model_only_the_newer_mapper_knows(monkeypatch):
    """End to end: an unmapped name plus a newer mapper that maps it must raise."""
    staged = REAL_MAPPER.replace(
        "__INT_TO_FLOAT_MAPPER = \\\n{\n",
        "__INT_TO_FLOAT_MAPPER = \\\n{\n"
        '    "vendor/brand-new-bnb-4bit": ("vendor/brand-new",),\n',
        1,
    )
    assert staged != REAL_MAPPER, "the source table header moved"
    monkeypatch.setattr(loader_utils, "_env_says_offline", lambda: False)
    with _serving(staged, monkeypatch):
        with pytest.raises(NotImplementedError, match = "not supported in your current"):
            loader_utils.get_model_name("vendor/brand-new", load_in_4bit = True)
