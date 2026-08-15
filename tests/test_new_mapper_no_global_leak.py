"""Regression test for ``_get_new_mapper`` leaking into ``loader_utils`` globals.

``get_model_name`` calls ``_get_new_mapper()`` whenever a name misses the local
tables, purely to answer "would a newer Unsloth support this?". It fetches
``mapper.py`` from GitHub main, prefixes the three mappers it wants with
``NEW_``, and ``exec``s the result into ``globals()``.

The slice starts at ``__INT_TO_FLOAT_MAPPER``, so it also carries
``FLOAT_TO_FP8_BLOCK_MAPPER``/``FLOAT_TO_FP8_ROW_MAPPER`` and the two
``_add_*`` helpers, and those names are NOT renamed. Exec'ing into
``globals()`` therefore rebinds the FP8 tables that ``loader_utils`` imported
from the installed ``mapper``, so every later ``get_model_name(...,
load_in_fp8 = ...)`` in the process resolves through GitHub main's table
instead of the installed one. The probe is supposed to read, not to swap the
installed mappings out from under the caller.

``loader_utils`` imports torch, so ast-extract ``_get_new_mapper`` and run it
against a stubbed ``requests`` rather than importing unsloth (which needs a GPU).
"""

import ast
import os
import sys
import types

_MODELS = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth", "models")


def _mapper_source():
    with open(os.path.join(_MODELS, "mapper.py"), encoding = "utf-8") as f:
        return f.read()


def _extract_get_new_mapper(namespace):
    with open(os.path.join(_MODELS, "loader_utils.py"), encoding = "utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_get_new_mapper":
            exec(compile(ast.Module([node], []), node.name, "exec"), namespace)
            return namespace["_get_new_mapper"]
    raise AssertionError("_get_new_mapper not found in loader_utils.py")


class _FakeResponse:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _install_fake_requests(monkeypatch, text):
    module = types.ModuleType("requests")
    module.get = lambda url, timeout = None: _FakeResponse(text)
    monkeypatch.setitem(sys.modules, "requests", module)


def test_get_new_mapper_does_not_rebind_the_installed_fp8_tables(monkeypatch):
    _install_fake_requests(monkeypatch, _mapper_source())

    installed = {}
    exec(compile(_mapper_source(), "mapper.py", "exec"), installed)
    block = installed["FLOAT_TO_FP8_BLOCK_MAPPER"]
    row = installed["FLOAT_TO_FP8_ROW_MAPPER"]
    assert block and row, "the installed FP8 tables should not be empty"

    # Stand in for loader_utils' module globals, which import the FP8 tables.
    namespace = {"FLOAT_TO_FP8_BLOCK_MAPPER": block, "FLOAT_TO_FP8_ROW_MAPPER": row}
    get_new_mapper = _extract_get_new_mapper(namespace)

    int_to_float, float_to_int, map_to_16bit, fp8_block, fp8_row = get_new_mapper()

    # _get_new_mapper swallows every exception and returns empty dicts, so assert
    # it actually ran before trusting anything below.
    assert int_to_float and float_to_int and map_to_16bit, "the fetch/exec path did not run"

    # the probe has to hand the FETCHED fp8 tables back, or a newly added fp8 repo would
    # miss both the installed tables and the probe and skip the upgrade message
    assert fp8_block and fp8_row
    assert fp8_block is not block and fp8_row is not row

    assert namespace["FLOAT_TO_FP8_BLOCK_MAPPER"] is block
    assert namespace["FLOAT_TO_FP8_ROW_MAPPER"] is row


def test_get_new_mapper_leaves_no_helpers_behind(monkeypatch):
    _install_fake_requests(monkeypatch, _mapper_source())

    namespace = {}
    get_new_mapper = _extract_get_new_mapper(namespace)
    before = set(namespace)

    assert all(get_new_mapper()), "the fetch/exec path did not run"

    leaked = set(namespace) - before
    assert not leaked, f"_get_new_mapper leaked {sorted(leaked)} into its module globals"
