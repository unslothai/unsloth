"""Regression test for ``_get_new_mapper`` leaking into ``loader_utils`` globals.

``get_model_name`` calls ``_get_new_mapper()`` whenever a name misses the local
tables, purely to answer "would a newer Unsloth support this?". It fetches
``mapper.py`` from GitHub main, prefixes the three mappers it wants with
``NEW_``, and ``exec``s the result.

The slice starts at ``__INT_TO_FLOAT_MAPPER``, so it also carries
``FLOAT_TO_FP8_BLOCK_MAPPER``/``FLOAT_TO_FP8_ROW_MAPPER`` and the two
``_add_*`` helpers, and those names are NOT renamed. Exec'ing into
``globals()`` therefore rebinds the FP8 tables that ``loader_utils`` imported
from the installed ``mapper``, so every later ``get_model_name(...,
load_in_fp8 = ...)`` in the process resolves through GitHub main's table
instead of the installed one. The probe is supposed to read, not to swap the
installed mappings out from under the caller.

Isolating the exec is only half of it: the probe still has to *use* the fetched
FP8 tables, and it must survive a fetched ``mapper.py`` that has no FP8 tables
at all. The last two tests pin those, driving ``get_model_name`` end to end
rather than inspecting what ``_get_new_mapper`` returns, because a table that
merely looks fresh is not the same as one the resolver actually consults.

``loader_utils`` imports torch, so ast-extract the resolvers and run them
against a stubbed ``requests`` rather than importing unsloth (which needs a GPU).
"""

import ast
import os
import sys
import types
from importlib.metadata import version as _dist_version

import pytest
from packaging.version import Version

# loader_utils reads both; via metadata so transformers stays unimported here.
transformers_version = _dist_version("transformers")

_MODELS = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth", "models")

# A repo that only the fetched mapper knows about, so resolving it proves the
# probe read the FETCHED FP8 table and not the installed one.
FETCH_ONLY_FP8_REPO = "zeta-org/Zeta-9B-PR7478-FP8"

_RESOLVERS = ("__get_model_name", "_get_new_mapper", "_resolve_with_mappers", "get_model_name")


def _mapper_source():
    with open(os.path.join(_MODELS, "mapper.py"), encoding = "utf-8") as f:
        return f.read()


def _loader_utils_source():
    with open(os.path.join(_MODELS, "loader_utils.py"), encoding = "utf-8") as f:
        return f.read()


def _extract_get_new_mapper(namespace):
    tree = ast.parse(_loader_utils_source())
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


class _NoVLLM:
    """Stand-in for ``importlib`` that reports vllm as absent.

    From vllm 0.12.0 on, ``__get_model_name`` returns the original name on the
    first FP8 pass, so the probe is never reached and an FP8 test would pass
    without exercising anything.
    """

    class util:
        @staticmethod
        def find_spec(name):
            return None


def _extract_resolvers(monkeypatch, fetched_source):
    """exec the four resolvers into a namespace standing in for loader_utils' globals."""
    installed = {}
    exec(compile(_mapper_source(), "mapper.py", "exec"), installed)
    _install_fake_requests(monkeypatch, fetched_source)

    namespace = {
        "os": os,
        "importlib": _NoVLLM,
        "SUPPORTS_FOURBIT": True,
        "BAD_MAPPINGS": {},
        "_env_says_offline": lambda: False,
        # Unreached today (vllm stubbed absent, SUPPORTS_FOURBIT True), but present
        # so a change to either gate fails on the assertion, not on a bare NameError.
        "Version": Version,
        "transformers_version": transformers_version,
        "INT_TO_FLOAT_MAPPER": installed["INT_TO_FLOAT_MAPPER"],
        "FLOAT_TO_INT_MAPPER": installed["FLOAT_TO_INT_MAPPER"],
        "MAP_TO_UNSLOTH_16bit": installed["MAP_TO_UNSLOTH_16bit"],
        "FLOAT_TO_FP8_BLOCK_MAPPER": installed["FLOAT_TO_FP8_BLOCK_MAPPER"],
        "FLOAT_TO_FP8_ROW_MAPPER": installed["FLOAT_TO_FP8_ROW_MAPPER"],
    }
    nodes = [
        node
        for node in ast.parse(_loader_utils_source()).body
        if isinstance(node, ast.FunctionDef) and node.name in _RESOLVERS
    ]
    assert len(nodes) == len(_RESOLVERS), (
        f"loader_utils.py no longer defines all of {_RESOLVERS}; "
        f"found {sorted(node.name for node in nodes)}"
    )
    exec(compile(ast.Module(nodes, []), "loader_utils.py", "exec"), namespace)
    return namespace, installed


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

    # Shape only: this fixture serves one mapper.py as both installed and fetched
    # source, so identity catches "handed back the installed tables" but not
    # "returned a stale copy". Provenance is pinned by
    # test_probe_answers_for_an_fp8_repo_only_the_fetched_mapper_knows.
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


def test_probe_answers_for_an_fp8_repo_only_the_fetched_mapper_knows(monkeypatch):
    # Row scaling needs FBGEMM; without it the resolver reads the block table only.
    monkeypatch.delenv("UNSLOTH_HAS_FBGEMM", raising = False)
    fetched = _mapper_source() + (
        f'\nFLOAT_TO_FP8_BLOCK_MAPPER["{FETCH_ONLY_FP8_REPO.lower()}"] = '
        f'"unsloth/Zeta-9B-PR7478-FP8"\n'
    )
    namespace, installed = _extract_resolvers(monkeypatch, fetched)
    assert FETCH_ONLY_FP8_REPO.lower() not in installed["FLOAT_TO_FP8_BLOCK_MAPPER"], (
        "fixture is stale: the installed table already knows this repo, so the test "
        "would pass without the probe reading the fetched one"
    )

    with pytest.raises(NotImplementedError, match = "not supported in your current Unsloth version"):
        namespace["get_model_name"](FETCH_ONLY_FP8_REPO, load_in_fp8 = True)


def test_fetched_mapper_without_fp8_tables_keeps_the_4bit_probe_alive(monkeypatch):
    # Stands in for a mapper.py from before the FP8 tables existed: the two names
    # are simply absent from the fetched module, which is what the probe has to
    # tolerate. Indexing them would KeyError into the bare except and return five
    # empty dicts, taking the 4bit/16bit half of the probe down with the FP8 half.
    fetched = _mapper_source() + "\ndel FLOAT_TO_FP8_BLOCK_MAPPER, FLOAT_TO_FP8_ROW_MAPPER\n"
    namespace, _ = _extract_resolvers(monkeypatch, fetched)

    int_to_float, float_to_int, map_to_16bit, fp8_block, fp8_row = namespace["_get_new_mapper"]()

    assert (
        int_to_float and float_to_int and map_to_16bit
    ), "a fetched mapper.py without FP8 tables killed the 4bit half of the probe"
    assert fp8_block == {} and fp8_row == {}
