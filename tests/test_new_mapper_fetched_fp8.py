# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests for what ``_get_new_mapper`` hands back to the upgrade probe.

``test_new_mapper_no_global_leak.py`` serves the repo's own ``mapper.py`` as both installed
and fetched source, so it cannot tell a fetched table from a fresh copy of the installed one.
Two gaps it misses:

1. The probe must answer for an fp8 repo only the FETCHED mapper knows, so an extra ``"8"``
   entry is spliced into the fetched source only. Isolating the exec without returning the
   fetched fp8 tables would silently drop the fp8 half of the upgrade check.
2. The probe must survive a fetched ``mapper.py`` with no fp8 tables (anything older, or a
   future rename): reading them with ``[]`` raises ``KeyError`` into the bare ``except``,
   taking the 4bit half, the probe's whole purpose, down with it.

Both of the above only reach the block table. The last two tests take the row branch, which
``load_in_fp8 = True`` plus ``UNSLOTH_HAS_FBGEMM`` selects ahead of block: deleting that branch,
or dropping ``_resolve_with_mappers``' ``fp8_row`` argument so it falls back to the installed
table, both leave every other test here green.

``loader_utils`` imports torch, so ast-extract the resolvers and run them against a stubbed
``requests``, as in ``tests/test_bad_mappings_redirect.py``.
"""

import ast
import os
import sys
import types

_MODELS = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth", "models")

_WANTED = {"__get_model_name", "_resolve_with_mappers", "_get_new_mapper", "get_model_name"}

# An fp8 ("8") model, spliced into the FETCHED mapper only.
_NEW_KEY = "unsloth/Zeta-9B-Only-On-Main"
_NEW_OFFICIAL = "zeta-org/Zeta-9B-Only-On-Main-FP8"
_NEW_BLOCK = "unsloth/Zeta-9B-Only-On-Main-FP8-Block"
_NEW_ROW = "unsloth/Zeta-9B-Only-On-Main-FP8-Row"
_ANCHOR = '    "unsloth/Kimi-K2-Instruct-BF16" : ('
# Row table only, so the block branch cannot answer for it and mask a row-path regression.
_ROW_ONLY = "zeta-org/Zeta-9B-Row-Only-FP8"


def _loader_utils_globals():
    """The real loader_utils module globals, for anything the stand-in needs verbatim."""
    import unsloth.models.loader_utils as loader_utils
    return vars(loader_utils)


def _mapper_source():
    with open(os.path.join(_MODELS, "mapper.py"), encoding = "utf-8") as f:
        return f.read()


def _with_extra_fp8_model(source):
    assert _ANCHOR in source, "anchor moved; update this test"
    entry = (
        f'    "{_NEW_KEY}" : {{\n'
        f'        "16" : ("{_NEW_KEY}", "zeta-org/Zeta-9B-Only-On-Main"),\n'
        f'        "8"  : ("{_NEW_OFFICIAL}", "{_NEW_BLOCK}", "{_NEW_ROW}"),\n'
        f"    }},\n"
    )
    return source.replace(_ANCHOR, entry + _ANCHOR, 1)


def _with_row_only_fp8_model(source):
    """Fetched row table only. Block must not know it, or the block branch answers instead."""
    return source + f'\nFLOAT_TO_FP8_ROW_MAPPER["{_ROW_ONLY.lower()}"] = "{_NEW_ROW}"\n'


def _without_fp8_tables(source):
    """A mapper.py from before the fp8 tables existed."""
    return source.replace("FLOAT_TO_FP8_BLOCK_MAPPER", "SOME_OTHER_BLOCK_TABLE").replace(
        "FLOAT_TO_FP8_ROW_MAPPER", "SOME_OTHER_ROW_TABLE"
    )


class _FakeRaw:
    """`read1` over a fixed list of chunks, returning b"" at the end."""

    def __init__(self, chunks):
        self._chunks = iter(chunks)

    def read1(self, amount = -1):
        return next(self._chunks, b"")


class _FakeResponse:
    """The streaming half of `requests.Response`, which is all the probe uses.

    `_get_new_mapper` reads the body in chunks and stops at its byte cap while
    reading, because `requests.get` would otherwise buffer and decode the whole
    response before any length check could run. It also follows redirects by hand,
    since `requests` drains each intermediate body inside `get`, so the fake has to
    carry a status and headers as well as `iter_content`.
    """

    def __init__(
        self,
        text,
        chunks = None,
        status_code = 200,
        headers = None,
    ):
        self.encoding = "utf-8"
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = chunks if chunks is not None else [text.encode("utf-8")]
        self._raw = None

    def iter_content(self, chunk_size = 1):
        yield from self._chunks

    @property
    def raw(self):
        """The urllib3 response the probe reads through.

        It calls `read1`, which returns whatever ONE socket read produced, so that a
        deadline check sits between every read rather than only between whole chunks.
        `iter_content` is kept because it is what `requests` exposes and the fake would
        otherwise drift from the real object.
        """
        if self._raw is None:
            self._raw = _FakeRaw(self._chunks)
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _install_fake_requests(
    monkeypatch,
    text,
    chunks = None,
):
    module = types.ModuleType("requests")
    module.compat = types.SimpleNamespace(urljoin = lambda base, url: url)
    module.get = lambda url, timeout = None, stream = False, allow_redirects = True: (
        _FakeResponse(text, chunks)
    )
    monkeypatch.setitem(sys.modules, "requests", module)


def _install_fake_vllm_absent(monkeypatch, namespace):
    """vllm >= 0.12.0 returns early from __get_model_name, leaving the probe unreachable."""
    monkeypatch.delitem(sys.modules, "vllm", raising = False)
    fake = types.ModuleType("importlib")
    fake.util = types.SimpleNamespace(find_spec = lambda name: None)
    namespace["importlib"] = fake


def _load_resolver(installed_source):
    """Stand-in for loader_utils' module globals, built from `installed_source`."""
    from unsloth_zoo.utils import Version

    # loader_utils imports this from .mapper; _get_new_mapper derives the fetched tables
    # with it, so the stand-in globals need it or the probe NameErrors into its own bare
    # except and returns empty tables.
    from unsloth.models.mapper import build_mappers

    mapper_ns = {}
    exec(compile(installed_source, "mapper.py", "exec"), mapper_ns)

    namespace = {
        "INT_TO_FLOAT_MAPPER": mapper_ns["INT_TO_FLOAT_MAPPER"],
        "FLOAT_TO_INT_MAPPER": mapper_ns["FLOAT_TO_INT_MAPPER"],
        "MAP_TO_UNSLOTH_16bit": mapper_ns["MAP_TO_UNSLOTH_16bit"],
        "FLOAT_TO_FP8_BLOCK_MAPPER": mapper_ns["FLOAT_TO_FP8_BLOCK_MAPPER"],
        "FLOAT_TO_FP8_ROW_MAPPER": mapper_ns["FLOAT_TO_FP8_ROW_MAPPER"],
        "build_mappers": build_mappers,
        # Imported from loader_utils rather than rebuilt, so a new helper added there
        # cannot silently drop out of this stand-in and make the probe look broken.
        "_MAPPER_HELPERS": _loader_utils_globals()["_MAPPER_HELPERS"],
        "SUPPORTS_FOURBIT": True,
        "transformers_version": Version("4.57.6"),
        "Version": Version,
        "os": os,
    }
    with open(os.path.join(_MODELS, "loader_utils.py"), encoding = "utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) in ("BAD_MAPPINGS", "_OFFLINE_ENV_VALUES", "_OFFLINE_ENV_KEYS")
            for t in node.targets
        ):
            exec(compile(ast.Module([node], []), "<assign>", "exec"), namespace)
        elif isinstance(node, ast.FunctionDef) and (
            node.name in _WANTED or node.name == "_env_says_offline"
        ):
            exec(compile(ast.Module([node], []), node.name, "exec"), namespace)
    return namespace


def test_probe_answers_for_an_fp8_repo_only_the_fetched_mapper_knows(monkeypatch):
    installed = _mapper_source()
    namespace = _load_resolver(installed)
    installed_block = namespace["FLOAT_TO_FP8_BLOCK_MAPPER"]
    installed_row = namespace["FLOAT_TO_FP8_ROW_MAPPER"]
    assert _NEW_OFFICIAL.lower() not in installed_block, "the installed table must not know it"

    _install_fake_requests(monkeypatch, _with_extra_fp8_model(installed))
    _install_fake_vllm_absent(monkeypatch, namespace)

    try:
        resolved = namespace["get_model_name"](
            _NEW_OFFICIAL, load_in_4bit = False, load_in_fp8 = "block"
        )
    except NotImplementedError as error:
        assert "not supported in your current Unsloth version" in str(error)
    else:
        raise AssertionError(
            f"a fetched-only fp8 repo must raise the upgrade error, got {resolved!r}"
        )

    # Answering must not have adopted the fetched tables.
    assert namespace["FLOAT_TO_FP8_BLOCK_MAPPER"] is installed_block
    assert namespace["FLOAT_TO_FP8_ROW_MAPPER"] is installed_row
    assert _NEW_OFFICIAL.lower() not in namespace["FLOAT_TO_FP8_BLOCK_MAPPER"]


def test_probe_survives_a_fetched_mapper_without_the_fp8_tables(monkeypatch):
    installed = _mapper_source()
    namespace = _load_resolver(installed)
    _install_fake_requests(monkeypatch, _without_fp8_tables(installed))

    int_to_float, float_to_int, map_to_16bit = namespace["_get_new_mapper"]()[:3]

    assert (
        int_to_float and float_to_int and map_to_16bit
    ), "a fetched mapper.py without the fp8 tables must not take the 4bit upgrade check down"


def test_fbgemm_prefers_the_row_table_over_the_block_one(monkeypatch):
    """With FBGEMM, `load_in_fp8 = True` must resolve row-scaled, not blockwise."""
    monkeypatch.setenv("UNSLOTH_HAS_FBGEMM", "1")
    namespace = _load_resolver(_mapper_source())
    row = namespace["FLOAT_TO_FP8_ROW_MAPPER"]
    block = namespace["FLOAT_TO_FP8_BLOCK_MAPPER"]

    key = next(k for k in row if k in block and row[k] != block[k])
    resolved = namespace["get_model_name"](key, load_in_4bit = False, load_in_fp8 = True)

    assert resolved == row[key], (
        f"FBGEMM must take the row branch for {key!r}, got {resolved!r} "
        f"(the blockwise answer is {block[key]!r})"
    )


def test_probe_answers_for_a_row_only_repo_the_fetched_mapper_knows(monkeypatch):
    """The row half of the probe needs the FETCHED row table, same as the block half."""
    monkeypatch.setenv("UNSLOTH_HAS_FBGEMM", "1")
    installed = _mapper_source()
    namespace = _load_resolver(installed)
    installed_row = namespace["FLOAT_TO_FP8_ROW_MAPPER"]
    key = _ROW_ONLY.lower()
    assert key not in installed_row, "the installed row table must not know it"
    assert key not in namespace["FLOAT_TO_FP8_BLOCK_MAPPER"], "no block entry, or block answers"

    _install_fake_requests(monkeypatch, _with_row_only_fp8_model(installed))
    _install_fake_vllm_absent(monkeypatch, namespace)

    try:
        resolved = namespace["get_model_name"](_ROW_ONLY, load_in_4bit = False, load_in_fp8 = True)
    except NotImplementedError as error:
        assert "not supported in your current Unsloth version" in str(error)
    else:
        raise AssertionError(
            f"a fetched-only row-scaled repo must raise the upgrade error, got {resolved!r}"
        )

    assert namespace["FLOAT_TO_FP8_ROW_MAPPER"] is installed_row


def test_a_fetched_mapper_that_uses_update_still_installs_its_entries(monkeypatch):
    """`.update({...})` adds entries exactly as the subscript spelling does.

    Only the subscript spelling was read, so a newer mapper.py written the ordinary
    way left the row table empty and the models it had just added got no upgrade
    notice. The subscript case is asserted alongside it, so the test cannot pass by
    the probe having stopped reading either spelling.
    """
    installed = _mapper_source()
    namespace = _load_resolver(installed)

    both = installed + (
        f"\nFLOAT_TO_FP8_ROW_MAPPER.update({{{_ROW_ONLY.lower()!r}: {_NEW_ROW!r}}})\n"
        f"FLOAT_TO_FP8_ROW_MAPPER[{_NEW_OFFICIAL.lower()!r}] = {_NEW_ROW!r}\n"
    )
    _install_fake_requests(monkeypatch, both)

    fetched = namespace["_get_new_mapper"]()
    row_table = fetched[4]
    assert row_table.get(_ROW_ONLY.lower()) == _NEW_ROW, f"update() entry missing: {row_table}"
    assert row_table.get(_NEW_OFFICIAL.lower()) == _NEW_ROW, f"subscript entry missing: {row_table}"


def test_update_on_a_name_the_probe_does_not_export_is_ignored(monkeypatch):
    """The receiver has to name one of the five tables, or nothing is read from it."""
    installed = _mapper_source()
    namespace = _load_resolver(installed)

    _install_fake_requests(
        monkeypatch,
        installed + f"\nSOMETHING_ELSE.update({{{_ROW_ONLY.lower()!r}: {_NEW_ROW!r}}})\n",
    )
    fetched = namespace["_get_new_mapper"]()
    assert _ROW_ONLY.lower() not in fetched[4]
