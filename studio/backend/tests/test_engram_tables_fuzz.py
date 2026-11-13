# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Property fuzz for swapping a GGUF quant's Engram table for a standalone one."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("hypothesis")
from hypothesis import assume, given, settings, strategies as st  # noqa: E402

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import test_llama_cpp_placement  # noqa: E402,F401 -- stubs jwt for routes.inference

from core.inference import llama_cpp as lc  # noqa: E402
from core.inference.engram_tables import (  # noqa: E402
    bundled_table,
    engram_table_precision,
    engram_tables,
    files_llama_reads,
)
from hub.utils.gguf_plan import build_gguf_variant_plans  # noqa: E402
from models.inference import LoadRequest  # noqa: E402
from test_engram_tables import F16, IQ4_NL, PLE, _gguf  # noqa: E402
from utils.openai_auto_switch_settings import (  # noqa: E402
    model_override_load_kwargs,
    normalize_model_override,
)

FUZZ = settings(max_examples = 300, deadline = None, derandomize = True, database = None)
PRECISION = st.from_regex(r"[A-Za-z0-9][A-Za-z0-9_]{0,10}", fullmatch = True)
MODEL = st.from_regex(r"[A-Za-z0-9][A-Za-z0-9._-]{0,20}", fullmatch = True)
SIZE = st.integers(min_value = 1, max_value = 300 * 10**9)


@st.composite
def repos(draw):
    """A repo listing: a split quant whose shard sizes may collide with the standalone tables."""
    model = draw(MODEL)
    precisions = draw(st.lists(PRECISION, min_size = 1, max_size = 5, unique_by = str.upper))
    sizes = draw(st.lists(SIZE, min_size = len(precisions), max_size = len(precisions), unique = True))
    tables = [(f"engram/{model}-engram-{p}.gguf_file", s) for p, s in zip(precisions, sizes)]
    count = draw(st.integers(min_value = 2, max_value = 6))
    shard_sizes = draw(
        st.lists(st.one_of(SIZE, st.sampled_from(sizes)), min_size = count, max_size = count)
    )
    shards = [
        (f"UD-IQ1_S/{model}-UD-IQ1_S-{i:05d}-of-{count:05d}.gguf", s)
        for i, s in enumerate(shard_sizes, start = 1)
    ]
    return model, tables, shards


def asked(data, tables):
    """A precision as a caller would send it: mostly a published one in any case, else anything."""
    published = [engram_table_precision(n) for n, _ in tables]
    pick = data.draw(st.one_of(st.sampled_from(published), PRECISION, st.none()))
    if pick and data.draw(st.booleans()):
        pick = data.draw(st.sampled_from([pick.lower(), pick.upper(), f" {pick} "]))
    return pick


@FUZZ
@given(MODEL, PRECISION, st.sampled_from([".gguf_file", ".gguf"]))
def test_the_table_name_yields_its_precision(model, precision, suffix):
    assert engram_table_precision(f"engram/{model}-engram-{precision}{suffix}") == precision


@FUZZ
@given(st.text(max_size = 80))
def test_any_other_name_is_not_a_table(name):
    precision = engram_table_precision(name)
    if precision is not None:
        normalized = name.replace("\\", "/")
        assert "engram/" in normalized and "-engram-" in normalized
        assert normalized.lower().endswith((".gguf", ".gguf_file"))


@FUZZ
@given(repos())
def test_the_listing_keeps_one_table_per_precision_largest_first(repo):
    _, tables, shards = repo
    listed = engram_tables(
        tables + shards + [("engram/README.md", 1), ("engram/x-engram-Z.gguf_file", 0)]
    )
    assert [t.size_bytes for t in listed] == sorted((t.size_bytes for t in listed), reverse = True)
    assert len({t.precision.upper() for t in listed}) == len(listed) == len(tables)


@FUZZ
@given(repos())
def test_the_bundled_shard_is_the_one_shard_sized_like_a_table(repo):
    _, tables, shards = repo
    found = bundled_table(shards, engram_tables(tables))
    sizes = {s for _, s in tables}
    matching = [name for name, size in shards if size in sizes]
    if len(matching) == 1:
        assert found is not None and found[0] == matching[0]
        assert found[1].size_bytes == dict(shards)[found[0]]
    else:
        assert found is None


@FUZZ
@given(repos(), st.data())
def test_a_swapped_plan_fetches_the_table_instead_of_the_shard(repo, data):
    _, tables, shards = repo
    precision = asked(data, tables)
    siblings = [
        SimpleNamespace(rfilename = name, size = size, lfs = {"sha256": f"{i:064x}"})
        for i, (name, size) in enumerate(tables + shards)
    ]
    plan = build_gguf_variant_plans(siblings)["ud-iq1_s"]
    swapped = plan.with_engram(precision)
    table = plan.engram_table(precision)
    if table is None or plan.engram_shard is None or table.size == plan.engram_shard.size:
        assert swapped is plan
        return
    paths = [f.path for f in swapped.expected_files]
    assert plan.engram_shard.path not in paths
    assert paths.count(table.path) == 1
    assert list(swapped.target_filenames) == paths
    assert swapped.download_size_bytes == sum(f.size for f in swapped.expected_files)
    assert swapped.required_hashes == {f.sha256 for f in swapped.expected_files if f.sha256}
    # Swapping again, to the same or another table, keeps exactly one table.
    assert swapped.with_engram(precision).expected_files == swapped.expected_files
    other = asked(data, tables)
    again = [f.path for f in swapped.with_engram(other).expected_files]
    assert sum(1 for p in again if engram_table_precision(p)) <= 1


@FUZZ
@given(st.lists(st.sampled_from(["model", "table", "absent"]), min_size = 1, max_size = 6))
def test_llama_reads_every_model_shard_and_the_external_table(kinds):
    kinds = ["model", *kinds]
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, kind in enumerate(kinds, start = 1):
            path = Path(tmp) / f"m-{i:05d}-of-{len(kinds):05d}.gguf"
            if kind == "model":
                path.write_bytes(
                    _gguf([(f"blk.{i}.w", [8, 8], F16)], "qwen4exp" if i == 1 else None)
                )
            elif kind == "table":
                path.write_bytes(_gguf([(PLE, [160, 1024], IQ4_NL)]))
            paths.append(str(path))
        ple = str(Path(tmp) / "engram.gguf_file")
        read = files_llama_reads(paths, ple)
        assert read[0] == paths[0] and read[-1] == ple
        assert read[1:-1] == [p for p, k in zip(paths[1:], kinds[1:]) if k == "model"]
        assert files_llama_reads(paths, None) == paths


@FUZZ
@given(
    st.one_of(
        st.none(), st.text(max_size = 40), st.integers(), st.booleans(), st.just([]), st.just({})
    )
)
def test_a_load_request_keeps_only_a_clean_precision(value):
    try:
        request = LoadRequest(model_path = "unsloth/X-GGUF", engram_precision = value)
    except Exception:  # noqa: BLE001 -- refusing a malformed value is the other allowed outcome
        return
    got = request.engram_precision
    assert got is None or (
        got == got.strip()
        and 1 <= len(got) <= 32
        and got.replace("_", "").isalnum()
        and got.isascii()
    )
    if isinstance(value, str) and not value.strip():
        assert got is None
    # A non-string is never read as "the built-in table".
    assert value is None or isinstance(value, str)


@FUZZ
@given(st.one_of(st.none(), st.text(max_size = 40), st.integers(), st.booleans()))
def test_the_override_store_keeps_only_a_clean_precision(value):
    entry = normalize_model_override({"engram_precision": value})
    stored = entry.get("engram_precision")
    if stored is not None:
        assert stored.isascii() and stored.replace("_", "").isalnum() and 1 <= len(stored) <= 32
        assert model_override_load_kwargs(entry, is_gguf = True)["engram_precision"] == stored
    assert "engram_precision" not in model_override_load_kwargs(entry, is_gguf = False)


@FUZZ
@given(repos(), st.data())
def test_resolving_a_precision_swaps_names_the_own_table_or_refuses(repo, data):
    _, tables, shards = repo
    precision = data.draw(st.one_of(st.just(asked(data, tables)), st.text(max_size = 12)))
    assume(precision and precision.strip())
    names = [n for n, _ in shards]
    listing = tables + shards

    class _Api:
        def __init__(self, token = None):
            pass

        def model_info(
            self,
            repo_id,
            files_metadata = False,
        ):
            return SimpleNamespace(
                siblings = [SimpleNamespace(rfilename = n, size = s) for n, s in listing]
            )

    import huggingface_hub

    saved = (lc._resolve_variant_gguf_files, huggingface_hub.HfApi)
    lc._resolve_variant_gguf_files = lambda *a: (names[0], names[1:])
    huggingface_hub.HfApi = _Api
    try:
        try:
            swap = lc._resolve_engram_swap("o/M-GGUF", "UD-IQ1_S", None, precision)
        except ValueError as e:
            assert "Engram table" in str(e)
            published = {p.upper() for p in (engram_table_precision(n) for n, _ in tables)}
            assert precision.strip().upper() not in published
            return
        bundled = bundled_table(shards, engram_tables(tables))
        if swap is None:
            assert bundled is not None and bundled[1].precision.upper() == precision.strip().upper()
        else:
            assert swap.table.precision.upper() == precision.strip().upper()
            assert swap.bundled_shard == (bundled[0] if bundled else None)
    finally:
        lc._resolve_variant_gguf_files, huggingface_hub.HfApi = saved
