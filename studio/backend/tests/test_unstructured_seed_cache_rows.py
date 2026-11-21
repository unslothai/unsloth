# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

_BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND))
_PKG = (
    _BACKEND
    / "plugins"
    / "data-designer-unstructured-seed"
    / "src"
    / "data_designer_unstructured_seed"
)


def _load_chunking():
    spec = importlib.util.spec_from_file_location("_uss_chunking_under_test", _PKG / "chunking.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chunking = _load_chunking()


def _oracle_preview(cached, preview_size):
    """The previous cache-hit path: every row to a dict, then the round-robin."""
    rows = pd.read_parquet(cached).to_dict(orient = "records")
    return chunking._round_robin_preview(rows, preview_size), len(rows)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(chunking, "_CACHE_DIR", tmp_path / "cache")
    return tmp_path


def _entries(tmp_path, texts):
    out = []
    for i, (name, text) in enumerate(texts):
        p = tmp_path / f"f{i}.extracted.txt"
        p.write_text(text, encoding = "utf-8")
        out.append((p, name))
    return out


def _text(seed, words):
    return " ".join(f"w{seed}_{j}{'.' if j % 11 == 10 else ''}" for j in range(words))


_LAYOUTS = {
    "one_source": [("a.pdf", _text(1, 900))],
    "three_sources": [("a.pdf", _text(1, 900)), ("b.docx", _text(2, 300)), ("c.txt", _text(3, 40))],
    "repeated_name": [("a.pdf", _text(1, 500)), ("b.pdf", _text(2, 500)), ("a.pdf", _text(3, 500))],
    "many_small": [(f"s{i}.txt", _text(i, 60 + 37 * i)) for i in range(9)],
}
_SIZES = [-3, 0, 1, 2, 3, 5, 10, 1000]


def _counting_to_dict(monkeypatch):
    converted = []
    real = pd.DataFrame.to_dict

    def spy(self, *args, **kwargs):
        converted.append(len(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_dict", spy)
    return converted


@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_cached_preview_matches_full_conversion(cache, layout, monkeypatch):
    entries = _entries(cache, _LAYOUTS[layout])
    kwargs = dict(file_entries = entries, chunk_size = 120, chunk_overlap = 20)
    cold_preview = {p: chunking.build_multi_file_preview(preview_size = p, **kwargs) for p in _SIZES}
    cached, _ = chunking.materialize_multi_file_unstructured_seed(**kwargs)
    before = cached.read_bytes()
    total = len(pd.read_parquet(cached))
    sources = pd.read_parquet(cached)["source_file"].nunique()
    for p in _SIZES:
        converted = _counting_to_dict(monkeypatch)
        got = chunking.build_multi_file_preview(preview_size = p, **kwargs)
        monkeypatch.undo()
        monkeypatch.setattr(chunking, "_CACHE_DIR", cache / "cache")
        assert got == _oracle_preview(cached, p) == cold_preview[p]
        assert sum(converted) <= max(0, p) * sources
        if p > 0 and total > p * sources:
            assert sum(converted) < total
    assert cached.read_bytes() == before


@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_dataset_uri_path_skips_rows_only_on_a_cache_hit(cache, layout):
    entries = _entries(cache, _LAYOUTS[layout])
    kwargs = dict(file_entries = entries, chunk_size = 120, chunk_overlap = 20)
    cold_path, cold_rows = chunking.materialize_multi_file_unstructured_seed(
        include_rows = False, **kwargs
    )
    assert cold_rows and cold_rows == pd.read_parquet(cold_path).to_dict(orient = "records")
    warm_path, warm_rows = chunking.materialize_multi_file_unstructured_seed(
        include_rows = False, **kwargs
    )
    assert warm_path == cold_path and warm_rows == []
    full_path, full_rows = chunking.materialize_multi_file_unstructured_seed(**kwargs)
    assert full_path == cold_path and full_rows == cold_rows


def _write_cache(cache, df):
    entries = _entries(cache, [("a.pdf", _text(1, 50))])
    kwargs = dict(file_entries = entries, chunk_size = 120, chunk_overlap = 20)
    cached, _ = chunking.materialize_multi_file_unstructured_seed(**kwargs)
    df.to_parquet(cached, index = False)
    return cached, kwargs


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"chunk_text": ["x", "y", "z", "w"], "source_file": ["a", None, "a", None]}),
        pd.DataFrame(
            {
                "chunk_text": ["x", "y", "z", "w"],
                "source_file": ["a", float("nan"), float("nan"), "b"],
            }
        ),
        pd.DataFrame({"chunk_text": ["x", "y", "z"], "source_file": [1, 1, 2]}),
        pd.DataFrame({"chunk_text": ["x", "y", "z"]}),
        pd.DataFrame(
            {
                "chunk_text": ["x", "y", "z", "q"],
                "source_file": ["a", "a", "a", "b"],
                "n": [1, 2, 3, 4],
            }
        ),
        pd.DataFrame({"chunk_text": [None, "y", "z", "q"], "source_file": ["a", "a", "b", "b"]}),
        pd.DataFrame(
            {"chunk_text": pd.Series([], dtype = object), "source_file": pd.Series([], dtype = object)}
        ),
    ],
    ids = [
        "none_source",
        "nan_source",
        "int_source",
        "no_source",
        "extra_column",
        "none_text",
        "empty",
    ],
)
def test_unusual_caches_give_the_same_preview(cache, frame):
    cached, kwargs = _write_cache(cache, frame)
    for p in _SIZES:
        got = chunking.build_multi_file_preview(preview_size = p, **kwargs)
        want = _oracle_preview(cached, p)
        assert repr(got) == repr(want)


def test_corrupt_cache_raises_the_same_error_on_every_path(cache):
    entries = _entries(cache, [("a.pdf", _text(1, 50))])
    kwargs = dict(file_entries = entries, chunk_size = 120, chunk_overlap = 20)
    cached, _ = chunking.materialize_multi_file_unstructured_seed(**kwargs)
    cached.write_bytes(b"PAR1 not a parquet file")
    with pytest.raises(Exception) as want:
        pd.read_parquet(cached)
    calls = [
        lambda: chunking.materialize_multi_file_unstructured_seed(include_rows = False, **kwargs),
        lambda: chunking.materialize_multi_file_unstructured_seed(**kwargs),
        lambda: chunking.build_multi_file_preview(preview_size = 3, **kwargs),
    ]
    for call in calls:
        with pytest.raises(type(want.value)) as got:
            call()
        assert str(got.value) == str(want.value)


def test_seed_reader_asks_for_the_path_only():
    tree = ast.parse((_PKG / "impl.py").read_text(encoding = "utf-8"))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "materialize_multi_file_unstructured_seed"
    ]
    assert len(calls) == 1
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert isinstance(kw.get("include_rows"), ast.Constant) and kw["include_rows"].value is False
