# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The HNSW tier's index, delta and files, without SQLite: this file runs on every platform the wheels cover."""

import os

import pytest

np = pytest.importorskip("numpy")
if os.environ.get("CI"):
    # CI installs studio.txt: a missing or unloadable usearch wheel there is a packaging failure, not a skip.
    import usearch  # noqa: F401
else:
    pytest.importorskip("usearch")

from core.rag import vec_ann


def _clustered(
    n,
    dim,
    seed = 0,
):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((max(8, n // 200), dim)).astype(np.float32)
    x = centers[rng.integers(0, len(centers), n)] + 0.5 * rng.standard_normal((n, dim)).astype(
        np.float32
    )
    return x * rng.uniform(0.5, 3.0, (n, 1)).astype(np.float32)


def _exact(rows, keys, q, k):
    unit = rows / np.linalg.norm(rows, axis = 1, keepdims = True)
    order = np.argsort(-(unit @ (q / np.linalg.norm(q))), kind = "stable")[:k]
    return [int(keys[i]) for i in order]


def _built(rows, keys):
    index = vec_ann.new_index(rows.shape[1])
    vec_ann.add(np, index, keys, rows, threads = 2)
    index.expansion_search = vec_ann.EXPANSION_SEARCH
    empty = np.empty(0, np.int64), np.empty((0, rows.shape[1]), np.float32)
    return vec_ann.Ann(index, len(index), None, *empty)


def test_search_recalls_the_exact_neighbours():
    rows = _clustered(5000, 32)
    keys = np.arange(5000) * 3 + 7
    ann = _built(rows, keys)
    rng = np.random.default_rng(1)
    hits = 0
    for q in rows[rng.integers(0, 5000, 20)] + 0.01:
        got, dist = vec_ann.search(np, ann, q / np.linalg.norm(q), 30)
        assert list(dist) == sorted(dist)
        hits += len(set(got.tolist()) & set(_exact(rows, keys, q, 30)))
    assert hits / (20 * 30) >= 0.99


def test_unscorable_rows_are_never_indexed():
    rows = _clustered(200, 16)
    rows[3] = np.nan
    rows[4] = 0.0
    rows[5, 0] = np.inf
    ann = _built(rows, np.arange(200))
    assert ann.n == 197
    got, _ = vec_ann.search(np, ann, rows[6] / np.linalg.norm(rows[6]), 200)
    assert not {3, 4, 5} & set(got.tolist())


def test_the_delta_is_searched_exactly_with_the_index():
    rows = _clustered(2000, 16)
    ann = _built(rows, np.arange(2000))
    new = _clustered(50, 16, seed = 9)
    keys, unit = vec_ann.unit_rows(np, np.arange(5000, 5050), new)
    merged = vec_ann.with_delta(np, ann, keys, unit)
    assert merged.size == 2050 and ann.size == 2000
    q = new[17] / np.linalg.norm(new[17])
    got, dist = vec_ann.search(np, merged, q, 10)
    assert got[0] == 5017 and abs(dist[0]) < 1e-6
    want = _exact(np.concatenate([rows, new]), list(range(2000)) + list(range(5000, 5050)), q, 10)
    assert len(set(got.tolist()) & set(want)) >= 9


def test_saved_generation_reloads_mapped_with_the_same_answers(tmp_path):
    rows = _clustered(3000, 24)
    ann = _built(rows, np.arange(3000))
    base = str(tmp_path / "g-1-0-3000")
    vec_ann.save(ann.index, base, {"dim": 24, "n": ann.n})
    view, meta = vec_ann.load(base, 24)
    assert meta["n"] == 3000 and len(view) == 3000
    q = rows[11] / np.linalg.norm(rows[11])
    reloaded = vec_ann.Ann(view, len(view), base + ".usearch", ann.delta_keys, ann.delta_mat)
    assert (
        vec_ann.search(np, reloaded, q, 20)[0].tolist()
        == vec_ann.search(np, ann, q, 20)[0].tolist()
    )
    del view, reloaded


def test_mismatched_or_corrupt_generations_are_rejected(tmp_path):
    rows = _clustered(300, 8)
    ann = _built(rows, np.arange(300))
    base = str(tmp_path / "g-1-0-300")
    vec_ann.save(ann.index, base, {"dim": 8, "n": ann.n})
    assert vec_ann.load(base, 16) is None
    with open(base + ".usearch", "wb") as fh:
        fh.write(b"junk" * 16)
    assert vec_ann.load(base, 8) is None
    assert vec_ann.load(str(tmp_path / "missing"), 8) is None


def test_remove_files_deletes_every_part(tmp_path):
    rows = _clustered(100, 8)
    base = str(tmp_path / "g-1-0-100")
    vec_ann.save(_built(rows, np.arange(100)).index, base, {"dim": 8, "n": 100})
    assert vec_ann.remove_files(base)
    assert not list(tmp_path.iterdir())


def test_rows_added_with_the_build_threads_stay_reachable():
    from core.rag import vec_cache

    # Parallel inserts strand nodes in usearch 2.26; the build's thread count must not.
    rows = _clustered(6000, 16, seed = 5)
    index = vec_ann.new_index(16)
    vec_ann.add(np, index, np.arange(6000), rows, threads = vec_cache._ann_threads())
    index.expansion_search = vec_ann.EXPANSION_SEARCH
    unit = rows / np.linalg.norm(rows, axis = 1, keepdims = True)
    sample = np.random.default_rng(2).choice(6000, 300, replace = False)
    assert all(int(index.search(unit[i], 1).keys[0]) == i for i in sample)
