# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Simulation suite: edge cases for the /v1/models servability cache.

The cache sits on a request path that several clients poll, so it has to be correct
under concurrency, correct when the catalog is replaced, and must never let a stale
residency answer through. A wrong answer here is worse than the latency it saves:
it would advertise a model the server cannot serve, or hide one it can.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import routes.inference as inf  # noqa: E402


@pytest.fixture(autouse = True)
def _clean_cache():
    def _reset():
        inf._SERVABLE_SCAN_CACHE["at"] = None
        inf._SERVABLE_SCAN_CACHE["rows"] = []
        inf._SERVABLE_SCAN_CACHE["catalog"] = None

    _reset()
    yield
    _reset()


def _catalog(n = 3, tag = "m"):
    return [
        SimpleNamespace(id = f"repo/{tag}{i}", path = f"/models/{tag}{i}", task = None) for i in range(n)
    ]


@pytest.fixture
def stub(monkeypatch):
    counts = {"servable": 0, "resident": 0}

    def _servable(info):
        counts["servable"] += 1
        return (True, ("Q4_K_M",))

    monkeypatch.setattr("core.inference.local_model_resolver.local_servable_model", _servable)
    monkeypatch.setattr("core.inference.local_model_resolver.local_load_dir", lambda p: p)

    def _resident(key, **kw):
        counts["resident"] += 1
        return False

    monkeypatch.setattr(inf, "_resolves_to_resident", _resident)
    return counts


def test_concurrent_callers_do_not_corrupt_the_cache(stub):
    catalog = _catalog(5)
    results: list[int] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(16)

    def _go():
        try:
            barrier.wait()
            for _ in range(20):
                rows = inf._servable_catalog_rows(catalog, 111.0)
                results.append(len(rows))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target = _go) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout = 60)
    assert not errors, f"concurrent access raised: {errors}"
    assert set(results) == {5}, "every caller must see the whole catalog"


def test_a_replaced_catalog_is_never_served_from_the_old_entry(stub):
    first, second = _catalog(2, "a"), _catalog(3, "b")
    rows = inf._servable_catalog_rows(first, 111.0)
    assert [r[0].id for r in rows] == ["repo/a0", "repo/a1"]
    # Same stamp, different catalog object: identity must defeat the stamp.
    rows = inf._servable_catalog_rows(second, 111.0)
    assert [r[0].id for r in rows] == ["repo/b0", "repo/b1", "repo/b2"]


def test_the_zero_stamp_of_a_fresh_process_does_not_pin_the_cache(stub):
    # _CATALOG_CACHE["at"] starts at 0.0 and only advances inside
    # _cached_local_catalog. A caller that supplies a catalog from anywhere else
    # keeps that 0.0, so the stamp alone cannot be trusted.
    first, second = _catalog(1, "a"), _catalog(1, "b")
    assert inf._servable_catalog_rows(first, 0.0)[0][0].id == "repo/a0"
    assert inf._servable_catalog_rows(second, 0.0)[0][0].id == "repo/b0"


def test_residency_is_never_cached(stub):
    catalog = _catalog(2)
    for _ in range(10):
        inf._servable_catalog_rows(catalog, 111.0)
    assert stub["servable"] == 2, "the scan is cached"
    assert stub["resident"] == 20, "residency is not"


def test_residency_flips_are_visible_immediately(monkeypatch):
    monkeypatch.setattr(
        "core.inference.local_model_resolver.local_servable_model",
        lambda info: (True, ()),
    )
    monkeypatch.setattr("core.inference.local_model_resolver.local_load_dir", lambda p: p)
    flag = {"resident": False}
    monkeypatch.setattr(inf, "_resolves_to_resident", lambda key, **kw: flag["resident"])

    catalog = _catalog(1)
    assert inf._servable_catalog_rows(catalog, 111.0)[0][3] is False
    flag["resident"] = True  # a /load happened
    assert inf._servable_catalog_rows(catalog, 111.0)[0][3] is True


def test_an_empty_catalog_is_cached_without_confusing_a_miss(stub):
    assert inf._servable_catalog_rows([], 111.0) == []
    assert inf._servable_catalog_rows([], 111.0) == []
    assert stub["servable"] == 0


def test_no_stamp_never_populates_the_cache(stub):
    catalog = _catalog(2)
    inf._servable_catalog_rows(catalog)
    assert inf._SERVABLE_SCAN_CACHE["at"] is None
    inf._servable_catalog_rows(catalog)
    assert stub["servable"] == 4


def test_media_and_stt_tasks_stay_excluded(monkeypatch):
    monkeypatch.setattr(
        "core.inference.local_model_resolver.local_servable_model",
        lambda info: (True, ()),
    )
    monkeypatch.setattr("core.inference.local_model_resolver.local_load_dir", lambda p: p)
    monkeypatch.setattr(inf, "_resolves_to_resident", lambda key, **kw: False)
    catalog = [
        SimpleNamespace(id = "repo/text", path = "/m/text", task = None),
        SimpleNamespace(id = "repo/stt", path = "/m/stt", task = inf._STT_MODEL_TASK),
        SimpleNamespace(id = "repo/tts", path = "/m/tts", task = inf._TTS_MODEL_TASK),
    ]
    rows = inf._servable_catalog_rows(catalog, 111.0)
    assert [r[0].id for r in rows] == ["repo/text"]


def test_a_raising_resolver_is_not_cached_as_an_empty_catalog(monkeypatch):
    # If the scan blows up, the failure must propagate rather than silently
    # caching "this server can serve nothing" for the life of the catalog.
    monkeypatch.setattr(
        "core.inference.local_model_resolver.local_servable_model",
        lambda info: (_ for _ in ()).throw(RuntimeError("scan failed")),
    )
    monkeypatch.setattr("core.inference.local_model_resolver.local_load_dir", lambda p: p)
    monkeypatch.setattr(inf, "_resolves_to_resident", lambda key, **kw: False)
    with pytest.raises(RuntimeError):
        inf._servable_catalog_rows(_catalog(1), 111.0)
    assert inf._SERVABLE_SCAN_CACHE["at"] is None, "a failed scan must not be cached"


def test_large_catalog_stays_correct(stub):
    catalog = _catalog(500)
    rows = inf._servable_catalog_rows(catalog, 111.0)
    assert len(rows) == 500
    assert stub["servable"] == 500
    rows = inf._servable_catalog_rows(catalog, 111.0)
    assert len(rows) == 500
    assert stub["servable"] == 500, "second call must be free"
