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
import time
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
        inf._SERVABLE_SCAN_CACHE["entry"] = None

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
    assert inf._SERVABLE_SCAN_CACHE["entry"] is None
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
    assert inf._SERVABLE_SCAN_CACHE["entry"] is None, "a failed scan must not be cached"


def test_the_cache_entry_is_published_atomically(stub):
    # The fast path reads without the lock. Separate fields could be caught
    # half-replaced: old stamp and old catalog still matching while rows already held
    # the next catalog's rows, so an in-flight request got another catalog's models.
    first, second = _catalog(2, "a"), _catalog(3, "b")
    inf._servable_catalog_rows(first, 111.0)
    entry = inf._SERVABLE_SCAN_CACHE["entry"]
    assert entry is not None and len(entry) == 5, "the tuple must be one object"
    at, cached, generation, _scanned_at, rows = entry
    assert at == 111.0 and cached is first and len(rows) == 2
    assert isinstance(generation, tuple) and len(generation) == 3

    inf._servable_catalog_rows(second, 222.0)
    at, cached, _generation, _at2, rows = inf._SERVABLE_SCAN_CACHE["entry"]
    assert (at, cached is second, len(rows)) == (
        222.0,
        True,
        3,
    ), "stamp, catalog, generation and rows move together"


def test_residency_resolves_the_current_snapshot_each_call(monkeypatch):
    # A non-GGUF HF repo path resolves to a snapshot dir. A download can move that
    # pointer inside the catalog's 30s lifetime, and caching the resolved dir would
    # report a freshly loaded model as unloaded until the catalog expired.
    monkeypatch.setattr(
        "core.inference.local_model_resolver.local_servable_model",
        lambda info: (False, ()),  # non-GGUF, so residency goes through local_load_dir
    )
    snapshot = {"dir": "/models/m0/snapshots/old"}
    monkeypatch.setattr(
        "core.inference.local_model_resolver.local_load_dir", lambda p: snapshot["dir"]
    )
    seen: list[str] = []

    def _resident(key, **kw):
        seen.append(key)
        return key == "/models/m0/snapshots/new"

    monkeypatch.setattr(inf, "_resolves_to_resident", _resident)

    catalog = _catalog(1)
    assert inf._servable_catalog_rows(catalog, 111.0)[0][3] is False
    snapshot["dir"] = "/models/m0/snapshots/new"  # a download moved the pointer
    assert (
        inf._servable_catalog_rows(catalog, 111.0)[0][3] is True
    ), "the snapshot must be re-resolved per call, not cached with the scan"
    assert seen == ["/models/m0/snapshots/old", "/models/m0/snapshots/new"]


def test_large_catalog_stays_correct(stub):
    catalog = _catalog(500)
    rows = inf._servable_catalog_rows(catalog, 111.0)
    assert len(rows) == 500
    assert stub["servable"] == 500
    rows = inf._servable_catalog_rows(catalog, 111.0)
    assert len(rows) == 500
    assert stub["servable"] == 500, "second call must be free"


# ------------------------------------------------------- deletion during the catalog TTL


def test_a_deleted_model_leaves_the_listing_within_the_catalog_ttl(monkeypatch):
    """A delete invalidates the resolver, not _CATALOG_CACHE, so the catalog behind this
    cache can stay standing for the rest of its 30s TTL. Keying on the resolver
    generation is what stops the removed model being advertised for that window, which
    the per-request scan used to drop at once."""
    from core.inference import local_model_resolver as resolver

    catalog = _catalog(2, "d")
    gone: set[str] = set()

    def _servable(info):
        return None if info.path in gone else (True, ("Q4_K_M",))

    monkeypatch.setattr("core.inference.local_model_resolver.local_servable_model", _servable)
    monkeypatch.setattr("core.inference.local_model_resolver.local_load_dir", lambda p: p)
    monkeypatch.setattr(inf, "_resolves_to_resident", lambda key, **kw: False)

    assert [r[0].id for r in inf._servable_catalog_rows(catalog, 111.0)] == ["repo/d0", "repo/d1"]
    gone.add("/models/d1")
    # Same catalog, same stamp: without the generation this still answers from the cache.
    resolver.invalidate_index()
    assert [r[0].id for r in inf._servable_catalog_rows(catalog, 111.0)] == ["repo/d0"]


def test_an_additions_only_invalidation_also_refreshes_the_scan(monkeypatch):
    """A finished download invalidates additions-only, and a new quant must appear
    without waiting out the catalog TTL."""
    from core.inference import local_model_resolver as resolver

    catalog = _catalog(1, "q")
    quants = {"/models/q0": ("Q4_K_M",)}

    monkeypatch.setattr(
        "core.inference.local_model_resolver.local_servable_model",
        lambda info: (True, quants[info.path]),
    )
    monkeypatch.setattr("core.inference.local_model_resolver.local_load_dir", lambda p: p)
    monkeypatch.setattr(inf, "_resolves_to_resident", lambda key, **kw: False)

    assert inf._servable_catalog_rows(catalog, 222.0)[0][2] == ("Q4_K_M",)
    quants["/models/q0"] = ("Q8_0", "Q4_K_M")
    resolver.invalidate_index(additions_only = True)
    assert inf._servable_catalog_rows(catalog, 222.0)[0][2] == ("Q8_0", "Q4_K_M")


def test_an_invalidation_during_a_scan_is_not_stamped_in(monkeypatch):
    """The generation is read before the scan, so an invalidation that lands while the
    scan runs makes the stored entry stale rather than being cached as already seen."""
    from core.inference import local_model_resolver as resolver

    catalog = _catalog(1, "r")
    state = {"racing": True}

    def _servable(info):
        if state["racing"]:
            state["racing"] = False
            resolver.invalidate_index()
        return (True, ("Q4_K_M",))

    monkeypatch.setattr("core.inference.local_model_resolver.local_servable_model", _servable)
    monkeypatch.setattr("core.inference.local_model_resolver.local_load_dir", lambda p: p)
    monkeypatch.setattr(inf, "_resolves_to_resident", lambda key, **kw: False)

    inf._servable_catalog_rows(catalog, 333.0)
    entry = inf._SERVABLE_SCAN_CACHE["entry"]
    assert entry is not None
    assert entry[2] != resolver.index_generation(), "the racing scan must not read as fresh"


def test_the_generation_only_moves_on_invalidation(monkeypatch):
    """A quiet process must still get the cache: if the generation drifted on its own,
    every request would rescan and the fix would undo the performance work."""
    from core.inference import local_model_resolver as resolver

    before = resolver.index_generation()
    catalog = _catalog(3, "s")
    monkeypatch.setattr(
        "core.inference.local_model_resolver.local_servable_model",
        lambda info: (True, ("Q4_K_M",)),
    )
    monkeypatch.setattr("core.inference.local_model_resolver.local_load_dir", lambda p: p)
    calls = {"n": 0}

    def _resident(key, **kw):
        calls["n"] += 1
        return False

    monkeypatch.setattr(inf, "_resolves_to_resident", _resident)
    for _ in range(5):
        inf._servable_catalog_rows(catalog, 444.0)
    assert resolver.index_generation() == before
    # Residency is deliberately per call; the scan behind it ran once.
    assert calls["n"] == 15


# ----------------------------------------------- every signal servability depends on


def test_a_hub_cache_deletion_leaves_the_listing(monkeypatch):
    """deletion.py invalidates the HF cache scans and nothing else, so the resolver
    generation alone would keep advertising a deleted cached repo for the catalog TTL."""
    from hub.utils import inventory_scan

    catalog = _catalog(2, "h")
    gone: set[str] = set()
    monkeypatch.setattr(
        "core.inference.local_model_resolver.local_servable_model",
        lambda info: None if info.path in gone else (True, ("Q4_K_M",)),
    )
    monkeypatch.setattr("core.inference.local_model_resolver.local_load_dir", lambda p: p)
    monkeypatch.setattr(inf, "_resolves_to_resident", lambda key, **kw: False)

    assert len(inf._servable_catalog_rows(catalog, 555.0)) == 2
    gone.add("/models/h1")
    inventory_scan.invalidate_hf_cache_scans()
    assert [r[0].id for r in inf._servable_catalog_rows(catalog, 555.0)] == ["repo/h0"]


def test_a_hardware_redetect_reveals_newly_servable_checkpoints(monkeypatch):
    """local_servable_model decides non-GGUF servability from hardware.DEVICE. An Apple
    Silicon MLX self-repair flips CPU to MLX after startup, so an early /v1/models must
    not pin 'unservable' for the rest of the catalog TTL."""
    from utils.hardware import hardware as hw

    catalog = _catalog(1, "k")
    servable = {"ok": False}
    monkeypatch.setattr(
        "core.inference.local_model_resolver.local_servable_model",
        lambda info: (False, ()) if servable["ok"] else None,
    )
    monkeypatch.setattr("core.inference.local_model_resolver.local_load_dir", lambda p: p)
    monkeypatch.setattr(inf, "_resolves_to_resident", lambda key, **kw: False)

    assert inf._servable_catalog_rows(catalog, 666.0) == []
    servable["ok"] = True
    monkeypatch.setattr(hw, "DETECTION_GENERATION", hw.DETECTION_GENERATION + 1)
    assert len(inf._servable_catalog_rows(catalog, 666.0)) == 1


def test_the_generation_key_names_all_three_signals():
    """A missing counter would silently pin the key and undo the invalidation, so the
    shape is asserted rather than left to the two behavioural tests above."""
    from core.inference.local_model_resolver import index_generation
    from hub.utils.inventory_scan import hf_cache_scans_epoch
    from utils.hardware import hardware as hw

    assert inf._servability_generation() == (
        index_generation(),
        int(hf_cache_scans_epoch()),
        int(hw.DETECTION_GENERATION),
    )


def test_every_delete_branch_invalidates_the_scan():
    """An outputs/exports directory can be a registered scan folder, so a model deleted
    through delete_finetuned_model may be one /v1/models is advertising. Both successful
    branches count: deleting a single GGUF variant returns earlier than the full-model
    delete, and only the later one was covered at first."""
    import inspect

    from routes import models as models_route

    lines = inspect.getsource(models_route.delete_finetuned_model).split("\n")
    # The variant branch returns a multi-line dict, so match the status rather than a
    # one-line prefix; a mid-function `return {` with no status is an error path.
    successes = [
        n
        for n, line in enumerate(lines)
        if '"status": "deleted"' in line or '"deleted"' in line and "status" in line
    ]
    invalidations = [n for n, line in enumerate(lines) if "_invalidate_local_scans()" in line]
    assert len(successes) == 2, f"expected two success returns, found {len(successes)}"
    assert (
        len(invalidations) == 2
    ), f"expected one invalidation per success branch, found {len(invalidations)}"
    for at in successes:
        before = [n for n in invalidations if n < at]
        assert before, "a successful deletion branch reports success without invalidating"
        # Belongs to THIS branch: nothing else returns between the two.
        assert not [
            n for n in successes if before[-1] < n < at
        ], "the invalidation must belong to this branch, not to one above it"


def test_a_generation_bump_while_waiting_for_the_lock_is_not_accepted(monkeypatch):
    """Two callers miss together, one scans while the other queues on the lock, and a
    delete lands during that scan. The scanner stamps its entry with the generation it
    STARTED with, which is correct, so a waiter comparing against the value it captured
    before queueing would be handed rows examined before the delete."""
    from core.inference import local_model_resolver as resolver

    catalog = _catalog(1, "w")
    scanned = {"n": 0}

    def _servable(info):
        scanned["n"] += 1
        return (True, ("Q4_K_M",))

    monkeypatch.setattr("core.inference.local_model_resolver.local_servable_model", _servable)
    monkeypatch.setattr("core.inference.local_model_resolver.local_load_dir", lambda p: p)
    monkeypatch.setattr(inf, "_resolves_to_resident", lambda key, **kw: False)

    generation_at_queue = inf._servability_generation()
    stale_rows = [("scanned-before-the-delete",)]

    class _LockThatLosesTheRace:
        """Stands in for the wait: the other scanner publishes and a delete lands here."""

        def __enter__(self):
            inf._SERVABLE_SCAN_CACHE["entry"] = (
                777.0,
                catalog,
                generation_at_queue,
                time.monotonic(),
                stale_rows,
            )
            resolver.invalidate_index()
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(inf, "_SERVABLE_SCAN_CACHE_LOCK", _LockThatLosesTheRace())
    rows = inf._servable_catalog_rows(catalog, 777.0)
    assert rows != stale_rows, "the waiter accepted rows scanned before the delete"
    assert scanned["n"] > 0, "it must have rescanned rather than reused the entry"


def test_the_invalidation_helper_stays_off_the_event_loop():
    """invalidate_index takes the resolver lock and _index() holds it across a full
    multi-root filesystem scan, so an async route calling it inline would stall unrelated
    requests and in-flight inference streams behind a rebuild."""
    import asyncio as _asyncio
    import inspect

    from routes import models as models_route

    assert _asyncio.iscoroutinefunction(models_route._invalidate_local_scans)
    source = inspect.getsource(models_route._invalidate_local_scans)
    assert (
        "asyncio.to_thread(invalidate_index)" in source
    ), "the invalidation must be offloaded, matching the other async sites in this file"
    # And every call site must await it, or the coroutine is created and dropped.
    route = inspect.getsource(models_route.delete_finetuned_model)
    calls = route.count("_invalidate_local_scans()")
    awaited = route.count("await _invalidate_local_scans()")
    assert calls == awaited == 2, f"{calls} call(s), {awaited} awaited"


def test_a_hit_is_rejected_when_the_generation_moves_mid_read(monkeypatch):
    """Seqlock: generation, entry, generation. A delete completing between the first two
    reads was accepted anyway, and the lock-free path is where most requests land."""
    from core.inference import local_model_resolver as resolver

    catalog = _catalog(1, "s")
    scanned = {"n": 0}

    def _servable(info):
        scanned["n"] += 1
        return (True, ("Q4_K_M",))

    monkeypatch.setattr("core.inference.local_model_resolver.local_servable_model", _servable)
    monkeypatch.setattr("core.inference.local_model_resolver.local_load_dir", lambda p: p)
    monkeypatch.setattr(inf, "_resolves_to_resident", lambda key, **kw: False)

    real = inf._servability_generation
    entry_generation = real()
    inf._SERVABLE_SCAN_CACHE["entry"] = (
        888.0,
        catalog,
        entry_generation,
        time.monotonic(),
        [("stale",)],
    )
    calls = {"n": 0}

    def _moves_after_the_first_read():
        # The entry read sits between the two generation reads. The delete lands there.
        calls["n"] += 1
        if calls["n"] == 1:
            return entry_generation
        resolver.invalidate_index()
        return real()

    monkeypatch.setattr(inf, "_servability_generation", _moves_after_the_first_read)
    rows = inf._servable_catalog_rows(catalog, 888.0)
    assert rows != [("stale",)], "a delete completing mid-read must not be served"
    assert scanned["n"] > 0, "the request must fall through to a real scan"


def test_an_out_of_band_file_change_is_picked_up_within_the_scan_ttl(monkeypatch):
    """No counter moves when a file is removed from a scan folder by hand, and the
    per-request scan used to notice immediately. The entry's own TTL bounds that."""
    catalog = _catalog(2, "o")
    gone: set[str] = set()
    monkeypatch.setattr(
        "core.inference.local_model_resolver.local_servable_model",
        lambda info: None if info.path in gone else (True, ("Q4_K_M",)),
    )
    monkeypatch.setattr("core.inference.local_model_resolver.local_load_dir", lambda p: p)
    monkeypatch.setattr(inf, "_resolves_to_resident", lambda key, **kw: False)

    clock = {"t": 1000.0}
    monkeypatch.setattr(inf.time, "monotonic", lambda: clock["t"])

    assert len(inf._servable_catalog_rows(catalog, 999.0)) == 2
    gone.add("/models/o1")  # rm, outside every instrumented path
    assert len(inf._servable_catalog_rows(catalog, 999.0)) == 2, "still inside the TTL"
    clock["t"] += inf._SERVABLE_SCAN_TTL_S + 0.1
    assert [r[0].id for r in inf._servable_catalog_rows(catalog, 999.0)] == ["repo/o0"]
