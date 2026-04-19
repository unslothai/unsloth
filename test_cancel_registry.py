"""
Standalone test for the cancel-registry logic extracted from
studio/backend/routes/inference.py
"""

import threading
import time
import unittest

# ── Extract the exact production code ────────────────────────────────────────

_CANCEL_REGISTRY: dict[str, set] = {}
_CANCEL_LOCK = threading.Lock()
_PENDING_CANCELS: dict[str, float] = {}
_PENDING_CANCEL_TTL_S = 30.0


def _prune_pending(now: float) -> None:
    for k in [
        k for k, ts in _PENDING_CANCELS.items() if now - ts > _PENDING_CANCEL_TTL_S
    ]:
        _PENDING_CANCELS.pop(k, None)


class _TrackedCancel:
    def __init__(self, event: threading.Event, *keys):
        self.event = event
        self.keys = tuple(k for k in keys if k)

    def __enter__(self):
        should_cancel = False
        with _CANCEL_LOCK:
            for k in self.keys:
                _CANCEL_REGISTRY.setdefault(k, set()).add(self.event)
            now = time.monotonic()
            _prune_pending(now)
            for k in self.keys:
                if k and _PENDING_CANCELS.pop(k, None) is not None:
                    should_cancel = True
        if should_cancel:
            self.event.set()
        return self.event

    def __exit__(self, *exc):
        with _CANCEL_LOCK:
            for k in self.keys:
                bucket = _CANCEL_REGISTRY.get(k)
                if bucket is None:
                    continue
                bucket.discard(self.event)
                if not bucket:
                    _CANCEL_REGISTRY.pop(k, None)
        return False


def _cancel_by_keys(keys) -> int:
    if not keys:
        return 0
    events: set = set()
    with _CANCEL_LOCK:
        _prune_pending(time.monotonic())
        for k in keys:
            bucket = _CANCEL_REGISTRY.get(k)
            if bucket:
                events.update(bucket)
    for ev in events:
        ev.set()
    return len(events)


def _cancel_by_cancel_id_or_stash(cancel_id: str) -> int:
    now = time.monotonic()
    events: set = set()
    with _CANCEL_LOCK:
        _prune_pending(now)
        bucket = _CANCEL_REGISTRY.get(cancel_id)
        if bucket:
            events.update(bucket)
        else:
            _PENDING_CANCELS[cancel_id] = now
    for ev in events:
        ev.set()
    return len(events)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reset():
    """Clear all global state between tests."""
    with _CANCEL_LOCK:
        _CANCEL_REGISTRY.clear()
        _PENDING_CANCELS.clear()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCancelRegistry(unittest.TestCase):

    def setUp(self):
        _reset()

    # ------------------------------------------------------------------
    # 1. Race: cancel POST arrives BEFORE stream registers (stash+replay)
    # ------------------------------------------------------------------
    def test_cancel_before_register_stashed_and_replayed(self):
        cid = "run-001"
        # Cancel POST fires first
        n = _cancel_by_cancel_id_or_stash(cid)
        self.assertEqual(n, 0, "No events registered yet, count must be 0")
        self.assertIn(cid, _PENDING_CANCELS, "cancel_id must be stashed")

        # Stream now registers
        ev = threading.Event()
        with _TrackedCancel(ev, cid):
            self.assertTrue(ev.is_set(),
                "Event must be set immediately on __enter__ when a pending cancel exists")
        self.assertNotIn(cid, _PENDING_CANCELS, "Stash must be consumed on __enter__")

    # ------------------------------------------------------------------
    # 2. Race: cancel POST arrives AFTER stream registers (immediate fire)
    # ------------------------------------------------------------------
    def test_cancel_after_register_fires_immediately(self):
        cid = "run-002"
        ev = threading.Event()
        tracker = _TrackedCancel(ev, cid)
        tracker.__enter__()
        self.assertFalse(ev.is_set(), "Event must not be set before cancel")

        n = _cancel_by_cancel_id_or_stash(cid)
        self.assertEqual(n, 1)
        self.assertTrue(ev.is_set(), "Event must be set after cancel POST")
        tracker.__exit__(None, None, None)

    # ------------------------------------------------------------------
    # 3. Multiple concurrent streams sharing the same session_id
    # ------------------------------------------------------------------
    def test_multiple_streams_same_session_id(self):
        sid = "session-shared"
        ev1, ev2 = threading.Event(), threading.Event()
        t1 = _TrackedCancel(ev1, sid)
        t2 = _TrackedCancel(ev2, sid)
        t1.__enter__()
        t2.__enter__()

        bucket = _CANCEL_REGISTRY.get(sid)
        self.assertIsNotNone(bucket)
        self.assertEqual(len(bucket), 2, "Both events must be in the same bucket")

        n = _cancel_by_keys([sid])
        self.assertEqual(n, 2)
        self.assertTrue(ev1.is_set())
        self.assertTrue(ev2.is_set())

        t1.__exit__(None, None, None)
        t2.__exit__(None, None, None)
        self.assertNotIn(sid, _CANCEL_REGISTRY, "Bucket must be removed when empty")

    # ------------------------------------------------------------------
    # 4. _TrackedCancel.__exit__ called twice (double-cleanup)
    # ------------------------------------------------------------------
    def test_double_exit_is_safe(self):
        cid = "run-double-exit"
        ev = threading.Event()
        tracker = _TrackedCancel(ev, cid)
        tracker.__enter__()
        tracker.__exit__(None, None, None)   # first exit removes the bucket
        try:
            tracker.__exit__(None, None, None)  # second exit must not raise
        except Exception as exc:
            self.fail(f"Second __exit__ raised: {exc}")
        self.assertNotIn(cid, _CANCEL_REGISTRY)

    # ------------------------------------------------------------------
    # 5. _prune_pending with expired TTL entries
    # ------------------------------------------------------------------
    def test_prune_pending_removes_expired(self):
        old_ts = time.monotonic() - (_PENDING_CANCEL_TTL_S + 1)
        _PENDING_CANCELS["expired-key"] = old_ts
        _PENDING_CANCELS["fresh-key"] = time.monotonic()

        _prune_pending(time.monotonic())

        self.assertNotIn("expired-key", _PENDING_CANCELS, "Expired entry must be pruned")
        self.assertIn("fresh-key", _PENDING_CANCELS, "Fresh entry must survive")

    # ------------------------------------------------------------------
    # 6. cancel_id is None / falsy — must be filtered out of keys
    # ------------------------------------------------------------------
    def test_falsy_cancel_id_is_filtered(self):
        ev = threading.Event()
        # _TrackedCancel filters falsy keys in __init__
        tracker = _TrackedCancel(ev, None, "", "valid-key")
        self.assertEqual(tracker.keys, ("valid-key",),
            "_TrackedCancel.keys must strip None/''/falsy values")
        tracker.__enter__()
        self.assertNotIn(None, _CANCEL_REGISTRY)
        self.assertNotIn("", _CANCEL_REGISTRY)
        self.assertIn("valid-key", _CANCEL_REGISTRY)
        tracker.__exit__(None, None, None)

    # ------------------------------------------------------------------
    # 7. _cancel_by_cancel_id_or_stash must NOT fall through to session_id
    # ------------------------------------------------------------------
    def test_cancel_id_exclusive_no_session_fallthrough(self):
        """When cancel_id is supplied, only that exact key is looked up/stashed.
        A matching session_id in the registry must NOT be fired."""
        sid = "session-ABC"
        cid = "cancel-XYZ"
        ev = threading.Event()
        # Register event under session_id only
        _TrackedCancel(ev, sid).__enter__()

        # POST cancel with cancel_id that doesn't match the session_id
        n = _cancel_by_cancel_id_or_stash(cid)
        self.assertEqual(n, 0, "Must not hit session_id bucket when cancel_id is given")
        self.assertFalse(ev.is_set(), "Event must NOT be set — wrong key was used")
        self.assertIn(cid, _PENDING_CANCELS, "Unmatched cancel_id must be stashed")

    # ------------------------------------------------------------------
    # 8. Memory leak: _CANCEL_REGISTRY grows if streams never exit
    # ------------------------------------------------------------------
    def test_registry_does_not_shrink_without_exit(self):
        """Streams that never call __exit__ keep their entries alive forever —
        this is the expected/documented behaviour (no TTL on the registry
        side), but we want to surface it as a known issue."""
        for i in range(5):
            ev = threading.Event()
            tracker = _TrackedCancel(ev, f"leak-key-{i}")
            tracker.__enter__()
            # NOTE: intentionally NOT calling __exit__

        self.assertEqual(len(_CANCEL_REGISTRY), 5,
            "KNOWN ISSUE: registry grows unbounded when __exit__ is never called "
            "(no TTL on _CANCEL_REGISTRY — this is a potential memory leak)")

    # ------------------------------------------------------------------
    # 9. Pruning is partial: _CANCEL_REGISTRY is NOT pruned alongside _PENDING_CANCELS
    # ------------------------------------------------------------------
    def test_pruning_asymmetry(self):
        """
        _prune_pending only cleans _PENDING_CANCELS.
        _CANCEL_REGISTRY has no TTL — stale entries accumulate if __exit__ is skipped.
        Demonstrate that _cancel_by_keys (which calls _prune_pending) does NOT
        clean up abandoned _CANCEL_REGISTRY buckets.
        """
        abandoned_key = "abandoned-stream"
        ev = threading.Event()
        tracker = _TrackedCancel(ev, abandoned_key)
        tracker.__enter__()
        # Simulate stream dying without calling __exit__

        # Now call _cancel_by_keys with an unrelated key — this triggers _prune_pending
        _cancel_by_keys(["unrelated"])

        self.assertIn(abandoned_key, _CANCEL_REGISTRY,
            "KNOWN ISSUE: abandoned entry in _CANCEL_REGISTRY is never reaped by "
            "_prune_pending — only _PENDING_CANCELS is pruned on TTL")

    # ------------------------------------------------------------------
    # 10. Thread-safety smoke test for the stash-replay race
    # ------------------------------------------------------------------
    def test_concurrent_stash_and_register(self):
        """
        Fire _cancel_by_cancel_id_or_stash and _TrackedCancel.__enter__
        concurrently many times; the event must always end up set.
        """
        errors = []
        for iteration in range(100):
            _reset()
            cid = f"race-{iteration}"
            ev = threading.Event()
            barrier = threading.Barrier(2)

            def do_cancel():
                barrier.wait()
                _cancel_by_cancel_id_or_stash(cid)

            def do_register():
                barrier.wait()
                t = _TrackedCancel(ev, cid)
                t.__enter__()
                # Give cancel thread a moment to stash if it ran first
                time.sleep(0.001)
                # If cancel ran first, event should already be set via stash replay;
                # if register ran first, _cancel_by_cancel_id_or_stash sets it directly.

            ct = threading.Thread(target=do_cancel)
            rt = threading.Thread(target=do_register)
            ct.start(); rt.start()
            ct.join(); rt.join()

            if not ev.is_set():
                errors.append(f"iteration {iteration}: event not set after cancel+register race")

        self.assertEqual(errors, [], f"Race condition failures:\n" + "\n".join(errors))


if __name__ == "__main__":
    unittest.main(verbosity=2)
