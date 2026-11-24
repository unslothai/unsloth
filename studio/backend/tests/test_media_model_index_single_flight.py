# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Callers asking for the same media index while one is being built wait for that build instead of
walking every model root again: at startup the early quant probe and the post-warm diffusers gate
ask at once, and each walk is a full local model scan."""

from __future__ import annotations

import threading

import pytest

import core.inference.media_model_index as index
from hub.utils import inventory_scan
from utils.account_context import AccountContext, run_as

TASK = index.IMAGE_TASK
ALICE = AccountContext("a" * 32, "alice")


class _CountingEvent(threading.Event):
    waiting = 0
    _lock = threading.Lock()

    def wait(self, timeout = None):
        with _CountingEvent._lock:
            _CountingEvent.waiting += 1
        return super().wait(timeout)


@pytest.fixture(autouse = True)
def counted_waits(monkeypatch):
    """Count callers parked on an in-flight build (absent on a tree without single-flight)."""
    _CountingEvent.waiting = 0
    build_cls = getattr(index, "_Build", None)
    if build_cls is not None:
        original_init = build_cls.__init__

        def init(self, epoch):
            original_init(self, epoch)
            self.done = _CountingEvent()

        monkeypatch.setattr(build_cls, "__init__", init)


def _until(predicate):
    for _ in range(5000):
        if predicate():
            return
        threading.Event().wait(0.002)
    raise AssertionError("timed out")


@pytest.fixture
def gated_build(monkeypatch):
    """_build_index that blocks until released and counts its calls."""
    calls = []
    started = threading.Event()
    release = threading.Event()

    def fake_build(task):
        calls.append((threading.get_ident(), task))
        started.set()
        assert release.wait(10)
        return {"flux-dev": index.MediaModelPick("flux-dev", "/models/flux-dev", None, None)}

    monkeypatch.setattr(index, "_build_index", fake_build)
    index.invalidate_index()
    yield calls, started, release
    release.set()
    index.invalidate_index()


def _ids_in_thread(results, account = None):
    def run():
        if account is None:
            results.append(index.available_media_model_ids(TASK))
        else:
            results.append(run_as(account, index.available_media_model_ids, TASK))

    thread = threading.Thread(target = run)
    thread.start()
    return thread


def test_concurrent_callers_share_one_build(gated_build):
    calls, started, release = gated_build
    results = []
    first = _ids_in_thread(results)
    assert started.wait(10)
    second = _ids_in_thread(results)
    third = _ids_in_thread(results)
    # Joined (parked on the first build) or, without single-flight, walking on their own.
    _until(lambda: _CountingEvent.waiting + len(calls) >= 3)
    release.set()
    for thread in (first, second, third):
        thread.join(10)
    assert results == [["flux-dev"]] * 3
    assert len(calls) == 1


def test_a_build_started_before_an_invalidation_is_not_joined(gated_build):
    calls, started, release = gated_build
    results = []
    first = _ids_in_thread(results)
    assert started.wait(10)
    # A download or delete landed after that walk began: the next caller must walk again.
    inventory_scan.invalidate_hf_cache_scans(sizes = False)
    second = _ids_in_thread(results)
    _until(lambda: len(calls) >= 2 or _CountingEvent.waiting >= 1)
    release.set()
    first.join(10)
    second.join(10)
    assert len(calls) == 2


def test_invalidate_index_also_starts_a_new_build(gated_build):
    calls, started, release = gated_build
    results = []
    first = _ids_in_thread(results)
    assert started.wait(10)
    index.invalidate_index()
    second = _ids_in_thread(results)
    _until(lambda: len(calls) >= 2 or _CountingEvent.waiting >= 1)
    release.set()
    first.join(10)
    second.join(10)
    assert len(calls) == 2


def test_another_account_never_joins(gated_build):
    calls, started, release = gated_build
    results = []
    first = _ids_in_thread(results)
    assert started.wait(10)
    second = _ids_in_thread(results, ALICE)
    _until(lambda: len(calls) >= 2 or _CountingEvent.waiting >= 1)
    release.set()
    first.join(10)
    second.join(10)
    assert len(calls) == 2


def test_a_failed_build_is_retried_by_its_waiters(monkeypatch):
    calls = []
    started = threading.Event()
    release = threading.Event()

    def flaky_build(task):
        calls.append(task)
        if len(calls) == 1:
            started.set()
            assert release.wait(10)
            raise OSError("root vanished mid-walk")
        return {"flux-dev": index.MediaModelPick("flux-dev", "/models/flux-dev", None, None)}

    monkeypatch.setattr(index, "_build_index", flaky_build)
    index.invalidate_index()
    errors, results = [], []

    def owner():
        try:
            index.available_media_model_ids(TASK)
        except OSError as exc:
            errors.append(exc)

    first = threading.Thread(target = owner)
    first.start()
    assert started.wait(10)
    second = _ids_in_thread(results)
    _until(lambda: _CountingEvent.waiting + len(calls) >= 2)
    release.set()
    first.join(10)
    second.join(10)
    index.invalidate_index()
    assert len(errors) == 1
    assert results == [["flux-dev"]]
    assert len(calls) == 2


def test_a_reentrant_call_from_the_building_thread_does_not_deadlock(monkeypatch):
    depth = []

    def nested_build(task):
        depth.append(task)
        if len(depth) == 1:
            # Same thread, same key, while its own build is in flight.
            index.available_media_model_ids(TASK)
        return {}

    monkeypatch.setattr(index, "_build_index", nested_build)
    index.invalidate_index()
    done = []
    thread = threading.Thread(target = lambda: done.append(index.available_media_model_ids(TASK)))
    thread.start()
    thread.join(10)
    index.invalidate_index()
    assert not thread.is_alive()
    assert done == [[]]
    assert len(depth) == 2


def test_a_fresh_entry_still_answers_without_a_build(gated_build):
    calls, started, release = gated_build
    release.set()
    assert index.available_media_model_ids(TASK) == ["flux-dev"]
    assert index.available_media_model_ids(TASK) == ["flux-dev"]
    assert len(calls) == 1


@pytest.mark.parametrize("invalidate", ["index", "hf_cache"])
def test_a_build_overtaken_by_an_invalidation_is_not_cached(monkeypatch, invalidate):
    """A walk that began before a download finishes after the fresh walk: its answer must not shadow the fresh one."""
    calls = []
    started = threading.Event()
    release = threading.Event()

    def build(task):
        calls.append(task)
        if len(calls) == 1:
            started.set()
            assert release.wait(10)
            return {"old": index.MediaModelPick("old", "/models/old", None, None)}
        return {"new": index.MediaModelPick("new", "/models/new", None, None)}

    monkeypatch.setattr(index, "_build_index", build)
    index.invalidate_index()
    results = []
    first = _ids_in_thread(results)
    assert started.wait(10)
    if invalidate == "index":
        index.invalidate_index()
    else:
        inventory_scan.invalidate_hf_cache_scans(sizes = False)
    assert index.available_media_model_ids(TASK) == ["new"]
    release.set()
    first.join(10)
    index_after = index.available_media_model_ids(TASK)
    index.invalidate_index()
    assert results == [["old"]]
    assert index_after == ["new"]
    assert len(calls) == 2


def test_a_build_waiting_on_a_thread_that_asks_the_index_does_not_deadlock(monkeypatch):
    """A waiter gives up on a build that never finishes and scans on its own, so a builder that hands work to a
    helper thread asking the index cannot lock both threads."""
    monkeypatch.setattr(index, "_JOIN_TIMEOUT_S", 0.2)
    calls = []

    def build(task):
        calls.append(1)
        if len(calls) == 1:
            box = []
            helper = threading.Thread(
                target = lambda: box.append(index.available_media_model_ids(TASK))
            )
            helper.start()
            helper.join(5)
            assert not helper.is_alive() and box == [["helper"]]
        return {"helper": index.MediaModelPick("helper", "/models/helper", None, None)}

    monkeypatch.setattr(index, "_build_index", build)
    index.invalidate_index()
    assert index.available_media_model_ids(TASK) == ["helper"]
    assert len(calls) == 2
