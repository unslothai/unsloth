# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The load-token contract of the image and video unload routes, against a fake engine."""

from __future__ import annotations

import gc
import threading
import time
import weakref

import pytest

from core.inference import media_load_ids
from hub.services.models import account_access
from utils.account_context import AccountContext, run_as


class _State:
    """Stands in for an engine's resident state (pipeline and weights): weakly referable, as theirs are."""


class _Engine:
    """begin_load / unload bump the epoch as the real engines do; a load stays in flight until finish()."""

    def __init__(self) -> None:
        self._load_token = 0
        self._state = None
        self.loading: tuple = ()
        self.unloads = 0

    def begin_load(self, repo: str) -> dict:
        self._load_token += 1
        self.loading = (repo,)
        return {"loaded": False}

    def finish(self) -> None:
        self._state = _State()
        self.loading = ()

    def fail(self) -> None:
        self.loading = ()

    def loading_repo_ids(self) -> tuple:
        return self.loading

    def unload(self, only_if = None) -> dict:
        if only_if is not None and not only_if():
            return None
        self.unloads += 1
        self._load_token += 1
        self._state = None
        self.loading = ()
        return {"loaded": False}


@pytest.fixture(autouse = True)
def _fresh_records():
    media_load_ids._reset_for_tests()
    yield
    media_load_ids._reset_for_tests()


def _load(
    engine: _Engine,
    token,
    repo = "org/model",
    modality = "video",
):
    # As the routes call it: inside admit_media_load, so under media_load_lock.
    with account_access.media_load_lock:
        prior = media_load_ids.admitting(modality, engine, token)
        engine.begin_load(repo)
        media_load_ids.admitted(modality, engine, token, prior)


def _unload(
    engine: _Engine,
    token,
    modality = "video",
):
    return media_load_ids.unload_if_current(modality, engine, token, engine.unload)


@pytest.mark.parametrize("finished", [False, True])
def test_the_token_unloads_its_own_load_in_flight_or_resident(finished):
    engine = _Engine()
    _load(engine, "t1")
    if finished:
        engine.finish()
    assert _unload(engine, "t1") == ("unloaded", {"loaded": False})
    assert engine.unloads == 1
    assert _unload(engine, "t1") == ("superseded", None)
    assert engine.unloads == 1


@pytest.mark.parametrize("newer_token", [None, "t2"])
@pytest.mark.parametrize("finished", [False, True])
def test_a_stale_token_leaves_a_same_model_replacement_alone(newer_token, finished):
    # A newer load of the very same repo, from the UI (no token) or any other client.
    engine = _Engine()
    _load(engine, "t1")
    engine.finish()
    _load(engine, newer_token)
    if finished:
        engine.finish()
    assert _unload(engine, "t1") == ("superseded", None)
    assert engine.unloads == 0
    assert engine.loading or engine._state is not None


def test_a_token_whose_load_failed_leaves_the_previous_model_alone():
    engine = _Engine()
    _load(engine, None)
    engine.finish()
    previous = engine._state
    _load(engine, "t1")
    engine.fail()
    assert _unload(engine, "t1") == ("superseded", None)
    assert engine._state is previous and engine.unloads == 0


def test_a_token_whose_model_was_evicted_elsewhere_leaves_the_slot_alone():
    engine = _Engine()
    _load(engine, "t1")
    engine.finish()
    # The arbiter's evictor, Studio's idle unload: an unload that is not this token's.
    engine.unload()
    engine.unloads = 0
    assert _unload(engine, "t1") == ("superseded", None)
    assert engine.unloads == 0


def test_a_token_names_a_load_on_one_engine_only():
    # The image router switched engines since: the record names the old one.
    old, new = _Engine(), _Engine()
    _load(old, "t1")
    new.begin_load("org/model")
    assert _unload(new, "t1") == ("superseded", None)
    assert new.unloads == 0 and old.unloads == 0


def test_an_unseen_token_cancels_its_load_before_it_starts():
    # The POST is still on its way, or its answer was lost: the unload comes first.
    engine = _Engine()
    assert _unload(engine, "t1") == ("cancelled", None)
    assert engine.unloads == 0
    # The route's early check refuses it (and so does admission, for one already past that check).
    with pytest.raises(media_load_ids.LoadCancelledError):
        media_load_ids.refuse_cancelled("video", "t1")
    assert _unload(engine, "t3") == ("cancelled", None)
    with pytest.raises(media_load_ids.LoadCancelledError):
        _load(engine, "t3")
    assert engine._load_token == 0 and not engine.loading
    _load(engine, "t2")
    _load(engine, None)
    assert engine._load_token == 2


def test_slots_keep_their_own_tokens():
    image, video = _Engine(), _Engine()
    _load(image, "t1", modality = "diffusion")
    _load(video, "t2")
    assert _unload(video, "t1") == ("cancelled", None)
    assert _unload(image, "t1", modality = "diffusion")[0] == "unloaded"
    assert video.loading and video.unloads == 0


def test_the_unload_runs_under_the_admission_lock():
    # No load can be admitted between the check and the unload.
    engine = _Engine()
    _load(engine, "t1")
    held = []

    def unload(only_if):
        held.append(account_access.media_load_lock.locked())
        return engine.unload(only_if)

    assert media_load_ids.unload_if_current("video", engine, "t1", unload)[0] == "unloaded"
    assert held == [True]


def test_a_load_admitted_while_the_unload_waits_is_not_cancelled():
    # The unload is blocked on the lock while a newer load is admitted: it then finds that load.
    engine = _Engine()
    _load(engine, "t1")
    outcome = []
    with account_access.media_load_lock:
        worker = threading.Thread(target = lambda: outcome.append(_unload(engine, "t1")))
        worker.start()
        worker.join(0.2)
        assert worker.is_alive()
        prior = media_load_ids.admitting("video", engine, "t2")
        engine.begin_load("org/model")
        media_load_ids.admitted("video", engine, "t2", prior)
    worker.join(5)
    assert outcome == [("superseded", None)]
    assert engine.loading and engine.unloads == 0


def test_a_cancellation_is_pushed_out_only_by_the_same_accounts_tokens(monkeypatch):
    monkeypatch.setattr(media_load_ids, "_PER_ACCOUNT", 3)
    engine = _Engine()
    assert _as("a", _unload, engine, "mine") == ("cancelled", None)
    # Another account cancelling many tokens cannot evict this account's cancellation.
    for n in range(10):
        _as("b", _unload, engine, f"theirs{n}")
    assert [t for t in media_load_ids._cancelled[("video", "b")]] == [
        "theirs7",
        "theirs8",
        "theirs9",
    ]
    with pytest.raises(media_load_ids.LoadCancelledError):
        _as("a", _load, engine, "mine")


def test_a_cancellation_is_consumed_by_its_load_or_expires(monkeypatch):
    engine = _Engine()
    assert _unload(engine, "t1") == ("cancelled", None)
    with pytest.raises(media_load_ids.LoadCancelledError):
        _load(engine, "t1")
    assert not any("t1" in tokens for tokens in media_load_ids._cancelled.values())
    clock = [1000.0]
    monkeypatch.setattr(media_load_ids, "_now", lambda: clock[0])
    assert _unload(engine, "t2") == ("cancelled", None)
    clock[0] += media_load_ids._CANCELLED_TTL_S + 1
    _load(engine, "t2")
    assert engine.loading


def _as(account: str, fn, *args):
    return run_as(AccountContext(account_id = account, username = account), fn, *args)


def test_a_cancelled_token_refuses_only_the_cancelling_accounts_load():
    # Another account naming a token in advance cannot stop this account's load with it.
    engine = _Engine()
    assert _as("b", _unload, engine, "t1") == ("cancelled", None)
    _as("a", _load, engine, "t1")
    assert engine.loading and engine._load_token == 1
    assert _as("b", _unload, engine, "t2") == ("cancelled", None)
    with pytest.raises(media_load_ids.LoadCancelledError):
        _as("b", _load, engine, "t2")


def test_status_names_the_load_in_the_slot():
    engine = _Engine()
    assert media_load_ids.current_id("video", engine) is None
    _load(engine, "t1")
    assert media_load_ids.current_id("video", engine) == "t1"
    engine.finish()
    assert media_load_ids.current_id("video", engine) == "t1"
    _load(engine, None)
    made = media_load_ids.current_id("video", engine)
    assert made.startswith("studio-") and made != "t1"
    assert _unload(engine, "t1") == ("superseded", None)
    assert _unload(engine, made)[0] == "unloaded"
    assert media_load_ids.current_id("video", engine) is None


def test_a_resident_stays_named_after_a_newer_load_fails():
    engine = _Engine()
    _load(engine, "t1")
    engine.finish()
    resident = engine._state
    _load(engine, "t2")
    assert media_load_ids.current_id("video", engine) == "t2"
    engine.fail()
    assert engine._state is resident
    assert media_load_ids.current_id("video", engine) == "t1"
    assert _unload(engine, "t2") == ("superseded", None)
    assert _unload(engine, "t1")[0] == "unloaded"


def test_a_blocking_status_read_holds_up_no_admission_or_unload():
    engine = _Engine()
    _load(engine, "t1")
    engine.finish()
    release, reads, result = threading.Event(), [], []

    def read():
        reads.append(account_access.media_load_lock.locked())
        if len(reads) == 1:
            release.wait(10)
        return {"read": len(reads)}

    reader = threading.Thread(
        target = lambda: result.append(media_load_ids.snapshot("video", engine, read))
    )
    reader.start()
    while not reads:
        time.sleep(0.01)
    # While the read blocks: a token unload and a new admission both go through.
    done = []
    worker = threading.Thread(
        target = lambda: done.extend([_unload(engine, "t1")[0], _load(engine, "t2")])
    )
    worker.start()
    worker.join(5)
    assert not worker.is_alive() and done[0] == "unloaded"
    release.set()
    reader.join(5)
    # The first read straddled them, so it was read again: the id names the state the answer shows.
    assert result == [({"read": 2}, "t2")]
    assert reads == [False, False]


def test_a_status_read_an_admission_lands_in_every_time_names_no_load(monkeypatch):
    engine = _Engine()
    _load(engine, "t1")
    engine.finish()
    tokens = iter(f"n{n}" for n in range(10))

    def read():
        _load(engine, next(tokens))
        return {"loaded": True}

    assert media_load_ids.snapshot("video", engine, read) == ({"loaded": True}, None)


def test_a_token_another_account_used_is_cancelled_for_this_account():
    # Account a once loaded with t1 (now replaced); account b naming t1 records its own cancellation.
    engine = _Engine()
    _as("a", _load, engine, "t1")
    _as("a", _load, engine, "t2")
    assert _as("a", _unload, engine, "t1") == ("superseded", None)
    assert _as("b", _unload, engine, "t1") == ("cancelled", None)
    with pytest.raises(media_load_ids.LoadCancelledError):
        _as("b", _load, engine, "t1")
    assert engine.unloads == 0 and engine._load_token == 2


@pytest.mark.parametrize("named", ["t2", "t1"])
def test_a_load_that_ends_before_the_teardown_costs_no_other_model(named):
    # t2 is in flight over t1's resident model; between the check and the engine's own recheck,
    # t2's worker fails (naming t2) or the resident is replaced (naming t1).
    engine = _Engine()
    _load(engine, "t1")
    engine.finish()
    resident = engine._state
    _load(engine, "t2")
    if named == "t1":
        engine.fail()

    def unload(only_if):
        if named == "t2":
            engine.fail()
        else:
            engine._state = _State()
        return engine.unload(only_if)

    assert media_load_ids.unload_if_current("video", engine, named, unload) == ("superseded", None)
    assert engine.unloads == 0
    assert engine._state is (resident if named == "t2" else engine._state)


def test_the_real_engines_tear_nothing_down_when_the_recheck_fails():
    from core.inference.diffusion import DiffusionBackend
    from core.inference.video import VideoBackend
    for engine in (VideoBackend(), DiffusionBackend()):
        epoch = engine._load_token
        assert engine.unload(only_if = lambda: False) is None
        assert engine._load_token == epoch
        # The recheck reads the engine's in-flight marker without its lock.
        assert media_load_ids._in_flight_unlocked(engine) is False
        assert engine.unload(only_if = lambda: True)["loaded"] is False
        assert engine._load_token == epoch + 1


def _collected(ref) -> bool:
    gc.collect()
    return ref() is None


@pytest.mark.parametrize("how", ["replaced", "unloaded", "evicted", "token_unload"])
def test_the_registry_keeps_no_torn_down_model_alive(how):
    engine = _Engine()
    _load(engine, "t1")
    engine.finish()
    old = weakref.ref(engine._state)
    # A newer load that fails leaves t1's state resident, recorded as the published resident.
    _load(engine, "t2")
    engine.fail()
    assert media_load_ids.current_id("video", engine) == "t1"
    if how == "replaced":
        _load(engine, "t3")
        engine.finish()
    elif how == "token_unload":
        assert _unload(engine, "t1")[0] == "unloaded"
    else:
        # Studio's own unload route without a token, or the arbiter's evictor.
        engine.unload()
    assert _collected(old)
    assert media_load_ids.current_id("video", engine) == ("t3" if how == "replaced" else None)


def test_the_real_engine_states_are_weakly_referable():
    from dataclasses import fields

    from core.inference.diffusion import _LoadState
    from core.inference.video import _VideoLoadState

    for cls in (_LoadState, _VideoLoadState):
        state = cls(**{f.name: None for f in fields(cls) if f.init})
        assert media_load_ids._weak(state)() is state


def test_the_resident_load_id_names_the_model_under_a_load_in_flight():
    engine = _Engine()
    read = lambda: {"loaded": engine._state is not None}
    assert media_load_ids.snapshot_ids("video", engine, read) == ({"loaded": False}, None, None)
    _load(engine, "t1")
    engine.finish()
    assert media_load_ids.snapshot_ids("video", engine, read) == ({"loaded": True}, "t1", "t1")
    # A replacement in flight: load_id names it, resident_load_id the model still loaded under it.
    _load(engine, "t2")
    assert media_load_ids.snapshot_ids("video", engine, read)[1:] == ("t2", "t1")
    engine.finish()
    assert media_load_ids.snapshot_ids("video", engine, read)[1:] == ("t2", "t2")
    # Replaced again, and that replacement fails: the resident stays named by both ids.
    _load(engine, "t3")
    engine.fail()
    assert media_load_ids.snapshot_ids("video", engine, read)[1:] == ("t2", "t2")
    assert media_load_ids.snapshot("video", engine, read) == ({"loaded": True}, "t2")
