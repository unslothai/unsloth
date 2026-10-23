# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The load-token contract of the image and video unload routes, against a fake engine."""

from __future__ import annotations

import threading

import pytest

from core.inference import media_load_ids
from hub.services.models import account_access
from utils.account_context import AccountContext, run_as


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
        self._state = object()
        self.loading = ()

    def fail(self) -> None:
        self.loading = ()

    def loading_repo_ids(self) -> tuple:
        return self.loading

    def unload(self) -> dict:
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
    # Done: the same token again is a no-op.
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
    with pytest.raises(media_load_ids.LoadCancelledError):
        media_load_ids.refuse_cancelled("video", "t1")
    with pytest.raises(media_load_ids.LoadCancelledError):
        _load(engine, "t1")
    assert engine._load_token == 0 and not engine.loading
    # Another token, or none, loads as before.
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

    def unload():
        held.append(account_access.media_load_lock.locked())
        return engine.unload()

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


def test_remembered_tokens_are_bounded(monkeypatch):
    monkeypatch.setattr(media_load_ids, "_REMEMBERED", 3)
    engine = _Engine()
    for n in range(5):
        _unload(engine, f"gone{n}")
    assert [t for t, _a in media_load_ids._cancelled["video"]] == ["gone2", "gone3", "gone4"]


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
    # A load without a token gets one from Studio, which an unload can name.
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
