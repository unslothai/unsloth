# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whether a failed image generation can still say WHY once it is no longer running.

A POST lost past the proxy's ~100s window leaves the client polling generate-progress as
its only channel. Both engines clear ``_gen`` in a ``finally``, so without a retained
reason that poll answers "not running" for a failure, and the settling path reads it as
success and can advance a batch past an output that never arrived.

Static plus behavioural: generating needs a GPU, so what runs here is the part deciding
what a CALLER sees, the route's classification, while the engines' retention, clearing and
publishing are asserted against their own source.
"""

from __future__ import annotations

import ast
import re
import sys
import types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

ENGINES = ("core/inference/diffusion.py", "core/inference/sd_cpp_backend.py")


def _src(rel: str) -> str:
    return (Path(_BACKEND_DIR) / rel).read_text(encoding = "utf-8")


@pytest.mark.parametrize("engine", ENGINES)
def test_an_engine_retains_the_reason_past_the_clear_of_gen(engine):
    src = _src(engine)
    assert "self._last_generate_error = str(exc) or type(exc).__name__" in src, (
        f"{engine} no longer retains a failed generation's reason, so idle progress "
        "cannot distinguish a failure from a finished run"
    )
    # Cleared when the NEXT run starts, or a later poll would read a stale failure as
    # belonging to the generation now in flight.
    assert re.search(
        r"self\._gen = _(?:Gen|Sd)[A-Za-z]*\(total_steps[^\n]*\n\s*#[^\n]*\n(?:\s*#[^\n]*\n)?\s*self\._last_generate_error = None",
        src,
    ), f"{engine} does not clear the retained reason when a generation starts"


@pytest.mark.parametrize("engine", ENGINES)
def test_the_idle_branch_is_what_publishes_it(engine):
    """On the idle branch only. Published while active it would describe the previous run."""
    src = _src(engine)
    tree = ast.parse(src)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "generate_progress":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Dict):
                    keys = [k.value for k in sub.keys if isinstance(k, ast.Constant)]
                    if "active" in keys:
                        found.append(keys)
    assert found, f"{engine} has no generate_progress dict to check"
    idle = [k for k in found if "error" in k]
    assert len(idle) == 1, f"{engine}: expected exactly one branch to carry error, got {found}"


def test_the_route_classifies_it_and_never_relays_engine_text(monkeypatch):
    """The engine keeps the raw string; only the route decides what a caller may see.

    Otherwise the field becomes a hole for the engine's own text, which carries local paths
    and argv -- the thing _generate_failure_detail exists to keep out of a message.
    """
    import routes.inference as route

    raw = (
        "CUDA out of memory. Tried to allocate 2.00 GiB while loading "
        "/home/somebody/.unsloth/studio/models/secret-project/model.safetensors "
        "with --argv-that-should-not-ship"
    )
    classified = route._generate_failure_detail(raw)
    assert "/home/somebody" not in classified, "a local path reached a client-visible message"
    assert "--argv" not in classified
    assert classified != raw and classified, "the raw text was relayed unchanged"
    # And it still says something specific rather than collapsing to the bare fallback,
    # which is the whole reason this PR touched these messages.
    assert (
        classified != route._GENERATE_FAILURE_FALLBACK
    ), "an out-of-memory failure classified to the bare fallback"

    # Nothing to say when nothing failed.
    assert route._generate_failure_detail("") == route._GENERATE_FAILURE_FALLBACK


def test_the_progress_route_puts_the_classified_reason_on_the_response():
    """Wiring, read from the route's own source: the response must carry the CLASSIFIED
    value, not the engine's."""
    src = _src("routes/inference.py")
    at = src.index("async def diffusion_generate_progress")
    body = src[at : at + 4500]
    assert (
        '"error": _generate_failure_detail(raw_error) if raw_error else None' in body
    ), "the progress route no longer classifies the retained reason"


@pytest.mark.parametrize("engine", ENGINES)
def test_an_engine_identifies_the_reason_with_the_attempt_that_caused_it(engine):
    """A retained reason is only useful if a caller can tell WHOSE run it came from.

    A counter only answers "later", and later includes a concurrent client's run as well as
    this attempt's, while a generation whose POST never reached the backend started no run at
    all. So the engine keeps the id the request carried, and a caller matches it exactly.
    """
    src = _src(engine)
    assert (
        "self._last_generate_attempt = attempt_id" in src
    ), f"{engine} does not keep the attempt id, so a reason cannot be identified"
    assert (
        '"generation_attempt": getattr(self, "_last_generate_attempt", None),' in src
    ), f"{engine} does not publish the attempt id beside the reason"
    assert (
        "attempt_id: Optional[str] = None," in src
    ), f"{engine} does not accept an attempt id from the route"
    # Recorded where the run STARTS, beside the clear, or the id and the reason would
    # describe different moments.
    recorded = src.index("self._last_generate_attempt = attempt_id")
    clear = src.index("self._last_generate_error = None")
    assert (
        0 < recorded - clear < 600
    ), f"{engine} records the attempt id away from where the reason is cleared"
    # And no counter left behind: a monotonic lower bound is what this replaced, and leaving
    # it published invites the comparison it cannot support.
    assert "generation_seq" not in src, f"{engine} still publishes the superseded counter"


def test_the_route_forwards_the_attempt_id_and_sends_it_only_beside_a_reason():
    """Forwarded from the request, returned only with the failure it dates.

    Without a reason there is nothing to attribute, and only the client that sent the id can
    match it -- so a concurrent client has no business reading which attempt last failed.
    """
    src = _src("routes/inference.py")
    at = src.index("async def generate_diffusion_image")
    assert (
        "attempt_id = request.attempt_id," in src[at : at + 4000]
    ), "the generate route no longer forwards the attempt id to the engine"
    at = src.index("async def diffusion_generate_progress")
    body = src[at : at + 4500]
    assert 'if not progress.get("error"):' in body
    assert 'progress.pop("generation_attempt", None)' in body


def test_the_attempt_id_is_bounded_and_patterned_on_the_way_in():
    """It comes off a request and goes back out on a response, so it is validated there."""
    from models.inference import DiffusionGenerateRequest

    ok = DiffusionGenerateRequest(prompt = "p", attempt_id = "a-Z_09")
    assert ok.attempt_id == "a-Z_09"
    assert DiffusionGenerateRequest(prompt = "p").attempt_id is None
    for bad in ("has space", "semi;colon", "<script>", "x" * 65, "sl/ash"):
        with pytest.raises(Exception):
            DiffusionGenerateRequest(prompt = "p", attempt_id = bad)


def test_a_failure_survives_the_runs_that_follow_it():
    """One retained slot is not enough when a second client is queued.

    A client whose POST was lost polls once a second. A queued client can take the slot in
    that window, and the run it starts clears the slot: the settling client then reads the
    newcomer going active and idle as its OWN success, and advances a batch past an output
    that never arrived. Outcomes are therefore kept per attempt.
    """
    from core.inference.generate_outcomes import (
        _RETAINED_GENERATE_FAILURES,
        _retain_generate_failure,
        generate_failure_for_attempt,
    )

    engine = types.SimpleNamespace()
    _retain_generate_failure(engine, "attempt-a", "CUDA out of memory")
    # B starts and fails; A has not polled yet.
    _retain_generate_failure(engine, "attempt-b", "model was replaced")
    assert (
        generate_failure_for_attempt(engine, "attempt-a") == "CUDA out of memory"
    ), "a later run discarded the reason the settling client is waiting for"
    assert generate_failure_for_attempt(engine, "attempt-b") == "model was replaced"
    # An attempt that never failed, or never ran, has nothing to report.
    assert generate_failure_for_attempt(engine, "attempt-c") is None
    assert generate_failure_for_attempt(engine, None) is None
    assert generate_failure_for_attempt(engine, "") is None

    # Bounded, oldest first out: only a settling caller reads one, and it reads it within
    # seconds, so this cannot grow with uptime.
    for i in range(_RETAINED_GENERATE_FAILURES + 4):
        _retain_generate_failure(engine, f"attempt-{i}", f"reason {i}")
    assert len(engine._generate_outcomes) == _RETAINED_GENERATE_FAILURES
    assert generate_failure_for_attempt(engine, "attempt-0") is None, "the bound is not enforced"
    assert generate_failure_for_attempt(engine, f"attempt-{_RETAINED_GENERATE_FAILURES + 3}")

    # An id with no reason is not remembered at all, so a successful run leaves nothing.
    fresh = types.SimpleNamespace()
    _retain_generate_failure(fresh, None, "should not be kept")
    assert getattr(fresh, "_generate_outcomes", None) is None


@pytest.mark.parametrize("engine", ENGINES)
def test_an_engine_records_the_failure_against_its_own_attempt(engine):
    """Both engines, at the point where they already retain the reason."""
    src = _src(engine)
    assert (
        "_retain_generate_failure(" in src
    ), f"{engine} does not record the failure against the attempt that ran it"
    retained = src.index("self._last_generate_error = str(exc) or type(exc).__name__")
    recorded = src.index("_retain_generate_failure(", retained)
    assert (
        recorded - retained < 300
    ), f"{engine} records the per-attempt outcome away from where it retains the reason"


def test_the_progress_route_answers_about_the_attempt_it_was_asked_about():
    """Named, the answer is that attempt's; unnamed, the retained slot answers as before."""
    src = _src("routes/inference.py")
    at = src.index("async def diffusion_generate_progress")
    body = src[at : at + 4500]
    assert (
        "attempt_id: Optional[str] = Query(" in body
    ), "the progress route cannot be asked about a particular attempt"
    assert "generate_failure_for_attempt(engine, attempt_id)" in body
    assert (
        "if attempt_id is not None:" in body
    ), "an older client with no attempt id no longer gets the retained slot"


def test_the_native_cancel_branch_records_its_attempt_too():
    """sd.cpp catches SdCppCancelled before the generic handler.

    An unload or a superseding load cancels the run; no image was produced, so a client
    settling a lost POST has to be told. The keyed lookup answers whenever the client names
    its attempt, so a branch that only wrote the single slot left that client reading idle
    as success. The diffusers engine raises RuntimeError for the same case and reaches its
    generic handler, which already records.
    """
    src = _src("core/inference/sd_cpp_backend.py")
    at = src.index("except SdCppCancelled as exc:")
    branch = src[at : at + 700]
    assert (
        "_retain_generate_failure(self, attempt_id, DIFFUSION_CANCELLED_MSG)" in branch
    ), "a cancelled native generation records nothing against its attempt"
    # Before the re-raise, or it never runs.
    assert branch.index("_retain_generate_failure") < branch.index("raise RuntimeError(")


@pytest.mark.parametrize("engine", ENGINES)
def test_an_engine_says_whose_run_is_active(engine):
    """On the active branch as well as the idle one.

    A caller settling a lost POST asks whether ITS generation is running. Told only that
    SOMETHING is, it counts a concurrent client's run as its own and reads that run going
    idle as its own success.
    """
    src = _src(engine)
    at = src.index("def generate_progress")
    body = src[at : at + 2200]
    assert (
        body.count('"generation_attempt": getattr(self, "_last_generate_attempt", None),') == 2
    ), f"{engine} does not publish the running attempt on both progress branches"


def test_an_attempt_specific_progress_answer_is_only_about_that_attempt():
    """The route narrows active, the step counter and the reason to the attempt asked about.

    Driven rather than read: the failure mode is a field left over from the global answer,
    which source-shape assertions are bad at catching.
    """
    import asyncio

    import routes.inference as route

    class _Engine:
        def __init__(self, payload):
            self.payload = payload

        def generate_progress(self):
            return dict(self.payload)

        def status(self):
            return {"loaded": True, "repo_id": "someone/model"}

    running_for_someone_else = {
        "active": True,
        "step": 7,
        "total_steps": 30,
        "fraction": 7 / 30,
        "eta_seconds": 12.0,
        "generation_attempt": "attempt-theirs",
    }

    def answer(payload, attempt_id):
        engine = _Engine(payload)
        original = route.account_access
        try:
            route.account_access = types.SimpleNamespace(
                managed_account = lambda: False,
                generation_is_foreign = lambda *_a, **_k: False,
                generation_is_mine = lambda *_a, **_k: True,
                resident_hidden = lambda *_a, **_k: False,
                hidden_generate_progress_response = lambda cls: cls(),
            )
            import core.inference.diffusion_engine_router as router

            original_get = router.get_active_diffusion_engine
            router.get_active_diffusion_engine = lambda: engine
            try:
                return asyncio.run(
                    route.diffusion_generate_progress(
                        attempt_id = attempt_id, current_subject = "owner"
                    )
                )
            finally:
                router.get_active_diffusion_engine = original_get
        finally:
            route.account_access = original

    mine = answer(running_for_someone_else, "attempt-mine")
    assert mine.active is False, "a run belonging to another attempt was reported as this caller's"
    assert (mine.step, mine.total_steps, mine.eta_seconds) == (
        0,
        0,
        None,
    ), "another attempt's step counter was reported as this caller's progress"

    theirs = answer(running_for_someone_else, "attempt-theirs")
    assert theirs.active is True, "the attempt that IS running must be told so"
    assert theirs.step == 7

    # And an unnamed poll still gets the global answer, which is what the progress bar and
    # an older client read.
    unnamed = answer(running_for_someone_else, None)
    assert unnamed.active is True and unnamed.step == 7


def test_a_retained_outcome_is_the_callers_own_account(monkeypatch):
    """Keyed by account as well as attempt, and answered before the guards that hide
    another account's generation.

    With managed accounts, a second account can start a generation between this caller's
    failure and its next poll. The foreign-generation guard answers idle and hidden, which a
    client that had already seen its own run active reads as success. Answering the named
    lookup first is only safe because the key is account-qualified, so this test pins both
    halves: the caller reads its own reason, and another account cannot read it.
    """
    from types import SimpleNamespace

    from core.inference.generate_outcomes import (
        _retain_generate_failure,
        generate_failure_for_attempt,
    )
    from utils.account_context import AccountContext, run_as

    engine = SimpleNamespace()
    ada = AccountContext("acct-a", "ada")
    bo = AccountContext("acct-b", "bo")

    run_as(ada, _retain_generate_failure, engine, "attempt-1", "CUDA out of memory")
    assert run_as(ada, generate_failure_for_attempt, engine, "attempt-1") == ("CUDA out of memory")
    assert (
        run_as(bo, generate_failure_for_attempt, engine, "attempt-1") is None
    ), "another account read an attempt's retained failure"
    # And the owner is a third scope again.
    assert generate_failure_for_attempt(engine, "attempt-1") is None

    # The route answers that lookup ahead of the hiding guards, or the caller never reaches
    # it while someone else is generating.
    src = _src("routes/inference.py")
    at = src.index("async def diffusion_generate_progress")
    body = src[at : at + 4500]
    lookup = body.index("generate_failure_for_attempt(get_active_diffusion_engine()")
    guard = body.index('account_access.generation_is_foreign("diffusion")')
    assert (
        lookup < guard
    ), "the named lookup runs after the guard that hides another account's generation"


def test_a_persisting_generation_counts_as_active_only_for_its_own_attempt():
    """The persist window is global; a poll naming an attempt is not.

    Records being written for any other Studio or OpenAI image request held the override on,
    so a settling caller saw active and took the end of that window for its own success,
    skipping the gallery proof.
    """
    import asyncio

    import routes.inference as route
    from core.inference.generate_outcomes import attempt_scope_key

    class _Idle:
        def generate_progress(self):
            return {
                "active": False,
                "step": 0,
                "total_steps": 0,
                "fraction": 0.0,
                "eta_seconds": None,
                "error": None,
                "generation_attempt": None,
            }

        def status(self):
            return {"loaded": True, "repo_id": "someone/model"}

    def answer(attempt_id):
        original = route.account_access
        try:
            route.account_access = types.SimpleNamespace(
                managed_account = lambda: False,
                generation_is_foreign = lambda *_a, **_k: False,
                generation_is_mine = lambda *_a, **_k: True,
                resident_hidden = lambda *_a, **_k: False,
                hidden_generate_progress_response = lambda cls: cls(),
            )
            import core.inference.diffusion_engine_router as router

            original_get = router.get_active_diffusion_engine
            router.get_active_diffusion_engine = lambda: _Idle()
            try:
                return asyncio.run(
                    route.diffusion_generate_progress(
                        attempt_id = attempt_id, current_subject = "owner"
                    )
                )
            finally:
                router.get_active_diffusion_engine = original_get
        finally:
            route.account_access = original

    original_count = route._diffusion_persist_active
    original_attempts = dict(route._diffusion_persist_attempts)
    try:
        # Somebody else's records are being written.
        route._diffusion_persist_active = 1
        route._diffusion_persist_attempts.clear()
        route._note_persisting_attempt(attempt_scope_key("attempt-theirs"), 1)

        assert (
            answer("attempt-mine").active is False
        ), "another attempt's persist window was reported as this caller's activity"
        assert (
            answer("attempt-theirs").active is True
        ), "the attempt whose records ARE being written must still be told so"
        # An unnamed poll keeps the global answer: that is what the reload mount probe reads.
        assert answer(None).active is True
    finally:
        route._diffusion_persist_active = original_count
        route._diffusion_persist_attempts.clear()
        route._diffusion_persist_attempts.update(original_attempts)
