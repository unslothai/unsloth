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
    body = src[at : at + 6000]
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
    # describe different moments. Searched from the clear, since the pre-admission handler
    # writes the same field earlier in the file for the raises that never reach the start.
    clear = src.index("self._last_generate_error = None")
    recorded = src.index("self._last_generate_attempt = attempt_id", clear)
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
    # Widened as the route grows: the window has to reach the end of this handler, or the
    # assertions below silently stop measuring anything.
    at = src.index("async def diffusion_generate_progress")
    body = src[at : at + 9000]
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
    import core.inference.generate_outcomes as outcomes
    from core.inference.generate_outcomes import (
        _RETAINED_GENERATE_FAILURES,
        _retain_generate_failure,
        generate_failure_for_attempt,
    )

    _retain_generate_failure("attempt-a", "CUDA out of memory")
    # B starts and fails; A has not polled yet.
    _retain_generate_failure("attempt-b", "model was replaced")
    assert (
        generate_failure_for_attempt("attempt-a") == "CUDA out of memory"
    ), "a later run discarded the reason the settling client is waiting for"
    assert generate_failure_for_attempt("attempt-b") == "model was replaced"
    # An attempt that never failed, or never ran, has nothing to report.
    assert generate_failure_for_attempt("attempt-c") is None
    assert generate_failure_for_attempt(None) is None
    assert generate_failure_for_attempt("") is None

    # Bounded, oldest first out: only a settling caller reads one, and it reads it within
    # seconds, so this cannot grow with uptime.
    for i in range(_RETAINED_GENERATE_FAILURES + 4):
        _retain_generate_failure(f"attempt-{i}", f"reason {i}")
    assert generate_failure_for_attempt("attempt-0") is None, "the bound is not enforced"
    assert generate_failure_for_attempt(f"attempt-{_RETAINED_GENERATE_FAILURES + 3}")

    # An id with no reason is not remembered at all, so a successful run leaves nothing.
    before = len(outcomes._OUTCOMES)
    _retain_generate_failure(None, "should not be kept")
    assert len(outcomes._OUTCOMES) == before


@pytest.mark.parametrize("engine", ENGINES)
def test_an_engine_records_the_failure_against_its_own_attempt(engine):
    """Both engines, at the point where they already retain the reason."""
    src = _src(engine)
    assert (
        "_retain_generate_failure(attempt_id," in src
    ), f"{engine} does not record the failure against the attempt that ran it"
    # Both places that set the reason: the handler inside the run and the helper for the
    # raises that precede it. Each has to record the per-attempt outcome beside it.
    at = 0
    seen = 0
    while True:
        try:
            retained = src.index("self._last_generate_error = str(exc) or type(exc).__name__", at)
        except ValueError:
            break
        recorded = src.index("_retain_generate_failure(", retained)
        assert (
            recorded - retained < 500
        ), f"{engine} records the per-attempt outcome away from where it retains the reason"
        seen += 1
        at = retained + 1
    assert seen >= 2, f"{engine} no longer records a reason for the raises before the run"


def test_the_progress_route_answers_about_the_attempt_it_was_asked_about():
    """Named, the answer is that attempt's; unnamed, the retained slot answers as before."""
    src = _src("routes/inference.py")
    at = src.index("async def diffusion_generate_progress")
    body = src[at : at + 6000]
    assert (
        "attempt_id: Optional[str] = Query(" in body
    ), "the progress route cannot be asked about a particular attempt"
    assert "generate_failure_for_attempt(attempt_id)" in body
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
        "_retain_generate_failure(attempt_id, DIFFUSION_CANCELLED_MSG)" in branch
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
                # None is the single-identity installation: there is nobody for the
                # engine's slot to belong to but the caller.
                account_scope = lambda: None,
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
    from core.inference.generate_outcomes import (
        _retain_generate_failure,
        generate_failure_for_attempt,
    )
    from utils.account_context import AccountContext, run_as

    ada = AccountContext("acct-a", "ada")
    bo = AccountContext("acct-b", "bo")

    run_as(ada, _retain_generate_failure, "attempt-scoped", "CUDA out of memory")
    assert run_as(ada, generate_failure_for_attempt, "attempt-scoped") == ("CUDA out of memory")
    assert (
        run_as(bo, generate_failure_for_attempt, "attempt-scoped") is None
    ), "another account read an attempt's retained failure"
    # And the owner is a third scope again.
    assert generate_failure_for_attempt("attempt-scoped") is None

    # The route answers that lookup ahead of the hiding guards, or the caller never reaches
    # it while someone else is generating.
    src = _src("routes/inference.py")
    at = src.index("async def diffusion_generate_progress")
    body = src[at : at + 6000]
    lookup = body.index("generate_failure_for_attempt(attempt_id)")
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
                # None is the single-identity installation: there is nobody for the
                # engine's slot to belong to but the caller.
                account_scope = lambda: None,
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


def test_a_retained_outcome_survives_an_engine_switch():
    """The store belongs to the process, not to the engine instance that ran the generation.

    A request can switch the active engine between diffusers and sd.cpp while a client is
    still settling a lost POST. Kept on the instance, that client's reason became
    unreachable the moment the other engine took over, and it was told either that its
    request never arrived or that another run going idle was its own success.
    """
    import core.inference.generate_outcomes as outcomes
    from core.inference.generate_outcomes import (
        _retain_generate_failure,
        generate_failure_for_attempt,
    )

    _retain_generate_failure("attempt-across-engines", "CUDA out of memory")
    # Whatever the active engine is now, the lookup takes no engine at all.
    assert generate_failure_for_attempt("attempt-across-engines") == "CUDA out of memory"
    assert not any(
        hasattr(engine, "_generate_outcomes") for engine in (outcomes, object())
    ), "the outcomes are stored on an engine instance again"

    src = _src("core/inference/generate_outcomes.py")
    assert (
        "_OUTCOMES" in src and "getattr(engine" not in src
    ), "the store reads from an engine instance again"
    # And the two engines record into it without passing themselves.
    for engine_src in ENGINES:
        assert "_retain_generate_failure(attempt_id," in _src(engine_src)


def test_a_cancelled_attempt_settles_as_a_cancellation_not_a_failure():
    """The sentinel survives classification, because it is the one reason that is not a failure.

    A POST that returns normally answers 409 with the sentinel and the page treats that as
    the requested outcome. A POST that was LOST has only the retained reason, and the generic
    classifier turned the sentinel into "Image generation failed.", so the user's own Stop
    toasted a failure.
    """
    from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG
    from routes.inference import _generate_failure_detail

    assert (
        _generate_failure_detail(DIFFUSION_CANCELLED_MSG) == DIFFUSION_CANCELLED_MSG
    ), "a cancellation read as a generic failure to the client settling a lost post"
    # Everything else still goes through the classifier, engine text included.
    named = _generate_failure_detail("CUDA out of memory. Tried to allocate 20.00 GiB")
    assert named.startswith("Image generation failed.")
    assert "20.00 GiB" not in named, "engine text escaped into a client-visible message"
    assert _generate_failure_detail("something nobody classified") == ("Image generation failed.")


def test_an_unscoped_poll_on_a_multi_account_install_is_not_told_someone_elses_reason():
    """The ENGINE's error slot is one per process; the keyed store is per account.

    A poll that names no attempt reads the engine slot, and the account guards above only
    hide a generation while it is ACTIVE. So once A's run had failed and left
    media_generation, B's unscoped poll was answered with A's classified reason and the
    attempt id that produced it. The keyed store is the authority: a reason this caller can
    look up is a reason this caller owns.
    """
    import asyncio
    import types

    import routes.inference as route
    from core.inference.generate_outcomes import _retain_generate_failure
    from utils.account_context import AccountContext, run_as

    class _Failed:
        def generate_progress(self):
            return {
                "active": False,
                "step": 0,
                "total_steps": 0,
                "fraction": 0.0,
                "eta_seconds": None,
                "error": "CUDA out of memory. Tried to allocate 20.00 GiB",
                "generation_attempt": "attempt-a",
            }

        def status(self):
            return {"loaded": True, "repo_id": "someone/model"}

    def answer(scope):
        original = route.account_access
        try:
            route.account_access = types.SimpleNamespace(
                managed_account = lambda: scope is not None,
                account_scope = lambda: scope,
                generation_is_foreign = lambda *_a, **_k: False,
                generation_is_mine = lambda *_a, **_k: True,
                resident_hidden = lambda *_a, **_k: False,
                hidden_generate_progress_response = lambda cls: cls(),
            )
            import core.inference.diffusion_engine_router as router

            original_get = router.get_active_diffusion_engine
            router.get_active_diffusion_engine = lambda: _Failed()
            try:
                return asyncio.run(
                    route.diffusion_generate_progress(attempt_id = None, current_subject = "someone")
                )
            finally:
                router.get_active_diffusion_engine = original_get
        finally:
            route.account_access = original

    ada = AccountContext("acct-a", "ada")
    bo = AccountContext("acct-b", "bo")
    run_as(ada, _retain_generate_failure, "attempt-a", "CUDA out of memory")

    mine = run_as(ada, answer, "acct-a")
    assert mine.error, "the account that ran the failed generation was told nothing"
    theirs = run_as(bo, answer, "acct-b")
    assert theirs.error is None, "an unscoped poll was answered with another account's failure"
    assert (
        theirs.generation_attempt is None
    ), "another account's attempt id came back with the empty reason"

    # A single-identity installation has nobody else for the slot to belong to, so the
    # legacy answer an older client depends on is unchanged.
    solo = answer(None)
    assert solo.error, "a single-account install lost the unscoped legacy answer"


def test_a_client_input_failure_is_not_reported_as_one_the_log_explains():
    """Classification hides WHICH failure it was, so the record has to carry it.

    The route answers a ValueError with its own reason and deliberately never logs it, and
    a caller settling a lost POST reads the RETAINED reason rather than that response. By
    then it wears the same "Image generation failed." prefix as an internal failure, so a
    page judging the message alone offered "View logs" for a failure no log can explain.
    """
    import core.inference.generate_outcomes as outcomes
    from core.inference.generate_outcomes import (
        _retain_generate_failure,
        generate_failure_for_attempt,
        generate_failure_was_logged,
        mark_generate_failure_unlogged,
    )

    _retain_generate_failure("attempt-logged", "CUDA out of memory")
    assert (
        generate_failure_was_logged("attempt-logged") is True
    ), "the engine's own retention defaults to logged, which is what its handlers do"

    _retain_generate_failure("attempt-input", "negative_prompt is not supported")
    mark_generate_failure_unlogged("attempt-input")
    assert generate_failure_was_logged("attempt-input") is False
    # The reason itself is untouched: the client still gets told why.
    assert generate_failure_for_attempt("attempt-input") == "negative_prompt is not supported"
    # Nothing retained says nothing either way.
    assert generate_failure_was_logged("attempt-never-ran") is None
    assert generate_failure_was_logged(None) is None
    # And marking an attempt nothing is held for does not invent a record.
    before = len(outcomes._OUTCOMES)
    mark_generate_failure_unlogged("attempt-never-ran")
    assert len(outcomes._OUTCOMES) == before


def test_the_route_marks_and_publishes_whether_the_failure_was_logged():
    """Wiring, from the route's own source: the 400 branch marks, the poll publishes."""
    src = _src("routes/inference.py")
    at = src.index("async def generate_diffusion_image")
    body = src[at : at + 6000]
    assert (
        "mark_generate_failure_unlogged(request.attempt_id)" in body
    ), "the branch that answers without logging leaves the record saying it logged"

    at = src.index("async def diffusion_generate_progress")
    body = src[at : at + 7000]
    assert '"error_logged"' in body, "the poll does not say whether the reason was logged"
    assert "generate_failure_was_logged(attempt_id)" in body


def test_a_persist_failure_is_retained_for_the_client_that_cannot_read_the_response():
    """The one failure raised AFTER the attempt was reported active.

    A settling client whose POST was lost watches its attempt go active-to-idle and takes
    that for success, and the images it never saved are not in the gallery to contradict it.
    So the reason has to be readable from the progress poll, recorded before the finally
    drops the persist marker.
    """
    src = _src("routes/inference.py")
    at = src.index('logger.error("diffusion.persist_failed')
    window = src[at : at + 1600]
    assert (
        "_retain_generate_failure(request.attempt_id, _PERSIST_FAILURE_MSG)" in window
    ), "a persist failure leaves the settling client reading its lost run as a success"
    # Before the finally, which drops the marker this attempt was active under.
    retained = src.index("_retain_generate_failure(request.attempt_id", at)
    released = src.index("_note_persisting_attempt(persisting_attempt, -1)", at)
    assert retained < released, "the failure is recorded after the attempt stops being active"
    # Retained as LOGGED, since the line above is the log and the disk error is only there.
    assert "logged = False" not in window


def _answer_progress(payload, attempt_id):
    """Drive diffusion_generate_progress against *payload* on a single-identity install."""
    import asyncio

    import routes.inference as route

    class _Engine:
        def generate_progress(self):
            return dict(payload)

        def status(self):
            return {"loaded": True, "repo_id": "someone/model"}

    original = route.account_access
    try:
        route.account_access = types.SimpleNamespace(
            managed_account = lambda: False,
            account_scope = lambda: None,
            generation_is_foreign = lambda *_a, **_k: False,
            generation_is_mine = lambda *_a, **_k: True,
            resident_hidden = lambda *_a, **_k: False,
            hidden_generate_progress_response = lambda cls: cls(),
        )
        import core.inference.diffusion_engine_router as router

        original_get = router.get_active_diffusion_engine
        router.get_active_diffusion_engine = lambda: _Engine()
        try:
            return asyncio.run(
                route.diffusion_generate_progress(attempt_id = attempt_id, current_subject = "owner")
            )
        finally:
            router.get_active_diffusion_engine = original_get
    finally:
        route.account_access = original


def test_a_queued_attempt_is_pending_not_absent():
    """The engine cannot name an attempt until it holds the generation slot.

    A second run queued behind an active one therefore answered a named poll with active
    False, and a settling client whose POST was lost read that as "the request never
    arrived", or took the running run's newly saved record for proof that its own finished.
    Driven, not read: the failure mode is a field carrying the global answer.
    """
    import routes.inference as route
    from core.inference.generate_outcomes import attempt_scope_key

    running_for_someone_else = {
        "active": True,
        "step": 7,
        "total_steps": 30,
        "fraction": 7 / 30,
        "eta_seconds": 12.0,
        "generation_attempt": "attempt-theirs",
    }

    key = attempt_scope_key("attempt-queued")
    route._note_queued_attempt(key, 1)
    try:
        queued = _answer_progress(running_for_someone_else, "attempt-queued")
    finally:
        route._note_queued_attempt(key, -1)
    assert queued.active is True, "a queued attempt was reported as one that never arrived"
    # Pending, not progressing: the running run's step counter is still not this caller's.
    assert (queued.step, queued.total_steps, queued.eta_seconds) == (0, 0, None)

    # Once the request is gone the attempt really is absent, and the answer goes back.
    absent = _answer_progress(running_for_someone_else, "attempt-queued")
    assert absent.active is False, "the marker outlived the request that held it"


def test_a_reloaded_page_hears_about_a_persist_failure():
    """A mount probe after a reload has lost its attempt id.

    The keyed store is reachable only by id, so a persist failure recorded there alone was
    invisible to the resumed poll: it saw an error-free idle state and refreshed a gallery
    that had not changed, and the user was never told saving had failed.
    """
    import routes.inference as route

    class _Backend:
        pass

    backend = _Backend()
    route._note_unscoped_generate_failure(
        backend, "attempt-reloaded", "Failed to save the generated image."
    )
    assert backend._last_generate_error == "Failed to save the generated image."
    assert backend._last_generate_attempt == "attempt-reloaded"

    idle_after_the_failure = {
        "active": False,
        "step": 0,
        "total_steps": 0,
        "fraction": 0.0,
        "eta_seconds": None,
        "error": backend._last_generate_error,
        "generation_attempt": backend._last_generate_attempt,
    }
    resumed = _answer_progress(idle_after_the_failure, None)
    assert (
        resumed.error == "Failed to save the generated image."
    ), "a reloaded page read a failed save as a finished generation"
    assert resumed.error_logged is True, "the log that holds the disk error is not offered"

    # Wired at the persist failure, and before the marker that made the attempt active drops.
    src = _src("routes/inference.py")
    at = src.index('logger.error("diffusion.persist_failed')
    window = src[at : at + 1600]
    assert "_note_unscoped_generate_failure(" in window


def test_an_unscoped_poll_reads_the_log_flag_of_the_attempt_it_is_told_about():
    """A reloaded page polls without an attempt id, and the record still knows.

    A client-input failure is answered with its own reason and never logged, and the route
    marks the record so. Looking the flag up under None answered "logged" anyway, so the
    reloaded page offered a log that cannot hold that failure.
    """
    from core.inference.generate_outcomes import (
        _retain_generate_failure,
        mark_generate_failure_unlogged,
    )

    _retain_generate_failure("attempt-unlogged", "Image generation failed. Bad size.")
    mark_generate_failure_unlogged("attempt-unlogged")
    idle = {
        "active": False,
        "step": 0,
        "total_steps": 0,
        "fraction": 0.0,
        "eta_seconds": None,
        "error": "Image generation failed. Bad size.",
        "generation_attempt": "attempt-unlogged",
    }
    resumed = _answer_progress(idle, None)
    assert resumed.error, "the reason itself was dropped"
    assert (
        resumed.error_logged is False
    ), "an unscoped poll was offered the log of a failure that was never logged"

    # A failure that WAS logged still gets its log through the same channel.
    _retain_generate_failure("attempt-logged", "boom")
    logged = _answer_progress({**idle, "generation_attempt": "attempt-logged"}, None)
    assert logged.error_logged is True


def test_a_failure_class_is_matched_as_a_whole_word():
    """Every exception class reaches this classifier now, so its needles have to be tokens.

    "oom" as a substring made any message containing "boom", "bathroom" or "zoom" report
    that the device had run out of memory, and told the user to shrink an image that was
    never too big.
    """
    from routes.inference import _GENERATE_FAILURE_FALLBACK, _generate_failure_detail

    for innocent in (
        "the sandbox blew up with a boom",
        "no space left in the bathroom volume",
        "zoom factor must be positive",
    ):
        assert (
            _generate_failure_detail(innocent) == _GENERATE_FAILURE_FALLBACK
        ), f"{innocent!r} was reported as an out-of-memory failure"

    # And the real thing, in each of the spellings the engines actually produce.
    for real in (
        "CUDA out of memory. Tried to allocate 20.00 GiB",
        "torch.OutOfMemoryError: OOM",
        "MPS backend out of memory",
    ):
        assert "ran out of memory" in _generate_failure_detail(
            real
        ), f"a real out-of-memory failure stopped being named: {real!r}"


def test_a_new_execution_supersedes_the_outcome_retained_under_its_id():
    """The Tauri client retries a dropped POST with the same body, so the same attempt id.

    A first execution's retained failure then outranked the retry, which can be queued,
    running or already successful, and the settling client declared the attempt failed and
    ignored the images the retry had produced.
    """
    from core.inference.generate_outcomes import (
        _retain_generate_failure,
        clear_generate_failure,
        generate_failure_for_attempt,
    )

    _retain_generate_failure("attempt-retried", "CUDA out of memory")
    assert generate_failure_for_attempt("attempt-retried") == "CUDA out of memory"
    clear_generate_failure("attempt-retried")
    assert (
        generate_failure_for_attempt("attempt-retried") is None
    ), "a retry of the same attempt still reads the previous execution's failure"
    # An id that was never retained, and no id at all, are both no-ops rather than errors.
    clear_generate_failure("attempt-never-seen")
    clear_generate_failure(None)

    # Wired where the execution takes the id, beside the marker that makes it pending.
    src = _src("routes/inference.py")
    at = src.index("_note_queued_attempt(queued_attempt, 1)")
    assert (
        "clear_generate_failure(request.attempt_id)" in src[at : at + 700]
    ), "a new execution does not supersede the outcome retained under its id"


def test_a_live_execution_outranks_an_outcome_retained_under_its_id():
    """A Tauri retry reuses the id, and the predecessor can fail after the retry starts.

    The early clear happens when the retry takes the id, so a predecessor failing later
    retains an outcome under an id the retry owns; the progress route answered from that
    retained outcome before looking at the markers, so the client reported failure and
    stopped settling while the retry was queued or running.
    """
    import routes.inference as route
    from core.inference.generate_outcomes import _retain_generate_failure, attempt_scope_key

    _retain_generate_failure("attempt-shared", "CUDA out of memory")
    idle = {
        "active": False,
        "step": 0,
        "total_steps": 0,
        "fraction": 0.0,
        "eta_seconds": None,
        "error": None,
        "generation_attempt": None,
    }

    key = attempt_scope_key("attempt-shared")
    route._note_queued_attempt(key, 1)
    try:
        assert route._attempt_execution_is_live("attempt-shared") is True
        pending = _answer_progress(idle, "attempt-shared")
        assert (
            pending.error is None
        ), "the predecessor's failure answered for an execution that is still running"
        assert pending.active is True, "a queued retry was reported as absent"
    finally:
        route._note_queued_attempt(key, -1)

    # With no execution holding the id, the retained outcome answers again, which is what a
    # settling client whose run really did fail needs.
    settled = _answer_progress(idle, "attempt-shared")
    assert settled.error, "a genuinely failed attempt stopped being told its reason"

    # And the persist window counts as live too, so the gap between them is covered.
    route._note_persisting_attempt(key, 1)
    try:
        assert route._attempt_execution_is_live("attempt-shared") is True
    finally:
        route._note_persisting_attempt(key, -1)
    assert route._attempt_execution_is_live("attempt-shared") is False

    # The successful path clears the stale outcome once its own run has happened.
    src = _src("routes/inference.py")
    persist_at = src.index("def _persist() -> list[dict]:")
    assert (
        "_clear_outcome(request.attempt_id)" in src[persist_at - 900 : persist_at]
    ), "a retry that ran leaves its predecessor's failure to resurface"


def test_a_stalled_persist_does_not_answer_for_a_retry_that_already_saved():
    """Persist runs after the engine slot is released, so two executions can overlap.

    A Tauri retry of one lost POST can generate and persist successfully while the original's
    persist stalls; the original failing afterwards recorded that failure under the shared id,
    so the settling client reported failure while the retry's images sat in the gallery.
    """
    import routes.inference as route

    first = next(route._diffusion_execution_serial)
    retry = next(route._diffusion_execution_serial)
    key = "acct\x00attempt-overlapped"

    assert (
        route._attempt_run_was_superseded(key, first) is False
    ), "nothing has finished yet, so nothing is superseded"
    # The retry's run finishes first.
    route._note_attempt_run_finished(key, retry)
    try:
        assert (
            route._attempt_run_was_superseded(key, first) is True
        ), "the original execution still answers after a later one finished its run"
        assert (
            route._attempt_run_was_superseded(key, retry) is False
        ), "the newest execution was told it had been superseded by itself"
        # An id nobody is sharing is unaffected, and no id at all is never superseded.
        assert route._attempt_run_was_superseded("acct\x00attempt-alone", first) is False
        assert route._attempt_run_was_superseded(None, first) is False
        # An older serial cannot move the mark backwards.
        route._note_attempt_run_finished(key, first)
        assert route._diffusion_attempt_last_run[key] == retry
    finally:
        route._diffusion_attempt_last_run.pop(key, None)

    # Wired: the persist failure records nothing on either channel once superseded.
    src = _src("routes/inference.py")
    at = src.index('logger.error("diffusion.persist_failed')
    window = src[at : at + 1600]
    assert (
        "if not _attempt_run_was_superseded(persisting_attempt, execution_serial):" in window
    ), "a stalled persist still answers for a retry that already saved its images"
    guard = window.index("_attempt_run_was_superseded")
    assert guard < window.index("_retain_generate_failure"), "the guard runs after the record"
    assert guard < window.index(
        "_note_unscoped_generate_failure"
    ), "the unscoped slot is written even when a later execution has superseded this one"
    # And a finished run marks itself, or nothing is ever superseded.
    assert "_note_attempt_run_finished(queued_attempt, execution_serial)" in src


def test_the_finished_run_bookkeeping_is_bounded():
    """Every generation supplies a fresh id, so this cannot grow per generation forever.

    The serial only matters while a duplicate of the SAME id is still persisting, which is
    seconds, so an evicted key is an id nothing is racing over any more.
    """
    import routes.inference as route

    saved = dict(route._diffusion_attempt_last_run)
    try:
        route._diffusion_attempt_last_run.clear()
        for index in range(route._RETAINED_ATTEMPT_RUNS + 10):
            route._note_attempt_run_finished(f"acct\x00attempt-{index}", index + 1)
        assert (
            len(route._diffusion_attempt_last_run) == route._RETAINED_ATTEMPT_RUNS
        ), "the finished-run bookkeeping grows without bound"
        # Oldest out first: the newest ids, which are the ones a retry could still share,
        # are the ones kept.
        assert "acct\x00attempt-0" not in route._diffusion_attempt_last_run
        newest = f"acct\x00attempt-{route._RETAINED_ATTEMPT_RUNS + 9}"
        assert newest in route._diffusion_attempt_last_run
        # An id with no attempt is not stored at all.
        route._note_attempt_run_finished(None, 1)
        assert None not in route._diffusion_attempt_last_run
    finally:
        route._diffusion_attempt_last_run.clear()
        route._diffusion_attempt_last_run.update(saved)


def test_clearing_a_retained_outcome_takes_the_same_lock():
    """Every other mutation of the store holds it, and the bound is read-then-pop.

    Unlocked, a clear could land between the eviction test in _retain_generate_failure and
    its popitem, dropping one MORE outcome than the bound calls for and leaving that client
    reading its failed generation as absent.
    """
    import inspect

    from core.inference import generate_outcomes

    src = inspect.getsource(generate_outcomes.clear_generate_failure)
    assert "with _OUTCOMES_LOCK:" in src, "the outcome store is mutated without its lock"
    assert src.index("with _OUTCOMES_LOCK:") < src.index(
        "_OUTCOMES.pop("
    ), "the pop happens outside the lock"
    # Still does its job.
    generate_outcomes._retain_generate_failure("attempt-locked-clear", "boom")
    assert generate_outcomes.generate_failure_for_attempt("attempt-locked-clear") == "boom"
    generate_outcomes.clear_generate_failure("attempt-locked-clear")
    assert generate_outcomes.generate_failure_for_attempt("attempt-locked-clear") is None


@pytest.mark.parametrize("engine", ENGINES)
def test_a_failure_before_the_run_starts_is_retained_too(engine):
    """The handler inside generate cannot see the raises that precede it.

    Cancelled while queued for the generation slot, no model loaded, a dead resident server,
    a superseding load: each raises before the try that retains, so a client whose POST was
    lost found nothing recorded and settleLostGeneration reported that its request never
    reached the server rather than the cancellation that actually happened.
    """
    src = _src(engine)
    assert (
        "def _retained_generate_failure(self, exc, attempt_id):" in src
    ), f"{engine} has no way to record a failure raised before the run starts"
    helper = src[src.index("def _retained_generate_failure") :][:900]
    assert "_retain_generate_failure(attempt_id, self._last_generate_error" in helper
    # The attempt too, or the reason is attributed to whichever attempt ran last.
    assert (
        "self._last_generate_attempt = attempt_id" in helper
    ), f"{engine} records a pre-run reason without the attempt it belongs to"
    # Every pre-run raise goes through it: a bare one would be the hole again.
    body = src[src.index("    def generate(") :]
    body = body[: body.index("    def generate_progress(")]
    head = body[: body.index("            try:")]
    # The slot's own admission raises count as pre-run too, and are where the cancelled-in-
    # queue case actually lives.
    if "def _generation_slot(" in src:
        at = src.index("def _generation_slot(")
        head += src[at : src.index("    def ", at + 10)]
    for bare in (
        "raise RuntimeError(DIFFUSION_NOT_LOADED_MSG)",
        "raise RuntimeError(DIFFUSION_CANCELLED_MSG)",
        "raise DiffusionModelReplacedError(",
    ):
        assert bare not in head, f"{engine} still raises {bare} without retaining it"


def test_the_cancelled_slot_wait_records_the_reason():
    """Driven: a generation cancelled while queued for the slot leaves its reason behind."""
    import threading

    from core.inference import diffusion as diffusion_module
    from core.inference.generate_outcomes import (
        clear_generate_failure,
        generate_failure_for_attempt,
    )

    engine = diffusion_module.DiffusionBackend.__new__(diffusion_module.DiffusionBackend)
    engine._generation_cancel_lock = threading.Lock()
    engine._queued_generate_cancels = set()
    engine._generate_lock = threading.Lock()
    engine._lock = threading.Lock()
    cancel = threading.Event()
    cancel.set()

    clear_generate_failure("attempt-cancelled-in-queue")
    with pytest.raises(RuntimeError):
        with engine._generation_slot(cancel, attempt_id = "attempt-cancelled-in-queue"):
            pass
    try:
        assert (
            generate_failure_for_attempt("attempt-cancelled-in-queue")
            == diffusion_module.DIFFUSION_CANCELLED_MSG
        ), "a generation cancelled while queued recorded nothing for its attempt"
        assert engine._last_generate_attempt == "attempt-cancelled-in-queue"
    finally:
        clear_generate_failure("attempt-cancelled-in-queue")


def test_a_saved_generation_clears_what_was_retained_under_its_id():
    """Images exist, so nothing under this id is a failure of this request any more.

    A duplicate execution of one retried POST can fail while this one is still writing, and
    its reason would then answer a settling client whose images are already in the gallery.
    """
    src = _src("routes/inference.py")
    at = src.index("records = await asyncio.to_thread(_persist)")
    window = src[at : at + 600]
    assert (
        "_clear_outcome(request.attempt_id)" in window
    ), "a successful save leaves a duplicate execution's failure to answer for it"
    # After the save, not before it: clearing first would drop the reason of the execution
    # that is still the newest if this save then fails.
    assert window.index("_clear_outcome") > window.index("_persist)")


def test_a_queued_attempt_survives_the_foreign_run_guard():
    """The guards answer idle for anything foreign, including our own queued request.

    A managed account's attempt can be queued behind ANOTHER account's generation; a poll
    naming it was answered with the hidden idle response, and a settling client whose POST
    was lost read that as "the request never arrived" while it was still queued and would
    later spend GPU time and create images.
    """
    import asyncio
    import types as _types

    import routes.inference as route
    from core.inference.generate_outcomes import attempt_scope_key
    from models.inference import DiffusionGenerateProgressResponse

    running_for_another_account = {
        "active": True,
        "step": 11,
        "total_steps": 30,
        "fraction": 11 / 30,
        "eta_seconds": 9.0,
        "generation_attempt": "attempt-theirs",
    }

    class _Engine:
        def generate_progress(self):
            return dict(running_for_another_account)

        def status(self):
            return {"loaded": True, "repo_id": "someone/model"}

    def answer(attempt_id):
        original = route.account_access
        try:
            route.account_access = _types.SimpleNamespace(
                managed_account = lambda: True,
                account_scope = lambda: "acct-mine",
                # Someone ELSE is generating.
                generation_is_foreign = lambda *_a, **_k: True,
                generation_is_mine = lambda *_a, **_k: False,
                resident_hidden = lambda *_a, **_k: True,
                hidden_generate_progress_response = lambda cls: cls(
                    active = False, step = 0, total_steps = 0, fraction = 0.0
                ),
            )
            import core.inference.diffusion_engine_router as router

            original_get = router.get_active_diffusion_engine
            router.get_active_diffusion_engine = lambda: _Engine()
            try:
                return asyncio.run(
                    route.diffusion_generate_progress(attempt_id = attempt_id, current_subject = "mine")
                )
            finally:
                router.get_active_diffusion_engine = original_get
        finally:
            route.account_access = original

    hidden = answer("attempt-queued-mine")
    assert isinstance(hidden, DiffusionGenerateProgressResponse)
    assert hidden.active is False, "nothing is queued yet, so the guards still hide the run"

    key = attempt_scope_key("attempt-queued-mine")
    route._note_queued_attempt(key, 1)
    try:
        pending = answer("attempt-queued-mine")
        assert (
            pending.active is True
        ), "a queued attempt of the caller's own was hidden as someone else's run"
        # And nothing of the foreign run leaks with it.
        assert (pending.step, pending.total_steps, pending.eta_seconds) == (
            0,
            0,
            None,
        ), "another account's step counter was published to this caller"
        assert pending.error is None
    finally:
        route._note_queued_attempt(key, -1)

    # Someone else's attempt id gets nothing from the marker: the key is account-qualified,
    # so a foreign poll cannot even name it.
    assert answer("attempt-theirs").active is False


def test_a_successful_save_clears_the_unscoped_reason():
    """That slot means "the last thing that happened", and the engine clears it at a START.

    So a persist failure written after a newer run began outlived it: the newer run could
    generate and save successfully and an unscoped probe still read the older failure.
    """
    import routes.inference as route

    class _Backend:
        pass

    backend = _Backend()
    route._note_unscoped_generate_failure(
        backend, "attempt-older", "Failed to save the generated image."
    )
    route._clear_unscoped_generate_failure(backend)
    assert (
        backend._last_generate_error is None
    ), "an older run's failure outlived a newer run that saved successfully"
    # The attempt is left alone: a run in flight publishes it, and it is only read beside a
    # reason.
    assert backend._last_generate_attempt == "attempt-older"

    # Wired at the successful save, and only where a NEWER execution has not already
    # described itself in the slot, which would be the same race the other way round.
    src = _src("routes/inference.py")
    at = src.index("records = await asyncio.to_thread(_persist)")
    window = src[at : at + 1200]
    assert (
        "_clear_unscoped_generate_failure(backend)" in window
    ), "a successful save leaves an older run's failure in the unscoped slot"
    guard = window.index("_unscoped_slot_is_newer_than(execution_serial)")
    assert guard < window.index(
        "_clear_unscoped_generate_failure("
    ), "an older save wipes the failure of a run that started after it"


def test_the_unscoped_slot_belongs_to_the_newest_execution():
    """One slot, two executions that can overlap, and the engine clears it only at a START.

    So whoever finishes LAST used to win: an older save wiped a newer failure, and an older
    failure hid a newer success. The serial decides instead.
    """
    import routes.inference as route

    saved = route._diffusion_unscoped_slot_serial
    try:
        older = next(route._diffusion_execution_serial)
        newer = next(route._diffusion_execution_serial)

        assert route._unscoped_slot_is_newer_than(older) is False, "nothing has claimed it yet"
        route._note_unscoped_slot_serial(newer)
        assert (
            route._unscoped_slot_is_newer_than(older) is True
        ), "an older execution still gets to speak for the slot"
        assert (
            route._unscoped_slot_is_newer_than(newer) is False
        ), "the newest execution was locked out of its own slot"
        # Monotone: an older claim cannot take it back.
        route._note_unscoped_slot_serial(older)
        assert route._unscoped_slot_is_newer_than(older) is True
    finally:
        route._diffusion_unscoped_slot_serial = saved

    # Both writes are guarded by it, and the engine's own write is stamped for the execution
    # whose call it happened in.
    src = _src("routes/inference.py")
    at = src.index('logger.error("diffusion.persist_failed')
    window = src[at : at + 1600]
    guard = window.index("_unscoped_slot_is_newer_than(execution_serial)")
    assert guard < window.index(
        "_note_unscoped_generate_failure("
    ), "an older persist failure still overwrites a newer execution's outcome"
    assert (
        "_note_unscoped_slot_serial(execution_serial)"
        in src[src.index("_note_queued_attempt(queued_attempt, -1)") :][:400]
    ), "the engine's own write is not attributed to the execution it happened in"


def test_a_validation_failure_is_recorded_and_marked_unlogged():
    """The pre-lock guards raise before the engine's handler, so nothing was recorded.

    Marking corrects a record; it does not create one, so a settling client whose POST was
    lost found nothing and was told its request never reached the server, when the answer was
    a validation refusal the route had already written.
    """
    src = _src("routes/inference.py")
    # The generate route's own handler, not the first ValueError in the file.
    at = src.index("_retain_generate_failure(request.attempt_id, str(exc))")
    window = src[at - 900 : at + 900]
    assert (
        "except ValueError as exc:" in window
    ), "the retained refusal is not the generate route's validation branch"
    assert (
        "_retain_generate_failure(request.attempt_id, str(exc))" in window
    ), "a validation refusal records nothing for a client that cannot read the response"
    assert (
        "mark_generate_failure_unlogged(request.attempt_id)" in window
    ), "the recorded refusal claims a log the route never wrote"
    assert window.index("_retain_generate_failure(") < window.index(
        "mark_generate_failure_unlogged("
    ), "the mark runs before there is a record to mark"

    # Driven: retained then marked leaves a reason the poll can read and no log offered.
    from core.inference.generate_outcomes import (
        _retain_generate_failure,
        clear_generate_failure,
        generate_failure_for_attempt,
        generate_failure_was_logged,
        mark_generate_failure_unlogged,
    )

    clear_generate_failure("attempt-bad-size")
    _retain_generate_failure("attempt-bad-size", "width must be a multiple of 8")
    mark_generate_failure_unlogged("attempt-bad-size")
    try:
        assert generate_failure_for_attempt("attempt-bad-size") == ("width must be a multiple of 8")
        assert generate_failure_was_logged("attempt-bad-size") is False
    finally:
        clear_generate_failure("attempt-bad-size")


@pytest.mark.parametrize("engine", ENGINES)
def test_a_client_state_failure_is_retained_as_unlogged(engine):
    """Cancelled, not loaded, a superseding load: every one is answered WITHOUT logging.

    Retained as logged, the settling poll published error_logged true and the page offered
    the log for an event that is not in it.
    """
    src = _src(engine)
    helper = src[src.index("def _retained_generate_failure") :][:1200]
    assert (
        "logged = False" in helper
    ), f"{engine} records a client-state failure as one the log can hold"
