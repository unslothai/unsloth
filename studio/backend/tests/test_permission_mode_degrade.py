# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

""" "off" (Run automatically) asks before a risky python/terminal call without OS isolation."""

import itertools
import os
import socket
import sys
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.inference import os_sandbox, sandbox_probe
from state import tool_policy

MODES = ("ask", "auto", "off", "full")
TOOLS = ("python", "terminal", "web_search", "edit_file")
ISOLATION = (True, False, None)


def _risk(name, arguments):
    return bool(arguments.get("risky"))


def _never(name):
    return name == "search_conversation"


def _expected(mode, tool, isolated, risky):
    if mode == "full":
        return False
    if mode == "ask":
        return True
    if mode == "auto":
        return risky
    return tool in ("python", "terminal") and isolated is not True and risky


@pytest.fixture
def isolation(monkeypatch):
    state = {}
    monkeypatch.setattr(os_sandbox, "cached_tool_isolation", lambda tool: state.get(tool))
    return state


@pytest.mark.parametrize(
    "mode,tool,isolated,risky", list(itertools.product(MODES, TOOLS, ISOLATION, (True, False)))
)
def test_decision_matrix(isolation, mode, tool, isolated, risky):
    isolation["python"] = isolation["terminal"] = isolated
    got = tool_policy.needs_tool_confirmation(
        confirm_tool_calls = True,
        bypass_permissions = mode == "full",
        permission_mode = mode,
        name = tool,
        arguments = {"risky": risky},
        is_high_risk = _risk,
        never_needs = _never,
    )
    assert got is _expected(mode, tool, isolated, risky)


@pytest.mark.parametrize("mode", MODES)
def test_an_unarmed_gate_never_asks(isolation, mode):
    isolation["python"] = False
    assert not tool_policy.needs_tool_confirmation(
        confirm_tool_calls = False,
        bypass_permissions = False,
        permission_mode = mode,
        name = "python",
        arguments = {"risky": True},
        is_high_risk = _risk,
        never_needs = _never,
    )


@pytest.mark.parametrize("mode", ("ask", "auto", "off"))
def test_search_conversation_never_asks(isolation, mode):
    assert not tool_policy.needs_tool_confirmation(
        confirm_tool_calls = True,
        bypass_permissions = False,
        permission_mode = mode,
        name = "search_conversation",
        arguments = {"risky": True},
        is_high_risk = _risk,
        never_needs = _never,
    )


def test_default_classifiers_are_the_tools_module_ones(isolation):
    isolation["python"] = False
    common = dict(confirm_tool_calls = True, bypass_permissions = False, permission_mode = "off")
    assert not tool_policy.needs_tool_confirmation(
        name = "python", arguments = {"code": "print(1)"}, **common
    )
    assert tool_policy.needs_tool_confirmation(
        name = "python", arguments = {"code": 'open("/etc/shadow").read()'}, **common
    )


@pytest.mark.parametrize("isolated", ISOLATION)
def test_may_prompt_before_arguments(isolation, isolated):
    isolation["python"] = isolation["terminal"] = isolated
    kw = dict(confirm_tool_calls = True, bypass_permissions = False)
    assert tool_policy.tool_call_may_prompt(permission_mode = "ask", name = "render_html", **kw)
    assert not tool_policy.tool_call_may_prompt(permission_mode = "off", name = "render_html", **kw)
    assert tool_policy.tool_call_may_prompt(permission_mode = "off", name = "python", **kw) is (
        isolated is not True
    )
    assert not tool_policy.tool_call_may_prompt(
        permission_mode = "ask", name = "python", confirm_tool_calls = True, bypass_permissions = True
    )


class _Payload:
    def __init__(self, **kw):
        self.permission_mode = kw.get("permission_mode")
        self.bypass_permissions = kw.get("bypass_permissions", False)
        self.stream = kw.get("stream", True)


@pytest.mark.parametrize(
    "mode,stream,ui_events,bypass,armed",
    [
        ("off", True, True, False, True),
        ("off", False, True, False, False),
        ("off", True, False, False, False),
        ("off", True, True, True, False),
        ("auto", True, True, False, False),
        ("ask", True, True, False, False),
        (None, True, True, False, False),
    ],
)
def test_off_gate_is_armed_only_where_a_prompt_can_reach_the_caller(
    mode, stream, ui_events, bypass, armed
):
    import routes.inference as inference_route
    payload = _Payload(permission_mode = mode, stream = stream, bypass_permissions = bypass)
    assert inference_route._off_mode_sandbox_gate(payload, ui_events) is armed


@pytest.mark.parametrize(
    "body,armed",
    [
        ({}, True),
        ({"confirm_tool_calls": False}, False),
        ({"confirm_tool_calls": True}, True),
    ],
)
def test_an_explicit_confirm_opt_out_keeps_off_unprompted(body, armed):
    import routes.inference as inference_route
    from models.inference import ChatCompletionRequest

    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "hi"}],
        stream = True,
        permission_mode = "off",
        **body,
    )
    assert payload.confirm_tool_calls is False
    assert inference_route._off_mode_sandbox_gate(payload, True) is armed


def _gguf_client(monkeypatch, captured):
    import routes.inference as inference_route
    from .llama_backend_double import FakeLlamaCppBackend

    class _Backend(FakeLlamaCppBackend):
        supports_tools = True
        context_length = 8192

        def generate_chat_completion_with_tools(self, **kwargs):
            captured.update(kwargs)
            yield {"type": "content", "text": "done"}
            yield {
                "type": "metadata",
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                "timings": {"prompt_n": 3, "predicted_n": 1},
                "finish_reason": "stop",
            }

    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _Backend())
    monkeypatch.setattr(inference_route, "_effective_enable_tools", lambda payload: True)

    async def _select(payload, **_kwargs):
        return [{"type": "function", "function": {"name": "python"}}]

    monkeypatch.setattr(inference_route, "_select_request_tools", _select)
    app = FastAPI()
    app.include_router(inference_route.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


@pytest.mark.parametrize(
    "stream,headers,armed",
    [
        (True, {"X-Unsloth-Events": "1"}, True),
        (False, {}, False),
    ],
)
def test_gguf_route_hands_the_loop_an_armed_gate_for_off(monkeypatch, stream, headers, armed):
    captured = {}
    response = _gguf_client(monkeypatch, captured).post(
        "/chat/completions",
        json = {
            "messages": [{"role": "user", "content": "run it"}],
            "stream": stream,
            "enable_tools": True,
            "permission_mode": "off",
        },
        headers = headers,
    )
    assert response.status_code == 200, response.text
    assert captured["permission_mode"] == "off"
    assert captured["confirm_tool_calls"] is armed
    assert captured["bypass_permissions"] is False


def test_cached_isolation_never_blocks_and_refreshes_once(monkeypatch):
    monkeypatch.delenv(os_sandbox.WARMUP_DISABLE_ENV, raising = False)
    release = threading.Event()
    calls = []

    def slow_refresh(tool, *, force = False):
        calls.append(tool)
        release.wait(5)
        os_sandbox.note_tool_isolation(tool, True, backend = "bubblewrap")
        return True

    monkeypatch.setattr(os_sandbox, "refresh_tool_isolation", slow_refresh)
    started = time.monotonic()
    assert os_sandbox.cached_tool_isolation("python") is None
    assert os_sandbox.cached_tool_isolation("python") is None
    assert time.monotonic() - started < 1.0
    release.set()
    deadline = time.monotonic() + 5
    while os_sandbox.cached_tool_isolation("python") is not True and time.monotonic() < deadline:
        time.sleep(0.01)
    assert os_sandbox.cached_tool_isolation("python") is True
    assert calls == ["python"]


def test_cached_isolation_starts_nothing_when_background_probes_are_off(monkeypatch):
    monkeypatch.setenv(os_sandbox.WARMUP_DISABLE_ENV, "1")
    monkeypatch.setattr(
        os_sandbox,
        "refresh_tool_isolation",
        lambda *a, **k: pytest.fail("probed in the background"),
    )
    assert os_sandbox.cached_tool_isolation("python") is None
    assert os_sandbox.start_tool_isolation_warmup() is None


def test_a_real_launch_updates_the_cached_answer():
    os_sandbox.note_tool_isolation("terminal", False, backend = "none", reason = "blocked")
    assert os_sandbox.cached_tool_capability("terminal") == (False, "none", "blocked")
    os_sandbox.note_tool_isolation("unknown-tool", True)
    assert os_sandbox.cached_tool_capability("unknown-tool") is None


def test_forget_discards_a_refresh_that_started_before_it(monkeypatch):
    generation = os_sandbox.tool_isolation_generation()
    os_sandbox.forget_tool_isolation()
    os_sandbox.note_tool_isolation("python", True, generation = generation)
    assert os_sandbox.cached_tool_capability("python") is None


def test_an_older_check_that_finishes_last_never_overwrites_a_newer_one():
    stale = os_sandbox.tool_isolation_generation()
    forced = os_sandbox.tool_isolation_generation()
    os_sandbox.note_tool_isolation("python", True, backend = "bubblewrap", generation = forced)
    os_sandbox.note_tool_isolation("python", False, backend = "none", generation = stale)
    assert os_sandbox.cached_tool_capability("python")[:2] == (True, "bubblewrap")
    os_sandbox.note_tool_isolation("python", False, backend = "none")
    assert os_sandbox.cached_tool_capability("python")[:2] == (False, "none")
    os_sandbox.note_tool_isolation("terminal", True, generation = stale)
    assert os_sandbox.cached_tool_capability("terminal")[0] is True


def test_probe_cache_reset_forgets_the_cached_answer():
    os_sandbox.note_tool_isolation("python", True)
    sandbox_probe.reset_probe_cache()
    assert os_sandbox.cached_tool_capability("python") is None


def test_warmup_runs_off_the_calling_thread(monkeypatch):
    monkeypatch.delenv(os_sandbox.WARMUP_DISABLE_ENV, raising = False)
    monkeypatch.setattr(sys, "platform", "linux")
    seen = []
    done = threading.Event()

    def refresh(tool, *, force = False):
        seen.append((tool, threading.current_thread().name))
        if len(seen) == 2:
            done.set()
        return False

    monkeypatch.setattr(os_sandbox, "refresh_tool_isolation", refresh)
    thread = os_sandbox.start_tool_isolation_warmup()
    assert thread is not None and thread.daemon
    assert done.wait(5)
    assert [tool for tool, _ in seen] == ["python", "terminal"]
    assert all(name == "unsloth-sandbox-warmup" for _, name in seen)


def test_no_startup_warmup_on_windows(monkeypatch):
    # The MXC check launches a container and applies read grants: never at every server start.
    monkeypatch.delenv(os_sandbox.WARMUP_DISABLE_ENV, raising = False)
    monkeypatch.setattr(sys, "platform", "win32")
    called = []
    monkeypatch.setattr(os_sandbox, "refresh_tool_isolation", lambda *a, **k: called.append(a))
    assert os_sandbox.start_tool_isolation_warmup() is None
    assert called == []


class _LeapingClock:
    """Each read is a day later, so any TTL left in the code expires between two calls."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        self.now += 86400.0
        return self.now


def _leap_time(monkeypatch, module):
    import types
    clock = types.SimpleNamespace(monotonic = _LeapingClock(), time = time.time, sleep = time.sleep)
    monkeypatch.setattr(module, "time", clock, raising = False)


@pytest.fixture
def probe_env(monkeypatch):
    monkeypatch.setenv(os_sandbox.WARMUP_DISABLE_ENV, "0")
    sandbox_probe.reset_probe_cache()
    _leap_time(monkeypatch, sandbox_probe)
    identity = ["identity"]
    monkeypatch.setattr(os_sandbox, "_runtime_identity", lambda: identity[0])
    verdicts = []
    gate = threading.Event()
    gate.set()
    calls = []

    def run_probe(backend, name, plan_cls):
        calls.append(threading.current_thread().name)
        gate.wait(5)
        return verdicts.pop(0)

    monkeypatch.setattr(sandbox_probe, "_run_probe", run_probe)
    yield identity, verdicts, gate, calls
    sandbox_probe.reset_probe_cache()


class _Backend:
    BACKEND_NAME = "fake"


def _wait_for(predicate, timeout = 5.0):
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        time.sleep(0.01)
    return predicate()


@pytest.mark.parametrize("first", [(True, "ok"), (False, "blocked")])
def test_a_verdict_is_held_until_an_event_not_a_timer(probe_env, first):
    identity, verdicts, gate, calls = probe_env
    verdicts.extend([first, (True, "after reset"), (False, "forced"), (True, "new runtime")])
    assert sandbox_probe.probe(_Backend()) == first
    for _ in range(20):
        assert sandbox_probe.probe(_Backend()) == first
    assert len(calls) == 1
    sandbox_probe.reset_probe_cache()
    assert sandbox_probe.probe(_Backend()) == (True, "after reset")
    assert sandbox_probe.probe(_Backend(), force = True) == (False, "forced")
    identity[0] = "swapped venv"
    assert sandbox_probe.probe(_Backend()) == (True, "new runtime")
    assert len(calls) == 4
    assert all(name == threading.current_thread().name for name in calls)


def test_an_undecided_verdict_is_retried_by_the_next_caller(probe_env):
    identity, verdicts, gate, calls = probe_env
    verdicts.extend([(False, sandbox_probe.UndecidedReason("out of disk")), (True, "ok")])
    assert sandbox_probe.probe(_Backend())[0] is False
    assert sandbox_probe.probe(_Backend()) == (True, "ok")
    assert sandbox_probe.probe(_Backend()) == (True, "ok")
    assert len(calls) == 2


def test_a_reset_during_a_probe_keeps_its_result_out(probe_env):
    identity, verdicts, gate, calls = probe_env
    verdicts.extend([(True, "from before the reset"), (False, "after")])
    gate.clear()
    worker = threading.Thread(target = sandbox_probe.probe, args = (_Backend(),))
    worker.start()
    assert _wait_for(lambda: len(calls) == 1)
    sandbox_probe.reset_probe_cache()
    gate.set()
    worker.join(5)
    assert sandbox_probe.probe(_Backend()) == (False, "after")


def test_an_older_probe_finishing_last_never_replaces_a_newer_verdict(probe_env, monkeypatch):
    """With no TTL, a slow ordinary probe landing after a forced Refresh would have stuck for good."""
    identity, verdicts, gate, calls = probe_env
    slow_started, release_slow = threading.Event(), threading.Event()

    def run_probe(backend, name, plan_cls):
        if not slow_started.is_set():
            slow_started.set()
            release_slow.wait(5)
            return False, "old failure"
        return True, "forced pass"

    monkeypatch.setattr(sandbox_probe, "_run_probe", run_probe)
    slow = threading.Thread(target = sandbox_probe.probe, args = (_Backend(),))
    slow.start()
    assert slow_started.wait(5)
    assert sandbox_probe.probe(_Backend(), force = True) == (True, "forced pass")
    release_slow.set()
    slow.join(5)
    assert sandbox_probe.probe(_Backend()) == (True, "forced pass")


def test_an_undecided_answer_is_checked_again_by_the_next_read(monkeypatch):
    monkeypatch.delenv(os_sandbox.WARMUP_DISABLE_ENV, raising = False)
    started = []

    def refresh(tool, *, force = False):
        started.append(tool)
        os_sandbox.note_tool_isolation(tool, True, backend = "bubblewrap")
        return True

    monkeypatch.setattr(os_sandbox, "refresh_tool_isolation", refresh)
    os_sandbox.note_tool_isolation(
        "python", False, backend = "none", reason = sandbox_probe.UndecidedReason("out of disk")
    )
    assert os_sandbox.cached_tool_capability("python")[0] is False
    assert _wait_for(lambda: os_sandbox.cached_tool_isolation("python") is True)
    for _ in range(5):
        os_sandbox.cached_tool_capability("python")
    assert started == ["python"]


def test_polling_an_answered_tool_never_probes_again(monkeypatch):
    monkeypatch.delenv(os_sandbox.WARMUP_DISABLE_ENV, raising = False)
    _leap_time(monkeypatch, os_sandbox)
    started = []
    import types

    monkeypatch.setattr(
        os_sandbox,
        "threading",
        types.SimpleNamespace(
            Thread = lambda *a, **k: started.append(k.get("name")) or threading.Thread()
        ),
    )
    os_sandbox.note_tool_isolation("python", False, backend = "none", reason = "blocked")
    for _ in range(50):
        assert os_sandbox.cached_tool_capability("python")[:3] == (False, "none", "blocked")
        assert tool_policy.off_mode_still_gates("python") is True
    assert started == [], "polling an answered tool started a re-probe"


def test_an_answer_noted_under_another_host_key_is_checked_once(monkeypatch):
    monkeypatch.delenv(os_sandbox.WARMUP_DISABLE_ENV, raising = False)
    key = ["no bwrap"]
    monkeypatch.setattr(os_sandbox, "_host_key", lambda: key[0])
    started = []

    def refresh(tool, *, force = False):
        started.append(tool)
        os_sandbox.note_tool_isolation(tool, True, backend = "bubblewrap")
        return True

    monkeypatch.setattr(os_sandbox, "refresh_tool_isolation", refresh)
    os_sandbox.note_tool_isolation("python", False, backend = "none", reason = "no bwrap")
    for _ in range(5):
        assert os_sandbox.cached_tool_capability("python")[0] is False
    assert started == []
    key[0] = "bwrap installed"
    os_sandbox.cached_tool_capability("python")
    assert _wait_for(lambda: os_sandbox.cached_tool_isolation("python") is True)
    for _ in range(5):
        os_sandbox.cached_tool_capability("python")
    assert started == ["python"]


@pytest.fixture(autouse = True)
def _no_host_setup_probe(monkeypatch):
    from core.inference import sandbox_setup_plan
    monkeypatch.setattr(
        sandbox_setup_plan,
        "detect",
        lambda *a, **k: sandbox_setup_plan.SetupPlan(platform = sys.platform, reason = "test"),
    )


def _capability_client(authenticated = True):
    from routes.sandbox_capability import router

    app = FastAPI()
    app.include_router(router, prefix = "/api/sandbox")
    if authenticated:
        app.dependency_overrides[get_current_subject] = lambda: "someone"
    return TestClient(app)


def test_capability_requires_a_signed_in_user():
    response = _capability_client(authenticated = False).get("/api/sandbox/capability")
    assert response.status_code in (401, 403)


def test_capability_before_the_first_answer():
    body = _capability_client().get("/api/sandbox/capability").json()
    assert body["python_os_isolated"] is False
    assert body["terminal_os_isolated"] is False
    assert body["backend"] == "unknown"
    assert body["setup_action"] is None and body["can_run_setup"] is False


@pytest.mark.parametrize("known", ["python", "terminal"])
def test_capability_stays_unknown_until_both_tools_have_an_answer(known):
    os_sandbox.note_tool_isolation(known, True, backend = "bubblewrap", reason = "passed")
    body = _capability_client().get("/api/sandbox/capability").json()
    assert body["backend"] == "unknown"
    assert body["setup_action"] is None and body["can_run_setup"] is False


def test_capability_reports_the_cached_answers():
    os_sandbox.note_tool_isolation("python", True, backend = "bubblewrap", reason = "passed")
    os_sandbox.note_tool_isolation("terminal", False, backend = "none", reason = "no bash")
    body = _capability_client().get("/api/sandbox/capability").json()
    assert body["python_os_isolated"] is True
    assert body["terminal_os_isolated"] is False
    assert body["backend"] == "bubblewrap"
    assert body["reason"] == "no bash"


@pytest.mark.parametrize(
    "mode,tool,isolated,risky", list(itertools.product(MODES, TOOLS, ISOLATION, (True, False)))
)
def test_strict_launch_matrix(isolation, mode, tool, isolated, risky):
    isolation["python"] = isolation["terminal"] = isolated
    decision = dict(
        confirm_tool_calls = True,
        bypass_permissions = mode == "full",
        permission_mode = mode,
        name = tool,
        arguments = {"risky": risky},
        is_high_risk = _risk,
    )
    prompted = tool_policy.needs_tool_confirmation(**decision, never_needs = lambda _n: False)
    strict = tool_policy.requires_os_isolation(**decision, prompted = prompted)
    assert strict is (
        mode == "off" and tool in ("python", "terminal") and isolated is True and risky
    )


def test_a_refresh_between_the_two_decisions_cannot_skip_both(isolation):
    # The launch must follow the prompt decision, not re-read a cache refreshed in between.
    isolation["python"] = True
    decision = dict(
        confirm_tool_calls = True,
        bypass_permissions = False,
        permission_mode = "off",
        name = "python",
        arguments = {"risky": True},
        is_high_risk = _risk,
    )
    prompted = tool_policy.needs_tool_confirmation(**decision, never_needs = lambda _n: False)
    isolation["python"] = False
    assert prompted is False
    assert tool_policy.requires_os_isolation(**decision, prompted = prompted) is True


def test_an_approved_prompt_runs_without_the_strict_launch(isolation):
    isolation["python"] = False
    assert not tool_policy.requires_os_isolation(
        confirm_tool_calls = True,
        bypass_permissions = False,
        permission_mode = "off",
        name = "python",
        arguments = {"risky": True},
        is_high_risk = _risk,
        prompted = True,
    )


def test_strict_launch_needs_an_armed_gate(isolation):
    isolation["python"] = True
    assert not tool_policy.requires_os_isolation(
        confirm_tool_calls = False,
        bypass_permissions = False,
        permission_mode = "off",
        name = "python",
        arguments = {"risky": True},
        is_high_risk = _risk,
        prompted = False,
    )


class _ModeRecordingExecuteTool:
    def __init__(self):
        self.modes = []

    def __call__(
        self,
        name,
        arguments,
        *,
        cancel_event = None,
        timeout = None,
        session_id = None,
        thread_id = None,
        rag_scope = None,
        disable_sandbox = False,
        tool_execution_mode = "auto",
    ):
        self.modes.append(tool_execution_mode)
        return f"RESULT[{name}]"


@pytest.mark.parametrize(
    "mode,code,expected",
    [
        ("off", 'import os; os.remove(\\"x\\")', "required"),
        ("off", "print(1)", "auto"),
        ("auto", "print(1)", "auto"),
    ],
)
def test_the_loop_launches_an_unasked_risky_call_strictly(mode, code, expected):
    import uuid

    from core.inference.safetensors_agentic import run_safetensors_tool_loop

    os_sandbox.note_tool_isolation("python", True, backend = "bubblewrap")
    turns = iter(
        [f'<tool_call>{{"name": "python", "arguments": {{"code": "{code}"}}}}</tool_call>', "final"]
    )

    def single_turn(_messages):
        try:
            yield next(turns)
        except StopIteration:
            return

    exec_fn = _ModeRecordingExecuteTool()
    events = list(
        run_safetensors_tool_loop(
            single_turn = single_turn,
            messages = [{"role": "user", "content": "hi"}],
            tools = [{"type": "function", "function": {"name": "python"}}],
            execute_tool = exec_fn,
            session_id = f"strict-{uuid.uuid4().hex}",
            confirm_tool_calls = True,
            permission_mode = mode,
        )
    )
    starts = [e for e in events if e["type"] == "tool_start"]
    assert starts and starts[0]["awaiting_confirmation"] is False
    assert exec_fn.modes == [expected]


def _first_python_call(session_id):
    from core.inference.safetensors_agentic import run_safetensors_tool_loop

    turns = iter(
        [
            '<tool_call>{"name": "python", "arguments": {"code": "import shutil; shutil.rmtree(\\"dist\\")"}}</tool_call>',
            "done",
        ]
    )

    def single_turn(_messages):
        try:
            yield next(turns)
        except StopIteration:
            return

    exec_fn = _ModeRecordingExecuteTool()
    loop = run_safetensors_tool_loop(
        single_turn = single_turn,
        messages = [{"role": "user", "content": "clean dist"}],
        tools = [{"type": "function", "function": {"name": "python"}}],
        execute_tool = exec_fn,
        session_id = session_id,
        confirm_tool_calls = True,
        permission_mode = "off",
    )
    start = next(e for e in loop if e["type"] == "tool_start")
    if start["awaiting_confirmation"]:
        loop.close()
    else:
        list(loop)
    return start["awaiting_confirmation"], exec_fn.modes


def test_run_automatically_asks_once_the_chat_folder_is_too_large_to_check(monkeypatch, tmp_path):
    import uuid

    from core.inference import tools

    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path))
    monkeypatch.setattr(os_sandbox, "WORKDIR_SCAN_ENTRIES", 3)
    os_sandbox.note_tool_isolation("python", True, backend = "bubblewrap")
    big, small = f"big-{uuid.uuid4().hex}", f"small-{uuid.uuid4().hex}"
    workdir = tools._get_workdir(big)
    tools._get_workdir(small)
    for i in range(8):
        with open(os.path.join(workdir, f"f{i}"), "w"):
            pass

    assert os_sandbox.scan_workdir_for_host_channels(workdir) == (
        os_sandbox.WORKDIR_SCAN_INCOMPLETE,
    )
    assert _first_python_call(big) == (True, [])
    assert _first_python_call(small) == (False, ["required"])

    for i in range(8):
        os.remove(os.path.join(workdir, f"f{i}"))
    assert os_sandbox.scan_workdir_for_host_channels(workdir) == ()
    assert _first_python_call(big) == (False, ["required"])


def test_a_chosen_project_folder_stops_asking_once_it_is_small_again(monkeypatch, tmp_path):
    from core.inference import tools

    monkeypatch.setattr(os_sandbox, "WORKDIR_SCAN_ENTRIES", 3)
    monkeypatch.setattr(os_sandbox, "OVERRUN_RECHECK_SECONDS", 0.0)
    for i in range(8):
        (tmp_path / f"f{i}").write_text("")
    os_sandbox.scan_workdir_for_host_channels(str(tmp_path))
    plan = os_sandbox.ToolLaunchPlan(
        argv = ("/bin/true",), workdir = str(tmp_path), env = {}, requested_mode = "auto"
    )

    tools._prepare_tool_launch(plan, user_folder = True)
    assert os_sandbox.workdir_scan_overran(str(tmp_path))

    for i in range(8):
        (tmp_path / f"f{i}").unlink()
    scratch = tmp_path / os_sandbox.TOOL_TEMP_DIRNAME
    scratch.mkdir(mode = os_sandbox.TOOL_TEMP_MODE)
    monkeypatch.chdir(scratch)
    dead = socket.socket(socket.AF_UNIX)
    dead.bind("s")
    dead.close()
    tools._prepare_tool_launch(plan, host_access_approved = True, user_folder = True)
    assert not os_sandbox.workdir_scan_overran(str(tmp_path))


def test_an_oversized_project_folder_is_rechecked_at_most_once_a_minute(monkeypatch, tmp_path):
    from core.inference import tools

    monkeypatch.setattr(os_sandbox, "WORKDIR_SCAN_ENTRIES", 3)
    for i in range(8):
        (tmp_path / f"f{i}").write_text("")
    os_sandbox.scan_workdir_for_host_channels(str(tmp_path))
    walks = []
    real = os_sandbox._walk_for_host_channels
    monkeypatch.setattr(
        os_sandbox, "_walk_for_host_channels", lambda *a: walks.append(a) or real(*a)
    )
    plan = os_sandbox.ToolLaunchPlan(
        argv = ("/bin/true",), workdir = str(tmp_path), env = {}, requested_mode = "auto"
    )

    for _ in range(3):
        tools._prepare_tool_launch(plan, user_folder = True)
    assert len(walks) == 1
    assert os_sandbox.workdir_scan_overran(str(tmp_path))


def test_windows_asks_the_model_to_retry_when_the_chat_folder_is_too_large(monkeypatch, tmp_path):
    from core.inference import mxc_policy

    monkeypatch.setattr(os_sandbox, "WORKDIR_SCAN_ENTRIES", 3)
    for i in range(8):
        (tmp_path / f"f{i}").write_text("")

    with pytest.raises(mxc_policy.MxcPolicyError, match = "Call the tool again"):
        mxc_policy._scan_workdir(str(tmp_path), required = True)
    assert os_sandbox.workdir_scan_overran(str(tmp_path))


@pytest.mark.skipif(
    sys.platform != "linux" or not os_sandbox.capability_snapshot().available,
    reason = "needs a host that can build the OS sandbox",
)
def test_an_oversized_chat_folder_refuses_once_then_asks_and_runs_isolated(monkeypatch, tmp_path):
    import uuid

    from core.inference import tools
    from core.inference.safetensors_agentic import run_safetensors_tool_loop
    from state.tool_approvals import resolve_tool_decision

    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path))
    monkeypatch.setattr(os_sandbox, "WORKDIR_SCAN_ENTRIES", 3)
    session = f"big-real-{uuid.uuid4().hex}"
    workdir = tools._get_workdir(session)
    for i in range(8):
        with open(os.path.join(workdir, f"f{i}"), "w"):
            pass
    capability = os_sandbox.capability_snapshot()
    os_sandbox.note_tool_isolation("python", True, backend = capability.backend)
    call = '<tool_call>{"name": "python", "arguments": {"code": "import shutil; shutil.rmtree(\\"dist%d\\", ignore_errors=True); print(\\"RAN\\" + \\"_OK\\")"}}</tool_call>'
    turns = iter([call % 1, call % 2, "done"])

    def single_turn(_messages):
        try:
            yield next(turns)
        except StopIteration:
            return

    starts, results, records = [], [], []
    for ev in run_safetensors_tool_loop(
        single_turn = single_turn,
        messages = [{"role": "user", "content": "clean dist"}],
        tools = [{"type": "function", "function": {"name": "python"}}],
        execute_tool = tools.execute_tool,
        session_id = session,
        confirm_tool_calls = True,
        permission_mode = "off",
    ):
        if ev["type"] == "tool_start":
            starts.append(ev["awaiting_confirmation"])
            if ev["awaiting_confirmation"]:
                resolve_tool_decision(ev["approval_id"], "allow", session_id = session)
        elif ev["type"] == "tool_end":
            results.append(ev["result"])
            records.append(tools._last_tool_execution_record)

    assert starts == [False, True]
    assert "Call the tool again" in results[0] and "RAN_OK" not in results[0]
    assert "RAN_OK" in results[1]
    assert records[1].os_isolation is True
    assert os_sandbox.WORKDIR_SCAN_INCOMPLETE in records[1].limitations


def test_a_strict_launch_is_refused_when_the_sandbox_stopped_working(monkeypatch):
    from core.inference import tools

    os_sandbox.note_tool_isolation("python", True, backend = "bubblewrap")
    monkeypatch.setattr(
        os_sandbox,
        "capability_snapshot",
        lambda **_kw: os_sandbox.SandboxCapability(
            backend = "bubblewrap",
            available = False,
            reason = "bwrap: setting up uid map: Permission denied",
            protection_state = "unavailable",
            limitations = (),
            remediation = "Load the AppArmor profile.",
        ),
    )
    tools._last_tool_execution_record = None
    out = tools.execute_tool(
        "python",
        {"code": "print('RAN')"},
        session_id = "__LOCALID_strict_refusal",
        timeout = 60,
        tool_execution_mode = "required",
    )
    assert "OS_ISOLATION_UNAVAILABLE" in out and "RAN" not in out
    record = tools._last_tool_execution_record
    assert record is None or record.effective_mode != "software_safeguards"
    auto = tools.execute_tool(
        "python", {"code": "print('RAN')"}, session_id = "__LOCALID_strict_refusal", timeout = 60
    )
    assert "RAN" in auto


def test_every_loop_launches_strictly_through_the_mode_parameter():
    import inspect

    from core.inference import llama_cpp, safetensors_agentic, studio_tool_loop
    for module in (studio_tool_loop, llama_cpp, safetensors_agentic):
        source = inspect.getsource(module)
        assert "requires_os_isolation(" in source, module.__name__
        assert 'kwargs["tool_execution_mode"] = "required"' in source, module.__name__


def test_a_launch_does_not_republish_an_answer_a_reset_cleared(monkeypatch, tmp_path):
    def snapshot(**_kw):
        os_sandbox.forget_tool_isolation()
        return os_sandbox.SandboxCapability(
            backend = "bubblewrap",
            available = False,
            reason = "stubbed",
            protection_state = "unavailable",
            limitations = (),
        )

    monkeypatch.setattr(os_sandbox, "capability_snapshot", snapshot)
    plan = os_sandbox.ToolLaunchPlan(
        argv = (sys.executable, "-c", "pass"),
        workdir = str(tmp_path),
        env = {},
        execution_kind = "python",
    )
    try:
        os_sandbox.prepare_tool_launch(plan)
    except Exception:  # noqa: BLE001 - only the cached answer matters here
        pass
    assert not os_sandbox.has_tool_isolation_answer("python")


@pytest.mark.skipif(sys.platform == "win32", reason = "the Windows backend prepares through MXC")
@pytest.mark.parametrize(
    "error,forgotten",
    [
        (os_sandbox.SandboxUnavailableError("bubblewrap (bwrap) is not installed"), True),
        (os_sandbox.WorkdirUnsafeError("a host channel in the workdir"), False),
    ],
)
def test_a_backend_that_refuses_at_preparation_drops_the_cached_pass(
    monkeypatch, tmp_path, error, forgotten
):
    import importlib

    backend = importlib.import_module(
        "core.inference.sandbox_linux"
        if sys.platform == "linux"
        else "core.inference.sandbox_macos"
    )
    monkeypatch.setenv(os_sandbox.WARMUP_DISABLE_ENV, "1")
    monkeypatch.setattr(
        os_sandbox,
        "capability_snapshot",
        lambda **_kw: os_sandbox.SandboxCapability(
            backend = backend.BACKEND_NAME, available = True, reason = "cached pass"
        ),
    )

    def refuse(_plan, *_args):
        raise error

    monkeypatch.setattr(backend, "prepare", refuse)
    plan = os_sandbox.ToolLaunchPlan(
        argv = (sys.executable, "-c", "pass"),
        workdir = str(tmp_path),
        env = {},
        requested_mode = "required",
        timeout_seconds = 10,
        execution_kind = "python",
    )
    with pytest.raises(type(error)):
        os_sandbox.prepare_tool_launch(plan)
    assert (os_sandbox.cached_tool_capability("python") is None) is forgotten
