# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

""" "off" (Full access in sandbox) asks before a risky python/terminal call without OS isolation.

The decision every tool loop shares (state.tool_policy.needs_tool_confirmation), the route
arming that makes it reachable only where a prompt can be shown, the cached capability it
reads (never waiting on a probe), the probe cache that serves a stale PASS while it re-probes,
the startup warm-up, and GET /api/sandbox/capability.
"""

import itertools
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
    # off: only python/terminal, only without isolation, only when risky.
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
    # Without injected classifiers the real ones decide: a benign python call under "off" and
    # no isolation does not ask, a credential read does.
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


# --- route arming ------------------------------------------------------------------------


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


# --- cached capability ---------------------------------------------------------------------


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
    with os_sandbox._tool_isolation_lock:
        generation = os_sandbox._tool_isolation_generation
    os_sandbox.forget_tool_isolation()
    os_sandbox.note_tool_isolation("python", True, generation = generation)
    assert os_sandbox.cached_tool_capability("python") is None


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


# --- probe cache: stale-while-revalidate ---------------------------------------------------


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def probe_env(monkeypatch):
    # The re-probe runs in the background only while background probes are on (conftest turns them off).
    monkeypatch.setenv(os_sandbox.WARMUP_DISABLE_ENV, "0")
    sandbox_probe.reset_probe_cache()
    clock = _Clock()
    import types

    monkeypatch.setattr(
        sandbox_probe,
        "time",
        types.SimpleNamespace(monotonic = clock, time = time.time, sleep = time.sleep),
    )
    monkeypatch.setattr(os_sandbox, "_runtime_identity", lambda: "identity")
    verdicts = []
    gate = threading.Event()
    gate.set()
    calls = []

    def run_probe(backend, name, plan_cls):
        calls.append(threading.current_thread().name)
        gate.wait(5)
        return verdicts.pop(0)

    monkeypatch.setattr(sandbox_probe, "_run_probe", run_probe)
    yield clock, verdicts, gate, calls
    sandbox_probe.reset_probe_cache()


class _Backend:
    BACKEND_NAME = "fake"


def _wait_for(predicate, timeout = 5.0):
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        time.sleep(0.01)
    return predicate()


def test_a_stale_pass_is_served_while_one_reprobe_runs(probe_env):
    clock, verdicts, gate, calls = probe_env
    verdicts.extend([(True, "first"), (True, "second")])
    assert sandbox_probe.probe(_Backend()) == (True, "first")
    clock.now += sandbox_probe._CACHE_TTL_SECONDS + 1
    gate.clear()
    # Served at once, twice, with exactly one background re-probe.
    assert sandbox_probe.probe(_Backend()) == (True, "first")
    assert sandbox_probe.probe(_Backend()) == (True, "first")
    assert _wait_for(lambda: len(calls) == 2)
    assert calls[1] == "unsloth-sandbox-reprobe"
    gate.set()
    assert _wait_for(lambda: sandbox_probe.probe(_Backend()) == (True, "second"))
    assert len(calls) == 2


def test_a_failed_reprobe_replaces_the_stale_pass(probe_env):
    clock, verdicts, gate, calls = probe_env
    verdicts.extend([(True, "ok"), (False, "now blocked")])
    sandbox_probe.probe(_Backend())
    clock.now += sandbox_probe._CACHE_TTL_SECONDS + 1
    assert sandbox_probe.probe(_Backend()) == (True, "ok")
    assert _wait_for(lambda: sandbox_probe.probe(_Backend()) == (False, "now blocked"))


def test_a_fail_is_never_served_stale(probe_env):
    clock, verdicts, gate, calls = probe_env
    verdicts.extend([(False, "blocked"), (True, "fixed")])
    assert sandbox_probe.probe(_Backend()) == (False, "blocked")
    clock.now += sandbox_probe._CACHE_TTL_UNAVAILABLE_SECONDS + 1
    # Probed in the foreground, not served stale.
    assert sandbox_probe.probe(_Backend()) == (True, "fixed")
    assert all(name != "unsloth-sandbox-reprobe" for name in calls)


def test_a_pass_past_the_grace_is_probed_in_the_foreground(probe_env):
    clock, verdicts, gate, calls = probe_env
    verdicts.extend([(True, "old"), (True, "new")])
    sandbox_probe.probe(_Backend())
    clock.now += sandbox_probe._CACHE_TTL_SECONDS + sandbox_probe._STALE_GRACE_SECONDS + 1
    assert sandbox_probe.probe(_Backend()) == (True, "new")
    assert all(name != "unsloth-sandbox-reprobe" for name in calls)


def test_a_stale_pass_is_reprobed_in_the_foreground_when_background_probes_are_off(
    probe_env, monkeypatch
):
    clock, verdicts, gate, calls = probe_env
    monkeypatch.setenv(os_sandbox.WARMUP_DISABLE_ENV, "1")
    verdicts.extend([(True, "old"), (False, "broke")])
    sandbox_probe.probe(_Backend())
    clock.now += sandbox_probe._CACHE_TTL_SECONDS + 1
    assert sandbox_probe.probe(_Backend()) == (False, "broke")
    assert all(name != "unsloth-sandbox-reprobe" for name in calls)
    assert not sandbox_probe._refreshing


def test_a_reset_during_a_reprobe_keeps_its_result_out(probe_env):
    clock, verdicts, gate, calls = probe_env
    verdicts.extend([(True, "ok"), (True, "from before the reset"), (False, "after")])
    sandbox_probe.probe(_Backend())
    clock.now += sandbox_probe._CACHE_TTL_SECONDS + 1
    gate.clear()
    sandbox_probe.probe(_Backend())
    assert _wait_for(lambda: len(calls) == 2)
    sandbox_probe.reset_probe_cache()
    gate.set()
    assert _wait_for(lambda: not sandbox_probe._refreshing)
    # The pre-reset re-probe did not republish: the next read probes again.
    assert sandbox_probe.probe(_Backend()) == (False, "after")


# --- GET /api/sandbox/capability -----------------------------------------------------------


@pytest.fixture(autouse = True)
def _no_host_setup_probe(monkeypatch):
    # The setup plan would run `sudo -n true` on this host; the capability tests only need its shape.
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


def test_capability_reports_the_cached_answers():
    os_sandbox.note_tool_isolation("python", True, backend = "bubblewrap", reason = "passed")
    os_sandbox.note_tool_isolation("terminal", False, backend = "none", reason = "no bash")
    body = _capability_client().get("/api/sandbox/capability").json()
    assert body["python_os_isolated"] is True
    assert body["terminal_os_isolated"] is False
    assert body["backend"] == "bubblewrap"
    assert body["reason"] == "no bash"


# --- "off" skipped the prompt because the sandbox was on: that launch must not fall back ------


@pytest.mark.parametrize(
    "mode,tool,isolated,risky", list(itertools.product(MODES, TOOLS, ISOLATION, (True, False)))
)
def test_strict_launch_matrix(isolation, mode, tool, isolated, risky):
    isolation["python"] = isolation["terminal"] = isolated
    strict = tool_policy.requires_os_isolation(
        confirm_tool_calls = True,
        bypass_permissions = mode == "full",
        permission_mode = mode,
        name = tool,
        arguments = {"risky": risky},
        is_high_risk = _risk,
    )
    assert strict is (
        mode == "off" and tool in ("python", "terminal") and isolated is True and risky
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


def test_a_strict_launch_is_refused_when_the_sandbox_stopped_working(monkeypatch):
    from core.inference import tools

    # The cache still says isolated; the launch-time check says the backend is gone.
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
    # The same call in auto falls back, which is exactly what the strict launch prevents.
    auto = tools.execute_tool(
        "python", {"code": "print('RAN')"}, session_id = "__LOCALID_strict_refusal", timeout = 60
    )
    assert "RAN" in auto


def test_every_loop_launches_strictly_through_the_mode_parameter():
    # The GGUF and external-provider loops are driven elsewhere; the wiring must be the same.
    import inspect

    from core.inference import llama_cpp, safetensors_agentic, studio_tool_loop
    for module in (studio_tool_loop, llama_cpp, safetensors_agentic):
        source = inspect.getsource(module)
        assert "requires_os_isolation(" in source, module.__name__
        assert 'kwargs["tool_execution_mode"] = "required"' in source, module.__name__


def test_a_launch_does_not_republish_an_answer_a_reset_cleared(monkeypatch, tmp_path):
    # Settings or a failed launch reset the cached answers while this launch was still checking.
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
