# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deep Research's internal hop must never enter the local tool loop.

Its prompts carry gathered web and document text and go back through /v1/chat/completions,
where --enable-tools overrides a per-request enable_tools and an omitted enabled_tools
resolves to every built-in, python and terminal included. These tests pin the opt-out at the
route, where the decision is made, and pin that it costs an ordinary run nothing.
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.inference.llama_admission import reset_llama_admission_queues
from core.inference.llama_preemption import (
    PREEMPT_ENV,
    get_preemption_controller,
    preemption_enabled,
    reset_preemption_controllers,
)
import routes.inference as inference_route
from state.tool_policy import reset_tool_policy, set_tool_policy
from .llama_backend_double import FakeLlamaCppBackend


@pytest.fixture(autouse = True)
def _clean_policy():
    reset_tool_policy()
    reset_preemption_controllers()
    reset_llama_admission_queues()
    yield
    reset_tool_policy()
    reset_preemption_controllers()
    reset_llama_admission_queues()


class _Backend(FakeLlamaCppBackend):
    """Records which generation entry point the route picked."""

    supports_tools = True

    def __init__(self):
        self.calls = []

    def generate_chat_completion(self, **kwargs):
        self.calls.append(("plain", kwargs))
        yield "the answer"

    def generate_chat_completion_with_tools(self, **kwargs):
        self.calls.append(("tool_loop", kwargs))
        yield {"type": "content", "text": "the answer"}


def _client(monkeypatch, backend):
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    app = FastAPI()
    app.include_router(inference_route.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


def _research_payload(opt_out: bool):
    """The payload ResearchSupervisor._stream_completion builds, with the opt-out on or off."""
    body = {
        "model": "test/model.gguf",
        "messages": [
            {"role": "user", "content": "<untrusted_web_evidence>...</untrusted_web_evidence>"}
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.2,
        "max_tokens": 512,
    }
    if opt_out:
        body["tool_choice"] = "none"
        body["enabled_tools"] = []
    return body


def _entry_point(monkeypatch, *, policy, opt_out):
    backend = _Backend()
    if policy is not None:
        set_tool_policy(policy)
    response = _client(monkeypatch, backend).post(
        "/chat/completions",
        json = _research_payload(opt_out),
        # Unrestricted without the opt-out, so the confirm gate arms and needs a channel.
        # Research sends enabled_tools: [] and never arms it.
        headers = {"X-Unsloth-Events": "1"},
    )
    assert response.status_code == 200
    assert "the answer" in response.text
    return backend.calls[0][0], backend.calls[0][1]


def test_forced_tool_policy_would_reach_the_tool_loop_without_the_opt_out(monkeypatch):
    # Guards the test below: without this, it would pass if the route ever stopped forcing
    # tools on here, for entirely the wrong reason.
    entry, kwargs = _entry_point(monkeypatch, policy = True, opt_out = False)
    assert entry == "tool_loop"
    assert {t["function"]["name"] for t in kwargs["tools"]} >= {"python", "terminal"}


@pytest.mark.parametrize("policy", [None, True, False])
def test_the_research_payload_never_enters_the_tool_loop(monkeypatch, policy):
    entry, kwargs = _entry_point(monkeypatch, policy = policy, opt_out = True)
    assert entry == "plain"
    assert not kwargs.get("tools")


@pytest.mark.parametrize("policy", [None, False])
def test_the_opt_out_changes_nothing_a_default_install_does(monkeypatch, policy):
    # Without --enable-tools the hop was already tool-free, so the two fields must not
    # perturb what the model is handed: same entry point, same generation kwargs, and in
    # particular no tool catalogue on either side. The one kwarg that may differ is
    # `tools_withheld` (#9162), which is not handed to the model at all; it is pinned
    # explicitly below rather than excluded, so a regression either way still fails here.
    before_entry, before_kwargs = _entry_point(monkeypatch, policy = policy, opt_out = False)
    reset_tool_policy()
    after_entry, after_kwargs = _entry_point(monkeypatch, policy = policy, opt_out = True)

    assert (before_entry, after_entry) == ("plain", "plain")
    # Both are fresh per request (a new Event, and the monitor's per-request tok/s closure), so
    # comparing them by identity would fail for any pair of requests.
    # Fresh per request, so comparing them by identity fails for any pair. Asserted for
    # presence below instead, exactly as perf_callback is.
    drop = {
        "cancel_event",
        "perf_callback",
        "tools_withheld",
        "preempt_event",
        "preempt_policy",
        "on_tokens",
        "on_prompt_fitted",
    }
    # But dropping perf_callback outright would also pass if the opt-out stopped supplying it at
    # all, silently costing that path its tok/s readout. Compare presence first, then exclude.
    assert callable(before_kwargs.get("perf_callback")) == callable(
        after_kwargs.get("perf_callback")
    ), "the opt-out must not decide whether llama.cpp timings are collected"
    # Same shape for preemption, except the DEFAULT is the nothing-changes case: nothing in
    # this test sets UNSLOTH_LLAMA_ADMISSION_PREEMPT, so neither side may be armed and the
    # assertion is absence on both sides rather than presence. Pinned rather than dropped,
    # because a surface that arms a default install is invisible to every other test here.
    assert (
        preemption_enabled() is False
    ), "this test speaks for a default install, which does not opt into preemption"
    for _kwargs in (before_kwargs, after_kwargs):
        # llama.cpp folds a non-None preempt_event into a POLLED composite cancel event, so
        # handing one over is not free even when nobody ever sets it.
        assert _kwargs.get("preempt_event") is None
        # And no sweep: `on_tokens` is the only thing that tells the controller a chat has
        # grown, and llama.cpp skips the callback outright when it is None.
        assert _kwargs.get("on_tokens") is None
        # The policy object is still handed over, unbound, which is what it is for: it is
        # the callback a pause would land on, and there is no pause to land. Pinned inert
        # rather than absent, since it cannot be reached without a preempt_event.
        assert _kwargs["preempt_policy"].bound is False
        assert _kwargs["preempt_policy"].should_preempt() is False
        # The re-pricing hook is admission, not preemption, and it stays: without it an
        # overlong chat is sent the one-token floor the fit lifted.
        assert callable(_kwargs.get("on_prompt_fitted"))
    # `tools_withheld` reaches the compaction gate, never the prompt: it tells
    # `_can_reset_epoch` that THIS request withdrew the tool loop, which the process-wide
    # policy cannot see. A default install can still re-admit `search_conversation` alone
    # through the checkpoint repair, so resetting the epoch there is safe; under the opt-out
    # that repair is closed on this turn and on every identical turn after it, so a reset
    # would strand the epoch behind a tool that never arrives. It MUST differ, in this
    # direction, and the two must never both be False.
    assert (before_kwargs["tools_withheld"], after_kwargs["tools_withheld"]) == (False, True)
    # Nothing that reaches the model may differ, tool catalogue included.
    assert not before_kwargs.get("tools") and not after_kwargs.get("tools")
    assert {k: v for k, v in before_kwargs.items() if k not in drop} == {
        k: v for k, v in after_kwargs.items() if k not in drop
    }


def test_json_mode_research_calls_send_llama_server_an_unchanged_body():
    # The JSON-mode phases take the llama-server passthrough, not the loop above, so pin
    # that wire body too: no tools means no tool_choice is forwarded, and Unsloth-only
    # extensions never leave Unsloth.
    from models.inference import ChatCompletionRequest

    class _PassthroughBackend:
        supports_tools = True
        supports_tool_passthrough = True
        markup_profile = None

        def _request_reasoning_kwargs(self, enable_thinking, reasoning_effort, preserve):
            return None

    backend = _PassthroughBackend()
    bodies = []
    for opt_out in (False, True):
        payload = ChatCompletionRequest(
            **_research_payload(opt_out), response_format = {"type": "json_object"}
        )
        assert inference_route._takes_tool_passthrough(payload, backend) is True
        bodies.append(
            inference_route._build_openai_passthrough_body(payload, llama_backend = backend)
        )

    assert bodies[0] == bodies[1]
    assert "tool_choice" not in bodies[1] and "tools" not in bodies[1]
    assert "enabled_tools" not in bodies[1] and "enable_tools" not in bodies[1]
    assert json.loads(json.dumps(bodies[1]))["response_format"] == {"type": "json_object"}


# ── A default install takes no preemption path ────────────────────────────────
# The tests above pin what the research opt-out costs an ordinary run. These pin the same
# thing for the KV preemption controller, which is opt-in: unless
# `UNSLOTH_LLAMA_ADMISSION_PREEMPT=1` is set, a chat through these routes must reach
# llama.cpp exactly as it did before the controller existed.

_PREEMPT_KEY = "research-gate-default-install"


class _EligibleBackend(_Backend):
    """Everything preemption needs EXCEPT the opt-in: one shared cache, a budget, slots.

    Without these the switch would not be the reason nothing arms -- `controller.active` is
    False on any backend that is not `--kv-unified` with a known budget -- and the tests
    below would pass for the wrong reason.
    """

    base_url = "http://127.0.0.1:10301/"
    admission_key = _PREEMPT_KEY
    context_length = 16384
    _kv_cache_context_total = 16384
    effective_parallel_slots = 4
    _kv_cache_unified = True


def _one_chat(monkeypatch, *, tool_loop):
    """One chat through the real route, plain or through the server-side tool loop."""
    backend = _EligibleBackend()
    if tool_loop:
        set_tool_policy(True)
    response = _client(monkeypatch, backend).post(
        "/chat/completions",
        json = _research_payload(opt_out = not tool_loop),
        headers = {"X-Unsloth-Events": "1"},
    )
    assert response.status_code == 200
    entry, kwargs = backend.calls[0]
    assert entry == ("tool_loop" if tool_loop else "plain")
    return response, kwargs


@pytest.mark.parametrize("tool_loop", [False, True])
def test_a_default_install_takes_no_preemption_path(monkeypatch, tool_loop):
    """A plain GGUF chat and a tool-loop chat, on a backend preemption WOULD apply to."""
    assert preemption_enabled() is False, "nothing here opts in"
    response, kwargs = _one_chat(monkeypatch, tool_loop = tool_loop)

    # Nothing that could pause the stream. llama.cpp folds a non-None preempt_event into a
    # polled composite cancel event, and skips the token callback outright when on_tokens
    # is None, so both of these are the difference between arming and not.
    assert kwargs.get("preempt_event") is None
    assert kwargs.get("on_tokens") is None
    assert kwargs["preempt_policy"].bound is False

    # Nothing registered on the controller either: no participant to be chosen, and no
    # residency probe, which is the only thing that would ever send a GET /slots.
    controller = get_preemption_controller(_PREEMPT_KEY)
    assert controller._residency_probe is None, "a default install registered a slots probe"
    snapshot = controller.snapshot()
    assert (snapshot.holders, snapshot.paused, snapshot.decoding) == (0, 0, 0)

    # And the client hears nothing about a pause that never happened.
    assert ": preempt-" not in response.text


def test_a_default_install_is_charged_the_share(monkeypatch):
    """The optimism is off, so the reservation is the plain fair-share arithmetic.

    Charging less than that is safe only where a pause hands the difference back. Off, the
    wire is still permitted `share - prompt` less the reserve, so a smaller charge would be
    KV nobody reserved and nothing to reclaim it.
    """
    backend = _EligibleBackend()
    budget = inference_route._openai_llama_admission_budget(backend)
    share = budget // inference_route._openai_llama_admission_capacity(None, backend)
    active = inference_route._openai_llama_preemption_will_apply(backend, budget)
    assert active is False, "a default install must not price against a pause"
    for prompt in (100, 1000, 3000):
        assert inference_route._openai_llama_admission_output_allowance(
            None,
            budget = budget,
            prompt_tokens = prompt,
            context_window = budget,
            share = share,
            preemption_active = active,
        ) == min(share - prompt, budget - prompt), f"prompt {prompt} is not charged its share"
    # And a stated cap is charged in full, which is what serialises rather than overruns.
    assert inference_route._openai_llama_admission_output_allowance(
        5000,
        budget = budget,
        prompt_tokens = 1000,
        context_window = budget,
        share = share,
        preemption_active = active,
    ) == 5000


@pytest.mark.parametrize("tool_loop", [False, True])
def test_the_opt_in_arms_the_same_call(monkeypatch, tool_loop):
    """The other half of the switch: `=1` and the identical request is armed."""
    monkeypatch.setenv(PREEMPT_ENV, "1")
    assert preemption_enabled() is True
    _response, kwargs = _one_chat(monkeypatch, tool_loop = tool_loop)
    assert kwargs.get("preempt_event") is not None, "the opt-in did not hand over a signal"
    assert callable(kwargs.get("on_tokens")), "the opt-in did not arm the sweep"
    # And the probe the resume wait re-reads the cache with, refused on the default path.
    assert get_preemption_controller(_PREEMPT_KEY)._residency_probe is not None
