# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import threading

import pytest

from core.inference import skill_mentions as mentions, skills
from core.inference.tools import READ_SKILL_TOOL
from .test_explicit_skill_loading import mention_client  # noqa: F401 -- shared isolated fixture


@pytest.mark.parametrize(
    "text, expected",
    [
        ("@skill-creator what did i just paste?", ["skill-creator"]),
        ("wait so @skill-creator didnt load it for you?", ["skill-creator"]),
        ("@skill-creator @another @skill-creator", ["skill-creator", "another"]),
        ("@skill-creator-extra @skill", ["skill-creator-extra", "skill"]),
        ("foo@skill-creator https://example/@skill-creator @skill-creator.com", []),
        ("He said \"@skill-creator\" and '@skill-creator'", []),
        ("“@skill-creator” ‘@skill-creator’", []),
        ("`@skill-creator` `` @skill-creator ``", []),
        ("```python\n@skill-creator\n```\n@another", ["another"]),
        ("~~~\n@skill-creator\n~~~", []),
        ('"quoted\n@skill-creator\n"', []),
        ("````\n```\n@skill-creator\n````", []),
        ("> @skill-creator\n\n    @skill-creator\n\n\t@skill-creator", []),
        ("@Skill-creator @skill-creator/README.md @skill-creator_foo", []),
        ("@skill-creator, please", ["skill-creator"]),
        ("```@literal```\n@skill-creator", ["skill-creator"]),
        ("````\n````python\n@skill-creator\n````\n@another", ["another"]),
        ("```\n```x```\n@skill-creator\n```\n@another", ["another"]),
        ("```\n    ```\n@skill-creator\n```\n@another", ["another"]),
        ("~~~\n\t~~~\n@skill-creator\n~~~\n@another", ["another"]),
        ("   ```\n@skill-creator\n   ```\n@another", ["another"]),
        ("    ~~~\n@skill-creator", ["skill-creator"]),
        ("> quoted prose\n@skill-creator replies below it", ["skill-creator"]),
        ("> quoted prose\ncontinued prose\n@skill-creator", ["skill-creator"]),
        ("> quoted prose\n> @skill-creator stays quoted\n@another", ["another"]),
        ("- > quoted @skill-creator\n  > still @skill-creator\n  @another", ["another"]),
        ("> quoted prose\n\n@skill-creator", ["skill-creator"]),
        ("> quoted prose\n# @skill-creator", ["skill-creator"]),
        ("> # heading\n@skill-creator", ["skill-creator"]),
        ("> ```\n> code\n@skill-creator", ["skill-creator"]),
        ("> ```\n> code\n> ```\n@skill-creator", ["skill-creator"]),
        (r'He said "type \" carefully, then @skill-creator literally"', []),
        (r"He said 'type \' carefully, then @skill-creator literally'", []),
        (r'He said "type \" carefully" then @skill-creator', ["skill-creator"]),
        (r"He said “type \” carefully, then @skill-creator literally”", []),
        ("```\r@skill-creator\r```\r@another", ["another"]),
        ("'please don't use @skill-creator here'", []),
        ("‘please don’t use @skill-creator here’", []),
        ("don't use @skill-creator here", ["skill-creator"]),
        ('‘unclosed "quoted @skill-creator here" then @another', ["another"]),
        ("unclosed \" then 'quoted @skill-creator here' and @another", ["another"]),
        ("‘please don’t use it’ then @skill-creator", ["skill-creator"]),
        ("`` code ``` xx ` @skill-creator ``", []),
        ("`` code ``` xx ` literal `` then @skill-creator", ["skill-creator"]),
        (r"\` @skill-creator \`", ["skill-creator"]),
        (r"\\` @skill-creator `", []),
        (r"` @skill-creator \`", []),
        ("` unmatched\n\n@skill-creator\n\nclosing `", ["skill-creator"]),
    ],
)
def test_plain_text_intent_contract(text, expected):
    assert mentions.mentioned_skill_names(text) == expected


def _load(
    text = "@skill-creator",
    *,
    messages = None,
    tools = None,
    **kwargs,
):
    messages = messages if messages is not None else [{"role": "user", "content": text}]
    events = list(
        mentions.load_mentioned_skills(
            messages, [READ_SKILL_TOOL] if tools is None else tools, **kwargs
        )
    )
    return messages, events


def test_duplicates_and_multiple_skills_are_complete(mention_client):
    _, _, path = mention_client
    other = path.parent.parent / "another" / "SKILL.md"
    other.parent.mkdir()
    other.write_text("---\nname: another\ndescription: Another.\n---\nSecond body.\n")
    messages, events = _load("@skill-creator @another @skill-creator")
    assert [e["name"] for e in events if e["status"] == "loaded"] == ["skill-creator", "another"]
    assert messages[0]["content"].count(path.read_text()) == 1
    assert other.read_text() in messages[0]["content"]


@pytest.mark.parametrize("failure", ["disabled", "symlink", "invalid"])
def test_secure_discovery_failures_are_not_success(mention_client, failure):
    _, _, path = mention_client
    if failure == "disabled":
        skills.set_skill_enabled("skill-creator", False)
    elif failure == "symlink":
        old = path.read_text()
        target = path.parent / "other.md"
        target.write_text(old)
        path.unlink()
        path.symlink_to(target)
    elif failure == "invalid":
        path.write_text("no frontmatter")
    messages, events = _load("@skill-creator")
    assert events[-1]["status"] == "unavailable"
    assert "not loaded" in events[-1]["detail"]
    assert not any(e["status"] == "loaded" for e in events)
    assert "Complete manifest sentinel" not in messages[0]["content"]


def test_complete_snapshot_larger_than_resource_page(mention_client):
    _, _, path = mention_client
    path.write_text(path.read_text() + "Instruction.\n" * 1000)
    assert "Resource continues" in skills.read_skill_resource("skill-creator")
    messages, events = _load(context_length = 16384)
    assert events[-1]["status"] == "loaded"
    assert events[-1]["characters"] == len(path.read_text())
    assert path.read_text() in messages[0]["content"]


def test_budget_refuses_whole_manifest_not_a_truncated_success(mention_client):
    _, _, path = mention_client
    messages, events = _load(context_length = 16)
    assert events[-1]["status"] == "unavailable"
    assert "budget" in events[-1]["detail"]
    assert path.read_text() not in messages[0]["content"]


def test_existing_actual_instructions_deduplicate_but_changed_resources_reload(mention_client):
    _, _, path = mention_client
    original = path.read_text()
    history = [{"role": "tool", "content": original}, {"role": "user", "content": "@skill-creator"}]
    messages, events = _load(messages = history)
    assert events[-1]["already_in_context"] is True
    assert len(messages) == 2
    path.write_text(original + "Changed resource.\n")
    messages, events = _load(messages = history)
    assert events[-1]["already_in_context"] is False
    assert path.read_text() in messages[0]["content"]


def test_retry_without_instructions_reloads_and_old_mentions_never_activate(mention_client):
    _, _, path = mention_client
    for _ in range(2):
        messages, events = _load()
        assert path.read_text() in messages[0]["content"]
        assert events[-1]["already_in_context"] is False
    for history in [
        [{"role": "assistant", "content": "@skill-creator"}],
        [{"role": "user", "content": "@skill-creator"}, {"role": "user", "content": "hello"}],
        [{"role": "system", "content": "@skill-creator"}, {"role": "user", "content": "hello"}],
    ]:
        assert _load(messages = history)[1] == []
    assert _load(continue_final_message = True)[1] == []


def test_deduplicated_native_result_is_protected_through_context_fitting(mention_client):
    from core.inference.context_window import truncate_oldest_messages

    _, _, path = mention_client
    manifest = path.read_text()
    history = [
        {"role": "user", "content": "earlier task"},
        {"role": "assistant", "content": "earlier reply"},
        {"role": "tool", "content": manifest},
        {"role": "user", "content": "another task"},
        {"role": "assistant", "content": "another reply"},
        {"role": "user", "content": "@skill-creator"},
    ]
    pins = set()
    messages, events = _load(messages = history, protected_message_ids = pins)
    assert events[-1]["already_in_context"] is True
    assert id(history[2]) in pins
    fitted, _ = truncate_oldest_messages(messages, keep_ratio = 0.1, protected_message_ids = pins)
    assert any(manifest in message.get("content", "") for message in fitted)


def test_mentions_that_are_not_skills_are_silent(mention_client, monkeypatch):
    monkeypatch.setattr(mentions, "begin_tool_decision", lambda *a: pytest.fail("no approval"))
    text = "ask @john and @everyone about it"
    messages, events = _load(text, permission_mode = "ask", confirm_tool_calls = True)
    assert events == []
    assert messages == [{"role": "user", "content": text}]


def test_disabled_skill_reports_unavailable_without_asking_approval(mention_client, monkeypatch):
    skills.set_skill_enabled("skill-creator", False)
    monkeypatch.setattr(mentions, "begin_tool_decision", lambda *a: pytest.fail("no approval"))
    _, events = _load(permission_mode = "ask", confirm_tool_calls = True)
    assert events[-1]["status"] == "unavailable"


def test_read_error_preserves_a_single_system_message_and_original_history(mention_client):
    _, _, path = mention_client
    path.write_text("no frontmatter")
    system = {"role": "system", "content": "Standing instructions."}
    messages, events = _load(
        "@skill-creator", messages = [system, {"role": "user", "content": "@skill-creator"}]
    )
    assert events[-1]["status"] == "unavailable"
    assert [m["role"] for m in messages] == ["system", "user"]
    assert system["content"] == "Standing instructions."
    assert "Standing instructions." in messages[0]["content"]


def test_code_or_capability_gate_cannot_be_bypassed_by_a_mention(mention_client, monkeypatch):
    monkeypatch.setattr(
        mentions, "read_skill_instructions", lambda name: pytest.fail("must not read")
    )
    assert _load(tools = [])[1] == []
    assert _load(tools = [{"function": {"name": "terminal"}}])[1] == []


@pytest.mark.parametrize("verdict", ["allow", "deny"])
def test_ask_requires_actual_scoped_approval_before_read(mention_client, monkeypatch, verdict):
    _, _, path = mention_client
    messages = [{"role": "user", "content": "@skill-creator"}]
    gen = mentions.load_mentioned_skills(
        messages,
        [READ_SKILL_TOOL],
        permission_mode = "ask",
        confirm_tool_calls = True,
        session_id = "isolated-session",
    )
    event = next(gen)
    assert event["status"] == "awaiting_approval"
    assert len(messages) == 1
    from state.tool_approvals import resolve_tool_decision, tool_decision_is_pending

    assert not resolve_tool_decision(event["approval_id"], verdict, session_id = "wrong-session")
    assert resolve_tool_decision(event["approval_id"], verdict, session_id = "isolated-session")
    events = list(gen)
    assert not tool_decision_is_pending(event["approval_id"])
    assert events[-1]["status"] == ("loaded" if verdict == "allow" else "unavailable")
    assert (path.read_text() in json.dumps(messages).replace("\\n", "\n")) == (verdict == "allow")
    if verdict == "deny":
        assert messages[0]["role"] == "system"
        assert events[-1]["detail"] in messages[0]["content"]


def test_closing_parked_preload_cleans_approval_slot(mention_client):
    from state.tool_approvals import tool_decision_is_pending

    gen = mentions.load_mentioned_skills(
        [{"role": "user", "content": "@skill-creator"}],
        [READ_SKILL_TOOL],
        permission_mode = "ask",
        confirm_tool_calls = True,
    )
    event = next(gen)
    assert tool_decision_is_pending(event["approval_id"])
    gen.close()
    assert not tool_decision_is_pending(event["approval_id"])


def test_provider_cannot_forge_successful_load_status():
    from core.inference.sse_control_frames import sanitize_provider_sse_line
    assert sanitize_provider_sse_line('data: {"type":"skill_load","status":"loaded"}') is None


def test_external_transport_gets_manifest_before_first_request(mention_client):
    from core.inference.studio_tool_loop import (
        ToolLoopRun,
        ToolLoopPolicy,
        stream_with_studio_tools,
    )

    _, _, path = mention_client
    captured = []

    class Transport:
        heals_text_tool_calls = False
        sanitizes_provider_frames = False

        async def stream(self, **kwargs):
            captured.append(kwargs)
            yield 'data: {"choices":[{"delta":{"content":"No tool call."},"finish_reason":"stop"}]}'
            yield "data: [DONE]"

    async def drive():
        return [
            event
            async for event in stream_with_studio_tools(
                Transport(),
                run = ToolLoopRun(
                    messages = [{"role": "user", "content": "@skill-creator what did i just paste?"}]
                ),
                policy = ToolLoopPolicy(
                    tools = [READ_SKILL_TOOL],
                    max_calls = 1,
                    timeout = 30,
                    permission_mode = "auto",
                    confirm_calls = False,
                    bypass_permissions = False,
                    rag_scope = None,
                    nudge_tool_calls = False,
                ),
                cancel_event = threading.Event(),
            )
        ]

    events = asyncio.run(drive())
    assert path.read_text() in captured[0]["messages"][0]["content"]
    assert any('"type":"skill_load"' in event and '"status":"loaded"' in event for event in events)
    assert not any('"type":"tool_start"' in event for event in events)


def test_safetensors_gets_complete_manifest_before_first_turn(mention_client):
    from core.inference.safetensors_agentic import run_safetensors_tool_loop

    _, _, path = mention_client
    captured = []

    def single_turn(messages):
        captured.append(messages)
        yield "No tool call."

    events = list(
        run_safetensors_tool_loop(
            single_turn = single_turn,
            messages = [{"role": "user", "content": "@skill-creator"}],
            tools = [READ_SKILL_TOOL],
            execute_tool = lambda *args, **kwargs: pytest.fail("no script/tool execution"),
            nudge_tool_calls = False,
            permission_mode = "auto",
            context_length = 8192,
        )
    )
    assert path.read_text() in captured[0][0]["content"]
    assert any(event["type"] == "skill_load" and event["status"] == "loaded" for event in events)


@pytest.mark.parametrize("verdict", ["allow", "deny"])
def test_external_ask_flushes_skill_approval_before_waiting(mention_client, monkeypatch, verdict):
    from core.inference.studio_tool_loop import (
        ToolLoopRun,
        ToolLoopPolicy,
        stream_with_studio_tools,
    )

    def slow_verdict(*args, **kwargs):
        import time
        time.sleep(0.3)
        return verdict

    monkeypatch.setattr(mentions, "wait_tool_decision", slow_verdict)
    captured = []

    class Transport:
        heals_text_tool_calls = False
        sanitizes_provider_frames = False

        async def stream(self, **kwargs):
            captured.append(kwargs["messages"])
            yield 'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}'
            yield "data: [DONE]"

    async def drive():
        return [
            event
            async for event in stream_with_studio_tools(
                Transport(),
                run = ToolLoopRun(
                    messages = [{"role": "user", "content": "@skill-creator"}], session_id = "s"
                ),
                policy = ToolLoopPolicy(
                    tools = [READ_SKILL_TOOL],
                    max_calls = 1,
                    timeout = 30,
                    permission_mode = "ask",
                    confirm_calls = True,
                    bypass_permissions = False,
                    rag_scope = None,
                    nudge_tool_calls = False,
                ),
                cancel_event = threading.Event(),
            )
        ]

    events = asyncio.run(drive())
    gate = next(i for i, e in enumerate(events) if '"status":"awaiting_approval"' in e)
    assert events[gate + 1].startswith(":"), events[gate : gate + 2]
    assert any('"status":"loaded"' in e for e in events) == (verdict == "allow")
    if verdict == "deny":
        assert "@skill-creator not loaded" in captured[0][0]["content"]
        assert "approval was denied" in captured[0][0]["content"]


@pytest.mark.parametrize("tool_choice, max_calls", [("none", 5), ("auto", 0)])
def test_external_withdrawn_tools_skip_preload(mention_client, tool_choice, max_calls):
    from core.inference.studio_tool_loop import (
        ToolLoopRun,
        ToolLoopPolicy,
        stream_with_studio_tools,
    )

    _, _, path = mention_client
    captured = []

    class Transport:
        heals_text_tool_calls = False
        sanitizes_provider_frames = False

        async def stream(self, **kwargs):
            captured.append(kwargs)
            yield 'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}'
            yield "data: [DONE]"

    async def drive():
        return [
            event
            async for event in stream_with_studio_tools(
                Transport(),
                run = ToolLoopRun(
                    messages = [{"role": "user", "content": "@skill-creator"}],
                    tool_choice = tool_choice,
                ),
                policy = ToolLoopPolicy(
                    tools = [READ_SKILL_TOOL],
                    max_calls = max_calls,
                    timeout = 30,
                    permission_mode = "auto",
                    confirm_calls = False,
                    bypass_permissions = False,
                    rag_scope = None,
                    nudge_tool_calls = False,
                ),
                cancel_event = threading.Event(),
            )
        ]

    events = asyncio.run(drive())
    assert path.read_text() not in json.dumps([c.get("messages") for c in captured])
    assert not any('"type":"skill_load"' in event for event in events)


@pytest.mark.parametrize("choice", ["auto", "none"])
def test_safetensors_route_respects_withdrawn_tools(mention_client, monkeypatch, choice):
    from routes import inference as api

    client, gguf, manifest = mention_client
    gguf.is_loaded = False
    captured = []

    class Backend:
        active_model_name = "qwen"
        models = {"qwen": {"chat_template_info": {"template": "qwen"}, "is_vision": False}}

        def generate_chat_response(self, **kwargs):
            captured.append(kwargs["messages"])
            yield "No model tool call."

        def generate_chat_completion_with_tools(self, *, messages, tools, **kwargs):
            from core.inference.safetensors_agentic import run_safetensors_tool_loop
            return run_safetensors_tool_loop(
                single_turn = lambda conversation: self.generate_chat_response(messages = conversation),
                messages = messages,
                tools = tools,
                execute_tool = lambda *a, **kw: pytest.fail("no model tool call"),
                nudge_tool_calls = False,
                permission_mode = kwargs["permission_mode"],
            )

        def reset_generation_state(self, *args):
            pass

    monkeypatch.setattr(api, "get_inference_backend", lambda: Backend())
    monkeypatch.setattr(
        api, "_detect_safetensors_features", lambda *a, **kw: {"supports_tools": True}
    )
    response = client.post(
        "/chat/completions",
        json = {
            "messages": [{"role": "user", "content": "@skill-creator"}],
            "stream": True,
            "enable_tools": True,
            "enabled_tools": ["read_skill"],
            "permission_mode": "auto",
            "tool_choice": choice,
        },
        headers = {"X-Unsloth-Events": "1"},
    )
    assert response.status_code == 200, response.text
    assert captured, response.text
    assert (manifest.read_text() in json.dumps(captured).replace("\\n", "\n")) == (choice == "auto")
    assert ('"status": "loaded"' in response.text) == (choice == "auto")


@pytest.mark.parametrize("extra", [{"tool_choice": "none"}, {"max_tool_calls_per_message": 0}])
def test_local_withdrawn_tools_skip_preload(mention_client, extra):
    client, backend, manifest = mention_client
    response = client.post(
        "/chat/completions",
        json = {
            "messages": [{"role": "user", "content": "@skill-creator hi"}],
            "stream": True,
            "enable_tools": True,
            "enabled_tools": ["read_skill"],
            "permission_mode": "auto",
            **extra,
        },
        headers = {"X-Unsloth-Events": "1"},
    )
    assert response.status_code == 200, response.text
    assert backend.requests, response.text
    assert manifest.read_text() not in json.dumps(backend.requests[0]["messages"])
    assert '"type": "skill_load"' not in response.text
