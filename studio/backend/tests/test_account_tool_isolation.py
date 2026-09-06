# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Account boundaries for sandbox tools, permission admission and MCP state."""

import asyncio
import json
import os
import socket
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from auth import policy
from core.inference import mcp_client, tools
from state import tool_policy
from storage import mcp_servers_db
from utils.account_context import OWNER, AccountContext, arun_as, current_account_id, run_as

ALICE = AccountContext("alice-id", "alice")
BOB = AccountContext("bob-id", "bob")
ACCOUNTS = (OWNER, ALICE, BOB)


@pytest.fixture(autouse = True)
def isolated(tmp_path, monkeypatch):
    sweep = tools._start_detached_sweep
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_HOME", raising = False)
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(tools, "_workdirs", {})
    monkeypatch.setattr(tools, "_active_sessions", {})
    monkeypatch.setattr(tools, "_pending_removals", {})
    monkeypatch.setattr(tools, "_removing_sessions", set())
    monkeypatch.setattr(tools, "_legacy_sandbox_migrated", True)
    monkeypatch.setattr(tools, "_start_detached_sweep", lambda: None)
    monkeypatch.setattr(tools, "_legacy_sandbox_root", lambda: str(tmp_path / "legacy"))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", False)
    if hasattr(mcp_servers_db, "_account_schema_ready"):
        monkeypatch.setattr(mcp_servers_db, "_account_schema_ready", set())
    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    monkeypatch.setattr(mcp_client, "_probe_cooloff_until", {})
    monkeypatch.setattr(mcp_client, "_oauth_token_store", None)
    if hasattr(mcp_client, "_account_oauth_token_stores"):
        monkeypatch.setattr(mcp_client, "_account_oauth_token_stores", {})
    return sweep


@pytest.mark.parametrize("session", [None, "same-chat", "../invalid", "project-same"])
def test_workdirs_and_memos_are_account_scoped(session, tmp_path):
    paths = [run_as(account, tools._get_workdir, session) for account in ACCOUNTS]
    assert len(set(paths)) == 3
    for account, path in zip(ACCOUNTS, paths):
        root = tmp_path / "studio"
        if account != OWNER:
            root = root / "accounts" / account.account_id
        assert Path(path).parent == root / "sandbox"
        assert run_as(account, tools.resolve_sandbox_workdir, session) == path
        assert tools._workdirs[(account.account_id, session or tools._ANON_KEY)] == path


def test_override_preserves_owner_and_namespaces_managed_accounts(tmp_path, monkeypatch):
    override = tmp_path / "custom-sandbox"
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(override))
    assert tools.sandbox_root() == str(override)
    for account in (ALICE, BOB):
        assert run_as(account, tools.sandbox_root) == str(
            override / "accounts" / account.account_id
        )


@pytest.mark.parametrize(
    "resolver,name",
    [
        (tools._orphan_records_dir, "orphaned-projects"),
        (tools._spill_records_dir, "tool-output-records"),
    ],
)
def test_records_are_account_scoped(resolver, name, tmp_path):
    assert resolver() == str(tmp_path / "studio" / name)
    for account in (ALICE, BOB):
        assert run_as(account, resolver) == str(
            tmp_path / "studio" / "accounts" / account.account_id / name
        )


def test_managed_account_never_reads_or_migrates_owner_legacy_sandbox(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy" / "same-chat"
    legacy.mkdir(parents = True)
    (legacy / "secret").write_text("owner", encoding = "utf-8")
    monkeypatch.setattr(tools, "_legacy_sandbox_migrated", False)
    assert run_as(ALICE, tools._legacy_session_dir, "same-chat") is None
    path = Path(run_as(ALICE, tools._get_workdir, "same-chat"))
    assert not (path / "secret").exists()
    assert (legacy / "secret").read_text(encoding = "utf-8") == "owner"
    assert not tools._legacy_sandbox_migrated


def test_managed_project_does_not_resolve_host_workspace(monkeypatch):
    from storage import studio_db
    monkeypatch.setattr(
        studio_db,
        "ensure_chat_project_workspace",
        lambda *_: pytest.fail("host workspace consulted"),
    )
    assert run_as(ALICE, tools._get_project_workdir, "project-demo") is None


def test_session_lifecycle_does_not_block_another_account():
    def alice_call():
        with tools._session_in_flight("same-chat"):
            assert tools.wait_for_sessions_idle(["same-chat"], timeout = 0) is False
            assert run_as(BOB, tools.wait_for_sessions_idle, ["same-chat"], timeout = 0) is True
            assert run_as(OWNER, tools.wait_for_sessions_idle, ["same-chat"], timeout = 0) is True

    run_as(ALICE, alice_call)


def test_removing_alices_sandbox_keeps_bobs(tmp_path):
    alice = Path(run_as(ALICE, tools._get_workdir, "same-chat"))
    bob = Path(run_as(BOB, tools._get_workdir, "same-chat"))
    assert run_as(ALICE, tools.remove_session_sandbox, "same-chat") is True
    assert not alice.exists()
    assert bob.is_dir()
    assert (BOB.account_id, "same-chat") in tools._workdirs
    assert (ALICE.account_id, "same-chat") not in tools._workdirs


@pytest.mark.parametrize("other", [OWNER, BOB])
@pytest.mark.parametrize("kind", ["absolute", "parent", "symlink"])
def test_edit_file_cannot_read_or_write_foreign_sandbox(other, kind):
    alice = Path(run_as(ALICE, tools._get_workdir, "same-chat"))
    foreign = Path(run_as(other, tools._get_workdir, "same-chat")) / "secret.txt"
    foreign.write_text("private", encoding = "utf-8")
    if kind == "absolute":
        raw = str(foreign)
    elif kind == "parent":
        raw = os.path.relpath(foreign, alice)
    else:
        (alice / "link").symlink_to(foreign)
        raw = "link"
    result = run_as(
        ALICE,
        tools.execute_tool,
        "edit_file",
        {
            "path": raw,
            "edits": [{"old_string": "private", "new_string": "changed"}],
        },
        session_id = "same-chat",
    )
    assert "outside" in result
    assert foreign.read_text(encoding = "utf-8") == "private"


@pytest.mark.parametrize("account", ACCOUNTS)
@pytest.mark.parametrize(
    "flags", [{"permission_mode": "full"}, {"bypass_permissions": True}, {"disable_sandbox": True}]
)
def test_full_access_admission_refuses_every_account_in_multi_mode(account, flags):
    with pytest.raises(HTTPException) as exc:
        run_as(account, tool_policy.require_tool_access, **flags)
    assert exc.value.status_code == 400
    assert "full access" in exc.value.detail.lower()
    assert "more than one account exists" in exc.value.detail


@pytest.mark.parametrize(
    "mode,bypass,want",
    [
        (None, False, ("auto", False)),
        ("unknown", False, ("ask", False)),
        ("off", False, ("off", False)),
        ("full", False, ("full", True)),
        ("ask", True, ("full", True)),
    ],
)
def test_single_account_permissions_keep_historical_normalization(mode, bypass, want, monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)
    assert tool_policy.normalize_tool_permissions(mode, bypass) == want


def test_sandboxed_admission_never_queries_installation_policy(monkeypatch):
    monkeypatch.setattr(
        policy, "full_access_permitted", lambda: pytest.fail("unexpected policy lookup")
    )
    tool_policy.require_tool_access("auto")
    tool_policy.require_tool_access("ask")
    tool_policy.require_tool_access("off")


def test_direct_tool_bypass_is_rejected_before_dispatch(monkeypatch):
    monkeypatch.setattr(tools, "_bash_exec", lambda *a, **kw: pytest.fail("tool ran"))
    with pytest.raises(HTTPException) as exc:
        tools.execute_tool("terminal", {"command": "pwd"}, disable_sandbox = True)
    assert exc.value.status_code == 400


@pytest.mark.parametrize("account", ACCOUNTS)
@pytest.mark.parametrize(
    "name",
    [
        "HF_TOKEN",
        "WANDB_API_KEY",
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GH_TOKEN",
        "CUSTOM_TOKEN",
        "CUSTOM_KEY",
        "CUSTOM_SECRET",
        "custom_key",
    ],
)
def test_safe_environment_excludes_host_credentials(account, name, monkeypatch):
    monkeypatch.setenv(name, "private-value")
    workdir = run_as(account, tools._get_workdir, "env")
    env = run_as(account, tools._build_safe_env, workdir)
    assert name not in env
    assert env["HOME"] == workdir
    assert Path(env["TMPDIR"]).parent == Path(workdir)


def test_owner_environment_remains_identical_for_managed_context(monkeypatch):
    workdir = tools._get_workdir("env")
    monkeypatch.setenv("HF_TOKEN", "owner-token")
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    assert run_as(ALICE, tools._build_safe_env, workdir) == tools._build_safe_env(workdir)


def test_tool_stream_carries_account_into_worker_thread():
    from core.inference.tool_stream_exec import stream_tool_execution

    seen = []
    stream = run_as(ALICE, tool_policy.account_tool_stream, stream_tool_execution)
    list(stream(lambda output: seen.append(current_account_id()) or "done", tool_name = "test"))
    assert seen == [ALICE.account_id]
    assert tool_policy.account_tool_stream(stream_tool_execution) is stream_tool_execution


@pytest.mark.parametrize(
    "model_name", ["ChatCompletionRequest", "ChatCountTokensRequest", "AnthropicMessagesRequest"]
)
def test_full_access_validators_remain_policy_free(model_name, monkeypatch):
    from models import inference

    monkeypatch.setattr(
        policy, "full_access_permitted", lambda: pytest.fail("model consulted policy")
    )
    model = getattr(inference, model_name)
    kwargs = {
        "model": "test",
        "permission_mode": "full",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
    }
    payload = model(**kwargs)
    assert payload.bypass_permissions is True


def test_mcp_configs_are_private_and_schema_is_initialized_per_database(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))],
    )
    for account in ACCOUNTS:
        run_as(
            account,
            mcp_servers_db.create_server,
            id = "same",
            display_name = account.username,
            url = "https://public.example/mcp",
        )
    for account in ACCOUNTS:
        assert (
            run_as(account, mcp_servers_db.get_server, "same")["display_name"] == account.username
        )
    run_as(ALICE, mcp_servers_db.delete_server, "same")
    assert run_as(ALICE, mcp_servers_db.list_servers) == []
    assert run_as(BOB, mcp_servers_db.get_server, "same") is not None
    assert mcp_servers_db.get_server("same") is not None


@pytest.mark.parametrize(
    "url",
    [
        "python server.py",
        "http://127.0.0.1/mcp",
        "http://10.0.0.1/mcp",
        "http://169.254.169.254/mcp",
        "http://[::1]/mcp",
        "http://[::ffff:127.0.0.1]/mcp",
    ],
)
def test_managed_mcp_registration_refuses_local_servers(url):
    with pytest.raises(HTTPException) as exc:
        run_as(ALICE, mcp_servers_db.create_server, id = "unsafe", display_name = "Unsafe", url = url)
    assert exc.value.status_code == 400
    assert run_as(ALICE, mcp_servers_db.get_server, "unsafe") is None


def test_mcp_update_cannot_introduce_local_endpoint(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))],
    )
    run_as(
        ALICE,
        mcp_servers_db.create_server,
        id = "server",
        display_name = "Public",
        url = "https://public.example/mcp",
    )
    with pytest.raises(HTTPException):
        run_as(ALICE, mcp_servers_db.update_server, "server", {"url": "python local.py"})
    assert run_as(ALICE, mcp_servers_db.get_server, "server")["url"] == "https://public.example/mcp"


def test_mcp_owner_keeps_local_registration_without_dns_cost(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: pytest.fail("owner DNS validation"))
    for index, url in enumerate(("python server.py", "http://127.0.0.1/mcp")):
        mcp_servers_db.create_server(id = str(index), display_name = "Owner", url = url)
        assert mcp_servers_db.get_server(str(index))["url"] == url


def test_stdio_gate_refuses_alice_even_when_installation_enables_it(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    assert not run_as(ALICE, mcp_client.stdio_mcp_enabled)
    assert "owner" in run_as(ALICE, mcp_client.stdio_mcp_disabled_reason)


def test_mcp_dns_rejects_mixed_public_and_private_answers(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **kw: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443)) for ip in ("8.8.8.8", "10.0.0.1")
        ],
    )
    with pytest.raises(HTTPException):
        run_as(ALICE, mcp_client.validate_mcp_address, "https://mixed.example/mcp")


def test_mcp_cache_and_failure_cooloffs_are_private():
    for account in ACCOUNTS:
        run_as(account, mcp_client.cache_tools, "same", [{"name": account.username}])
    run_as(ALICE, mcp_client.record_probe_failure, "same")
    assert run_as(ALICE, mcp_client.in_failure_cooloff, "same")
    assert not run_as(BOB, mcp_client.in_failure_cooloff, "same")
    run_as(ALICE, mcp_client.invalidate_tool_cache)
    assert run_as(ALICE, mcp_client.get_cached_tools, "same") is None
    assert run_as(BOB, mcp_client.get_cached_tools, "same") == [{"name": "bob"}]
    assert mcp_client.get_cached_tools("same") == [{"name": "unsloth"}]


def test_oauth_stores_and_files_are_private(tmp_path):
    stores = [run_as(account, mcp_client._oauth_store) for account in ACCOUNTS]
    assert len({id(store) for store in stores}) == 3
    assert str(stores[0]._data_directory) == str(tmp_path / "studio" / "mcp-oauth-tokens")
    for account, store in zip(ACCOUNTS, stores):
        assert run_as(account, mcp_client._oauth_store) is store
    for account, store in zip((ALICE, BOB), stores[1:]):
        assert str(store._data_directory) == str(
            tmp_path / "studio" / "accounts" / account.account_id / "mcp-oauth-tokens"
        )


def test_closing_mcp_sessions_only_closes_acting_account(monkeypatch):
    sessions = {}
    closed = []
    for account in ACCOUNTS:
        key = run_as(
            account, mcp_client._session_key, "https://public.example/mcp", None, "same-chat"
        )
        sessions[key] = account.account_id
    assert len(sessions) == 3
    monkeypatch.setattr(mcp_client, "_mcp_sessions", sessions)
    monkeypatch.setattr(mcp_client, "_close_all", lambda values: closed.extend(values))
    monkeypatch.setattr(mcp_client, "_drain_cleanup_queue", lambda: ([], None))
    before = {
        account.account_id: run_as(
            account, mcp_client._mcp_close_generation, "https://public.example/mcp", None
        )
        for account in ACCOUNTS
    }
    run_as(ALICE, mcp_client.close_mcp_sessions)
    assert closed == [ALICE.account_id]
    for account in (OWNER, BOB):
        assert (
            run_as(account, mcp_client._mcp_close_generation, "https://public.example/mcp", None)
            == before[account.account_id]
        )
    assert (
        run_as(ALICE, mcp_client._mcp_close_generation, "https://public.example/mcp", None)
        != before[ALICE.account_id]
    )


@pytest.mark.parametrize("account", ACCOUNTS)
@pytest.mark.parametrize("loop_name", ["gguf", "safetensors", "studio", "codex"])
def test_every_tool_loop_refuses_full_access_before_model_or_tool_dispatch(account, loop_name):
    def check():
        if loop_name == "gguf":
            from core.inference.llama_cpp import LlamaCppBackend
            backend = LlamaCppBackend.__new__(LlamaCppBackend)
            list(
                backend.generate_chat_completion_with_tools(
                    messages = [], tools = [], permission_mode = "full"
                )
            )
        elif loop_name == "safetensors":
            from core.inference.safetensors_agentic import run_safetensors_tool_loop
            list(
                run_safetensors_tool_loop(
                    single_turn = lambda *_: pytest.fail("model ran"),
                    messages = [],
                    tools = [],
                    execute_tool = lambda *a, **kw: pytest.fail("tool ran"),
                    permission_mode = "full",
                )
            )
        else:
            from core.inference.studio_tool_loop import (
                ToolLoopPolicy,
                ToolLoopRun,
                stream_with_studio_tools,
            )

            transport = SimpleNamespace(heals_text_tool_calls = False)
            run = ToolLoopRun(model = "test", messages = [])
            loop_policy = ToolLoopPolicy(
                tools = [],
                max_calls = 1,
                timeout = 1,
                rag_scope = None,
                permission_mode = "full",
                confirm_calls = False,
                bypass_permissions = False,
            )
            if loop_name == "studio":
                stream = stream_with_studio_tools(
                    transport, run = run, policy = loop_policy, cancel_event = threading.Event()
                )
            else:
                from core.inference.openai_codex_tool_loop import (
                    CodexRunContext,
                    CodexToolPolicy,
                    stream_codex_with_studio_tools,
                )
                stream = stream_codex_with_studio_tools(
                    client = None,
                    cancel_event = threading.Event(),
                    run = CodexRunContext(
                        model = "test",
                        messages = [],
                        provider_id = "test",
                        thread_id = None,
                        session_id = None,
                        reasoning_effort = None,
                    ),
                    policy = CodexToolPolicy(
                        tools = [],
                        max_calls = 1,
                        timeout = 1,
                        rag_scope = None,
                        permission_mode = "full",
                        confirm_calls = False,
                        bypass_permissions = False,
                    ),
                )

            async def consume():
                return [event async for event in stream]

            asyncio.run(consume())

    with pytest.raises(HTTPException) as exc:
        run_as(account, check)
    assert exc.value.status_code == 400


def test_mcp_lru_never_evicts_another_accounts_session(monkeypatch):
    key = run_as(BOB, mcp_client._session_key, "https://public.example/mcp", None, "chat")
    session = SimpleNamespace(last_used = 0, in_flight = 0)
    monkeypatch.setattr(mcp_client, "_mcp_sessions", {key: session})
    monkeypatch.setattr(mcp_client, "_MAX_SESSIONS", 1)
    assert run_as(ALICE, mcp_client._evict_lru_locked) == []
    assert mcp_client._mcp_sessions[key] is session


def test_mcp_public_transport_pins_dns_and_checks_redirects(monkeypatch):
    seen = []

    async def check():
        client = mcp_client._public_http_client_factory()
        http = sys.modules[type(client).__module__.split(".")[0]]

        async def send(transport, request):
            seen.append(
                (str(request.url), request.headers["host"], request.extensions["sni_hostname"])
            )
            return http.Response(302, headers = {"location": "http://127.0.0.1/private"})

        monkeypatch.setattr(http.AsyncHTTPTransport, "handle_async_request", send)

        def resolve(host, port, **kwargs):
            ip = "8.8.8.8" if host == "public.example" else host
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]

        monkeypatch.setattr(socket, "getaddrinfo", resolve)
        async with client:
            with pytest.raises(HTTPException):
                await client.get("https://public.example/mcp")

    asyncio.run(check())
    assert seen == [("https://8.8.8.8/mcp", "public.example", "public.example")]


def test_managed_transport_and_oauth_use_public_http_factory(monkeypatch):
    import fastmcp
    import fastmcp.client.transports as transports

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))],
    )
    monkeypatch.setattr(fastmcp, "Client", lambda transport: transport)
    monkeypatch.setattr(transports, "StreamableHttpTransport", lambda **kwargs: kwargs)
    oauth = SimpleNamespace(httpx_client_factory = None)
    monkeypatch.setattr(mcp_client, "_oauth", lambda url: oauth)
    result = run_as(ALICE, mcp_client._client, "https://public.example/mcp", None, True)
    assert result["httpx_client_factory"] is mcp_client._public_http_client_factory
    assert oauth.httpx_client_factory is mcp_client._public_http_client_factory
    owner = mcp_client._client("http://127.0.0.1/mcp", None)
    assert "httpx_client_factory" not in owner


def test_workspace_cleanup_thread_retains_account(monkeypatch):
    seen = []
    monkeypatch.setattr(tools, "wait_for_sessions_idle", lambda *a, **kw: True)
    monkeypatch.setattr(
        tools, "collect_orphaned_project_workspaces", lambda: seen.append(current_account_id())
    )
    worker = run_as(ALICE, tools.finish_workspace_delete_when_idle, "project")
    worker.join(5)
    assert not worker.is_alive()
    assert seen == [ALICE.account_id]


@pytest.mark.parametrize("loop_name", ["gguf", "safetensors", "studio"])
def test_tool_loops_keep_account_in_real_worker_thread(loop_name, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    from core.inference import studio_tool_loop

    seen = []

    def execute(name, arguments, **kwargs):
        seen.append(current_account_id())
        return "completed"

    monkeypatch.setattr(tools, "execute_tool", execute)
    monkeypatch.setattr(studio_tool_loop, "execute_tool", execute)
    monkeypatch.setattr(tools, "build_rag_autoinject", lambda *a, **kw: None)
    monkeypatch.setattr(studio_tool_loop, "build_rag_autoinject", lambda *a, **kw: None)
    schema = [
        {"type": "function", "function": {"name": "python", "parameters": {"type": "object"}}}
    ]
    if loop_name == "gguf":
        from test_llama_cpp_tool_loop import _make_backend, _sse, _done, _structured_tool_call
        backend = _make_backend(
            monkeypatch,
            [
                _structured_tool_call("python", {"code": "print(1)"}, "call1"),
                [_sse({"content": "done"}), _done()],
            ],
            [],
        )

        def run():
            return list(
                backend.generate_chat_completion_with_tools(
                    messages = [{"role": "user", "content": "run"}],
                    tools = schema,
                    permission_mode = "off",
                    max_tool_iterations = 1,
                )
            )
    elif loop_name == "safetensors":
        from core.inference.safetensors_agentic import run_safetensors_tool_loop
        turns = iter(
            ['<tool_call>{"name":"python","arguments":{"code":"print(1)"}}</tool_call>', "done"]
        )

        def run():
            return list(
                run_safetensors_tool_loop(
                    single_turn = lambda *_: iter([next(turns)]),
                    messages = [{"role": "user", "content": "run"}],
                    tools = schema,
                    execute_tool = execute,
                    permission_mode = "off",
                    max_tool_iterations = 1,
                )
            )
    else:
        from test_studio_tool_loop import FakeTransport, _run, _sse, _DONE
        transport = FakeTransport(
            [
                [
                    _sse(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call1",
                                    "function": {
                                        "name": "python",
                                        "arguments": '{"code":"print(1)"}',
                                    },
                                }
                            ]
                        }
                    ),
                    _sse(finish = "tool_calls"),
                    _DONE,
                ],
                [_sse({"content": "done"}), _sse(finish = "stop"), _DONE],
            ]
        )

        def run():
            return _run(transport, tools = schema)

    run_as(ALICE, run)
    assert seen == [ALICE.account_id]


def test_retrieval_worker_retains_account():
    result = run_as(
        ALICE,
        tools._search_knowledge_base_with_budget,
        {},
        None,
        5,
        None,
        search_fn = lambda *args: current_account_id(),
    )
    assert result == ALICE.account_id


def test_managed_sandbox_does_not_fall_back_to_owner_on_root_error(monkeypatch):
    from utils.paths import storage_roots

    def unavailable():
        raise RuntimeError("root unavailable")

    monkeypatch.setattr(storage_roots, "workspace_root", unavailable)
    with pytest.raises(RuntimeError, match = "root unavailable"):
        run_as(ALICE, tools.sandbox_root)
    assert tools.sandbox_root() == tools._legacy_sandbox_root()


def test_transport_refuses_managed_stdio_before_constructing_client(monkeypatch):
    import fastmcp

    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    monkeypatch.setattr(fastmcp, "Client", lambda *a, **kw: pytest.fail("local client constructed"))
    with pytest.raises(HTTPException) as exc:
        run_as(ALICE, mcp_client._client, "python server.py", None)
    assert exc.value.status_code == 400


def test_sandbox_recovery_runs_once_for_each_account(isolated, monkeypatch):
    seen = []
    monkeypatch.setattr(tools, "_swept_detached", True)
    monkeypatch.setattr(tools, "_swept_detached_accounts", set(), raising = False)
    monkeypatch.setattr(
        tools, "sweep_detached_sandboxes", lambda: seen.append(current_account_id())
    )
    monkeypatch.setattr(tools, "collect_orphaned_project_workspaces", lambda: None)
    for account in (ALICE, BOB):
        worker = run_as(account, isolated)
        assert worker is not None
        worker.join(5)
        assert not worker.is_alive()
        assert run_as(account, isolated) is None
    assert seen == [ALICE.account_id, BOB.account_id]
