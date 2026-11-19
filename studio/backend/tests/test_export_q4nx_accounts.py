# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Q4NX export: only the owner of a machine with an AMD NPU and FastFlowLM, and never another account's GGUF."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from models import ConvertQ4NXRequest
from routes import export as export_routes
from utils.account_context import OWNER, AccountContext, arun_as
from utils.paths import workspace_root

ALICE = AccountContext("alice-q4nx-account", "alice")
BOB = AccountContext("bob-q4nx-account", "bob")


@pytest.fixture(autouse = True)
def npu_machine(monkeypatch):
    from utils import q4nx_export_gate
    monkeypatch.setattr(q4nx_export_gate, "q4nx_export_allowed", lambda: True)


def _convert(
    account,
    gguf_path = None,
    **fields,
):
    request = ConvertQ4NXRequest(
        save_directory = "q4nx",
        gguf_path = str(gguf_path) if gguf_path else None,
        base_model = fields.pop("base_model", "unsloth/Qwen3-0.6B"),
        hf_token = "own-token",
        **fields,
    )
    return asyncio.run(
        arun_as(account, export_routes.convert_q4nx(request, "subject", allow_ambient = False))
    )


@pytest.fixture
def authorized(monkeypatch):
    checked = []
    monkeypatch.setattr(
        export_routes.account_access,
        "authorize_download",
        lambda repo, repo_type, token: checked.append(repo),
    )
    return checked


@pytest.fixture
def converted(monkeypatch, authorized):
    calls = []
    real_to_thread = asyncio.to_thread

    async def to_thread(fn, *args, **kwargs):
        if fn is export_routes.account_access.authorize_download:
            return await real_to_thread(fn, *args, **kwargs)
        calls.append(fn)
        raise RuntimeError("stop before converting")

    monkeypatch.setattr(export_routes.asyncio, "to_thread", to_thread)
    return calls


def test_hub_repos_are_authorized_for_the_caller_before_converting(converted, authorized):
    with pytest.raises(HTTPException):
        _convert(ALICE, repo_id = "org/private-GGUF", filename = "m.Q4_1.gguf")
    assert authorized == ["org/private-GGUF", "unsloth/Qwen3-0.6B"]


def test_a_refused_hub_repo_stops_before_converting(monkeypatch, converted):
    def refuse(repo, repo_type, token):
        raise HTTPException(status_code = 404, detail = "Repository not found")

    monkeypatch.setattr(export_routes.account_access, "authorize_download", refuse)
    with pytest.raises(HTTPException) as exc:
        _convert(ALICE, repo_id = "org/private-GGUF", filename = "m.Q4_1.gguf")
    assert exc.value.status_code == 404
    assert converted == []


def test_a_companion_symlink_out_of_the_workspace_is_refused(tmp_path):
    from core.export import q4nx
    from utils.account_context import run_as

    base = run_as(ALICE, workspace_root) / "models" / "base"
    base.mkdir(parents = True, exist_ok = True)
    secret = tmp_path / "other_account_template.jinja"
    secret.write_text("private")
    link = base / "chat_template.jinja"
    link.unlink(missing_ok = True)
    link.symlink_to(secret)

    with pytest.raises(HTTPException) as exc:
        run_as(ALICE, q4nx._fetch_base_file, str(base), "chat_template.jinja", None)
    assert exc.value.status_code == 403
    assert run_as(OWNER, q4nx._fetch_base_file, str(base), "chat_template.jinja", None) == link


def test_managed_account_cannot_convert_another_accounts_gguf(converted):
    from utils.account_context import run_as

    foreign = run_as(BOB, workspace_root) / "exports" / "private.Q4_1.gguf"
    with pytest.raises(HTTPException) as exc:
        _convert(ALICE, foreign)
    assert exc.value.status_code == 403
    assert converted == []


def test_managed_account_and_owner_reach_the_converter_for_allowed_paths(converted):
    from utils.account_context import run_as

    own = run_as(ALICE, workspace_root) / "exports" / "mine.Q4_1.gguf"
    for account, path in ((ALICE, own), (OWNER, "/anywhere/model.Q4_1.gguf")):
        with pytest.raises(HTTPException) as exc:
            _convert(account, path)
        assert exc.value.status_code == 400 and "stop before converting" in exc.value.detail
    assert len(converted) == 2


@pytest.mark.parametrize("depth, registered", [(("q4nx",), False), (("a", "b", "c", "d"), True)])
def test_a_conversion_deeper_than_the_npu_scan_is_registered(
    monkeypatch, authorized, depth, registered
):
    from utils.paths.storage_roots import exports_root

    out = exports_root().joinpath(*depth, "m.Q4_1-q4nx")
    calls = []
    real_to_thread = asyncio.to_thread

    async def to_thread(fn, *args, **kwargs):
        if fn is export_routes.account_access.authorize_download:
            return None
        if fn is export_routes._try_register_external_export:
            calls.append(args[0])
            return True, str(args[0])
        if fn is export_routes._export_details:
            return {"output_path": str(args[0])}
        return str(out)

    monkeypatch.setattr(export_routes.asyncio, "to_thread", to_thread)
    _convert(OWNER, repo_id = "org/r-GGUF", filename = "m.Q4_1.gguf")
    assert calls == ([out] if registered else [])


@pytest.mark.parametrize("depth, registered", [(("gguf",), False), (("a", "b", "c", "d"), True)])
def test_a_gguf_export_with_q4nx_deeper_than_the_npu_scan_registers_its_save_folder(
    monkeypatch, depth, registered
):
    from models import ExportGGUFRequest
    from utils.paths.storage_roots import exports_root

    save = exports_root().joinpath(*depth)
    calls = []

    async def supported():
        return None

    async def to_thread(fn, *args, **kwargs):
        if fn is export_routes._try_register_external_export:
            calls.append(args[0])
            return True, str(args[0])
        if fn is export_routes._export_details:
            return {"output_path": str(args[0])}
        return True, "ok", str(save)

    monkeypatch.setattr(export_routes, "_ensure_export_supported", supported)
    backend = type("Backend", (), {"export_gguf": None, "decision": None})()
    monkeypatch.setattr(export_routes, "get_export_backend", lambda: backend)
    monkeypatch.setattr(export_routes, "_with_drafter", lambda details, path: details)
    monkeypatch.setattr(export_routes.asyncio, "to_thread", to_thread)
    request = ExportGGUFRequest(
        save_directory = "/".join(depth), quantization_method = "q4_1", npu_q4nx = True
    )
    asyncio.run(export_routes.export_gguf(request, "subject", allow_ambient = False))
    assert calls == ([save] if registered else [])


def test_a_machine_without_an_npu_and_fastflowlm_is_refused_before_anything(
    monkeypatch, authorized
):
    from utils import q4nx_export_gate

    monkeypatch.setattr(q4nx_export_gate, "q4nx_export_allowed", lambda: False)
    with pytest.raises(HTTPException) as exc:
        _convert(ALICE, repo_id = "org/private-GGUF", filename = "m.Q4_1.gguf")
    assert exc.value.status_code == 403
    assert authorized == []


@pytest.mark.parametrize(
    "status, allowed",
    [
        ({"supported": True, "ready": True}, True),
        # An NPU whose FastFlowLM runtime is not installed yet.
        ({"supported": True, "ready": False}, False),
        # NVIDIA or any other machine without a supported NPU.
        ({"supported": False, "ready": False}, False),
        ({"supported": False, "ready": True}, False),
    ],
)
def test_q4nx_export_needs_an_npu_and_its_runtime(monkeypatch, status, allowed):
    from core.inference import npu_backend
    from utils import q4nx_export_gate as gate

    monkeypatch.undo()
    monkeypatch.setattr(
        npu_backend, "get_npu_backend", lambda: type("B", (), {"status": lambda self: status})()
    )
    assert gate.q4nx_export_allowed() is allowed


def test_a_failed_npu_probe_refuses_q4nx_export(monkeypatch):
    from core.inference import npu_backend
    from utils import q4nx_export_gate as gate

    def broken():
        raise RuntimeError("probe failed")

    monkeypatch.undo()
    monkeypatch.setattr(npu_backend, "get_npu_backend", broken)
    assert gate.q4nx_export_allowed() is False


def test_a_managed_account_is_refused_as_if_there_were_no_npu(monkeypatch):
    from core.inference import npu_backend
    from utils import q4nx_export_gate as gate

    monkeypatch.undo()
    monkeypatch.setattr(
        npu_backend,
        "get_npu_backend",
        lambda: type("B", (), {"status": lambda self: {"supported": True, "ready": True}})(),
    )
    real_to_thread = asyncio.to_thread
    calls = []

    async def to_thread(fn, *args, **kwargs):
        if fn is export_routes.account_access.authorize_download:
            return await real_to_thread(fn, *args, **kwargs)
        calls.append(fn)
        raise RuntimeError("stop before converting")

    monkeypatch.setattr(export_routes.asyncio, "to_thread", to_thread)
    monkeypatch.setattr(
        export_routes.account_access, "authorize_download", lambda repo, repo_type, token: None
    )
    # Same answer as a machine without an NPU, so the status stays hidden from the account.
    with pytest.raises(HTTPException) as exc:
        _convert(ALICE, repo_id = "org/m-GGUF", filename = "m.Q4_1.gguf")
    assert exc.value.status_code == 403
    assert exc.value.detail == gate.Q4NX_EXPORT_UNAVAILABLE_DETAIL
    with pytest.raises(HTTPException) as exc:
        _convert(OWNER, repo_id = "org/m-GGUF", filename = "m.Q4_1.gguf")
    assert exc.value.status_code == 400 and "stop before converting" in exc.value.detail
    assert len(calls) == 1
