import asyncio
import logging
import sys
import types
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

# Keep this lifecycle-focused module independent of the optional structured logging dependency.
_loggers = types.ModuleType("loggers")
_loggers.get_logger = lambda name: logging.getLogger(name)
sys.modules.setdefault("loggers", _loggers)
_structlog = types.ModuleType("structlog")
_structlog.get_logger = lambda *args, **kwargs: logging.getLogger(
    args[0] if args else "structlog"
)
sys.modules.setdefault("structlog", _structlog)

import core.inference.llama_keepwarm as keepwarm
import routes.inference as inference_route
from models.inference import LoadAcceptedResponse, LoadRequest, LoadResponse


@pytest.fixture(autouse = True)
def reset_state():
    inference_route._load_admissions.clear()
    inference_route._pending_async_load = None
    inference_route._last_async_load_error = None
    yield
    for operation in list(inference_route._load_admissions.values()):
        if operation.task is not None and not operation.task.done():
            operation.task.cancel()
    inference_route._load_admissions.clear()
    inference_route._pending_async_load = None


def request(model = "unsloth/test", *, async_load = False, request_id = "load-1"):
    return LoadRequest(model_path = model, async_load = async_load, load_request_id = request_id)


def run(coro):
    return asyncio.run(coro)


def response(model):
    return LoadResponse(status = "loaded", model = model, display_name = "test", inference = {})


def test_defaults_to_sync_and_preserves_response(monkeypatch):
    calls = []

    async def load(req, fastapi_request, subject, **kwargs):
        calls.append(req.async_load)
        return response(req.model_path)

    monkeypatch.setattr(inference_route, "_load_model_impl", load)
    result = run(inference_route.load_model(request(), object(), "subject"))
    assert isinstance(result, LoadResponse)
    assert calls == [False]


def test_async_requires_non_empty_request_id():
    with pytest.raises(Exception):
        LoadRequest(model_path = "m", async_load = True, load_request_id = "")


def test_async_returns_before_background_work(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()

    async def load(req, fastapi_request, subject, **kwargs):
        started.set()
        await release.wait()
        return response(req.model_path)

    monkeypatch.setattr(inference_route, "_load_model_impl", load)
    monkeypatch.setattr(keepwarm, "acquire_inference_lifecycle_gate_nowait", lambda: True)
    monkeypatch.setattr(keepwarm, "release_inference_lifecycle_gate", lambda: None)

    async def scenario():
        accepted = await inference_route.load_model(request(async_load = True), object(), "subject")
        assert isinstance(accepted, LoadAcceptedResponse)
        assert inference_route.get_pending_async_load_model() == "unsloth/test"
        release.set()
        await inference_route._pending_async_load.task

    run(scenario())


def test_duplicate_subject_and_id_are_rejected(monkeypatch):
    release = asyncio.Event()

    async def load(req, fastapi_request, subject, **kwargs):
        await release.wait()
        return response(req.model_path)

    monkeypatch.setattr(inference_route, "_load_model_impl", load)
    monkeypatch.setattr(keepwarm, "acquire_inference_lifecycle_gate_nowait", lambda: True)
    monkeypatch.setattr(keepwarm, "release_inference_lifecycle_gate", lambda: None)

    async def scenario():
        await inference_route.load_model(request(async_load = True), object(), "a")
        with pytest.raises(HTTPException) as duplicate:
            await inference_route.load_model(request("other", async_load = True), object(), "a")
        with pytest.raises(HTTPException) as isolated:
            await inference_route.load_model(request("other", async_load = True), object(), "b")
        release.set()
        await inference_route._pending_async_load.task
        return duplicate.value, isolated.value

    duplicate, isolated = run(scenario())
    assert duplicate.status_code == isolated.status_code == 409


def test_busy_gate_does_not_admit(monkeypatch):
    monkeypatch.setattr(keepwarm, "acquire_inference_lifecycle_gate_nowait", lambda: False)
    with pytest.raises(HTTPException) as exc:
        run(inference_route.load_model(request(async_load = True), object(), "subject"))
    assert exc.value.status_code == 409
    assert not inference_route._load_admissions


def test_status_pending_error_is_published_in_idle_branch(monkeypatch):
    monkeypatch.setattr(
        inference_route,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(is_loaded = False),
    )
    monkeypatch.setattr(
        inference_route,
        "_probe_llama_cpp_status",
        lambda _backend: (False, {}),
    )
    monkeypatch.setattr(inference_route, "_peek_inference_backend", lambda: None)
    inference_route._last_async_load_error = "Model load failed"
    status = run(inference_route.get_status("subject"))
    assert status.loading == []
    assert status.load_error == "Model load failed"


def test_pending_admission_keeps_public_label_separate_from_delete_identity(monkeypatch):
    release = asyncio.Event()

    async def load(req, fastapi_request, subject, **kwargs):
        await release.wait()
        return response(req.model_path)

    monkeypatch.setattr(inference_route, "_load_model_impl", load)
    monkeypatch.setattr(keepwarm, "acquire_inference_lifecycle_gate_nowait", lambda: True)
    monkeypatch.setattr(keepwarm, "release_inference_lifecycle_gate", lambda: None)

    async def scenario():
        accepted = await inference_route.load_model(
            request("org/test", async_load = True), object(), "subject"
        )
        assert accepted.model == "org/test"
        assert inference_route.get_pending_async_load_model() == "org/test"
        assert inference_route.get_pending_async_load_deletion_path() == "org/test"
        release.set()
        await inference_route._pending_async_load.task

    run(scenario())


def test_async_load_preserves_active_generation_callback(monkeypatch):
    release = asyncio.Event()
    callbacks = []

    async def load(req, fastapi_request, subject, **kwargs):
        kwargs["on_reload_confirmed"](cancel = False)
        await release.wait()
        return response(req.model_path)

    monkeypatch.setattr(inference_route, "_load_model_impl", load)
    monkeypatch.setattr(keepwarm, "acquire_inference_lifecycle_gate_nowait", lambda: True)
    monkeypatch.setattr(keepwarm, "release_inference_lifecycle_gate", lambda: None)
    monkeypatch.setattr(
        inference_route,
        "_raise_or_cancel_active_generations",
        lambda **kwargs: callbacks.append(kwargs),
    )

    async def scenario():
        await inference_route.load_model(
            request(async_load = True), object(), "subject"
        )
        await asyncio.sleep(0)
        release.set()
        await inference_route._pending_async_load.task

    run(scenario())
    assert callbacks == [{"force": False, "action": "Loading a model", "cancel": False}]


def test_cancelling_async_waiter_keeps_lease_until_load_finishes(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()
    released = []

    async def load(req, fastapi_request, subject, **kwargs):
        started.set()
        await release.wait()
        return response(req.model_path)

    monkeypatch.setattr(inference_route, "_load_model_impl", load)
    monkeypatch.setattr(keepwarm, "acquire_inference_lifecycle_gate_nowait", lambda: True)
    monkeypatch.setattr(
        keepwarm,
        "release_inference_lifecycle_gate",
        lambda: released.append(True),
    )

    async def scenario():
        await inference_route.load_model(request(async_load = True), object(), "subject")
        task = inference_route._pending_async_load.task
        await started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert released == []
        assert inference_route.get_pending_async_load_model() == "unsloth/test"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(scenario())
    assert released == [True]


def test_cancellation_does_not_publish_error(monkeypatch):
    async def load(req, fastapi_request, subject, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(inference_route, "_load_model_impl", load)
    monkeypatch.setattr(keepwarm, "acquire_inference_lifecycle_gate_nowait", lambda: True)
    monkeypatch.setattr(keepwarm, "release_inference_lifecycle_gate", lambda: None)

    async def scenario():
        await inference_route.load_model(request(async_load = True), object(), "subject")
        with pytest.raises(asyncio.CancelledError):
            await inference_route._pending_async_load.task

    run(scenario())
    assert inference_route._last_async_load_error is None
