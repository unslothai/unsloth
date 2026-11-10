# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import contextlib
import functools
from typing import Any


def node(
    nid: str,
    type_: str,
    group: str | None = None,
    **params: Any,
) -> dict:
    found = {"id": nid, "type": type_, "v": 1, "params": params, "pos": [0, 0]}
    if group:
        found["group"] = group
    return found


def edge(
    src: str,
    sport: str,
    dst: str,
    dport: str,
    kind: str = "data",
) -> dict:
    return {
        "id": f"{src}-{sport}-{dst}-{dport}",
        "source": src,
        "source_port": sport,
        "target": dst,
        "target_port": dport,
        "kind": kind,
    }


def doc(
    nodes: list[dict],
    edges: list[dict] = (),
    **extra: Any,
) -> dict:
    found = {
        "format": "unsloth-flow",
        "version": 1,
        "id": "f1",
        "name": "Test flow",
        "nodes": nodes,
        "edges": list(edges),
    }
    found.update(extra)
    return found


def run_async(fn):
    """Run an async test on a fresh loop (the suite has no pytest-asyncio)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


def chat_entry_spec(type: str, **flags) -> dict:
    """A live entrypoint on the loaded chat model, like the ones a game player calls."""
    from core.flows.specs import P, inp, out, spec
    return spec(
        type,
        type,
        "test",
        "test",
        inputs = [inp("state", "json", optional = True)],
        outputs = [out("reply", "json")],
        params = [P("model", "model", None, "Model", model_kind = "chat")],
        lane = "gpu",
        model_kind = "chat",
        entrypoint = True,
        cacheable = False,
        shares_chat = True,
        **flags,
    )


# ---------------------------------------------------------------------------
# Engine and run helpers shared by every flows test module.


def make_engine(handlers: dict | None = None, cls = None):
    """A FlowEngine (or subclass ``cls``) with no real API key minting, running the given handlers."""
    from core.flows.engine import FlowEngine

    eng = (cls or FlowEngine)()
    eng.handlers = handlers if handlers is not None else {}
    eng.mint_key = lambda account: ("sk-unsloth-test", None)
    eng.revoke_key = lambda key_id: None
    return eng


async def run_doc(
    eng,
    d: dict,
    timeout: float = 30,
    **options,
):
    """Parse ``d``, start a run on ``eng`` and wait (bounded) until it ends."""
    from core.flows.compiler import parse_document

    parsed, errors = parse_document(d)
    assert not errors, errors
    rs = await eng.start_run(app = None, flow_id = d["id"], doc = parsed, options = options)
    await asyncio.wait_for(rs.task, timeout = timeout)
    return rs


def node_status(rs, nid: str) -> str:
    return rs.nodes[nid].status


async def ret(value):
    """An awaitable of ``value``: ``lambda ctx: ret({...})`` is a one-line node handler."""
    return value


async def text_handler(ctx):
    return {"text": str(ctx.params.get("value") or "")}


async def template_handler(ctx):
    values = ctx.inputs.get("values") or []
    return {"text": (ctx.params.get("template") or "{}").replace("{}", "+".join(map(str, values)))}


async def result_handler(ctx):
    return {"value": ctx.inputs.get("value")}


BASE_HANDLERS = {
    "input.text": text_handler,
    "text.template": template_handler,
    "output.result": result_handler,
}


async def wait_until(
    predicate,
    timeout: float = 10.0,
    what: str = "condition",
) -> None:
    """Yield to the loop until ``predicate()`` holds; fail after ``timeout`` seconds instead of hanging."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.001)


async def cancelled_event(ctx) -> asyncio.Event:
    """An Event set by the run's cancel hook, so a fake handler can block until Stop with no polling."""
    stopped = asyncio.Event()

    async def hook() -> None:
        stopped.set()

    ctx.on_cancel(hook)
    return stopped


# ---------------------------------------------------------------------------
# Routes client.

ACCOUNT_HEADER = "x-test-account"


def flow_accounts() -> dict:
    from utils.account_context import AccountContext, OWNER
    return {
        "owner": OWNER,
        "alice": AccountContext("alice-flows", "alice"),
        "bob": AccountContext("bob-flows", "bob"),
    }


@contextlib.contextmanager
def flows_test_client(
    monkeypatch,
    handlers: dict | None = None,
    hooks: bool = False,
    api_key: bool | None = None,
    accounts: dict | None = None,
):
    """A TestClient on the flows router with a fresh ENGINE; ``x-test-account`` picks the caller.

    ``handlers`` defaults to input.text + output.result, ``hooks`` mounts the trigger router,
    ``api_key`` pins ``authenticated_via_api_key`` and ``accounts`` replaces the header-to-account map.
    """
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient

    from auth.authentication import get_current_subject
    from core.flows import engine as engine_mod
    from routes import flows as flows_routes
    from utils.account_context import bind_account, reset_account

    eng = make_engine(
        handlers
        if handlers is not None
        else {"input.text": text_handler, "output.result": result_handler}
    )
    monkeypatch.setattr(engine_mod, "ENGINE", eng)
    accounts = accounts if accounts is not None else flow_accounts()
    app = FastAPI()

    @app.middleware("http")
    async def as_account(request: Request, call_next):
        token = bind_account(accounts[request.headers.get(ACCOUNT_HEADER, "owner")])
        try:
            return await call_next(request)
        finally:
            reset_account(token)

    if hooks:
        app.include_router(flows_routes.hooks_router, prefix = "/api/flows/hooks")
    app.include_router(flows_routes.router, prefix = "/api/flows")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    if api_key is not None:
        from auth.authentication import authenticated_via_api_key
        app.dependency_overrides[authenticated_via_api_key] = lambda: api_key
    with TestClient(app) as client:
        yield client


def wait_run(
    client,
    run_id: str,
    headers = None,
    timeout: float = 30,
) -> dict:
    """Block until the run's engine task has fully ended (final DB write included), then return its snapshot.

    Waits on the task itself through the client's event loop, so there is no polling and no race with the
    final ``update_run`` that lands after the in-memory status already reads terminal.
    """
    from core.flows import engine as engine_mod

    rs = engine_mod.ENGINE.runs.get(run_id)
    if rs is not None and rs.task is not None:

        async def settle():
            done, _ = await asyncio.wait({rs.task}, timeout = timeout)
            assert done, f"run {run_id} did not finish within {timeout}s"

        client.portal.call(settle)
    snap = client.get(f"/api/flows/runs/{run_id}", headers = headers).json()
    assert snap["run"]["status"] not in ("queued", "running"), snap["run"]["status"]
    return snap


# ---------------------------------------------------------------------------
# A node context whose Studio calls go to a fake route table.


class RouteCtx:
    """The ctx a node handler sees, with ``ctx.call`` answered by ``routes(method, path, params, json)``.

    Every call is recorded in ``calls`` as ``(method, path, params, json, files)``. ``node_type``
    fills ``params`` from the catalog defaults; keyword arguments override them. Subclasses set
    ``node_type``, ``default_inputs`` and ``seed`` and override ``answer`` for routes every test shares.
    """

    node_type: str | None = None
    default_inputs: dict = {}
    seed = 1

    def __init__(
        self,
        routes,
        *,
        inputs: dict | None = None,
        **params,
    ):
        from core.flows.specs import SPECS_BY_TYPE

        self.routes = routes
        self.calls: list[tuple] = []
        self.hooks: list = []
        self.partials: list[tuple] = []
        self.caller = object()
        self.cancelled = False
        self.inputs = dict(self.default_inputs if inputs is None else inputs)
        spec = SPECS_BY_TYPE.get(self.node_type) if self.node_type else None
        self.params = {p["id"]: p["default"] for p in spec["params"]} if spec else {}
        self.params.update(params)

    def answer(self, method, path, params, json):
        return self.routes(method, path, params, json)

    async def call(
        self,
        method,
        path,
        json = None,
        params = None,
        files = None,
        **_,
    ):
        self.calls.append((method, path, params, json, files))
        return self.answer(method, path, params, json)

    def check_cancelled(self):
        if self.cancelled:
            from core.flows.registry import FlowCancelled
            raise FlowCancelled()

    def on_cancel(self, fn):
        self.hooks.append(fn)
        self.cancel = fn

    def partial(self, port, value, **k):
        self.partials.append((port, value))

    def progress(self, *a, **k):
        pass

    def log(self, *a, **k):
        pass
