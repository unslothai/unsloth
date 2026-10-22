# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Flows for agents: list the saved flows and run one (or a flow document the agent wrote), with its inputs.
Runs go on in Unsloth Studio after the call returns; follow them with get_job(kind="flow") and stop them with
cancel(kind="flow")."""

from __future__ import annotations

import asyncio
import time
from typing import Annotated, Any, Optional
from urllib.parse import quote

from fastmcp.exceptions import ToolError
from pydantic import Field

from studio_mcp import loading
from studio_mcp.caller import Caller, current_caller
from studio_mcp.outputs import FlowRunInfo, JobStatus, RouteText, ToolOutput
from studio_mcp.tools import READ_ONLY, WRITES, as_dict, opt_text, route_json

FLOWS = "/api/flows"
TERMINAL = ("completed", "failed", "cancelled", "interrupted")
POLL_S = 2.0


class FlowRef(ToolOutput):
    id: RouteText
    name: RouteText
    kind: RouteText
    description: Optional[RouteText] = None
    node_count: Optional[int] = None
    example: bool = False


class FlowList(ToolOutput):
    flows: list[FlowRef] = []
    examples: list[FlowRef] = []


def _ref(row: dict, example: bool = False) -> FlowRef:
    return FlowRef(
        id = str(row.get("id")),
        name = str(row.get("name") or "Untitled flow"),
        kind = str(row.get("kind") or "pipeline"),
        description = opt_text(row.get("description")),
        node_count = row.get("nodeCount") if isinstance(row.get("nodeCount"), int) else None,
        example = example,
    )


async def list_flows() -> FlowList:
    """The flows saved in Unsloth Studio (newest first) and the bundled examples. A flow is a node graph of Unsloth Studio steps: LLM text, images, video, audio, datasets, training, logic. Run one with run_flow; copy an example first by running it with ``example_id``."""
    caller = current_caller()
    saved = as_dict(await route_json("GET", f"{FLOWS}/", caller = caller))
    examples = as_dict(await route_json("GET", f"{FLOWS}/examples", caller = caller))
    return FlowList(
        flows = [_ref(r) for r in saved.get("flows") or [] if isinstance(r, dict)],
        examples = [_ref(r, True) for r in examples.get("examples") or [] if isinstance(r, dict)],
    )


def _redacted(value: Any) -> Any:
    """Flow outputs are model text and media links; host paths in them are scrubbed like every route text."""
    from hub.utils.host_paths import redact_paths_in_text

    if isinstance(value, str):
        return redact_paths_in_text(value)
    if isinstance(value, list):
        return [_redacted(v) for v in value]
    if isinstance(value, dict):
        return {k: _redacted(v) for k, v in value.items()}
    return value


def flow_status(snapshot: dict) -> JobStatus:
    run = as_dict(snapshot.get("run"))
    nodes = as_dict(snapshot.get("nodes"))
    finished = [
        n
        for n, s in nodes.items()
        if as_dict(s).get("status") in ("done", "cached", "skipped", "failed", "cancelled")
    ]
    failed = [n for n, s in nodes.items() if as_dict(s).get("status") == "failed"]
    total = len(nodes)
    errors = [
        f"{n}: {as_dict(nodes[n]).get('error')}" for n in failed if as_dict(nodes[n]).get("error")
    ]
    return JobStatus(
        kind = "flow",
        id = opt_text(run.get("id")),
        status = opt_text(run.get("status")),
        progress_percent = round(100.0 * len(finished) / total, 1) if total else None,
        error = opt_text(run.get("error")) or ("; ".join(errors)[:2000] if errors else None),
        flow = FlowRunInfo(
            flow_id = opt_text(run.get("flow_id")),
            outputs = _redacted(run["outputs"]) if isinstance(run.get("outputs"), dict) else None,
            nodes_done = len(finished),
            nodes_total = total,
            failed_nodes = failed or None,
        ),
    )


async def read_run(caller: Caller, run_id: str) -> JobStatus:
    snapshot = await route_json("GET", f"{FLOWS}/runs/{quote(run_id, safe = '')}", caller = caller)
    return flow_status(as_dict(snapshot))


async def wait_run(caller: Caller, run_id: str, wait_s: float) -> JobStatus:
    deadline = time.monotonic() + max(0.0, min(wait_s, loading.MAX_WAIT_S))
    status = await read_run(caller, run_id)
    while status.status not in TERMINAL and time.monotonic() < deadline:
        await asyncio.sleep(min(POLL_S, max(0.01, deadline - time.monotonic())))
        status = await read_run(caller, run_id)
    if status.status not in TERMINAL:
        status = status.model_copy(update = {"poll_after_s": 10})
    return status


async def run_flow(
    flow_id: Annotated[
        Optional[str], Field(description = "A saved flow's id from list_flows.")
    ] = None,
    example_id: Annotated[
        Optional[str],
        Field(description = "A bundled example's id from list_flows: copied, then run."),
    ] = None,
    document: Annotated[
        Optional[dict[str, Any]],
        Field(
            description = 'A flow document (format "unsloth-flow") to save and run instead of a saved flow.'
        ),
    ] = None,
    params: Annotated[
        Optional[dict[str, Any]],
        Field(description = "Values for the flow's Flow Input nodes, by their name."),
    ] = None,
    wait_s: Annotated[
        int,
        Field(
            ge = 0,
            le = loading.MAX_WAIT_S,
            description = "Seconds to wait for the run to finish, 0 to 240.",
        ),
    ] = 0,
) -> JobStatus:
    """Run a flow in Unsloth Studio: one of ``flow_id``, ``example_id`` or ``document``. Returns the run's status at once (or after ``wait_s``); when it completes, ``flow.outputs`` holds the results of its Flow Output nodes. A run keeps going after this returns: follow it with get_job(kind="flow", id=…, wait_s=…) and stop it with cancel(kind="flow", id=…). Steps load the models their nodes name, unloading others to make room; GPU steps run one at a time."""
    caller = current_caller()
    if sum(x is not None for x in (flow_id, example_id, document)) != 1:
        raise ToolError("Pass exactly one of flow_id, example_id or document.")
    if example_id is not None:
        copied = as_dict(
            await route_json(
                "POST", f"{FLOWS}/examples/{quote(example_id, safe = '')}/copy", caller = caller
            )
        )
        flow_id = opt_text(copied.get("id"))
    elif document is not None:
        created = as_dict(
            await route_json("POST", f"{FLOWS}/", caller = caller, json_body = {"doc": document})
        )
        flow_id = opt_text(created.get("id"))
    if not flow_id:
        raise ToolError("Unsloth Studio did not save the flow.")
    started = as_dict(
        await route_json(
            "POST",
            f"{FLOWS}/{quote(flow_id, safe = '')}/runs",
            caller = caller,
            json_body = {"params": params or {}},
            hints = {400: "Fix the flow: the error lists each node or connection that is wrong."},
        )
    )
    run_id = opt_text(started.get("run_id"))
    if not run_id:
        raise ToolError("Unsloth Studio did not start the run.")
    return await wait_run(caller, run_id, wait_s)


TOOLS = (
    (list_flows, READ_ONLY, FlowList.model_json_schema()),
    (run_flow, WRITES, JobStatus.model_json_schema()),
)
