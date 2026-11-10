# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Flows and everything stored under them (runs, events, node results, saves, Recipe Dataset runs, capabilities,
triggers, assets) in the account's studio.db and workspace."""

from .factory_base import Factory, seeder

MARKER = "flows-recipe-matrix-sentinel"
EDITED = "flows-matrix-edited"
FLOW_ID = "flows-recipe-matrix-flow"
NODE_ID = "recipe"
REPLY_ID = "reply"
HOOK_ID = "hook"
RUN_ID = "flows-recipe-matrix-run"
FLOW_RUN_ID = "flows-matrix-run"
SLOT = "matrix-slot"
EXAMPLE_ID = "lantern-picnic"
ASSET_TEXT = f"{MARKER} asset\n".encode()


def _doc(name: str = MARKER) -> dict:
    return {
        "format": "unsloth-flow",
        "version": 1,
        "id": FLOW_ID,
        "name": name,
        "nodes": [
            {"id": NODE_ID, "type": "input.text", "v": 1, "params": {}, "pos": [0, 0]},
            {"id": REPLY_ID, "type": "chat.output", "v": 1, "params": {}, "pos": [200, 0]},
        ],
        "edges": [
            {
                "id": "e1",
                "source": NODE_ID,
                "source_port": "text",
                "target": REPLY_ID,
                "target_port": "reply",
            }
        ],
    }


@seeder("flow")
def seed_flow(account) -> dict[str, str]:
    from core.flows import assets as asset_store
    from storage import flows_db
    from utils.account_context import run_as

    def seed() -> str:
        flows_db.save_flow(_doc())
        flows_db.save_recipe_run(
            FLOW_ID, NODE_ID, {"id": RUN_ID, "createdAt": 1000, "status": MARKER}
        )
        flows_db.create_run(FLOW_RUN_ID, FLOW_ID, {})
        flows_db.update_run(
            FLOW_RUN_ID,
            status = "completed",
            outputs = {"text": MARKER},
            started_at = 1000,
            finished_at = 2000,
        )
        flows_db.append_event(
            FLOW_RUN_ID,
            1,
            {
                "seq": 1,
                "run_id": FLOW_RUN_ID,
                "type": "run.completed",
                "node_id": None,
                "ts": 1.0,
                "data": {},
            },
        )
        flows_db.save_node_result(FLOW_RUN_ID, NODE_ID, "done", {"text": MARKER}, None)
        flows_db.put_session(FLOW_ID, SLOT, {"note": MARKER}, MARKER)
        flows_db.set_capabilities(FLOW_ID, {"hosts": [MARKER]})
        flows_db.replace_triggers(
            FLOW_ID,
            [{"node_id": HOOK_ID, "kind": "webhook", "config": {"enabled": False}}],
            lambda: "matrix-webhook-token-" + "0" * 24,
        )
        digest, _mime = asset_store.store(ASSET_TEXT, "text/plain")
        return digest

    digest = run_as(account, seed)
    return {
        "flow_id": FLOW_ID,
        "node_id": NODE_ID,
        "run_id": RUN_ID,
        "slot": SLOT,
        "ref": f"sha256:{digest}",
    }


@seeder("flow-run")
def seed_flow_run(account) -> dict[str, str]:
    params = seed_flow(account)
    return {**params, "run_id": FLOW_RUN_ID}


@seeder("flow-trigger")
def seed_flow_trigger(account) -> dict[str, str]:
    params = seed_flow(account)
    return {**params, "node_id": HOOK_ID}


@seeder("flow-example")
def seed_flow_example(account) -> dict[str, str]:
    seed_flow(account)
    return {"example_id": EXAMPLE_ID}


OWNER_ONLY = "owner-only route: a managed account gets 403 before its flow is looked up"
IDEMPOTENT_DELETE = (
    "idempotent delete: another account's call deletes nothing in its own studio.db and gets 204; "
    "the seeded account's snapshot must stay unchanged"
)

FACTORIES = {
    "routes.flows:GET:/{flow_id}": Factory("flow", fragment = MARKER, absent = MARKER),
    "routes.flows:PUT:/{flow_id}": Factory(
        "flow",
        {"doc": _doc(EDITED)},
        fragment = EDITED,
        owner = (200,),
        wrong = (200,),
        absent = MARKER,
        reason = "upsert: another account's PUT saves a new flow in its own studio.db; the seeded account's "
        "snapshot must stay unchanged",
    ),
    "routes.flows:DELETE:/{flow_id}": Factory(
        "flow",
        success = 204,
        owner = (204,),
        wrong = (204,),
        absent = MARKER,
        reason = IDEMPOTENT_DELETE,
    ),
    "routes.flows:GET:/{flow_id}/runs": Factory(
        "flow",
        fragment = FLOW_RUN_ID,
        owner = (200,),
        wrong = (200,),
        absent = FLOW_RUN_ID,
        reason = "no flow lookup: another account reads its own studio.db and gets an empty run list",
    ),
    # The route compiles before starting: an invalid doc shows the flow was found without a live engine run.
    "routes.flows:POST:/{flow_id}/runs": Factory(
        "flow",
        {"doc": {"nodes": "invalid"}},
        right = (400,),
        self_expected = (400,),
        fragment = "Invalid flow",
        absent = MARKER,
        reason = "no live engine run in the matrix: an invalid doc gets 400 only once the flow is found",
    ),
    "routes.flows:POST:/{flow_id}/plan": Factory("flow", {}, fragment = '"gpu_ops"', absent = MARKER),
    "routes.flows:POST:/{flow_id}/invoke": Factory(
        "flow", {"entry": REPLY_ID, "warm": True}, fragment = '"warm"', absent = MARKER
    ),
    "routes.flows:GET:/{flow_id}/story": Factory(
        "flow",
        right = (200, 422),
        self_expected = (200, 422),
        absent = MARKER,
        reason = "a flow without story blocks may be refused with 422 after it is found",
    ),
    "routes.flows:GET:/{flow_id}/latest-outputs": Factory("flow", fragment = MARKER, absent = MARKER),
    "routes.flows:GET:/{flow_id}/nodes/{node_id}/history": Factory(
        "flow", fragment = MARKER, absent = MARKER
    ),
    "routes.flows:GET:/{flow_id}/nodes/{node_id}/recipe-runs": Factory(
        "flow", fragment = MARKER, absent = MARKER
    ),
    "routes.flows:GET:/{flow_id}/nodes/{node_id}/recipe-runs/{run_id}": Factory(
        "flow", fragment = MARKER, absent = MARKER
    ),
    "routes.flows:GET:/{flow_id}/sessions": Factory("flow", fragment = MARKER, absent = MARKER),
    "routes.flows:GET:/{flow_id}/sessions/{slot}": Factory("flow", fragment = MARKER, absent = MARKER),
    "routes.flows:PUT:/{flow_id}/sessions/{slot}": Factory(
        "flow", {"state": {"note": EDITED}, "label": EDITED}, fragment = EDITED, absent = MARKER
    ),
    "routes.flows:DELETE:/{flow_id}/sessions/{slot}": Factory(
        "flow",
        success = 204,
        owner = (204,),
        wrong = (204,),
        absent = MARKER,
        reason = IDEMPOTENT_DELETE,
    ),
    "routes.flows:GET:/{flow_id}/export": Factory(
        "flow", query = {"mode": "json"}, fragment = MARKER, absent = MARKER
    ),
    "routes.flows:GET:/{flow_id}/capabilities": Factory("flow", fragment = MARKER, absent = MARKER),
    "routes.flows:POST:/{flow_id}/approve": Factory(
        "flow",
        {},
        right = (403,),
        wrong = (403,),
        self_expected = (200,),
        absent = MARKER,
        reason = OWNER_ONLY,
    ),
    "routes.flows:GET:/{flow_id}/triggers": Factory("flow", fragment = '"owner_only"', absent = MARKER),
    "routes.flows:POST:/{flow_id}/triggers/{node_id}/rotate": Factory(
        "flow-trigger",
        right = (403,),
        wrong = (403,),
        self_expected = (200,),
        absent = MARKER,
        reason = OWNER_ONLY,
    ),
    "routes.flows:GET:/runs/{run_id}": Factory("flow-run", fragment = MARKER, absent = MARKER),
    "routes.flows:POST:/runs/{run_id}/cancel": Factory(
        "flow-run", fragment = "completed", absent = MARKER
    ),
    "routes.flows:GET:/runs/{run_id}/events": Factory("flow-run", fragment = MARKER, absent = MARKER),
    "routes.flows:POST:/runs/{run_id}/events": Factory("flow-run", fragment = MARKER, absent = MARKER),
    "routes.flows:GET:/assets/{ref}": Factory("flow", fragment = MARKER, absent = MARKER),
    # Built-in examples are shared read-only templates; a copy lands in the caller's own studio.db.
    "routes.flows:POST:/examples/{example_id}/copy": Factory(
        "flow-example",
        owner = (200,),
        wrong = (200,),
        absent = MARKER,
        reason = "built-in example, not an account object: every account copies it into its own studio.db",
    ),
}

SKIPPED: dict[str, str] = {}
