# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A flow's Recipe Dataset runs in the account's studio.db."""

from .factory_base import Factory, seeder

MARKER = "flows-recipe-matrix-sentinel"
FLOW_ID = "flows-recipe-matrix-flow"
NODE_ID = "recipe"
RUN_ID = "flows-recipe-matrix-run"


@seeder("flow-recipe-run")
def seed_recipe_run(account) -> dict[str, str]:
    from storage import flows_db
    from utils.account_context import run_as

    def seed() -> None:
        flows_db.save_flow(
            {
                "format": "unsloth-flow",
                "version": 1,
                "id": FLOW_ID,
                "name": MARKER,
                "nodes": [
                    {"id": NODE_ID, "type": "input.text", "v": 1, "params": {}, "pos": [0, 0]}
                ],
                "edges": [],
            }
        )
        flows_db.save_recipe_run(
            FLOW_ID, NODE_ID, {"id": RUN_ID, "createdAt": 1000, "status": MARKER}
        )

    run_as(account, seed)
    return {"flow_id": FLOW_ID, "node_id": NODE_ID, "run_id": RUN_ID}


FACTORIES = {
    "routes.flows:GET:/{flow_id}/nodes/{node_id}/recipe-runs": Factory(
        "flow-recipe-run", fragment = MARKER, absent = MARKER
    ),
    "routes.flows:GET:/{flow_id}/nodes/{node_id}/recipe-runs/{run_id}": Factory(
        "flow-recipe-run", fragment = MARKER, absent = MARKER
    ),
}

SKIPPED: dict[str, str] = {}
