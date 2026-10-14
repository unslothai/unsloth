# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Account isolation cases for saved Studio routines and their runs."""

from utils.account_context import run_as

from .factory_base import Factory, seeder

MARKER = "routine-matrix-sentinel"
EDITED = "routine-matrix-edited"
STEP = {
    "kind": "recipe",
    "label": "Generate a sample",
    "recipe": {
        "recipe": {
            "columns": [
                {
                    "column_type": "sampler",
                    "name": "topic",
                    "sampler_type": "category",
                    "params": {"values": [MARKER]},
                }
            ]
        },
        "run": {"rows": 1},
    },
}


@seeder("routine")
def seed_routine(account) -> dict[str, str]:
    from storage import routines_db

    routine = run_as(
        account,
        routines_db.save_routine,
        routine_id = None,
        name = MARKER,
        steps = [STEP],
        interval_minutes = 60,
        enabled = False,
    )
    return {"routine_id": routine["id"]}


@seeder("routine-with-run")
def seed_routine_with_run(account) -> dict[str, str]:
    from storage import routines_db

    params = seed_routine(account)
    run_as(account, routines_db.run_now, params["routine_id"])
    return params


FACTORIES = {
    "routes.routines:GET:/{routine_id}/runs": Factory(
        "routine-with-run", fragment = MARKER
    ),
    "routes.routines:PATCH:/{routine_id}/enabled": Factory(
        "routine", {"enabled": True}, fragment = MARKER
    ),
    "routes.routines:POST:/{routine_id}/run": Factory("routine"),
    "routes.routines:PUT:/{routine_id}": Factory(
        "routine",
        {"name": EDITED, "steps": [STEP], "interval_minutes": 60, "enabled": False},
        fragment = EDITED,
    ),
}

SKIPPED = {}
