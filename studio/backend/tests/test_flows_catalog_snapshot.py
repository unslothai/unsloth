# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from pathlib import Path

from core.flows.specs import catalog

SNAPSHOT = (
    Path(__file__).resolve().parents[2] / "frontend" / "tests" / "fixtures" / "flows-catalog.json"
)


def test_frontend_catalog_snapshot_matches_the_registry():
    assert (
        json.loads(SNAPSHOT.read_text()) == catalog()
    ), "refresh it: cd studio/backend && python -m core.flows.specs > ../frontend/tests/fixtures/flows-catalog.json"
