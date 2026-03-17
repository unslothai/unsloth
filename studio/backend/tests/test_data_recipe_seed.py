# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path


def test_seed_inspect_load_kwargs_disables_remote_code_execution():
    seed_route = (
        Path(__file__).resolve().parent.parent / "routes" / "data_recipe" / "seed.py"
    ).read_text()

    assert '"trust_remote_code": False' in seed_route
