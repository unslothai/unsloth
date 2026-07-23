# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Run the TypeScript vision-capability contract from the collected pytest suite."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def test_model_vision_capability_node_contract():
    subprocess.run(
        [
            "node",
            "--test",
            str(ROOT / "tests/studio/model_vision_capability.test.mjs"),
        ],
        cwd = ROOT,
        check = True,
    )
