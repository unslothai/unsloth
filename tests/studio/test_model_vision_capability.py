# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Run the TypeScript vision-capability contract from the collected pytest suite.

The warning decision lives in ``studio/frontend`` which carries no JS test runner, so the
real module is executed under node's type stripping instead. On a tree whose chat warning
still reads only the catalog row (the pre-fix behaviour) the module is absent and this
fails, which is what keeps the fix attached to a red baseline.
"""

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_model_vision_capability_node_contract():
    if shutil.which("node") is None:
        pytest.skip("node not available")
    probe = subprocess.run(
        ["node", "--experimental-strip-types", "--version"],
        capture_output = True,
        text = True,
        timeout = 5,
    )
    if probe.returncode != 0:
        pytest.skip("node --experimental-strip-types not available")
    subprocess.run(
        [
            "node",
            "--experimental-strip-types",
            "--test",
            str(ROOT / "tests/studio/model_vision_capability.test.mjs"),
        ],
        cwd = ROOT,
        check = True,
        timeout = 30,
    )
