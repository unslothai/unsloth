# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_PROVIDERS = REPO_ROOT / "studio/frontend/src/features/chat/external-providers.ts"


def test_minimax_vision_capability_matches_the_selected_model():
    if shutil.which("node") is None:
        pytest.skip("node not available")

    script = f"""
import {{ providerTypeSupportsVision }} from {json.dumps(EXTERNAL_PROVIDERS.as_uri())};
console.log(JSON.stringify({{
  m3: providerTypeSupportsVision("minimax", "MiniMax-M3"),
  m27: providerTypeSupportsVision("minimax", "MiniMax-M2.7"),
}}));
"""
    result = subprocess.run(
        [
            "node",
            "--experimental-strip-types",
            "--input-type=module",
            "--eval",
            script,
        ],
        capture_output = True,
        text = True,
        timeout = 30,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"m3": True, "m27": False}
