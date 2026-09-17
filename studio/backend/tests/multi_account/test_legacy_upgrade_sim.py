# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import os
import subprocess
import sys
from pathlib import Path


def test_populated_legacy_install_survives_app_import_and_owner_reads(tmp_path):
    backend = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env.update(
        UNSLOTH_STUDIO_HOME = str(tmp_path / "legacy-install"),
        UNSLOTH_STUDIO_DOCUMENTS_HOME = str(tmp_path / "Documents"),
        PYTHONPATH = str(backend),
        PYTHONDONTWRITEBYTECODE = "1",
    )
    result = subprocess.run(
        [sys.executable, "-m", "tests.multi_account.legacy_probe"],
        cwd = backend,
        env = env,
        capture_output = True,
        text = True,
        timeout = 120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {"preserved_files": 7, "owner_login": True}
