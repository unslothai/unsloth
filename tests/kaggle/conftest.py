# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""No test in this directory may see a real Kaggle credential.

Found the hard way. `gate.py` now considers two accounts, and a developer box
with `KAGGLE_API_TOKEN_2` exported turned the "a fork gets no secret" test into a
LIVE run: the gate authenticated to a real account, read that week's actual
remaining hours over the network, and answered `should_run=true`. The test failed
for the right-looking reason and would have PASSED had the quota been low, which
is the worse version -- a unit test whose verdict depends on somebody's usage.

Every credential env var is therefore cleared before each test, and a test that
wants one sets it explicitly afterwards. Autouse, because the hazard is in the
tests that never think about credentials at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CI_DIR = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "kaggle_t4_ci"
sys.path.insert(0, str(CI_DIR))

# Read from the gate rather than restated, so an account added there is covered
# here without anyone remembering to.
try:
    from gate import DEFAULT_ACCOUNT_ENVS
except Exception:  # noqa: BLE001 - the suite has its own import guards
    DEFAULT_ACCOUNT_ENVS = ("KAGGLE_API_TOKEN", "KAGGLE_API_TOKEN_2")

# The client reads these too, and a stray one authenticates just as well as the
# token does.
_OTHER_CREDENTIAL_ENVS = ("KAGGLE_KEY", "KAGGLE_USERNAME", "KAGGLE_ACCESS_TOKEN_GH")


@pytest.fixture(autouse = True)
def _no_ambient_kaggle_credentials(monkeypatch):
    for name in (*DEFAULT_ACCOUNT_ENVS, *_OTHER_CREDENTIAL_ENVS):
        monkeypatch.delenv(name, raising = False)
