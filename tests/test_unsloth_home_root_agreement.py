# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""UNSLOTH_HOME has to mean the same install to the CLI and to the backend: if
the two disagree the backend writes studio.db, auth and the pid file under
<UNSLOTH_HOME>/studio while the CLI reads ~/.unsloth/studio.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_PROBE = """
import json, sys
sys.path.insert(0, {backend!r})
from unsloth_cli.commands.studio import STUDIO_HOME, _STUDIO_HOME_IS_CUSTOM
from utils.paths.storage_roots import studio_root
print(json.dumps({{
    "cli": str(STUDIO_HOME),
    "cli_is_custom": bool(_STUDIO_HOME_IS_CUSTOM),
    "backend": str(studio_root()),
}}))
"""


def _probe(env_overrides: dict) -> dict:
    env = dict(os.environ)
    for key in ("UNSLOTH_HOME", "UNSLOTH_STUDIO_HOME", "STUDIO_HOME"):
        env.pop(key, None)
    env.update(env_overrides)
    env["PYTHONPATH"] = str(REPO_ROOT)
    source = _PROBE.format(backend = str(REPO_ROOT / "studio" / "backend"))
    out = subprocess.run(
        [sys.executable, "-c", source],
        capture_output = True,
        text = True,
        cwd = str(REPO_ROOT),
        env = env,
        check = True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_unsloth_home_resolves_to_one_studio_root(tmp_path):
    master = tmp_path / "portable"
    result = _probe({"UNSLOTH_HOME": str(master)})

    assert result["backend"] == str(master / "studio")
    assert result["cli"] == result["backend"]
    # Custom, so `unsloth studio ...` re-exports UNSLOTH_STUDIO_HOME.
    assert result["cli_is_custom"] is True


def test_studio_home_still_outranks_unsloth_home(tmp_path):
    explicit = tmp_path / "explicit"
    result = _probe(
        {"UNSLOTH_HOME": str(tmp_path / "portable"), "UNSLOTH_STUDIO_HOME": str(explicit)}
    )

    assert result["backend"] == str(explicit)
    assert result["cli"] == str(explicit)


def test_no_unsloth_home_keeps_the_legacy_default(tmp_path):
    result = _probe({"HOME": str(tmp_path / "home"), "USERPROFILE": str(tmp_path / "home")})

    legacy = str(tmp_path / "home" / ".unsloth" / "studio")
    assert result["cli"] == legacy
    assert result["backend"] == legacy
    assert result["cli_is_custom"] is False
