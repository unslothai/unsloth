# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""The labext lockfile stays in the format JupyterLab's bundled jlpm (Yarn 3) installs with --immutable.

The Studio images build the extension with `jlpm install --immutable`. A lockfile rewritten by
Yarn 4 (as an unpinned Dependabot update does) fails that with YN0028, so both image builds break.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LABEXT = REPO_ROOT / "docker" / "jupyter" / "unsloth_labext"


def test_the_lockfile_is_in_yarn_3_format():
    head = (LABEXT / "yarn.lock").read_text(encoding = "utf-8")[:500]
    version = re.search(r"^__metadata:\n  version: (\d+)\n  cacheKey: (\S+)$", head, re.M)
    assert version, "yarn.lock has no Yarn Berry __metadata block"
    assert version.groups() == ("6", "8"), (
        f"yarn.lock __metadata is version {version.group(1)} / cacheKey {version.group(2)}: "
        "regenerate it with `jlpm install` (Yarn 3), not Yarn 4, or `jlpm install --immutable` "
        "fails in the Docker image builds"
    )


def test_the_package_manager_pin_matches_jlpm():
    manifest = json.loads((LABEXT / "package.json").read_text(encoding = "utf-8"))
    # Dependabot updates the lockfile with this Yarn (via corepack); unpinned, it uses Yarn 4.
    assert re.fullmatch(r"yarn@3\.\d+\.\d+", manifest.get("packageManager", "")), manifest.get(
        "packageManager"
    )
