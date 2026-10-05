# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The unsloth_cli suite must not resolve the real studio home (#9586).

Never import ``conftest`` here: re-executing it allocates a second home mid-test.
"""

import os
from pathlib import Path


def test_the_resolved_constant_points_off_the_users_home():
    # The import-time constant, not the env var: a too-late redirect still gets the env right.
    from unsloth_cli.commands.studio import STUDIO_HOME

    resolved = Path(STUDIO_HOME).resolve()
    assert Path.home().resolve() in resolved.parents
    assert any(p.startswith("unsloth-cli-tests-home-") for p in resolved.parts), resolved


def test_the_isolation_does_not_present_as_a_custom_studio_home():
    from unsloth_cli.commands import studio as studio_mod

    assert studio_mod._STUDIO_HOME_IS_CUSTOM is False
    assert "UNSLOTH_STUDIO_HOME" not in os.environ
    assert "STUDIO_HOME" not in os.environ
