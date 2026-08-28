# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for issue #9586 channel 3: tests writing the user's real studio home.

Two things make this channel distinct, and there is a test for each.

The BINDING: ``STUDIO_HOME`` resolves at import, so the usual remedy -- an autouse fixture
-- runs too late, and a test that only read the environment would pass while the constant
every write goes through still pointed at the user's home.

The LEVER: ``UNSLOTH_STUDIO_HOME`` is the obvious way to redirect it and the wrong one,
because the production code reports whether it was set. Isolating with it changes output
this root asserts on elsewhere.

Nothing here imports ``conftest``. Importing it under its package path re-executes the
module, which allocates a SECOND temporary home and repoints ``HOME`` mid-test.
"""

from __future__ import annotations

import os
from pathlib import Path


_ISOLATION_PREFIX = "unsloth-cli-tests-home-"
_CONFTEST = Path(__file__).resolve().parent / "conftest.py"


def test_the_resolved_constant_points_off_the_users_home():
    """The pin that matters: the constant, not the environment variable.

    ``unsloth_cli.commands.studio.STUDIO_HOME`` is bound by ``_resolve_studio_home()`` at
    import. Applied in a fixture or in ``pytest_configure`` the redirect would be too late
    and this would still resolve under the real user's home.
    """
    from unsloth_cli.commands.studio import STUDIO_HOME

    resolved = Path(STUDIO_HOME).resolve()
    assert Path.home().resolve() in resolved.parents
    assert any(part.startswith(_ISOLATION_PREFIX) for part in resolved.parts), resolved


def test_the_isolation_does_not_present_as_a_custom_studio_home():
    """Why HOME is redirected and ``UNSLOTH_STUDIO_HOME`` is not.

    ``_resolve_studio_home()`` reports whether the home was explicitly set, and
    ``_fail_if_install_damaged()`` puts ``UNSLOTH_STUDIO_HOME=...`` into the repair command
    it prints when it was. Isolating through that variable changes output other tests in
    this root assert on -- measured, it breaks
    ``test_a_no_torch_install_keeps_that_mode_in_the_reinstall``. The variable is an input
    to the behaviour under test, so it cannot also be the isolation mechanism.
    """
    from unsloth_cli.commands import studio as studio_mod

    assert studio_mod._STUDIO_HOME_IS_CUSTOM is False
    assert "UNSLOTH_STUDIO_HOME" not in os.environ


def test_the_redirect_is_applied_at_module_scope():
    """Pins the placement, which is the whole mechanism.

    A fixture cannot be substituted: the constants above are already bound by the time one
    runs. Read from disk rather than via ``inspect``, because importing the module to
    inspect it would re-run the redirect.
    """
    source = _CONFTEST.read_text(encoding = "utf-8")

    redirect = source.index('_os.environ["HOME"]')
    assert redirect < source.index("\ndef "), "must precede every function in the module"
    assert redirect < source.index("@pytest.fixture"), "must not be inside or after a fixture"
