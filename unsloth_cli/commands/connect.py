# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deprecated shim for the renamed `unsloth_cli.commands.start` module.

Keeps `import unsloth_cli.commands.connect` (and `connect_app`) working for
existing callers; new code should import from `unsloth_cli.commands.start`.
"""

from unsloth_cli.commands.start import *  # noqa: F401, F403
from unsloth_cli.commands.start import start_app as connect_app  # noqa: F401
