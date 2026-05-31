# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Maps a --provider name to its client.

To add a cloud:
  1. Implement `Provider` (deploy/base.py) in a new ``<name>_client.py``.
  2. Register it below.
That's the whole surface -- the command layer (commands/deploy.py) talks only to
the `Provider` contract, so no command code changes.
"""

from __future__ import annotations

from unsloth_cli.deploy import DeployError
from unsloth_cli.deploy.base import Provider
from unsloth_cli.deploy.runpod_client import RunPod


PROVIDERS: dict[str, type[Provider]] = {"runpod": RunPod}


def get_provider(name: str) -> Provider:
    if name not in PROVIDERS:
        raise DeployError(f"Unknown provider '{name}'. Available: {', '.join(PROVIDERS)}.")
    return PROVIDERS[name]()
