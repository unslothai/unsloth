# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Maps a --provider name to its client. Add a provider by registering it here."""

from __future__ import annotations

from unsloth_cli.deploy import DeployError
from unsloth_cli.deploy.runpod_client import RunPod


PROVIDERS = {"runpod": RunPod}


def get_provider(name: str):
    if name not in PROVIDERS:
        raise DeployError(f"Unknown provider '{name}'. Available: {', '.join(PROVIDERS)}.")
    return PROVIDERS[name]()
