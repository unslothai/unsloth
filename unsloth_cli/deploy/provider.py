# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


from __future__ import annotations

from unsloth_cli.deploy import DeployError
from unsloth_cli.deploy.base import Provider
from unsloth_cli.deploy.modal_client import Modal
from unsloth_cli.deploy.runpod_client import RunPod


PROVIDERS: dict[str, type[Provider]] = {"runpod": RunPod, "modal": Modal}


def get_provider(name: str) -> Provider:
    if name not in PROVIDERS:
        raise DeployError(
            f"Unknown provider '{name}'. Available: {', '.join(PROVIDERS)}."
        )
    return PROVIDERS[name]()
