# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class DeployError(Exception):
    pass


@dataclass(frozen = True)
class Gpu:
    id: str
    name: str
    vram_gb: int
    cost_per_hour_usd: float
    stock: Optional[str] = None


@dataclass(frozen = True)
class SshTarget:
    user: str
    host: str
    port: int


@dataclass(frozen = True)
class StagedModel:
    model_path: str
    storage_id: Optional[str]
    summary: str
    placement: Optional[str] = None
