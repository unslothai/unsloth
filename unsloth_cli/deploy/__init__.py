# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared types for `unsloth deploy`."""

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
    stock: Optional[str] = None  # RunPod availability band: High/Medium/Low, or None


@dataclass(frozen = True)
class SshTarget:
    user: str
    host: str
    port: int
