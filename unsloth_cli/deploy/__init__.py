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
    # Optional, provider-defined availability hint shown in the picker (e.g.
    # RunPod's High/Medium/Low band). None when the provider has no live stock
    # signal -- e.g. a fixed-capacity cloud like Modal.
    stock: Optional[str] = None


@dataclass(frozen = True)
class SshTarget:
    user: str
    host: str
    port: int


@dataclass(frozen = True)
class StagedModel:
    """A local model a provider has placed on its own storage, ready for the
    instance to read. Returned by Provider.stage_local_model so the command
    layer never touches provider storage internals (volumes, S3, datacenters)."""
    model_path: str                  # path Studio loads from inside the container
    storage_id: Optional[str]        # provider handle to delete when done, or None
    summary: str                     # one-line note for the deploy preview/hints
    placement: Optional[str] = None  # provider hint the instance must honor (opaque)
