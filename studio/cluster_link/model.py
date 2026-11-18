# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The records shared by pairing, the agent, the relay and cluster serving."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen = True)
class LocalIdentity:
    """This node's own certificate, key and the cluster root it trusts."""

    node_id: str
    role: str  # "master" | "worker"
    cluster_id: str
    certfile: str
    keyfile: str
    rootfile: str
    fingerprint: str


@dataclass
class Peer:
    """A paired node as the OTHER side recorded it. Trust is the exact fingerprint, never an address."""

    node_id: str
    name: str
    role: str
    fingerprint: str
    cert_pem: str
    addrs: List[str] = field(default_factory = list)  # last verified, most recent first
    agent_port: int = 0
    added_at: float = 0.0
    removed: bool = False  # a tombstone: kept so a removed worker cannot quietly come back
