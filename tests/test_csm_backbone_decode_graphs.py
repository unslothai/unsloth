# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""CSM generate(): the backbone steps one audio frame at a time with input_ids of shape (batch, 1, codebooks). The
growable KV cache must size the call from dim 1 (frames), not the last dim (codebooks), or its host counter is
re-synced to the codebook count and every step "fills its bucket", so the backbone never replays a graph."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_csm_depth_decode_graphs import ENV, GG, _gen, csm, host  # noqa: E402,F401


def test_backbone_frames_replay(host, csm, monkeypatch):
    # The backbone steps one frame (batch, 1, codebooks) at a time: the growable cache must count frames, not
    # the codebooks of input_ids' last dim, or every step "fills its bucket" and decodes eagerly.
    monkeypatch.setenv(ENV, "0")
    eager = _gen(csm, frames = 8)
    monkeypatch.delenv(ENV)
    graphed = _gen(csm, frames = 8)
    stats = GG.stats(csm) or {"captures": 0, "replays": 0, "refusals": {}}
    assert torch.equal(graphed, eager)
    assert stats["replays"] > 0 and "kv cache fills its bucket" not in stats["refusals"], stats
