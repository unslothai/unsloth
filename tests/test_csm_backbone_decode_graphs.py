# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""CSM backbone: the growable KV cache must count frames (dim 1), not codebooks, or no step replays a graph."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_csm_depth_decode_graphs import ENV, GG, _gen, csm, host  # noqa: E402,F401


def test_backbone_frames_replay(host, csm, monkeypatch):
    monkeypatch.setenv(ENV, "0")
    eager = _gen(csm, frames = 8)
    monkeypatch.delenv(ENV)
    graphed = _gen(csm, frames = 8)
    stats = GG.stats(csm) or {"captures": 0, "replays": 0, "refusals": {}}
    assert torch.equal(graphed, eager)
    assert stats["replays"] > 0 and "kv cache fills its bucket" not in stats["refusals"], stats
