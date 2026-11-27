# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The cast cache reader copies device tensors to the host on a side stream of the tensor's own card.

A layer split puts blocks of one component on two cards. The reader kept one side stream, made on the first tensor's
card, so a tensor on cuda:1 was copied under a cuda:0 stream: the copy then ran on cuda:1's default stream, behind the
render the side stream exists to stay out of. CPU only: a fake two-card ``torch.cuda``."""

from __future__ import annotations

import contextlib

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_cast_cache as CC  # noqa: E402


class _Stream:
    def __init__(self, device = None):
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

    def synchronize(self):
        pass


class _DeviceTensor:
    def __init__(self, index: int, log: list, current: list):
        self.device = torch.device("cuda", index)
        self.log = log
        self.current = current

    def to(
        self,
        where,
        non_blocking = False,
    ):
        stream = self.current[-1] if self.current else None
        self.log.append((self.device.index, None if stream is None else stream.device.index))
        return torch.zeros(1)


def test_each_card_gets_its_own_side_stream(monkeypatch):
    log: list = []
    current: list = []
    made: list = []

    def make_stream(device = None):
        s = _Stream(device)
        made.append(s)
        return s

    @contextlib.contextmanager
    def use(stream):
        current.append(stream)
        try:
            yield
        finally:
            current.pop()

    monkeypatch.setattr(torch.cuda, "Stream", make_stream)
    monkeypatch.setattr(torch.cuda, "stream", use)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device = None: None)
    tensors = {"a": _DeviceTensor(0, log, current), "b": _DeviceTensor(1, log, current)}
    monkeypatch.setattr(CC, "_resolve_ref", lambda module, name: tensors[name])
    monkeypatch.setattr(CC, "_offload_groups", lambda module: [])
    monkeypatch.setattr(CC, "_hooked", lambda module: False)
    reader = CC._Reader(object(), lambda: None)
    monkeypatch.setattr(reader, "_kept_copy", lambda ref, name: None)
    for name in ("a", "b", "a", "b"):
        assert reader.tensor(name).device.type == "cpu"
    # every copy ran under a side stream on the tensor's own card, and one stream was made per card
    assert log == [(0, 0), (1, 1), (0, 0), (1, 1)]
    assert sorted(s.device.index for s in made) == [0, 1]
