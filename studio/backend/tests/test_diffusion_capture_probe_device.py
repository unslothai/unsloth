# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cold probes ask whether the operand's card is capturing: torch.cuda.is_current_stream_capturing() reads the
current device's stream, so with card 0 current and a capture on the operand's card 1 the timing / equality probe
must still stay out of that capture. CPU: the CUDA device API is faked."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_flux3_qk_rope as qr  # noqa: E402
from core.inference import video_minimax_h3_qknorm as qk  # noqa: E402


class _Operand:
    def __init__(self, index):
        self.shape = (1, 4, 2, 64)
        self.dtype = torch.bfloat16
        self.device = torch.device("cuda", index)


@pytest.fixture
def card_1_capturing(monkeypatch):
    state = {"current": 0}

    class _Device:
        def __init__(self, device):
            self.index = device.index if hasattr(device, "index") else int(device)

        def __enter__(self):
            self.prev, state["current"] = state["current"], self.index

        def __exit__(self, *exc):
            state["current"] = self.prev

    monkeypatch.setattr(torch.cuda, "device", _Device)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: state["current"])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: state["current"] == 1)
    return state


def test_flux3_qk_rope_probe_stays_out_of_the_operands_capture(monkeypatch, card_1_capturing):
    checks = []
    monkeypatch.setattr(qr, "_VERDICTS", {})
    monkeypatch.setattr(
        qr, "_check", lambda *a: checks.append(a) or {"exact": True, "faster": True, "ms": {}}
    )
    weight = torch.ones(64, dtype = torch.bfloat16)
    assert qr.verdict(_Operand(1), weight, 1e-6) is None
    assert checks == [] and qr._VERDICTS == {} and card_1_capturing["current"] == 0


def test_h3_qknorm_probe_stays_out_of_the_operands_capture(monkeypatch, card_1_capturing):
    checks = []
    monkeypatch.setattr(qk, "_VERDICTS", {})
    monkeypatch.setattr(qk, "_check", lambda *a: checks.append(a) or {"exact": True, "use": True})
    assert qk.verdict(_Operand(1), 64) is None
    assert checks == [] and qk._VERDICTS == {} and card_1_capturing["current"] == 0
