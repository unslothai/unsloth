# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One nvidia-smi reading per probe_devices call, same probes as per-helper reads (CPU, fake smi)."""

from __future__ import annotations

import itertools

import pytest
import torch

from core.inference import diffusion_device as dd
from core.inference import diffusion_multigpu as mg


def _rows(cards):
    """cards: {physical index: (total MiB, used MiB, util %)}"""
    return {
        "devices": [
            {
                "index": i,
                "vram_total_gb": t / 1024,
                "vram_used_gb": u / 1024,
                "gpu_utilization_pct": util,
            }
            for i, (t, u, util) in sorted(cards.items())
        ]
    }


class _Smi:
    """Fake ``_smi_visible_utilization``: answers ``readings`` in turn (the last one repeats), counting the runs."""

    def __init__(self, *readings):
        self.readings = list(readings)
        self.calls = 0

    def __call__(self, spec):
        self.calls += 1
        reading = self.readings[min(self.calls, len(self.readings)) - 1]
        return reading() if callable(reading) else reading


def _host(
    monkeypatch,
    smi,
    visible = (0, 1, 2),
):
    import utils.hardware.hardware as hw

    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw, "_cuda_order_matches_smi", lambda: True)
    monkeypatch.setattr(hw, "_cuda_instance_ids_active", lambda: False)
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: list(visible))
    monkeypatch.setattr(hw, "_integrated_cuda_rows", lambda v: ({}, ""))
    monkeypatch.setattr(hw, "_nvidia_mig_gpu_indices_or_none", lambda: set())
    monkeypatch.setattr(hw, "_get_parent_visible_gpu_spec", lambda: {"numeric_ids": list(visible)})
    monkeypatch.setattr(hw, "_smi_visible_utilization", smi)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda o = 0: "Tesla T4")
    from utils import hardware as hw_pkg

    monkeypatch.setattr(
        hw_pkg, "trusted_mem_get_info", lambda o, module = None: (15000 << 20, 15360 << 20)
    )


def _per_helper(ordinals, primary):
    """What probe_devices reported before: each secondary's memory and utilisation from their own reads."""
    out = {}
    for o in ordinals:
        if o == primary:
            out[o] = (15000, 15360, None)
        else:
            free, total = dd.smi_free_total_mib(o)
            out[o] = (free, total, dd.smi_util_pct(o))
    return out


def _as_tuples(probes):
    return {o: (p.free_mib, p.total_mib, p.util_pct) for o, p in probes.items()}


_FREE_USED = (300, 9000, 15360)  # idle, half used, full
_UTIL = (0, 3, 90, None)


@pytest.mark.parametrize(
    "used1,used2,util1,util2",
    list(itertools.product(_FREE_USED, _FREE_USED, _UTIL, _UTIL)),
)
def test_probe_matches_the_per_helper_reads_with_one_run(monkeypatch, used1, used2, util1, util2):
    rows = _rows({0: (15360, 300, 0), 1: (15360, used1, util1), 2: (15360, used2, util2)})
    smi = _Smi(rows)
    _host(monkeypatch, smi)
    selection = mg.MultiGpuSelection(primary = 0, secondaries = (1, 2), automatic = True)
    with mg.selection_scope(selection):
        expected = _per_helper((0, 1, 2), 0)
        assert smi.calls == 4  # two per secondary, outside a probe
        smi.calls = 0
        probes = mg.probe_devices((0, 1, 2))
    assert smi.calls == 1
    assert _as_tuples(probes) == expected
    # every card is the faked model, so the partner decisions below compare like with like
    assert {p.name for p in probes.values()} == {"Tesla T4"}
    # the decisions built from the probes
    with mg.selection_scope(selection):
        assert mg.best_secondary(selection, probes) == mg.best_secondary(
            selection,
            {o: mg.DeviceProbe(o, "Tesla T4", t, f, u) for o, (f, t, u) in expected.items()},
        )
    primary = probes[0]
    cards = [probes[o] for o in (1, 2) if probes[o].free_mib is not None]
    assert mg.automatic_partner(primary, cards, 3900) == mg.automatic_partner(
        mg.DeviceProbe(0, "Tesla T4", *expected[0][1::-1], None),
        [
            mg.DeviceProbe(o, "Tesla T4", expected[o][1], expected[o][0], expected[o][2])
            for o in (1, 2)
            if expected[o][0] is not None
        ],
        3900,
    )


def test_each_probe_reads_afresh(monkeypatch):
    first = _rows({0: (15360, 300, 0), 1: (15360, 300, 0), 2: (15360, 300, 0)})
    # card 2 vanishes and card 1 gets busy between two probes
    second = _rows({0: (15360, 300, 0), 1: (15360, 9000, 80)})
    smi = _Smi(first, second)
    _host(monkeypatch, smi)
    selection = mg.MultiGpuSelection(primary = 0, secondaries = (1, 2), automatic = True)
    with mg.selection_scope(selection):
        a = mg.probe_devices((0, 1, 2))
        b = mg.probe_devices((0, 1, 2))
    assert smi.calls == 2
    assert _as_tuples(a)[2] == (15060, 15360, 0.0)
    assert _as_tuples(b)[1] == (15360 - 9000, 15360, 80.0)
    assert _as_tuples(b)[2] == (None, None, None)


def test_a_failed_read_is_not_shared(monkeypatch):
    rows = _rows({0: (15360, 300, 0), 1: (15360, 300, 5)})
    smi = _Smi(None, rows)
    _host(monkeypatch, smi, visible = (0, 1))
    selection = mg.MultiGpuSelection(primary = 0, secondaries = (1,), automatic = True)
    with mg.selection_scope(selection):
        probes = mg.probe_devices((0, 1))
    # memory unread (the run failed), utilisation from the retry: the same as two independent reads
    assert smi.calls == 2
    assert _as_tuples(probes)[1] == (None, None, 5.0)


def test_nothing_is_kept_after_the_probe(monkeypatch):
    smi = _Smi(_rows({0: (15360, 300, 0), 1: (15360, 300, 0)}))
    _host(monkeypatch, smi, visible = (0, 1))
    selection = mg.MultiGpuSelection(primary = 0, secondaries = (1,), automatic = True)
    with mg.selection_scope(selection):
        mg.probe_devices((0, 1))
        assert smi.calls == 1
        dd.smi_free_total_mib(1)
        dd.smi_util_pct(1)
    assert smi.calls == 3
    assert dd._SMI_ROWS.get() is None


def test_explicit_selection_reads_no_smi(monkeypatch):
    smi = _Smi(_rows({0: (15360, 300, 0), 1: (15360, 300, 0)}))
    _host(monkeypatch, smi, visible = (0, 1))
    with mg.selection_scope(mg.MultiGpuSelection(primary = 0, secondaries = (1,))):
        probes = mg.probe_devices((0, 1))
    assert smi.calls == 0
    assert probes[1].util_pct is None
