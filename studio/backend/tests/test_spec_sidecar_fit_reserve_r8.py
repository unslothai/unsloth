# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Eighth-round cells for the DSpark/DFlash sidecar reserve.

The reserve composes with the VRAM budget's per-card margins: each card's entry is its own
budget margin plus its share, a --split-mode none launch keeps the other cards' margins, an
--rpc launch (remote devices first in the child's list) gets no positional list, and the
text-only retry does not rewrite the reserve-bearing pair.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_spec_sidecar_fit_reserve_r2 as _r2  # noqa: E402
import test_spec_sidecar_fit_reserve_r7 as _r7  # noqa: E402
from core.inference import llama_cpp as _lc  # noqa: E402

_MIXED = ((0, 49_140, 49_140), (1, 24_576, 24_576))


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    for k in ("LLAMA_ARG_DEVICE", "LLAMA_ARG_RPC", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(k, raising = False)


@pytest.fixture
def _budget(monkeypatch):
    """A 30% VRAM budget on a host whose device order is provable, so each card gets its
    own --fit-target margin."""
    monkeypatch.setattr(_lc, "_metal_capable_host", lambda: False)
    B = _lc.LlamaCppBackend
    monkeypatch.setattr(B, "_apple_metal_memory_budget_bytes", staticmethod(lambda: 0))
    monkeypatch.setattr(_lc, "_active_vram_fraction", lambda: 0.3)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setattr(B, "_GPU_IDS_ARE_PCI_INDICES", True)
    monkeypatch.setattr(B, "_SMI_MIG_INDICES", frozenset())
    monkeypatch.setattr(B, "_visibility_mask_is_unmappable", staticmethod(lambda: False))
    monkeypatch.setattr(B, "_host_torch_is_rocm", staticmethod(lambda: False))


def _two_cards(tmp_path):
    backend, gguf = _r2._cuda_backend(tmp_path, memory = _MIXED)
    backend._arch_gate_survivors = lambda binary = None: []
    return backend, gguf


def _base_margins(tmp_path, **kw):
    (tmp_path / "base").mkdir()
    backend, gguf = _two_cards(tmp_path / "base")
    return _r7._margins(_r2._placement._launch(backend, gguf, **kw)["cmd"])


def test_each_card_keeps_its_budget_margin_under_the_reserve(tmp_path, _budget):
    base = _base_margins(tmp_path)
    assert len(base) == 2 and base[0] != base[1]
    backend, gguf = _two_cards(tmp_path)
    reserved = _r7._margins(_r7._dspark(backend, gguf, tmp_path))
    assert len(reserved) == 2
    # Each entry sits on that card's own margin; the smaller card still has room.
    assert all(r > b for r, b in zip(reserved, base))
    assert reserved[1] < 24_576


def test_split_mode_none_keeps_the_other_cards_margins(tmp_path, _budget):
    extras = ["--split-mode", "none"]
    base = _base_margins(tmp_path, extra_args = extras)
    backend, gguf = _two_cards(tmp_path)
    reserved = _r7._margins(_r7._dspark(backend, gguf, tmp_path, extra_args = extras))
    assert len(reserved) == 2
    assert reserved[0] >= base[0] + 8 * 1024
    assert reserved[1] == base[1]


def test_an_rpc_launch_gets_no_positional_reserve_list(tmp_path, _budget):
    backend, gguf = _two_cards(tmp_path)
    cmd = _r7._dspark(backend, gguf, tmp_path, extra_args = ["--rpc", "10.0.0.2:50052"])
    assert len(_r7._margins(cmd)) == 1


def test_studios_per_card_margins_are_kept_under_the_reserve():
    flags = _lc._sidecar_reserve_margin_flags(
        ["--model-draft", "/d.gguf", "--fit", "on"],
        [],
        {},
        4000,
        5000,
        per_device = [5000, 500],
        base_by_device = [4000, 1000],
    )
    # The smaller card's lower margin is not raised to the scalar.
    assert flags == ["--fit-target", "9000,1500"]


def test_a_users_margin_still_wins_over_a_lower_per_card_base():
    flags = _lc._sidecar_reserve_margin_flags(
        ["--model-draft", "/d.gguf", "--fit", "on"],
        ["--fit-target", "2048"],
        {},
        1024,
        100,
        per_device = [100, 100],
        base_by_device = [1024, 4096],
    )
    assert flags == ["--fit-target", "2148,4196"]


def test_the_text_only_retry_leaves_the_reserve_pair_alone():
    source = "".join(inspect.getsource(_lc.LlamaCppBackend.load_model).split())
    at = source.index("cmd=self._strip_mmproj_args(_vision_gpu_cmd)")
    window = source[at : at + 400]
    assert "notin_sidecar_override_pairs" in window
    assert window.index("notin_sidecar_override_pairs") < window.index(
        "_per_device_fit_target_argv("
    )
