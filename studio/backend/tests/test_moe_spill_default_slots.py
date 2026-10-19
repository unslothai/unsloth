# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Default serving slots on a spilled MoE with a drafter (``_MOE_SPILL_DEFAULT_SLOTS``).

Once the fit spills experts to host RAM, every slot past one carries KV, recurrent state,
draft reserve and compute buffer that llama.cpp's fitter pays for with experts, and
drafting slots verify multi-token batches whose expert misses multiply. Measured on an
RTX PRO 6000 held to 16 GB (Qwen3.6-35B-A3B UD-Q4_K_XL, MTP forced): four slots kept
1.1 GB less MoE cache than one, and four concurrent chats ran 158 tok/s in aggregate
against 278 at two slots; Flash-Next with the cache at 24 GB ran 59 vs 85. Without a
drafter four slots won four chats (316 vs 273), and deep in a --fit split (Flash-Next,
under a third of the experts resident, no cache) they still did, so only a drafting load
near the fit edge or over the MoE cache is cut. A chosen count, a resident MoE and a
dense model keep theirs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_llama_cpp_placement as placement  # noqa: E402
from test_llama_cpp_placement import (  # noqa: E402, F401  (fixtures)
    _CACHE_CAPS_ON,
    _cache_launch,
    _discrete_linux_host,
    _moe_cache_host,
    _ubatch_values,
    _write_gguf,
    llama_cpp_module,
)

_CAPS = dict(
    _CACHE_CAPS_ON,
    supports_kv_unified = True,
    mtp_token = "draft-mtp",
    spec_draft_n_max_flag = "--spec-draft-n-max",
)


def _flag(cmd, flag):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


_GIB = 1024**3
# ~53% of the 15 GiB of experts resident (the fixture default) vs ~25% (deep spill).
_DEEP_SPILL_CARD = [(0, 11_000, 32_000)]


@pytest.fixture(autouse = True)
def _sized_drafter(monkeypatch):
    """A drafter the fit prices by bytes per slot, as a real GGUF's dims allow; a flat
    cushion leaves the resident share unpriceable, and an unpriced share keeps the ask."""
    monkeypatch.delenv("UNSLOTH_MOE_SPILL_SLOTS", raising = False)
    # The fixture drafter is a bare header; the standalone check is not under test.
    monkeypatch.setattr(llama_cpp_module, "_mtp_drafter_loads_standalone", lambda *a, **k: True)
    make = placement._moe_backend

    def sized(*args, **kwargs):
        backend, gguf = make(*args, **kwargs)
        backend._estimate_mtp_overhead_bytes = lambda _ctx, *a, n_parallel = None, **k: int(
            0.25 * _GIB * int(n_parallel or 1)
        )
        backend._mtp_draft_kv_bytes = lambda _ctx, *a, n_parallel = None, **k: int(
            0.1 * _GIB * int(n_parallel or 1)
        )
        return backend, gguf

    monkeypatch.setattr(placement, "_moe_backend", sized)


def _load(
    tmp_path,
    *,
    n_parallel = 4,
    from_default = True,
    drafter = True,
    speculative_type = "mtp",
    **kw,
):
    if drafter:
        kw["mtp_draft_path"] = str(_write_gguf(tmp_path / "mtp-draft.gguf"))
    return _cache_launch(
        tmp_path,
        caps = _CAPS,
        n_parallel = n_parallel,
        n_parallel_from_default = from_default,
        speculative_type = speculative_type,
        **kw,
    )


@pytest.fixture
def _hybrid_head(monkeypatch):
    """The MoE fixture read as a Qwen3.6-style hybrid with an embedded MTP head."""
    make = placement._moe_backend

    def hybrid(*args, **kwargs):
        backend, gguf = make(*args, **kwargs)

        def read_metadata(_path):
            backend._nextn_predict_layers = 1
            backend._full_attention_interval = 4
            backend._ssm_inner_size = 4096
            backend._ssm_state_size = 128
            backend._ssm_group_count = 16
            backend._ssm_conv_kernel = 4

        backend._read_gguf_metadata = read_metadata
        return backend, gguf

    monkeypatch.setattr(placement, "_moe_backend", hybrid)


@pytest.mark.parametrize("speculative_type", ["mtp", "auto"])
def test_a_spilled_moe_with_a_drafter_launches_fewer_default_slots(
    tmp_path, _moe_cache_host, speculative_type
):
    backend, cmd = _load(tmp_path, speculative_type = speculative_type)

    assert _flag(cmd, "--fit") == "on", cmd
    assert _flag(cmd, "--spec-type") == "draft-mtp", cmd
    assert _flag(cmd, "--parallel") == "2", cmd
    assert getattr(llama_cpp_module, "_MOE_SPILL_DEFAULT_SLOTS", None) == 2
    # Two slots still share one context window.
    assert "--kv-unified" in cmd, cmd
    # The micro-batch raise is still taken, priced at the cut count.
    assert _ubatch_values(cmd) == ["2048"], cmd
    # The ask is kept, so the UI and a reload compare against what the user left set.
    assert backend.requested_parallel_slots == 4
    assert backend.effective_parallel_slots == 2


def test_a_forced_embedded_hybrid_head_is_cut_too(tmp_path, _moe_cache_host, _hybrid_head):
    _backend, cmd = _load(tmp_path, drafter = False, speculative_type = "mtp")

    assert _flag(cmd, "--spec-type") == "draft-mtp", cmd
    assert _flag(cmd, "--parallel") == "2", cmd


def test_auto_standing_an_embedded_hybrid_head_down_keeps_the_slots(
    tmp_path, _moe_cache_host, _hybrid_head
):
    """Auto runs no drafter under this spill (embedded_mtp_partial_offload)."""
    backend, cmd = _load(tmp_path, drafter = False, speculative_type = "auto")

    assert _flag(cmd, "--spec-type") == "none", cmd
    assert backend.spec_fallback_reason == "mtp_partial_offload"
    assert _flag(cmd, "--parallel") == "4", cmd


def test_a_deep_split_spill_keeps_the_default_slots(tmp_path, _moe_cache_host, monkeypatch):
    """Most experts already run from host RAM under the --fit split: the slots move
    little and four chats gain."""
    monkeypatch.setenv("UNSLOTH_MOE_CACHE", "off")
    _backend, cmd = _load(tmp_path, memory = _DEEP_SPILL_CARD)

    assert _flag(cmd, "--fit") == "on", cmd
    assert "--moe-cache-mib" not in cmd, cmd
    assert _flag(cmd, "--spec-type") == "draft-mtp", cmd
    assert _flag(cmd, "--parallel") == "4", cmd


def test_a_deep_spill_the_cache_serves_is_cut(tmp_path, _moe_cache_host, monkeypatch):
    """Four drafting slots thrash the MoE cache at any resident share."""
    monkeypatch.setenv("UNSLOTH_MOE_CACHE", "on")
    _backend, cmd = _load(tmp_path, memory = _DEEP_SPILL_CARD)

    assert "--moe-cache-mib" in cmd, cmd
    assert _flag(cmd, "--parallel") == "2", cmd


def test_the_switch_on_cuts_a_deep_spill_too(tmp_path, _moe_cache_host, monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_CACHE", "off")
    monkeypatch.setenv("UNSLOTH_MOE_SPILL_SLOTS", "on")
    _backend, cmd = _load(tmp_path, memory = _DEEP_SPILL_CARD)

    assert _flag(cmd, "--parallel") == "2", cmd


def test_a_spilled_moe_without_a_drafter_keeps_the_default_slots(tmp_path, _moe_cache_host):
    _backend, cmd = _load(tmp_path, drafter = False, speculative_type = "auto")

    assert _flag(cmd, "--fit") == "on", cmd
    assert _flag(cmd, "--spec-type") != "draft-mtp", cmd
    assert _flag(cmd, "--parallel") == "4", cmd


def test_a_chosen_slot_count_is_kept_on_a_spilled_moe(tmp_path, _moe_cache_host):
    backend, cmd = _load(tmp_path, from_default = False)

    assert _flag(cmd, "--fit") == "on", cmd
    assert _flag(cmd, "--parallel") == "4", cmd
    assert backend.effective_parallel_slots == 4


@pytest.mark.parametrize("n_parallel", [1, 2])
def test_a_default_at_or_below_the_cap_is_unchanged(tmp_path, _moe_cache_host, n_parallel):
    _backend, cmd = _load(tmp_path, n_parallel = n_parallel)

    assert _flag(cmd, "--parallel") == str(n_parallel), cmd


def test_a_resident_moe_keeps_the_default_slots(tmp_path, _moe_cache_host):
    _backend, cmd = _load(tmp_path, spilled = False)

    assert _flag(cmd, "--fit") == "off", cmd
    assert _flag(cmd, "--parallel") == "4", cmd


def test_a_spilled_dense_model_keeps_the_default_slots(tmp_path, _moe_cache_host):
    _backend, cmd = _load(tmp_path, moe = False)

    assert _flag(cmd, "--fit") == "on", cmd
    assert _flag(cmd, "--parallel") == "4", cmd


def test_a_user_expert_offload_on_a_fitting_moe_keeps_the_slots(tmp_path, _moe_cache_host):
    """-ncmoe on a model that fits is the user's placement, not a VRAM shortfall."""
    _backend, cmd = _load(tmp_path, spilled = False, extra_args = ["-ncmoe", "12"])

    assert _ubatch_values(cmd) == ["2048"], cmd
    assert _flag(cmd, "--parallel") == "4", cmd


def test_the_switch_restores_the_default_slots(tmp_path, _moe_cache_host, monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_SPILL_SLOTS", "off")
    backend, cmd = _load(tmp_path)

    assert _flag(cmd, "--spec-type") == "draft-mtp", cmd
    assert _flag(cmd, "--parallel") == "4", cmd
    assert backend.effective_parallel_slots == 4
