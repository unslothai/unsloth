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


def test_a_chosen_count_does_not_dedupe_against_slots_cut_for_the_spill(tmp_path, _moe_cache_host):
    """A default four that launched two still matches; four chosen by hand reloads."""
    import dataclasses

    backend, cmd = _load(tmp_path)
    assert _flag(cmd, "--parallel") == "2", cmd

    default = backend.last_load_intent
    assert default.n_parallel_from_default
    chosen = dataclasses.replace(default, n_parallel_from_default = False)
    assert backend._runtime_matches_intent(default, None)
    assert not backend._runtime_matches_intent(chosen, None)


def test_a_chosen_count_still_dedupes_when_the_switch_kept_the_slots(
    tmp_path, _moe_cache_host, monkeypatch
):
    import dataclasses

    monkeypatch.setenv("UNSLOTH_MOE_SPILL_SLOTS", "off")
    backend, cmd = _load(tmp_path)
    assert _flag(cmd, "--parallel") == "4", cmd

    chosen = dataclasses.replace(backend.last_load_intent, n_parallel_from_default = False)
    assert backend._runtime_matches_intent(chosen, None)


@pytest.fixture
def _failing_drafter(monkeypatch):
    """Every launch carrying the drafter fails its health check; the rest come up."""
    cmds = []

    def launch(
        backend,
        gguf,
        model_identifier = "test",
        **load_kwargs,
    ):
        def fake_popen(cmd, **kwargs):
            if not cmd or str(cmd[0]) != "/fake/llama-server":
                return placement._REAL_POPEN(cmd, **kwargs)
            cmds.append(list(cmd))
            return type(
                "Process",
                (),
                {
                    "pid": 2**31 - 1,  # never a real process
                    "stdout": (),
                    "poll": lambda self: None,
                    "terminate": lambda self: None,
                    "wait": lambda self, timeout = None: 0,
                    "kill": lambda self: None,
                },
            )()

        def fake_health(timeout = None, **_kw):
            backend._stdout_lines = []
            return "draft-mtp" not in cmds[-1]

        backend._wait_for_health = fake_health
        with placement.patch.object(placement.subprocess, "Popen", side_effect = fake_popen):
            assert backend.load_model(
                placement.GgufLoadIntent(
                    gguf_path = str(gguf), model_identifier = model_identifier, **load_kwargs
                )
            )
        return {"cmd": cmds[-1]}

    monkeypatch.setattr(placement, "_launch", launch)
    return cmds


def test_a_drafter_that_fails_to_start_gives_the_spill_slots_back(
    tmp_path, _moe_cache_host, _failing_drafter
):
    """The cut needs a drafter; without one the launch is main's, at the asked count
    (as the drafter's own slot cut does, and as the mid-generation replay does)."""
    backend, _cmd = _load(tmp_path)

    first, retry = _failing_drafter[0], _failing_drafter[-1]
    assert _flag(first, "--parallel") == "2", first
    assert "draft-mtp" not in retry and "--spec-default" in retry, retry
    assert _flag(retry, "--parallel") == "4", retry
    assert backend.effective_parallel_slots == 4
    assert not backend._slots_cut_for_drafter


def test_the_mid_generation_replay_runs_at_the_asked_count(tmp_path, _moe_cache_host):
    """The crash replay is a load with speculation off: no drafter, so no cut."""
    import dataclasses

    backend, cmd = _load(tmp_path)
    assert _flag(cmd, "--parallel") == "2", cmd

    replay = dataclasses.replace(backend.last_load_intent, speculative_type = "off")
    _backend, cmd = _load(tmp_path, drafter = False, speculative_type = "off")
    assert replay.n_parallel == 4 and replay.n_parallel_from_default
    assert _flag(cmd, "--spec-type") != "draft-mtp", cmd
    assert _flag(cmd, "--parallel") == "4", cmd


@pytest.fixture
def _mla_head(monkeypatch):
    """The MoE fixture read as an MLA model (GLM/DeepSeek style) with an embedded head."""
    make = placement._moe_backend

    def mla(*args, **kwargs):
        backend, gguf = make(*args, **kwargs)

        def read_metadata(_path):
            backend._nextn_predict_layers = 1
            backend._kv_lora_rank = 512

        backend._read_gguf_metadata = read_metadata
        return backend, gguf

    monkeypatch.setattr(placement, "_moe_backend", mla)


@pytest.mark.parametrize("ngram_mod", [True, False])
def test_auto_standing_an_mla_head_down_keeps_the_slots(
    tmp_path, _moe_cache_host, _mla_head, monkeypatch, ngram_mod
):
    monkeypatch.delenv("UNSLOTH_MLA_MTP_ENABLED", raising = False)
    caps = dict(_CAPS, supports_ngram_mod = ngram_mod)
    backend, cmd = _cache_launch(
        tmp_path,
        caps = caps,
        n_parallel = 4,
        n_parallel_from_default = True,
        speculative_type = "auto",
    )

    assert backend.spec_fallback_reason == "mla_mtp_disabled"
    assert _flag(cmd, "--spec-type") != "draft-mtp", cmd
    assert _flag(cmd, "--parallel") == "4", cmd


def test_a_forced_mla_head_is_cut(tmp_path, _moe_cache_host, _mla_head):
    _backend, cmd = _load(tmp_path, drafter = False, speculative_type = "mtp")

    assert _flag(cmd, "--spec-type") == "draft-mtp", cmd
    assert _flag(cmd, "--parallel") == "2", cmd


@pytest.mark.parametrize(
    "how",
    [
        dict(load_mode = "mmap"),
        dict(extra_args = ["--load-mode", "mmap"]),
    ],
)
def test_a_deep_spill_loading_by_mmap_gets_no_cache_and_keeps_the_slots(
    tmp_path, _moe_cache_host, monkeypatch, how
):
    """The cache only launches pinned, so a user's mmap is a --fit split, not a cache."""
    monkeypatch.setenv("UNSLOTH_MOE_CACHE", "on")
    _backend, cmd = _load(tmp_path, memory = _DEEP_SPILL_CARD, **how)

    assert "--moe-cache-mib" not in cmd, cmd
    assert _flag(cmd, "--parallel") == "4", cmd


def test_a_shallow_spill_loading_by_mmap_is_still_cut(tmp_path, _moe_cache_host):
    """Near the fit edge the cut never needed the cache."""
    _backend, cmd = _load(tmp_path, load_mode = "mmap")

    assert _flag(cmd, "--parallel") == "2", cmd


@pytest.mark.parametrize("speculative_type", ["dspark", "dflash"])
def test_a_sidecar_mode_with_no_sidecar_keeps_the_slots(
    tmp_path, _moe_cache_host, _hybrid_head, speculative_type
):
    """DSpark / DFlash forced with no sidecar launch without a drafter, even with an
    embedded head in the model, so there is nothing to cut for."""
    caps = dict(_CAPS, supports_dspark = True, supports_dflash = True)
    backend, cmd = _cache_launch(
        tmp_path,
        caps = caps,
        n_parallel = 4,
        n_parallel_from_default = True,
        speculative_type = speculative_type,
    )

    assert "--model-draft" not in cmd, cmd
    assert _flag(cmd, "--parallel") == "4", cmd


def test_an_mtp_named_model_whose_drafter_is_dropped_keeps_the_slots(
    tmp_path, _moe_cache_host, monkeypatch
):
    """No embedded head and the sidecar unopenable: the launch has no drafter."""
    monkeypatch.setattr(llama_cpp_module, "_mtp_drafter_loads_standalone", lambda *a, **k: False)
    kw = dict(mtp_draft_path = str(_write_gguf(tmp_path / "mtp-draft.gguf")))
    backend, cmd = _cache_launch(
        tmp_path,
        caps = _CAPS,
        n_parallel = 4,
        n_parallel_from_default = True,
        speculative_type = "auto",
        model_identifier = "unsloth/Qwen3-MoE-MTP-GGUF",
        **kw,
    )

    assert _flag(cmd, "--spec-type") != "draft-mtp", cmd
    assert "--model-draft" not in cmd, cmd
    assert _flag(cmd, "--parallel") == "4", cmd


def test_an_mtp_named_model_with_its_drafter_is_still_cut(tmp_path, _moe_cache_host):
    _backend, cmd = _load(
        tmp_path, speculative_type = "auto", model_identifier = "unsloth/Qwen3-MoE-MTP-GGUF"
    )

    assert _flag(cmd, "--spec-type") == "draft-mtp", cmd
    assert _flag(cmd, "--parallel") == "2", cmd


@pytest.fixture
def _plain_head(monkeypatch):
    """The MoE fixture with an embedded MTP head that is neither hybrid nor MLA."""
    make = placement._moe_backend

    def plain(*args, **kwargs):
        backend, gguf = make(*args, **kwargs)

        def read_metadata(_path):
            backend._nextn_predict_layers = 1

        backend._read_gguf_metadata = read_metadata
        return backend, gguf

    monkeypatch.setattr(placement, "_moe_backend", plain)


def test_extras_running_the_embedded_head_under_a_sidecar_mode_are_cut(
    tmp_path, _moe_cache_host, _plain_head
):
    """DSpark picked with no sidecar, but the extras own --spec-type and run the head."""
    caps = dict(_CAPS, supports_dspark = True)
    _backend, cmd = _cache_launch(
        tmp_path,
        caps = caps,
        n_parallel = 4,
        n_parallel_from_default = True,
        speculative_type = "dspark",
        extra_args = ["--spec-type", "draft-mtp"],
    )

    assert "draft-mtp" in cmd, cmd
    assert _flag(cmd, "--parallel") == "2", cmd


def test_a_missing_drafter_file_keeps_the_slots(tmp_path, _moe_cache_host):
    """The launch drops a drafter that is not on disk (_resolve_launch_mtp_path)."""
    _backend, cmd = _cache_launch(
        tmp_path,
        caps = _CAPS,
        n_parallel = 4,
        n_parallel_from_default = True,
        speculative_type = "mtp",
        mtp_draft_path = str(tmp_path / "missing-mtp-draft.gguf"),
    )

    assert "--model-draft" not in cmd, cmd
    assert _flag(cmd, "--parallel") == "4", cmd


@pytest.mark.parametrize("smart_offload", [False, True])
def test_a_cache_the_ram_check_refuses_keeps_the_cut_placement(
    tmp_path, _moe_cache_host, monkeypatch, smart_offload
):
    """The cache is refused after placement was priced at the cut count (with smart
    offload a --fit off tensor plan), so the launch keeps the count it was priced at:
    raising it here would run four slots on a placement sized for two."""
    monkeypatch.setenv("UNSLOTH_MOE_CACHE", "on")
    if smart_offload:
        monkeypatch.setenv("UNSLOTH_SMART_OFFLOAD", "on")
    else:
        monkeypatch.delenv("UNSLOTH_SMART_OFFLOAD", raising = False)
    backend, cmd = _load(tmp_path, memory = [(0, 15_000, 32_000)], ram_gib = 24)

    assert "--moe-cache-mib" not in cmd, cmd
    assert _flag(cmd, "--spec-type") == "draft-mtp", cmd
    assert _flag(cmd, "--parallel") == "2", cmd
    assert backend.effective_parallel_slots == 2


def _cache_crash_launch(
    tmp_path,
    monkeypatch,
    *,
    memory,
    healthy = None,
    **load_kw,
):
    """First launch dies allocating the MoE cache; the uncached retry comes up unless
    ``healthy(cmd)`` says otherwise."""
    spawned = []
    crash_line = "failed to allocate the MoE cache buffers"

    def launch(
        backend,
        gguf,
        model_identifier = "test",
        **load_kwargs,
    ):
        def fake_popen(cmd, **kwargs):
            if not cmd or str(cmd[0]) != "/fake/llama-server":
                return placement._REAL_POPEN(cmd, **kwargs)
            spawned.append(list(cmd))
            crashed = len(spawned) == 1
            return type(
                "Process",
                (),
                {
                    "pid": 2**31 - 1,  # never a real process
                    "stdout": iter([crash_line + "\n"]) if crashed else iter(()),
                    "returncode": 1 if crashed else None,
                    "poll": lambda self: 1 if crashed else None,
                    "terminate": lambda self: None,
                    "wait": lambda self, timeout = None: 1 if crashed else 0,
                    "kill": lambda self: None,
                },
            )()

        def health(timeout = None, **_kw):
            if len(spawned) == 1:
                backend._stdout_lines = [crash_line]
                return False
            backend._stdout_lines = []
            return True if healthy is None else healthy(spawned[-1])

        backend._wait_for_health = health
        with placement.patch.object(placement.subprocess, "Popen", side_effect = fake_popen):
            try:
                backend.load_model(
                    placement.GgufLoadIntent(
                        gguf_path = str(gguf), model_identifier = model_identifier, **load_kwargs
                    )
                )
            except RuntimeError:
                # A ladder that ends without a healthy server; the argv list is the test.
                if healthy is None:
                    raise
        return {"cmd": spawned[-1]}

    monkeypatch.setattr(placement, "_launch", launch)
    backend, _cmd = _load(tmp_path, memory = memory, **load_kw)
    return backend, spawned


def test_a_cache_only_cut_gives_the_slots_back_when_the_cache_retry_drops_it(
    tmp_path, _moe_cache_host, monkeypatch
):
    monkeypatch.setenv("UNSLOTH_MOE_CACHE", "on")
    backend, spawned = _cache_crash_launch(tmp_path, monkeypatch, memory = _DEEP_SPILL_CARD)

    first, retry = spawned[0], spawned[-1]
    assert "--moe-cache-mib" in first and _flag(first, "--parallel") == "2", first
    assert "--moe-cache-mib" not in retry, retry
    assert _flag(retry, "--parallel") == "4", retry
    assert backend.effective_parallel_slots == 4
    assert not backend._slots_cut_for_drafter


def test_slots_given_back_for_the_cache_keep_host_lookup_tables_off_the_vram_plan(
    tmp_path, _moe_cache_host, monkeypatch
):
    """The restored slots reprice the plan other loads place against; a lookup table
    llama.cpp streams from the host stays out of it, as at the first plan."""
    monkeypatch.setenv("UNSLOTH_MOE_CACHE", "on")
    make = placement._moe_backend

    def big_drafter(*args, **kwargs):
        # A drafter large enough to spill experts that would fit beside the table.
        backend, gguf = make(*args, **kwargs)
        backend._estimate_mtp_overhead_bytes = lambda _ctx, *a, n_parallel = None, **k: int(
            15.5 * _GIB + 0.25 * _GIB * int(n_parallel or 1)
        )
        return backend, gguf

    monkeypatch.setattr(placement, "_moe_backend", big_drafter)
    backend, spawned = _cache_crash_launch(
        tmp_path,
        monkeypatch,
        memory = [(0, 24_576, 24_576)],
        size_gib = 40,
        expert_gib = 10,
        arch = "qwen4exp",
        lazy = {
            "per_layer_token_embd.weight": 26 * _GIB,
            "token_embd.weight": _GIB // 2,
            "output.weight": _GIB // 2,
        },
    )

    first, retry = spawned[0], spawned[-1]
    assert _flag(first, "--parallel") == "2" and _flag(retry, "--parallel") == "4", spawned
    # 40 GiB less 26.5 GiB of host tables, plus the 0.25 GiB KV stub.
    assert backend._planned_vram_mib == {0: 14_080}, backend._planned_vram_mib


def test_a_shallow_cut_survives_the_cache_retry(tmp_path, _moe_cache_host, monkeypatch):
    """Near the fit edge the cut never needed the cache, so dropping it keeps two."""
    backend, spawned = _cache_crash_launch(tmp_path, monkeypatch, memory = None)

    first, retry = spawned[0], spawned[-1]
    assert "--moe-cache-mib" in first and _flag(first, "--parallel") == "2", first
    assert "--moe-cache-mib" not in retry and _flag(retry, "--parallel") == "2", retry
    assert backend.effective_parallel_slots == 2


def test_slots_given_back_for_the_cache_are_not_cut_again_by_the_drafter_fallback(
    tmp_path, _moe_cache_host, monkeypatch
):
    """Once the asked count is back the spill cut is gone: a failing drafterless retry
    must not relaunch the identical four-slot argv as the cut count."""
    monkeypatch.setenv("UNSLOTH_MOE_CACHE", "on")
    backend, spawned = _cache_crash_launch(
        tmp_path,
        monkeypatch,
        memory = _DEEP_SPILL_CARD,
        healthy = lambda cmd: "draft-mtp" not in cmd and "--kv-unified" not in cmd,
    )

    later = spawned[1:]
    assert all(_flag(c, "--parallel") == "4" for c in later[:2]), later
    assert not any(a == b for a, b in zip(later, later[1:])), "identical relaunch"
    assert not backend._slots_cut_for_drafter


def test_a_first_launch_that_turns_unpinned_logs_its_own_stage(
    tmp_path, _moe_cache_host, monkeypatch
):
    """The first spawn drops a cache its load mode no longer pins and gives the slots
    back; the log names that launch, not a retry that never ran."""
    monkeypatch.setenv("UNSLOTH_MOE_CACHE", "on")
    real = llama_cpp_module._argv_loads_pinned
    # Emission (no cache yet in argv) still sees a pinned load; the spawn does not.
    monkeypatch.setattr(
        llama_cpp_module,
        "_argv_loads_pinned",
        lambda argv, env = None: "--moe-cache-mib" not in list(argv) and real(argv, env),
    )
    lines = []
    monkeypatch.setattr(
        llama_cpp_module.logger,
        "info",
        lambda msg, *a, **kw: lines.append(msg % a if a else msg),
    )
    backend, cmd = _load(tmp_path, memory = _DEEP_SPILL_CARD)

    assert "--moe-cache-mib" not in cmd, cmd
    assert _flag(cmd, "--parallel") == "4", cmd
    gave_back = [l for l in lines if "serving slots go back" in l]
    assert gave_back == [
        "MoE cache: not on the unpinned launch, so the serving slots go back 2 -> 4."
    ], lines


def _scripted_launch(
    tmp_path,
    monkeypatch,
    *,
    outcome,
    spawned = None,
    **load_kw,
):
    """A spilled MoE with a drafter whose spawns ``outcome(cmd)`` scripts as (exit code,
    None for healthy or "stall" for no loading progress, output lines). ``spawned``
    collects every argv, for a load expected to raise."""
    spawned = [] if spawned is None else spawned

    def launch(
        backend,
        gguf,
        model_identifier = "test",
        **load_kwargs,
    ):
        def fake_popen(cmd, **kwargs):
            if not cmd or str(cmd[0]) != "/fake/llama-server":
                return placement._REAL_POPEN(cmd, **kwargs)
            spawned.append(list(cmd))
            rc, _lines = outcome(cmd)
            rc = None if rc == "stall" else rc
            return type(
                "Process",
                (),
                {
                    "pid": 2**31 - 1,  # never a real process
                    "stdout": iter(()),
                    "returncode": rc,
                    "poll": lambda self: rc,
                    "terminate": lambda self: None,
                    "wait": lambda self, timeout = None: rc or 0,
                    "kill": lambda self: None,
                },
            )()

        def health(timeout = None, **_kw):
            rc, lines = outcome(spawned[-1])
            backend._stdout_lines = list(lines)
            backend._health_wait_stalled = rc == "stall"
            backend._health_wait_stall_kind = "no_progress" if rc == "stall" else None
            return rc is None

        backend._wait_for_health = health
        with placement.patch.object(placement.subprocess, "Popen", side_effect = fake_popen):
            assert backend.load_model(
                placement.GgufLoadIntent(
                    gguf_path = str(gguf), model_identifier = model_identifier, **load_kwargs
                )
            )
        return {"cmd": spawned[-1]}

    monkeypatch.setattr(placement, "_launch", launch)
    backend, _cmd = _load(tmp_path, **load_kw)
    return backend, spawned


def _vision_spill_launch(tmp_path, monkeypatch, *, outcome, **load_kw):
    """A multimodal deep spill the cache serves, spawns scripted as in _scripted_launch."""
    monkeypatch.setenv("UNSLOTH_MOE_CACHE", "on")
    make = placement._moe_backend

    def with_projector(*args, **kwargs):
        backend, gguf = make(*args, **kwargs)
        mmproj = _write_gguf(tmp_path / "mmproj.gguf", architecture = "clip")
        backend._resolve_launch_mmproj_path = lambda **_kw: str(mmproj)
        return backend, gguf

    monkeypatch.setattr(placement, "_moe_backend", with_projector)
    return _scripted_launch(
        tmp_path,
        monkeypatch,
        outcome = outcome,
        memory = _DEEP_SPILL_CARD,
        is_vision = True,
        **load_kw,
    )


def test_the_text_only_retry_after_a_failed_cpu_projector_runs_the_restored_slots(
    tmp_path, _moe_cache_host, monkeypatch
):
    """The CPU-projector retry drops the cache and gives the cut slots back; when it fails
    too, the text-only replay of the saved vision argv must launch that count, not the
    cut one the effective count no longer reports."""

    def outcome(cmd):
        if "--mmproj" not in cmd:
            return None, []
        if "--no-mmproj-offload" in cmd:
            return 1, ["srv    load_model: failed to load model"]
        return -11, []

    draft = _write_gguf(tmp_path / "user-draft.gguf")
    backend, spawned = _vision_spill_launch(
        tmp_path,
        monkeypatch,
        outcome = outcome,
        drafter = False,
        speculative_type = "off",
        extra_args = ["--flash-attn", "off", "--spec-type", "draft-simple", "-md", str(draft)],
    )

    first, text_only = spawned[0], spawned[-1]
    assert "--moe-cache-mib" in first and _flag(first, "--parallel") == "2", first
    assert any("--no-mmproj-offload" in c for c in spawned), spawned
    assert "--mmproj" not in text_only, text_only
    assert _flag(text_only, "--parallel") == str(backend.effective_parallel_slots), text_only
    assert backend.effective_parallel_slots == 4
    # The slots came back because the cache went; four over the cache is the launch the
    # cut exists to avoid.
    assert "--moe-cache-mib" not in text_only, text_only


def test_auto_standing_a_hybrid_head_down_keeps_the_slots_beside_a_bare_draft_model(
    tmp_path, _moe_cache_host, _hybrid_head
):
    """A bare -md in the extras does not stop Auto emitting --spec-type none, under
    which the draft model never drafts."""
    draft = _write_gguf(tmp_path / "user-draft.gguf")
    _backend, cmd = _load(
        tmp_path,
        drafter = False,
        speculative_type = "auto",
        extra_args = ["-md", str(draft)],
    )

    assert _flag(cmd, "--spec-type") == "none", cmd
    assert _flag(cmd, "--parallel") == "4", cmd


@pytest.mark.parametrize("ngram_mod", [True, False])
def test_a_bare_draft_model_beside_a_stood_down_mla_head(
    tmp_path, _moe_cache_host, _mla_head, monkeypatch, ngram_mod
):
    """Auto's ngram-mod replaces the bare -md; with no ngram-mod nothing is emitted and
    the -md runs as the drafter, so that one is still cut."""
    monkeypatch.delenv("UNSLOTH_MLA_MTP_ENABLED", raising = False)
    draft = _write_gguf(tmp_path / "user-draft.gguf")
    _backend, cmd = _cache_launch(
        tmp_path,
        caps = dict(_CAPS, supports_ngram_mod = ngram_mod),
        n_parallel = 4,
        n_parallel_from_default = True,
        speculative_type = "auto",
        extra_args = ["-md", str(draft)],
    )

    if ngram_mod:
        assert _flag(cmd, "--spec-type") == "ngram-mod", cmd
        assert _flag(cmd, "--parallel") == "4", cmd
    else:
        assert "--spec-type" not in cmd, cmd
        assert _flag(cmd, "--parallel") == "2", cmd


def test_a_vulkan_crash_replayed_on_cpu_runs_at_the_asked_count(
    tmp_path, _moe_cache_host, monkeypatch
):
    """The CPU replay streams no experts (the cut skips a CPU fallback), so it gets the
    slots back with the spill micro-batch and the cache it already drops."""
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(
        LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _binary = None: True)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_vulkan_prebuilt_was_auto_selected", staticmethod(lambda _binary: True)
    )
    make = placement._moe_backend

    def staged_cpu_runtime(*args, **kwargs):
        backend, gguf = make(*args, **kwargs)
        backend._cpu_isolated_binary = lambda _binary: "/fake/llama-server"
        backend._llama_server_env_for_binary = lambda _binary: {placement._loader_path_var(): ""}
        # The fixture answers "not Vulkan" on the instance.
        backend._is_vulkan_backend = lambda _binary = None: True
        return backend, gguf

    monkeypatch.setattr(placement, "_moe_backend", staged_cpu_runtime)

    def on_cpu(cmd):
        return "--device" in cmd and cmd[cmd.index("--device") + 1] == "none"

    backend, spawned = _scripted_launch(
        tmp_path,
        monkeypatch,
        outcome = lambda cmd: (None, []) if on_cpu(cmd) else (-11, ["ggml_vulkan: crash"]),
    )

    first, replay = spawned[0], spawned[-1]
    assert _flag(first, "--parallel") == "2", first
    assert on_cpu(replay), replay
    assert _flag(replay, "--parallel") == "4", replay
    assert backend.effective_parallel_slots == 4
    assert not backend._slots_cut_for_drafter


def test_a_single_sequence_recovery_is_not_recorded_as_the_spill_cut(
    tmp_path, _moe_cache_host, monkeypatch
):
    """One slot is the build's limit there: a chosen four would relaunch the same
    one-slot server, so it must dedupe."""

    def outcome(cmd):
        if _flag(cmd, "--parallel") == "1":
            return None, []
        return 1, ["a unified KV cache is only supported with a single sequence"]

    backend, spawned = _scripted_launch(tmp_path, monkeypatch, outcome = outcome)

    assert _flag(spawned[0], "--parallel") == "2", spawned[0]
    assert _flag(spawned[-1], "--parallel") == "1", spawned[-1]
    assert backend.effective_parallel_slots == 1
    assert not backend._slots_cut_for_drafter


def test_a_vulkan_with_no_device_starts_on_cpu_at_the_asked_count(
    tmp_path, _moe_cache_host, monkeypatch
):
    """The zero-device probe routes a load to --device none after the cut was priced."""
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(
        LlamaCppBackend, "_vulkan_prebuilt_was_auto_selected", staticmethod(lambda _binary: True)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_vulkan_probe_outcome", staticmethod(lambda _binary = None: "none")
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _binary = None: True)
    )
    make = placement._moe_backend

    def staged_cpu_runtime(*args, **kwargs):
        backend, gguf = make(*args, **kwargs)
        backend._is_vulkan_backend = lambda _binary = None: True
        backend._cpu_isolated_binary = lambda _binary: "/fake/llama-server"
        backend._llama_server_env_for_binary = lambda _binary: {placement._loader_path_var(): ""}
        return backend, gguf

    monkeypatch.setattr(placement, "_moe_backend", staged_cpu_runtime)
    # So the drafter can be pinned to the CPU and the route takes the replay.
    monkeypatch.setitem(_CAPS, "spec_draft_ngl_flag", "--gpu-layers-draft")
    backend, spawned = _scripted_launch(tmp_path, monkeypatch, outcome = lambda cmd: (None, []))

    (only,) = spawned
    assert only[only.index("--device") + 1] == "none", only
    assert _flag(only, "--parallel") == "4", only
    assert backend.effective_parallel_slots == 4


def _vulkan_cpu_host(monkeypatch, *, staged = True):
    """Auto-selected Vulkan with a staged CPU runtime (``staged`` off: none to stage)."""
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(
        LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _binary = None: True)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_vulkan_prebuilt_was_auto_selected", staticmethod(lambda _binary: True)
    )
    make = placement._moe_backend

    def staged_cpu_runtime(*args, **kwargs):
        backend, gguf = make(*args, **kwargs)
        # The fixture answers "not Vulkan" on the instance.
        backend._is_vulkan_backend = lambda _binary = None: True
        backend._cpu_isolated_binary = lambda _binary: "/fake/llama-server" if staged else None
        backend._llama_server_env_for_binary = lambda _binary: {placement._loader_path_var(): ""}
        # Checkpoints big enough that the host caps them below the build's default.
        backend._ctx_checkpoint_bytes = lambda *_a, **_k: _GIB // 4
        backend._host_memory_capacity_mib = lambda: 32 * 1024
        return backend, gguf

    monkeypatch.setattr(placement, "_moe_backend", staged_cpu_runtime)
    # Checkpoints emitted, and a drafter the CPU replay can pin.
    monkeypatch.setitem(_CAPS, "ctx_checkpoints_flag", "--ctx-checkpoints")
    monkeypatch.setitem(_CAPS, "spec_draft_ngl_flag", "--gpu-layers-draft")


def _on_cpu(cmd):
    return "--device" in cmd and cmd[cmd.index("--device") + 1] == "none"


def test_a_failed_cpu_drafter_replay_leaves_the_drafterless_one_one_checkpoint_cap(
    tmp_path, _moe_cache_host, _hybrid_head, monkeypatch
):
    """The drafter CPU replay fails; the drafterless one, saved at the cut, still sheds
    the cap it launched with instead of carrying two."""
    _vulkan_cpu_host(monkeypatch)

    def outcome(cmd):
        if not _on_cpu(cmd):
            return -11, ["ggml_vulkan: crash"]
        return (1, ["mtp failed"]) if _flag(cmd, "--spec-type") == "draft-mtp" else (None, [])

    backend, spawned = _scripted_launch(
        tmp_path, monkeypatch, outcome = outcome, drafter = False, speculative_type = "mtp"
    )

    replay = spawned[-1]
    assert any(_on_cpu(c) and _flag(c, "--spec-type") == "draft-mtp" for c in spawned), spawned
    assert _on_cpu(replay) and _flag(replay, "--spec-type") != "draft-mtp", replay
    assert replay.count("--ctx-checkpoints") == 1, replay
    assert _flag(replay, "--parallel") == "4", replay
    assert backend.effective_parallel_slots == 4


def test_a_cpu_replay_that_cannot_stage_leaves_the_gpu_ladder_at_the_cut(
    tmp_path, _moe_cache_host, monkeypatch
):
    """A stalled suspect Vulkan with no CPU runtime to stage goes on down the GPU
    rungs, which must still launch and report the cut count."""
    from core.inference.llama_cpp import LlamaCppBackend

    _vulkan_cpu_host(monkeypatch, staged = False)
    monkeypatch.setattr(
        LlamaCppBackend, "_vulkan_probe_outcome", staticmethod(lambda _binary = None: "timeout")
    )

    def outcome(cmd):
        if not spawned_before[0]:
            spawned_before[0] = cmd
        if cmd is spawned_before[0] or cmd == spawned_before[0]:
            return "stall", []
        return (1, ["boom"]) if _flag(cmd, "--parallel") == "4" else (None, [])

    spawned_before = [None]
    backend, spawned = _scripted_launch(tmp_path, monkeypatch, outcome = outcome)

    final = spawned[-1]
    assert len(spawned) > 2 and not _on_cpu(final), spawned
    assert _flag(final, "--parallel") == "2", final
    assert backend.effective_parallel_slots == 2
    assert backend._slots_cut_for_drafter


def test_a_cpu_start_at_the_asked_count_is_not_retried_as_a_cut(
    tmp_path, _moe_cache_host, monkeypatch
):
    """The zero-device route already runs the asked count, so a failing drafter fallback
    has no cut to retry at: no identical relaunch."""
    from core.inference.llama_cpp import LlamaCppBackend

    _vulkan_cpu_host(monkeypatch)
    monkeypatch.setattr(
        LlamaCppBackend, "_vulkan_probe_outcome", staticmethod(lambda _binary = None: "none")
    )
    seen = []
    with pytest.raises(RuntimeError):
        _scripted_launch(tmp_path, monkeypatch, outcome = lambda cmd: (1, ["boom"]), spawned = seen)

    assert all(_flag(c, "--parallel") == "4" for c in seen), seen
    # One drafterless launch: a "-retry-cut" would run the same four slots again.
    assert sum("--spec-default" in c for c in seen) == 1, seen
