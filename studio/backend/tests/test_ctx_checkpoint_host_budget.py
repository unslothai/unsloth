# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Host-memory budgeting and pricing for llama-server context checkpoints."""

from __future__ import annotations

import pytest

from core.inference.llama_cpp import (
    LlamaCppBackend,
    ctx_checkpoints_allocated,
    ctx_checkpoints_default_for_caps,
    effective_ctx_checkpoints_for_caps,
)
from core.inference.llama_server_args import (
    CTX_CHECKPOINTS_MIN_USEFUL,
    LLAMA_CTX_CHECKPOINTS_DEFAULT,
    ctx_checkpoints_within_host_budget,
    effective_ctx_checkpoints,
)

GIB = 1024**3
MIB = 1024**2

# unsloth/Qwen3.8-27B-GGUF, read from the shipped file: arch qwen35, 65 blocks,
# full_attention_interval 4, ssm.inner_size 6144, state_size 128, group_count 16, conv_kernel 4.
QWEN38_27B = {
    "_n_layers": 65,
    "_n_kv_heads": 4,
    "_n_heads": 24,
    "_embedding_length": 5120,
    "_kv_key_length": 256,
    "_kv_value_length": 256,
    "_full_attention_interval": 4,
    "_ssm_inner_size": 6144,
    "_ssm_state_size": 128,
    "_ssm_group_count": 16,
    "_ssm_conv_kernel": 4,
}


@pytest.fixture(autouse = True)
def _neutral_host(monkeypatch):
    """Keep every case off the runner's own cgroup and inherited llama.cpp settings."""
    monkeypatch.setattr(LlamaCppBackend, "_cgroup_memory_limit_mib", staticmethod(lambda: None))
    monkeypatch.delenv("LLAMA_ARG_CTX_CHECKPOINTS", raising = False)


def _backend(**overrides):
    b = LlamaCppBackend()
    for key, value in {**QWEN38_27B, **overrides}.items():
        setattr(b, key, value)
    return b


def _plain_attention_backend():
    return _backend(
        _full_attention_interval = None,
        _ssm_inner_size = None,
        _ssm_state_size = None,
        _ssm_group_count = None,
        _ssm_conv_kernel = None,
    )


# --------------------------------------------------------------- the per-snapshot cost


def test_one_snapshot_is_the_whole_recurrent_state():
    """Cross-check the helper against the model metadata."""
    b = _backend()
    d_inner, d_state, n_group, d_conv = 6144, 128, 16, 4
    n_embd_r = (d_conv - 1) * (d_inner + 2 * n_group * d_state)
    n_embd_s = d_state * d_inner
    n_recurrent = 65 - -(-65 // 4)
    assert b._rollback_state_bytes(1) == n_recurrent * (n_embd_r + n_embd_s) * 4
    assert b._rollback_state_bytes(1) == pytest.approx(149.6 * MIB, rel = 0.01)
    assert b._rollback_state_bytes(1) == _backend()._rollback_state_bytes(1)


def test_a_plain_attention_model_pays_nothing_per_snapshot():
    assert _plain_attention_backend()._rollback_state_bytes(1) == 0


# --------------------------------------------------------------- the budget arithmetic


def test_the_default_stands_where_a_snapshot_costs_nothing():
    assert ctx_checkpoints_within_host_budget(0, 4, 94 * GIB) == LLAMA_CTX_CHECKPOINTS_DEFAULT


def test_the_default_stands_where_the_host_cannot_be_read():
    assert ctx_checkpoints_within_host_budget(150 * MIB, 4, None) == LLAMA_CTX_CHECKPOINTS_DEFAULT
    assert ctx_checkpoints_within_host_budget(150 * MIB, 4, 0) == LLAMA_CTX_CHECKPOINTS_DEFAULT


def test_the_default_stands_where_it_already_fits():
    assert ctx_checkpoints_within_host_budget(16 * MIB, 4, 256 * GIB) == (
        LLAMA_CTX_CHECKPOINTS_DEFAULT
    )


def test_the_budget_binds_on_the_reported_hardware():
    got = ctx_checkpoints_within_host_budget(int(149.625 * MIB), 4, 94 * GIB)
    assert got == 8
    assert got * int(149.625 * MIB) * 4 <= int(94 * GIB * 0.05)


def test_a_small_host_still_keeps_the_floor():
    assert ctx_checkpoints_within_host_budget(600 * MIB, 4, 8 * GIB) == CTX_CHECKPOINTS_MIN_USEFUL


def test_more_slots_buy_fewer_snapshots_each():
    one = ctx_checkpoints_within_host_budget(int(149.625 * MIB), 1, 94 * GIB)
    four = ctx_checkpoints_within_host_budget(int(149.625 * MIB), 4, 94 * GIB)
    assert one == LLAMA_CTX_CHECKPOINTS_DEFAULT
    assert four == 8


# --------------------------------------------------------------- what the launcher emits


def _caps(flag = "--ctx-checkpoints"):
    return {"found": True, "ctx_checkpoints_flag": flag}


def test_the_launcher_bounds_a_hybrid_recurrent_load(monkeypatch):
    b = _backend()
    monkeypatch.setattr(
        LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: 94 * 1024)
    )
    assert b._bounded_ctx_checkpoints(4, _caps()) == 8


def test_the_launcher_leaves_every_other_load_alone(monkeypatch):
    monkeypatch.setattr(
        LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: 94 * 1024)
    )
    assert _plain_attention_backend()._bounded_ctx_checkpoints(4, _caps()) is None
    assert _backend()._bounded_ctx_checkpoints(
        4, {"found": True, "ctx_checkpoints_flag": None}
    ) is (None)
    assert _backend()._bounded_ctx_checkpoints(1, _caps()) is None


def test_the_launcher_reads_the_older_flag_spelling(monkeypatch):
    monkeypatch.setattr(
        LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: 94 * 1024)
    )
    assert _backend()._bounded_ctx_checkpoints(4, _caps("--swa-checkpoints")) == 8


def test_an_unreadable_host_leaves_the_launch_alone(monkeypatch):
    monkeypatch.setattr(LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: None))
    assert _backend()._bounded_ctx_checkpoints(4, _caps()) is None


# --------------------------------------------------------------- what the estimate prices


def test_the_estimator_charges_the_snapshots_on_the_hybrid_path():
    b = _backend()
    without = b._estimate_kv_cache_bytes(159744, "q4_0", n_parallel = 4, ctx_checkpoints = 0)
    with_32 = b._estimate_kv_cache_bytes(159744, "q4_0", n_parallel = 4, ctx_checkpoints = 32)
    assert with_32 - without == 4 * 32 * b._rollback_state_bytes(1)
    assert (with_32 - without) / GIB == pytest.approx(18.7, rel = 0.01)


def test_the_estimator_charges_them_on_the_legacy_hybrid_arm_too():
    b = _backend(_kv_key_length = None, _kv_value_length = None)
    without = b._estimate_kv_cache_bytes(8192, "f16", n_parallel = 2, ctx_checkpoints = 0)
    with_8 = b._estimate_kv_cache_bytes(8192, "f16", n_parallel = 2, ctx_checkpoints = 8)
    assert with_8 - without == 2 * 8 * b._rollback_state_bytes(1)


def test_the_checkpoint_charge_does_not_scale_with_context():
    b = _backend()
    delta = lambda ctx: (  # noqa: E731
        b._estimate_kv_cache_bytes(ctx, "q4_0", n_parallel = 4, ctx_checkpoints = 32)
        - b._estimate_kv_cache_bytes(ctx, "q4_0", n_parallel = 4, ctx_checkpoints = 0)
    )
    assert delta(8192) == delta(159744)


def test_a_plain_attention_model_is_unchanged_by_the_new_term():
    b = _plain_attention_backend()
    without = b._estimate_kv_cache_bytes(32768, "f16", n_parallel = 4, ctx_checkpoints = 0)
    with_32 = b._estimate_kv_cache_bytes(32768, "f16", n_parallel = 4, ctx_checkpoints = 32)
    assert with_32 == without


# --------------------------------------------------------------- blank is 32, not 0


def test_a_blank_field_prices_llama_cpps_own_default():
    assert effective_ctx_checkpoints(None, None, supports_flag = True) == (
        LLAMA_CTX_CHECKPOINTS_DEFAULT
    )


def test_a_blank_field_on_a_hybrid_prices_the_bound_the_launcher_applies():
    assert (
        effective_ctx_checkpoints(
            None,
            None,
            supports_flag = True,
            per_checkpoint_bytes = int(149.625 * MIB),
            n_parallel = 4,
            total_host_bytes = 94 * GIB,
        )
        == 8
    )


def test_an_explicit_count_is_honoured_whatever_it_costs():
    common = dict(
        supports_flag = True,
        per_checkpoint_bytes = int(149.625 * MIB),
        n_parallel = 4,
        total_host_bytes = 94 * GIB,
    )
    assert effective_ctx_checkpoints(None, 64, **common) == 64
    assert effective_ctx_checkpoints(["--ctx-checkpoints", "64"], None, **common) == 64
    assert effective_ctx_checkpoints(["--ctx-checkpoints", "64"], 4, **common) == 64


def test_an_explicit_zero_still_means_zero():
    common = dict(
        supports_flag = True,
        per_checkpoint_bytes = int(149.625 * MIB),
        n_parallel = 4,
        total_host_bytes = 94 * GIB,
    )
    assert effective_ctx_checkpoints(None, 0, **common) == 0
    assert effective_ctx_checkpoints(["--ctx-checkpoints", "0"], None, **common) == 0


def test_a_build_without_the_flag_allocates_none():
    assert effective_ctx_checkpoints(None, 32, supports_flag = False) == 0


# --------------------------------------------------------------- the probe's third state


class TestAnInconclusiveProbeIsNotProofOfAbsence:
    """An unanswered help probe is not proof that checkpoints are absent."""

    # A real successful parse always yields a catalogue, so every "the help ran" case
    # here carries one; an empty catalogue is silence, covered separately below.
    _PARSED = {"--flash-attn": "...", "--cache-ram": "..."}

    def test_a_help_that_ran_and_named_neither_alias_is_zero(self):
        assert (
            ctx_checkpoints_allocated({"found": True, "help_probe_ok": True, "flags": self._PARSED})
            is False
        )

    def test_an_exit_zero_probe_that_parsed_nothing_is_not_absence(self):
        """A wrapper that prints no help would otherwise zero the whole snapshot pool."""
        empty = {"found": True, "help_probe_ok": True, "flags": {}}
        assert ctx_checkpoints_allocated(empty) is True
        assert ctx_checkpoints_allocated({"found": True, "help_probe_ok": True}) is True
        assert (
            effective_ctx_checkpoints_for_caps(
                empty,
                None,
                None,
                per_checkpoint_bytes = int(149.625 * MIB),
                n_parallel = 4,
                total_host_bytes = 94 * GIB,
            )
            == LLAMA_CTX_CHECKPOINTS_DEFAULT
        )

    def test_a_real_empty_help_probe_prices_the_default(self, tmp_path):
        """End to end: a binary that exits 0 and says nothing must not read as absence."""
        fake = tmp_path / "llama-server"
        fake.write_text("#!/bin/sh\nexit 0\n")
        fake.chmod(0o755)
        caps = LlamaCppBackend.probe_server_capabilities(str(fake))
        assert caps["found"] is True and caps["help_probe_ok"] is True
        assert not caps["flags"] and caps["ctx_checkpoints_flag"] is None
        assert ctx_checkpoints_allocated(caps) is True
        backend = _backend()
        assert (
            effective_ctx_checkpoints_for_caps(
                caps,
                None,
                None,
                per_checkpoint_bytes = backend._rollback_state_bytes(1),
                n_parallel = 4,
                total_host_bytes = 94 * GIB,
            )
            == LLAMA_CTX_CHECKPOINTS_DEFAULT
        )

    def test_a_named_flag_is_priced(self):
        assert (
            ctx_checkpoints_allocated(
                {"found": True, "help_probe_ok": True, "ctx_checkpoints_flag": "--ctx-checkpoints"}
            )
            is True
        )

    @pytest.mark.parametrize(
        "caps",
        [
            pytest.param({"found": True, "help_probe_ok": False}, id = "probe-crashed"),
            pytest.param({"found": False, "help_probe_ok": False}, id = "no-binary-read"),
            pytest.param({}, id = "empty-caps"),
        ],
    )
    def test_an_unanswered_probe_keeps_the_default(self, caps):
        assert ctx_checkpoints_allocated(caps) is True
        got = effective_ctx_checkpoints_for_caps(
            caps,
            None,
            None,
            per_checkpoint_bytes = int(149.625 * MIB),
            n_parallel = 4,
            total_host_bytes = 94 * GIB,
        )
        assert got == LLAMA_CTX_CHECKPOINTS_DEFAULT

    def test_the_budget_narrows_only_once_the_flag_is_named(self):
        budget = dict(
            per_checkpoint_bytes = int(149.625 * MIB), n_parallel = 4, total_host_bytes = 94 * GIB
        )
        named = {"found": True, "help_probe_ok": True, "ctx_checkpoints_flag": "--ctx-checkpoints"}
        unnamed = {"found": True, "help_probe_ok": False, "ctx_checkpoints_flag": None}
        assert effective_ctx_checkpoints_for_caps(named, None, None, **budget) == 8
        assert effective_ctx_checkpoints_for_caps(unnamed, None, None, **budget) == (
            LLAMA_CTX_CHECKPOINTS_DEFAULT
        )

    @pytest.mark.parametrize(
        "caps",
        [
            pytest.param(
                {"found": True, "help_probe_ok": True, "ctx_checkpoints_flag": "--ctx-checkpoints"},
                id = "flag-named",
            ),
            pytest.param(
                {"found": True, "help_probe_ok": True, "ctx_checkpoints_flag": "--swa-checkpoints"},
                id = "older-alias",
            ),
            pytest.param(
                {"found": True, "help_probe_ok": False, "ctx_checkpoints_flag": None},
                id = "probe-crashed",
            ),
            pytest.param({"found": False, "help_probe_ok": False}, id = "no-binary-read"),
            pytest.param({}, id = "empty-caps"),
            pytest.param(
                {"found": True, "help_probe_ok": True, "ctx_checkpoints_flag": None},
                id = "help-ran-named-neither",
            ),
        ],
    )
    def test_what_is_priced_is_what_the_launch_will_run(self, caps, monkeypatch):
        """The priced count must match the count Studio can place in the command."""
        monkeypatch.setattr(
            LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: 94 * 1024)
        )
        backend = _backend()
        for requested in (None, 0, 4, 64):
            priced = effective_ctx_checkpoints_for_caps(
                caps,
                None,
                requested,
                per_checkpoint_bytes = backend._rollback_state_bytes(1),
                n_parallel = 4,
                total_host_bytes = 94 * GIB,
            )
            if not ctx_checkpoints_allocated(caps):
                expected = 0
            elif not caps.get("ctx_checkpoints_flag") and caps.get("found"):
                expected = LLAMA_CTX_CHECKPOINTS_DEFAULT
            elif not caps.get("ctx_checkpoints_flag"):
                expected = requested if requested is not None else LLAMA_CTX_CHECKPOINTS_DEFAULT
            elif requested is not None:
                expected = requested
            else:
                emitted = backend._bounded_ctx_checkpoints(4, caps)
                expected = emitted if emitted is not None else LLAMA_CTX_CHECKPOINTS_DEFAULT
            assert priced == expected, (caps, requested)

    def test_a_pass_through_count_is_honoured_even_on_an_unreadable_probe(self):
        caps = {"found": True, "help_probe_ok": False, "ctx_checkpoints_flag": None}
        budget = dict(
            per_checkpoint_bytes = int(149.625 * MIB), n_parallel = 4, total_host_bytes = 94 * GIB
        )
        assert (
            effective_ctx_checkpoints_for_caps(caps, ["--ctx-checkpoints", "64"], None, **budget)
            == 64
        )
        assert (
            effective_ctx_checkpoints_for_caps(caps, ["--ctx-checkpoints", "0"], None, **budget)
            == 0
        )
        assert (
            effective_ctx_checkpoints_for_caps(caps, ["--ctx-checkpoints", "64"], 4, **budget) == 64
        )

    def test_the_launcher_still_will_not_name_a_flag_it_cannot_confirm(self, monkeypatch):
        monkeypatch.setattr(
            LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: 94 * 1024)
        )
        caps = {"found": True, "help_probe_ok": False, "ctx_checkpoints_flag": None}
        assert ctx_checkpoints_allocated(caps) is True
        assert _backend()._bounded_ctx_checkpoints(4, caps) is None

    def test_every_pricing_site_goes_through_the_one_entry_point(self):
        import inspect

        from routes import inference as inference_routes
        from routes import models as models_routes

        for source in (
            inspect.getsource(inference_routes._gguf_runtime_bytes),
            inspect.getsource(models_routes.get_kv_cache_estimate),
            inspect.getsource(LlamaCppBackend.load_model),
        ):
            assert "effective_ctx_checkpoints_for_caps(" in source
            assert "supports_flag = " not in source


# --------------------------------------------------------------- and never against VRAM


class TestCheckpointsNeverReachAVramFigure:
    """Checkpoint storage must not be charged to VRAM."""

    def test_the_planner_and_the_launcher_agree_on_one_count(self):
        import inspect

        source = inspect.getsource(LlamaCppBackend.load_model)
        # One rule, re-asked wherever the slot count changes, never a second spelling.
        assert source.count("def _decide_auto_ctx_checkpoints(") == 1
        # The emitted cap and the load-mode host charge size one snapshot the same way.
        compact = "".join(source.split())
        sized = (
            "self._ctx_checkpoint_bytes(cache_type_kv,swa_full=swa_full,"
            "flash_attn=planned_flash_attn)"
        )
        assert f"per_checkpoint_bytes={sized}" in compact
        assert (
            "self._bounded_ctx_checkpoints(n_parallel,server_caps,extra_args,"
            "cache_type_kv=cache_type_kv,swa_full=swa_full,flash_attn=planned_flash_attn,)"
        ) in compact
        assert "per_checkpoint_bytes=self._rollback_state_bytes" not in compact
        assert source.count("self._bounded_ctx_checkpoints(") == 1
        assert "str(_auto_ctx_checkpoints)" in source

    def test_the_snapshots_are_host_only_in_the_load_mode_footprint(self):
        import inspect

        source = inspect.getsource(LlamaCppBackend.load_model)
        assert "kv_cache_bytes = _kv_bytes(effective_ctx, 0)," in source
        assert "host_only_bytes = (_cpu_draft_fit_bytes or 0) + _ckpt_host_bytes," in source
        assert "_kv_bytes(effective_ctx, _effective_ctx_checkpoints)" in source
        assert "- _kv_bytes(effective_ctx, 0)," in source

    def test_the_windows_tuning_is_not_handed_a_second_count(self):
        import inspect

        source = inspect.getsource(LlamaCppBackend.load_model)
        emit = source.index("str(_auto_ctx_checkpoints)")
        tuning = source.index(
            '_cache_flags_emitted.extend([str(server_caps["ctx_checkpoints_flag"]), "0"])'
        )
        assert emit > tuning, "the automatic cap must be emitted after the Windows tuning"
        assert "_flag_name(str(token)) in _CTX_CHECKPOINTS_FLAGS" in source

    def test_the_fit_does_not_charge_a_recurrent_snapshot(self):
        import inspect

        source = inspect.getsource(LlamaCppBackend.load_model)
        assert "0 if self._rollback_state_bytes(1) > 0 else _requested_ctx_checkpoints" in source
        assert source.count("ctx_checkpoints = _fit_ctx_checkpoints,") == 7
        assert source.count("_kv_bytes(effective_ctx, _effective_ctx_checkpoints)") == 1

    def test_the_fit_is_unmoved_by_the_new_term(self):
        b = _backend()
        assert (
            b._fit_context_to_vram(262144, 24 * 1024, 16 * 1024**3, "q4_0", n_parallel = 4) == 262144
        )
        assert b._estimate_kv_cache_bytes(
            262144, "q4_0", n_parallel = 4, ctx_checkpoints = 32
        ) > b._estimate_kv_cache_bytes(262144, "q4_0", n_parallel = 4, ctx_checkpoints = 0)

    def test_the_training_guard_nets_them_out(self):
        import inspect

        from routes import inference as inference_routes

        source = inspect.getsource(inference_routes._estimate_gguf_kv_gb)
        assert "runtime.kv_bytes - runtime.kv_checkpoint_bytes" in source
        assert "runtime.kv_bytes + runtime.compute_bytes" not in source
        # ...but only against a pool that is really separate from host RAM.
        assert "if shared_memory_pool" in source


def test_the_launcher_stands_down_for_a_typed_count(monkeypatch):
    monkeypatch.setattr(
        LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: 94 * 1024)
    )
    b = _backend()
    assert b._bounded_ctx_checkpoints(4, _caps(), ["--ctx-checkpoints", "64"]) is None
    assert b._bounded_ctx_checkpoints(4, _caps(), ["--swa-checkpoints=0"]) is None
    assert b._bounded_ctx_checkpoints(4, _caps(), ["--top-k", "20"]) == 8


def test_a_kda_hybrid_is_priced_wherever_it_is_bounded():
    b = LlamaCppBackend()
    for key, value in {
        "_n_layers": 8,
        "_n_heads": 8,
        "_n_kv_heads": 1,
        "_kv_lora_rank": 512,
        "_key_length_mla": 64,
        "_kda_head_dim": 128,
        "_ssm_conv_kernel": 4,
        "_n_kv_heads_by_layer": [1, 0, 1, 0, 1, 0, 1, 0],
    }.items():
        setattr(b, key, value)
    assert b._rollback_state_bytes(1) > 0
    without = b._estimate_kv_cache_bytes(8192, "f16", n_parallel = 2, ctx_checkpoints = 0)
    with_8 = b._estimate_kv_cache_bytes(8192, "f16", n_parallel = 2, ctx_checkpoints = 8)
    assert with_8 - without == 2 * 8 * b._rollback_state_bytes(1)


# --------------------------------------------------------------- inside a memory-limited container


class TestTheBudgetIsWhatThisProcessMayCharge:
    """MemTotal is the host's; a cgroup limit is what the child may actually allocate."""

    def _container(self, monkeypatch, host_gib, cgroup_gib):
        monkeypatch.setattr(
            LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: host_gib * 1024)
        )
        monkeypatch.setattr(
            LlamaCppBackend,
            "_cgroup_memory_limit_mib",
            staticmethod(lambda: None if cgroup_gib is None else cgroup_gib * 1024),
        )

    def test_the_capacity_is_the_tighter_of_the_two(self, monkeypatch):
        self._container(monkeypatch, 256, 8)
        assert LlamaCppBackend._host_memory_capacity_mib() == 8 * 1024
        self._container(monkeypatch, 8, 256)
        assert LlamaCppBackend._host_memory_capacity_mib() == 8 * 1024

    def test_an_uncapped_host_keeps_its_own_total(self, monkeypatch):
        self._container(monkeypatch, 94, None)
        assert LlamaCppBackend._host_memory_capacity_mib() == 94 * 1024

    def test_an_unreadable_host_falls_back_to_the_limit(self, monkeypatch):
        monkeypatch.setattr(LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: None))
        monkeypatch.setattr(
            LlamaCppBackend, "_cgroup_memory_limit_mib", staticmethod(lambda: 8 * 1024)
        )
        assert LlamaCppBackend._host_memory_capacity_mib() == 8 * 1024
        monkeypatch.setattr(LlamaCppBackend, "_cgroup_memory_limit_mib", staticmethod(lambda: None))
        assert LlamaCppBackend._host_memory_capacity_mib() is None

    def test_the_cap_fits_the_container_not_the_host(self, monkeypatch):
        """A 256 GiB host with an 8 GiB cgroup must not be given the host's 5%."""
        backend = _backend()
        per_slot_round = backend._rollback_state_bytes(1) * 4
        self._container(monkeypatch, 256, None)
        uncapped = backend._bounded_ctx_checkpoints(4, _caps())
        assert uncapped * per_slot_round > 8 * GIB, "the host-wide budget overruns the container"
        self._container(monkeypatch, 256, 8)
        bounded = backend._bounded_ctx_checkpoints(4, _caps())
        assert bounded == CTX_CHECKPOINTS_MIN_USEFUL
        assert bounded * per_slot_round < 8 * GIB

    def test_the_priced_count_is_still_the_emitted_one(self, monkeypatch):
        """The container must not split the planner's figure from the launched argv."""
        self._container(monkeypatch, 256, 8)
        backend = _backend()
        capacity = LlamaCppBackend._host_memory_capacity_mib()
        assert effective_ctx_checkpoints_for_caps(
            _caps(),
            None,
            None,
            per_checkpoint_bytes = backend._rollback_state_bytes(1),
            n_parallel = 4,
            total_host_bytes = capacity * MIB,
        ) == backend._bounded_ctx_checkpoints(4, _caps())

    def test_every_budget_site_reads_the_capacity(self):
        import inspect

        from routes import inference as inference_routes
        from routes import models as models_routes

        for source in (
            inspect.getsource(inference_routes._gguf_runtime_bytes),
            inspect.getsource(models_routes.get_kv_cache_estimate),
            inspect.getsource(LlamaCppBackend.load_model),
            inspect.getsource(LlamaCppBackend._bounded_ctx_checkpoints),
        ):
            assert "_host_memory_capacity_mib" in source
            assert "_total_system_memory_mib" not in source


# --------------------------------------------------------------- across the arch-crash retry


class TestTheCapSurvivesAWindowsDeviceRetry:
    """The retry re-decides the Windows cache tuning; the automatic cap must follow it."""

    def test_a_stated_cap_would_swallow_the_windows_zero(self):
        """Why the retry strips the pair first: the skip means 'the user typed one'."""
        caps = {"supports_cache_ram": True, "ctx_checkpoints_flag": "--ctx-checkpoints"}
        with_cap = ["llama-server", "-m", "x.gguf", "--ctx-checkpoints", "8"]
        assert LlamaCppBackend._retry_cache_tuning_flags(
            with_cap, cache_ram = None, ctx_checkpoints = None, server_caps = caps
        ) == ["--cache-ram", "0"]
        stripped = LlamaCppBackend._without_flag_pairs(with_cap, ["--ctx-checkpoints", "8"])
        assert LlamaCppBackend._retry_cache_tuning_flags(
            stripped, cache_ram = None, ctx_checkpoints = None, server_caps = caps
        ) == ["--cache-ram", "0", "--ctx-checkpoints", "0"]

    def test_a_typed_count_still_outranks_the_retry(self):
        caps = {"supports_cache_ram": True, "ctx_checkpoints_flag": "--ctx-checkpoints"}
        typed = ["llama-server", "--ctx-checkpoints", "64"]
        assert LlamaCppBackend._retry_cache_tuning_flags(
            typed, cache_ram = None, ctx_checkpoints = None, server_caps = caps
        ) == ["--cache-ram", "0"]

    def test_the_retry_strips_the_cap_then_re_emits_it(self):
        import inspect

        source = inspect.getsource(LlamaCppBackend.load_model)
        # One emission rule for the launch and every respawn, not a spelling each.
        assert source.count("def _emit_auto_ctx_checkpoints(") == 1
        # The launch, the arch-crash respawn, and the single-sequence retry.
        assert source.count("_emit_auto_ctx_checkpoints(cmd)") == 3
        # Every respawn that re-emits first takes back the pair it is replacing.
        assert source.count("self._without_flag_pairs(cmd, _auto_ckpt_emitted)") == 2
        tuning = source.index("_retry_cache_tuning_flags(")
        strip = source.index("self._without_flag_pairs(cmd, _auto_ckpt_emitted)")
        re_emit = source.index("_emit_auto_ctx_checkpoints(cmd)", tuning)
        assert strip < tuning < re_emit, "strip, re-decide the tuning, then re-apply the cap"

    def test_the_single_sequence_retry_re_decides_the_cap_for_one_slot(self):
        """llama-server refusing a unified cache drops to one slot, which affords more."""
        import inspect

        source = inspect.getsource(LlamaCppBackend.load_model)
        clamp = source.index("n_parallel = 1  # allow-slot-clamp: llama-server refused more")
        strip = source.index("self._without_flag_pairs(cmd, _auto_ckpt_emitted)", clamp)
        decide = source.index("_auto_ctx_checkpoints = _decide_auto_ctx_checkpoints()", clamp)
        emit = source.index("_emit_auto_ctx_checkpoints(cmd)", clamp)
        spawn = source.index('_spawn_and_wait(cmd, label = "-single-seq")')
        assert clamp < strip < decide < emit < spawn, "re-decide for one slot before spawning"


# --------------------------------------------------------------- what THIS build defaults to


class TestTheCapNeverRaisesTheBuildsOwnDefault:
    """The flag shipped at 3 as --swa-checkpoints (ggml-org/llama.cpp#15293) and is 32 now."""

    def _caps_with_default(
        self,
        advertised,
        flag = "--ctx-checkpoints",
    ):
        return {
            "found": True,
            "help_probe_ok": True,
            "ctx_checkpoints_flag": flag,
            "ctx_checkpoints_default": advertised,
        }

    @pytest.mark.parametrize(
        "block, expected",
        [
            pytest.param(
                "-ctxcp, --ctx-checkpoints, --swa-checkpoints N max number of context checkpoints"
                " to create per slot (default: 32)[(more info)](https://github.com/ggml-org/"
                "llama.cpp/pull/15293) (env: LLAMA_ARG_CTX_CHECKPOINTS)",
                32,
                id = "today-wrapped-across-columns",
            ),
            pytest.param(
                "--swa-checkpoints N max number of SWA checkpoints per slot to create"
                " (default: 3)",
                3,
                id = "as-it-shipped",
            ),
            pytest.param(
                "--ctx-checkpoints N number of context checkpoints (default: 0)", 0, id = "zero"
            ),
            pytest.param("--ctx-checkpoints N number of context checkpoints", None, id = "silent"),
            pytest.param(None, None, id = "no-block"),
        ],
    )
    def test_the_advertised_default_is_read_from_the_help_block(self, block, expected):
        assert LlamaCppBackend._advertised_int_default(block) == expected

    def test_a_silent_build_falls_back_to_upstreams_default_today(self):
        assert ctx_checkpoints_default_for_caps({}) == LLAMA_CTX_CHECKPOINTS_DEFAULT
        assert (
            ctx_checkpoints_default_for_caps(self._caps_with_default(None))
            == LLAMA_CTX_CHECKPOINTS_DEFAULT
        )
        assert ctx_checkpoints_default_for_caps(self._caps_with_default(3)) == 3
        assert ctx_checkpoints_default_for_caps(self._caps_with_default(0)) == 0

    def test_a_build_that_keeps_three_is_never_pushed_to_eight(self, monkeypatch):
        """94 GiB affords 8, but this build would only have kept 3 unflagged."""
        monkeypatch.setattr(
            LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: 94 * 1024)
        )
        backend = _backend()
        caps = self._caps_with_default(3, flag = "--swa-checkpoints")
        assert backend._bounded_ctx_checkpoints(4, _caps("--swa-checkpoints")) == 8
        assert backend._bounded_ctx_checkpoints(4, caps) is None

    def test_a_build_that_disables_them_is_left_disabled(self, monkeypatch):
        monkeypatch.setattr(
            LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: 94 * 1024)
        )
        backend = _backend()
        caps = self._caps_with_default(0)
        assert backend._bounded_ctx_checkpoints(4, caps) is None
        assert (
            effective_ctx_checkpoints_for_caps(
                caps,
                None,
                None,
                per_checkpoint_bytes = backend._rollback_state_bytes(1),
                n_parallel = 4,
                total_host_bytes = 94 * GIB,
            )
            == 0
        )

    def test_a_small_default_still_gets_capped_on_a_tiny_host(self, monkeypatch):
        """Capping below the build's own default is still allowed, and still floored at 2."""
        monkeypatch.setattr(
            LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: 4 * 1024)
        )
        backend = _backend()
        assert backend._bounded_ctx_checkpoints(4, self._caps_with_default(3)) == (
            CTX_CHECKPOINTS_MIN_USEFUL
        )

    def test_the_floor_never_climbs_above_the_default(self):
        for default in (0, 1, 2, 3):
            got = ctx_checkpoints_within_host_budget(
                int(149.625 * MIB), 4, 1 * GIB, upstream_default = default
            )
            assert got <= default, default

    def test_the_priced_count_is_the_builds_own_default_for_a_blank_field(self):
        for default in (0, 3, 32):
            caps = self._caps_with_default(default)
            assert (
                effective_ctx_checkpoints_for_caps(
                    caps, None, None, per_checkpoint_bytes = 0, n_parallel = 4, total_host_bytes = None
                )
                == default
            )

    def test_an_explicit_count_still_outranks_the_advertised_default(self):
        caps = self._caps_with_default(3)
        budget = dict(
            per_checkpoint_bytes = int(149.625 * MIB), n_parallel = 4, total_host_bytes = 94 * GIB
        )
        assert effective_ctx_checkpoints_for_caps(caps, None, 64, **budget) == 64
        assert (
            effective_ctx_checkpoints_for_caps(caps, ["--ctx-checkpoints", "64"], None, **budget)
            == 64
        )


# --------------------------------------------------------------- an inherited env setting


class TestAnInheritedEnvCountIsTheOperatorsSetting:
    """llama.cpp applies LLAMA_ARG_CTX_CHECKPOINTS before argv, so an emitted flag overrules it."""

    def _host(
        self,
        monkeypatch,
        gib = 94,
    ):
        monkeypatch.setattr(
            LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: gib * 1024)
        )

    def test_the_launcher_stands_down_for_an_inherited_count(self, monkeypatch):
        self._host(monkeypatch)
        backend = _backend()
        assert backend._bounded_ctx_checkpoints(4, _caps()) == 8
        monkeypatch.setenv("LLAMA_ARG_CTX_CHECKPOINTS", "256")
        assert backend._bounded_ctx_checkpoints(4, _caps()) is None
        monkeypatch.setenv("LLAMA_ARG_CTX_CHECKPOINTS", "0")
        assert backend._bounded_ctx_checkpoints(4, _caps()) is None

    def test_an_unparseable_env_value_is_left_to_llama_cpp(self, monkeypatch):
        self._host(monkeypatch)
        for raw in ("", "   ", "many", "-1"):
            monkeypatch.setenv("LLAMA_ARG_CTX_CHECKPOINTS", raw)
            assert _backend()._bounded_ctx_checkpoints(4, _caps()) == 8, raw

    def test_the_estimate_prices_the_inherited_count(self, monkeypatch):
        backend = _backend()
        budget = dict(
            per_checkpoint_bytes = backend._rollback_state_bytes(1),
            n_parallel = 4,
            total_host_bytes = 94 * GIB,
        )
        monkeypatch.setenv("LLAMA_ARG_CTX_CHECKPOINTS", "256")
        assert effective_ctx_checkpoints_for_caps(_caps(), None, None, **budget) == 256
        monkeypatch.setenv("LLAMA_ARG_CTX_CHECKPOINTS", "0")
        assert effective_ctx_checkpoints_for_caps(_caps(), None, None, **budget) == 0
        monkeypatch.delenv("LLAMA_ARG_CTX_CHECKPOINTS")
        assert effective_ctx_checkpoints_for_caps(_caps(), None, None, **budget) == 8

    def test_argv_still_outranks_the_environment(self, monkeypatch):
        """Studio appends its flag after llama.cpp has read the variable, so argv wins."""
        monkeypatch.setenv("LLAMA_ARG_CTX_CHECKPOINTS", "256")
        budget = dict(
            per_checkpoint_bytes = int(149.625 * MIB), n_parallel = 4, total_host_bytes = 94 * GIB
        )
        assert effective_ctx_checkpoints_for_caps(_caps(), None, 4, **budget) == 4
        assert (
            effective_ctx_checkpoints_for_caps(_caps(), ["--ctx-checkpoints", "64"], None, **budget)
            == 64
        )
        assert (
            effective_ctx_checkpoints_for_caps(_caps(), ["--ctx-checkpoints", "0"], None, **budget)
            == 0
        )

    def test_the_windows_tuning_does_not_zero_an_inherited_count(self):
        """The tuning stands down for the field; the variable is the same instruction."""
        import inspect

        source = inspect.getsource(LlamaCppBackend.load_model)
        assert "_ctx_checkpoints_owned = (" in source
        assert "else _env_ctx_checkpoints_override()" in source
        # The tuning's zero and the arch-crash retry both read the combined value.
        assert "if _ctx_checkpoints_owned is not None:" in source
        assert "ctx_checkpoints = _ctx_checkpoints_owned," in source
        owned = source.index("_ctx_checkpoints_owned = (")
        zero = source.index(
            '_cache_flags_emitted.extend([str(server_caps["ctx_checkpoints_flag"]), "0"])'
        )
        assert owned < zero, "decide ownership before the tuning would zero it"

    def test_the_retry_tuning_also_respects_an_inherited_count(self):
        """_retry_cache_tuning_flags reads the same combined value as the launch."""
        caps = {"supports_cache_ram": True, "ctx_checkpoints_flag": "--ctx-checkpoints"}
        cmd = ["llama-server", "-m", "x.gguf"]
        assert LlamaCppBackend._retry_cache_tuning_flags(
            cmd, cache_ram = None, ctx_checkpoints = None, server_caps = caps
        ) == ["--cache-ram", "0", "--ctx-checkpoints", "0"]
        # What load_model now passes when LLAMA_ARG_CTX_CHECKPOINTS is set.
        assert LlamaCppBackend._retry_cache_tuning_flags(
            cmd, cache_ram = None, ctx_checkpoints = 256, server_caps = caps
        ) == ["--cache-ram", "0"]

    def test_an_explicit_env_map_is_used_over_the_process_environment(self, monkeypatch):
        self._host(monkeypatch)
        monkeypatch.delenv("LLAMA_ARG_CTX_CHECKPOINTS", raising = False)
        assert (
            _backend()._bounded_ctx_checkpoints(
                4, _caps(), None, {"LLAMA_ARG_CTX_CHECKPOINTS": "256"}
            )
            is None
        )
        assert (
            effective_ctx_checkpoints_for_caps(
                _caps(),
                None,
                None,
                per_checkpoint_bytes = int(149.625 * MIB),
                n_parallel = 4,
                total_host_bytes = 94 * GIB,
                env = {"LLAMA_ARG_CTX_CHECKPOINTS": "256"},
            )
            == 256
        )


# --------------------------------------------------------------- after the fit picks the slots


class TestTheCapFollowsTheSlotCountTheChildGets:
    """The fit can cut n_parallel below the request; a cap sized for the request is wrong."""

    def test_the_cap_is_slot_sensitive_at_all(self, monkeypatch):
        monkeypatch.setattr(
            LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: 94 * 1024)
        )
        backend = _backend()
        assert backend._bounded_ctx_checkpoints(32, _caps()) == CTX_CHECKPOINTS_MIN_USEFUL
        # One slot affords the whole default, so nothing is emitted and nothing is lost.
        assert backend._bounded_ctx_checkpoints(1, _caps()) is None

    def test_both_counts_are_decided_after_the_fit_rebinds_the_slots(self):
        """Structural, because the reduction sits ~2000 lines inside one function."""
        import ast
        import inspect
        import textwrap

        source, first = inspect.getsourcelines(LlamaCppBackend.load_model)
        tree = ast.parse(textwrap.dedent("".join(source)))
        fit_rebinds = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Tuple)
                and any(isinstance(e, ast.Name) and e.id == "n_parallel" for e in t.elts)
                for t in node.targets
            )
        ]
        assert fit_rebinds, "the fit no longer rebinds n_parallel; re-check this ordering"
        decisions = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_auto_ctx_checkpoints" for t in node.targets
            )
        ] + [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_ctx_checkpoints_for_final_slots"
        ]
        assert len(decisions) >= 2, decisions
        assert min(decisions) > max(fit_rebinds), (decisions, fit_rebinds)


# --------------------------------------------------------------- one pool or two


class TestTheGuardOnlyDropsHostBytesFromADiscretePool:
    """On an iGPU or an APU the driver's free VRAM IS the host heap, so nothing may leave."""

    def _shares(self, **kwargs):
        from routes import inference as inference_routes
        return inference_routes._admission_pool_shares_host_ram(**kwargs)

    def test_an_integrated_vulkan_device_shares(self):
        # Vulkan reports total 0 only for an integrated GPU.
        assert self._shares(is_vulkan_backend = True, vulkan_gpu_memory = [(0, 8192, 0)]) is True

    def test_a_discrete_vulkan_device_does_not(self):
        assert self._shares(is_vulkan_backend = True, vulkan_gpu_memory = [(0, 8192, 24576)]) is False

    def test_a_confirmed_ordinal_pin_narrows_the_rows(self):
        """A mixed host pinned to the discrete adapter is checked against that adapter."""
        mixed = [(0, 8192, 0), (1, 8192, 24576)]
        assert (
            self._shares(
                is_vulkan_backend = True,
                vulkan_gpu_memory = mixed,
                requested_gpu_ids = [1],
                gpu_ids_are_vulkan_ordinals = True,
            )
            is False
        )
        assert (
            self._shares(
                is_vulkan_backend = True,
                vulkan_gpu_memory = mixed,
                requested_gpu_ids = [0],
                gpu_ids_are_vulkan_ordinals = True,
            )
            is True
        )

    def test_a_pin_in_an_unconfirmed_index_space_reads_every_row(self):
        """A physical id read as a Vulkan ordinal would answer for the wrong adapter."""
        mixed = [(0, 8192, 0), (1, 8192, 24576)]
        assert (
            self._shares(is_vulkan_backend = True, vulkan_gpu_memory = mixed, requested_gpu_ids = [1])
            is True
        )

    def test_a_pin_that_matches_no_row_fails_closed(self):
        assert (
            self._shares(
                is_vulkan_backend = True,
                vulkan_gpu_memory = [(0, 8192, 24576)],
                requested_gpu_ids = [7],
                gpu_ids_are_vulkan_ordinals = True,
            )
            is True
        )

    def test_a_mixed_or_unreadable_vulkan_inventory_fails_closed(self):
        assert self._shares(is_vulkan_backend = True, vulkan_gpu_memory = []) is True
        assert self._shares(is_vulkan_backend = True, vulkan_gpu_memory = None) is True
        assert (
            self._shares(is_vulkan_backend = True, vulkan_gpu_memory = [(0, 8192, 24576), (1, 4096, 0)])
            is True
        )

    def _rocm(self, monkeypatch, *, answered, unified):
        monkeypatch.setattr(LlamaCppBackend, "_torch_is_rocm", staticmethod(lambda _t: True))
        monkeypatch.setattr(
            LlamaCppBackend, "_rocm_classification_answered", staticmethod(lambda: answered)
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: unified)
        )

    def _cuda(self, monkeypatch, *, probe_free, integrated):
        monkeypatch.setattr(LlamaCppBackend, "_torch_is_rocm", staticmethod(lambda _t: False))
        monkeypatch.setattr(
            LlamaCppBackend, "_integrated_cuda_probe_is_free", staticmethod(lambda: probe_free)
        )

        def _must_not_probe():
            raise AssertionError("must not probe CUDA while training holds the card")

        monkeypatch.setattr(
            LlamaCppBackend,
            "_integrated_cuda_gpu_ids",
            staticmethod((lambda: integrated) if probe_free else _must_not_probe),
        )

    def test_a_discrete_cuda_host_keeps_the_subtraction(self, monkeypatch):
        self._cuda(monkeypatch, probe_free = True, integrated = set())
        assert self._shares(is_vulkan_backend = False, requested_gpu_ids = [0]) is False

    def test_an_integrated_cuda_part_shares(self, monkeypatch):
        """Jetson and DGX Spark set cudaDeviceProp::integrated and have one pool."""
        self._cuda(monkeypatch, probe_free = True, integrated = {0})
        assert self._shares(is_vulkan_backend = False, requested_gpu_ids = [0]) is True

    def test_an_uncached_cuda_classification_is_never_probed_for(self, monkeypatch):
        """The probe pins ~700 MiB per card for the process, and training holds them."""
        self._cuda(monkeypatch, probe_free = False, integrated = set())
        assert self._shares(is_vulkan_backend = False, requested_gpu_ids = [0]) is True

    def test_a_rocm_apu_shares_whether_or_not_it_is_pinned(self, monkeypatch):
        self._rocm(monkeypatch, answered = True, unified = {0})
        assert self._shares(is_vulkan_backend = False, requested_gpu_ids = [0]) is True
        assert self._shares(is_vulkan_backend = False, requested_gpu_ids = None) is True
        # A discrete sibling that is explicitly pinned is its own pool.
        assert self._shares(is_vulkan_backend = False, requested_gpu_ids = [1]) is False

    def test_a_rocm_host_with_no_apu_is_discrete(self, monkeypatch):
        self._rocm(monkeypatch, answered = True, unified = set())
        assert self._shares(is_vulkan_backend = False, requested_gpu_ids = [0]) is False

    def test_an_unanswered_rocm_classification_is_not_evidence_of_a_discrete_pool(
        self, monkeypatch
    ):
        """The classifier skips a device it cannot query, so empty is ambiguous there."""
        self._rocm(monkeypatch, answered = False, unified = set())
        assert self._shares(is_vulkan_backend = False, requested_gpu_ids = [0]) is True

    def test_an_unreadable_classifier_keeps_the_charge(self, monkeypatch):
        def _boom():
            raise RuntimeError("no driver")

        monkeypatch.setattr(LlamaCppBackend, "_torch_is_rocm", staticmethod(lambda _t: True))
        monkeypatch.setattr(LlamaCppBackend, "_rocm_classification_answered", staticmethod(_boom))
        assert self._shares(is_vulkan_backend = False, requested_gpu_ids = [0]) is True

    def test_the_figure_follows_the_verdict(self, tmp_path, monkeypatch):
        """The same runtime, priced against one pool and against two."""
        from routes import inference as inference_routes

        class _Runtime:
            kv_bytes = 10 * GIB
            kv_checkpoint_bytes = 4 * GIB
            compute_bytes = 1 * GIB

        monkeypatch.setattr(inference_routes, "_gguf_runtime_bytes", lambda *a, **k: _Runtime())
        shared = inference_routes._estimate_gguf_kv_gb("x.gguf", 4096)
        discrete = inference_routes._estimate_gguf_kv_gb("x.gguf", 4096, shared_memory_pool = False)
        assert shared == 11.0 and discrete == 7.0
        assert shared > discrete, "the shared pool must not be credited the host share"


# --------------------------------------------------------------- sliding-window snapshots

# unsloth/gemma-4-31B-it-GGUF's shape: every sixth block global (4 KV heads x 512), the rest a
# 1024-token window (16 KV heads x 256), so one f16 snapshot is 50 x 1024 x 16 x 256 x K+V x 2.
_GEMMA4_31B_SWA = [(i + 1) % 6 != 0 for i in range(60)]
_GEMMA4_31B = {
    "context_length": 262144,
    "block_count": 60,
    "embedding_length": 5376,
    "attention.head_count": 32,
    "attention.head_count_kv": [16 if swa else 4 for swa in _GEMMA4_31B_SWA],
    "attention.key_length": 512,
    "attention.value_length": 512,
    "attention.key_length_swa": 256,
    "attention.value_length_swa": 256,
    "attention.sliding_window": 1024,
    "attention.sliding_window_pattern": _GEMMA4_31B_SWA,
}
GEMMA4_SNAPSHOT = 800 * MIB


@pytest.fixture
def gemma4_gguf(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "_kv_cache_estimation_for_checkpoints",
        Path(__file__).parent / "test_kv_cache_estimation.py",
    )
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    kv = {"general.architecture": "gemma4"}
    kv.update({f"gemma4.{k}": v for k, v in _GEMMA4_31B.items()})
    path = tmp_path / "gemma-4-31B-it-Q6_K.gguf"
    path.write_bytes(helpers._make_gguf_bytes("gemma4", kv))
    monkeypatch.delenv("LLAMA_ARG_SWA_FULL", raising = False)
    monkeypatch.delenv("LLAMA_ARG_CTX_SIZE", raising = False)
    monkeypatch.setattr(
        LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: 94 * 1024)
    )
    caps = {**_caps(), "supports_kv_unified": True, "supports_flash_attn": True}
    monkeypatch.setattr(
        LlamaCppBackend, "probe_server_capabilities", classmethod(lambda cls, *a, **k: caps)
    )
    return path


def _gemma4_backend(path):
    b = LlamaCppBackend()
    b._read_gguf_metadata(str(path))
    return b


def test_an_swa_snapshot_is_its_window(gemma4_gguf):
    b = _gemma4_backend(gemma4_gguf)
    assert b._rollback_state_bytes(1) == 0
    assert b._ctx_checkpoint_bytes("f16") == GEMMA4_SNAPSHOT
    assert b._ctx_checkpoint_bytes("q8_0") < GEMMA4_SNAPSHOT
    # --swa-full keeps the whole context, so llama-server takes no SWA checkpoints.
    assert b._ctx_checkpoint_bytes("f16", swa_full = True) == 0
    assert _plain_attention_backend()._ctx_checkpoint_bytes("f16") == 0
    assert _backend()._ctx_checkpoint_bytes("f16") == _backend()._rollback_state_bytes(1)


def test_the_launcher_bounds_an_swa_load(gemma4_gguf):
    b = _gemma4_backend(gemma4_gguf)
    # 5% of 94 GiB is 4.7 GiB: six 800 MiB snapshots for one slot, the floor for four.
    assert b._bounded_ctx_checkpoints(1, _caps(), cache_type_kv = "f16") == 6
    assert b._bounded_ctx_checkpoints(4, _caps(), cache_type_kv = "f16") == (
        CTX_CHECKPOINTS_MIN_USEFUL
    )
    assert b._bounded_ctx_checkpoints(4, _caps(), cache_type_kv = "f16", swa_full = True) is None
    # Without flash attention a quantized V snapshot is priced as the retry's f16.
    assert b._bounded_ctx_checkpoints(1, _caps(), cache_type_kv = "q8_0", flash_attn = False) < (
        b._bounded_ctx_checkpoints(1, _caps(), cache_type_kv = "q8_0")
    )


@pytest.mark.parametrize("flash_attn", [None, False], ids = ["managed", "fa-off"])
def test_both_estimates_price_the_count_the_launcher_emits(gemma4_gguf, monkeypatch, flash_attn):
    """The reported load: Auto context, f16, four slots. Uncapped this was 100 GiB of host RAM."""
    import asyncio

    from routes import inference as inference_routes
    from routes import models as models_routes

    sized_with = []
    snapshot = LlamaCppBackend._ctx_checkpoint_bytes

    def spy(self, *args, **kwargs):
        sized_with.append(kwargs.get("flash_attn"))
        return snapshot(self, *args, **kwargs)

    monkeypatch.setattr(LlamaCppBackend, "_ctx_checkpoint_bytes", spy)
    emitted = _gemma4_backend(gemma4_gguf)._bounded_ctx_checkpoints(4, _caps(), cache_type_kv = "f16")
    expected = 4 * emitted * GEMMA4_SNAPSHOT

    extras = None if flash_attn is None else ["--flash-attn", "off"]
    sized_with.clear()
    panel = inference_routes._gguf_runtime_bytes(str(gemma4_gguf), 0, extras, 4, "f16", False)
    assert panel.n_ctx == 262144
    assert panel.kv_checkpoint_bytes == expected

    monkeypatch.setattr(
        models_routes, "_resolve_quant_gguf", lambda *_a: (str(gemma4_gguf), 25 * GIB)
    )
    hub = asyncio.run(
        models_routes.get_kv_cache_estimate(
            repo_id = "unsloth/gemma-4-31B-it-GGUF",
            quant = "Q6_K",
            n_ctx = 262144,
            cache_type_kv = "f16",
            n_parallel = 4,
            speculative_type = None,
            spec_draft_n_max = None,
            spec_draft_cache_type = None,
            ctx_checkpoints = None,
            disable_vision = True,
            n_batch = None,
            n_ubatch = None,
            tensor_parallel = False,
            flash_attn = flash_attn,
            kv_unified = None,
            swa_full = None,
            no_mmproj_offload = None,
            request = None,
            current_subject = "test",
        )
    )
    assert hub["kv_checkpoint_bytes"] == expected
    # The snapshot is sized at the attention mode each estimate prices its cache with.
    assert set(sized_with) == {flash_attn is None}

