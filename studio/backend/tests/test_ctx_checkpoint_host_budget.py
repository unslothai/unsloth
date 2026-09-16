# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Host-memory budgeting and pricing for llama-server context checkpoints."""

from __future__ import annotations

import pytest

from core.inference.llama_cpp import (
    LlamaCppBackend,
    ctx_checkpoints_allocated,
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
def _no_cgroup_limit(monkeypatch):
    """Keep the cases that are not about containers off the runner's own cgroup."""
    monkeypatch.setattr(LlamaCppBackend, "_cgroup_memory_limit_mib", staticmethod(lambda: None))


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

    def test_a_help_that_ran_and_named_neither_alias_is_zero(self):
        assert ctx_checkpoints_allocated({"found": True, "help_probe_ok": True}) is False

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
        assert source.count("_auto_ctx_checkpoints = (") == 1
        assert "self._bounded_ctx_checkpoints(n_parallel, server_caps, extra_args)" in source
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
        monkeypatch.setattr(
            LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: None)
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_cgroup_memory_limit_mib", staticmethod(lambda: 8 * 1024)
        )
        assert LlamaCppBackend._host_memory_capacity_mib() == 8 * 1024
        monkeypatch.setattr(
            LlamaCppBackend, "_cgroup_memory_limit_mib", staticmethod(lambda: None)
        )
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
        # One emission rule for the launch and the respawn, not two spellings of it.
        assert source.count("def _emit_auto_ctx_checkpoints(") == 1
        assert source.count("_emit_auto_ctx_checkpoints(cmd)") == 2
        strip = source.index("self._without_flag_pairs(cmd, _auto_ckpt_emitted)")
        tuning = source.index("_retry_cache_tuning_flags(")
        re_emit = source.rindex("_emit_auto_ctx_checkpoints(cmd)")
        assert strip < tuning < re_emit, "strip, re-decide the tuning, then re-apply the cap"
