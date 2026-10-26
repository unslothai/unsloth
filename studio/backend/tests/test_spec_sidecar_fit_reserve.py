# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Single-card --fit-target reserve for DSpark/DFlash sidecars llama.cpp's fitter cannot size."""

from __future__ import annotations

import inspect
import math
import os
import pathlib
from pathlib import Path

import pytest

from . import _sidecar_fit_helpers as h
from ._sidecar_fit_helpers import FLOOR, GIB, MIB, lc

B = lc.LlamaCppBackend


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    monkeypatch.delenv("LLAMA_ARG_DEVICE", raising = False)
    monkeypatch.delenv("LLAMA_ARG_SPLIT_MODE", raising = False)
    monkeypatch.setattr(lc, "_metal_capable_host", lambda: False)
    monkeypatch.setattr(B, "_apple_metal_memory_budget_bytes", staticmethod(lambda: 0))


def _launch_kind(
    tmp_path,
    kind = "dspark",
    use_fit = True,
    extra_args = None,
):
    backend, gguf = h.cuda_backend(tmp_path, draft_key = kind)
    backend._select_gpus = lambda *a, **k: (None, True) if use_fit else ([0], False)
    path = h.sidecar(tmp_path, kind)
    cmd = h.placement._launch(
        backend,
        gguf,
        speculative_type = kind,
        extra_args = extra_args,
        **{f"{kind}_draft_path": path},
    )["cmd"]
    return cmd, path


class TestIntegrityFlags:
    def test_reserve_rides_on_top_of_the_auto_fit_margin(self):
        base = B._ctx_integrity_flags(1, True, True, 0, 0, h.CAPS)
        flags = B._ctx_integrity_flags(1, True, True, 0, 0, h.CAPS, fit_target_reserve_mib = 7630)
        assert int(h.last(flags, "--fit-target")) == int(h.last(base, "--fit-target")) + 7630

    def test_reserve_rides_on_top_of_the_metal_floor(self):
        flags = B._ctx_integrity_flags(
            1, True, False, 0, 0, h.CAPS, fit_target_floor_mib = 6464, fit_target_reserve_mib = 7630
        )
        assert int(h.last(flags, "--fit-target")) == 6464 + 7630

    def test_reserve_makes_the_legacy_path_speak(self):
        # The legacy path is silent at llama.cpp's own default; a reserve is not that.
        assert "--fit-target" not in B._ctx_integrity_flags(1, True, False, 0, 0, h.CAPS)
        flags = B._ctx_integrity_flags(1, True, False, 0, 0, h.CAPS, fit_target_reserve_mib = 100)
        assert int(h.last(flags, "--fit-target")) == 1024 + 100

    @pytest.mark.parametrize("use_fit, caps", [(False, h.CAPS), (True, {})])
    def test_no_fit_or_no_flag_emits_nothing(self, use_fit, caps):
        assert "--fit-target" not in B._ctx_integrity_flags(
            1, use_fit, False, 0, 0, caps, fit_target_reserve_mib = 100
        )


class TestReserveArithmetic:
    def _backend(self, file_bytes):
        backend = B()
        backend._get_gguf_size_bytes = lambda _path: file_bytes
        return backend

    def test_planner_reserve_wins_when_larger(self):
        assert self._backend(8 * GIB)._sidecar_fit_reserve_mib(9 * GIB, "/d.gguf") == 9 * 1024

    def test_file_size_floors_a_flat_fallback(self):
        assert self._backend(8 * GIB)._sidecar_fit_reserve_mib(0, "/d.gguf") == 8 * 1024

    def test_rounds_up(self):
        assert self._backend(0)._sidecar_fit_reserve_mib(MIB + 1, "/d.gguf") == 2

    def test_nothing_known_is_zero(self):
        assert self._backend(0)._sidecar_fit_reserve_mib(0, None) == 0


def test_an_unsized_sidecar_carries_the_flat_reserve():
    b = B.__new__(B)
    b._get_gguf_size_bytes = lambda path: None
    assert b._sidecar_fit_reserve_mib(0, "o/remote-dspark", flat_mib = 1200) == 1200
    assert b._sidecar_fit_reserve_mib(2 * GIB, "o/remote-dspark", flat_mib = 1200) == 2048


class TestDiscreteLaunch:
    @pytest.mark.parametrize("kind", ["dspark", "dflash"])
    def test_fit_on_keeps_the_sidecar_free(self, tmp_path, kind):
        cmd, path = _launch_kind(tmp_path, kind = kind)
        assert h.last(cmd, "--model-draft") == path
        assert h.margin(cmd) >= FLOOR

    def test_fit_off_has_no_fitter_to_tell(self, tmp_path):
        cmd, path = _launch_kind(tmp_path, use_fit = False)
        assert h.last(cmd, "--model-draft") == path
        assert "--fit-target" not in cmd

    def test_a_cpu_pinned_drafter_reserves_nothing(self, tmp_path):
        cmd, path = _launch_kind(tmp_path, extra_args = ["--spec-draft-ngl", "0"])
        assert h.last(cmd, "--model-draft") == path
        assert h.margin(cmd) < FLOOR


def test_the_integrity_margin_never_carries_the_reserve():
    """The reserve rides only on the trailing pair the spawn appends, so it can neither make
    an unpriced child look budget-priced nor be mistaken for a user's equal margin."""
    src = Path(lc.__file__).read_text(encoding = "utf-8")
    assert "fit_target_reserve_mib = _sidecar_fit_reserve_mib" not in src
    assert "fit_target_reserve_mib = 0," in src


@pytest.mark.parametrize("spec_type", ["draft-dspark", "draft-dflash"])
def test_a_pass_through_sidecar_is_reserved_too(tmp_path, spec_type):
    """Extras owning --spec-type replace Studio's emission, so the reserve follows the
    sidecar the child really loads, even with Studio's own speculation off."""
    backend, gguf = h.cuda_backend(tmp_path, draft_key = "sidecar")
    path = tmp_path / "custom-sidecar-Q8_0.gguf"
    path.write_bytes(b"draft")
    cmd = h.placement._launch(
        backend,
        gguf,
        speculative_type = "off",
        extra_args = ["--spec-type", spec_type, "--model-draft", str(path)],
    )["cmd"]
    assert str(path) in cmd
    assert h.margin(cmd) >= FLOOR


# _sidecar_reserve_margin_flags: the spawn's one question of the argv + env it launches.


@pytest.mark.parametrize(
    "args, env",
    [
        (["--model-draft", "/d.gguf"], {}),
        (["-md", "/d.gguf"], {}),
        (["--spec-draft-model", "/d.gguf"], {}),
        (["--model-draft=/d.gguf"], {}),
        ([], {"LLAMA_ARG_SPEC_DRAFT_MODEL": "/d.gguf"}),
    ],
    ids = ["long", "short", "spec-draft-model", "equals", "env"],
)
def test_every_draft_spelling_gets_the_reserve(args, env):
    flags = lc._sidecar_reserve_margin_flags(["--fit", "on", *args], (), env, 1024, 8192)
    assert flags == ["--fit-target", str(1024 + 8192)]


@pytest.mark.parametrize(
    "args, env",
    [(["--fit", "off"], {}), ([], {"LLAMA_ARG_FIT": "off"}), (["-fit", "off"], {})],
    ids = ["argv", "env", "short"],
)
def test_no_reserve_when_the_fitter_is_off(args, env):
    assert lc._sidecar_reserve_margin_flags(["-md", "/d.gguf", *args], (), env, 1024, 8192) == []


def test_no_reserve_without_a_drafter():
    assert lc._sidecar_reserve_margin_flags(["--fit", "on"], (), {}, 1024, 8192) == []


@pytest.mark.parametrize(
    "user_args, env, want",
    [
        ((), {"LLAMA_ARG_FIT_TARGET": "12000"}, 12000 + 8192),
        (("--fit-target", "16384"), {}, 16384 + 8192),
        # A single-device launch keeps device 0's entry, not the list maximum.
        (("-fitt", "512/20000"), {}, 1024 + 8192),
    ],
    ids = ["env", "argv", "per-device-list"],
)
def test_the_reserve_sits_on_top_of_the_users_margin(user_args, env, want):
    flags = lc._sidecar_reserve_margin_flags(
        ["-md", "/d.gguf", *user_args], user_args, env, 1024, 8192
    )
    assert flags == ["--fit-target", str(want)]


@pytest.mark.parametrize("value", ["1024,65536", "1024/65536"], ids = ["comma", "slash"])
def test_single_device_reserve_uses_the_first_list_entry(value):
    flags = lc._sidecar_reserve_margin_flags(
        ["--fit", "on", "-md", "/d.gguf", "--fit-target", value],
        ("--fit-target", value),
        {},
        1024,
        8192,
    )
    assert flags == ["--fit-target", str(1024 + 8192)]


@pytest.mark.parametrize(
    "extras",
    [["--fit-target", "1024"], ["-fitt", "1024"], ["--fit-target=1024"]],
    ids = ["spaced", "short", "equals"],
)
def test_a_user_fit_target_keeps_the_reserve(tmp_path, extras):
    backend, gguf = h.cuda_backend(tmp_path)
    cmd = h.dspark(backend, gguf, tmp_path, extra_args = extras)
    values = [
        cmd[i + 1] if tok in ("--fit-target", "-fitt") else tok.split("=", 1)[1]
        for i, tok in enumerate(cmd)
        if tok in ("--fit-target", "-fitt") or tok.startswith("--fit-target=")
    ]
    assert int(values[-1]) >= FLOOR


@pytest.mark.parametrize(
    "extras",
    [["--fit", "on"], ["--fit=on"], ["-fit", "on"]],
    ids = ["spaced", "equals", "short"],
)
def test_extras_that_turn_the_fitter_on_get_the_reserve(tmp_path, extras):
    backend, gguf = h.cuda_backend(tmp_path)
    backend._select_gpus = lambda *a, **k: ([0], False)
    assert h.margin(h.dspark(backend, gguf, tmp_path, extra_args = extras)) >= FLOOR


def test_an_inherited_env_margin_is_kept_under_the_reserve(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_FIT_TARGET", "12000")
    backend, gguf = h.cuda_backend(tmp_path)
    assert h.margin(h.dspark(backend, gguf, tmp_path)) >= 12000 + 8 * 1024


def test_the_reserve_carries_the_draft_compute_floor(tmp_path):
    backend, gguf = h.cuda_backend(tmp_path)
    assert h.margin(h.dspark(backend, gguf, tmp_path)) >= FLOOR + 224


def test_full_draft_offload_still_gets_the_reserve(tmp_path):
    backend, gguf = h.cuda_backend(tmp_path)
    assert h.margin(h.dspark(backend, gguf, tmp_path)) >= FLOOR


def test_manual_auto_context_prices_the_native_context(tmp_path):
    backend, gguf = h.cuda_backend(tmp_path)
    backend._context_length = 131072
    h.per_token_estimate(backend, 64 * 1024)  # 8 GiB of draft KV at 131072
    cmd = h.dspark(backend, gguf, tmp_path, n_ctx = 0, **h.MANUAL_AUTO)
    if "--fit-target" not in cmd:
        pytest.skip("this host plans no fitter for Manual + Auto")
    assert h.margin(cmd) >= 512 + 16 * 1024


def test_explicit_manual_context_is_priced_at_that_context(tmp_path):
    backend, gguf = h.cuda_backend(tmp_path)
    backend._context_length = 131072
    h.per_token_estimate(backend, 64 * 1024)
    cmd = h.dspark(backend, gguf, tmp_path, n_ctx = 4096, **h.MANUAL_AUTO)
    if "--fit-target" not in cmd:
        pytest.skip("this host plans no fitter for Manual + Auto")
    assert h.last(cmd, "-c") == "4096"
    assert h.margin(cmd) < 512 + 12 * 1024


def test_manual_auto_context_prices_an_inherited_context(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", "8192")
    backend, gguf = h.cuda_backend(tmp_path)
    backend._context_length = 131072
    h.per_token_estimate(backend, 64 * 1024)  # 0.5 GiB of draft KV at 8192
    cmd = h.dspark(backend, gguf, tmp_path, n_ctx = 0, **h.MANUAL_AUTO)
    assert "--fit-target" in cmd
    assert h.margin(cmd) < 512 + 12 * 1024


def test_an_automatic_context_is_priced_at_the_final_slots(tmp_path):
    backend, gguf = h.cuda_backend(tmp_path)
    backend._context_length = 32768
    h.per_token_estimate(backend, 32 * 1024)  # 1 GiB at 32768, 8 GiB at 8 x 32768
    cmd = h.dspark(
        backend,
        gguf,
        tmp_path,
        n_ctx = 0,
        n_parallel = 1,
        extra_args = ("--parallel", "8", "--no-kv-unified"),
        **h.MANUAL_AUTO,
    )
    assert "--fit-target" in cmd
    assert h.margin(cmd) >= 512 + 16 * 1024


def test_a_trailing_zero_context_is_priced_at_the_native_length(tmp_path):
    backend, gguf = h.cuda_backend(tmp_path)
    backend._context_length = 131072
    h.per_token_estimate(backend, 64 * 1024)
    cmd = h.dspark(
        backend, gguf, tmp_path, gpu_memory_mode = "auto", n_ctx = 4096, extra_args = ["-c", "0"]
    )
    assert h.last(cmd, "-c") == "0", cmd
    assert h.margin(cmd) >= 16 * 1024, cmd


def test_an_unread_native_length_prices_the_fitters_minimum_context():
    assert lc._sidecar_child_ctx([], {}, None, 1, True) == lc._LLAMA_FIT_MIN_CTX
    assert lc._sidecar_child_ctx([], {}, None, 4, False) == 4 * lc._LLAMA_FIT_MIN_CTX
    assert lc._sidecar_child_ctx(["-c", "8192"], {}, None, 4, False) == 8192


@pytest.mark.parametrize(
    "args, env, want",
    [
        # No -c: the pool is n_parallel * per-slot limit (tools/server/server.cpp).
        (["--kv-unified-per-slot", "2048"], {}, 4 * 2048),
        (["--kv-unified-per-slot=65536"], {}, 4 * 65536),
        ([], {"LLAMA_ARG_KV_UNIFIED_PER_SLOT": "2048"}, 4 * 2048),
        (["-c", "16384", "--kv-unified-per-slot", "2048"], {}, 16384),
        # -c 0 asks for the full window; a later --fit-ctx clears that.
        (["-c", "0", "--kv-unified-per-slot", "2048"], {}, 131072),
        (["-c", "0", "--fit-ctx", "8192", "--kv-unified-per-slot", "2048"], {}, 4 * 2048),
        ([], {}, 131072),
    ],
)
def test_the_per_slot_limit_sizes_the_priced_context(args, env, want):
    assert lc._sidecar_child_ctx(args, env, 131072, 4, True) == want


@pytest.mark.parametrize(
    "extras",
    [("--parallel", "8"), ("-np", "8"), ("--parallel=8",)],
    ids = ["long", "short", "equals"],
)
def test_pass_through_slots_are_priced_on_the_first_spawn(tmp_path, extras):
    backend, gguf = h.cuda_backend(tmp_path)
    h.slot_priced(backend)
    cmd = h.dspark(backend, gguf, tmp_path, n_ctx = 4096, n_parallel = 1, extra_args = extras)
    assert h.margin(cmd) >= FLOOR + 8 * 1024


def test_fewer_final_slots_lower_the_reserve(tmp_path):
    backend, gguf = h.cuda_backend(tmp_path)
    h.slot_priced(backend)
    cmd = h.dspark(
        backend, gguf, tmp_path, n_ctx = 4096, n_parallel = 8, extra_args = ("--parallel", "1")
    )
    assert FLOOR + 1024 <= h.margin(cmd) < FLOOR + 2 * 1024
    assert cmd.count("--fit-target") == 1


def _ubatch_priced(backend):
    backend._estimate_mtp_overhead_bytes = lambda n_ctx, n_ubatch = None, **kw: (
        8 * GIB + int(n_ubatch or 512) * MIB
    )


@pytest.mark.parametrize(
    "extras, env",
    [
        (("--batch-size", "4096", "--spec-draft-n-ubatch", "4096"), {}),
        ((), {"LLAMA_ARG_BATCH": "4096", "LLAMA_ARG_SPEC_DRAFT_N_UBATCH": "4096"}),
    ],
    ids = ["flag", "env"],
)
def test_the_draft_micro_batch_is_priced(tmp_path, monkeypatch, extras, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    backend, gguf = h.cuda_backend(tmp_path)
    _ubatch_priced(backend)
    cmd = h.dspark(backend, gguf, tmp_path, n_ctx = 4096, extra_args = extras)
    assert h.margin(cmd) >= FLOOR + 4096


def test_the_draft_micro_batch_is_capped_at_the_batch(tmp_path):
    """llama_context caps n_ubatch at n_batch; the draft context inherits the target's 2048."""
    backend, gguf = h.cuda_backend(tmp_path)
    _ubatch_priced(backend)
    cmd = h.dspark(
        backend, gguf, tmp_path, n_ctx = 4096, extra_args = ["--spec-draft-n-ubatch", "8192"]
    )
    assert FLOOR + 2048 <= h.margin(cmd) < FLOOR + 4096


def test_a_zero_draft_micro_batch_follows_the_target():
    assert lc._draft_n_ubatch(["--spec-draft-n-ubatch", "0"], {}) is None
    assert lc._draft_n_ubatch(["--spec-draft-n-ubatch=256"], {}) == 256


def test_underscore_spellings_are_read_like_llama_cpp():
    assert lc._draft_card_share_unknown(["--spec_draft_n_cpu_moe", "8"], {})
    assert lc._draft_n_ubatch(["--spec_draft_n_ubatch", "2048"], {}) == 2048


def _dsv41_dspark_pair():
    """DeepSeek-V4.1 target and its DSpark drafter, dims read off the published GGUFs."""
    target = B.__new__(B)
    target._embedding_length, target._vocab_size = 5120, 129280
    draft = B.__new__(B)
    for k, v in dict(
        _architecture = "dflash",
        _embedding_length = 5120,
        _n_layers = 3,
        _n_heads = 64,
        _n_kv_heads = 1,
        _kv_key_length = 512,
        _kv_value_length = 512,
        _feed_forward_length = 2304,
        _expert_feed_forward_length = 2304,
        _vocab_size = 129280,
    ).items():
        setattr(draft, k, v)
    target._draft_backend_for = lambda path: draft
    return target


@pytest.mark.parametrize("ub, measured_mib", [(512, 725.5), (2048, 2900.5)])
def test_a_dspark_prompt_graph_is_priced_with_the_micro_batch(ub, measured_mib):
    """llama.cpp b11734 reserves 725.5 MiB at -ub 512 and 2900.5 MiB at 2048 for this
    drafter's compute; the old flat 349 MiB let the reserve fall 2.2 GiB short."""
    b = _dsv41_dspark_pair()
    got = (
        b._mtp_draft_compute_bytes(8192, drafter_path = "/d.gguf", n_ubatch = ub)
        + B._MTP_DRAFT_COMPUTE_BYTES
    ) / MIB
    assert measured_mib <= got <= measured_mib * 1.25


_ROLLBACK = 4 * GIB


def _rollback_backend(tmp_path):
    backend, gguf = h.cuda_backend(tmp_path)
    backend._rollback_state_bytes = lambda n_parallel = 1: _ROLLBACK
    backend._estimate_mtp_overhead_bytes = (
        lambda n_ctx, target_rollback = True, spec_draft_n_max = 0, **kw: (
            8 * GIB + (_ROLLBACK * spec_draft_n_max if target_rollback else 0)
        )
    )
    return backend, gguf


def test_initial_reserve_leaves_target_rollback_to_the_fitter(tmp_path):
    backend, gguf = _rollback_backend(tmp_path)
    assert FLOOR <= h.margin(h.dspark(backend, gguf, tmp_path)) < FLOOR + _ROLLBACK // MIB


@pytest.mark.parametrize(
    "extras", [("-ngld", "1"), ("--spec-draft-ngl", "1")], ids = ["short", "long"]
)
def test_partial_draft_offload_gets_no_whole_file_reserve(tmp_path, extras):
    backend, gguf = h.cuda_backend(tmp_path)
    assert h.margin(h.dspark(backend, gguf, tmp_path, extra_args = extras)) < FLOOR


@pytest.mark.parametrize(
    "extras",
    [
        ("-otd", ".*=CPU"),
        ("--override-tensor-draft", ".*=CPU"),
        ("--spec-draft-override-tensor", ".*=CPU"),
        ("--spec-draft-override-tensor=.*=CPU",),
    ],
    ids = ["otd", "override-tensor-draft", "spec-draft-override-tensor", "equals"],
)
def test_a_draft_tensor_override_gets_no_whole_file_reserve(tmp_path, extras):
    backend, gguf = h.cuda_backend(tmp_path)
    cmd = h.dspark(backend, gguf, tmp_path, extra_args = extras)
    assert h.margin(cmd) < FLOOR
    assert extras[-1] in cmd


def test_a_target_tensor_override_keeps_the_reserve(tmp_path):
    backend, gguf = h.cuda_backend(tmp_path)
    cmd = h.dspark(backend, gguf, tmp_path, extra_args = ("-ot", "blk\\.0\\.ffn_up=CPU"))
    assert h.margin(cmd) >= FLOOR


@pytest.mark.parametrize(
    "extras",
    [
        ("--spec-draft-cpu-moe",),
        ("-cmoed",),
        ("--spec-draft-n-cpu-moe", "99"),
        ("--n-cpu-moe-draft=4",),
    ],
    ids = ["cpu-moe", "short", "n-cpu-moe", "n-equals"],
)
def test_draft_cpu_moe_gets_no_whole_file_reserve(tmp_path, extras):
    backend, gguf = h.cuda_backend(tmp_path)
    assert h.margin(h.dspark(backend, gguf, tmp_path, extra_args = extras)) < FLOOR


@pytest.mark.parametrize(
    "env",
    [{"LLAMA_ARG_SPEC_DRAFT_CPU_MOE": "1"}, {"LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE": "8"}],
    ids = ["cpu-moe", "n-cpu-moe"],
)
def test_draft_cpu_moe_from_the_env_is_a_placement_too(env):
    assert lc._draft_card_share_unknown([], env)
    assert not lc._draft_card_share_unknown([], {"LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE": "0"})


def test_a_positive_draft_cpu_moe_count_is_not_undone_by_a_later_zero():
    """llama.cpp appends the draft CPU overrides per positive count; a later 0 adds none
    but removes none either."""
    assert lc._draft_card_share_unknown(["--spec-draft-n-cpu-moe", "8", "-ncmoed", "0"], {})
    assert not lc._draft_card_share_unknown(["-ncmoed", "0"], {})


@pytest.mark.parametrize("extras", [("-ngld", "1"), ("-otd", "blk\\.0\\.=CPU")])
def test_a_partly_placed_drafter_keeps_its_compute_floor(tmp_path, extras):
    backend, gguf = h.cuda_backend(tmp_path)
    floor = math.ceil(B._MTP_DRAFT_COMPUTE_BYTES / MIB)
    assert (
        1024 + floor <= h.margin(h.dspark(backend, gguf, tmp_path, extra_args = list(extras))) < FLOOR
    )


@pytest.mark.parametrize(
    "extras, env",
    [(["-nkvo"], {}), (["--no-kv-offload"], {}), ([], {"LLAMA_ARG_NO_KV_OFFLOAD": "1"})],
)
def test_host_resident_draft_kv_is_not_reserved_on_the_card(tmp_path, monkeypatch, extras, env):
    """The draft context copies the target's params (common_base_params_to_speculative), so
    -nkvo puts the draft KV in host buffers too."""
    for d in ("on", "off"):
        (tmp_path / d).mkdir()

    def margin_of(d, extras):
        backend, gguf = h.cuda_backend(d)
        backend._mtp_draft_kv_bytes = lambda *a, **k: h.DRAFT_KV
        return h.margin(h.dspark(backend, gguf, d, extra_args = list(extras)))

    on = margin_of(tmp_path / "on", [])
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert on - margin_of(tmp_path / "off", extras) >= 760


@pytest.mark.parametrize("device", ["none", "cpu"])
def test_a_cpu_child_gets_no_reserve_and_no_report(tmp_path, monkeypatch, device):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    messages = h.info_messages(monkeypatch)
    backend, gguf = h.cuda_backend(tmp_path)
    cmd = h.dspark(backend, gguf, tmp_path, extra_args = ["--device", device])
    assert h.margin(cmd) < 1024 + 4 * 1024
    assert not any("cannot size the sidecar" in m for m in messages)


def test_a_draft_with_its_own_embeddings_and_head_gets_no_reserve(tmp_path, monkeypatch):
    """llama_context needs the target's context only when the draft lacks token_embd or output."""
    monkeypatch.setattr(lc, "_sidecar_borrows_target_tensors", lambda path: False)
    backend, gguf = h.cuda_backend(tmp_path)
    assert h.margin(h.dspark(backend, gguf, tmp_path)) < FLOOR


def _draft_with(monkeypatch, tmp_path, tensors):
    import utils.models.gguf_metadata as gm
    import utils.models.model_config as mc

    path = tmp_path / "draft.gguf"
    path.write_bytes(b"GGUF")
    monkeypatch.setattr(mc, "colocated_split_shards", lambda p: ([p], True))
    monkeypatch.setattr(gm, "_gguf_shard_has_named_tensor", lambda shard, name: name in tensors)
    return str(path)


@pytest.mark.parametrize(
    "present, borrows",
    [
        ({"token_embd.weight", "output.weight"}, False),
        ({"token_embd.weight"}, True),
        ({"output.weight"}, True),
        (set(), True),
    ],
)
def test_borrowed_tensor_detection(tmp_path, monkeypatch, present, borrows):
    assert (
        lc._sidecar_borrows_target_tensors(_draft_with(monkeypatch, tmp_path, present)) is borrows
    )
    assert lc._sidecar_borrows_target_tensors(None) is True


@pytest.mark.parametrize(
    "tensors, borrows, borrows_output",
    [
        # Upstream DFlash borrows the target's head when a draft ships none, and no marker
        # tells a build that ties it to the draft's embeddings apart.
        ({"token_embd.weight"}, True, True),
        ({"output.weight"}, True, False),
        (set(), True, True),
        ({"token_embd.weight", "output.weight"}, False, False),
    ],
)
def test_a_draft_missing_a_tensor_borrows_it(
    monkeypatch, tmp_path, tensors, borrows, borrows_output
):
    path = _draft_with(monkeypatch, tmp_path, tensors)
    assert lc._sidecar_borrows_target_tensors(path) is borrows
    assert lc._sidecar_borrows_output(path) is borrows_output


def test_an_unread_draft_borrows_the_output():
    assert lc._sidecar_borrows_output(None) is True


def test_native_measurement_is_read_from_the_shared_common_library(tmp_path):
    binary = h.elf(tmp_path / "llama-server", b"libllama-common.so.0")
    real = h.elf(tmp_path / "libllama-common.so.0.0.1", b"..." + h.NATIVE_MARKER + b"...")
    (tmp_path / "libllama-common.so.0").symlink_to(real.name)
    (tmp_path / "libllama-common.so").symlink_to("libllama-common.so.0")
    assert lc._fit_measures_borrowed_drafts(str(binary))


def test_native_measurement_is_read_from_a_static_server(tmp_path):
    assert lc._fit_measures_borrowed_drafts(str(h.elf(tmp_path / "llama-server", h.NATIVE_MARKER)))


def test_native_measurement_is_read_through_the_installer_wrapper(tmp_path):
    h.elf(tmp_path / "llama-server-real", h.NATIVE_MARKER)
    wrapper = tmp_path / "llama-server"
    wrapper.write_text('#!/bin/sh\nexec "$(dirname "$0")/llama-server-real" "$@"\n')
    wrapper.chmod(0o755)
    assert lc._fit_measures_borrowed_drafts(str(wrapper))


def test_an_older_build_does_not_claim_native_measurement(tmp_path):
    binary = h.elf(tmp_path / "llama-server")
    h.elf(
        tmp_path / "libllama-common.so",
        b"failed to measure the memory of the extra model, fitting without it",
    )
    assert not lc._fit_measures_borrowed_drafts(str(binary))
    assert not lc._fit_measures_borrowed_drafts(None)


def test_only_the_linked_common_library_speaks_for_the_build(tmp_path):
    binary = h.elf(tmp_path / "llama-server", b"libllama-common.so.0")
    h.elf(tmp_path / "libllama-common.so.0")
    h.elf(tmp_path / "libllama-common.so.0.9.9", h.NATIVE_MARKER)
    assert not lc._fit_measures_borrowed_drafts(str(binary))
    h.elf(tmp_path / "libllama-common.so.0", h.NATIVE_MARKER)
    assert lc._fit_measures_borrowed_drafts(str(binary))


def test_a_static_server_beside_a_stale_common_library_keeps_the_reserve(tmp_path):
    binary = h.elf(tmp_path / "llama-server")
    h.elf(tmp_path / "libllama-common.so.0", h.NATIVE_MARKER)
    assert not lc._fit_measures_borrowed_drafts(str(binary))


def test_the_common_library_is_followed_through_the_server_impl_library(tmp_path):
    """b11734 layout: llama-server links libllama-server-impl.so, which links
    libllama-common.so.0, where fit.cpp lives."""
    binary = h.elf(tmp_path / "llama-server", b"libllama-server-impl.so")
    h.elf(tmp_path / "libllama-server-impl.so", b"libllama-common.so.0")
    h.elf(tmp_path / "libllama-common.so.0", h.NATIVE_MARKER)
    assert lc._fit_measures_borrowed_drafts(str(binary))
    h.elf(tmp_path / "libllama-server-impl.so")
    assert not lc._fit_measures_borrowed_drafts(str(binary))


@pytest.mark.parametrize(
    "linked", ["libllama-common.so.0", "libllama-common.0.dylib", "llama-common.dll"]
)
def test_a_server_that_names_its_common_library_is_read_through_it(tmp_path, linked):
    binary = h.elf(tmp_path / "llama-server", linked.encode())
    h.elf(tmp_path / linked, h.NATIVE_MARKER)
    assert lc._fit_measures_borrowed_drafts(str(binary))


def test_the_linked_library_names_are_read_once_per_binary_version(tmp_path, monkeypatch):
    server = tmp_path / "llama-server"
    server.write_bytes(b"\0libllama-common.so.0\0" * 4)
    lib = tmp_path / "libllama-common.so.0"
    lib.write_bytes(b"lib")
    reads = []
    real = pathlib.Path.read_bytes

    def read_bytes(self):
        reads.append(self.name)
        return real(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", read_bytes)
    for _ in range(3):
        assert lc._linked_llama_libs(server, tmp_path) == [lib]
    assert reads.count("llama-server") == 1
    # Existence is not cached.
    lib.unlink()
    assert lc._linked_llama_libs(server, tmp_path) == []
    assert reads.count("llama-server") == 1
    lib.write_bytes(b"lib")
    server.write_bytes(b"\0libllama-server-impl.so\0")
    st = server.stat()
    os.utime(server, ns = (st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    (tmp_path / "libllama-server-impl.so").write_bytes(b"impl")
    assert lc._linked_llama_libs(server, tmp_path) == [tmp_path / "libllama-server-impl.so"]
    assert reads.count("llama-server") == 2


def test_a_missing_binary_fails_closed(tmp_path):
    caps = B.probe_server_capabilities(str(tmp_path / "missing"))
    assert caps["fit_measures_borrowed_drafts"] is False


def test_a_cached_probe_rereads_the_shared_library(tmp_path, monkeypatch):
    """The capability cache is keyed on the server alone; a swapped libllama-common
    must not keep an older answer."""
    binary = h.elf(tmp_path / "llama-server", b"libllama-common.so")
    st = binary.stat()
    key = (str(binary), st.st_mtime_ns, st.st_size)
    monkeypatch.setattr(B, "_exec_path_for_launch", staticmethod(lambda b: b))
    monkeypatch.setitem(
        B._capability_cache,
        key,
        {"found": True, "supports_fit_target": True, "fit_measures_borrowed_drafts": True},
    )
    assert B.probe_server_capabilities(str(binary))["fit_measures_borrowed_drafts"] is False
    h.elf(tmp_path / "libllama-common.so", h.NATIVE_MARKER)
    assert B.probe_server_capabilities(str(binary))["fit_measures_borrowed_drafts"] is True


@pytest.mark.parametrize("native", [False, True], ids = ["older", "native"])
def test_the_reserve_stands_down_when_the_fitter_measures_the_draft(tmp_path, native):
    backend, gguf = h.cuda_backend(tmp_path)
    backend.probe_server_capabilities = lambda _binary = None: {
        **h.CAPS,
        "fit_measures_borrowed_drafts": native,
    }
    cmd = h.dspark(backend, gguf, tmp_path)
    assert "--model-draft" in cmd
    assert (h.margin(cmd) >= FLOOR) is not native


def test_a_default_slider_single_card_reserve_is_logged(tmp_path, monkeypatch):
    backend, gguf = h.cuda_backend(tmp_path)
    messages = h.info_messages(monkeypatch)
    assert h.margin(h.dspark(backend, gguf, tmp_path)) >= FLOOR
    assert any("llama.cpp cannot size the sidecar, so" in m for m in messages)


def _warned(
    tmp_path,
    monkeypatch,
    kind,
    fit_target = True,
):
    messages = h.info_messages(monkeypatch)
    backend, gguf = h.cuda_backend(tmp_path, draft_key = kind)
    caps = dict(h.CAPS, supports_fit_target = fit_target)
    backend.probe_server_capabilities = lambda _binary = None: dict(caps)
    path = h.sidecar(tmp_path, kind)
    cmd = h.placement._launch(backend, gguf, speculative_type = kind, **{f"{kind}_draft_path": path})[
        "cmd"
    ]
    assert f"draft-{kind}" in cmd
    return any("is not reserved" in m for m in messages)


@pytest.mark.parametrize("kind", ["dspark", "dflash"])
def test_a_self_contained_drafter_is_not_reported_unreserved(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(lc, "_sidecar_borrows_target_tensors", lambda path: False)
    assert _warned(tmp_path, monkeypatch, kind) is False


@pytest.mark.parametrize("kind", ["dspark", "dflash"])
def test_a_borrowing_drafter_without_a_reserve_is_reported(tmp_path, monkeypatch, kind):
    assert _warned(tmp_path, monkeypatch, kind, fit_target = False) is True


def test_a_one_card_tensor_split_is_reserved_and_reported(tmp_path, monkeypatch):
    """Tensor mode folds several cards into one meta device; one card has nothing to fold."""
    backend, gguf = h.cards(tmp_path, h.ONE)
    seen = h.spy_fit_sized(backend)
    messages = h.info_messages(monkeypatch)
    cmd = h.dspark(backend, gguf, tmp_path, extra_args = ["--split-mode", "tensor"])
    assert h.margin(cmd) >= FLOOR
    assert seen["fit_sized"] is True
    assert not any("is not reserved" in m for m in messages)


# Retries: the reserve is reconciled at every spawn off the argv being launched.


def _flash_crash(i, cmd):
    return -11 if h.last(cmd, "--flash-attn") == "on" else None


def _flash_off(i, cmd):
    return h.last(cmd, "--flash-attn") == "off"


def test_the_drafter_free_startup_retry_drops_the_reserve(tmp_path):
    backend, gguf = h.cuda_backend(tmp_path)
    cmds = h.spawns(
        backend,
        gguf,
        healthy = lambda i, cmd: "draft-dspark" not in cmd,
        n_ctx = 4096,
        speculative_type = "dspark",
        dspark_draft_path = h.sidecar(tmp_path),
    )
    assert cmds.loaded
    first, retry = cmds[0], cmds[-1]
    assert int(h.last(first, "--fit-target")) >= FLOOR
    assert "--model-draft" not in retry
    assert h.margin(retry) < FLOOR


def test_the_drafter_free_fallback_keeps_the_reserve_for_a_user_drafter(tmp_path):
    """The fallback removes Studio's own block; a bare user --model-draft survives and
    still loads, so its reserve must too."""
    backend, gguf = h.cuda_backend(tmp_path)
    custom = tmp_path / "custom-dspark-Q8_0.gguf"
    custom.write_bytes(b"draft")
    cmds = h.spawns(
        backend,
        gguf,
        healthy = lambda i, cmd: "draft-dspark" not in cmd,
        n_ctx = 4096,
        speculative_type = "dspark",
        dspark_draft_path = h.sidecar(tmp_path),
        extra_args = ("--model-draft", str(custom)),
    )
    retry = cmds[-1]
    if "draft-dspark" in retry:
        pytest.skip("the user drafter kept the speculative block; no fallback ran")
    assert str(custom) in retry
    assert h.margin(retry) >= FLOOR


def _spawn_source():
    src = inspect.getsource(B.load_model)
    spawn = src[src.index("def _spawn_and_wait(") :]
    return src, spawn[: spawn.index("_last_spawn_cmd = list(run_cmd)")]


def test_the_fit_on_startup_retry_carries_the_reserve():
    """Every attempt of _spawn_and_wait (the --fit on retry re-enters its loop) reconciles
    the reserve before spawning. Checked at the source: the retry needs a real crash."""
    _src, spawn = _spawn_source()
    assert "_sidecar_fit_reserve_eligible_mib" in spawn
    assert "_sidecar_reserve_margin_flags(" in spawn


def test_the_startup_retry_reads_the_child_not_a_literal_flag():
    src, spawn = _spawn_source()
    retry = src[src.index("retrying once with --fit on so it can offload") :]
    retry = retry[: retry.index("_did_fit_retry = True")]
    assert "_sidecar_reserve_margin_flags(" in spawn
    assert '"--model-draft" in _run' not in retry
    assert "fit_target_margin_in(extra_args, None)" not in retry


def test_the_noflash_fit_flip_carries_the_reserve(tmp_path):
    backend, gguf = h.cuda_backend(tmp_path)
    backend._select_gpus = lambda *a, **k: ([0], False)  # planned full offload: --fit off
    cmds = h.spawns(
        backend,
        gguf,
        crash = _flash_crash,
        healthy = _flash_off,
        late_exit = True,
        n_ctx = 4096,
        speculative_type = "dspark",
        dspark_draft_path = h.sidecar(tmp_path),
    )
    noflash = [c for c in cmds if h.last(c, "--flash-attn") == "off"]
    if not noflash or h.last(noflash[0], "--fit") != "on":
        pytest.skip("no --flash-attn off retry with the fitter on in this harness")
    assert "--model-draft" in noflash[0]
    assert h.margin(noflash[0]) >= FLOOR


def test_the_noflash_retry_reprices_the_reserve(tmp_path):
    """--flash-attn off pads the draft V to f16, so the reserve grows."""
    backend, gguf = h.cuda_backend(tmp_path)
    backend._estimate_mtp_overhead_bytes = lambda n_ctx, draft_cache_type_v = None, **kw: (
        8 * GIB + (1 if str(draft_cache_type_v or "f16").lower().startswith("q") else 3) * GIB
    )
    cmds = h.spawns(
        backend,
        gguf,
        crash = _flash_crash,
        healthy = _flash_off,
        n_ctx = 4096,
        speculative_type = "dspark",
        dspark_draft_path = h.sidecar(tmp_path),
        extra_args = ("--cache-type-k-draft", "q4_0", "--cache-type-v-draft", "q4_0"),
    )
    noflash = [c for c in cmds if h.last(c, "--flash-attn") == "off"]
    if not noflash or "--model-draft" not in noflash[0]:
        pytest.skip("no --flash-attn off retry with the drafter in this harness")
    assert h.margin(noflash[0]) >= h.margin(cmds[0]) + 2 * 1024


def test_noflash_reprice_leaves_target_rollback_to_the_fitter(tmp_path):
    backend, gguf = _rollback_backend(tmp_path)
    cmds = h.spawns(
        backend,
        gguf,
        crash = _flash_crash,
        healthy = _flash_off,
        n_ctx = 4096,
        speculative_type = "dspark",
        dspark_draft_path = h.sidecar(tmp_path),
    )
    noflash = [c for c in cmds if h.last(c, "--flash-attn") == "off" and "--model-draft" in c]
    if not noflash:
        pytest.skip("no --flash-attn off retry with the drafter in this harness")
    assert h.margin(noflash[0]) < FLOOR + _ROLLBACK // MIB


def test_a_retry_added_reserve_leaves_with_the_drafter(tmp_path):
    backend, gguf = h.cuda_backend(tmp_path)
    backend._select_gpus = lambda *a, **k: ([0], False)  # planned full offload: --fit off
    cmds = h.spawns(
        backend,
        gguf,
        # flash on: signal crash (fit-on retry, then the --flash-attn off retry carries the
        # reserve forward); flash off with the drafter: plain exit, so the drafter goes.
        crash = lambda i, cmd: (
            -11 if h.last(cmd, "--flash-attn") == "on" else (1 if "--model-draft" in cmd else None)
        ),
        healthy = lambda i, cmd: "--model-draft" not in cmd,
        n_ctx = 4096,
        speculative_type = "dspark",
        dspark_draft_path = h.sidecar(tmp_path),
    )
    reserved = [c for c in cmds if "--model-draft" in c and h.margin(c) >= FLOOR]
    if not reserved or "--model-draft" in cmds[-1]:
        pytest.skip("no reserved retry followed by a drafter-free launch in this harness")
    assert h.margin(cmds[-1]) < FLOOR


_REFUSED = ("a unified KV cache is only supported with a single sequence",)


def _one_slot_retry(backend, gguf, tmp_path, **kw):
    return h.spawns(
        backend,
        gguf,
        stdout = lambda i, cmd: _REFUSED if i == 0 else (),
        healthy = lambda i, cmd: i > 0,
        crash = lambda i, cmd: None if i > 0 else 1,
        n_ctx = 4096,
        n_parallel = 4,
        speculative_type = "dspark",
        dspark_draft_path = h.sidecar(tmp_path),
        **kw,
    )


def test_the_one_slot_retry_reprices_the_reserve(tmp_path):
    backend, gguf = h.cuda_backend(tmp_path)
    cmds = _one_slot_retry(h.slot_priced(backend), gguf, tmp_path)
    assert len(cmds) >= 2, [c[-6:] for c in cmds]
    first, retry = cmds[0], cmds[1]
    assert h.last(first, "--parallel") == "4"
    assert h.last(retry, "--parallel") in (None, "1")
    assert "--model-draft" in retry
    assert h.margin(first) >= FLOOR + 4 * 1024
    # Still reserved for the drafter and its one slot, not for the three slots dropped.
    assert FLOOR + 1024 <= h.margin(retry) < FLOOR + 2 * 1024
    assert retry.count("--fit-target") <= first.count("--fit-target")


def test_the_one_slot_retry_keeps_a_user_margin_equal_to_the_reserve(tmp_path):
    """A user --fit-target equal to Studio's reserved margin stays theirs: the retry drops
    only Studio's trailing pair and reprices it for the one slot left."""
    (tmp_path / "plain").mkdir()
    backend, gguf = h.cuda_backend(tmp_path / "plain")
    plain = h.spawns(
        h.slot_priced(backend),
        gguf,
        healthy = lambda i, cmd: True,
        n_ctx = 4096,
        n_parallel = 4,
        speculative_type = "dspark",
        dspark_draft_path = h.sidecar(tmp_path),
    )[0]
    reserved = str(h.margin(plain))
    backend, gguf = h.cuda_backend(tmp_path)
    cmds = _one_slot_retry(
        h.slot_priced(backend),
        gguf,
        tmp_path,
        extra_args = ["--fit-target", reserved, "--fit-target", "1024"],
    )
    assert len(cmds) >= 2, [c[-6:] for c in cmds]
    retry = h.fit_targets(cmds[1])
    assert retry[-3:-1] == [reserved, "1024"], retry
    assert FLOOR + 1024 <= int(retry[-1]) < FLOOR + 2 * 1024, retry


# _drop_sidecar_margins: the fallback's cleanup takes back exactly the pairs Studio added.


@pytest.mark.parametrize(
    "cmd, integrity, rebuilt, pairs, want",
    [
        # Single-sequence recovery already dropped --kv-unified and set --parallel 1.
        (
            ["--parallel", "1", "--fit", "on", "--fit-ctx", "4096", "--fit-target", "9252"],
            ["--kv-unified", "--fit-ctx", "4096", "--fit-target", "9252"],
            ["--kv-unified", "--fit-ctx", "4096", "--fit-target", "1024"],
            [],
            ["--parallel", "1", "--fit", "on", "--fit-ctx", "4096", "--fit-target", "1024"],
        ),
        (
            ["--fit", "on", "--fit-target", "9225", "--fit-target", "1024", "--fit-target", "9225"],
            [],
            [],
            [["--fit-target", "9225"]],
            ["--fit", "on", "--fit-target", "9225", "--fit-target", "1024"],
        ),
        (
            ["--fit", "on", "--fit-target", "9225", "--fit-target", "11273"],
            [],
            [],
            [["--fit-target", "9225"], ["--fit-target", "11273"]],
            ["--fit", "on"],
        ),
    ],
    ids = [
        "test_cleanup_swaps_only_the_margin_pair_after_a_rewrite",
        "test_cleanup_removes_studios_trailing_pair_not_the_users",
        "test_cleanup_removes_every_recorded_pair",
    ],
)
def test_drop_sidecar_margins(cmd, integrity, rebuilt, pairs, want):
    assert lc._drop_sidecar_margins(cmd, integrity, rebuilt, pairs) == want


def test_a_repeated_cleanup_keeps_the_users_identical_margin():
    integrity, rebuilt, user = (
        ["--fit-target", "9449"],
        ["--fit-target", "1024"],
        ["--fit-target", "9449"],
    )
    # Studio's copy is already gone; only the user's remains.
    cmd = ["llama-server", "-m", "m.gguf", "--fit-target", "1024", *user]
    assert lc._drop_sidecar_margins(cmd, integrity, rebuilt, [], user_args = user) == cmd
    cmd = ["llama-server", "--fit-target", "9449", *user]
    assert lc._drop_sidecar_margins(cmd, integrity, rebuilt, [], user_args = user) == [
        "llama-server",
        "--fit-target",
        "1024",
        *user,
    ]
    # A trailing pair a launch appended goes; the user's identical earlier one stays.
    pair = ["--fit-target", "17874"]
    assert lc._drop_sidecar_margins(
        ["llama-server", *pair, *pair], [], [], [pair], user_args = pair
    ) == [
        "llama-server",
        *pair,
    ]
    assert lc._drop_sidecar_margins(["llama-server", *pair], [], [], [pair], user_args = pair) == [
        "llama-server",
        *pair,
    ]
