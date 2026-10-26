# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lookup-only embedding tables (per-layer embeddings, the DeepSeek-V4.1 Engram table) in the launch fit, the offload layout and the planner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
_TESTS_DIR = str(Path(__file__).resolve().parent)
for _d in (_BACKEND_DIR, _TESTS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from test_gpu_arch_gate_os_matrix_7624 import probe_env  # noqa: F401, I001 -- fixture by name
from test_llama_cpp_placement import (  # noqa: F401, I001 -- fixtures by name
    _cache_launch,
    _discrete_linux_host,
    _moe_cache_host,
)
from test_offload_planner import _StubReader, _StubTensor, _shard_fields, _shard_tensors

from core.inference import llama_cpp as llama_cpp_module  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
from core.inference.offload_layout import (  # noqa: E402
    BlockLayout,
    ModelLayout,
    _layout_from_reader,
)
from core.inference.offload_planner import PlanOptions, _finish  # noqa: E402


def _host_pinned_input_bytes(*args):
    return llama_cpp_module._host_pinned_input_bytes(*args)


def _override_places_tensor_on_device(*args):
    return llama_cpp_module._override_places_tensor_on_device(*args)


GIB = 1024**3
PLE = "per_layer_token_embd.weight"

DSV41_TOTAL = int(216.98 * GIB)
DSV41_ENGRAM = int(103.00 * GIB)
DSV41_TOKEN_EMBD = int(0.51 * GIB)
DSV41_OUTPUT = int(0.51 * GIB)

FLASH_NEXT_TOTAL = int(73.45 * GIB)
FLASH_NEXT_PLE = int(26.82 * GIB)
FLASH_NEXT_TOKEN_EMBD = int(0.41 * GIB)
FLASH_NEXT_OUTPUT = int(0.33 * GIB)

GEMMA3N_TOTAL = int(4.23 * GIB)
GEMMA3N_PLE = int(1.64 * GIB)
GEMMA3N_TOKEN_EMBD = int(0.28 * GIB)

_CAPS = {"supports_lazy_mode": True}


def _scan_backend(arch, named):
    backend = LlamaCppBackend()
    backend._gguf_tensor_scan = lambda _path: (arch, dict(named), 0)
    return backend


@pytest.mark.parametrize(
    "extra_args, auto_off, expected",
    [
        ([], False, DSV41_ENGRAM),
        ([], True, 0),
        (["-lzm", "off"], False, 0),
        (["-lzm", "on"], True, DSV41_ENGRAM),
    ],
    ids = ["auto", "igpu_auto", "off", "on_igpu"],
)
def test_the_deepseek41_engram_table_is_read_lazily(extra_args, auto_off, expected):
    backend = _scan_backend("deepseek41", {PLE: DSV41_ENGRAM})
    got = backend._lazy_read_host_bytes(
        "/m.gguf", caps = _CAPS, extra_args = extra_args, env = {}, auto_resolves_off = auto_off
    )
    assert got == expected


def test_the_dsv41_host_preflight_no_longer_charges_the_engram_table(monkeypatch):
    """DSV4.1 UD-IQ1_S on a 24 GiB card with 64 GiB of RAM."""
    backend = _scan_backend("deepseek41", {PLE: DSV41_ENGRAM})
    backend._get_gguf_size_bytes = lambda _path: DSV41_TOTAL
    backend.probe_server_capabilities = lambda _binary = None: dict(_CAPS)
    backend._is_vulkan_backend = lambda _binary = None: False
    monkeypatch.setattr(LlamaCppBackend, "_lazy_auto_resolves_off", lambda self, **_kw: False)
    monkeypatch.delenv("UNSLOTH_ALLOW_HOST_OFFLOAD", raising = False)
    seen = {}

    def _message(offload_bytes, avail_mib):
        seen["offload"] = offload_bytes
        return "short"

    backend._host_offload_shortfall_message = _message
    msg = backend._launch_host_shortfall_message(
        ["/fake/llama-server", "-m", "/m.gguf"],
        [(0, 23_800)],
        env = {},
        avail_mib = 64 * 1024,
    )
    assert msg == "short"
    expected = DSV41_TOTAL - DSV41_ENGRAM - 23_800 * 1024 * 1024
    assert seen["offload"] == expected
    assert 90 * GIB < seen["offload"] < 91 * GIB


def _gemma3n_layout(with_output = False):
    tensors = _shard_tensors(range(4)) + [
        _StubTensor("token_embd.weight", GEMMA3N_TOKEN_EMBD),
        _StubTensor(PLE, GEMMA3N_PLE),
        _StubTensor("output_norm.weight", 4096),
    ]
    if with_output:
        tensors.append(_StubTensor("output.weight", GEMMA3N_TOKEN_EMBD))
    return _layout_from_reader(_StubReader(_shard_fields(**{"llama.block_count": 4}), tensors))


def test_the_per_layer_table_is_not_token_embd():
    layout = _gemma3n_layout()
    assert layout.complete
    assert layout.token_embd_bytes == GEMMA3N_TOKEN_EMBD
    assert layout.per_layer_embd_bytes == GEMMA3N_PLE


def test_a_tied_file_duplicates_token_embd_but_not_the_per_layer_table():
    """gemma-3n is tied."""
    layout = _gemma3n_layout()
    assert layout.other_resident_bytes == GEMMA3N_TOKEN_EMBD + 4096
    untied = _gemma3n_layout(with_output = True)
    assert untied.other_resident_bytes == 4096
    assert untied.lm_head_bytes == GEMMA3N_TOKEN_EMBD


def _flash_next_layout():
    blocks = tuple(
        BlockLayout(index = i, spillable_bytes = GIB, resident_bytes = int(0.14 * GIB)) for i in range(40)
    )
    return ModelLayout(
        arch = "qwen4exp",
        n_layers = 40,
        n_attention_layers = 10,
        blocks = blocks,
        lm_head_bytes = FLASH_NEXT_OUTPUT,
        token_embd_bytes = FLASH_NEXT_TOKEN_EMBD,
        per_layer_embd_bytes = FLASH_NEXT_PLE,
        tensor_bytes = FLASH_NEXT_TOTAL,
        other_resident_bytes = int(0.29 * GIB),
        kv_bytes_per_token_f16 = 40 * 1024,
        n_ctx_train = 262144,
        is_moe = True,
        n_expert = 256,
        n_expert_used = 8,
        complete = True,
    )


def test_a_lazy_table_does_not_deny_load_mode_none_to_spilled_experts():
    """Flash-Next on 24 GiB + a 64 GB box with 48 GiB available."""
    layout = _flash_next_layout()
    chosen = list(layout.blocks[:25])
    lazy = _finish(
        layout, PlanOptions(lazy_read_bytes = FLASH_NEXT_PLE), 8192, chosen, False, 48 * GIB
    )
    assert lazy.load_mode_none
    assert lazy.host_bytes == FLASH_NEXT_TOKEN_EMBD + 25 * GIB
    resident = _finish(layout, PlanOptions(), 8192, chosen, False, 48 * GIB)
    assert not resident.load_mode_none
    assert resident.host_bytes == FLASH_NEXT_TOKEN_EMBD + FLASH_NEXT_PLE + 25 * GIB


def test_host_pinned_inputs_for_the_three_families():
    assert (
        _host_pinned_input_bytes(
            ("gemma3n", {PLE: GEMMA3N_PLE, "token_embd.weight": GEMMA3N_TOKEN_EMBD}, 0)
        )
        == GEMMA3N_PLE
    )
    assert (
        _host_pinned_input_bytes(
            (
                "qwen4exp",
                {
                    PLE: FLASH_NEXT_PLE,
                    "token_embd.weight": FLASH_NEXT_TOKEN_EMBD,
                    "output.weight": FLASH_NEXT_OUTPUT,
                },
                0,
            )
        )
        == FLASH_NEXT_PLE + FLASH_NEXT_TOKEN_EMBD
    )
    assert (
        _host_pinned_input_bytes(
            (
                "deepseek41",
                {
                    PLE: DSV41_ENGRAM,
                    "token_embd.weight": DSV41_TOKEN_EMBD,
                    "output.weight": DSV41_OUTPUT,
                },
                0,
            )
        )
        == DSV41_ENGRAM + DSV41_TOKEN_EMBD
    )
    assert _host_pinned_input_bytes(None) == 0


def test_an_override_onto_a_device_keeps_a_resident_table_charged():
    ot = ["-ot", "per_layer_token_embd=CUDA0"]
    assert _override_places_tensor_on_device(PLE, ot)
    assert not _override_places_tensor_on_device(PLE, ["-ot", "per_layer=CPU"])
    assert not _override_places_tensor_on_device(PLE, ["-ot", "per_layer=CPU,embd=CUDA0"])
    assert not _override_places_tensor_on_device(PLE, ["-ot", "exps=CPU"])
    assert _override_places_tensor_on_device(PLE, [], {"LLAMA_ARG_OVERRIDE_TENSOR": "embd=CUDA0"})
    assert _host_pinned_input_bytes(("qwen4exp", {PLE: FLASH_NEXT_PLE}, 0), ot, {}) == 0
    # Upstream b11870 gathers the Engram rows in the graph, so -ot places it like any other table;
    # a build that keeps it on the host whatever -ot says still discounts it.
    engram = ("deepseek41", {PLE: DSV41_ENGRAM}, 0)
    assert _host_pinned_input_bytes(engram, ot, {}) == 0
    assert (
        llama_cpp_module._host_pinned_input_bytes(engram, ot, {}, host_forced = {"deepseek41"})
        == DSV41_ENGRAM
    )


def _gemma3n_named():
    return {PLE: GEMMA3N_PLE, "token_embd.weight": GEMMA3N_TOKEN_EMBD}


def _fit_flags(cmd):
    i = cmd.index("--fit")
    return cmd[i + 1]


def test_gemma3n_fits_a_small_card_once_its_tables_leave_the_vram_charge(tmp_path, _moe_cache_host):
    """gemma-3n-E4B Q4_K_M (4.23 GiB)."""
    _backend, cmd = _cache_launch(
        tmp_path,
        moe = False,
        size_gib = GEMMA3N_TOTAL / GIB,
        memory = [(0, 9_000, 10_240)],
        lazy = _gemma3n_named(),
        arch = "gemma3n",
        ram_gib = 16,
    )
    assert _fit_flags(cmd) == "off", cmd
    assert cmd[cmd.index("-ngl") + 1] == "-1", cmd
    assert "--lazy-mode" not in cmd and "-lzm" not in cmd, cmd


def test_an_override_moving_the_table_to_the_gpu_keeps_it_charged(tmp_path, _moe_cache_host):
    _backend, cmd = _cache_launch(
        tmp_path,
        moe = False,
        size_gib = GEMMA3N_TOTAL / GIB,
        memory = [(0, 9_000, 10_240)],
        lazy = _gemma3n_named(),
        arch = "gemma3n",
        ram_gib = 16,
        extra_args = ["-ot", "per_layer_token_embd=CUDA0"],
    )
    assert cmd[cmd.index("--fit") + 1] == "on", cmd


@pytest.mark.parametrize("ram_gib, none_ok", [(2, False), (16, True)])
def test_resident_input_tables_are_host_ram_in_the_load_mode_fit(
    tmp_path, _moe_cache_host, ram_gib, none_ok
):
    """gemma3n's table is resident (never lazily read) and host-pinned."""
    _backend, cmd = _cache_launch(
        tmp_path,
        moe = False,
        size_gib = GEMMA3N_TOTAL / GIB,
        memory = [(0, 23_000, 24_576)],
        lazy = _gemma3n_named(),
        arch = "gemma3n",
        ram_gib = ram_gib,
    )
    assert ("--load-mode" in cmd and cmd[cmd.index("--load-mode") + 1] == "none") is none_ok, cmd


def _lazy_values(cmd):
    return [cmd[i + 1] for i, tok in enumerate(cmd) if tok in ("--lazy-mode", "-lzm")]


def _flash_next_launch(tmp_path, *, ram_gib, **kwargs):
    return _cache_launch(
        tmp_path,
        size_gib = FLASH_NEXT_TOTAL / GIB,
        memory = [(0, 23_800, 24_576)],
        ram_gib = ram_gib,
        expert_gib = 40,
        lazy = {
            PLE: FLASH_NEXT_PLE,
            "token_embd.weight": FLASH_NEXT_TOKEN_EMBD,
            "output.weight": FLASH_NEXT_OUTPUT,
        },
        arch = "qwen4exp",
        **kwargs,
    )


def test_a_stale_spelling_from_another_build_is_not_the_users_choice(
    tmp_path, _moe_cache_host, monkeypatch
):
    """llama.cpp reads only its own spelling, so an older build's env twin left set does not stop the managed resident choice."""
    monkeypatch.setenv("LLAMA_ARG_TENSOR_READ_LAZY", "off")
    backend, cmd = _flash_next_launch(tmp_path, ram_gib = 128)
    assert _lazy_values(cmd) == ["off"], cmd


def test_flash_next_keeps_its_table_resident_where_ram_holds_it(tmp_path, _moe_cache_host):
    backend, cmd = _flash_next_launch(tmp_path, ram_gib = 128)
    assert _lazy_values(cmd) == ["off"], cmd
    assert backend._lazy_mode_flags == ["--lazy-mode", "off"]


def test_the_resident_table_is_charged_once_in_the_load_mode_fit(tmp_path, _moe_cache_host):
    """After the resident verdict the table moves from the pooled weights to host-only RAM."""
    _backend, cmd = _flash_next_launch(tmp_path, ram_gib = 72)
    assert _lazy_values(cmd) == ["off"], cmd
    assert cmd[cmd.index("--load-mode") + 1] == "none", cmd


def test_a_retry_hands_the_table_back_to_llama_cpp():
    backend = LlamaCppBackend()
    backend._lazy_mode_flags = ["--lazy-mode", "off"]
    argv = ["/fake/llama-server", "-m", "/m.gguf", "--lazy-mode", "off", "-c", "4096"]
    assert backend._drop_managed_lazy_mode(argv, "CPU fallback") == [
        "/fake/llama-server",
        "-m",
        "/m.gguf",
        "-c",
        "4096",
    ]
    backend._lazy_mode_flags = []
    assert backend._drop_managed_lazy_mode(argv, "CPU fallback") == argv


def test_a_device_override_keeps_the_table_streamed(tmp_path, _moe_cache_host):
    """Read lazily, the table sits on a CPU buffer whatever -ot says."""
    backend, cmd = _flash_next_launch(
        tmp_path, ram_gib = 128, extra_args = ["-ot", "per_layer_token_embd=CUDA0"]
    )
    assert _lazy_values(cmd) == [], cmd
    assert backend._lazy_mode_flags == []


def test_the_in_place_fit_on_retry_hands_the_table_back(tmp_path, _moe_cache_host, monkeypatch):
    """A full offload launches with --fit off."""
    import subprocess
    import test_llama_cpp_placement as placement

    cmds = []

    def launch(
        backend,
        gguf,
        model_identifier = "test",
        **kwargs,
    ):
        def fake_popen(cmd, **kw):
            if not cmd or str(cmd[0]) != "/fake/llama-server":
                return placement._REAL_POPEN(cmd, **kw)
            spawn = "-m" in cmd
            if spawn:
                cmds.append(list(cmd))
            rc = (1 if len(cmds) == 1 else None) if spawn else 0
            return type(
                "Process",
                (),
                {
                    "pid": 2**31 - 1,
                    "stdout": (),
                    "returncode": rc,
                    "poll": lambda self: rc,
                    "terminate": lambda self: None,
                    "wait": lambda self, timeout = None: rc or 0,
                    "kill": lambda self: None,
                },
            )()

        backend._wait_for_health = lambda timeout = None, **_kw: len(cmds) > 1
        with placement.patch.object(subprocess, "Popen", side_effect = fake_popen):
            backend.load_model(
                placement.GgufLoadIntent(
                    gguf_path = str(gguf), model_identifier = model_identifier, **kwargs
                )
            )
        return {"cmd": cmds[-1]}

    monkeypatch.setattr(placement, "_launch", launch)
    _cache_launch(
        tmp_path,
        size_gib = FLASH_NEXT_TOTAL / GIB,
        memory = [(0, 96_000, 98_304)],
        ram_gib = 128,
        expert_gib = 40,
        lazy = {
            PLE: FLASH_NEXT_PLE,
            "token_embd.weight": FLASH_NEXT_TOKEN_EMBD,
            "output.weight": FLASH_NEXT_OUTPUT,
        },
        arch = "qwen4exp",
    )
    assert len(cmds) == 2, cmds
    assert _lazy_values(cmds[0]) == ["off"], cmds[0]
    assert _lazy_values(cmds[1]) == [], cmds[1]


def test_flash_next_streams_its_table_on_a_64gb_box(tmp_path, _moe_cache_host):
    """24 GiB card + 64 GB RAM."""
    backend, cmd = _flash_next_launch(tmp_path, ram_gib = 58)
    assert _lazy_values(cmd) == [], cmd
    assert backend._lazy_mode_flags == []


@pytest.mark.parametrize(
    "extra_args, expected",
    [(["-lzm", "auto"], ["auto"]), (["--lazy-mode", "on"], ["on"]), (["-lzm", "off"], ["off"])],
    ids = ["user_auto", "user_on", "user_off"],
)
def test_a_users_lazy_mode_wins(tmp_path, _moe_cache_host, extra_args, expected):
    backend, cmd = _flash_next_launch(tmp_path, ram_gib = 128, extra_args = extra_args)
    assert _lazy_values(cmd) == expected, cmd
    assert backend._lazy_mode_flags == []


def test_an_inherited_lazy_mode_wins(tmp_path, _moe_cache_host, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_LAZY_MODE", "on")
    backend, cmd = _flash_next_launch(tmp_path, ram_gib = 128)
    assert _lazy_values(cmd) == [], cmd


def test_an_igpu_child_is_left_to_llama_cpp(tmp_path, _moe_cache_host, monkeypatch):
    monkeypatch.setattr(LlamaCppBackend, "_lazy_auto_resolves_off", lambda self, **_kw: True)
    backend, cmd = _flash_next_launch(tmp_path, ram_gib = 128)
    assert _lazy_values(cmd) == [], cmd


def test_a_build_without_lazy_mode_gets_no_flag(tmp_path, _moe_cache_host):
    from test_llama_cpp_placement import _CACHE_CAPS_ON

    caps = dict(_CACHE_CAPS_ON, supports_lazy_mode = False)
    backend, cmd = _flash_next_launch(tmp_path, ram_gib = 128, caps = caps)
    assert _lazy_values(cmd) == [], cmd


def test_dsv41_streams_its_engram_table_on_24gb_plus_64gb(tmp_path, _moe_cache_host):
    _backend, cmd = _cache_launch(
        tmp_path,
        size_gib = DSV41_TOTAL / GIB,
        memory = [(0, 23_800, 24_576)],
        ram_gib = 58,
        expert_gib = 100,
        lazy = {
            PLE: DSV41_ENGRAM,
            "token_embd.weight": DSV41_TOKEN_EMBD,
            "output.weight": DSV41_OUTPUT,
        },
        arch = "deepseek41",
    )
    assert _lazy_values(cmd) == [], cmd


def test_the_decision_is_logged_once(monkeypatch):
    seen = []
    monkeypatch.setattr(llama_cpp_module.logger, "info", lambda msg, *a, **k: seen.append(str(msg)))
    backend = _scan_backend("qwen4exp", {PLE: FLASH_NEXT_PLE})
    for _ in range(3):
        flags = backend._managed_lazy_mode_flags(
            "/m.gguf", caps = _CAPS, extra_args = [], env = {}, resident_fits = lambda _t: True
        )
        assert flags == ["--lazy-mode", "off"]
    assert sum("Lookup tables: keeping" in m for m in seen) == 1


def test_an_unpriced_fit_keeps_llama_cpp_default():
    backend = _scan_backend("qwen4exp", {PLE: FLASH_NEXT_PLE})

    def boom(_table):
        raise RuntimeError("unpriced")

    for verdict in (lambda _t: None, lambda _t: False, boom):
        assert (
            backend._managed_lazy_mode_flags(
                "/m.gguf", caps = _CAPS, extra_args = [], env = {}, resident_fits = verdict
            )
            == []
        )
    old = {"supports_lazy_mode": True, "lazy_mode_flag": "--tensor-read-lazy"}
    assert backend._managed_lazy_mode_flags(
        "/m.gguf", caps = old, extra_args = [], env = {}, resident_fits = lambda _t: True
    ) == ["--tensor-read-lazy", "off"]


def _apple_backend(arch, named, avail_gib):
    backend = _scan_backend(arch, named)
    backend._available_system_memory_mib = lambda: int(avail_gib * 1024)
    backend._can_estimate_kv = lambda: True
    return backend


def _apple_discount(
    backend,
    *,
    probe,
    model_size_fit,
    budget_gib,
    need_at_ctx = GIB,
):
    return backend._apple_lookup_table_discount(
        probe,
        "/m.gguf",
        caps = dict(_CAPS, supports_cache_ram = True),
        extra_args = [],
        model_size_fit = model_size_fit,
        need_at_ctx = need_at_ctx,
        budget_mib = int(budget_gib * 1024),
        cache_ram = None,
        env = {},
    )


def test_a_failed_metal_probe_still_leaves_the_engram_table_out():
    """DSV4.1 UD-IQ1_S on a 128 GB Mac."""
    backend = _apple_backend("deepseek41", {PLE: DSV41_ENGRAM}, avail_gib = 119)
    got = _apple_discount(backend, probe = 0, model_size_fit = DSV41_TOTAL, budget_gib = 96)
    assert got == DSV41_ENGRAM
    assert backend._apple_lazy_flags == []
    assert (DSV41_TOTAL - got) / GIB == pytest.approx(113.98, abs = 0.01)


def test_a_successful_probe_keeps_its_larger_measured_discount():
    backend = _apple_backend("deepseek41", {PLE: DSV41_ENGRAM}, avail_gib = 119)
    measured = DSV41_ENGRAM + DSV41_TOKEN_EMBD
    assert (
        _apple_discount(backend, probe = measured, model_size_fit = DSV41_TOTAL, budget_gib = 96)
        == measured
    )


def test_flash_next_goes_resident_on_a_128gb_mac_and_stays_charged():
    backend = _apple_backend("qwen4exp", {PLE: FLASH_NEXT_PLE}, avail_gib = 119)
    got = _apple_discount(backend, probe = 0, model_size_fit = FLASH_NEXT_TOTAL, budget_gib = 96)
    assert backend._apple_lazy_flags == ["--lazy-mode", "off"]
    assert got == 0


def test_an_unsized_kv_cache_proves_nothing_on_a_mac():
    backend = _apple_backend("qwen4exp", {PLE: FLASH_NEXT_PLE}, avail_gib = 119)
    backend._can_estimate_kv = lambda: False
    got = _apple_discount(backend, probe = 0, model_size_fit = FLASH_NEXT_TOTAL, budget_gib = 96)
    assert backend._apple_lazy_flags == []
    assert got == FLASH_NEXT_PLE


def test_a_users_lazy_off_on_a_mac_takes_the_table_back_out_of_the_probe_discount():
    backend = _apple_backend("qwen4exp", {PLE: FLASH_NEXT_PLE}, avail_gib = 119)
    got = backend._apple_lookup_table_discount(
        FLASH_NEXT_PLE + FLASH_NEXT_TOKEN_EMBD,
        "/m.gguf",
        caps = dict(_CAPS),
        extra_args = ["-lzm", "off"],
        model_size_fit = FLASH_NEXT_TOTAL,
        need_at_ctx = GIB,
        budget_mib = 96 * 1024,
        cache_ram = None,
        env = {},
    )
    assert got == FLASH_NEXT_TOKEN_EMBD
    assert backend._apple_lazy_flags == []


def test_flash_next_streams_on_a_64gb_mac():
    backend = _apple_backend("qwen4exp", {PLE: FLASH_NEXT_PLE}, avail_gib = 58)
    got = _apple_discount(backend, probe = 0, model_size_fit = FLASH_NEXT_TOTAL, budget_gib = 48)
    assert backend._apple_lazy_flags == []
    assert got == FLASH_NEXT_PLE


def test_gemma3n_gets_no_metal_table_discount():
    """Not TENSOR_READ_LAZY upstream and under 4 GiB: only the probe can discount it."""
    backend = _apple_backend("gemma3n", _gemma3n_named(), avail_gib = 12)
    assert _apple_discount(backend, probe = 0, model_size_fit = GEMMA3N_TOTAL, budget_gib = 10) == 0
    assert backend._apple_lazy_flags == []


@pytest.mark.parametrize(
    "lazy_args, expected", [([], FLASH_NEXT_PLE), (["-lzm", "off"], 0)], ids = ["auto", "off"]
)
def test_a_metal_override_moves_the_table_only_when_it_is_not_streamed(lazy_args, expected):
    """llama.cpp picks the lazy buffer before -ot, so a streamed table stays off Metal."""
    backend = _apple_backend("qwen4exp", {PLE: FLASH_NEXT_PLE}, avail_gib = 58)
    got = backend._apple_lookup_table_discount(
        0,
        "/m.gguf",
        caps = dict(_CAPS),
        extra_args = ["-ot", "per_layer_token_embd=MTL0", *lazy_args],
        model_size_fit = FLASH_NEXT_TOTAL,
        need_at_ctx = GIB,
        budget_mib = 48 * 1024,
        cache_ram = None,
        env = {},
    )
    assert got == expected


def test_module_table_names_deepseek41():
    assert "deepseek41" in llama_cpp_module._LAZY_READ_TENSORS


@pytest.mark.parametrize("name", ["LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL"])
def test_an_inherited_projector_on_a_mac_keeps_the_table_streamed(name):
    """The projector the env names is unpriced in the Metal footprint, so the 128 GB Mac that would read Flash-Next's table resident leaves it to lazy auto instead."""
    backend = _apple_backend("qwen4exp", {PLE: FLASH_NEXT_PLE}, avail_gib = 119)
    got = backend._apple_lookup_table_discount(
        0,
        "/m.gguf",
        caps = dict(_CAPS),
        extra_args = [],
        model_size_fit = FLASH_NEXT_TOTAL,
        need_at_ctx = GIB,
        budget_mib = 96 * 1024,
        cache_ram = None,
        env = {name: "/p/mmproj.gguf"},
    )
    assert backend._apple_lazy_flags == []
    assert got == FLASH_NEXT_PLE


_RPC_LIST = ["--rpc", "10.0.0.2:50052", "--device", "Vulkan0,RPC0"]
_ONE = _RPC_LIST + ["--split-mode", "none"]


@pytest.mark.parametrize(
    "extra_args, env",
    [
        (_RPC_LIST, {}),
        (["--rpc", "10.0.0.2:50052"], {}),
        ([], {"LLAMA_ARG_RPC": "10.0.0.2:50052"}),
        (["--rpc", "10.0.0.2:50052", "--device", "RPC0"], {}),
        (["--rpc", "10.0.0.2:50052", "--device", "Vulkan0"], {}),
        ([], {}),
        (_ONE + ["--main-gpu", "0"], {}),
        (_ONE + ["--main-gpu", "1"], {}),
        (_ONE + ["--main-gpu", "-1"], {}),
        (_RPC_LIST, {"LLAMA_ARG_SPLIT_MODE": "none", "LLAMA_ARG_MAIN_GPU": "0"}),
        (["--rpc", "10.0.0.2:50052", "--split-mode", "none", "--main-gpu", "-1"], {}),
        (["--rpc", "10.0.0.2:50052", "--split-mode", "none", "--main-gpu", "0"], {}),
    ],
)
def test_an_rpc_model_device_keeps_lazy_auto_on(extra_args, env):
    """RPC devices report mmap support (ggml-rpc.cpp device_get_props in the shipped
    b11734-mix-c208e5f and upstream b11870), and llama.cpp turns auto off only for a
    device without it, so RPC alone never reads the table resident."""
    from core.inference.llama_cpp import LlamaCppBackend

    backend = LlamaCppBackend()
    got = backend._lazy_auto_resolves_off(
        gpu_indices = [0],
        detected_gpus = [(0, 24_000)],
        shared_gpu_ids = set(),
        is_vulkan_backend = True,
        probe_integrated_cuda = False,
        extra_args = extra_args,
        env = env,
    )
    assert got is False


def test_an_integrated_gpu_beside_an_rpc_device_still_turns_lazy_auto_off():
    from core.inference.llama_cpp import LlamaCppBackend
    got = LlamaCppBackend()._lazy_auto_resolves_off(
        gpu_indices = [0, 1],
        detected_gpus = [(0, 24_000), (1, 8_000)],
        shared_gpu_ids = {1},
        is_vulkan_backend = True,
        probe_integrated_cuda = False,
        extra_args = ["--rpc", "10.0.0.2:50052", "--device", "Vulkan1,RPC0"],
        env = {},
    )
    assert got is True


def test_a_streamed_table_stays_on_the_host_under_a_device_override():
    """llama.cpp picks the lazy buffer before -ot (llama-model-loader.cpp create_tensor)."""
    ot = ["-ot", "per_layer_token_embd=CUDA0"]
    scan = ("qwen4exp", {PLE: FLASH_NEXT_PLE}, 0)
    pinned = llama_cpp_module._host_pinned_input_bytes
    assert pinned(scan, ot, {}, lazy_mode = "auto") == FLASH_NEXT_PLE
    assert pinned(scan, ot, {}, lazy_mode = "off") == 0
    small = ("gemma4", {PLE: GEMMA3N_PLE}, 0)
    assert pinned(small, ot, {}, lazy_mode = "auto") == 0
    assert pinned(small, ot, {}, lazy_mode = "on") == GEMMA3N_PLE


def _flash_next_on_two_cards(tmp_path, extra_args):
    """Flash-Next (73.45 GiB, 26.82 GiB table) on 2x32 GiB."""
    return _cache_launch(
        tmp_path,
        moe = False,
        size_gib = FLASH_NEXT_TOTAL / GIB,
        memory = [(0, 32_000, 32_768), (1, 32_000, 32_768)],
        ram_gib = 128,
        lazy = {
            PLE: FLASH_NEXT_PLE,
            "token_embd.weight": FLASH_NEXT_TOKEN_EMBD,
            "output.weight": FLASH_NEXT_OUTPUT,
        },
        arch = "qwen4exp",
        extra_args = extra_args,
    )


@pytest.mark.parametrize(
    "lazy_args, fit",
    [([], "off"), (["-lzm", "on"], "off"), (["-lzm", "off"], "on")],
    ids = ["auto", "on", "off"],
)
def test_a_device_override_leaves_a_streamed_table_off_the_vram_charge(
    tmp_path, _moe_cache_host, lazy_args, fit
):
    _backend, cmd = _flash_next_on_two_cards(tmp_path, ["-ot", "embd=CUDA0", *lazy_args])
    assert _fit_flags(cmd) == fit, cmd
    if fit == "off":
        assert cmd[cmd.index("-ngl") + 1] == "-1", cmd


def test_a_streamed_table_under_an_override_keeps_the_resident_embedding_host_only(
    tmp_path, _moe_cache_host, monkeypatch
):
    """token_embd stays pinned on the host beside a streamed table the override names."""
    seen = []
    real = LlamaCppBackend._fit_derived_load_mode

    def spy(self, **kwargs):
        if kwargs.get("announce", True):
            seen.append(kwargs)
        return real(self, **kwargs)

    monkeypatch.setattr(LlamaCppBackend, "_fit_derived_load_mode", spy)
    _flash_next_on_two_cards(tmp_path, ["-ot", "per_layer_token_embd=CUDA0"])
    assert seen and seen[-1]["lazy_read_bytes"] == FLASH_NEXT_PLE
    assert seen[-1]["host_only_bytes"] >= FLASH_NEXT_TOKEN_EMBD, seen[-1]


@pytest.mark.parametrize(
    "forced, flags", [(frozenset(), []), (frozenset({"deepseek41"}), ["--lazy-mode", "off"])]
)
def test_a_device_override_on_deepseek41_follows_the_builds_placement(monkeypatch, forced, flags):
    """Where -ot moves the Engram table, -lzm off would make it GPU-resident while it is priced on
    the host; a build that keeps it on the host still gets the resident choice."""
    monkeypatch.setattr(
        LlamaCppBackend, "_host_forced_table_archs", staticmethod(lambda binary = None: forced)
    )
    backend = _scan_backend("deepseek41", {PLE: DSV41_ENGRAM})
    assert (
        backend._managed_lazy_mode_flags(
            "/m.gguf",
            caps = _CAPS,
            extra_args = ["-ot", "per_layer_token_embd=CUDA0"],
            env = {},
            resident_fits = lambda _t: True,
        )
        == flags
    )


@pytest.mark.parametrize("marked", [True, False])
def test_the_host_forced_probe_reads_the_builds_loader(tmp_path, marked):
    """Studio's prebuilt b11734-mix-c208e5f carries the loader warning; upstream b11870 does not."""
    server = tmp_path / "llama-server"
    server.write_bytes(b"\x7fELF server")
    marker = llama_cpp_module._HOST_FORCED_TABLE_MARKER if marked else b"place it anywhere"
    (tmp_path / "libllama.so.0.7.0").write_bytes(b"\x00" * 64 + marker + b" %s, using %s\n")
    got = LlamaCppBackend._host_forced_table_archs(str(server))
    assert got == (frozenset({"deepseek41"}) if marked else frozenset())
    assert (
        LlamaCppBackend._host_forced_table_archs(str(tmp_path / "gone" / "llama-server"))
        == frozenset()
    )


@pytest.mark.parametrize(
    "name, forced",
    [
        ("libllama.so.0.7.0", True),
        ("libllama.0.dylib", True),
        ("LLAMA.DLL", True),
        ("libllama.so.0.bak", False),
        ("llama.dll.disabled", False),
    ],
)
def test_the_host_forced_probe_reads_only_a_loadable_library(tmp_path, name, forced):
    server = tmp_path / "llama-server"
    server.write_bytes(b"\x7fELF server")
    (tmp_path / name).write_bytes(llama_cpp_module._HOST_FORCED_TABLE_MARKER)
    got = LlamaCppBackend._host_forced_table_archs(str(server))
    assert got == (frozenset({"deepseek41"}) if forced else frozenset())


def test_the_lazy_mode_choices_probe_the_launch_binary(monkeypatch):
    seen = []

    def probe(binary = None):
        seen.append(binary)
        return frozenset()

    monkeypatch.setattr(LlamaCppBackend, "_host_forced_table_archs", staticmethod(probe))
    backend = _apple_backend("deepseek41", {PLE: DSV41_ENGRAM}, avail_gib = 119)
    ot = ["-ot", "per_layer_token_embd=MTL0"]
    backend._managed_lazy_mode_flags(
        "/m.gguf",
        caps = _CAPS,
        extra_args = ot,
        env = {},
        resident_fits = lambda _t: True,
        binary = "/launch/llama-server",
    )
    backend._apple_lookup_table_discount(
        0,
        "/m.gguf",
        caps = dict(_CAPS),
        extra_args = ot,
        model_size_fit = DSV41_TOTAL,
        need_at_ctx = GIB,
        budget_mib = 96 * 1024,
        cache_ram = None,
        env = {},
        binary = "/launch/llama-server",
    )
    assert seen and set(seen) == {"/launch/llama-server"}


@pytest.mark.parametrize(
    "extra_args, resident",
    [([], True), (["--parallel", "1"], True), (["-c", "0"], False), (["--parallel", "8"], False)],
    ids = ["none", "same_parallel", "ctx_zero", "larger_parallel"],
)
def test_a_pass_through_that_reprices_the_cache_keeps_the_table_streamed(
    tmp_path, _moe_cache_host, extra_args, resident
):
    """-c 0 or a larger --parallel wins over the cache the resident verdict priced."""
    _backend, cmd = _flash_next_launch(tmp_path, ram_gib = 128, extra_args = extra_args)
    assert (_lazy_values(cmd) == ["off"]) is resident, cmd


@pytest.mark.parametrize(
    "extra_args, n_parallel, reprices",
    [
        ([], 1, False),
        (["-c", "0"], 1, True),
        (["--ctx-size", "8192"], 1, False),
        (["--parallel", "4"], 4, False),
        (["--parallel", "4"], 2, True),
        (["--parallel", "1"], 2, False),
    ],
)
def test_the_metal_resident_verdict_abstains_on_the_same_overrides(
    extra_args, n_parallel, reprices
):
    assert llama_cpp_module._pass_through_reprices_cache(extra_args, n_parallel) is reprices


@pytest.mark.parametrize("cpu_build", [False, True])
def test_a_cpu_child_keeps_llama_cpps_lazy_default(
    tmp_path, _moe_cache_host, monkeypatch, cpu_build
):
    """The resident verdict credits the cards; a CPU-only build or --device none has none."""
    monkeypatch.setattr(
        LlamaCppBackend, "_binary_ships_no_gpu_backend", staticmethod(lambda *a, **k: cpu_build)
    )
    _backend, cmd = _flash_next_launch(tmp_path, ram_gib = 128)
    assert (_lazy_values(cmd) == ["off"]) is not cpu_build, cmd
    _backend, cmd = _flash_next_launch(tmp_path, ram_gib = 128, extra_args = ["--device", "none"])
    assert _lazy_values(cmd) == [], cmd


def _metal_launched(tmp_path, monkeypatch, **kw):
    """A Metal load_model (16 GiB budget, 12 GiB weights): (None, None, argv)."""
    import test_metal_explicit_context_guard as metal

    captured = metal._launch(
        tmp_path,
        monkeypatch,
        n_ctx = 8192,
        kv_per_token = 32 * 1024,
        backend = metal._backend_with_embeddings(monkeypatch, embd = 0),
        real_fit = True,
        budget_bytes = 16 * GIB,
        weights_bytes = 12 * GIB,
        **kw,
    )
    return None, None, captured["cmd"]


def _metal_tokens(cmd):
    from test_metal_explicit_context_guard import TestInputEmbeddingsLeftInTheFileMapping
    return TestInputEmbeddingsLeftInTheFileMapping._fit_tokens(cmd)


@pytest.mark.parametrize(
    "extra_args, pinned",
    [
        (None, True),
        (["-ngl", "20"], False),
        (["--fit-target", "2048"], False),
        (["-ot", "blk=CPU"], False),
    ],
    ids = ["studio_owns", "user_layers", "user_fit_target", "user_cpu_override"],
)
def test_a_streamed_table_alone_never_pins_a_placement_the_user_owns(
    tmp_path, monkeypatch, extra_args, pinned
):
    """The table discount sizes the Metal fit; only Studio's own placement may be pinned on it."""
    monkeypatch.setattr(
        LlamaCppBackend,
        "_apple_lookup_table_discount",
        lambda self, probe, *a, **k: max(int(probe), 6 * GIB),
    )
    _pub, _ctx, cmd = _metal_launched(tmp_path, monkeypatch, extra_args = extra_args)
    assert (_metal_tokens(cmd)[:4] == ["-ngl", "-1", "--fit", "off"]) is pinned, cmd


def _spy_resident_inputs(monkeypatch):
    seen = []

    def spy(self, probe, *a, **k):
        seen.append(k)
        return int(probe)

    monkeypatch.setattr(LlamaCppBackend, "_apple_lookup_table_discount", spy)
    return seen


def test_the_metal_resident_verdict_prices_the_context_checkpoints(tmp_path, monkeypatch):
    needs = []
    for ckpt in (0, GIB):
        seen = _spy_resident_inputs(monkeypatch)
        _metal_launched(
            tmp_path, monkeypatch, ckpt_bytes = ckpt, extra_args = ["--ctx-checkpoints", "8"]
        )
        needs.append(seen[-1]["need_at_ctx"])
    assert needs[1] > needs[0], needs


@pytest.mark.parametrize("wired_gib, budget_gib", [(0, 16), (8, 8), (32, 16)])
def test_the_metal_resident_verdict_obeys_the_wired_headroom(
    tmp_path, monkeypatch, wired_gib, budget_gib
):
    seen = _spy_resident_inputs(monkeypatch)
    _metal_launched(tmp_path, monkeypatch, wired_bytes = wired_gib * GIB)
    assert seen[-1]["budget_mib"] == budget_gib * 1024


@pytest.mark.parametrize(
    "extra_args, cpu_forced",
    [(["--device", "none"], False), ([], True)],
    ids = ["device_none", "paravirtual"],
)
def test_a_cpu_child_on_a_mac_keeps_the_table_streamed(extra_args, cpu_forced):
    backend = _apple_backend("qwen4exp", {PLE: FLASH_NEXT_PLE}, avail_gib = 119)
    got = backend._apple_lookup_table_discount(
        0,
        "/m.gguf",
        caps = dict(_CAPS, supports_cache_ram = True),
        extra_args = extra_args,
        model_size_fit = FLASH_NEXT_TOTAL,
        need_at_ctx = GIB,
        budget_mib = 96 * 1024,
        cache_ram = None,
        env = {},
        cpu_forced = cpu_forced,
    )
    assert backend._apple_lazy_flags == []
    assert got == FLASH_NEXT_PLE


def test_a_paravirtual_launch_keeps_llama_cpps_lazy_default(tmp_path, _moe_cache_host, monkeypatch):
    monkeypatch.setattr(llama_cpp_module, "_metal_device_is_paravirtual", lambda: True)
    _backend, cmd = _flash_next_launch(tmp_path, ram_gib = 128)
    assert _lazy_values(cmd) == [], cmd


def test_the_metal_resident_verdict_abstains_on_an_unsized_mtp_draft(tmp_path, monkeypatch):
    """A flat MTP reserve prices no draft KV, so no resident verdict (as the discrete one)."""
    seen = _spy_resident_inputs(monkeypatch)
    _metal_launched(tmp_path, monkeypatch)
    assert seen[-1]["need_at_ctx"] is not None
    _metal_launched(tmp_path, monkeypatch, extra_args = ["--spec-type", "draft-mtp"])
    assert seen[-1]["need_at_ctx"] is None


def test_used_up_wired_headroom_leaves_no_resident_budget(tmp_path, monkeypatch):
    seen = _spy_resident_inputs(monkeypatch)
    _metal_launched(tmp_path, monkeypatch, wired_exhausted = True)
    assert seen[-1]["budget_mib"] == 0


def test_a_rocm_arch_gated_cpu_launch_is_a_cpu_child(tmp_path, monkeypatch, probe_env):
    """No installed kernel covers either card, so the child runs on the CPU."""
    import test_gpu_arch_gate_os_matrix_7624 as gate

    forced = []
    real = LlamaCppBackend._child_runs_on_cpu

    def spy(self, *a, **k):
        forced.append(bool(k.get("cpu_forced")))
        return real(self, *a, **k)

    monkeypatch.setattr(LlamaCppBackend, "_child_runs_on_cpu", spy)
    backend = LlamaCppBackend()
    backend._gguf_tensor_scan = lambda _p: ("qwen4exp", {PLE: FLASH_NEXT_PLE}, 0)
    gate._apply_os(monkeypatch, "linux", is_rocm = True)
    torch = gate._fake_torch(
        [gate._device("gfx1030", free_mib = 12049), gate._device("gfx1036", free_mib = 12176)],
        vendor = "amd",
    )
    gate._run_auto_load(
        monkeypatch,
        tmp_path,
        torch,
        ["gfx908"],
        returncode = None,
        backend = backend,
        server_caps = dict(_CAPS),
        intent_kwargs = {"gpu_memory_mode": "auto"},
    )
    assert backend._arch_gate_forced_cpu is True
    assert forced and all(forced), forced


@pytest.mark.parametrize("node", [False, True])
def test_an_empty_inventory_beside_a_gpu_prices_the_table_resident_at_launch(
    tmp_path, _moe_cache_host, monkeypatch, node
):
    """A failed probe may hide an iGPU, where lazy auto is off (as the listing prices it)."""
    seen = []
    real = LlamaCppBackend._fit_derived_load_mode

    def spy(self, **kwargs):
        if kwargs.get("announce", True):
            seen.append(kwargs)
        return real(self, **kwargs)

    monkeypatch.setattr(LlamaCppBackend, "_fit_derived_load_mode", spy)
    monkeypatch.setattr(llama_cpp_module, "_gpu_device_node_present", lambda: node)
    _cache_launch(
        tmp_path,
        size_gib = FLASH_NEXT_TOTAL / GIB,
        memory = [],
        ram_gib = 58,
        expert_gib = 40,
        lazy = {
            PLE: FLASH_NEXT_PLE,
            "token_embd.weight": FLASH_NEXT_TOKEN_EMBD,
            "output.weight": FLASH_NEXT_OUTPUT,
        },
        arch = "qwen4exp",
    )
    assert seen, "no load-mode fit ran"
    assert seen[-1]["lazy_read_bytes"] == (0 if node else FLASH_NEXT_PLE), seen[-1]
