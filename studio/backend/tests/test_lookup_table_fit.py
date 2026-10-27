# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lookup-only embedding tables (per-layer embeddings, the DeepSeek-V4.1 Engram table) in the launch fit, the offload layout and the planner."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
_TESTS_DIR = str(Path(__file__).resolve().parent)
for _d in (_BACKEND_DIR, _TESTS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import test_llama_cpp_placement as placement  # noqa: E402, I001
from test_gpu_arch_gate_os_matrix_7624 import probe_env  # noqa: F401, I001 -- fixture by name
from test_llama_cpp_placement import (  # noqa: F401, I001 -- fixtures by name
    _cache_launch,
    _discrete_linux_host,
    _moe_cache_host,
)
from test_offload_planner import _StubReader, _StubTensor, _shard_fields, _shard_tensors

from core.inference import llama_cpp as lc  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
from core.inference.offload_layout import (  # noqa: E402
    BlockLayout,
    ModelLayout,
    _layout_from_reader,
)
from core.inference.offload_planner import PlanOptions, _finish  # noqa: E402

GIB = 1024**3
PLE = "per_layer_token_embd.weight"
OT_TABLE = ["-ot", "per_layer_token_embd=CUDA0"]

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
_FLASH_NAMED = {
    PLE: FLASH_NEXT_PLE,
    "token_embd.weight": FLASH_NEXT_TOKEN_EMBD,
    "output.weight": FLASH_NEXT_OUTPUT,
}
_DSV41_NAMED = {
    PLE: DSV41_ENGRAM,
    "token_embd.weight": DSV41_TOKEN_EMBD,
    "output.weight": DSV41_OUTPUT,
}
_GEMMA3N_NAMED = {PLE: GEMMA3N_PLE, "token_embd.weight": GEMMA3N_TOKEN_EMBD}
_INHERITED_CPU_SPLIT = {"LLAMA_ARG_SPLIT_MODE": "none", "LLAMA_ARG_MAIN_GPU": "-1"}
_RPC_LIST = ["--rpc", "10.0.0.2:50052", "--device", "Vulkan0,RPC0"]
_ONE = _RPC_LIST + ["--split-mode", "none"]


def _scan_backend(arch, named):
    backend = LlamaCppBackend()
    backend._gguf_tensor_scan = lambda _path: (arch, dict(named), 0)
    return backend


def _arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def _lazy_values(cmd):
    return [cmd[i + 1] for i, tok in enumerate(cmd) if tok in ("--lazy-mode", "-lzm")]


def _setenv(monkeypatch, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)


def _full_caps():
    caps = dict(LlamaCppBackend.probe_server_capabilities.__func__(LlamaCppBackend, None))
    caps.update(placement._CACHE_CAPS_ON)
    return lambda _binary = None: caps


def _flash_next_launch(
    tmp_path,
    *,
    ram_gib = 128,
    memory = ((0, 23_800, 24_576),),
    **kwargs,
):
    return _cache_launch(
        tmp_path,
        size_gib = FLASH_NEXT_TOTAL / GIB,
        memory = list(memory),
        ram_gib = ram_gib,
        expert_gib = 40,
        lazy = _FLASH_NAMED,
        arch = "qwen4exp",
        **kwargs,
    )


def _flash_next_on_two_cards(
    tmp_path,
    extra_args = (),
    **kwargs,
):
    """Flash-Next (73.45 GiB, 26.82 GiB table) on 2x32 GiB."""
    return _cache_launch(
        tmp_path,
        moe = False,
        size_gib = FLASH_NEXT_TOTAL / GIB,
        memory = [(0, 32_000, 32_768), (1, 32_000, 32_768)],
        ram_gib = 128,
        lazy = _FLASH_NAMED,
        arch = "qwen4exp",
        extra_args = list(extra_args),
        **kwargs,
    )


def _gemma3n_launch(tmp_path, *, memory, ram_gib, **kwargs):
    return _cache_launch(
        tmp_path,
        moe = False,
        size_gib = GEMMA3N_TOTAL / GIB,
        memory = memory,
        lazy = _GEMMA3N_NAMED,
        arch = "gemma3n",
        ram_gib = ram_gib,
        **kwargs,
    )


def _spy(monkeypatch, name, record):
    """Wrap a LlamaCppBackend method; ``record(answer, args, kwargs)`` appends what it saw."""
    seen = []
    real = getattr(LlamaCppBackend, name)

    def spy(self, *a, **k):
        answer = real(self, *a, **k)
        entry = record(answer, a, k)
        if entry is not None:
            seen.append(entry)
        return answer

    monkeypatch.setattr(LlamaCppBackend, name, spy)
    return seen


def _spy_fit_load_mode(monkeypatch):
    return _spy(
        monkeypatch,
        "_fit_derived_load_mode",
        lambda _ans, _a, k: k if k.get("announce", True) else None,
    )


def _apple_backend(arch, named, avail_gib):
    backend = _scan_backend(arch, named)
    backend._available_system_memory_mib = lambda: int(avail_gib * 1024)
    backend._can_estimate_kv = lambda: True
    return backend


def _apple_discount(
    backend,
    probe = 0,
    *,
    model_size_fit = FLASH_NEXT_TOTAL,
    budget_gib = 96,
    caps = None,
    extra_args = (),
    env = None,
    **kwargs,
):
    return backend._apple_lookup_table_discount(
        probe,
        "/m.gguf",
        caps = dict(caps if caps is not None else dict(_CAPS, supports_cache_ram = True)),
        extra_args = list(extra_args),
        model_size_fit = model_size_fit,
        need_at_ctx = kwargs.pop("need_at_ctx", GIB),
        budget_mib = int(budget_gib * 1024),
        cache_ram = None,
        env = env or {},
        **kwargs,
    )


def _managed_flags(
    backend,
    extra_args = (),
    resident_fits = lambda _t: True,
    caps = _CAPS,
    **kwargs,
):
    return backend._managed_lazy_mode_flags(
        "/m.gguf",
        caps = caps,
        extra_args = list(extra_args),
        env = {},
        resident_fits = resident_fits,
        **kwargs,
    )


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
    seen = []
    backend._host_offload_shortfall_message = (
        lambda offload, _avail: seen.append(offload) or "short"
    )
    msg = backend._launch_host_shortfall_message(
        ["/fake/llama-server", "-m", "/m.gguf"], [(0, 23_800)], env = {}, avail_mib = 64 * 1024
    )
    assert msg == "short"
    assert seen == [DSV41_TOTAL - DSV41_ENGRAM - 23_800 * 1024 * 1024]
    assert 90 * GIB < seen[0] < 91 * GIB


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
    assert _gemma3n_layout().other_resident_bytes == GEMMA3N_TOKEN_EMBD + 4096
    untied = _gemma3n_layout(with_output = True)
    assert untied.other_resident_bytes == 4096
    assert untied.lm_head_bytes == GEMMA3N_TOKEN_EMBD


def test_a_lazy_table_does_not_deny_load_mode_none_to_spilled_experts():
    """Flash-Next on 24 GiB + a 64 GB box with 48 GiB available."""
    layout = ModelLayout(
        arch = "qwen4exp",
        n_layers = 40,
        n_attention_layers = 10,
        blocks = tuple(
            BlockLayout(index = i, spillable_bytes = GIB, resident_bytes = int(0.14 * GIB))
            for i in range(40)
        ),
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
    pinned = lc._host_pinned_input_bytes
    assert pinned(("gemma3n", _GEMMA3N_NAMED, 0)) == GEMMA3N_PLE
    assert pinned(("qwen4exp", _FLASH_NAMED, 0)) == FLASH_NEXT_PLE + FLASH_NEXT_TOKEN_EMBD
    assert pinned(("deepseek41", _DSV41_NAMED, 0)) == DSV41_ENGRAM + DSV41_TOKEN_EMBD
    assert pinned(None) == 0


def test_an_override_onto_a_device_keeps_a_resident_table_charged():
    on_device, pinned = lc._override_places_tensor_on_device, lc._host_pinned_input_bytes
    assert on_device(PLE, OT_TABLE)
    assert not on_device(PLE, ["-ot", "per_layer=CPU"])
    assert not on_device(PLE, ["-ot", "per_layer=CPU,embd=CUDA0"])
    assert not on_device(PLE, ["-ot", "exps=CPU"])
    assert on_device(PLE, [], {"LLAMA_ARG_OVERRIDE_TENSOR": "embd=CUDA0"})
    assert pinned(("qwen4exp", {PLE: FLASH_NEXT_PLE}, 0), OT_TABLE, {}) == 0
    # Upstream b11870 gathers the Engram rows in the graph, so -ot places it like any other table;
    # a build that keeps it on the host whatever -ot says still discounts it.
    engram = ("deepseek41", {PLE: DSV41_ENGRAM}, 0)
    assert pinned(engram, OT_TABLE, {}) == 0
    assert pinned(engram, OT_TABLE, {}, host_forced = {"deepseek41"}) == DSV41_ENGRAM


def test_gemma3n_fits_a_small_card_once_its_tables_leave_the_vram_charge(tmp_path, _moe_cache_host):
    """gemma-3n-E4B Q4_K_M (4.23 GiB)."""
    _backend, cmd = _gemma3n_launch(tmp_path, memory = [(0, 9_000, 10_240)], ram_gib = 16)
    assert (_arg(cmd, "--fit"), _arg(cmd, "-ngl")) == ("off", "-1"), cmd
    assert _lazy_values(cmd) == [], cmd


def test_an_override_moving_the_table_to_the_gpu_keeps_it_charged(tmp_path, _moe_cache_host):
    _backend, cmd = _gemma3n_launch(
        tmp_path, memory = [(0, 9_000, 10_240)], ram_gib = 16, extra_args = OT_TABLE
    )
    assert _arg(cmd, "--fit") == "on", cmd


@pytest.mark.parametrize("ram_gib, none_ok", [(2, False), (16, True)])
def test_resident_input_tables_are_host_ram_in_the_load_mode_fit(
    tmp_path, _moe_cache_host, ram_gib, none_ok
):
    """gemma3n's table is resident (never lazily read) and host-pinned."""
    _backend, cmd = _gemma3n_launch(tmp_path, memory = [(0, 23_000, 24_576)], ram_gib = ram_gib)
    assert ("--load-mode" in cmd and _arg(cmd, "--load-mode") == "none") is none_ok, cmd


@pytest.mark.parametrize(
    "ram_gib, lazy",
    [(128, ["off"]), (58, [])],
    ids = [
        "flash_next_keeps_its_table_resident_where_ram_holds_it",
        "flash_next_streams_its_table_on_a_64gb_box",
    ],
)
def test_flash_next_keeps_its_table_resident_where_ram_holds_it(
    tmp_path, _moe_cache_host, ram_gib, lazy
):
    """24 GiB card; 58 GiB of RAM is a 64 GB box."""
    backend, cmd = _flash_next_launch(tmp_path, ram_gib = ram_gib)
    assert _lazy_values(cmd) == lazy, cmd
    assert backend._lazy_mode_flags == (["--lazy-mode", "off"] if lazy else [])


def test_a_stale_spelling_from_another_build_is_not_the_users_choice(
    tmp_path, _moe_cache_host, monkeypatch
):
    """llama.cpp reads only its own spelling, so an older build's env twin is not a user choice."""
    monkeypatch.setenv("LLAMA_ARG_TENSOR_READ_LAZY", "off")
    assert _lazy_values(_flash_next_launch(tmp_path)[1]) == ["off"]


def test_the_resident_table_is_charged_once_in_the_load_mode_fit(tmp_path, _moe_cache_host):
    """After the resident verdict the table moves from the pooled weights to host-only RAM."""
    _backend, cmd = _flash_next_launch(tmp_path, ram_gib = 72)
    assert _lazy_values(cmd) == ["off"], cmd
    assert _arg(cmd, "--load-mode") == "none", cmd


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
    backend, cmd = _flash_next_launch(tmp_path, extra_args = OT_TABLE)
    assert _lazy_values(cmd) == [], cmd
    assert backend._lazy_mode_flags == []


def test_the_in_place_fit_on_retry_hands_the_table_back(tmp_path, _moe_cache_host, monkeypatch):
    """A full offload launches with --fit off; its failed first spawn retries without the flag."""
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
    _flash_next_launch(tmp_path, memory = [(0, 96_000, 98_304)])
    assert [_lazy_values(c) for c in cmds] == [["off"], []], cmds


@pytest.mark.parametrize(
    "extra_args, expected",
    [(["-lzm", "auto"], ["auto"]), (["--lazy-mode", "on"], ["on"]), (["-lzm", "off"], ["off"])],
    ids = ["user_auto", "user_on", "user_off"],
)
def test_a_users_lazy_mode_wins(tmp_path, _moe_cache_host, extra_args, expected):
    backend, cmd = _flash_next_launch(tmp_path, extra_args = extra_args)
    assert _lazy_values(cmd) == expected, cmd
    assert backend._lazy_mode_flags == []


def test_an_inherited_lazy_mode_wins(tmp_path, _moe_cache_host, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_LAZY_MODE", "on")
    assert _lazy_values(_flash_next_launch(tmp_path)[1]) == []


def test_an_igpu_child_is_left_to_llama_cpp(tmp_path, _moe_cache_host, monkeypatch):
    monkeypatch.setattr(LlamaCppBackend, "_lazy_auto_resolves_off", lambda self, **_kw: True)
    assert _lazy_values(_flash_next_launch(tmp_path)[1]) == []


def test_a_build_without_lazy_mode_gets_no_flag(tmp_path, _moe_cache_host):
    caps = dict(placement._CACHE_CAPS_ON, supports_lazy_mode = False)
    assert _lazy_values(_flash_next_launch(tmp_path, caps = caps)[1]) == []


def test_a_paravirtual_launch_keeps_llama_cpps_lazy_default(tmp_path, _moe_cache_host, monkeypatch):
    monkeypatch.setattr(lc, "_metal_device_is_paravirtual", lambda: True)
    assert _lazy_values(_flash_next_launch(tmp_path)[1]) == []


def test_dsv41_streams_its_engram_table_on_24gb_plus_64gb(tmp_path, _moe_cache_host):
    _backend, cmd = _cache_launch(
        tmp_path,
        size_gib = DSV41_TOTAL / GIB,
        memory = [(0, 23_800, 24_576)],
        ram_gib = 58,
        expert_gib = 100,
        lazy = _DSV41_NAMED,
        arch = "deepseek41",
    )
    assert _lazy_values(cmd) == [], cmd


def test_the_decision_is_logged_once(monkeypatch):
    seen = []
    monkeypatch.setattr(lc.logger, "info", lambda msg, *a, **k: seen.append(str(msg)))
    backend = _scan_backend("qwen4exp", {PLE: FLASH_NEXT_PLE})
    for _ in range(3):
        assert _managed_flags(backend) == ["--lazy-mode", "off"]
    assert sum("Lookup tables: keeping" in m for m in seen) == 1


def test_an_unpriced_fit_keeps_llama_cpp_default():
    backend = _scan_backend("qwen4exp", {PLE: FLASH_NEXT_PLE})

    def boom(_table):
        raise RuntimeError("unpriced")

    for verdict in (lambda _t: None, lambda _t: False, boom):
        assert _managed_flags(backend, resident_fits = verdict) == []
    old = {"supports_lazy_mode": True, "lazy_mode_flag": "--tensor-read-lazy"}
    assert _managed_flags(backend, caps = old) == ["--tensor-read-lazy", "off"]


@pytest.mark.parametrize(
    "probe, want",
    [(0, DSV41_ENGRAM), (DSV41_ENGRAM + DSV41_TOKEN_EMBD, DSV41_ENGRAM + DSV41_TOKEN_EMBD)],
    ids = [
        "a_failed_metal_probe_still_leaves_the_engram_table_out",
        "a_successful_probe_keeps_its_larger_measured_discount",
    ],
)
def test_a_failed_metal_probe_still_leaves_the_engram_table_out(probe, want):
    """DSV4.1 UD-IQ1_S on a 128 GB Mac."""
    backend = _apple_backend("deepseek41", {PLE: DSV41_ENGRAM}, avail_gib = 119)
    got = _apple_discount(backend, probe, model_size_fit = DSV41_TOTAL)
    assert got == want
    assert backend._apple_lazy_flags == []
    if not probe:
        assert (DSV41_TOTAL - got) / GIB == pytest.approx(113.98, abs = 0.01)


@pytest.mark.parametrize(
    "avail_gib, budget_gib, flags, want",
    [(119, 96, ["--lazy-mode", "off"], 0), (58, 48, [], FLASH_NEXT_PLE)],
    ids = [
        "flash_next_goes_resident_on_a_128gb_mac_and_stays_charged",
        "flash_next_streams_on_a_64gb_mac",
    ],
)
def test_flash_next_goes_resident_on_a_128gb_mac_and_stays_charged(
    avail_gib, budget_gib, flags, want
):
    backend = _apple_backend("qwen4exp", {PLE: FLASH_NEXT_PLE}, avail_gib = avail_gib)
    assert _apple_discount(backend, budget_gib = budget_gib) == want
    assert backend._apple_lazy_flags == flags


def test_an_unsized_kv_cache_proves_nothing_on_a_mac():
    backend = _apple_backend("qwen4exp", {PLE: FLASH_NEXT_PLE}, avail_gib = 119)
    backend._can_estimate_kv = lambda: False
    assert _apple_discount(backend) == FLASH_NEXT_PLE
    assert backend._apple_lazy_flags == []


def test_a_users_lazy_off_on_a_mac_takes_the_table_back_out_of_the_probe_discount():
    backend = _apple_backend("qwen4exp", {PLE: FLASH_NEXT_PLE}, avail_gib = 119)
    got = _apple_discount(
        backend, FLASH_NEXT_PLE + FLASH_NEXT_TOKEN_EMBD, caps = _CAPS, extra_args = ["-lzm", "off"]
    )
    assert got == FLASH_NEXT_TOKEN_EMBD
    assert backend._apple_lazy_flags == []


def test_gemma3n_gets_no_metal_table_discount():
    """Not TENSOR_READ_LAZY upstream and under 4 GiB: only the probe can discount it."""
    backend = _apple_backend("gemma3n", _GEMMA3N_NAMED, avail_gib = 12)
    assert _apple_discount(backend, model_size_fit = GEMMA3N_TOTAL, budget_gib = 10) == 0
    assert backend._apple_lazy_flags == []


@pytest.mark.parametrize(
    "lazy_args, expected", [([], FLASH_NEXT_PLE), (["-lzm", "off"], 0)], ids = ["auto", "off"]
)
def test_a_metal_override_moves_the_table_only_when_it_is_not_streamed(lazy_args, expected):
    """llama.cpp picks the lazy buffer before -ot, so a streamed table stays off Metal."""
    backend = _apple_backend("qwen4exp", {PLE: FLASH_NEXT_PLE}, avail_gib = 58)
    got = _apple_discount(
        backend,
        caps = _CAPS,
        extra_args = ["-ot", "per_layer_token_embd=MTL0", *lazy_args],
        budget_gib = 48,
    )
    assert got == expected


def test_module_table_names_deepseek41():
    assert "deepseek41" in lc._LAZY_READ_TENSORS


@pytest.mark.parametrize("name", ["LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL"])
def test_an_inherited_projector_on_a_mac_keeps_the_table_streamed(name):
    """The env's projector is unpriced in the Metal footprint, so a Mac that would read the table resident leaves it to auto."""
    backend = _apple_backend("qwen4exp", {PLE: FLASH_NEXT_PLE}, avail_gib = 119)
    assert _apple_discount(backend, caps = _CAPS, env = {name: "/p/mmproj.gguf"}) == FLASH_NEXT_PLE
    assert backend._apple_lazy_flags == []


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
    """RPC devices report mmap support (ggml-rpc.cpp device_get_props, b11734-mix-c208e5f and b11870)."""
    got = LlamaCppBackend()._lazy_auto_resolves_off(
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
    pinned = lc._host_pinned_input_bytes
    scan = ("qwen4exp", {PLE: FLASH_NEXT_PLE}, 0)
    assert pinned(scan, OT_TABLE, {}, lazy_mode = "auto") == FLASH_NEXT_PLE
    assert pinned(scan, OT_TABLE, {}, lazy_mode = "off") == 0
    small = ("gemma4", {PLE: GEMMA3N_PLE}, 0)
    assert pinned(small, OT_TABLE, {}, lazy_mode = "auto") == 0
    assert pinned(small, OT_TABLE, {}, lazy_mode = "on") == GEMMA3N_PLE


@pytest.mark.parametrize(
    "lazy_args, fit",
    [([], "off"), (["-lzm", "on"], "off"), (["-lzm", "off"], "on")],
    ids = ["auto", "on", "off"],
)
def test_a_device_override_leaves_a_streamed_table_off_the_vram_charge(
    tmp_path, _moe_cache_host, lazy_args, fit
):
    _backend, cmd = _flash_next_on_two_cards(tmp_path, ["-ot", "embd=CUDA0", *lazy_args])
    assert _arg(cmd, "--fit") == fit, cmd
    if fit == "off":
        assert _arg(cmd, "-ngl") == "-1", cmd


def test_a_streamed_table_under_an_override_keeps_the_resident_embedding_host_only(
    tmp_path, _moe_cache_host, monkeypatch
):
    seen = _spy_fit_load_mode(monkeypatch)
    _flash_next_on_two_cards(tmp_path, OT_TABLE)
    assert seen and seen[-1]["lazy_read_bytes"] == FLASH_NEXT_PLE
    assert seen[-1]["host_only_bytes"] >= FLASH_NEXT_TOKEN_EMBD, seen[-1]


@pytest.mark.parametrize(
    "forced, flags", [(frozenset(), []), (frozenset({"deepseek41"}), ["--lazy-mode", "off"])]
)
def test_a_device_override_on_deepseek41_follows_the_builds_placement(monkeypatch, forced, flags):
    """Where -ot moves the Engram table, -lzm off would make it GPU-resident while priced on the host."""
    monkeypatch.setattr(
        LlamaCppBackend, "_host_forced_table_archs", staticmethod(lambda binary = None: forced)
    )
    backend = _scan_backend("deepseek41", {PLE: DSV41_ENGRAM})
    assert _managed_flags(backend, OT_TABLE) == flags


def _server_beside(tmp_path, name, payload):
    server = tmp_path / "llama-server"
    server.write_bytes(b"\x7fELF server")
    (tmp_path / name).write_bytes(payload)
    return LlamaCppBackend._host_forced_table_archs(str(server))


@pytest.mark.parametrize("marked", [True, False])
def test_the_host_forced_probe_reads_the_builds_loader(tmp_path, marked):
    """Studio's prebuilt b11734-mix-c208e5f carries the loader warning; upstream b11870 does not."""
    marker = lc._HOST_FORCED_TABLE_MARKER if marked else b"place it anywhere"
    got = _server_beside(tmp_path, "libllama.so.0.7.0", b"\x00" * 64 + marker + b" %s, using %s\n")
    assert got == (frozenset({"deepseek41"}) if marked else frozenset())
    gone = str(tmp_path / "gone" / "llama-server")
    assert LlamaCppBackend._host_forced_table_archs(gone) == frozenset()


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
    got = _server_beside(tmp_path, name, lc._HOST_FORCED_TABLE_MARKER)
    assert got == (frozenset({"deepseek41"}) if forced else frozenset())


def test_the_lazy_mode_choices_probe_the_launch_binary(monkeypatch):
    seen = []
    monkeypatch.setattr(
        LlamaCppBackend,
        "_host_forced_table_archs",
        staticmethod(lambda binary = None: seen.append(binary) or frozenset()),
    )
    backend = _apple_backend("deepseek41", {PLE: DSV41_ENGRAM}, avail_gib = 119)
    ot = ["-ot", "per_layer_token_embd=MTL0"]
    _managed_flags(backend, ot, binary = "/launch/llama-server")
    _apple_discount(
        backend,
        caps = _CAPS,
        extra_args = ot,
        model_size_fit = DSV41_TOTAL,
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
    _backend, cmd = _flash_next_launch(tmp_path, extra_args = extra_args)
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
    assert lc._pass_through_reprices_cache(extra_args, n_parallel) is reprices


@pytest.mark.parametrize("cpu_build", [False, True])
def test_a_cpu_child_keeps_llama_cpps_lazy_default(
    tmp_path, _moe_cache_host, monkeypatch, cpu_build
):
    """The resident verdict credits the cards; a CPU-only build or --device none has none."""
    monkeypatch.setattr(
        LlamaCppBackend, "_binary_ships_no_gpu_backend", staticmethod(lambda *a, **k: cpu_build)
    )
    _backend, cmd = _flash_next_launch(tmp_path)
    assert (_lazy_values(cmd) == ["off"]) is not cpu_build, cmd
    _backend, cmd = _flash_next_launch(tmp_path, extra_args = ["--device", "none"])
    assert _lazy_values(cmd) == [], cmd


def _metal_launched(tmp_path, monkeypatch, **kw):
    """A Metal load_model (16 GiB budget, 12 GiB weights); returns argv."""
    import test_metal_explicit_context_guard as metal
    return metal._launch(
        tmp_path,
        monkeypatch,
        n_ctx = 8192,
        kv_per_token = 32 * 1024,
        backend = metal._backend_with_embeddings(monkeypatch, embd = 0),
        real_fit = True,
        budget_bytes = 16 * GIB,
        weights_bytes = 12 * GIB,
        **kw,
    )["cmd"]


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
    from test_metal_explicit_context_guard import TestInputEmbeddingsLeftInTheFileMapping as T

    monkeypatch.setattr(
        LlamaCppBackend,
        "_apple_lookup_table_discount",
        lambda self, probe, *a, **k: max(int(probe), 6 * GIB),
    )
    cmd = _metal_launched(tmp_path, monkeypatch, extra_args = extra_args)
    assert (T._fit_tokens(cmd)[:4] == ["-ngl", "-1", "--fit", "off"]) is pinned, cmd


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


@pytest.mark.parametrize(
    "kw, budget_gib",
    [
        ({"wired_bytes": 0}, 16),
        ({"wired_bytes": 8 * GIB}, 8),
        ({"wired_bytes": 32 * GIB}, 16),
        ({"wired_exhausted": True}, 0),
    ],
    ids = ["0-16", "8-8", "32-16", "used_up_wired_headroom_leaves_no_resident_budget"],
)
def test_the_metal_resident_verdict_obeys_the_wired_headroom(tmp_path, monkeypatch, kw, budget_gib):
    seen = _spy_resident_inputs(monkeypatch)
    _metal_launched(tmp_path, monkeypatch, **kw)
    assert seen[-1]["budget_mib"] == budget_gib * 1024


@pytest.mark.parametrize(
    "extra_args, cpu_forced",
    [(["--device", "none"], False), ([], True)],
    ids = ["device_none", "paravirtual"],
)
def test_a_cpu_child_on_a_mac_keeps_the_table_streamed(extra_args, cpu_forced):
    backend = _apple_backend("qwen4exp", {PLE: FLASH_NEXT_PLE}, avail_gib = 119)
    got = _apple_discount(backend, extra_args = extra_args, cpu_forced = cpu_forced)
    assert backend._apple_lazy_flags == []
    assert got == FLASH_NEXT_PLE


def test_the_metal_resident_verdict_abstains_on_an_unsized_mtp_draft(tmp_path, monkeypatch):
    """A flat MTP reserve prices no draft KV, so no resident verdict (as the discrete one)."""
    seen = _spy_resident_inputs(monkeypatch)
    _metal_launched(tmp_path, monkeypatch)
    assert seen[-1]["need_at_ctx"] is not None
    _metal_launched(tmp_path, monkeypatch, extra_args = ["--spec-type", "draft-mtp"])
    assert seen[-1]["need_at_ctx"] is None


def test_a_rocm_arch_gated_cpu_launch_is_a_cpu_child(tmp_path, monkeypatch, probe_env):
    """No installed kernel covers either card, so the child runs on the CPU."""
    import test_gpu_arch_gate_os_matrix_7624 as gate

    forced = _spy(monkeypatch, "_child_runs_on_cpu", lambda _ans, _a, k: bool(k.get("cpu_forced")))
    backend = _scan_backend("qwen4exp", {PLE: FLASH_NEXT_PLE})
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
    seen = _spy_fit_load_mode(monkeypatch)
    monkeypatch.setattr(lc, "_gpu_device_node_present", lambda: node)
    _flash_next_launch(tmp_path, ram_gib = 58, memory = [])
    assert seen, "no load-mode fit ran"
    assert seen[-1]["lazy_read_bytes"] == (0 if node else FLASH_NEXT_PLE), seen[-1]


@pytest.mark.parametrize(
    "extra_args, env, cpu",
    [
        (["--split-mode", "none", "--main-gpu", "-1"], {}, True),
        (["-sm", "none", "-mg", "-1"], {}, True),
        ([], _INHERITED_CPU_SPLIT, True),
        (["-sm", "none", "-mg", "0"], {}, False),
        (["--main-gpu", "-1"], {}, False),
        (["-sm", "layer", "-mg", "-1"], {}, False),
    ],
)
def test_split_mode_none_with_a_negative_main_gpu_is_a_cpu_child(extra_args, env, cpu):
    """llama.cpp clears the model's devices for it (src/llama.cpp, b11734-mix-c208e5f)."""
    backend = LlamaCppBackend()
    backend._binary_ships_no_gpu_backend = lambda *a, **k: False
    assert backend._child_runs_on_cpu(None, extra_args, dict(env)) is cpu


@pytest.mark.parametrize(
    "extra_args, env, resident",
    [
        (["-sm", "none", "-mg", "0"], {}, True),
        (["--split-mode", "none", "--main-gpu", "-1"], {}, False),
        (["--split-mode", "none"], {"LLAMA_ARG_MAIN_GPU": "-1"}, False),
        # load_model drops an inherited non-layer split mode on a layer launch: layer split on CUDA0.
        ([], _INHERITED_CPU_SPLIT, True),
        (["--main-gpu", "-1"], {"LLAMA_ARG_SPLIT_MODE": "none"}, True),
    ],
)
def test_a_main_gpu_of_minus_one_keeps_llama_cpps_lazy_default(
    tmp_path, _moe_cache_host, monkeypatch, extra_args, env, resident
):
    _setenv(monkeypatch, env)
    _backend, cmd = _flash_next_launch(tmp_path, extra_args = extra_args)
    assert (_lazy_values(cmd) == ["off"]) is resident, cmd


def _spy_placement_previews(monkeypatch):
    """(helper, caller, env split mode, tensor_parallel) for every placement preview."""
    seen = []
    for name in ("_lazy_auto_resolves_off", "_child_runs_on_cpu", "_apple_lookup_table_discount"):
        real = getattr(LlamaCppBackend, name)

        def make(real, name):
            def spy(self, *a, **k):
                env = k.get("env", a[2] if name == "_child_runs_on_cpu" and len(a) > 2 else None)
                env = os.environ if env is None else env
                seen.append(
                    (
                        name,
                        sys._getframe(1).f_code.co_name,
                        env.get("LLAMA_ARG_SPLIT_MODE"),
                        k.get("tensor_parallel"),
                    )
                )
                return real(self, *a, **k)

            return spy

        monkeypatch.setattr(LlamaCppBackend, name, make(real, name))
    return seen


@pytest.mark.parametrize("cards", [1, 2])
def test_every_launch_preview_sees_the_split_mode_the_child_gets(
    tmp_path, _moe_cache_host, monkeypatch, cards
):
    """load_model drops an inherited non-layer split mode before spawn; every preview must price that child."""
    _setenv(monkeypatch, _INHERITED_CPU_SPLIT)
    seen = _spy_placement_previews(monkeypatch)
    launch = _flash_next_launch if cards == 1 else _flash_next_on_two_cards
    _backend, cmd = launch(tmp_path)
    callers = {caller for _n, caller, _sm, _tp in seen}
    assert {"_host_input_bytes_now", "load_model", "_table_resident_fits"} <= callers, seen
    assert cards == 2 or "_fitter_vram_shortfall" in callers, seen
    assert [s for s in seen if s[2] is not None] == [], seen
    assert _lazy_values(cmd) == ["off"], cmd


def test_a_tensor_launch_is_not_a_cpu_child_under_an_inherited_split_mode(
    tmp_path, _moe_cache_host, monkeypatch
):
    """Studio emits --split-mode tensor (argv beats env), so the inherited CPU split does not apply."""
    _setenv(monkeypatch, _INHERITED_CPU_SPLIT)
    seen = _spy_placement_previews(monkeypatch)
    _backend, cmd = _flash_next_on_two_cards(tmp_path, tensor_parallel = True)
    assert "tensor" in cmd, cmd
    cpu_calls = [s for s in seen if s[0] == "_child_runs_on_cpu" and s[1] == "_table_resident_fits"]
    assert cpu_calls and cpu_calls[-1][3] is True, seen
    assert _lazy_values(cmd) == ["off"], cmd


@pytest.mark.parametrize("tensor, cpu", [(None, True), (False, True), (True, False)])
def test_the_cpu_child_check_follows_the_launch_tensor_state(tensor, cpu):
    backend = LlamaCppBackend()
    backend._binary_ships_no_gpu_backend = lambda *a, **k: False
    got = backend._child_runs_on_cpu(None, [], dict(_INHERITED_CPU_SPLIT), tensor_parallel = tensor)
    assert got is cpu


def _apu_flash_next(tmp_path, monkeypatch, apu_ids):
    backend, gguf = placement._apu_backend(
        tmp_path, gguf_gb = 73.45, avail_mib = 58 * 1024, monkeypatch = monkeypatch
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: set(apu_ids))
    )
    backend._gguf_tensor_scan = lambda _path: ("qwen4exp", {PLE: placement._QWEN38_PLE}, 0)
    backend.probe_server_capabilities = _full_caps()
    return backend, gguf


def test_a_failed_probe_apu_guard_ignores_a_split_mode_the_launch_drops(tmp_path, monkeypatch):
    """The HIP child runs layer split on the APU once load_model drops split mode none, so the guard charges its table."""
    _setenv(monkeypatch, _INHERITED_CPU_SPLIT)
    backend, gguf = _apu_flash_next(tmp_path, monkeypatch, {0})
    backend._get_gpu_memory = lambda *_a, **_kw: []
    backend._get_gpu_free_memory = lambda *_a, **_kw: []

    res = placement._launch(backend, gguf, extra_args = ["--load-mode", "none"])

    assert res["env"].get("LLAMA_ARG_SPLIT_MODE") is None
    assert "unified-memory APU" in (backend.last_load_warning or "")
    assert not placement._unmapped_tokens(res["cmd"]), res["cmd"]


def test_the_metal_discount_sees_the_split_mode_the_child_gets(tmp_path, monkeypatch):
    _setenv(monkeypatch, _INHERITED_CPU_SPLIT)
    seen = _spy_placement_previews(monkeypatch)
    _metal_launched(tmp_path, monkeypatch)
    metal = [s for s in seen if s[0] == "_apple_lookup_table_discount"]
    assert metal and all(s[2] is None for s in metal), seen


@pytest.mark.parametrize(
    "extra_args",
    [["--device", "ROCm0"], ["--device", "ROCm0", "--load-mode", "none"], []],
    ids = ["device", "device_unmapped", "no_device"],
)
def test_a_compact_device_names_the_card_the_child_mask_puts_first(
    tmp_path, monkeypatch, extra_args
):
    """Auto pins a ROCm APU at physical 1 beside a discrete card at 0, so the child's ROCm0 is the APU, not the parent's."""
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising = False)
    backend, gguf = _apu_flash_next(tmp_path, monkeypatch, {1})
    rows = [(0, 16_000, 16_384), (1, 90_000, 92_160)]
    backend._get_gpu_memory = lambda *_a, **_kw: list(rows)
    backend._get_gpu_free_memory = lambda *_a, **_kw: [(i, f) for i, f, _t in rows]
    backend._amd_apu_wants_unified_memory = lambda ids = None, *a, **k: (
        ids is not None and bool(list(ids)) and {int(i) for i in ids} <= {1}
    )
    backend._select_gpus = lambda *a, **k: ([1], True)
    backend._can_estimate_kv = lambda: True
    backend._estimate_kv_cache_bytes = lambda *a, **k: GIB // 4
    seen = _spy(
        monkeypatch,
        "_lazy_auto_resolves_off",
        lambda answer, _a, _k: (sys._getframe(2).f_code.co_name, answer),
    )

    res = placement._launch(backend, gguf, extra_args = extra_args)

    assert res["env"].get("CUDA_VISIBLE_DEVICES") == "1", res["env"]
    previews = [s for s in seen if s[0] in ("load_model", "_fitter_vram_shortfall")]
    assert previews and all(answer for _c, answer in previews), seen
    assert "unified-memory APU" in (backend.last_load_warning or "")
    assert not placement._unmapped_tokens(res["cmd"]), res["cmd"]


@pytest.mark.parametrize("ram_gib", [58, 256])
@pytest.mark.parametrize("layout", ["dgpu0_igpu1", "igpu0_dgpu1"])
def test_split_mode_none_reads_the_main_device_in_picker_order(
    tmp_path, _moe_cache_host, layout, ram_gib
):
    """gpu_ids [1, 0] emits --device Vulkan1,Vulkan0, so --split-mode none puts every weight on Vulkan1, not Vulkan0."""
    dgpu, igpu = (0, 1) if layout == "dgpu0_igpu1" else (1, 0)
    rows = sorted([(dgpu, 35 * 1024, 36 * 1024), (igpu, 55 * 1024, 0)])
    backend, gguf = placement._moe_backend(tmp_path, size_gib = 73.45, memory = rows, moe = False)
    backend._is_vulkan_backend = lambda _binary = None: True
    backend._can_estimate_kv = lambda: True
    backend._estimate_kv_cache_bytes = lambda *a, **k: GIB // 4
    backend._available_system_memory_mib = lambda: int(ram_gib * 1024)
    backend._gguf_tensor_scan = lambda _path: ("qwen4exp", {PLE: placement._QWEN38_PLE}, 0)
    backend.probe_server_capabilities = _full_caps()

    cmd = placement._launch(backend, gguf, gpu_ids = [1, 0], extra_args = ["--split-mode", "none"])[
        "cmd"
    ]

    assert _arg(cmd, "--device") == "Vulkan1,Vulkan0", cmd
    main_is_igpu = igpu == 1
    # An iGPU main device keeps the table resident (auto is off there): 73.45 GiB does
    # not fit 58 GiB of RAM unmapped, so the load stays mmap'd.
    if main_is_igpu and ram_gib == 58:
        assert placement._load_mode(cmd) is None, cmd
    # A discrete main device streams it unless RAM holds it, where Studio pins it resident.
    expected_lazy = ["off"] if not main_is_igpu and ram_gib == 256 else []
    assert _lazy_values(cmd) == expected_lazy, cmd


@pytest.mark.parametrize(
    "gpu_ids, gpu_indices, split, inherited, order, mask",
    [
        (None, [0, 1], None, None, [0, 1], [0, 1]),
        ([1, 0], [0, 1], None, None, [1, 0], [1, 0]),
        ([1, 0], None, None, None, [1, 0], [1, 0]),
        ([1, 0], [0, 1], [3, 1], None, [1, 0], [1, 0]),
        ([1, 0], [0, 1], [3], None, [0, 1], [0, 1]),
        (None, [0, 1], None, [1, 0], [0, 1], [1, 0]),
        (None, [0, 1], [3, 1, 2], [1, 0], [0, 1], [0, 1]),
        (None, None, None, None, None, None),
    ],
    ids = [
        "ascending",
        "picker_reorder",
        "picker_unnarrowed",
        "picker_moves_shares",
        "picker_veto_shares",
        "inherited_order",
        "inherited_veto_shares",
        "unpinned",
    ],
)
def test_the_planned_child_order_matches_the_spawn(
    monkeypatch, gpu_ids, gpu_indices, split, inherited, order, mask
):
    monkeypatch.setattr(
        LlamaCppBackend, "_resolve_visible_physical_ids", staticmethod(lambda: inherited)
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_pinned_child_visibility",
        staticmethod(lambda ids: ",".join(str(int(i)) for i in ids)),
    )
    backend = LlamaCppBackend()
    kwargs = dict(
        gpu_ids = gpu_ids, gpu_indices = gpu_indices, tp_tensor_split = split, extra_args = [], env = {}
    )
    plan = backend._planned_child_gpu_order
    assert plan(**kwargs, is_vulkan_backend = False) == (order, mask)
    # Vulkan has no mask; a CPU child gets none either.
    assert plan(**kwargs, is_vulkan_backend = True) == (order, None)
    assert plan(**kwargs, is_vulkan_backend = False, cpu_forced = True) == (order, None)
    # Ids the mask cannot name keep the inherited one: the parent's numbering stands.
    monkeypatch.setattr(LlamaCppBackend, "_pinned_child_visibility", staticmethod(lambda ids: None))
    assert plan(**kwargs, is_vulkan_backend = False) == (order, None)


def test_a_compact_device_maps_through_the_planned_mask(monkeypatch):
    backend = LlamaCppBackend()
    backend._integrated_cuda_gpu_ids = lambda: set()
    monkeypatch.setattr(LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: {1}))
    common = dict(
        gpu_indices = [1],
        detected_gpus = [(0, 16_000), (1, 90_000)],
        shared_gpu_ids = set(),
        is_vulkan_backend = False,
        extra_args = ["--device", "ROCm0"],
        env = {},
    )
    # The parent's numbering: ROCm0 is the discrete card; the child's mask [1]: ROCm0 is the APU.
    assert backend._lazy_auto_resolves_off(**common) is False
    assert backend._lazy_auto_resolves_off(**common, child_gpu_order = [1]) is True
