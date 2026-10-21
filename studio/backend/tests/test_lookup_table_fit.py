# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lookup-only embedding tables (per-layer embeddings, the DeepSeek-V4.1 Engram table)
in the launch fit, the offload layout and the planner.

Sizes are the real GGUF headers:
  DeepSeek-V4.1-Flash UD-IQ1_S   deepseek41  216.98 GiB, per_layer_token_embd 103.00 GiB
  Qwen3.8-Flash-Next UD-Q2_K_XL  qwen4exp     73.45 GiB, per_layer_token_embd  26.82 GiB
  gemma-3n-E4B-it Q4_K_M         gemma3n       4.23 GiB, per_layer_token_embd   1.64 GiB,
                                               token_embd 0.28 GiB, tied (no output.weight)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
_TESTS_DIR = str(Path(__file__).resolve().parent)
for _d in (_BACKEND_DIR, _TESTS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

# First: these install the dependency stubs the backend imports need on a bare runner.
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


# ---------------------------------------------------------------- item 1: deepseek41 is lazy


@pytest.mark.parametrize(
    "extra_args, auto_off, expected",
    [
        ([], False, DSV41_ENGRAM),
        # An iGPU / APU child: llama.cpp turns auto into off, the table loads resident.
        ([], True, 0),
        (["-lzm", "off"], False, 0),
        # An explicit "on" is read lazily on any device.
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
    """DSV4.1 UD-IQ1_S on a 24 GiB card with 64 GiB of RAM. The host side the warning
    prices was ~194 GiB with the 103 GiB Engram table charged as resident weights (a
    ~132 GiB shortfall); with it on disk ~91 GiB remain, still short of 64 GiB, so the
    warning still fires, ~29 GiB short."""
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


# ------------------------------------------- item 2: the per-layer table is its own bucket


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
    """gemma-3n is tied: only the 0.28 GiB token_embd is copied onto the GPU as the
    output head. The 1.64 GiB PLE table was charged to VRAM with it."""
    layout = _gemma3n_layout()
    assert layout.other_resident_bytes == GEMMA3N_TOKEN_EMBD + 4096
    untied = _gemma3n_layout(with_output = True)
    assert untied.other_resident_bytes == 4096
    assert untied.lm_head_bytes == GEMMA3N_TOKEN_EMBD


# --------------------------------------- item 3: lazy bytes are not host RAM in the planner


def _flash_next_layout():
    # 40 MoE blocks of 1.0 GiB experts and 0.14 GiB attention/shared: 73.45 GiB in all.
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
    """Flash-Next on 24 GiB + a 64 GB box with 48 GiB available: 25 spilled expert blocks
    and token_embd fit pinned; adding the 26.82 GiB table that stays mmap'd anyway
    pushed the host side past the RAM and left the experts on the slow mmap path."""
    layout = _flash_next_layout()
    chosen = list(layout.blocks[:25])
    lazy = _finish(
        layout, PlanOptions(lazy_read_bytes = FLASH_NEXT_PLE), 8192, chosen, False, 48 * GIB
    )
    assert lazy.load_mode_none
    assert lazy.host_bytes == FLASH_NEXT_TOKEN_EMBD + 25 * GIB
    # Resident table (-lzm off, or auto resolved off on an iGPU): charged in full.
    resident = _finish(layout, PlanOptions(), 8192, chosen, False, 48 * GIB)
    assert not resident.load_mode_none
    assert resident.host_bytes == FLASH_NEXT_TOKEN_EMBD + FLASH_NEXT_PLE + 25 * GIB


# ----------------------------------- item 5: host-pinned inputs are not charged to VRAM


def test_host_pinned_inputs_for_the_three_families():
    # gemma-3n is tied: token_embd is duplicated onto the GPU, the PLE table is not.
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


def test_an_override_onto_a_device_keeps_the_table_charged_except_on_deepseek41():
    ot = ["-ot", "per_layer_token_embd=CUDA0"]
    assert _override_places_tensor_on_device(PLE, ot)
    assert not _override_places_tensor_on_device(PLE, ["-ot", "per_layer=CPU"])
    # First match wins, as in llama.cpp.
    assert not _override_places_tensor_on_device(PLE, ["-ot", "per_layer=CPU,embd=CUDA0"])
    assert not _override_places_tensor_on_device(PLE, ["-ot", "exps=CPU"])
    assert _override_places_tensor_on_device(PLE, [], {"LLAMA_ARG_OVERRIDE_TENSOR": "embd=CUDA0"})
    assert _host_pinned_input_bytes(("qwen4exp", {PLE: FLASH_NEXT_PLE}, 0), ot, {}) == 0
    # The deepseek41 loader forces the Engram table back onto a host buffer.
    assert _host_pinned_input_bytes(("deepseek41", {PLE: DSV41_ENGRAM}, 0), ot, {}) == DSV41_ENGRAM


def _gemma3n_named():
    return {PLE: GEMMA3N_PLE, "token_embd.weight": GEMMA3N_TOKEN_EMBD}


def _fit_flags(cmd):
    i = cmd.index("--fit")
    return cmd[i + 1]


def test_gemma3n_fits_a_small_card_once_its_tables_leave_the_vram_charge(tmp_path, _moe_cache_host):
    """gemma-3n-E4B Q4_K_M (4.23 GiB). The weights the GPU really holds are ~2.6 GiB:
    the 1.64 GiB PLE table is host-pinned, and the tied token_embd copy is the only
    embedding on the card. On a card whose free VRAM sits between the two charges
    (this harness prices ~9.8 GiB with the whole file, ~8.2 GiB without the table) the
    fit pins the layers; the whole file charged to VRAM sent it to the --fit on fallback."""
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
    # The table is not lazy on gemma3n, so no lazy mode is ever emitted.
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
    """gemma3n's table is resident (never lazily read) and host-pinned: free VRAM cannot pay
    for it, so a host whose RAM cannot hold it keeps llama.cpp's mmap fallback."""
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


# ------------------------------------------------- item 8: lazy mode as a managed choice


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
    """llama.cpp reads only its own spelling, so an older build's env twin left set does
    not stop the managed resident choice."""
    monkeypatch.setenv("LLAMA_ARG_TENSOR_READ_LAZY", "off")
    backend, cmd = _flash_next_launch(tmp_path, ram_gib = 128)
    assert _lazy_values(cmd) == ["off"], cmd


def test_flash_next_keeps_its_table_resident_where_ram_holds_it(tmp_path, _moe_cache_host):
    backend, cmd = _flash_next_launch(tmp_path, ram_gib = 128)
    assert _lazy_values(cmd) == ["off"], cmd
    assert backend._lazy_mode_flags == ["--lazy-mode", "off"]


def test_the_resident_table_is_charged_once_in_the_load_mode_fit(tmp_path, _moe_cache_host):
    """After the resident verdict the table moves from the pooled weights to host-only
    RAM. Charging it in both left a host that just proved the fit without --load-mode
    none. 72 GiB holds ~47 GiB of host weights plus the table once, not twice."""
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
    # A user's own pair is never in the record, so it is left alone.
    backend._lazy_mode_flags = []
    assert backend._drop_managed_lazy_mode(argv, "CPU fallback") == argv


def test_a_device_override_keeps_the_table_streamed(tmp_path, _moe_cache_host):
    """Read lazily, the table sits on a CPU buffer whatever -ot says; resident, the
    override would apply and move 26.8 GiB onto a 24 GiB card."""
    backend, cmd = _flash_next_launch(
        tmp_path, ram_gib = 128, extra_args = ["-ot", "per_layer_token_embd=CUDA0"]
    )
    assert _lazy_values(cmd) == [], cmd
    assert backend._lazy_mode_flags == []


def test_the_in_place_fit_on_retry_hands_the_table_back(tmp_path, _moe_cache_host, monkeypatch):
    """A full offload launches with --fit off; when that crashes, the --fit on retry can
    spill weights to RAM beside the table, so the resident verdict does not carry over."""
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
            # A device probe (--list-devices on Windows) is not a server spawn: it exits 0.
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
    # Whatever in-place retry the platform takes after the crash, it re-places the model.
    assert len(cmds) == 2, cmds
    assert _lazy_values(cmds[0]) == ["off"], cmds[0]
    assert _lazy_values(cmds[1]) == [], cmds[1]


def test_flash_next_streams_its_table_on_a_64gb_box(tmp_path, _moe_cache_host):
    """24 GiB card + 64 GB RAM: ~47 GiB of weights, the 8 GiB prompt cache and the
    26.8 GiB table do not fit together, so llama.cpp's lazy auto stays."""
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
    # The pre-rename spelling is what that build reads.
    old = {"supports_lazy_mode": True, "lazy_mode_flag": "--tensor-read-lazy"}
    assert backend._managed_lazy_mode_flags(
        "/m.gguf", caps = old, extra_args = [], env = {}, resident_fits = lambda _t: True
    ) == ["--tensor-read-lazy", "off"]


# -------------------------------------- item 4: Metal falls back to the static discount


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
    """DSV4.1 UD-IQ1_S on a 128 GB Mac. The probe failing used to charge all 216.98 GiB to
    unified memory; the Engram table is lazily read on a CPU buffer whatever the probe
    says, so 113.98 GiB is charged."""
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


def test_a_metal_override_onto_the_gpu_keeps_the_probe_answer():
    backend = _apple_backend("qwen4exp", {PLE: FLASH_NEXT_PLE}, avail_gib = 58)
    got = backend._apple_lookup_table_discount(
        0,
        "/m.gguf",
        caps = dict(_CAPS),
        extra_args = ["-ot", "per_layer_token_embd=MTL0"],
        model_size_fit = FLASH_NEXT_TOTAL,
        need_at_ctx = GIB,
        budget_mib = 48 * 1024,
        cache_ram = None,
        env = {},
    )
    assert got == 0


def test_module_table_names_deepseek41():
    assert "deepseek41" in llama_cpp_module._LAZY_READ_TENSORS
