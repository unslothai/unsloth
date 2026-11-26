"""Inkling 2 with every expert on the GPU launches with -ub 1024 when the larger compute buffer fits."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_llama_cpp_placement import (  # noqa: E402
    _CACHE_CAPS_ON,
    _GIB,
    _batch_values,
    _discrete_linux_host,  # noqa: F401 - fixture
    _launch,
    _moe_backend,
    _moe_cache_host,  # noqa: F401 - fixture
    _ubatch_values,
)
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
import core.inference.llama_cpp as llama_cpp_module  # noqa: E402

_MIB = 1024 * 1024
# Compute buffer at the default micro-batch and the extra cost per micro-batch step.
_CB_DEFAULT = 512 * _MIB


def _inkling_launch(
    tmp_path,
    *,
    arch = "inkling2",
    size_gib = 8,
    memory = ((0, 40_000, 48_000),),
    raise_cost_mib = 256,
    record = None,
    **load_kwargs,
):
    """One MoE launch whose compute buffer grows by ``raise_cost_mib`` per micro-batch doubling."""
    backend, gguf = _moe_backend(tmp_path, size_gib = size_gib, memory = list(memory))
    backend._architecture = arch
    backend._can_estimate_kv = lambda: True
    backend._estimate_kv_cache_bytes = lambda *a, **k: _GIB // 4
    backend._available_system_memory_mib = lambda: 256 * 1024
    backend._gguf_tensor_scan = lambda _path: (arch, {}, int(max(1, size_gib - 1) * _GIB))

    def estimate(
        *,
        n_ubatch = None,
        n_parallel = 1,
        per_device_tensor = False,
        **_kw,
    ):
        ub = int(n_ubatch or 512)
        if record is not None:
            record.append((ub, int(n_parallel)))
        steps = 0
        while ub > 512:
            ub //= 2
            steps += 1
        return _CB_DEFAULT + steps * raise_cost_mib * _MIB

    backend._estimate_compute_buffer_bytes = estimate
    caps = dict(LlamaCppBackend.probe_server_capabilities.__func__(LlamaCppBackend, None))
    caps.update(_CACHE_CAPS_ON, supports_kv_unified = True)
    backend.probe_server_capabilities = lambda _binary = None: caps
    load_kwargs.setdefault("n_ctx", 8192)
    cmd = _launch(backend, gguf, **load_kwargs)["cmd"]
    return backend, cmd


@pytest.fixture
def _inkling_host(monkeypatch, _moe_cache_host):
    monkeypatch.delenv("UNSLOTH_INKLING_UBATCH", raising = False)


@pytest.fixture
def _inkling_log(monkeypatch):
    """The Inkling raise lines Studio logs (its logger is structlog, not stdlib)."""
    lines: list[str] = []
    real = llama_cpp_module.logger.info

    def info(msg, *args, **kwargs):
        text = str(msg) % args if args else str(msg)
        if text.startswith("Inkling 2 on GPU"):
            lines.append(text)
        return real(msg, *args, **kwargs)

    monkeypatch.setattr(llama_cpp_module.logger, "info", info)
    return lines


@pytest.mark.parametrize("arch", ["inkling2", "inkling2_loop"])
def test_an_all_gpu_inkling_launches_with_micro_batch_1024(
    tmp_path, _inkling_host, _inkling_log, arch
):
    backend, cmd = _inkling_launch(tmp_path, arch = arch)

    assert cmd[cmd.index("--fit") + 1] == "off", cmd
    assert _ubatch_values(cmd) == ["1024"], cmd
    # llama.cpp's default batch (2048) already holds the micro-batch.
    assert all(int(b) >= 1024 for b in _batch_values(cmd)), cmd
    assert backend._n_ubatch == 1024
    assert backend.requested_n_ubatch is None
    # A CPU replay hands back the pair the user had.
    assert backend._moe_spill_batch_restore == (None, None)
    assert _inkling_log == [
        "Inkling 2 on GPU: raising --ubatch-size 512 -> 1024 for faster prompt processing "
        "(+256 MiB compute buffer)."
    ], _inkling_log


def test_a_spilled_inkling_keeps_the_2048_spill_raise(tmp_path, _inkling_host, _inkling_log):
    _backend, cmd = _inkling_launch(
        tmp_path, size_gib = 20, memory = [(0, 8_000, 16_000)], raise_cost_mib = 16
    )

    assert cmd[cmd.index("--fit") + 1] == "on", cmd
    assert _ubatch_values(cmd) == ["2048"], cmd
    assert _inkling_log == []


@pytest.mark.parametrize(
    "extra_args, ubatch",
    [(["-ub", "512"], ["512"]), (["-ub", "768"], ["768"]), (["-b", "4096"], [])],
    ids = ["ub512", "ub768", "b4096"],
)
def test_a_user_batch_pair_is_left_alone(tmp_path, _inkling_host, extra_args, ubatch):
    backend, cmd = _inkling_launch(tmp_path, extra_args = extra_args)

    assert _ubatch_values(cmd) == ubatch, cmd
    assert backend._moe_spill_batch_restore is None


def test_a_first_class_micro_batch_is_left_alone(tmp_path, _inkling_host):
    backend, cmd = _inkling_launch(tmp_path, n_ubatch = 512)

    assert _ubatch_values(cmd) == ["512"], cmd
    assert backend._moe_spill_batch_restore is None


@pytest.mark.parametrize("switch", ["off", "0", "never"])
def test_the_kill_switch_keeps_the_default_micro_batch(
    tmp_path, _inkling_host, monkeypatch, switch
):
    monkeypatch.setenv("UNSLOTH_INKLING_UBATCH", switch)
    backend, cmd = _inkling_launch(tmp_path)

    assert _ubatch_values(cmd) == [], cmd
    assert backend._n_ubatch == backend._DEFAULT_N_UBATCH


@pytest.mark.parametrize("arch", ["qwen3moe", "gemma4", "inkling"])
def test_other_moe_archs_keep_the_default_micro_batch(tmp_path, _inkling_host, arch):
    backend, cmd = _inkling_launch(tmp_path, arch = arch)

    assert _ubatch_values(cmd) == [], cmd
    assert backend._moe_spill_batch_restore is None


# 16 GB card, 12,000 MiB free, 8 GiB weights: default fits 97%; +4 GiB at -ub 1024 does not.
_TIGHT = dict(size_gib = 8, memory = [(0, 12_000, 16_000)], raise_cost_mib = 4096)


def test_a_raise_that_no_longer_fits_is_restored(tmp_path, _inkling_host, _inkling_log):
    backend, cmd = _inkling_launch(tmp_path, **_TIGHT)

    assert cmd[cmd.index("--fit") + 1] == "off", cmd
    assert _ubatch_values(cmd) == [] and _batch_values(cmd) == [], cmd
    assert backend._n_ubatch == backend._DEFAULT_N_UBATCH
    assert backend._moe_spill_batch_restore is None
    assert len(_inkling_log) == 1 and _inkling_log[0].startswith(
        "Inkling 2 on GPU: keeping --ubatch-size 512; 1024 (+4096 MiB compute buffer)"
    ), _inkling_log


def test_the_same_card_with_room_takes_the_raise(tmp_path, _inkling_host):
    _backend, cmd = _inkling_launch(tmp_path, **{**_TIGHT, "memory": [(0, 15_500, 16_000)]})

    assert _ubatch_values(cmd) == ["1024"], cmd


def test_the_switch_on_skips_the_headroom_gate(tmp_path, _inkling_host, monkeypatch):
    monkeypatch.setenv("UNSLOTH_INKLING_UBATCH", "on")
    _backend, cmd = _inkling_launch(tmp_path, **_TIGHT)

    assert _ubatch_values(cmd) == ["1024"], cmd


def test_the_raise_is_priced_at_the_served_slots(tmp_path, _inkling_host):
    seen: list = []
    backend, cmd = _inkling_launch(tmp_path, n_parallel = 4, record = seen)

    assert cmd[cmd.index("--parallel") + 1] == "4", cmd
    assert _ubatch_values(cmd) == ["1024"], cmd
    assert backend._n_ubatch == 1024
    assert (1024, 4) in seen, seen


def test_the_helper_only_raises():
    from core.inference.llama_cpp import _inkling_resident_batch_ubatch as raise_

    on = dict(architecture = "inkling2", experts_resident = True, user_named_batch = False)
    assert raise_(None, None, **on) == (None, 1024)
    assert raise_(None, 1120, **on) == (None, 1120)
    assert raise_(None, 2048, **on) == (None, 2048)
    # An unnamed batch below the target grows with it (llama.cpp caps ubatch at batch).
    assert raise_(512, None, **on) == (1024, 1024)
    assert raise_(None, None, **{**on, "architecture": "inkling2_loop"}) == (None, 1024)
    for off in (
        {"architecture": "qwen3moe"},
        {"architecture": None},
        {"experts_resident": False},
        {"user_named_batch": True},
    ):
        assert raise_(None, None, **{**on, **off}) == (None, None)


def test_the_switch_reads_like_the_moe_switches():
    switch = llama_cpp_module._moe_switch
    name = llama_cpp_module._INKLING_UBATCH_ENV

    assert name == "UNSLOTH_INKLING_UBATCH"
    assert switch(name, {}) == "auto"
    assert switch(name, {name: "OFF"}) == "off"
    assert switch(name, {name: "on"}) == "on"
    assert switch(name, {name: "bogus"}) == "auto"


def test_a_context_below_the_target_keeps_the_default(tmp_path, _inkling_host, _inkling_log):
    backend, cmd = _inkling_launch(tmp_path, n_ctx = 512)

    assert _ubatch_values(cmd) == [], cmd
    assert backend._moe_spill_batch_restore is None
    assert _inkling_log == []
