# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Thirteenth-round cells for the DSpark/DFlash sidecar reserve and the Metal OOM latch.

Whether the sidecar is reserved is reported by the spawn, off the argv it launches, so the
report cannot disagree with the reserve (RPC, no --fit-target, an off-target pin under
--split-mode none, one-card tensor mode, a renumbered companion pin). Host-resident draft
KV follows a named CUDA / ROCm device. A caller's load starts a fresh Metal OOM episode,
and a diffusion load drops the chat server's recovery state.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_llama_cpp_placement as _placement  # noqa: E402
import test_spec_sidecar_fit_reserve_r2 as _r2  # noqa: E402
import test_spec_sidecar_fit_reserve_r4 as _r4  # noqa: E402
import test_spec_sidecar_fit_reserve_r7 as _r7  # noqa: E402
from core.inference import llama_cpp as _lc  # noqa: E402

_TWO = ((0, 24_000, 24_000), (1, 24_000, 24_000))


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("LLAMA_ARG_DEVICE", raising = False)
    monkeypatch.delenv("LLAMA_ARG_SPLIT_MODE", raising = False)
    monkeypatch.setattr(_lc, "_metal_capable_host", lambda: False)
    monkeypatch.setattr(
        _lc.LlamaCppBackend, "_apple_metal_memory_budget_bytes", staticmethod(lambda: 0)
    )


def _messages(monkeypatch):
    seen = []
    real_info = _lc.logger.info

    def info(msg, *args, **kwargs):
        try:
            seen.append(str(msg) % args if args else str(msg))
        except TypeError:
            seen.append(str(msg))
        return real_info(msg, *args, **kwargs)

    monkeypatch.setattr(_lc.logger, "info", info)
    return seen


def _cards(tmp_path, memory = _TWO):
    backend, gguf = _r2._cuda_backend(tmp_path, memory = memory)
    backend._arch_gate_survivors = lambda binary = None: []
    return backend, gguf


# The report follows the reserve the spawned argv carries.


@pytest.mark.parametrize(
    "memory, extras, caps_fit_target",
    [
        # --rpc devices come first in the child's list: no reserve on one local card.
        (((0, 24_000, 24_000),), ["--rpc", "10.0.0.2:50052"], True),
        # A server without --fit-target cannot be told.
        (((0, 24_000, 24_000),), [], False),
        # --split-mode none keeps the target on card 0, the drafter on card 1.
        (_TWO, ["--split-mode", "none", "--device-draft", "CUDA1"], True),
    ],
)
def test_a_reserve_the_spawn_drops_is_reported_as_missing(
    tmp_path, monkeypatch, memory, extras, caps_fit_target
):
    backend, gguf = _cards(tmp_path, memory)
    caps = dict(_r2._CAPS, supports_fit_target = caps_fit_target)
    backend.probe_server_capabilities = lambda _binary = None: dict(caps)
    seen = _r2._spy_fit_sized(backend)
    messages = _messages(monkeypatch)
    cmd = _r7._dspark(backend, gguf, tmp_path, extra_args = list(extras))
    assert max(_r7._margins(cmd)) < 1024 + 4 * 1024
    assert seen["fit_sized"] is False
    assert not any("keeps" in m and "free for it" in m for m in messages)
    assert any("is not reserved" in m for m in messages)


def test_a_dflash_reserve_the_spawn_drops_is_reported_as_missing(tmp_path, monkeypatch):
    backend, gguf = _cards(tmp_path, ((0, 24_000, 24_000),))
    caps = dict(_r2._CAPS, supports_dflash = True)
    backend.probe_server_capabilities = lambda _binary = None: dict(caps)
    messages = _messages(monkeypatch)
    sidecar = tmp_path / "dflash-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    cmd = _placement._launch(
        backend,
        gguf,
        speculative_type = "dflash",
        dflash_draft_path = str(sidecar),
        extra_args = ["--rpc", "10.0.0.2:50052"],
    )["cmd"]
    assert "draft-dflash" in cmd
    assert any(m.startswith("DFlash") and "is not reserved" in m for m in messages)
    assert not any("keeps" in m and "free for it" in m for m in messages)


def test_a_one_card_tensor_split_is_reserved_and_reported(tmp_path, monkeypatch):
    """Tensor mode folds several cards into one meta device; on one card there is nothing
    to fold, and the spawn reserves it as usual."""
    backend, gguf = _cards(tmp_path, ((0, 24_000, 24_000),))
    seen = _r2._spy_fit_sized(backend)
    messages = _messages(monkeypatch)
    cmd = _r7._dspark(backend, gguf, tmp_path, extra_args = ["--split-mode", "tensor"])
    assert _r4._margin(cmd) >= 1024 + 8 * 1024
    assert seen["fit_sized"] is True
    assert not any("is not reserved" in m for m in messages)


def test_a_renumbered_companion_pin_is_reported_as_reserved(tmp_path, monkeypatch):
    """gpu_ids [1, 2] masks the child to those cards, so the unpinned CUDA2 of the request
    reaches the child as CUDA1, which the spawn reserves."""
    backend, gguf = _cards(
        tmp_path, ((0, 24_000, 24_000), (1, 24_000, 24_000), (2, 24_000, 24_000))
    )
    seen = _r2._spy_fit_sized(backend)
    cmd = _r7._dspark(
        backend, gguf, tmp_path, gpu_ids = [1, 2], extra_args = ["--device-draft", "CUDA2"]
    )
    margins = _r7._margins(cmd)
    assert len(margins) == 2 and max(margins) >= 1024 + 8 * 1024
    assert seen["fit_sized"] is True


def test_the_start_log_shows_the_argv_spawned(tmp_path, monkeypatch):
    """The per-card reserve is spelled at spawn, so the logged command must be that one."""
    backend, gguf = _cards(tmp_path)
    messages = _messages(monkeypatch)
    cmd = _r7._dspark(backend, gguf, tmp_path)
    spelled = _r2._last(cmd, "--fit-target")
    assert "," in str(spelled)
    starts = [m for m in messages if m.startswith("Starting llama-server:")]
    assert starts and f"--fit-target {spelled}" in starts[-1]


# Host-resident draft KV under a named CUDA / ROCm device.


@pytest.mark.parametrize("device, discrete", [("CUDA1", True), ("ROCm1", True), ("CUDA0", False)])
def test_a_named_card_beside_an_apu_decides_host_kv(tmp_path, monkeypatch, device, discrete):
    """APU 0 plus discrete card 1, the child pointed at one by --device: -nkvo keeps the
    draft KV in host memory, which only the discrete card does not hold."""
    out = []
    for name, extras in (("on", []), ("off", ["-nkvo"])):
        d = tmp_path / name
        d.mkdir()
        backend, gguf = _cards(d)
        backend._mtp_draft_kv_bytes = lambda *a, **k: 768 * 1024 * 1024
        backend._amd_apu_wants_unified_memory = (
            lambda gpu_indices = None: gpu_indices is None or 0 in gpu_indices
        )
        backend._integrated_cuda_unified_memory = lambda gpu_indices = None: False
        names = [f"{device[:-1]}0", f"{device[:-1]}1"]
        monkeypatch.setattr(
            _lc.LlamaCppBackend,
            "_enumerated_gpu_devices",
            classmethod(lambda cls, binary = None, env = None, _n = names: list(_n)),
        )
        out.append(
            _r4._margin(
                _r7._dspark(
                    backend,
                    gguf,
                    d,
                    extra_args = ["--device", device, *extras],
                    gpu_memory_mode = "manual",
                    gpu_layers = -1,
                )
            )
        )
    on, off = out
    if discrete:
        assert on - off >= 760
    else:
        assert on == off


# Auto's oversized-drafter drop sees a drafter pinned across the target's cards.


def test_auto_drops_a_drafter_pinned_across_both_target_cards(tmp_path):
    backend, gguf, sidecar = _placement._tight_vram_backend(tmp_path, drafter_gb = 12.0)
    backend._get_gpu_memory = lambda _binary = None, **_kw: [(0, 12288, 12288), (1, 12288, 12288)]
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: [(0, 12288), (1, 12288)]
    backend._select_gpus = lambda *a, **k: ([0, 1], False)
    backend._select_gpus_split_aware = lambda *a, **k: ([0, 1], False)
    backend._arch_gate_survivors = lambda binary = None: []
    cmd = _placement._launch_auto_8k(
        backend, gguf, dspark_draft_path = str(sidecar), extra_args = ["--device-draft", "CUDA0,CUDA1"]
    )["cmd"]
    assert "--model-draft" not in cmd
    assert backend.spec_fallback_reason == "drafter_no_vram"


# Metal OOM episodes.


def _stub_child():
    return type(
        "Process",
        (),
        {
            "pid": 123,
            "poll": lambda self: 0,
            "kill": lambda self: None,
            "wait": lambda self, timeout = None: 0,
        },
    )()


def _strike(backend):
    dead = _stub_child()
    backend._process = dead
    backend._retired_process = dead
    backend._after_metal_poisoned_server(False)


def test_a_callers_load_starts_a_fresh_metal_oom_episode(tmp_path):
    """A (one strike), then the user loads B and A again: A's new session must get its own
    two strikes, not die on its first OOM."""
    backend, gguf = _r2._cuda_backend(tmp_path)
    backend._speculative_type = "none"
    _placement._launch(backend, gguf, n_ctx = 4096)
    _strike(backend)
    assert backend._metal_oom_strikes is not None
    _placement._launch(backend, gguf, n_ctx = 8192)
    assert backend._metal_oom_strikes is None
    _placement._launch(backend, gguf, n_ctx = 4096)
    _strike(backend)
    assert backend._metal_oom_strikes[1] == 1
    assert backend._metal_oom_terminal_key is None


def test_automatic_recovery_keeps_the_strike(tmp_path):
    """Control: the respawn and the drafter-free reload replay under the recovery marker,
    which carries the count, so a second OOM still stops respawning."""
    backend, gguf = _r2._cuda_backend(tmp_path)
    backend._speculative_type = "none"
    _placement._launch(backend, gguf, n_ctx = 4096)
    _strike(backend)
    with _lc._automatic_recovery_load():
        _placement._launch(backend, gguf, n_ctx = 4096)
    assert backend._metal_oom_strikes[1] == 1
    _strike(backend)
    assert backend._metal_oom_terminal_key is not None


def test_the_respawn_and_drafter_free_reload_run_as_automatic_recovery():
    import inspect

    respawn = inspect.getsource(_lc.LlamaCppBackend._respawn_if_dead)
    assert "with _automatic_recovery_load():" in respawn
    recover = inspect.getsource(_lc.LlamaCppBackend._maybe_recover_from_mtp_crash)
    assert "with _automatic_recovery_load():" in recover


def test_a_diffusion_load_drops_the_chat_servers_recovery_state(tmp_path, monkeypatch):
    """A crashed diffusion runner must neither replay the chat load it replaced nor meet that
    load's Metal OOM latch."""
    import test_gpu_memory_mode as _gm

    real_init = _lc.LlamaCppBackend.__init__

    def init(self, *a, **k):
        real_init(self, *a, **k)
        self._last_load_intent = object()
        self._mtp_runtime_fallback_active = True
        self._metal_oom_strikes = ("chat", 2)
        self._metal_oom_terminal_key = "chat"

    monkeypatch.setattr(_lc.LlamaCppBackend, "__init__", init)
    _cmd, _env, backend = _gm._launch_diffusion(
        monkeypatch, tmp_path, gguf_gib = 4.0, gpus = _gm._TWO_3090, gpu_ids = None
    )
    assert backend._last_load_intent is None
    assert backend._mtp_runtime_fallback_active is False
    assert backend._metal_oom_strikes is None
    assert backend._metal_oom_terminal_key is None
