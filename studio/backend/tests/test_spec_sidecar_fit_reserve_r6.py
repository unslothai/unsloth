# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sixth-round cells for the DSpark/DFlash sidecar reserve and the Metal OOM episode.

A hinted explicit reload clears a terminal Metal OOM (the strike key ignores the per-request
cache hint), a drafter partly off the card (positive -ngld or a draft tensor override) is
neither reserved whole nor Auto-dropped for its whole file, the one-slot startup retry
reprices the reserve for one sequence, and a retry that lands on several GPUs carries no
scalar reserve.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_llama_cpp_placement as _placement  # noqa: E402
import test_spec_sidecar_fit_reserve_r2 as _r2  # noqa: E402
import test_spec_sidecar_fit_reserve_r4 as _r4  # noqa: E402
from core.inference import llama_cpp as _lc  # noqa: E402
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402

_GIB = 1024 * 1024 * 1024
_FLOOR = 1024 + 8 * 1024  # base margin + the 8 GiB sidecar
_HINT = ("someone/repo", "Q4_K_M", "/nonexistent/model.gguf", ())


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))


def _run(backend, gguf, *, stdout, healthy, **intent):
    """Every llama-server argv the load spawns; ``stdout(i, cmd)`` is spawn i's output and
    ``healthy(i, cmd)`` its health answer (an unhealthy child has exited with 1)."""
    cmds = []

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _placement._REAL_POPEN(cmd, **kwargs)
        cmds.append(list(cmd))
        rc = None if healthy(len(cmds) - 1, list(cmd)) else 1
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

    def fake_health(timeout = None, **_kw):
        i = len(cmds) - 1
        backend._stdout_lines = list(stdout(i, cmds[-1]))
        return healthy(i, cmds[-1])

    backend._wait_for_health = fake_health
    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        try:
            backend.load_model(
                GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", **intent)
            )
        except Exception:
            pass
    return cmds


# Finding 1/6: the verified-cache hint is per request; the strike key must not see it.


def test_the_strike_key_ignores_the_verified_cache_hint(tmp_path):
    backend, gguf = _r2._cuda_backend(tmp_path)
    plain = GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", n_ctx = 4096)
    hinted = GgufLoadIntent(
        gguf_path = str(gguf), model_identifier = "test", n_ctx = 4096, verified_gguf = _HINT
    )
    assert backend._metal_oom_key(hinted) == backend._metal_oom_key(plain)
    backend._metal_oom_terminal_key = backend._metal_oom_key(plain)
    assert backend._metal_oom_is_terminal(hinted)


def test_a_hinted_explicit_reload_clears_a_terminal_metal_oom(tmp_path):
    backend, gguf = _r2._cuda_backend(tmp_path)
    # The terminal key comes from a committed snapshot, which never carries the hint.
    snapshot = GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", n_ctx = 4096)
    key = backend._metal_oom_key(snapshot)
    backend._metal_oom_strikes = (key, 2)
    backend._metal_oom_terminal_key = key
    _run(
        backend,
        gguf,
        stdout = lambda i, cmd: (),
        healthy = lambda i, cmd: True,
        n_ctx = 4096,
        verified_gguf = _HINT,
    )
    assert backend._last_load_intent is not None
    assert backend._last_load_intent.verified_gguf is None
    assert backend._metal_oom_terminal_key is None
    assert backend._metal_oom_strikes is None
    assert not backend._metal_oom_is_terminal(backend._last_load_intent)


# Finding 3: a draft tensor override places the drafter per weight; no whole-file reserve.


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
    backend, gguf = _r2._cuda_backend(tmp_path)
    cmd = _placement._launch(
        backend,
        gguf,
        speculative_type = "dspark",
        dspark_draft_path = _r4._sidecar(tmp_path),
        extra_args = extras,
    )["cmd"]
    assert _r4._margin(cmd) < _FLOOR
    assert extras[-1] in cmd


def test_a_target_tensor_override_keeps_the_reserve(tmp_path):
    """Control: -ot places the target, not the drafter, so the drafter stays reserved."""
    backend, gguf = _r2._cuda_backend(tmp_path)
    cmd = _placement._launch(
        backend,
        gguf,
        speculative_type = "dspark",
        dspark_draft_path = _r4._sidecar(tmp_path),
        extra_args = ("-ot", "blk\\.0\\.ffn_up=CPU"),
    )["cmd"]
    assert _r4._margin(cmd) >= _FLOOR


# Finding 2: the Metal Auto drop charged a drafter split with the host its whole file.


def test_metal_auto_keeps_a_partially_offloaded_drafter(tmp_path, monkeypatch):
    """12 GiB holds the 4 GiB target alone but not target + the whole 4 GiB sidecar
    (test_spec_sidecar_fit_reserve.TestMetalLaunch); with -ngld 1 only one block is on
    the card, so Auto must not drop the drafter for the whole file."""
    cmd, backend = _r2._metal_cmd(
        tmp_path, monkeypatch, "auto", budget_gib = 12, extra_args = ["--spec-draft-ngl", "1"]
    )
    assert "--model-draft" in cmd
    assert backend.spec_fallback_reason != "drafter_no_vram"


def test_metal_auto_still_drops_a_whole_drafter_that_does_not_fit(tmp_path, monkeypatch):
    """Control for the cell above: without the split the same budget drops it."""
    cmd, backend = _r2._metal_cmd(tmp_path, monkeypatch, "auto", budget_gib = 12)
    assert "--model-draft" not in cmd
    assert backend.spec_fallback_reason == "drafter_no_vram"


# Finding 4: the one-slot startup retry reprices the reserve for one sequence.

_PER_SLOT = 1 * _GIB


def _slot_priced_backend(tmp_path):
    backend, gguf = _r2._cuda_backend(tmp_path)

    def estimate(
        n_ctx,
        n_parallel = 1,
        **kw,
    ):
        return 8 * _GIB + max(1, int(n_parallel or 1)) * _PER_SLOT

    backend._estimate_mtp_overhead_bytes = estimate
    return backend, gguf


def test_the_one_slot_retry_reprices_the_reserve(tmp_path):
    backend, gguf = _slot_priced_backend(tmp_path)
    refused = ("a unified KV cache is only supported with a single sequence",)
    cmds = _run(
        backend,
        gguf,
        stdout = lambda i, cmd: refused if i == 0 else (),
        healthy = lambda i, cmd: i > 0,
        n_ctx = 4096,
        n_parallel = 4,
        speculative_type = "dspark",
        dspark_draft_path = _r4._sidecar(tmp_path),
    )
    assert len(cmds) >= 2, [c[-6:] for c in cmds]
    first, retry = cmds[0], cmds[1]
    assert _r2._last(first, "--parallel") == "4"
    assert _r2._last(retry, "--parallel") in (None, "1")
    assert "--model-draft" in retry
    assert _r4._margin(first) >= _FLOOR + 4 * 1024
    # Still reserved for the drafter and its one slot, not for the three slots dropped.
    assert _FLOOR + 1024 <= _r4._margin(retry) < _FLOOR + 2 * 1024
    assert retry.count("--fit-target") <= first.count("--fit-target")


# Finding 5: the arch-crash retry re-masks onto several GPUs; the scalar reserve goes.


def test_an_arch_retry_onto_several_gpus_drops_the_scalar_reserve(tmp_path):
    rows = ((0, 24_000, 24_000), (1, 12_000, 12_000), (2, 12_000, 12_000))
    backend, gguf = _r2._cuda_backend(tmp_path, memory = rows)
    backend._select_gpus = lambda *a, **k: ([0], True)
    backend._arch_crash_retry_gpu_ids = lambda *a, **k: [1, 2]
    crash = ("ggml_cuda_compute_forward: ROCm error: device kernel image is invalid",)
    cmds = _run(
        backend,
        gguf,
        stdout = lambda i, cmd: crash if i == 0 else (),
        healthy = lambda i, cmd: i > 0,
        n_ctx = 4096,
        speculative_type = "dspark",
        dspark_draft_path = _r4._sidecar(tmp_path),
    )
    assert len(cmds) >= 2, [c[-6:] for c in cmds]
    assert _r4._margin(cmds[0]) >= _FLOOR
    assert "--model-draft" in cmds[1]
    assert _r4._margin(cmds[1]) < _FLOOR


# A llama.cpp whose fitter measures the borrowed-tensor draft itself needs no reserve.

_MARKER = b"measuring the extra model next to an allocation-free target context"


def _write(path, payload = b""):
    path.write_bytes(b"\x7fELF" + payload + b"\x00")
    return path


def test_native_measurement_is_read_from_the_shared_common_library(tmp_path):
    binary = _write(tmp_path / "llama-server")
    real = _write(tmp_path / "libllama-common.so.0.0.1", b"..." + _MARKER + b"...")
    (tmp_path / "libllama-common.so.0").symlink_to(real.name)
    (tmp_path / "libllama-common.so").symlink_to("libllama-common.so.0")
    assert _lc._fit_measures_borrowed_drafts(str(binary))


def test_native_measurement_is_read_from_a_static_server(tmp_path):
    binary = _write(tmp_path / "llama-server", _MARKER)
    assert _lc._fit_measures_borrowed_drafts(str(binary))


def test_an_older_build_does_not_claim_native_measurement(tmp_path):
    binary = _write(tmp_path / "llama-server")
    _write(
        tmp_path / "libllama-common.so",
        b"failed to measure the memory of the extra model, fitting without it",
    )
    assert not _lc._fit_measures_borrowed_drafts(str(binary))
    assert not _lc._fit_measures_borrowed_drafts(None)


def test_a_missing_binary_fails_closed(tmp_path):
    caps = LlamaCppBackend.probe_server_capabilities(str(tmp_path / "missing"))
    assert caps["fit_measures_borrowed_drafts"] is False


@pytest.mark.parametrize("native", [False, True], ids = ["older", "native"])
def test_the_reserve_stands_down_when_the_fitter_measures_the_draft(tmp_path, native):
    backend, gguf = _r2._cuda_backend(tmp_path)
    backend.probe_server_capabilities = lambda _binary = None: {
        **_r2._CAPS,
        "fit_measures_borrowed_drafts": native,
    }
    cmd = _placement._launch(
        backend, gguf, speculative_type = "dspark", dspark_draft_path = _r4._sidecar(tmp_path)
    )["cmd"]
    assert "--model-draft" in cmd
    if native:
        assert _r4._margin(cmd) < _FLOOR
    else:
        assert _r4._margin(cmd) >= _FLOOR


# Review round 7.

# The first spawn reprices from its final argv: pass-through slots raise the reserve.


@pytest.mark.parametrize(
    "extras",
    [("--parallel", "8"), ("-np", "8"), ("--parallel=8",)],
    ids = ["long", "short", "equals"],
)
def test_pass_through_slots_are_priced_on_the_first_spawn(tmp_path, extras):
    backend, gguf = _slot_priced_backend(tmp_path)
    cmd = _placement._launch(
        backend,
        gguf,
        n_ctx = 4096,
        n_parallel = 1,
        speculative_type = "dspark",
        dspark_draft_path = _r4._sidecar(tmp_path),
        extra_args = extras,
    )["cmd"]
    assert _r4._margin(cmd) >= _FLOOR + 8 * 1024


# Draft CPU-MoE switches become draft tensor overrides: the card share is unpriced.


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
    backend, gguf = _r2._cuda_backend(tmp_path)
    cmd = _placement._launch(
        backend,
        gguf,
        speculative_type = "dspark",
        dspark_draft_path = _r4._sidecar(tmp_path),
        extra_args = extras,
    )["cmd"]
    assert _r4._margin(cmd) < _FLOOR


@pytest.mark.parametrize(
    "env",
    [{"LLAMA_ARG_SPEC_DRAFT_CPU_MOE": "1"}, {"LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE": "8"}],
    ids = ["cpu-moe", "n-cpu-moe"],
)
def test_draft_cpu_moe_from_the_env_is_a_placement_too(env):
    assert _lc._draft_card_share_unknown([], env)
    assert not _lc._draft_card_share_unknown([], {"LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE": "0"})


def test_metal_auto_keeps_a_drafter_with_its_experts_on_cpu(tmp_path, monkeypatch):
    cmd, backend = _r2._metal_cmd(
        tmp_path, monkeypatch, "auto", budget_gib = 12, extra_args = ["--cpu-moe-draft"]
    )
    assert "--model-draft" in cmd
    assert backend.spec_fallback_reason != "drafter_no_vram"


# A surviving --device naming one card is a single-device launch on a multi-GPU host.


def test_one_named_device_on_a_two_gpu_host_keeps_the_reserve(tmp_path):
    backend, gguf = _r2._cuda_backend(
        tmp_path, memory = ((0, 24_000, 24_000), (1, 24_000, 24_000)), vulkan = True
    )
    cmd = _placement._launch(
        backend,
        gguf,
        speculative_type = "dspark",
        dspark_draft_path = _r4._sidecar(tmp_path),
        extra_args = ("--device", "Vulkan0"),
    )["cmd"]
    assert _r2._last(cmd, "--device") == "Vulkan0"
    assert _r4._margin(cmd) >= _FLOOR


def test_two_detected_gpus_without_a_device_pin_stay_unreserved(tmp_path):
    """Control: with nothing narrowing the launch it spans both cards."""
    backend, gguf = _r2._cuda_backend(
        tmp_path, memory = ((0, 24_000, 24_000), (1, 24_000, 24_000)), vulkan = True
    )
    cmd = _placement._launch(
        backend, gguf, speculative_type = "dspark", dspark_draft_path = _r4._sidecar(tmp_path)
    )["cmd"]
    assert _r4._margin(cmd) < _FLOOR


# Manual + Auto context with a positive inherited LLAMA_ARG_CTX_SIZE runs at that size.


def test_manual_auto_context_prices_an_inherited_context(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", "8192")
    backend, gguf = _r2._cuda_backend(tmp_path)
    backend._context_length = 131072
    per_token = 64 * 1024  # 8 GiB of draft KV at the native 131072, 0.5 GiB at 8192

    def estimate(n_ctx, **kw):
        return 8 * _GIB + max(0, int(n_ctx)) * per_token

    backend._estimate_mtp_overhead_bytes = estimate
    cmd = _placement._launch(
        backend,
        gguf,
        speculative_type = "dspark",
        dspark_draft_path = _r4._sidecar(tmp_path),
        gpu_memory_mode = "manual",
        gpu_layers = -1,
        n_ctx = 0,
    )["cmd"]
    assert "--fit-target" in cmd
    assert _r4._margin(cmd) < 512 + 12 * 1024


# A static server behind the installer's wrapper script carries the marker itself.


def test_native_measurement_is_read_through_the_installer_wrapper(tmp_path):
    _write(tmp_path / "llama-server-real", _MARKER)
    wrapper = tmp_path / "llama-server"
    wrapper.write_text('#!/bin/sh\nexec "$(dirname "$0")/llama-server-real" "$@"\n')
    wrapper.chmod(0o755)
    assert _lc._fit_measures_borrowed_drafts(str(wrapper))


def test_a_positive_draft_cpu_moe_count_is_not_undone_by_a_later_zero():
    """llama.cpp appends the draft CPU overrides per positive count; a later 0 adds none
    but removes none either."""
    assert _lc._draft_card_share_unknown(["--spec-draft-n-cpu-moe", "8", "-ncmoed", "0"], {})
    assert not _lc._draft_card_share_unknown(["-ncmoed", "0"], {})


# Review round 8.

# (A) An automatic context scales with the FINAL slots and KV layout.


def test_an_automatic_context_is_priced_at_the_final_slots(tmp_path):
    backend, gguf = _r2._cuda_backend(tmp_path)
    backend._context_length = 32768
    per_token = 32 * 1024  # 1 GiB of draft KV at 32768, 8 GiB at 8 x 32768

    def estimate(n_ctx, **kw):
        return 8 * _GIB + max(0, int(n_ctx)) * per_token

    backend._estimate_mtp_overhead_bytes = estimate
    cmd = _placement._launch(
        backend,
        gguf,
        speculative_type = "dspark",
        dspark_draft_path = _r4._sidecar(tmp_path),
        gpu_memory_mode = "manual",
        gpu_layers = -1,
        n_ctx = 0,
        n_parallel = 1,
        extra_args = ("--parallel", "8", "--no-kv-unified"),
    )["cmd"]
    assert "--fit-target" in cmd
    assert _r4._margin(cmd) >= 512 + 16 * 1024


# (C) The reserve falls when the final argv runs fewer slots than the request.


def test_fewer_final_slots_lower_the_reserve(tmp_path):
    backend, gguf = _slot_priced_backend(tmp_path)
    cmd = _placement._launch(
        backend,
        gguf,
        n_ctx = 4096,
        n_parallel = 8,
        speculative_type = "dspark",
        dspark_draft_path = _r4._sidecar(tmp_path),
        extra_args = ("--parallel", "1"),
    )["cmd"]
    assert _FLOOR + 1024 <= _r4._margin(cmd) < _FLOOR + 2 * 1024
    assert cmd.count("--fit-target") == 1


# (D) The draft context's own micro-batch sizes its compute.


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
    backend, gguf = _r2._cuda_backend(tmp_path)

    def estimate(
        n_ctx,
        n_ubatch = None,
        **kw,
    ):
        return 8 * _GIB + int(n_ubatch or 512) * 1024 * 1024

    backend._estimate_mtp_overhead_bytes = estimate
    cmd = _placement._launch(
        backend,
        gguf,
        n_ctx = 4096,
        speculative_type = "dspark",
        dspark_draft_path = _r4._sidecar(tmp_path),
        extra_args = extras,
    )["cmd"]
    assert _r4._margin(cmd) >= _FLOOR + 4096


def test_the_draft_micro_batch_is_capped_at_the_batch(tmp_path):
    """llama_context caps n_ubatch at n_batch; the draft context inherits the target's 2048."""
    backend, gguf = _r2._cuda_backend(tmp_path)
    backend._estimate_mtp_overhead_bytes = lambda n_ctx, n_ubatch = None, **kw: (
        8 * _GIB + int(n_ubatch or 512) * 1024 * 1024
    )
    cmd = _placement._launch(
        backend,
        gguf,
        n_ctx = 4096,
        speculative_type = "dspark",
        dspark_draft_path = _r4._sidecar(tmp_path),
        extra_args = ["--spec-draft-n-ubatch", "8192"],
    )["cmd"]
    assert _FLOOR + 2048 <= _r4._margin(cmd) < _FLOOR + 4096


def test_a_zero_draft_micro_batch_follows_the_target():
    assert _lc._draft_n_ubatch(["--spec-draft-n-ubatch", "0"], {}) is None
    assert _lc._draft_n_ubatch(["--spec-draft-n-ubatch=256"], {}) == 256


# Apple Silicon: one GPU. A draft pinned to it (MTL0) is reserved there; a pin to any
# other device is not on the Metal target, so it gets no reserve.


@pytest.mark.parametrize("pin, reserved", [("MTL0", True), ("mtl0", True), ("Vulkan0", False)])
def test_a_metal_draft_pin_is_reserved_only_on_the_metal_device(
    tmp_path, monkeypatch, pin, reserved
):
    cmd, _ = _r2._metal_cmd(
        tmp_path, monkeypatch, "dspark", budget_gib = 16, extra_args = ["--device-draft", pin]
    )
    assert "--model-draft" in cmd
    # The harness reads every GGUF as 4 GiB, the sidecar too.
    assert (_r4._margin(cmd) >= 1024 + 4 * 1024) is reserved


# Review round 8.


@pytest.mark.parametrize("devices", ["CUDA0,CUDA0", "CUDA0, cuda0"])
def test_a_repeated_device_is_one_card_for_the_reserve(tmp_path, devices):
    backend, gguf = _r2._cuda_backend(tmp_path)
    cmd = _placement._launch(
        backend,
        gguf,
        speculative_type = "dspark",
        dspark_draft_path = _r4._sidecar(tmp_path),
        extra_args = ["--device", devices],
    )["cmd"]
    assert _r4._margin(cmd) >= _FLOOR


@pytest.mark.parametrize(
    "pin, reserved", [("CUDA0", True), ("CUDA1", False), ("Vulkan0", False), ("RPC0", False)]
)
def test_a_cuda_draft_pin_on_the_only_target_card_keeps_the_reserve(tmp_path, pin, reserved):
    """The child's mask numbers the single target card 0, so CUDA0 is the target."""
    backend, gguf = _r2._cuda_backend(tmp_path)
    cmd = _placement._launch(
        backend,
        gguf,
        speculative_type = "dspark",
        dspark_draft_path = _r4._sidecar(tmp_path),
        extra_args = ["--device-draft", pin],
    )["cmd"]
    assert "--model-draft" in cmd
    assert (_r4._margin(cmd) >= _FLOOR) is reserved


def test_a_cached_probe_rereads_the_shared_library(tmp_path, monkeypatch):
    """The capability cache is keyed on the server alone; a swapped libllama-common
    must not keep an older answer."""
    binary = _write(tmp_path / "llama-server")
    st = binary.stat()
    key = (str(binary), st.st_mtime_ns, st.st_size)
    monkeypatch.setattr(LlamaCppBackend, "_exec_path_for_launch", staticmethod(lambda b: b))
    monkeypatch.setitem(
        LlamaCppBackend._capability_cache,
        key,
        {"found": True, "supports_fit_target": True, "fit_measures_borrowed_drafts": True},
    )
    assert (
        LlamaCppBackend.probe_server_capabilities(str(binary))["fit_measures_borrowed_drafts"]
        is False
    )
    _write(tmp_path / "libllama-common.so", _MARKER)
    assert (
        LlamaCppBackend.probe_server_capabilities(str(binary))["fit_measures_borrowed_drafts"]
        is True
    )


def _dsv41_dspark_pair():
    """DeepSeek-V4.1 target and its DSpark drafter, dims read off the published GGUFs."""
    target = LlamaCppBackend.__new__(LlamaCppBackend)
    target._embedding_length, target._vocab_size = 5120, 129280
    draft = LlamaCppBackend.__new__(LlamaCppBackend)
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
        + LlamaCppBackend._MTP_DRAFT_COMPUTE_BYTES
    ) / 2**20
    assert measured_mib <= got <= measured_mib * 1.25


def test_only_the_linked_common_library_speaks_for_the_build(tmp_path):
    """A stale versioned file beside the soname proves nothing about the running server."""
    binary = _write(tmp_path / "llama-server")
    _write(tmp_path / "libllama-common.so.0")
    _write(tmp_path / "libllama-common.so.0.9.9", _MARKER)
    assert not _lc._fit_measures_borrowed_drafts(str(binary))
    _write(tmp_path / "libllama-common.so.0", _MARKER)
    assert _lc._fit_measures_borrowed_drafts(str(binary))


def test_an_unsized_sidecar_carries_the_flat_reserve():
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._get_gguf_size_bytes = lambda path: None
    assert b._sidecar_fit_reserve_mib(0, "o/remote-dspark", flat_mib = 1200) == 1200
    # A sized one keeps its own bytes.
    assert b._sidecar_fit_reserve_mib(2 * 1024**3, "o/remote-dspark", flat_mib = 1200) == 2048
