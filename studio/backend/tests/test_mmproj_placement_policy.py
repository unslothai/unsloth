# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Projector placement order: GPU, then CPU pin, then drop the drafter; vision off means no projector."""

from __future__ import annotations

import inspect
import os
import struct
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.inference.llama_cpp import (
    GgufLoadIntent,
    LlamaCppBackend,
    _AUTO_OFFLOAD_CTX,
    _resolved_mmproj_offload,
)
import utils.models.gguf_metadata as _meta
from models.inference import InferenceStatusResponse, LoadResponse
from routes.inference import (
    _estimate_gguf_required_gb,
    _guard_chat_load_against_training,
    _llama_runtime_fields,
    _load_keeps_a_projector,
    _LoadPlacement,
)

MIB = 1024 * 1024
GIB = 1024**3
_REAL_POPEN = subprocess.Popen

# Fits at 4096 but not native: pricing residency at native length is the bug guarded here.
NATIVE_CTX = 262144
KV_PER_TOKEN = 64 * 1024  # 4096 ctx -> 256 MiB, NATIVE_CTX -> 16 GiB


def _write_gguf(path: Path) -> Path:
    def string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    metadata = string("general.architecture") + struct.pack("<I", 8) + string("llama")
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)
    return path


def _write_drafter_gguf(path: Path, *, with_token_embd: bool = True) -> Path:
    import numpy as np
    from gguf import GGUFWriter

    writer = GGUFWriter(str(path), "qwen35")
    names = ["output.weight", "blk.64.nextn.eh_proj.weight"]
    if with_token_embd:
        names += ["token_embd.weight", "output_norm.weight"]
    for name in names:
        writer.add_tensor(name, np.zeros((2, 2), dtype = np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _backend(
    tmp_path: Path,
    *,
    memory,
    model_bytes: int = 6 * GIB,
    mmproj_bytes: int = 1 * GIB,
    drafter_bytes: int = 0,
    native_ctx: int = NATIVE_CTX,
):
    """A GGUF vision load with pinned fit inputs; drafter tests lower native_ctx so draft KV can't
    decide."""
    backend = LlamaCppBackend()
    gguf = _write_gguf(tmp_path / "model.gguf")
    mmproj = _write_gguf(tmp_path / "mmproj-F16.gguf")
    drafter = _write_drafter_gguf(tmp_path / "mtp.gguf")

    def read_metadata(_path):
        backend._context_length = native_ctx
        backend._n_layers = 32
        backend._n_heads = 32
        backend._n_kv_heads = 8
        backend._embedding_length = 4096
        backend._vocab_size = 32000

    backend._read_gguf_metadata = read_metadata
    backend._get_gpu_memory = lambda _binary = None, **_kw: list(memory)
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: [
        (index, free) for index, free, _total in memory
    ]
    backend._estimate_kv_cache_bytes = lambda ctx, *_a, **_kw: max(0, ctx) * KV_PER_TOKEN
    backend._compute_buffer_ctx_bytes = lambda *_a, **_kw: 0
    backend._estimate_compute_buffer_bytes = lambda **_kw: 256 * MIB
    backend._get_gguf_size_bytes = lambda path: (
        drafter_bytes if Path(path).name == "mtp.gguf" else model_bytes
    )
    backend._mmproj_vram_bytes = lambda _path: mmproj_bytes
    backend._resolve_launch_mmproj_path = lambda **_kw: str(mmproj)
    # Only the speculative test wants a drafter; elsewhere MTP must not engage.
    backend._resolve_launch_mtp_path = lambda **_kw: str(drafter) if drafter_bytes else None
    backend._apu_ram_shortfall_message = lambda *_a, **_kw: None
    backend._amd_apu_wants_unified_memory = lambda *_a, **_kw: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: False
    backend._wait_for_health = lambda timeout, **_kw: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _detected: True
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_no_mmproj_offload": True,
        "mtp_token": "draft-mtp",
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }
    return backend, gguf


def _launch(backend, gguf, **load_kwargs):
    captured = {}

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)
        captured["cmd"] = list(cmd)
        captured["env"] = kwargs.get("env") or dict(os.environ)
        return type(
            "Process",
            (),
            {
                "pid": 123,
                "stdout": (),
                "poll": lambda self: None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    intent_kwargs = {
        "is_vision": True,
        # Auto context (0): a pinned 4096 would hide the floor question.
        "n_ctx": 0,
        **load_kwargs,
    }
    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(
            GgufLoadIntent(
                gguf_path = str(gguf),
                model_identifier = "test",
                **intent_kwargs,
            )
        )
    return captured


def test_projector_stays_on_gpu_when_it_fits_at_the_floor(tmp_path):
    """Context shrinks rather than spilling layers, so pinning would buy nothing and cost ~8.8x per
    image."""
    backend, gguf = _backend(tmp_path, memory = [(0, 12_000, 24_000)])

    cmd = _launch(backend, gguf)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" not in cmd
    assert cmd[cmd.index("--fit") + 1] == "off"
    assert int(cmd[cmd.index("-c") + 1]) < NATIVE_CTX


def test_projector_pinned_to_cpu_when_it_does_not_fit(tmp_path):
    """_MMPROJ_VRAM_SAFETY alone decides the pin; --mmproj still goes out, only its offload is disabled."""
    backend, gguf = _backend(tmp_path, memory = [(0, 8_692, 16_384)])

    cmd = _launch(backend, gguf)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" in cmd
    assert cmd[cmd.index("--fit") + 1] == "off"


def test_user_owns_the_placement_when_they_name_either_spelling(tmp_path):
    """llama.cpp is last-wins on the placement pair, so an explicit
    --mmproj-offload must not be raced by the automatic pin."""
    backend, gguf = _backend(tmp_path, memory = [(0, 8_692, 16_384)])

    cmd = _launch(backend, gguf, extra_args = ["--mmproj-offload"])["cmd"]

    assert cmd.count("--no-mmproj-offload") == 0


def test_vision_switched_off_loads_no_projector_anywhere(tmp_path, monkeypatch):
    """Not on the GPU, not on the CPU, and not through an inherited env var:
    common/arg.cpp reads LLAMA_ARG_MMPROJ straight into params.mmproj.path."""
    monkeypatch.setenv("LLAMA_ARG_MMPROJ", "/ambient/mmproj.gguf")
    monkeypatch.setenv("LLAMA_ARG_MMPROJ_URL", "https://example.invalid/mmproj.gguf")
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])

    result = _launch(backend, gguf, disable_vision = True)

    assert "--mmproj" not in result["cmd"]
    assert "--no-mmproj-offload" not in result["cmd"]
    assert "LLAMA_ARG_MMPROJ" not in result["env"]
    assert "LLAMA_ARG_MMPROJ_URL" not in result["env"]
    assert backend.is_vision is False


@pytest.mark.parametrize(
    "memory,label",
    [
        ([], "apple_unified"),
        ([(0, 7_600, 0)], "amd_apu_or_igpu"),
    ],
)
def test_shared_memory_pools_are_not_charged_as_discrete(tmp_path, memory, label):
    """Apple Silicon enumerates no GPU and an APU / iGPU reports free SYSTEM RAM
    as its free VRAM with a total of 0. Moving the encoder inside one pool frees
    nothing, so the same shortfall that pins a discrete card must not pin here."""
    backend, gguf = _backend(tmp_path, memory = memory)

    cmd = _launch(backend, gguf)["cmd"]

    assert "--no-mmproj-offload" not in cmd, label


DRAFTER_NATIVE_CTX = 8192


def _drafter_backend(tmp_path, memory):
    return _backend(
        tmp_path,
        memory = memory,
        drafter_bytes = 2 * GIB,
        native_ctx = DRAFTER_NATIVE_CTX,
    )


def _launch_with_drafter(backend, gguf, tmp_path):
    return _launch(
        backend,
        gguf,
        mtp_draft_path = str(tmp_path / "mtp.gguf"),
        speculative_type = "auto",
    )["cmd"]


def test_the_projector_is_pinned_before_the_drafter_is_dropped(tmp_path):
    """Pin the projector before dropping the drafter, since per-image cost is cheaper to concede."""
    backend, gguf = _drafter_backend(tmp_path, [(0, 12_470, 24_000)])

    cmd = _launch_with_drafter(backend, gguf, tmp_path)

    assert "--no-mmproj-offload" in cmd
    assert "--model-draft" in cmd


def test_both_are_given_up_when_pinning_alone_is_not_enough(tmp_path):
    """Step 3: the drafter goes too, but only after the projector has moved and
    the load still does not fit. Budget 8200 MiB against about 9250 for model +
    drafter with the projector already pinned."""
    backend, gguf = _drafter_backend(tmp_path, [(0, 8_692, 16_384)])

    cmd = _launch_with_drafter(backend, gguf, tmp_path)

    assert "--no-mmproj-offload" in cmd
    assert "--model-draft" not in cmd


def test_the_drafters_vram_is_part_of_the_pin_decision(tmp_path):
    """The pin runs with the drafter live, so its VRAM must be charged or the predicate is wrong."""
    memory = [(0, 12_000, 24_000)]

    with_drafter, gguf = _drafter_backend(tmp_path, memory)
    pinned = _launch_with_drafter(with_drafter, gguf, tmp_path)

    without_drafter, gguf2 = _backend(tmp_path, memory = memory, native_ctx = DRAFTER_NATIVE_CTX)
    unpinned = _launch(without_drafter, gguf2)["cmd"]

    assert "--no-mmproj-offload" in pinned
    assert "--model-draft" in pinned
    assert "--no-mmproj-offload" not in unpinned


@pytest.mark.parametrize(
    "is_vision,disable_vision,expect_disabled,expect_by_user",
    [
        (True, True, True, True),
        (False, True, True, False),
        (True, False, False, False),
        (False, False, False, False),
    ],
)
def test_load_and_status_both_report_the_vision_toggle(
    tmp_path, is_vision, disable_vision, expect_disabled, expect_by_user
):
    """Both fields must be on both responses; a missing one makes ?? false reseed the switch off."""
    backend, gguf = _backend(tmp_path, memory = [(0, 12_000, 24_000)])
    _launch(backend, gguf, is_vision = is_vision, disable_vision = disable_vision)

    fields = _llama_runtime_fields(backend)
    load = LoadResponse(
        status = "loaded", model = "test", display_name = "test", inference = {}, **fields
    ).model_dump()
    status = InferenceStatusResponse(active_model = "test", **fields).model_dump()

    for payload, where in ((load, "load"), (status, "status")):
        assert "disable_vision" in payload, where
        assert "vision_disabled_by_user" in payload, where
        assert payload["disable_vision"] is expect_disabled, where
        assert payload["vision_disabled_by_user"] is expect_by_user, where


def test_the_training_guard_does_not_charge_a_projector_the_load_will_not_open(tmp_path):
    """The switch is used on constrained machines, which is exactly where this
    guard bites: charging VRAM the load provably never takes would refuse a chat
    load for the memory the user just freed."""
    model = tmp_path / "model.gguf"
    model.write_bytes(b"\x00" * (4 * MIB))
    mmproj = tmp_path / "mmproj-F16.gguf"
    mmproj.write_bytes(b"\x00" * (1 * MIB))
    config = SimpleNamespace(
        gguf_file = str(model),
        gguf_mmproj_file = str(mmproj),
        gguf_mtp_file = None,
        gguf_dspark_file = None,
        gguf_dflash_file = None,
        gguf_hf_repo = None,
        gguf_variant = None,
        is_vision = True,
    )

    charged = _estimate_gguf_required_gb(config)
    freed = _estimate_gguf_required_gb(config, disable_vision = True)

    assert charged is not None and freed is not None
    assert round((charged - freed) * 1024) == 1
    assert freed < charged


def test_the_training_guard_forwards_the_switch_to_its_estimator(tmp_path):
    """Asserts the switch reaches the estimator keyword, not the verdict, so only wiring is tested."""
    seen = {}

    def capture(_config, **kwargs):
        seen.update(kwargs)
        return 0.0

    training = SimpleNamespace(is_training_active = lambda: True)
    request = SimpleNamespace(
        hf_token = None,
        max_seq_length = 0,
        speculative_type = None,
        cache_type_kv = None,
        gpu_memory_mode = "auto",
        gpu_layers = -1,
        tensor_parallel = False,
        n_parallel = 1,
        disable_vision = True,
    )
    config = SimpleNamespace(is_gguf = True, gguf_file = None, gguf_hf_repo = None)
    placement = _LoadPlacement(
        requested_gpu_ids = None,
        resolved_gpu_ids = None,
        gpu_ids_are_vulkan_ordinals = False,
        diffusion_kind = False,
    )

    with (
        patch("core.training.get_training_backend", lambda: training),
        patch("routes.inference._estimate_gguf_required_gb", side_effect = capture),
        patch.object(LlamaCppBackend, "_find_llama_server_binary", lambda *_a, **_k: None),
        patch.object(LlamaCppBackend, "_effective_gpu_count", lambda *_a, **_k: 1),
    ):
        try:
            _guard_chat_load_against_training(
                config, request, load_in_4bit = False, placement = placement
            )
        except Exception:
            pass

    assert seen.get("disable_vision") is True


@pytest.mark.parametrize(
    "has_audio,accepts_image,projector_expected,label",
    [
        (True, False, True, "audio_only_keeps_the_projector"),
        (True, True, False, "omni_honors_the_switch"),
        (False, True, False, "vision_only_honors_the_switch"),
    ],
)
def test_the_vision_switch_does_not_take_audio_only_projectors_away(
    tmp_path, has_audio, accepts_image, projector_expected, label
):
    """Audio-only projectors (ultravox, Voxtral, Qwen3-ASR) are kept, since dropping one frees no VRAM."""
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])
    with patch.object(_meta, "mmproj_capabilities", lambda _p: (has_audio, accepts_image)):
        cmd = _launch(backend, gguf, disable_vision = True)["cmd"]

    assert ("--mmproj" in cmd) is projector_expected, label


# The pin reads only probe outputs, no platform flag. 7600 MiB free pins on a discrete card,
# so a "no pin" cell means shared memory, not room.
_TOPOLOGIES = [
    ([(0, 7_600, 8_192)], True, "linux_nvidia_discrete"),
    ([(0, 7_600, 8_192)], True, "windows_nvidia_discrete"),
    ([(0, 7_600, 8_192)], True, "wsl_nvidia_discrete"),
    ([(0, 7_600, 8_192)], True, "linux_amd_discrete_rocm"),
    ([(0, 7_600, 8_192)], True, "windows_amd_discrete_vulkan"),
    ([(0, 7_600, 0)], False, "linux_amd_apu"),
    ([(0, 7_600, 0)], False, "windows_amd_igpu"),
    ([(0, 7_600, 0)], False, "wsl_amd_igpu"),
    ([(0, 7_600, 0)], False, "linux_intel_igpu"),
    ([], False, "mac_apple_silicon_unified"),
    ([], False, "linux_cpu_only"),
    ([], False, "windows_cpu_only"),
    ([], False, "wsl_cpu_only"),
    ([], False, "mac_cpu_only"),
]


@pytest.mark.parametrize("memory,expect_pin,label", _TOPOLOGIES)
def test_the_platform_and_gpu_matrix_pins_only_where_memory_is_discrete(
    tmp_path, memory, expect_pin, label
):
    """Moving the encoder out of a shared pool frees nothing, so only a card with
    its own memory may be charged for the projector. An APU, an iGPU and a
    virtualised Metal device all report free SYSTEM RAM as free VRAM."""
    backend, gguf = _backend(tmp_path, memory = memory)

    cmd = _launch(backend, gguf)["cmd"]

    assert ("--no-mmproj-offload" in cmd) is expect_pin, label
    assert "--mmproj" in cmd, label


@pytest.mark.parametrize(
    "memory,label",
    [
        ([(0, 7_600, 8_192), (1, 20_000, 24_576)], "big_card_second"),
        ([(0, 20_000, 24_576), (1, 7_600, 8_192)], "big_card_first"),
    ],
)
def test_a_heterogeneous_pair_is_ranked_before_the_projector_is_charged(tmp_path, memory, label):
    """Enumeration order must not decide placement: the same two cards in either
    order have to reach the same answer, or the pin is reading the device list
    rather than the memory on it."""
    backend, gguf = _backend(tmp_path, memory = memory)

    cmd = _launch(backend, gguf)["cmd"]

    assert "--mmproj" in cmd, label
    assert ("--no-mmproj-offload" in cmd) is False, label


def test_a_remembered_mmproj_auto_does_not_survive_the_vision_switch(tmp_path):
    """--mmproj-auto survives suppressing --mmproj and env, so the disable flag must come after extras."""
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])

    cmd = _launch(backend, gguf, disable_vision = True, extra_args = ["--mmproj-auto"])["cmd"]

    assert "--no-mmproj-auto" in cmd
    assert cmd.index("--no-mmproj-auto") > cmd.index("--mmproj-auto")


def test_an_audio_only_projector_is_not_taken_away_by_the_auto_override(tmp_path):
    """The override exists to stop a projector coming back. An audio-only one is kept
    on purpose, so --no-mmproj-auto must not follow it out the door."""
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])
    with patch.object(_meta, "mmproj_capabilities", lambda _p: (True, False)):
        cmd = _launch(backend, gguf, disable_vision = True, extra_args = ["--mmproj-auto"])["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-auto" not in cmd


def test_an_audio_only_projector_does_not_blame_the_switch_for_images(tmp_path):
    """vision_disabled_by_user drives the composer's "you turned it off" message, so
    on a model with no image encoder it would promise a capability that turning the
    switch back on cannot deliver."""
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])
    with patch.object(_meta, "mmproj_capabilities", lambda _p: (True, False)):
        _launch(backend, gguf, disable_vision = True)

    assert backend._disable_vision is True
    assert backend._vision_disabled_by_user is False


def test_the_training_guard_still_charges_an_audio_only_projector(tmp_path):
    """Audio-only projectors are still charged: the loader keeps them, and under-charging over-admits."""
    model = tmp_path / "model.gguf"
    model.write_bytes(b"\x00" * (4 * MIB))
    mmproj = tmp_path / "mmproj-F16.gguf"
    mmproj.write_bytes(b"\x00" * (1 * MIB))
    config = SimpleNamespace(
        gguf_file = str(model),
        gguf_mmproj_file = str(mmproj),
        gguf_mtp_file = None,
        gguf_dspark_file = None,
        gguf_dflash_file = None,
        gguf_hf_repo = None,
        gguf_variant = None,
        is_vision = True,
    )

    with patch.object(_meta, "mmproj_accepts_image", lambda _p: False):
        audio_only = _estimate_gguf_required_gb(config, disable_vision = True)
    with patch.object(_meta, "mmproj_accepts_image", lambda _p: True):
        vision = _estimate_gguf_required_gb(config, disable_vision = True)
    charged = _estimate_gguf_required_gb(config)

    assert audio_only is not None and vision is not None and charged is not None
    assert audio_only == charged
    assert vision < charged


def test_the_download_interlock_is_not_relaxed_by_the_vision_switch(tmp_path):
    """Interlock mirrors the download gate, not the switch: a vision-off load must still get the 409."""
    from core.inference.llama_cpp import GgufLoadIntent, _with_gguf_load_marker

    seen = {}

    def fake_blocks(
        repo,
        variant,
        *,
        require_mmproj,
        hf_token = None,
    ):
        seen["require_mmproj"] = require_mmproj
        return False

    def inner(
        self,
        intent,
        load_cancel_event = None,
    ):
        return True

    def _run(**intent_kwargs):
        seen.clear()
        _with_gguf_load_marker(inner)(
            object(),
            GgufLoadIntent(
                gguf_path = str(tmp_path / "model.gguf"),
                model_identifier = "test",
                hf_repo = "unsloth/some-vl-GGUF",
                is_vision = True,
                **intent_kwargs,
            ),
        )
        return seen["require_mmproj"]

    with patch("core.inference.llama_cpp._hub_download_blocks_gguf_load", fake_blocks):
        assert _run(disable_vision = True) is True
        assert _run(disable_vision = False) is True
        assert _run(disable_vision = True, extra_args = ["--no-mmproj"]) is False


def test_a_user_pinned_projector_is_not_charged_against_vram(tmp_path):
    """--no-mmproj-offload puts the projector in host RAM, so its bytes are not on
    the card. Charging them anyway shrank the context and spilled layers to make
    room for VRAM nothing occupies, which is worse placement than Unsloth's own."""
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])

    cmd = _launch(backend, gguf, extra_args = ["--no-mmproj-offload"])["cmd"]

    assert "--mmproj" in cmd
    assert "--fit" in cmd and cmd[cmd.index("--fit") + 1] == "off"
    assert cmd[cmd.index("-c") + 1] == "9984"


def test_a_user_demanding_gpu_offload_still_pays_for_it(tmp_path):
    """The mirror: --mmproj-offload asks for the projector ON the card, so its bytes
    stay in the budget and the context shrinks to fit them. Resolving the value is
    what separates this from the case above; merely detecting ownership cannot."""
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])

    cmd = _launch(backend, gguf, extra_args = ["--mmproj-offload"])["cmd"]

    # Lands on the Auto offload fallback; compare to the constant, not a literal.
    assert cmd[cmd.index("-c") + 1] == str(_AUTO_OFFLOAD_CTX)


def test_the_last_placement_spelling_is_what_gets_budgeted(tmp_path):
    """llama.cpp folds the pair into one option and takes the last occurrence, so a
    list ending in the disable form must budget as disabled however it starts."""
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])

    cmd = _launch(
        backend,
        gguf,
        extra_args = ["--mmproj-offload", "--no-mmproj-offload"],
    )["cmd"]

    assert cmd[cmd.index("-c") + 1] == "9984"


def test_the_projector_probe_agrees_with_the_layer_loop_it_gates(tmp_path):
    """Probe and layer loop must agree: the projector stays on GPU only if every model layer is placed."""
    backend, gguf = _backend(
        tmp_path,
        memory = [(0, 6_000, 8_192), (1, 6_000, 8_192)],
        model_bytes = 9 * GIB,
    )

    cmd = _launch(backend, gguf)["cmd"]

    pinned = "--no-mmproj-offload" in cmd
    fitted = "--fit" in cmd and cmd[cmd.index("--fit") + 1] == "off"
    assert pinned or fitted, (
        "the probe left the projector on the GPU and the fit then could not place "
        f"the model: {[c for c in cmd if 'fit' in str(c) or 'mmproj' in str(c)]}"
    )


def test_a_remote_projector_of_unknown_kind_is_charged_to_the_guard(tmp_path):
    """A remote projector of unknown kind is charged, since under-charging admits loads that need VRAM."""
    seen = {}

    def fake_companions(repo, *, hf_token, include_mmproj, **kw):
        seen["include_mmproj"] = include_mmproj
        return 0

    config = SimpleNamespace(
        gguf_file = None,
        gguf_mmproj_file = None,
        gguf_mtp_file = None,
        gguf_dspark_file = None,
        gguf_dflash_file = None,
        gguf_hf_repo = "unsloth/some-vl-GGUF",
        gguf_variant = "UD-Q4_K_XL",
        is_vision = True,
    )
    variant = SimpleNamespace(quant = "UD-Q4_K_XL", size_bytes = 4 * GIB)

    with (
        patch("routes.inference._remote_gguf_companion_bytes", fake_companions),
        patch(
            "utils.models.model_config.list_gguf_variants",
            lambda *a, **k: ([variant], True),
        ),
    ):
        _estimate_gguf_required_gb(config, disable_vision = True)

    assert seen.get("include_mmproj") is True


def test_the_training_guard_charges_a_hand_added_repo_root_projector(tmp_path):
    """A hand-added projector is charged though the listing cannot see it."""
    projector = tmp_path / "mmproj-F16.gguf"
    projector.write_bytes(b"\x00" * (3 * MIB))
    seen = {}

    def fake_companions(
        repo,
        *,
        hf_token,
        include_mmproj,
        local_mmproj_bytes = 0,
        **kw,
    ):
        seen["local_mmproj_bytes"] = local_mmproj_bytes
        return max(int(local_mmproj_bytes), 0)

    config = SimpleNamespace(
        gguf_file = None,
        gguf_mmproj_file = None,
        gguf_local_mmproj_file = str(projector),
        gguf_mtp_file = None,
        gguf_dspark_file = None,
        gguf_dflash_file = None,
        gguf_hf_repo = "unsloth/some-vl-GGUF",
        gguf_variant = "UD-Q4_K_XL",
        is_vision = True,
    )
    variant = SimpleNamespace(quant = "UD-Q4_K_XL", size_bytes = 4 * GIB)

    with (
        patch("routes.inference._remote_gguf_companion_bytes", fake_companions),
        patch(
            "utils.models.model_config.list_gguf_variants",
            lambda *a, **k: ([variant], False),
        ),
    ):
        charged = _estimate_gguf_required_gb(config)

    assert seen.get("local_mmproj_bytes") == 3 * MIB
    assert charged is not None and charged > 4 * GIB / (1024**3)


@pytest.mark.parametrize(
    "kwargs,accepts_image,charged,label",
    [
        ({}, True, True, "no switch charges it"),
        ({"disable_vision": True}, True, False, "the switch suppresses an image tower"),
        ({"disable_vision": True}, False, True, "an audio encoder survives the switch"),
        ({"llama_extra_args": ["--no-mmproj"]}, True, False, "the extras opt-out resolves none"),
    ],
)
def test_the_guard_asks_the_local_projector_whether_the_launch_opens_it(
    tmp_path, kwargs, accepts_image, charged, label
):
    """A local projector the Vision switch suppresses is not charged."""
    projector = tmp_path / "mmproj-F16.gguf"
    projector.write_bytes(b"\x00" * (3 * MIB))
    seen = {}

    def fake_companions(
        repo,
        *,
        hf_token,
        include_mmproj,
        local_mmproj_bytes = 0,
        **kw,
    ):
        seen["local_mmproj_bytes"] = local_mmproj_bytes
        return max(int(local_mmproj_bytes), 0)

    config = SimpleNamespace(
        gguf_file = None,
        gguf_mmproj_file = None,
        gguf_local_mmproj_file = str(projector),
        gguf_mtp_file = None,
        gguf_dspark_file = None,
        gguf_dflash_file = None,
        gguf_hf_repo = "unsloth/some-vl-GGUF",
        gguf_variant = "UD-Q4_K_XL",
        is_vision = True,
    )
    variant = SimpleNamespace(quant = "UD-Q4_K_XL", size_bytes = 4 * GIB)
    import utils.models.gguf_metadata as _meta

    with (
        patch("routes.inference._remote_gguf_companion_bytes", fake_companions),
        patch.object(_meta, "mmproj_accepts_image", lambda _p: accepts_image),
        patch(
            "utils.models.model_config.list_gguf_variants",
            lambda *a, **k: ([variant], False),
        ),
    ):
        _estimate_gguf_required_gb(config, **kwargs)

    assert seen.get("local_mmproj_bytes") == (3 * MIB if charged else 0), label


def test_gpu_ownership_reads_the_remote_configs_own_projector(tmp_path):
    """A suppressed local image tower does not claim the GPU."""
    projector = tmp_path / "mmproj-F16.gguf"
    projector.write_bytes(b"\x00" * MIB)
    config = SimpleNamespace(
        is_vision = True,
        gguf_mmproj_file = None,
        gguf_local_mmproj_file = str(projector),
    )
    import utils.models.gguf_metadata as _meta

    with patch.object(_meta, "mmproj_accepts_image", lambda _p: True):
        assert _load_keeps_a_projector(config, disable_vision = True) is False
        assert _load_keeps_a_projector(config, disable_vision = False) is True

    with patch.object(_meta, "mmproj_accepts_image", lambda _p: False):
        assert _load_keeps_a_projector(config, disable_vision = True) is True


def _ambient_mmproj(tmp_path, monkeypatch):
    ambient = tmp_path / "ambient-mmproj.gguf"
    ambient.write_bytes(b"\x00" * (1 * MIB))
    monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(ambient))
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])
    backend._resolve_launch_mmproj_path = lambda **_kw: None
    return backend, gguf


def test_the_vision_switch_does_not_record_an_inherited_audio_encoder(tmp_path, monkeypatch):
    """Don't record audio from an inherited LLAMA_ARG_MMPROJ the switch scrubs, or the composer
    offers it."""
    backend, gguf = _ambient_mmproj(tmp_path, monkeypatch)

    with patch.object(_meta, "mmproj_capabilities", lambda _p: (True, True)):
        result = _launch(backend, gguf, disable_vision = True)

    assert "LLAMA_ARG_MMPROJ" not in result["env"]
    assert backend._mmproj_has_audio is False


def test_the_vision_switch_keeps_an_inherited_audio_only_encoder(tmp_path, monkeypatch):
    """Audio-only inherited projectors (ultravox, Voxtral, Qwen3-ASR) stay: scrubbing them frees no VRAM."""
    backend, gguf = _ambient_mmproj(tmp_path, monkeypatch)

    with patch.object(_meta, "mmproj_capabilities", lambda _p: (True, False)):
        result = _launch(backend, gguf, disable_vision = True)

    assert result["env"].get("LLAMA_ARG_MMPROJ")
    assert backend._mmproj_has_audio is True
    assert backend._mmproj_accepts_image is False
    # --no-mmproj-auto does not unload, but makes the router advertise text-only.
    assert "--no-mmproj-auto" not in result["cmd"]


def test_an_inherited_projector_that_reads_images_still_goes(tmp_path, monkeypatch):
    """The asymmetry is deliberate: only a readable audio-only declaration is kept.
    An image-capable one is exactly what the switch is for."""
    backend, gguf = _ambient_mmproj(tmp_path, monkeypatch)

    with patch.object(_meta, "mmproj_capabilities", lambda _p: (False, True)):
        result = _launch(backend, gguf, disable_vision = True)

    assert "LLAMA_ARG_MMPROJ" not in result["env"]


def test_a_diffusion_runtime_is_not_torn_down_over_the_vision_switch(tmp_path):
    """Diffusion records the switch as False, so comparing it would reload an identical runtime."""
    from core.inference.llama_cpp import (
        GgufLoadIntent,
        LlamaCppBackend,
        _resolved_mmproj_offload,
    )

    backend = LlamaCppBackend()
    backend._is_diffusion = True
    backend._disable_vision = False
    backend._gguf_path = str(tmp_path / "diffusion.gguf")
    # Enough state for the comparison under test to be reached.
    backend._requested_n_ctx = 4096
    backend._cache_type_kv = None

    def _intent(disable_vision: bool):
        return GgufLoadIntent(
            gguf_path = str(tmp_path / "diffusion.gguf"),
            model_identifier = "test",
            disable_vision = disable_vision,
        )

    assert backend._runtime_matches_intent(_intent(True), None) == (
        backend._runtime_matches_intent(_intent(False), None)
    )


def test_an_advanced_argument_that_drops_the_projector_is_not_blamed_on_the_switch(tmp_path):
    """--no-mmproj in extras suppresses the projector by argument, so the switch is not to blame."""
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])

    _launch(backend, gguf, disable_vision = True, extra_args = ["--no-mmproj"])

    assert backend._disable_vision is True
    assert backend._vision_disabled_by_user is False


def test_a_projector_the_resolve_rejected_is_not_blamed_on_the_switch(tmp_path):
    """A None launch path is not the switch's doing; only a projector the switch itself dropped counts."""
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])
    backend._resolve_launch_mmproj_path = lambda **_kw: None

    _launch(backend, gguf, disable_vision = True)

    assert backend._disable_vision is True
    assert backend._vision_disabled_by_user is False


def test_an_explicit_context_is_priced_at_the_length_it_asked_for(tmp_path):
    """An explicit context is honored verbatim, so it is priced at that length, not the floor."""
    backend, gguf = _backend(tmp_path, memory = [(0, 12_000, 24_000)])

    cmd = _launch(backend, gguf, n_ctx = 65536)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" in cmd
    assert cmd[cmd.index("--fit") + 1] == "off"
    assert cmd[cmd.index("-c") + 1] == "65536"


def test_an_explicit_context_that_fits_with_the_projector_keeps_it_on_the_gpu(tmp_path):
    """A small explicit context that fits keeps the projector on GPU; pinning buys nothing there."""
    backend, gguf = _backend(tmp_path, memory = [(0, 12_000, 24_000)])

    cmd = _launch(backend, gguf, n_ctx = 8192)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" not in cmd
    assert cmd[cmd.index("--fit") + 1] == "off"
    assert cmd[cmd.index("-c") + 1] == "8192"


def test_an_environment_owned_placement_is_not_reversed_by_the_pin(tmp_path, monkeypatch):
    """Env applies before argv, so a user-set LLAMA_ARG_MMPROJ_OFFLOAD must not be reversed by the pin."""
    monkeypatch.setenv("LLAMA_ARG_MMPROJ_OFFLOAD", "1")
    backend, gguf = _backend(tmp_path, memory = [(0, 8_692, 16_384)])

    cmd = _launch(backend, gguf)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" not in cmd


def test_an_environment_pinned_projector_is_not_charged_against_vram(tmp_path, monkeypatch):
    """The mirror of the flag case: LLAMA_ARG_MMPROJ_OFFLOAD=0 puts the projector in
    host RAM just as --no-mmproj-offload does, so budgeting its bytes shrinks the
    context for VRAM nothing occupies."""
    monkeypatch.setenv("LLAMA_ARG_MMPROJ_OFFLOAD", "0")
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])

    cmd = _launch(backend, gguf)["cmd"]

    assert "--mmproj" in cmd
    assert cmd[cmd.index("--fit") + 1] == "off"
    assert cmd[cmd.index("-c") + 1] == "9984"


def test_the_negative_environment_spelling_pins_on_presence_alone(tmp_path, monkeypatch):
    """get_value_from_env checks the LLAMA_ARG_NO_ compatibility spelling first and
    forces falsey on getenv returning non-null, so an empty value still pins. Read
    any other way this charges VRAM the child never allocates."""
    monkeypatch.setenv("LLAMA_ARG_NO_MMPROJ_OFFLOAD", "")
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])

    cmd = _launch(backend, gguf)["cmd"]

    assert cmd[cmd.index("-c") + 1] == "9984"


@pytest.mark.parametrize(
    ("extras", "env", "expected"),
    [
        ([], {}, None),
        ([], {"LLAMA_ARG_MMPROJ_OFFLOAD": "1"}, True),
        ([], {"LLAMA_ARG_MMPROJ_OFFLOAD": "enabled"}, True),
        ([], {"LLAMA_ARG_MMPROJ_OFFLOAD": "off"}, False),
        # arg.cpp raises on a value that is neither, so there is no side to budget.
        ([], {"LLAMA_ARG_MMPROJ_OFFLOAD": "yes"}, None),
        # Presence, not value, and it wins over the positive spelling.
        ([], {"LLAMA_ARG_NO_MMPROJ_OFFLOAD": ""}, False),
        ([], {"LLAMA_ARG_MMPROJ_OFFLOAD": "1", "LLAMA_ARG_NO_MMPROJ_OFFLOAD": "0"}, False),
        # argv is parsed after the environment, so the flag wins either direction.
        (["--mmproj-offload"], {"LLAMA_ARG_MMPROJ_OFFLOAD": "0"}, True),
        (["--no-mmproj-offload"], {"LLAMA_ARG_MMPROJ_OFFLOAD": "1"}, False),
        (["--mmproj-offload"], {"LLAMA_ARG_NO_MMPROJ_OFFLOAD": "1"}, True),
    ],
)
def test_the_resolved_placement_follows_arg_cpps_own_precedence(extras, env, expected):
    """Environment first, argv on top, the negative spelling short-circuiting on
    presence. Anything else and Unsloth budgets for a placement the child does not
    run."""
    assert _resolved_mmproj_offload(extras, env) is expected


def test_an_unparseable_environment_value_is_still_the_callers_placement(tmp_path, monkeypatch):
    """No side to budget for, but the variable is set, so Unsloth must not append its
    own spelling on top: common_params_parse throws on the value and the load fails
    naming the caller's variable, not an Unsloth flag they never chose."""
    monkeypatch.setenv("LLAMA_ARG_MMPROJ_OFFLOAD", "yes")
    backend, gguf = _backend(tmp_path, memory = [(0, 8_692, 16_384)])

    cmd = _launch(backend, gguf)["cmd"]

    assert "--no-mmproj-offload" not in cmd


def test_an_explicit_context_too_large_for_either_still_gives_the_projector_up_first(tmp_path):
    """Even with the projector in host RAM the fit spills layers, so the projector is given up first."""
    backend, gguf = _backend(tmp_path, memory = [(0, 12_000, 24_000)])

    cmd = _launch(backend, gguf, n_ctx = 131072)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" in cmd
    assert cmd[cmd.index("--fit") + 1] == "on"


# Context-compute buffer: linear in ctx, _CTX_COMPUTE_SPLIT_MULT larger per device when split.
_CC_PER_TOKEN = 1536  # 6 MiB at 4096, the rate the bundled estimator produces


def _split_rate_backend(tmp_path, *, memory, **kwargs):
    backend, gguf = _backend(tmp_path, memory = memory, **kwargs)
    backend._compute_buffer_ctx_bytes = (
        lambda n_ctx, n_ubatch = None, cache_type_kv = None, *, layer_split = False, **_kw: (
            n_ctx * _CC_PER_TOKEN * (LlamaCppBackend._CTX_COMPUTE_SPLIT_MULT if layer_split else 1)
        )
    )
    return backend, gguf


def test_the_probe_prices_an_explicit_context_the_way_the_split_placement_does(tmp_path):
    """On two cards, an explicit context must price the compute buffer per device, as split
    placement does."""
    backend, gguf = _split_rate_backend(tmp_path, memory = [(0, 7_200, 8_200), (1, 7_200, 8_200)])

    cmd = _launch(backend, gguf, n_ctx = 65536)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" in cmd
    assert cmd[cmd.index("--fit") + 1] == "off"
    assert cmd[cmd.index("-c") + 1] == "65536"


def test_a_single_card_explicit_load_is_untouched_by_the_split_rate(tmp_path):
    """No split, no replication, so the split-aware selector must reach the same
    answer the plain one did. Guards the tightening from leaking onto one-GPU loads,
    where `_select_gpus_split_aware` returns before its retry."""
    backend, gguf = _split_rate_backend(tmp_path, memory = [(0, 12_000, 24_000)])

    cmd = _launch(backend, gguf, n_ctx = 8192)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" not in cmd
    assert cmd[cmd.index("-c") + 1] == "8192"


def test_auto_is_priced_at_the_split_rate_too_but_still_at_the_floor(tmp_path):
    """Auto is priced at the split rate; what keeps it honest is the floor, which shrinking cannot
    rescue."""
    backend, gguf = _split_rate_backend(tmp_path, memory = [(0, 4_900, 5_900), (1, 4_900, 5_900)])

    cmd = _launch(backend, gguf)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" in cmd


def test_auto_still_leaves_a_roomy_split_alone(tmp_path):
    """The floor is what stops the split rate from turning into a blanket pin: on cards
    with room at 4096 the projector stays on the GPU and Auto pays for the native
    context in context, exactly as it does on one card."""
    backend, gguf = _split_rate_backend(tmp_path, memory = [(0, 12_000, 16_000), (1, 12_000, 16_000)])

    cmd = _launch(backend, gguf)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" not in cmd


def test_the_probe_reserves_the_compute_buffer_on_every_split_device(tmp_path):
    """Compute buffer counts per device as well as in the total, or an unfit split is accepted."""
    backend, gguf = _split_rate_backend(tmp_path, memory = [(0, 6_000, 7_000), (1, 6_000, 7_000)])

    cmd = _launch(backend, gguf, n_ctx = 32768)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" in cmd
    assert cmd[cmd.index("--fit") + 1] == "off"
    assert cmd[cmd.index("-c") + 1] == "32768"


def test_the_predicted_pin_is_reported_through_both_responses(tmp_path):
    """A projector moved to host RAM by the fit must be reported on the startup-recovery pin's channel."""
    backend, gguf = _backend(tmp_path, memory = [(0, 8_692, 16_384)])
    cmd = _launch(backend, gguf)["cmd"]

    assert "--no-mmproj-offload" in cmd
    assert backend.mmproj_fallback_reason == "cpu_offload"

    fields = _llama_runtime_fields(backend)
    load = LoadResponse(
        status = "loaded", model = "test", display_name = "test", inference = {}, **fields
    ).model_dump()
    status = InferenceStatusResponse(active_model = "test", **fields).model_dump()
    for payload, where in ((load, "load"), (status, "status")):
        assert payload["mmproj_fallback_reason"] == "cpu_offload", where


def test_a_load_that_keeps_the_projector_on_the_gpu_reports_nothing(tmp_path):
    """The control: the reason is a report of something that happened, so a load that
    never moved the projector must leave it None rather than always claiming CPU."""
    backend, gguf = _backend(tmp_path, memory = [(0, 12_000, 24_000)])
    cmd = _launch(backend, gguf)["cmd"]

    assert "--no-mmproj-offload" not in cmd
    assert backend.mmproj_fallback_reason is None


@pytest.mark.parametrize(
    ("env", "expected_retry"),
    [
        # get_value_from_env checks LLAMA_ARG_NO_ first and forces falsey on presence.
        ({"LLAMA_ARG_NO_MMPROJ_OFFLOAD": "1"}, False),
        ({"LLAMA_ARG_NO_MMPROJ_OFFLOAD": ""}, False),
        ({"LLAMA_ARG_MMPROJ_OFFLOAD": "1", "LLAMA_ARG_NO_MMPROJ_OFFLOAD": "0"}, False),
        # is_falsey accepts `disabled`; the spelling list alone does not.
        ({"LLAMA_ARG_MMPROJ_OFFLOAD": "disabled"}, False),
        ({"LLAMA_ARG_MMPROJ_OFFLOAD": "enabled"}, True),
        ({}, True),
    ],
)
def test_the_recovery_retry_sees_every_environment_pin(env, expected_retry):
    """A projector the environment already pinned to CPU gets no retry; re-pinning changes nothing."""
    cmd = ["llama-server", "-m", "/cache/model.gguf", "--mmproj", "/cache/mmproj.gguf"]
    retry = LlamaCppBackend._with_mmproj_offload_disabled(cmd, env)

    assert (retry is not None) is expected_retry
    if expected_retry:
        assert retry[-1] == "--no-mmproj-offload"


def test_the_speculative_reserve_is_normalized_before_anything_prices_it(tmp_path):
    """Normalize CPU drafters before the projector probe prices the reserve; _mtp_bytes reads it lazily."""
    source = Path(inspect.getsourcefile(LlamaCppBackend)).read_text()
    normalize_at = source.index("if _draft_cpu_no_embedded and mtp_overhead_fn is not None:")
    probe_at = source.index("_mm_mtp_on_gpu = _mtp_will_engage and not _draft_cpu_no_embedded")
    assert (
        normalize_at < probe_at
    ), "the CPU-drafter reserve must be normalized before the projector probe prices it"


def test_a_shared_device_beside_a_discrete_one_does_not_veto_the_pin(tmp_path):
    """A shared device with total 0 beside a discrete card must not veto the projector pin."""
    backend, gguf = _backend(tmp_path, memory = [(0, 8_692, 16_384), (1, 7_600, 0)])

    cmd = _launch(backend, gguf)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" in cmd


def test_a_host_that_is_only_shared_memory_still_never_pins(tmp_path):
    """The rule that survives: with no budgeted device anywhere, moving the projector
    shuffles bytes inside one pool and frees nothing, so the 8.8x image-encode cost
    buys exactly nothing. Unchanged by the mixed-host relaxation above."""
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 0), (1, 7_600, 0)])

    cmd = _launch(backend, gguf)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" not in cmd


def test_a_user_pinned_projector_still_costs_a_shared_pool(tmp_path):
    """On a shared APU pool, --no-mmproj-offload frees nothing, so the projector still costs the pool."""
    memory = [(0, 9_000, 0)]

    backend, gguf = _backend(tmp_path, memory = memory)
    cmd = _launch(backend, gguf, extra_args = ["--no-mmproj-offload"])["cmd"]
    pinned_ctx = int(cmd[cmd.index("-c") + 1])

    reference, ref_gguf = _backend(tmp_path, memory = memory, model_bytes = 7 * GIB, mmproj_bytes = 0)
    ref_cmd = _launch(reference, ref_gguf)["cmd"]
    reference_ctx = int(ref_cmd[ref_cmd.index("-c") + 1])

    assert pinned_ctx == reference_ctx


def test_a_pinned_projector_costs_the_shared_pool_beside_a_discrete_card(tmp_path):
    """Fit prefixes can start at a shared pool, so that pool must still be charged the pinned projector."""
    memory = [(0, 9_000, 0), (1, 100, 16_384)]

    backend, gguf = _backend(tmp_path, memory = memory)
    cmd = _launch(backend, gguf, extra_args = ["--no-mmproj-offload"])["cmd"]
    pinned_ctx = int(cmd[cmd.index("-c") + 1])

    reference, ref_gguf = _backend(tmp_path, memory = memory, model_bytes = 7 * GIB, mmproj_bytes = 0)
    ref_cmd = _launch(reference, ref_gguf)["cmd"]

    assert pinned_ctx == int(ref_cmd[ref_cmd.index("-c") + 1])


def _estimator_config(model_path, mmproj_path = None):
    return SimpleNamespace(
        gguf_file = str(model_path),
        gguf_mmproj_file = str(mmproj_path) if mmproj_path else None,
        gguf_mtp_file = None,
        gguf_dspark_file = None,
        gguf_dflash_file = None,
        gguf_hf_repo = None,
        gguf_variant = None,
        is_vision = True,
    )


def test_the_guard_charges_a_projector_only_the_environment_names(tmp_path, monkeypatch):
    """An inherited LLAMA_ARG_MMPROJ stays GPU-resident with Vision off, so the guard must charge it."""
    model = tmp_path / "model.gguf"
    model.write_bytes(b"\x00" * (4 * MIB))
    ambient = tmp_path / "ambient-mmproj.gguf"
    ambient.write_bytes(b"\x00" * (1 * MIB))
    config = _estimator_config(model)

    bare = _estimate_gguf_required_gb(config, disable_vision = True)
    monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(ambient))

    with patch.object(_meta, "mmproj_capabilities", lambda _p: (True, False)):
        charged = _estimate_gguf_required_gb(config, disable_vision = True)
    with patch.object(_meta, "mmproj_capabilities", lambda _p: (False, True)):
        dropped = _estimate_gguf_required_gb(config, disable_vision = True)

    assert bare is not None and charged is not None and dropped is not None
    assert round((charged - bare) * 1024) == 1
    assert dropped == bare


def test_studios_own_projector_outranks_the_inherited_one_in_the_estimate(tmp_path, monkeypatch):
    """argv beats the environment (arg.cpp applies set_env first), so exactly one
    projector loads. Charging both billed a single file twice and refused loads
    that fit."""
    model = tmp_path / "model.gguf"
    model.write_bytes(b"\x00" * (4 * MIB))
    resolved = tmp_path / "mmproj-F16.gguf"
    resolved.write_bytes(b"\x00" * (1 * MIB))
    ambient = tmp_path / "ambient-mmproj.gguf"
    ambient.write_bytes(b"\x00" * (2 * MIB))

    expected = _estimate_gguf_required_gb(_estimator_config(model, resolved))
    monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(ambient))
    charged = _estimate_gguf_required_gb(_estimator_config(model, resolved))

    assert expected is not None and charged is not None
    assert charged == expected


def test_a_suppressed_image_projector_hands_the_budget_to_the_inherited_one(tmp_path, monkeypatch):
    """Suppressing the image projector leaves the inherited LLAMA_ARG_MMPROJ loading, so charge it."""
    model = tmp_path / "model.gguf"
    model.write_bytes(b"\x00" * (4 * MIB))
    configured = tmp_path / "mmproj-F16.gguf"
    configured.write_bytes(b"\x00" * (1 * MIB))
    ambient = tmp_path / "ambient-mmproj.gguf"
    ambient.write_bytes(b"\x00" * (2 * MIB))

    def _caps(path):
        return (False, True) if str(path) == str(configured) else (True, False)

    monkeypatch.delenv("LLAMA_ARG_MMPROJ", raising = False)
    with (
        patch.object(_meta, "mmproj_capabilities", _caps),
        patch.object(_meta, "mmproj_accepts_image", lambda p: _caps(p)[1]),
    ):
        weights_only = _estimate_gguf_required_gb(
            _estimator_config(model, configured), disable_vision = True
        )
        monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(ambient))
        charged = _estimate_gguf_required_gb(
            _estimator_config(model, configured), disable_vision = True
        )

    assert charged is not None and weights_only is not None
    assert round((charged - weights_only) * 1024) == 2


def test_the_extras_opt_out_does_not_excuse_an_inherited_projector(tmp_path, monkeypatch):
    """The load gates on mmproj.path, not no_mmproj, so --no-mmproj does not excuse an inherited file."""
    model = tmp_path / "model.gguf"
    model.write_bytes(b"\x00" * (4 * MIB))
    ambient = tmp_path / "ambient-mmproj.gguf"
    ambient.write_bytes(b"\x00" * (1 * MIB))
    config = _estimator_config(model)

    bare = _estimate_gguf_required_gb(config)
    monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(ambient))
    charged = _estimate_gguf_required_gb(config, llama_extra_args = ["--no-mmproj"])

    assert bare is not None and charged is not None
    assert round((charged - bare) * 1024) == 1


def test_the_extras_opt_out_moves_the_charge_to_the_inherited_projector(tmp_path, monkeypatch):
    """--no-mmproj skips the configured projector but not an inherited path, so charge that one instead."""
    model = tmp_path / "model.gguf"
    model.write_bytes(b"\x00" * (4 * MIB))
    configured = tmp_path / "mmproj-F16.gguf"
    configured.write_bytes(b"\x00" * (1 * MIB))
    ambient = tmp_path / "ambient-mmproj.gguf"
    ambient.write_bytes(b"\x00" * (2 * MIB))
    config = _estimator_config(model, configured)

    monkeypatch.delenv("LLAMA_ARG_MMPROJ", raising = False)
    normal = _estimate_gguf_required_gb(config)
    weights_only = _estimate_gguf_required_gb(_estimator_config(model))
    opted_out = _estimate_gguf_required_gb(config, llama_extra_args = ["--no-mmproj"])

    monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(ambient))
    inherited = _estimate_gguf_required_gb(config, llama_extra_args = ["--no-mmproj"])

    for value in (normal, weights_only, opted_out, inherited):
        assert value is not None
    assert round((normal - weights_only) * 1024) == 1
    assert opted_out == weights_only
    assert round((inherited - weights_only) * 1024) == 2


def _paravirtual(monkeypatch):
    import core.inference.llama_cpp as _llama_cpp
    monkeypatch.setattr(_llama_cpp, "_metal_device_is_paravirtual", lambda: True)


def test_a_virtualised_metal_device_does_not_keep_the_inherited_projector(tmp_path, monkeypatch):
    """A virtualised Metal device scrubs both projector env vars after the switch, so none survives."""
    ambient = tmp_path / "ambient-mmproj.gguf"
    ambient.write_bytes(b"\x00" * (1 * MIB))
    monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(ambient))
    _paravirtual(monkeypatch)
    backend, gguf = _backend(tmp_path, memory = [])
    backend._resolve_launch_mmproj_path = lambda **_kw: None

    with patch.object(_meta, "mmproj_capabilities", lambda _p: (True, False)):
        result = _launch(backend, gguf, disable_vision = True, extra_args = ["--mmproj-auto"])

    assert "LLAMA_ARG_MMPROJ" not in result["env"]
    assert "--no-mmproj-auto" in result["cmd"]
    assert backend._mmproj_has_audio is False


def test_dropping_an_inherited_image_projector_points_at_the_switch(tmp_path, monkeypatch):
    """Turning Vision back on restores an inherited image projector, so the composer
    must name the switch rather than send the user hunting for a valid mmproj."""
    ambient = tmp_path / "ambient-mmproj.gguf"
    ambient.write_bytes(b"\x00" * (1 * MIB))
    monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(ambient))
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])
    backend._resolve_launch_mmproj_path = lambda **_kw: None

    with patch.object(_meta, "mmproj_capabilities", lambda _p: (False, True)):
        _launch(backend, gguf, disable_vision = True)

    assert backend._vision_disabled_by_user is True


def test_a_stale_inherited_path_does_not_blame_the_switch(tmp_path, monkeypatch):
    """A path that names no file drops nothing, so turning Vision back on changes
    nothing either. Blaming the switch there points at a control that cannot help."""
    monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(tmp_path / "not-on-disk.gguf"))
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])
    backend._resolve_launch_mmproj_path = lambda **_kw: None

    _launch(backend, gguf, disable_vision = True)

    assert backend._vision_disabled_by_user is False


def test_a_cpu_recovery_records_the_vision_state_it_launched_with(tmp_path):
    """CPU recovery must record the Vision state it launched with, or unchanged disable_vision reloads."""
    backend = LlamaCppBackend()
    backend._disable_vision = False
    backend._vision_disabled_by_user = False
    intent = GgufLoadIntent(
        gguf_path = str(_write_gguf(tmp_path / "model.gguf")),
        model_identifier = "test",
        is_vision = True,
        disable_vision = True,
    )

    backend._apply_cpu_fallback_state(
        intent,
        is_vision = False,
        mmproj_has_audio = False,
        disable_vision = True,
        vision_disabled_by_user = True,
    )

    assert backend._disable_vision is True
    assert backend._vision_disabled_by_user is True
    assert backend._cpu_fallback_reason == "vulkan_startup_crash"


def test_both_cpu_recovery_call_sites_pass_the_vision_state(tmp_path):
    """Two call sites reach that helper and only one is on the common path, so a
    keyword added to one and not the other is a silent half-fix. Checked at the source
    because the second site needs a crash inside a replay this harness cannot stage."""
    source = inspect.getsource(LlamaCppBackend.load_model)
    calls = source.count("self._apply_cpu_fallback_state(")
    assert calls == 2, f"expected 2 recovery call sites, found {calls}"
    # The load's own assignment uses the same words, so subtract it.
    keyword_uses = source.count("vision_disabled_by_user = bool(") - source.count(
        "self._vision_disabled_by_user = bool("
    )
    assert keyword_uses == calls
    assert source.count("disable_vision = disable_vision,") == calls


@pytest.mark.parametrize("cache_type_kv", [None, "q8_0"])
def test_a_tensor_load_downgraded_to_layer_split_still_gives_the_projector_up(
    tmp_path, cache_type_kv
):
    """A tensor load downgraded to layer split must apply the projector probe, moving the encoder to CPU."""
    backend, gguf = _backend(tmp_path, memory = [(0, 4_400, 8_192), (1, 4_400, 8_192)])

    cmd = _launch(backend, gguf, tensor_parallel = True, cache_type_kv = cache_type_kv)["cmd"]

    # Reachable only via deferred application: tensor_parallel skips the probe-site verdict.
    assert "--no-mmproj-offload" in cmd
    assert "--mmproj" in cmd
    assert cmd[cmd.index("--fit") + 1] == "off"


def test_a_surviving_tensor_load_keeps_its_projector(tmp_path):
    """A surviving tensor load keeps its projector: layer-split probe numbers do not apply to it."""
    backend, gguf = _backend(tmp_path, memory = [(0, 4_800, 16_384), (1, 4_800, 16_384)])

    cmd = _launch(backend, gguf, tensor_parallel = True)["cmd"]

    assert cmd[cmd.index("--split-mode") + 1] == "tensor"
    assert "--no-mmproj-offload" not in cmd


def test_a_gpu_drafter_holds_the_deferred_pin_back(tmp_path):
    """A GPU drafter defers the projector pin: pinning first pays the encoder and still reaches --fit on."""
    backend, gguf = _drafter_backend(tmp_path, [(0, 4_400, 8_192), (1, 4_400, 8_192)])

    cmd = _launch(
        backend,
        gguf,
        tensor_parallel = True,
        mtp_draft_path = str(tmp_path / "mtp.gguf"),
        speculative_type = "auto",
    )["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" not in cmd


def test_an_unloadable_drafter_is_not_charged_before_it_is_dropped(tmp_path):
    """Judged after the fit, the drafter's 2 GiB would push model + drafter + projector
    past this 10550 MiB budget and pin the projector for a file that is dropped anyway."""
    backend, gguf = _drafter_backend(tmp_path, [(0, 12_470, 24_000)])
    _write_drafter_gguf(tmp_path / "mtp.gguf", with_token_embd = False)
    del backend._resolve_launch_mtp_path

    cmd = _launch_with_drafter(backend, gguf, tmp_path)

    assert "--model-draft" not in cmd
    assert "--no-mmproj-offload" not in cmd
    assert backend.mtp_draft_suppressed_path == str(tmp_path / "mtp.gguf")


def test_a_replayed_context_places_the_projector_like_a_fresh_forced_drafter(tmp_path):
    memory = [(0, 13_500, 24_000)]

    def load(**kwargs):
        backend, gguf = _backend(tmp_path, memory = memory, drafter_bytes = 2 * GIB, native_ctx = 65536)
        cmd = _launch(backend, gguf, mtp_draft_path = str(tmp_path / "mtp.gguf"), **kwargs)["cmd"]
        return backend, cmd, int(cmd[cmd.index("-c") + 1])

    _, auto, replayed = load(speculative_type = "auto", n_ctx = 0)
    assert "--model-draft" not in auto
    _, fresh, fresh_ctx = load(speculative_type = "mtp", n_ctx = 0)
    backend, replay, replay_ctx = load(
        speculative_type = "mtp", n_ctx = replayed, max_seq_length_auto_derived = True
    )

    assert "--no-mmproj-offload" not in fresh
    assert "--no-mmproj-offload" not in replay
    assert replay_ctx == fresh_ctx < replayed
    assert backend._requested_n_ctx == replay_ctx


def test_a_replayed_context_refits_for_a_cpu_pinned_drafter_too(tmp_path):
    # --spec-draft-ngl 0 keeps the drafter off the GPU; the hybrid target still pays rollback state.
    memory = [(0, 10_000, 24_000)]
    extras = ["--spec-draft-ngl", "0"]

    def load(**kwargs):
        backend, gguf = _backend(tmp_path, memory = memory, drafter_bytes = 2 * GIB, native_ctx = 65536)
        backend._rollback_state_bytes = lambda n_parallel = 1, *_a, **_kw: n_parallel * 256 * MIB
        cmd = _launch(
            backend,
            gguf,
            mtp_draft_path = str(tmp_path / "mtp.gguf"),
            extra_args = extras,
            **kwargs,
        )["cmd"]
        return cmd, int(cmd[cmd.index("-c") + 1])

    _, replayed = load(speculative_type = "off", n_ctx = 0)
    fresh, fresh_ctx = load(speculative_type = "mtp", n_ctx = 0)
    replay, replay_ctx = load(
        speculative_type = "mtp", n_ctx = replayed, max_seq_length_auto_derived = True
    )

    assert replay_ctx == fresh_ctx < replayed
    assert ("--no-mmproj-offload" in replay) == ("--no-mmproj-offload" in fresh)


@pytest.mark.parametrize("flag", ["--mmproj", "-mm"])
@pytest.mark.parametrize("vision_off", [False, True])
def test_custom_projector_replaces_discovery_and_respects_vision_switch(tmp_path, flag, vision_off):
    backend, gguf = _backend(tmp_path, memory = [(0, 12_000, 24_000)])
    backend._resolve_launch_mmproj_path = LlamaCppBackend._resolve_launch_mmproj_path.__get__(
        backend
    )
    gguf = gguf.rename(tmp_path / "Qwen-model.gguf")
    custom = _write_gguf(tmp_path / "gemma-custom projector.gguf")
    seen = []
    backend._mmproj_vram_bytes = lambda path: seen.append(path) or GIB
    cmd = _launch(
        backend,
        gguf,
        is_vision = False,
        disable_vision = vision_off,
        extra_args = [flag, str(custom)],
    )["cmd"]
    if vision_off:
        assert "--mmproj" not in cmd and "-mm" not in cmd
    else:
        assert cmd.count("--mmproj") == 1
        assert cmd[cmd.index("--mmproj") + 1] == str(custom)
        assert str(custom) in seen


def test_vision_off_still_charges_a_custom_audio_only_projector(tmp_path, monkeypatch):
    model = _write_gguf(tmp_path / "model.gguf")
    custom = _write_gguf(tmp_path / "custom-audio.gguf")
    monkeypatch.setattr(_meta, "mmproj_accepts_image", lambda _path: False)
    config = SimpleNamespace(
        gguf_file = str(model),
        gguf_mmproj_file = None,
        gguf_mtp_file = None,
        gguf_dspark_file = None,
        gguf_dflash_file = None,
        gguf_hf_repo = None,
        gguf_variant = None,
        is_vision = False,
    )
    enabled = _estimate_gguf_required_gb(config, llama_extra_args = ["--mmproj", str(custom)])
    disabled = _estimate_gguf_required_gb(
        config,
        llama_extra_args = ["--mmproj", str(custom)],
        disable_vision = True,
    )
    assert enabled == disabled


@pytest.mark.parametrize("vision_off", [False, True])
def test_custom_projector_does_not_hide_remote_model_bytes(tmp_path, monkeypatch, vision_off):
    import routes.inference as routes
    import utils.models.model_config as model_config

    custom = _write_gguf(tmp_path / "custom-vision.gguf")
    config = SimpleNamespace(
        gguf_file = None,
        gguf_mmproj_file = None,
        gguf_mtp_file = None,
        gguf_dspark_file = None,
        gguf_dflash_file = None,
        gguf_hf_repo = "org/vision-GGUF",
        gguf_variant = "Q4_K_M",
        is_vision = True,
    )
    monkeypatch.setattr(
        model_config,
        "list_gguf_variants",
        lambda *_a, **_kw: (
            [SimpleNamespace(quant = "Q4_K_M", size_bytes = 4 * GIB)],
            True,
        ),
    )
    seen = []
    monkeypatch.setattr(
        routes, "_remote_gguf_companion_bytes", lambda *_a, **kw: seen.append(kw) or 0
    )
    monkeypatch.setattr(routes, "_remote_gguf_compute_reserve_gb", lambda **_kw: 0)
    result = _estimate_gguf_required_gb(
        config,
        llama_extra_args = ["--mmproj", str(custom)],
        disable_vision = vision_off,
        speculative_type = "off",
    )
    expected = 4 * GIB + (0 if vision_off else custom.stat().st_size)
    assert result * GIB == expected
    assert seen[0]["include_mmproj"] is False


def test_missing_custom_projector_is_rejected_before_unloading(tmp_path, monkeypatch):
    backend = LlamaCppBackend()
    unloaded = []
    monkeypatch.setattr(backend, "unload_model", lambda: unloaded.append(True))
    with pytest.raises(ValueError, match = "custom mmproj path"):
        backend.load_model(
            GgufLoadIntent(
                model_identifier = "test",
                gguf_path = str(tmp_path / "model.gguf"),
                extra_args = ("--mmproj", str(tmp_path / "missing.gguf")),
            )
        )
    assert unloaded == []


@pytest.mark.parametrize("flag", ["--mmproj", "-mm"])
def test_hub_load_rechecks_custom_projector_replaced_in_place(tmp_path, flag):
    backend, gguf = _backend(tmp_path, memory = [(0, 12_000, 24_000)])
    backend._resolve_launch_mmproj_path = LlamaCppBackend._resolve_launch_mmproj_path.__get__(
        backend
    )
    custom = _write_gguf(tmp_path / "custom-projector.gguf")
    _launch(backend, gguf, is_vision = False, extra_args = [flag, str(custom)])
    backend._hf_variant = "Q4_K_M"
    intent = GgufLoadIntent(
        model_identifier = "test",
        hf_repo = "org/model",
        hf_variant = "Q4_K_M",
        extra_args = (flag, str(custom)),
    )
    assert backend.matches_load_source(intent)
    assert not backend.matches_load_source(replace(intent, hf_variant = "Q8_0"))
    replacement = _write_gguf(tmp_path / "replacement.gguf")
    replacement.replace(custom)
    assert not backend.matches_load_source(intent)


def test_remote_ubatch_sizes_the_custom_projector_not_the_repo_one(tmp_path, monkeypatch):
    import routes.inference as routes

    custom = _write_gguf(tmp_path / "custom-audio.gguf")
    monkeypatch.setattr(_meta, "mmproj_accepts_image", lambda _path: False)
    config = SimpleNamespace(is_vision = True)
    assert routes._remote_required_ubatch(config, []) > 0
    assert routes._remote_required_ubatch(config, ["--mmproj", str(custom)]) == 0


@pytest.mark.parametrize("flag", ["--mmproj", "-mm"])
@pytest.mark.parametrize("managed", [False, True])
def test_only_the_owner_may_name_a_custom_projector(monkeypatch, flag, managed):
    import routes.inference as routes
    from fastapi import HTTPException

    monkeypatch.setattr(routes.account_access, "managed_account", lambda: managed)
    routes._refuse_managed_custom_projector(["--ctx-size", "4096"])
    if not managed:
        routes._refuse_managed_custom_projector([flag, "/models/p.gguf"])
        return
    with pytest.raises(HTTPException) as err:
        routes._refuse_managed_custom_projector([flag, "/models/p.gguf"])
    assert err.value.status_code == 403


@pytest.mark.parametrize(
    "args",
    [
        ["--chat-template-file", "/home/owner/.ssh/id_ed25519"],
        ["--grammar-file", "/etc/shadow"],
        ["-jf", "/home/owner/secret.json"],
        ["--lora", "/home/owner/adapter.gguf"],
        ["--lora-scaled", "/home/owner/adapter.gguf:0.5"],
        ["--control-vector", "/home/owner/cv.gguf"],
        ["-md", "/home/owner/draft.gguf"],
        ["--spec-draft-model", "/home/owner/draft.gguf"],
        ["-lcs", "/home/owner/cache.bin"],
        ["--log-prompts-dir", "/home/owner/.config"],
        ["--video-ffmpeg-dir", "/srv/studio/accounts/m/sandbox"],
    ],
)
@pytest.mark.parametrize("managed", [False, True])
def test_only_the_owner_may_name_a_file_path_option(monkeypatch, args, managed):
    import routes.inference as routes
    from fastapi import HTTPException

    monkeypatch.setattr(routes.account_access, "managed_account", lambda: managed)
    routes._refuse_managed_custom_projector(["--ctx-size", "4096", "--temp", "0.7"])
    routes._refuse_managed_custom_projector(None)
    if not managed:
        routes._refuse_managed_custom_projector(["--ctx-size", "4096", *args])
        return
    with pytest.raises(HTTPException) as err:
        routes._refuse_managed_custom_projector(["--ctx-size", "4096", *args])
    assert err.value.status_code == 403 and args[0] in err.value.detail
    assert args[1] not in err.value.detail


_OWNER_PATHS = ["--chat-template-file", "/owner/t.jinja", "-md", "/owner/draft.gguf"]


def _managed_with_owner(
    monkeypatch,
    override = None,
    intent = None,
):
    import routes.inference as routes
    from types import SimpleNamespace
    from utils import openai_auto_switch_settings as settings

    monkeypatch.setattr(routes.account_access, "managed_account", lambda: True)
    monkeypatch.setattr(
        settings, "get_model_override", lambda key: dict(override.get(key, {})) if override else {}
    )
    monkeypatch.setattr(
        routes, "get_llama_cpp_backend", lambda: SimpleNamespace(last_load_intent = intent)
    )
    return routes


def test_managed_caller_may_replay_the_owners_saved_paths(monkeypatch):
    from fastapi import HTTPException

    routes = _managed_with_owner(monkeypatch, {"org/alias": {"llama_extra_args": _OWNER_PATHS}})
    routes._refuse_managed_custom_projector(
        ["--ctx-size", "4096", *_OWNER_PATHS], "m.gguf", "org/alias"
    )
    routes._refuse_managed_custom_projector(
        ["--chat-template-file=/owner/t.jinja"], "m.gguf", "org/alias"
    )
    with pytest.raises(HTTPException) as err:
        routes._refuse_managed_custom_projector(
            ["--chat-template-file", "/home/owner/.ssh/id"], "m.gguf", "org/alias"
        )
    assert err.value.status_code == 403
    with pytest.raises(HTTPException):
        routes._refuse_managed_custom_projector(_OWNER_PATHS, "other.gguf")


def test_managed_caller_may_resend_the_resident_same_model_paths(monkeypatch):
    from fastapi import HTTPException
    from types import SimpleNamespace

    import utils.hf_cache_settings as cache_settings

    monkeypatch.setattr(cache_settings, "known_hf_hub_caches", lambda: [Path("/hf/hub")])

    intent = SimpleNamespace(
        model_identifier = "m.gguf",
        hf_variant = None,
        extra_args = ("--lora", "/owner/a.gguf"),
    )
    routes = _managed_with_owner(monkeypatch, intent = intent)
    routes._refuse_managed_custom_projector(["--lora", "/owner/a.gguf"], "m.gguf")
    snapshot = SimpleNamespace(
        model_identifier = "/hf/hub/models--unsloth--B-GGUF/snapshots/abc/B-Q4_K_M.gguf",
        hf_variant = None,
        extra_args = ("--lora", "/owner/a.gguf"),
    )
    monkeypatch.setattr(
        routes, "get_llama_cpp_backend", lambda: SimpleNamespace(last_load_intent = snapshot)
    )
    routes._refuse_managed_custom_projector(["--lora", "/owner/a.gguf"], "unsloth/B-GGUF")
    monkeypatch.setattr(
        routes, "get_llama_cpp_backend", lambda: SimpleNamespace(last_load_intent = intent)
    )
    with pytest.raises(HTTPException):
        routes._refuse_managed_custom_projector(["--lora", "/owner/a.gguf"], "other.gguf")
    with pytest.raises(HTTPException):
        routes._refuse_managed_custom_projector(["--lora", "/owner/a.gguf"], "m.gguf", None, "Q8_0")


def test_resident_paths_need_a_real_cache_snapshot_and_follow_an_omitted_variant(
    monkeypatch, tmp_path
):
    from fastapi import HTTPException
    from types import SimpleNamespace

    import utils.hf_cache_settings as cache_settings

    hub = tmp_path / "hub"
    monkeypatch.setattr(cache_settings, "known_hf_hub_caches", lambda: [hub])
    intent = SimpleNamespace(
        model_identifier = str(hub / "models--unsloth--B-GGUF/snapshots/abc/B-Q4_K_M.gguf"),
        hf_variant = "Q4_K_M",
        extra_args = ("--lora", "/owner/a.gguf"),
    )
    routes = _managed_with_owner(monkeypatch, intent = intent)
    routes._refuse_managed_custom_projector(["--lora", "/owner/a.gguf"], "unsloth/B-GGUF")
    routes._refuse_managed_custom_projector(
        ["--lora", "/owner/a.gguf"], "unsloth/B-GGUF", None, "Q4_K_M"
    )
    routes._refuse_managed_custom_projector(["--lora", "/owner/a.gguf"], intent.model_identifier)
    fake = tmp_path / "ws/models--unsloth--B-GGUF/snapshots/x/B.gguf"
    with pytest.raises(HTTPException):
        routes._refuse_managed_custom_projector(["--lora", "/owner/a.gguf"], str(fake))
    outside = SimpleNamespace(
        **{
            **vars(intent),
            "model_identifier": str(tmp_path / "own/models--unsloth--B-GGUF/snapshots/x/B.gguf"),
        }
    )
    monkeypatch.setattr(
        routes, "get_llama_cpp_backend", lambda: SimpleNamespace(last_load_intent = outside)
    )
    with pytest.raises(HTTPException):
        routes._refuse_managed_custom_projector(["--lora", "/owner/a.gguf"], "unsloth/B-GGUF")
    with pytest.raises(HTTPException):
        routes._refuse_managed_custom_projector(
            ["--lora", "/owner/a.gguf"], "unsloth/B-GGUF", None, "Q8_0"
        )


def test_inherited_owner_paths_get_the_same_managed_check(monkeypatch, tmp_path):
    from fastapi import HTTPException
    from types import SimpleNamespace

    import utils.hf_cache_settings as cache_settings
    from models.inference import LoadRequest

    hub = tmp_path / "hub"
    monkeypatch.setattr(cache_settings, "known_hf_hub_caches", lambda: [hub])
    resident = str(hub / "models--unsloth--B-GGUF/snapshots/abc/B-Q4_K_M.gguf")
    backend = SimpleNamespace(
        extra_args = ["--lora", "/owner/a.gguf"],
        extra_args_source = (resident, "Q4_K_M"),
        last_load_intent = SimpleNamespace(
            model_identifier = resident,
            hf_variant = "Q4_K_M",
            extra_args = ("--lora", "/owner/a.gguf"),
        ),
    )
    routes = _managed_with_owner(monkeypatch)
    monkeypatch.setattr(routes, "get_llama_cpp_backend", lambda: backend)
    config = SimpleNamespace(is_gguf = True, gguf_variant = "Q4_K_M")

    same = LoadRequest(model_path = "unsloth/B-GGUF")
    assert routes._resolve_inherited_extra_args(same, config, "unsloth/B-GGUF", None) == [
        "--lora",
        "/owner/a.gguf",
    ]
    fake = str(tmp_path / "ws/models--unsloth--B-GGUF/snapshots/x/B-Q4_K_M.gguf")
    with pytest.raises(HTTPException):
        routes._resolve_inherited_extra_args(LoadRequest(model_path = fake), config, fake, None)
