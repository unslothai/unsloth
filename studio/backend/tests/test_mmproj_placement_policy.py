# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The vision-projector placement policy: GPU, CPU, or not loaded at all.

Model + speculative decoding + projector + 4096 context in VRAM if they fit;
otherwise pin the projector to the CPU; if that is still not enough, drop
speculative decoding as well; and load no projector at all, on either device,
when vision is switched off. Also covers the two fields the client reseeds the
switch from, which no single-side test exercises.
"""

from __future__ import annotations

import inspect
import os
import struct
import subprocess
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
from models.inference import InferenceStatusResponse, LoadResponse
from routes.inference import (
    _estimate_gguf_required_gb,
    _guard_chat_load_against_training,
    _llama_runtime_fields,
    _LoadPlacement,
)

MIB = 1024 * 1024
GIB = 1024**3
_REAL_POPEN = subprocess.Popen

# Model + projector fit at 4096 but not at the native length: the placement loop
# shrinks the context long before it would spill a layer, so pricing residency at
# the native length is the bug this policy exists to avoid.
NATIVE_CTX = 262144
KV_PER_TOKEN = 64 * 1024  # 4096 ctx -> 256 MiB, NATIVE_CTX -> 16 GiB


def _write_gguf(path: Path) -> Path:
    def string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    metadata = string("general.architecture") + struct.pack("<I", 8) + string("llama")
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)
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
    """A GGUF vision load with the fit inputs pinned to known numbers.

    ``native_ctx`` is lowered for the drafter tests: the drop probe prices the
    reserve at the context the target alone would reach, so a 262144 native
    length makes the draft KV, not the placement, decide every one of them.
    """
    backend = LlamaCppBackend()
    gguf = _write_gguf(tmp_path / "model.gguf")
    mmproj = _write_gguf(tmp_path / "mmproj-F16.gguf")
    drafter = _write_gguf(tmp_path / "mtp.gguf")

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
    # Only the speculative-decoding test asks for a drafter; everywhere else the
    # resolution has to come back empty or MTP engages behind the scenes.
    backend._resolve_launch_mtp_path = lambda **_kw: str(drafter) if drafter_bytes else None
    backend._apu_ram_shortfall_message = lambda *_a, **_kw: None
    backend._amd_apu_wants_unified_memory = lambda *_a, **_kw: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: False
    backend._wait_for_health = lambda timeout: True
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
        # Auto context, the mode the policy is about: 0 resolves to the model's
        # native length and lets the placement loop shrink it. A request pinned at
        # 4096 would hide the whole floor question.
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
    """Model + projector fit at 4096 but not at the native 262144 context.

    The placement loop shrinks the context, it does not spill layers, so both
    arms are fully GPU-resident and pinning would buy nothing while costing
    ~8.8x on every image encode.
    """
    backend, gguf = _backend(tmp_path, memory = [(0, 12_000, 24_000)])

    cmd = _launch(backend, gguf)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" not in cmd
    # The premise the pin would have been traded against: every layer is already
    # resident, so what the native length cost was context, not residency.
    assert cmd[cmd.index("--fit") + 1] == "off"
    assert int(cmd[cmd.index("-c") + 1]) < NATIVE_CTX


def test_projector_pinned_to_cpu_when_it_does_not_fit(tmp_path):
    """The 6 GiB model fits at 4096 on this card, the projector on top does not.

    Vision is preserved rather than dropped: --mmproj still goes out, only the
    offload is disabled. Sized so the _MMPROJ_VRAM_SAFETY surcharge decides it:
    the budget is 8692 - 3% of 16384 = 8200 MiB, the footprint at 4096 is 6144
    model + 1024 projector + 256 compute + 320 CUDA context + 256 KV = 8000 MiB,
    and only the 409 MiB a projector costs beyond its file size puts it over.
    """
    backend, gguf = _backend(tmp_path, memory = [(0, 8_692, 16_384)])

    cmd = _launch(backend, gguf)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" in cmd
    # And the trade was paid for: the model alone is fully resident, which is the
    # only thing the slower image encode is bought with.
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


# The drafter tests share one shape: a small native context so the drop probe
# prices the reserve near the floor, and a 2 GiB drafter. Only the budget moves.
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
    """The projector is the first thing given up, not the last.

    A bounded per-image cost is cheaper to concede than a per-token speedup, so pin
    the projector and drop the drafter only if that was not enough. Here it is enough:
    budget 12470 - 8% of 24000 = 10550 MiB, model + drafter + projector needs ~10682,
    and the pin hands back the projector's 1434 so the drafter survives.

    Deliberately tight: every term of the drafter's charge is decisive at this budget,
    its 224 MiB decode graph included, so dropping one shows up here.
    """
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
    """The pin runs with speculative decoding still live, so the drafter has to
    be charged or the predicate answers for a machine that does not exist.

    Differential, on one budget: the only thing that changes between the two
    loads is whether a drafter is present. Model + projector alone is about 8410
    MiB against a 10080 MiB budget and fits comfortably, so a predicate that
    leaves the drafter out cannot pin either load, and the asymmetry below is
    exactly the drafter's roughly 2272 MiB of weights and draft graph.
    """
    memory = [(0, 12_000, 24_000)]

    with_drafter, gguf = _drafter_backend(tmp_path, memory)
    pinned = _launch_with_drafter(with_drafter, gguf, tmp_path)

    without_drafter, gguf2 = _backend(tmp_path, memory = memory, native_ctx = DRAFTER_NATIVE_CTX)
    unpinned = _launch(without_drafter, gguf2)["cmd"]

    assert "--no-mmproj-offload" in pinned
    assert "--model-draft" in pinned
    # Same card, same model, same projector: only the drafter differs.
    assert "--no-mmproj-offload" not in unpinned


@pytest.mark.parametrize(
    "is_vision,disable_vision,expect_disabled,expect_by_user",
    [
        # A vision GGUF with the switch on: the projector exists and the user
        # turned it off, so both are true.
        (True, True, True, True),
        # A GGUF that never had a projector, switch on. The request still has to
        # round-trip or the toggle reseeds itself to off after every load, but
        # nothing was taken away from the user, so the narrow field stays false.
        (False, True, True, False),
        (True, False, False, False),
        (False, False, False, False),
    ],
)
def test_load_and_status_both_report_the_vision_toggle(
    tmp_path, is_vision, disable_vision, expect_disabled, expect_by_user
):
    """Both fields must reach the client on both responses.

    The frontend coalesces each with ``?? false``, so a field the backend omits
    reads as "vision is on" instead of raising: the switch silently reseeds to
    off after every load. Presence is therefore asserted on its own, not just
    the value.
    """
    backend, gguf = _backend(tmp_path, memory = [(0, 12_000, 24_000)])
    _launch(backend, gguf, is_vision = is_vision, disable_vision = disable_vision)

    # The shared resolver both responses are built from, drift guard included.
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
    # Exactly the projector, and nothing else moved.
    assert round((charged - freed) * 1024) == 1
    assert freed < charged


def test_the_training_guard_forwards_the_switch_to_its_estimator(tmp_path):
    """The gate above is only worth having if the request reaches it.

    Asserts on the keyword the guard hands the estimator, not on the verdict, so
    the test stays about the wiring and not about the rest of the guard.
    """
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
            # The verdict is not what this test is about; the forwarded keyword is,
            # and it is already captured by the time anything downstream can fail.
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
    """A projector is not always a vision tower. ultravox, Voxtral and Qwen3-ASR
    declare an audio encoder and no vision, so suppressing one would remove the
    model's audio input and free no image VRAM: the switch has nothing to turn
    off there and must leave it alone. A projector serving both modalities is
    still suppressed, because llama.cpp cannot load one modality without the
    other, and the switch is the user asking for the VRAM back."""
    import utils.models.gguf_metadata as _meta

    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])
    with patch.object(_meta, "mmproj_capabilities", lambda _p: (has_audio, accepts_image)):
        cmd = _launch(backend, gguf, disable_vision = True)["cmd"]

    assert ("--mmproj" in cmd) is projector_expected, label


# The pin reads no platform flag. It sees the GPU probe's output, the Metal
# budget, the paravirtual capability answer, tensor_parallel and the memory
# mode, and nothing else, so an OS reaches it only through what its probe
# reports. These are the probe signatures the supported pairs produce: what
# varies between a Windows and a WSL RTX 4090 is nothing the decision reads.
# Free VRAM is 7600 MiB against a footprint that needs more, i.e. the case that
# pins on a discrete card, so any cell reporting "no pin" is doing so because
# its memory is shared, not because it had room.
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
    # Vision survives either way: the pin moves the projector, it never drops it.
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
    """--mmproj-auto asks llama-server to find the adjacent projector on its own, so
    suppressing Studio's --mmproj and the env vars is not enough: vision would come
    back on a load that reports it off and never charged the projector's VRAM.
    llama.cpp is last-wins on the pair, so the disable form has to follow the extras."""
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])

    cmd = _launch(backend, gguf, disable_vision = True, extra_args = ["--mmproj-auto"])["cmd"]

    assert "--no-mmproj-auto" in cmd
    assert cmd.index("--no-mmproj-auto") > cmd.index("--mmproj-auto")


def test_an_audio_only_projector_is_not_taken_away_by_the_auto_override(tmp_path):
    """The override exists to stop a projector coming back. An audio-only one is kept
    on purpose, so --no-mmproj-auto must not follow it out the door."""
    import utils.models.gguf_metadata as _meta

    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])
    with patch.object(_meta, "mmproj_capabilities", lambda _p: (True, False)):
        cmd = _launch(backend, gguf, disable_vision = True, extra_args = ["--mmproj-auto"])["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-auto" not in cmd


def test_an_audio_only_projector_does_not_blame_the_switch_for_images(tmp_path):
    """vision_disabled_by_user drives the composer's "you turned it off" message, so
    on a model with no image encoder it would promise a capability that turning the
    switch back on cannot deliver."""
    import utils.models.gguf_metadata as _meta

    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])
    with patch.object(_meta, "mmproj_capabilities", lambda _p: (True, False)):
        _launch(backend, gguf, disable_vision = True)

    assert backend._disable_vision is True
    assert backend._vision_disabled_by_user is False


def test_the_training_guard_still_charges_an_audio_only_projector(tmp_path):
    """The switch turns vision off, and the loader keeps an audio-only projector
    anyway because there is no image tower to drop. Dropping its bytes here would
    let the guard admit a chat load the running training job cannot afford, which
    is the direction that costs someone else's job rather than merely annoying
    this user."""
    import utils.models.gguf_metadata as _meta

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
    # Kept for audio, so it is charged exactly as an enabled projector would be.
    assert audio_only == charged
    # An image projector really is dropped, so the switch still frees its bytes.
    assert vision < charged


def test_the_download_interlock_is_not_relaxed_by_the_vision_switch(tmp_path):
    """The load fetches a remote projector whenever the repo ships one and the extras
    have not opted out, switch or no switch, because only the file's metadata says
    whether it is an image tower or an audio encoder. So the interlock has to hold
    for it: relaxing it let a vision-off load skip the 409 and then write into the
    shared Hub cache beside a running download job, which is the race the check
    exists to stop. The predicate must mirror the download gate, not the switch."""
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
        # Vision off still downloads, so the interlock still applies.
        assert _run(disable_vision = True) is True
        assert _run(disable_vision = False) is True
        # The extras opting out is the one case that downloads nothing.
        assert _run(disable_vision = True, extra_args = ["--no-mmproj"]) is False


def test_a_user_pinned_projector_is_not_charged_against_vram(tmp_path):
    """--no-mmproj-offload puts the projector in host RAM, so its bytes are not on
    the card. Charging them anyway shrank the context and spilled layers to make
    room for VRAM nothing occupies, which is worse placement than Studio's own."""
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])

    cmd = _launch(backend, gguf, extra_args = ["--no-mmproj-offload"])["cmd"]

    assert "--mmproj" in cmd
    # Fully placed, exactly as an unpinned load on this card is.
    assert "--fit" in cmd and cmd[cmd.index("--fit") + 1] == "off"
    assert cmd[cmd.index("-c") + 1] == "9984"


def test_a_user_demanding_gpu_offload_still_pays_for_it(tmp_path):
    """The mirror: --mmproj-offload asks for the projector ON the card, so its bytes
    stay in the budget and the context shrinks to fit them. Resolving the value is
    what separates this from the case above; merely detecting ownership cannot."""
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])

    cmd = _launch(backend, gguf, extra_args = ["--mmproj-offload"])["cmd"]

    # Charging the projector leaves nothing that fits, so this lands on the Auto
    # offload fallback. The value is that constant, not a literal: the point of the
    # assertion is that the context shrank to pay for the projector, and pinning the
    # number here only records which release the test was written in.
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
    """The probe answers "is the projector resident", and the layer placement that
    follows must not then contradict it by spilling model layers. The context-compute
    buffer is replicated per device, so a probe pricing it once could call a
    multi-GPU split resident and hand the fit a load it cannot place. Whatever the
    probe decides, the launch has to be self-consistent: either the projector went
    to the CPU, or the model is fully placed."""
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
    """Remote, so nothing here has the file to ask whether it is an image tower the
    switch drops or an audio encoder the loader keeps. Under-charging is the
    direction that admits a chat load over VRAM a running training job needs, so
    the unknown one is charged."""
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


def _ambient_mmproj(tmp_path, monkeypatch):
    ambient = tmp_path / "ambient-mmproj.gguf"
    ambient.write_bytes(b"\x00" * (1 * MIB))
    monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(ambient))
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])
    backend._resolve_launch_mmproj_path = lambda **_kw: None
    return backend, gguf


def test_the_vision_switch_does_not_record_an_inherited_audio_encoder(tmp_path, monkeypatch):
    """The capability probe falls back to the ambient LLAMA_ARG_MMPROJ, which the
    switch then scrubs out of the child. Reading it anyway recorded an audio encoder
    the launched server does not have, and that becomes has_audio_input, so the
    composer would offer attachments the server cannot process.

    A projector serving both modalities is the case that still scrubs: llama.cpp
    cannot load half of it, so switching vision off takes the audio with it."""
    backend, gguf = _ambient_mmproj(tmp_path, monkeypatch)

    import utils.models.gguf_metadata as _meta

    with patch.object(_meta, "mmproj_capabilities", lambda _p: (True, True)):
        result = _launch(backend, gguf, disable_vision = True)

    assert "LLAMA_ARG_MMPROJ" not in result["env"]
    assert backend._mmproj_has_audio is False


def test_the_vision_switch_keeps_an_inherited_audio_only_encoder(tmp_path, monkeypatch):
    """The other half of the same rule, which the resolved path already had: an
    ultravox / Voxtral / Qwen3-ASR projector declares audio and no vision, so there is
    no image tower for the switch to turn off. Scrubbing it took the model's audio
    input away and handed back no image VRAM, and the probe agreed with the scrub, so
    the loss was silent on both sides.
    """
    backend, gguf = _ambient_mmproj(tmp_path, monkeypatch)

    import utils.models.gguf_metadata as _meta

    with patch.object(_meta, "mmproj_capabilities", lambda _p: (True, False)):
        result = _launch(backend, gguf, disable_vision = True)

    assert result["env"].get("LLAMA_ARG_MMPROJ")
    # And the probe follows the scrub, so the composer is told what the child has.
    assert backend._mmproj_has_audio is True
    assert backend._mmproj_accepts_image is False
    # --no-mmproj-auto does not unload it (server-context.cpp gates the load on a
    # non-empty mmproj.path and never reads no_mmproj), but it does make the router
    # advertise the model text-only, so a projector kept on purpose must not get it.
    assert "--no-mmproj-auto" not in result["cmd"]


def test_an_inherited_projector_that_reads_images_still_goes(tmp_path, monkeypatch):
    """The asymmetry is deliberate: only a readable audio-only declaration is kept.
    An image-capable one is exactly what the switch is for."""
    backend, gguf = _ambient_mmproj(tmp_path, monkeypatch)

    import utils.models.gguf_metadata as _meta

    with patch.object(_meta, "mmproj_capabilities", lambda _p: (False, True)):
        result = _launch(backend, gguf, disable_vision = True)

    assert "LLAMA_ARG_MMPROJ" not in result["env"]


def test_a_diffusion_runtime_is_not_torn_down_over_the_vision_switch(tmp_path):
    """_start_diffusion_server ignores the switch and records it False, so comparing
    it on that path makes every identical repeat request a mismatch and reloads a
    runtime that was already the one asked for. The switch must not be what decides,
    so both spellings of the request have to reach the same verdict."""
    from core.inference.llama_cpp import (
        GgufLoadIntent,
        LlamaCppBackend,
        _resolved_mmproj_offload,
    )

    backend = LlamaCppBackend()
    backend._is_diffusion = True
    backend._disable_vision = False
    backend._gguf_path = str(tmp_path / "diffusion.gguf")
    # Enough state for the comparison under test to be REACHED: the checks above it
    # return early on their own, and a test where both calls fail for an unrelated
    # reason passes whatever this line does (it did, until the mutant survived).
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
    """vision_disabled_by_user drives the composer's "you turned it off" message. With
    --no-mmproj in the extras the projector is suppressed by the ARGUMENT, resolution
    is skipped so nothing reads the file and the capability default stays True, and
    the switch would take the blame for images that turning it back on cannot
    restore while the argument still applies."""
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])

    _launch(backend, gguf, disable_vision = True, extra_args = ["--no-mmproj"])

    assert backend._disable_vision is True
    assert backend._vision_disabled_by_user is False


def test_a_projector_the_resolve_rejected_is_not_blamed_on_the_switch(tmp_path):
    """A None launch path does not mean the switch dropped it. The resolve also
    returns None for a missing file or a family mismatch, and reporting the switch
    there tells the user to turn Vision back on when the same projector will just be
    rejected again. Only a usable image projector the switch itself dropped counts."""
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])
    # What a family-check failure looks like from here.
    backend._resolve_launch_mmproj_path = lambda **_kw: None

    _launch(backend, gguf, disable_vision = True)

    assert backend._disable_vision is True
    assert backend._vision_disabled_by_user is False


def test_an_explicit_context_is_priced_at_the_length_it_asked_for(tmp_path):
    """The same card the auto test above leaves alone, with the context pinned.

    Auto shrinks the CONTEXT and never spills a layer, which is why it is asked at the
    4096 floor. An explicit context is honored verbatim, so the only give left is
    ``--fit on``, which offloads MODEL LAYERS: priced at the floor this load answers
    "the projector fits" and pays in the one currency the policy refuses to spend.

    Budget 12000 - 3% of 24000 = 11280 MiB. At 65536: 6144 model + 4096 KV + 256
    compute + 320 CUDA context = 10816, and the projector's real 1433 puts it at
    12249, over. So the projector goes to host RAM and every layer stays resident.
    """
    backend, gguf = _backend(tmp_path, memory = [(0, 12_000, 24_000)])

    cmd = _launch(backend, gguf, n_ctx = 65536)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" in cmd
    # What the pin bought, and the whole reason it was worth making: the requested
    # context survives intact with the model fully resident.
    assert cmd[cmd.index("--fit") + 1] == "off"
    assert cmd[cmd.index("-c") + 1] == "65536"


def test_an_explicit_context_that_fits_with_the_projector_keeps_it_on_the_gpu(tmp_path):
    """The floor is not simply replaced by "always pin under an explicit context".

    Same card, a context small enough that model + projector + KV all fit. Nothing
    is bought by moving the encoder off the GPU here, so it stays and image encode
    keeps its ~8.8x.
    """
    backend, gguf = _backend(tmp_path, memory = [(0, 12_000, 24_000)])

    cmd = _launch(backend, gguf, n_ctx = 8192)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" not in cmd
    assert cmd[cmd.index("--fit") + 1] == "off"
    assert cmd[cmd.index("-c") + 1] == "8192"


def test_an_environment_owned_placement_is_not_reversed_by_the_pin(tmp_path, monkeypatch):
    """common/arg.cpp applies every set_env option BEFORE argv, so an appended
    --no-mmproj-offload does not lose to LLAMA_ARG_MMPROJ_OFFLOAD=1, it overwrites
    it, and llama.cpp says so only in a stderr warning nobody reads. A user who set
    the variable globally owns the placement exactly as one who passed the flag
    does, which is already how the CPU-recovery gate reads it."""
    monkeypatch.setenv("LLAMA_ARG_MMPROJ_OFFLOAD", "1")
    # The card that pins when nobody has claimed the placement.
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
    # The same context the flag spelling earns on this card.
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
        # Silence on both sides is the only None: nobody has placed it.
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
    presence. Anything else and Studio budgets for a placement the child does not
    run."""
    assert _resolved_mmproj_offload(extras, env) is expected


def test_an_unparseable_environment_value_is_still_the_callers_placement(tmp_path, monkeypatch):
    """No side to budget for, but the variable is set, so Studio must not append its
    own spelling on top: common_params_parse throws on the value and the load fails
    naming the caller's variable, not a Studio flag they never chose."""
    monkeypatch.setenv("LLAMA_ARG_MMPROJ_OFFLOAD", "yes")
    backend, gguf = _backend(tmp_path, memory = [(0, 8_692, 16_384)])

    cmd = _launch(backend, gguf)["cmd"]

    assert "--no-mmproj-offload" not in cmd


def test_an_explicit_context_too_large_for_either_still_gives_the_projector_up_first(tmp_path):
    """Same card, a context that does not fit even with the projector in host RAM.

    The fit has to spill layers whatever happens here, which is exactly why the
    projector should not be holding 1433 MiB of the card while it does: every byte
    it gives back is a byte of model that stays resident. The order the policy is
    built on does not change just because the pin alone was not enough, and this is
    the same shape as dropping the drafter after the pin.
    """
    backend, gguf = _backend(tmp_path, memory = [(0, 12_000, 24_000)])

    cmd = _launch(backend, gguf, n_ctx = 131072)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" in cmd
    assert cmd[cmd.index("--fit") + 1] == "on"


# The context-compute buffer, at the shape the real one has: linear in context, and
# _CTX_COMPUTE_SPLIT_MULT times larger PER DEVICE once the model is layer-split. The
# shared _backend stubs it to a flat 0, which is fine for the single-GPU cases above
# and is exactly why none of them can see a split-rate error.
_CC_PER_TOKEN = 1536  # 6 MiB at 4096, the rate the bundled estimator produces


def _split_rate_backend(tmp_path, *, memory, **kwargs):
    backend, gguf = _backend(tmp_path, memory = memory, **kwargs)
    backend._compute_buffer_ctx_bytes = (
        lambda n_ctx, n_ubatch = None, cache_type_kv = None, *, layer_split = False: (
            n_ctx * _CC_PER_TOKEN * (LlamaCppBackend._CTX_COMPUTE_SPLIT_MULT if layer_split else 1)
        )
    )
    return backend, gguf


def test_the_probe_prices_an_explicit_context_the_way_the_split_placement_does(tmp_path):
    """Two cards, an explicit 65536, and a footprint that only a layer split can hold.

    The explicit branch selects through `_select_gpus_split_aware`, charging the
    context-compute buffer in the per-device overhead as well as the total and handing
    the single-to-split step to the retry. A layer split does not just replicate that
    buffer per card, it replicates a bigger one, so pricing it once understates the
    real cost several-fold: 96 MiB charged against 768 MiB spent at this context on
    two devices. That gap is wider than the projector surcharge being decided, so a
    plain probe answers "the projector fits" on exactly the loads whose placement then
    falls back to `--fit on` and spills model layers around the projector it kept.
    """
    backend, gguf = _split_rate_backend(tmp_path, memory = [(0, 7_200, 8_200), (1, 7_200, 8_200)])

    cmd = _launch(backend, gguf, n_ctx = 65536)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" in cmd
    # The point of the pin: the requested context is placed on the two cards rather
    # than the model being offloaded around a resident projector.
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
    """The split rate and the floor are separate questions and only the floor is Auto's.

    Auto's loop charges `_cc_bytes(ctx, n_gpus)` per card, so it is already stricter
    than plain `_select_gpus` and a plainly priced probe is more optimistic than the
    loop it gates. What keeps the pin honest under Auto is the FLOOR: a subset that
    cannot hold the projector at 4096 is one Auto cannot rescue by shrinking, so
    `--fit on` was coming either way.

    Two cards where plain accounting says the projector fits at 4096 and the split
    rate says it does not.
    """
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
    """The buffer belongs in the per-device overhead as well as the total.

    That is how the explicit branch charges it, and it is a separate term from the
    retry's single-to-split step: the step re-prices the FOOTPRINT once the count is
    known, while this is what makes each individual card carry its own copy. Drop it
    and a subset whose cards cannot each hold one is accepted; the placement then makes
    every device hold one anyway and falls back to `--fit on`, spilling model layers
    around the projector the probe just kept. Verified reachable by sweeping the launch
    path: without this term these numbers go from pinned with `--fit off` to unpinned
    with `--fit on`.
    """
    backend, gguf = _split_rate_backend(tmp_path, memory = [(0, 6_000, 7_000), (1, 6_000, 7_000)])

    cmd = _launch(backend, gguf, n_ctx = 32768)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" in cmd
    assert cmd[cmd.index("--fit") + 1] == "off"
    assert cmd[cmd.index("-c") + 1] == "32768"


def test_the_predicted_pin_is_reported_through_both_responses(tmp_path):
    """A projector the fit moved to host RAM must say so, on the same channel the
    startup-recovery pin uses.

    Both routes end at the same placement and the same user-visible cost, so the
    notice cannot depend on which one got there. Without this the predicted pin is
    silent outside the server log: image encoding simply gets slower with nothing in
    the UI to explain it, which is the complaint the recovery path exists to answer.
    The card here is the one the pin fires on.
    """
    backend, gguf = _backend(tmp_path, memory = [(0, 8_692, 16_384)])
    cmd = _launch(backend, gguf)["cmd"]

    # Premise: this really is the predicted pin, not a post-crash recovery.
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
        # get_value_from_env checks the LLAMA_ARG_NO_ spelling first and forces falsey
        # on presence alone, so the projector is already in host RAM either way.
        ({"LLAMA_ARG_NO_MMPROJ_OFFLOAD": "1"}, False),
        ({"LLAMA_ARG_NO_MMPROJ_OFFLOAD": ""}, False),
        # It wins over the positive spelling, exactly as arg.cpp orders them.
        ({"LLAMA_ARG_MMPROJ_OFFLOAD": "1", "LLAMA_ARG_NO_MMPROJ_OFFLOAD": "0"}, False),
        # is_falsey accepts `disabled`; the spelling list alone does not.
        ({"LLAMA_ARG_MMPROJ_OFFLOAD": "disabled"}, False),
        # Still a GPU projector, so the recovery retry is real work.
        ({"LLAMA_ARG_MMPROJ_OFFLOAD": "enabled"}, True),
        ({}, True),
    ],
)
def test_the_recovery_retry_sees_every_environment_pin(env, expected_retry):
    """A projector the environment already pinned to CPU cannot be rescued by pinning
    it again: the retry respawns a command identical in effect, cannot clear the
    allocation failure, and costs the caller the real error, because the branch that
    surfaces that OOM only runs when this returns None."""
    cmd = ["llama-server", "-m", "/cache/model.gguf", "--mmproj", "/cache/mmproj.gguf"]
    retry = LlamaCppBackend._with_mmproj_offload_disabled(cmd, env)

    assert (retry is not None) is expected_retry
    if expected_retry:
        assert retry[-1] == "--no-mmproj-offload"


def test_the_speculative_reserve_is_normalized_before_anything_prices_it(tmp_path):
    """Ordering invariant: `_mtp_bytes` reads `mtp_overhead_fn` at call time, so the
    CPU-drafter normalization has to run before the first thing that prices it.

    It used to run after the projector probe, which is wrong on its face: a CPU-pinned
    drafter allocates no VRAM, so charging the probe its full GPU footprint could pin
    the projector and cost ~8.8x per image encode for memory nothing holds.

    Asserted on the source rather than a launch, deliberately. No configuration I could
    build makes the two orders decide differently -- a separate CPU-pinned drafter is
    already excluded upstream, so `mtp_overhead_fn` is None at the probe either way.
    Reaching it needs a target keeping its own reserve while a sidecar displaces an
    embedded head. This locks the ordering that was fixed and claims no more.
    """
    source = Path(inspect.getsourcefile(LlamaCppBackend)).read_text()
    normalize_at = source.index("if _draft_cpu_no_embedded and mtp_overhead_fn is not None:")
    probe_at = source.index("_mm_mtp_on_gpu = _mtp_will_engage and not _draft_cpu_no_embedded")
    assert (
        normalize_at < probe_at
    ), "the CPU-drafter reserve must be normalized before the projector probe prices it"


def test_a_shared_device_beside_a_discrete_one_does_not_veto_the_pin(tmp_path):
    """A laptop pairing a discrete card with an APU enumerates both.

    Requiring EVERY enumerated device to have its own budget refused the pin there, so
    a model that would have fitted the discrete card once the projector moved kept it
    resident and either pulled the shared device into the split or went to `--fit on`.
    The shared device is dropped from the question instead: bytes handed back land on a
    card with its own pool.

    The discrete card is the one the single-GPU pin test uses, with a shared device
    (total 0) alongside it.
    """
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
    """--no-mmproj-offload on an APU moves the encoder inside one pool.

    On a discrete card the flag really does take those bytes off the device, so
    dropping them from the fit is right. A shared pool has nowhere to move them
    to: the projector sits in the very memory the context is being fitted
    against, and spending it on KV as well over-commits. The reference load
    charges the same bytes as model weights with no projector at all, which is
    exactly what a CPU-resident projector costs here, so the two must fit to the
    same context.
    """
    memory = [(0, 9_000, 0)]

    backend, gguf = _backend(tmp_path, memory = memory)
    cmd = _launch(backend, gguf, extra_args = ["--no-mmproj-offload"])["cmd"]
    pinned_ctx = int(cmd[cmd.index("-c") + 1])

    reference, ref_gguf = _backend(tmp_path, memory = memory, model_bytes = 7 * GIB, mmproj_bytes = 0)
    ref_cmd = _launch(reference, ref_gguf)["cmd"]
    reference_ctx = int(ref_cmd[ref_cmd.index("-c") + 1])

    assert pinned_ctx == reference_ctx


def test_a_pinned_projector_costs_the_shared_pool_beside_a_discrete_card(tmp_path):
    """The mixed host, where asking "is any device budgeted?" gets it backwards.

    _discrete_vram drops shared devices to decide whether the pin frees anything
    ANYWHERE, and one discrete card is enough for that. The fit is a different
    question: the selection walks prefixes of the enumerated list, so a shared
    device ranked first is a candidate subset on its own, and a context fitted
    against that pool without the pinned projector over-commits it. The discrete
    card here is nearly full, so the shared pool is what the load runs on.
    """
    memory = [(0, 9_000, 0), (1, 100, 16_384)]

    backend, gguf = _backend(tmp_path, memory = memory)
    cmd = _launch(backend, gguf, extra_args = ["--no-mmproj-offload"])["cmd"]
    pinned_ctx = int(cmd[cmd.index("-c") + 1])

    # Same reference as the single-device case: a CPU-resident projector in a
    # shared pool costs exactly what the same bytes cost as model weights.
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
    """The loader keeps an inherited audio-only LLAMA_ARG_MMPROJ when Vision is off.

    It is GPU-resident like any other projector, and this config never names it, so
    charging nothing let the coexistence guard admit a chat load the running
    training job cannot afford -- the direction that costs someone else's job.
    """
    model = tmp_path / "model.gguf"
    model.write_bytes(b"\x00" * (4 * MIB))
    ambient = tmp_path / "ambient-mmproj.gguf"
    ambient.write_bytes(b"\x00" * (1 * MIB))
    config = _estimator_config(model)

    bare = _estimate_gguf_required_gb(config, disable_vision = True)
    monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(ambient))

    import utils.models.gguf_metadata as _meta

    with patch.object(_meta, "mmproj_capabilities", lambda _p: (True, False)):
        charged = _estimate_gguf_required_gb(config, disable_vision = True)
    # An image-capable one is scrubbed out of the child, so it must stay uncharged.
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

    # What the launch really costs: weights plus the one projector argv names.
    expected = _estimate_gguf_required_gb(_estimator_config(model, resolved))
    monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(ambient))
    charged = _estimate_gguf_required_gb(_estimator_config(model, resolved))

    assert expected is not None and charged is not None
    # The 2 MiB ambient file is not in it, so the env changed nothing.
    assert charged == expected


def test_a_suppressed_image_projector_hands_the_budget_to_the_inherited_one(tmp_path, monkeypatch):
    """The combination that slipped through: the CONFIGURED projector is
    image-capable, so the switch drops it and Studio emits no --mmproj at all, while
    the inherited one is audio-only and is kept. argv only beats the environment when
    there IS argv, so the inherited projector is what loads, and it is what has to be
    charged."""
    model = tmp_path / "model.gguf"
    model.write_bytes(b"\x00" * (4 * MIB))
    configured = tmp_path / "mmproj-F16.gguf"
    configured.write_bytes(b"\x00" * (1 * MIB))
    ambient = tmp_path / "ambient-mmproj.gguf"
    ambient.write_bytes(b"\x00" * (2 * MIB))

    import utils.models.gguf_metadata as _meta

    def _caps(path):
        # Configured: images, so the switch drops it. Inherited: audio only, so it stays.
        return (False, True) if str(path) == str(configured) else (True, False)

    monkeypatch.delenv("LLAMA_ARG_MMPROJ", raising = False)
    with (
        patch.object(_meta, "mmproj_capabilities", _caps),
        patch.object(_meta, "mmproj_accepts_image", lambda p: _caps(p)[1]),
    ):
        # Weights alone: the switch drops the image projector and no env one exists.
        weights_only = _estimate_gguf_required_gb(
            _estimator_config(model, configured), disable_vision = True
        )
        monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(ambient))
        charged = _estimate_gguf_required_gb(
            _estimator_config(model, configured), disable_vision = True
        )

    assert charged is not None and weights_only is not None
    # The 2 MiB inherited projector, not the 1 MiB configured one that never loads.
    assert round((charged - weights_only) * 1024) == 2


def test_the_extras_opt_out_does_not_excuse_an_inherited_projector(tmp_path, monkeypatch):
    """--no-mmproj sets params.no_mmproj, which stops Studio resolving one of its own
    and stops the HF download, but server-context.cpp gates the load on a non-empty
    mmproj.path and never reads that field. The inherited projector loads straight
    through the opt-out, so the guard has to keep charging it."""
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
    """--no-mmproj makes llama_cpp.py skip the resolve, so Studio emits no --mmproj
    and the configured projector never loads. It does not unset an inherited path,
    which then loads unopposed. The estimate has to move with the launch: drop the
    configured file, charge the inherited one.
    """
    model = tmp_path / "model.gguf"
    model.write_bytes(b"\x00" * (4 * MIB))
    configured = tmp_path / "mmproj-F16.gguf"
    configured.write_bytes(b"\x00" * (1 * MIB))
    ambient = tmp_path / "ambient-mmproj.gguf"
    ambient.write_bytes(b"\x00" * (2 * MIB))
    config = _estimator_config(model, configured)

    monkeypatch.delenv("LLAMA_ARG_MMPROJ", raising = False)
    # The configured projector, charged, with no opt-out in play.
    normal = _estimate_gguf_required_gb(config)
    weights_only = _estimate_gguf_required_gb(_estimator_config(model))
    opted_out = _estimate_gguf_required_gb(config, llama_extra_args = ["--no-mmproj"])

    monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(ambient))
    inherited = _estimate_gguf_required_gb(config, llama_extra_args = ["--no-mmproj"])

    for value in (normal, weights_only, opted_out, inherited):
        assert value is not None
    assert round((normal - weights_only) * 1024) == 1
    # The opt-out drops the configured projector, because it never loads.
    assert opted_out == weights_only
    # And the inherited one takes its place at its own size.
    assert round((inherited - weights_only) * 1024) == 2


def _paravirtual(monkeypatch):
    import core.inference.llama_cpp as _llama_cpp
    monkeypatch.setattr(_llama_cpp, "_metal_device_is_paravirtual", lambda: True)


def test_a_virtualised_metal_device_does_not_keep_the_inherited_projector(tmp_path, monkeypatch):
    """The paravirtual scrub runs after the switch's and takes BOTH projector vars
    unconditionally, so a file the switch kept is gone by launch.

    Two things went wrong when the "kept" answer did not know that: the capability
    probe described an audio encoder the child does not have, and the --no-mmproj-auto
    override was skipped, leaving a remembered --mmproj-auto free to rediscover an
    adjacent image projector on a load the user asked to be text-only.
    """
    ambient = tmp_path / "ambient-mmproj.gguf"
    ambient.write_bytes(b"\x00" * (1 * MIB))
    monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(ambient))
    _paravirtual(monkeypatch)
    backend, gguf = _backend(tmp_path, memory = [])
    backend._resolve_launch_mmproj_path = lambda **_kw: None

    import utils.models.gguf_metadata as _meta

    with patch.object(_meta, "mmproj_capabilities", lambda _p: (True, False)):
        result = _launch(backend, gguf, disable_vision = True, extra_args = ["--mmproj-auto"])

    assert "LLAMA_ARG_MMPROJ" not in result["env"]
    # Nothing survives to rediscover a projector with.
    assert "--no-mmproj-auto" in result["cmd"]
    # And the probe describes the child that actually launched.
    assert backend._mmproj_has_audio is False


def test_dropping_an_inherited_image_projector_points_at_the_switch(tmp_path, monkeypatch):
    """Turning Vision back on restores an inherited image projector, so the composer
    must name the switch rather than send the user hunting for a valid mmproj."""
    ambient = tmp_path / "ambient-mmproj.gguf"
    ambient.write_bytes(b"\x00" * (1 * MIB))
    monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(ambient))
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])
    backend._resolve_launch_mmproj_path = lambda **_kw: None

    import utils.models.gguf_metadata as _meta

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
    """_apply_cpu_fallback_state is the last thing a Vulkan-crash recovery runs: its
    caller returns immediately after, before the load's own assignment of these two.

    Leaving them behind meant the response described the PREVIOUS load's Vision state,
    so the control flipped back on and an unchanged disable_vision request then failed
    runtime matching and reloaded the model.
    """
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
    # The rest of the recovery state still lands, so this is additive.
    assert backend._cpu_fallback_reason == "vulkan_startup_crash"


def test_both_cpu_recovery_call_sites_pass_the_vision_state(tmp_path):
    """Two call sites reach that helper and only one is on the common path, so a
    keyword added to one and not the other is a silent half-fix. Checked at the source
    because the second site needs a crash inside a replay this harness cannot stage."""
    source = inspect.getsource(LlamaCppBackend.load_model)
    calls = source.count("self._apply_cpu_fallback_state(")
    assert calls == 2, f"expected 2 recovery call sites, found {calls}"
    # The load's own `self._vision_disabled_by_user = ...` uses the same words, so
    # subtract it rather than matching loosely and passing on the wrong occurrence.
    keyword_uses = source.count("vision_disabled_by_user = bool(") - source.count(
        "self._vision_disabled_by_user = bool("
    )
    assert keyword_uses == calls
    assert source.count("disable_vision = disable_vision,") == calls


@pytest.mark.parametrize("cache_type_kv", [None, "q8_0"])
def test_a_tensor_load_downgraded_to_layer_split_still_gives_the_projector_up(
    tmp_path, cache_type_kv
):
    """The corner the probe's TP exclusion used to leave at main's behaviour.

    Tensor parallelism is requested, so the probe withholds its answer: layer-split
    numbers cannot price a per-device tensor buffer. The pooled weight-budget check
    then gives tensor mode up anyway -- and it prices weights PLUS projector, so it
    downgrades exactly the loads moving the encoder would have rescued. Once that
    downgrade is final the load is layer split, the probe's answer applies, and the
    projector goes to the CPU instead of the model spilling layers around it.

    Both cache types, because the downgrade leaves the requested one alone: the probe
    prices the cache that actually loads, so the verdict holds for either.

    Two cards too small to pool the 6 GiB model with its 1 GiB projector, but large
    enough to hold the model alone once the encoder moves.
    """
    backend, gguf = _backend(tmp_path, memory = [(0, 4_400, 8_192), (1, 4_400, 8_192)])

    cmd = _launch(backend, gguf, tensor_parallel = True, cache_type_kv = cache_type_kv)["cmd"]

    # Reachable ONLY through the deferred application: with tensor_parallel requested
    # the probe never applies its verdict at the probe site.
    assert "--no-mmproj-offload" in cmd
    assert "--mmproj" in cmd
    # And the trade was paid for: every layer stays resident.
    assert cmd[cmd.index("--fit") + 1] == "off"


def test_a_surviving_tensor_load_keeps_its_projector(tmp_path):
    """The other side of the deferral, and why the verdict is withheld rather than
    applied early: these numbers are layer-split numbers.

    Two cards that pool enough for tensor mode, so the weight-budget check keeps it.
    The layer-split probe would refuse the same footprint (it charges a per-device
    pipeline reserve and a replicated compute buffer that tensor mode allocates
    differently), so applying its answer here would move the encoder off a load that
    had room for it.
    """
    backend, gguf = _backend(tmp_path, memory = [(0, 4_800, 16_384), (1, 4_800, 16_384)])

    cmd = _launch(backend, gguf, tensor_parallel = True)["cmd"]

    assert cmd[cmd.index("--split-mode") + 1] == "tensor"
    assert "--no-mmproj-offload" not in cmd


def test_a_gpu_drafter_holds_the_deferred_pin_back(tmp_path):
    """The drafter-drop probe is gated off under tensor parallelism, so it has not
    run. The documented order is projector first and drafter second; pinning without
    being able to re-ask the second half pays the encoder cost and still reaches
    --fit on."""
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
