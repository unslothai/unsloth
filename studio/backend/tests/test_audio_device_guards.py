# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Guards that must not charge a CPU-placed audio load for VRAM it never takes.

A load held in system RAM still passed the training coexistence check, the GPU
arbiter and the memory preflight, so it could be refused on a full card, evict an
image or video pipeline, or be reported already-loaded while sitting on the GPU.
"""

import asyncio
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import HTTPException  # noqa: E402

import routes.inference as ri  # noqa: E402
from routes.training_vram import _stt_sidecar_holds_no_vram  # noqa: E402


def _audio(audio_type = "higgs_tts2", **kwargs):
    return types.SimpleNamespace(audio_type = audio_type, is_lora = False, identifier = "x/y", **kwargs)


def _request(audio_device = None):
    return types.SimpleNamespace(audio_device = audio_device)


# --- the gate itself -------------------------------------------------------


def test_only_a_native_audio_model_counts_as_a_cpu_audio_load():
    assert ri._native_audio_cpu_load(_audio(), _request("cpu"))
    assert not ri._native_audio_cpu_load(_audio(), _request("auto"))
    assert not ri._native_audio_cpu_load(_audio(), _request(None))


def test_a_chat_model_cannot_skip_the_guards_by_sending_audio_device():
    """audio_device is documented as ignored off the audio path. If it were not
    gated here, any load could set it and walk past the training guard."""
    assert not ri._native_audio_cpu_load(_audio(audio_type = None), _request("cpu"))
    assert not ri._native_audio_cpu_load(_audio(audio_type = "whisper"), _request("cpu"))


# --- the training coexistence guard ----------------------------------------


def test_a_cpu_audio_load_is_not_refused_while_training_runs(monkeypatch):
    """The guard 409s an unsized load, and refuses everything outright during
    diffusion training. Neither applies to weights that never reach the card."""
    monkeypatch.setattr(ri, "_diffusion_training_active", lambda: True)

    assert (
        ri._guard_chat_load_against_training(
            _audio(is_gguf = False),
            types.SimpleNamespace(
                audio_device = "cpu",
                gpu_memory_mode = "auto",
                gpu_layers = -1,
                tensor_parallel = False,
            ),
            load_in_4bit = False,
            placement = types.SimpleNamespace(
                requested_gpu_ids = None,
                gpu_ids_are_vulkan_ordinals = False,
                diffusion_kind = None,
            ),
        )
        is None
    )


def test_a_gpu_audio_load_is_still_refused_during_diffusion_training(monkeypatch):
    """The exemption must be the CPU placement, not the audio type."""
    monkeypatch.setattr(ri, "_diffusion_training_active", lambda: True)

    with pytest.raises(HTTPException) as excinfo:
        ri._guard_chat_load_against_training(
            _audio(is_gguf = False),
            types.SimpleNamespace(
                audio_device = "auto",
                gpu_memory_mode = "auto",
                gpu_layers = -1,
                tensor_parallel = False,
            ),
            load_in_4bit = False,
            placement = types.SimpleNamespace(
                requested_gpu_ids = None,
                gpu_ids_are_vulkan_ordinals = False,
                diffusion_kind = None,
            ),
        )
    assert excinfo.value.status_code == 409


# --- the memory preflight --------------------------------------------------


def test_a_cpu_load_skips_the_vram_preflight_entirely(monkeypatch):
    """Sizing it would refuse the load on a full GPU, which is the case the
    option exists for. The probe must not even be reached."""

    def _never():
        raise AssertionError("a CPU load must not size GPU memory")

    monkeypatch.setattr(ri, "_native_audio_post_handoff_free_gb", _never)
    placement = types.SimpleNamespace(requested_gpu_ids = None)

    result = asyncio.run(ri._preflight_native_audio_placement(_audio(), _request("cpu"), placement))
    assert result is placement


def test_minimax_on_cpu_is_refused_before_the_resident_model_is_evicted():
    """Its runtime needs CUDA. Failing later in the worker would cost the user
    the model they already had, since the switch evicts before the load runs."""
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            ri._preflight_native_audio_placement(
                _audio(audio_type = "minimax_music3"),
                _request("cpu"),
                types.SimpleNamespace(requested_gpu_ids = None),
            )
        )
    assert excinfo.value.status_code == 400
    assert "CPU RAM" in excinfo.value.detail


# --- the already-loaded shortcut -------------------------------------------


def _backend(audio_cpu, audio_type = "higgs_tts2"):
    entry = {"is_audio": True, "audio_type": audio_type}
    if audio_cpu is not None:
        entry["audio_cpu"] = audio_cpu
    return types.SimpleNamespace(active_model_name = "x/y", models = {"x/y": entry})


def test_a_resident_gpu_audio_model_does_not_satisfy_a_cpu_request():
    assert not ri._resident_audio_placement_matches(_backend(audio_cpu = False), _request("cpu"))


def test_a_resident_cpu_audio_model_satisfies_the_same_request_again():
    assert ri._resident_audio_placement_matches(_backend(audio_cpu = True), _request("cpu"))


def test_a_model_loaded_before_this_existed_is_read_as_gpu():
    """No recorded key means the load predates the option, which placed on GPU."""
    assert ri._resident_audio_placement_matches(_backend(audio_cpu = None), _request("auto"))
    assert not ri._resident_audio_placement_matches(_backend(audio_cpu = None), _request("cpu"))


def test_a_non_audio_model_keeps_the_shortcut():
    assert ri._resident_audio_placement_matches(
        _backend(audio_cpu = None, audio_type = None), _request("cpu")
    )


# --- training eviction -----------------------------------------------------


def test_a_cpu_placed_sidecar_is_left_alone_when_training_claims_vram():
    assert _stt_sidecar_holds_no_vram(types.SimpleNamespace(device = "cpu"))
    assert _stt_sidecar_holds_no_vram(types.SimpleNamespace(device = "whisper.cpp", _forced_cpu = True))
    assert _stt_sidecar_holds_no_vram(types.SimpleNamespace(device = "llama.cpp", _gpu_disabled = True))


def test_anything_that_might_hold_vram_is_still_evicted():
    """Default-deny: starving the run this makes room for is the worse failure."""
    assert not _stt_sidecar_holds_no_vram(types.SimpleNamespace(device = "cuda"))
    assert not _stt_sidecar_holds_no_vram(types.SimpleNamespace(device = "mps"))
    assert not _stt_sidecar_holds_no_vram(
        types.SimpleNamespace(device = "whisper.cpp", _forced_cpu = False)
    )
    assert not _stt_sidecar_holds_no_vram(types.SimpleNamespace())

    class _Raises:
        @property
        def device(self):
            raise RuntimeError("unreadable")

    assert not _stt_sidecar_holds_no_vram(_Raises())


def test_the_shortcut_reads_the_resident_model_not_the_requested_config():
    """It runs ahead of config resolution, so reading a config there raised
    UnboundLocalError and turned every repeat safetensors load into a 500."""
    import inspect
    assert list(inspect.signature(ri._resident_audio_placement_matches).parameters) == [
        "backend",
        "request",
    ]


# --- the GPU arbiter -------------------------------------------------------


def test_a_cpu_placed_audio_model_never_takes_the_arbiter():
    """It holds no GPU, so acquiring would cancel an image or video run for nothing."""
    assert ri._resident_audio_holds_no_gpu(_backend(audio_cpu = True))
    assert not ri._resident_audio_holds_no_gpu(_backend(audio_cpu = False))
    assert not ri._resident_audio_holds_no_gpu(_backend(audio_cpu = None))


def test_a_chat_model_cannot_reach_the_arbiter_skip():
    """Same audio-type gate as the writer, so a stray marker cannot skip it."""
    assert not ri._resident_audio_holds_no_gpu(_backend(audio_cpu = True, audio_type = None))


def test_nothing_resident_reads_as_holding_the_gpu():
    empty = types.SimpleNamespace(active_model_name = None, models = {})
    assert not ri._resident_audio_holds_no_gpu(empty)


def _inference_source() -> str:
    import pathlib
    return pathlib.Path(ri.__file__).read_text()


def test_the_already_loaded_branch_guards_its_acquire():
    src = _inference_source()
    assert "if not _resident_audio_holds_no_gpu(backend):\n" in src


def test_the_post_load_ownership_check_is_gated_like_the_gguf_one():
    """Ungated, it unloads the CPU audio model it just loaded and returns 409:
    the load skips acquire_for, so on a clean server the owner is None."""
    src = _inference_source()
    assert "if chat_load_needs_gpu and current_owner() != CHAT:" in src
    assert "\n        if current_owner() != CHAT:" not in src


# --- hiding the accelerators -----------------------------------------------


def test_a_cpu_audio_worker_hides_cuda_and_hip():
    from core.inference.audio_device import mask_accelerators_for_cpu_audio

    env = {"CUDA_VISIBLE_DEVICES": "0,1", "HIP_VISIBLE_DEVICES": "0"}
    mask_accelerators_for_cpu_audio(env)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    # HIP reads the CUDA variable only when its own is unset, so blanking one is
    # not enough; -1 is the sentinel the CPU embed server already uses.
    assert env["HIP_VISIBLE_DEVICES"] == "-1"


def test_an_inherited_rocr_mask_is_left_alone():
    """Clearing it exposes more agents to HSA enumeration, not fewer."""
    from core.inference.audio_device import mask_accelerators_for_cpu_audio

    env = {"ROCR_VISIBLE_DEVICES": "0"}
    mask_accelerators_for_cpu_audio(env)
    assert env["ROCR_VISIBLE_DEVICES"] == "0"


def test_the_mask_runs_before_hardware_detection():
    """detect_hardware() calls get_device_properties, which creates the context
    this load is supposed not to hold, so masking after it is too late."""
    import pathlib
    from core.inference import worker

    src = pathlib.Path(worker.__file__).read_text()
    assert src.index("mask_accelerators_for_cpu_audio(os.environ)") < src.index(
        "_hw.detect_hardware()"
    )


def test_a_zero_gpu_standard_load_drops_the_stale_chat_claim():
    """The load replaced whatever held CHAT. Leaving the claim makes the next
    Images/Video acquire run the CHAT evictor and unload a CPU audio model that
    was never on the GPU. The GGUF branch already releases; both do now."""
    src = _inference_source()
    assert src.count("await asyncio.to_thread(release, CHAT)") == 2


def test_the_release_is_gated_on_the_same_flag_as_the_409():
    src = _inference_source()
    assert src.count("if not chat_load_needs_gpu:") == 2


def test_every_http_device_field_pins_the_three_canonical_values():
    """A misspelled "cpu" that fell through to auto would put the model back on
    the GPU without saying so. 422 is the honest answer at the boundary."""
    import inspect
    import typing

    from models.inference import LoadRequest, SttLoadRequest, TranscribeRequest, ValidateModelRequest

    expected = typing.Optional[typing.Literal["auto", "cpu", "gpu"]]
    for model, field in (
        (LoadRequest, "audio_device"),
        (ValidateModelRequest, "audio_device"),
        (SttLoadRequest, "device"),
        (TranscribeRequest, "device"),
    ):
        assert model.model_fields[field].annotation == expected, f"{model.__name__}.{field}"
    # The raw endpoint takes it as a query param, so it is annotated rather than
    # declared on a model; it must not be the odd one out.
    assert (
        inspect.signature(ri.transcribe_audio_raw).parameters["device"].annotation == expected
    )
