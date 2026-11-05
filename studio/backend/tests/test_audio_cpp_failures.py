# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Each runtime string below is copied from audio.cpp v0.9.2 (`git grep` at that tag finds it), so
a reworded upstream message shows up here rather than as a silently generic error."""

from __future__ import annotations

from dataclasses import replace

import pytest

from core.inference import audio_cpp_failures as af
from core.inference.audio_cpp_models import AudioCppModel, AudioCppVariant, RepoFile


def _model(sizes = (8, 4, 3, 2, 1), current = 0) -> AudioCppModel:
    variants = tuple(
        AudioCppVariant(f"Q{gb}", (RepoFile(f"m-q{gb}.gguf", gb * 1024**3),), f"m-q{gb}.gguf")
        for gb in sizes
    )
    return AudioCppModel(
        id = "audio-cpp/audio.cpp-gguf/M-GGUF",
        repo_id = "audio-cpp/audio.cpp-gguf",
        folder = "M-GGUF",
        display_name = "Model M",
        family = "qwen3_tts",
        task = "tts",
        server_task = "tts",
        variant = variants[current],
        variants = variants,
        default_variant = variants[current].key,
    )


@pytest.mark.parametrize(
    "text, backend, code",
    [
        ("Could not load eSpeak-ng; install the shared library", None, "espeak_missing"),
        ("eSpeak-ng failed to initialize; check its data directory", None, "espeak_missing"),
        ("eSpeak data package missing phontab", None, "espeak_missing"),
        ("unsupported model family hint: fish_drama3", None, "family_unknown_to_runtime"),
        ("no registered model loader can load: /m/x.gguf", None, "family_unknown_to_runtime"),
        ("CUDA backend requested but registry 'CUDA' has no device 0", "cuda", "gpu_unavailable"),
        ("Failed to initialize Vulkan backend on device 0", "vulkan", "gpu_unavailable"),
        (
            "ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device",
            "cuda",
            "gpu_unavailable",
        ),
        ("ggml_vulkan: No devices found.", "vulkan", "gpu_unavailable"),
        ("failed to allocate CUDA0 backend weight buffer", "cuda", "out_of_memory"),
        (
            "ggml_gallocr_reserve_n: failed to allocate CUDA0 buffer of size 9663676416",
            "cuda",
            "out_of_memory",
        ),
        (
            "alloc: allocating 9216.00 MiB on device 0: cudaMalloc failed: out of memory",
            "cuda",
            "out_of_memory",
        ),
        ("CUDA error: out of memory\n  current device: 0", "cuda", "out_of_memory"),
        ("ggml_vulkan: Device memory allocation of size 123 failed.", "vulkan", "out_of_memory"),
        (
            "LFM2-Audio prefill graph does not fit in device memory at 9 prompt steps",
            "cuda",
            "out_of_memory",
        ),
        (
            "cannot load model 'm': estimated 9 GiB + 512 MiB headroom exceeds available CUDA memory",
            None,
            "out_of_memory",
        ),
        ("gguf_init_from_file_impl: failed to read magic", None, "file_incomplete"),
        (
            "gguf_init_from_file_impl: invalid magic characters: 'xxxx', expected 'GGUF'",
            None,
            "file_incomplete",
        ),
        ("gguf_init_from_file_impl: failed to read tensor data", None, "file_incomplete"),
        ("failed to read GGUF metadata: /m/x.gguf", None, "file_incomplete"),
        ("tensor data range is out of bounds: blk.0.attn_q.weight", None, "file_incomplete"),
        ("model path does not exist: /m/x.gguf", None, "file_incomplete"),
        # A CUDA build logs its init failure on CPU launches too.
        ("ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device", "cpu", None),
        ("failed to allocate GGUF context", "cuda", None),
        ("Failed to initialize GGML context for Granite 5 ASR encoder", "cuda", None),
        ("CosyVoice3 requires reference audio", "cuda", None),
        # A refused request that only mentions eSpeak keeps the runtime's own reason.
        ("zipvoice: tokenizer must be espeak, emilia, or simple", "cuda", None),
        ("Inflect v2 eSpeak-ng produced no phonemes", "cuda", None),
        ("The audio runtime is being updated. Try again in a moment.", "cuda", None),
        ("", "cuda", None),
    ],
)
def test_classifies_upstream_runtime_strings(text, backend, code):
    failure = af.classify_audio_cpp_failure(text, backend = backend, model = _model())
    if code is None:
        assert failure is None
        return
    assert failure.code == code and "/m/" not in failure.message
    detail = failure.detail()
    assert {"code", "message"} <= detail.keys() and not {"error", "type"} & detail.keys()
    assert af.AudioCppFailure.from_payload(failure.payload()) == failure


def test_studio_runtime_messages_classify(tmp_path, monkeypatch):
    from core.inference import audio_cpp_models as models
    from core.inference import audio_cpp_server as srv

    monkeypatch.setattr(srv, "find_audio_cpp_server_binary", lambda: None)
    missing = srv.model_runtime_problem(_model())
    assert af.classify_audio_cpp_failure(missing).code == "runtime_missing"
    binary = tmp_path / srv.BINARY_NAME
    binary.write_bytes(b"x")
    kokoro = replace(_model(), needs_espeak = True)
    espeak = af.classify_audio_cpp_failure(srv.model_runtime_problem(kokoro, str(binary)))
    assert espeak.code == "espeak_missing" and espeak.status == 503
    monkeypatch.setattr(models, "runtime_knows_family", lambda family: False)
    policy = models._family_from_spec_tasks("new_tts", {"tasks": ["tts"]}, "New TTS")
    newer = af.classify_audio_cpp_failure(policy.unsupported, model = _model())
    assert newer.code == "family_unknown_to_runtime" and "Model M" in newer.message


def test_estimate_uses_the_measured_family_profile():
    gb = 1024**3
    generic = af.estimate_resident_gb(3 * gb)
    assert generic == af.estimate_resident_gb(3 * gb, "qwen3_tts") == 9.0
    assert af.estimate_resident_gb(3 * gb, None) == generic
    assert af.estimate_resident_gb(3 * gb, "new") == generic
    # Peak grows with the input, so it keeps the generic formula.
    assert af.estimate_resident_gb(3 * gb, "vibevoice") == generic
    # Measured below the generic formula.
    assert af.estimate_resident_gb(3 * gb, "chatterbox") == 6.5
    assert af.estimate_resident_gb(int(4.4 * gb), "voxtral_tts") == 6.4
    # Measured above it: Kokoro q8 peaks at 2.7 GB, VoxCPM2 q8 at 9.7 GB and bf16 at 13.3 GB.
    assert af.estimate_resident_gb(int(0.1765 * gb), "kokoro_tts") == 3.68 > 2.7
    assert af.estimate_resident_gb(int(2.7521 * gb), "voxcpm2") == 10.75 > 9.7
    assert af.estimate_resident_gb(int(4.4445 * gb), "voxcpm2") == 14.48 > 13.3
    # ACE-Step1.5 turbo Q8_0 / BF16 (Hub bytes): 240 s cover peaks at 11.2 GB / 14.7 GB.
    assert af.estimate_resident_gb(6185468096, "ace_step") == 12.16 > 11.2
    assert af.estimate_resident_gb(10090406336, "ace_step") == 15.8 > 14.7


def test_past_the_measured_sizes_the_estimate_never_drops_below_either_profile():
    gb = 1024**3
    # A bigger model of a family measured low: the generic formula.
    assert af.estimate_resident_gb(8 * gb, "voxtral_tts") == af.estimate_resident_gb(8 * gb) == 21.5
    # A bigger model of a family measured high: its own profile.
    assert af.estimate_resident_gb(int(8.8887 * gb), "voxcpm2") == 24.26
    assert af.estimate_resident_gb(int(8.8887 * gb)) == 23.72


def test_smaller_variants_estimate_with_the_family(monkeypatch):
    model = replace(_model(sizes = (6, 3, 1), current = 0), family = "chatterbox")
    rows = af.smaller_variants(model)
    assert [(row["label"], row["estimate_gb"]) for row in rows] == [("Q3", 6.5), ("Q1", 4.5)]
    assert [row["label"] for row in af.smaller_variants(model, free_gb = 7.3)] == ["Q3", "Q1"]


def test_a_sibling_package_a_few_bytes_smaller_is_not_offered(monkeypatch):
    monkeypatch.setattr(af, "offload_options", lambda model: [])
    # ACE-Step1.5-GGUF on the Hub: base/Q8_0 is 1,152 bytes under turbo/Q8_0 and needs as much.
    sizes = {
        "turbo/BF16": 10090406336,
        "turbo/Q8_0": 6185468096,
        "base/Q8_0": 6185466944,
        "base/BF16": 10090405184,
    }
    variants = tuple(
        AudioCppVariant(key, (RepoFile(f"{key}.gguf", size),), f"{key}.gguf", label = key)
        for key, size in sizes.items()
    )
    model = replace(
        _model(),
        family = "ace_step",
        variant = variants[1],
        variants = variants,
        default_variant = variants[1].key,
    )
    assert af.smaller_variants(model) == []
    refusal = af.classify_audio_cpp_failure(
        "No single GPU has enough free memory for this native audio model.", model = model
    )
    assert "Smaller variants" not in refusal.message and "smaller_variants" not in refusal.extras


def test_placement_refusal_states_the_estimate_and_the_free_memory(monkeypatch):
    monkeypatch.setattr(af, "offload_options", lambda model: [])
    text = "No single GPU has enough free memory for this native audio model."
    model = _model(sizes = (4, 1), current = 0)
    required = af.estimate_resident_gb(4 * 1024**3)
    refusal = af.classify_audio_cpp_failure(text, model = model, free_gb = 9.3, required_gb = required)
    assert refusal.message.startswith(
        "No single GPU has enough free memory for Model M. It needs about 11.5 GB of GPU memory "
        "while running, and the most free on one GPU is 9.3 GB. Free GPU memory,"
    )
    assert refusal.message.endswith("Smaller variants: Q1 (~4.0 GB).")
    unknown = af.classify_audio_cpp_failure(text, model = model, required_gb = required)
    assert "It needs about 11.5 GB of GPU memory while running. Free GPU memory," in unknown.message
    bare = af.classify_audio_cpp_failure(text, model = model)
    assert bare.message.startswith("No single GPU has enough free memory for Model M. Free GPU")


def test_oom_lists_smaller_variants_and_offload_options(monkeypatch):
    from core.inference import audio_cpp_models as models

    monkeypatch.setattr(models, "runtime_spec", lambda family: None)
    model = _model(sizes = (4, 8, 3, 2, 1), current = 0)
    failure = af.classify_audio_cpp_failure(
        "cudaMalloc failed: out of memory", backend = "cuda", model = model
    )
    rows = failure.extras["smaller_variants"]
    assert [row["label"] for row in rows] == [
        "Q3",
        "Q2",
        "Q1",
    ] and "offload_options" not in failure.extras
    assert rows[0] == {
        "variant": model.variants[2].key,
        "label": "Q3",
        "size_bytes": 3 * 1024**3,
        "estimate_gb": af.estimate_resident_gb(3 * 1024**3),
    }
    assert "Smaller variants: Q3 (~9.0 GB)" in failure.message and failure.status == 503
    fits = af.smaller_variants(model, free_gb = af.estimate_resident_gb(2 * 1024**3))
    assert [row["label"] for row in fits] == ["Q2", "Q1"]
    assert af.smaller_variants(_model(current = 4)) == []
    on_cpu = af.classify_audio_cpp_failure(
        "failed to allocate CPU buffer of size 9", backend = "cpu", model = model
    )
    assert on_cpu.extras["device"] == "cpu" and "GPU" not in on_cpu.message.split(" Smaller")[0]
    assert af.AudioCppFailure.from_payload(on_cpu.payload()) == on_cpu
    text = "LFM2-Audio prefill graph does not fit in device memory at 9 prompt steps"
    capacity = af.classify_audio_cpp_failure(text, backend = "cuda", runtime_status = 400)
    assert (capacity.code, capacity.status, capacity.extras) == ("out_of_memory", 400, {})
    assert af.AudioCppFailure.from_payload(capacity.payload()) == capacity
    cpu = af.classify_audio_cpp_failure(text, backend = "cpu", runtime_status = 400)
    assert cpu.extras == {"device": "cpu"} and "CPU" not in cpu.message
    session = [{"name": "context_length", "type": "int"}]
    monkeypatch.setattr(models, "runtime_spec", lambda family: {"options": {"session": session}})
    assert af.offload_options(_model()) == []
    session += [
        {"name": "cpu_expert_layers", "type": "int"},
        {"name": "disk_experts", "type": "bool"},
    ]
    failure = af.classify_audio_cpp_failure(
        "cuda error: out of memory", backend = "cuda", model = _model()
    )
    assert failure.extras["offload_options"] == ["cpu_expert_layers", "disk_experts"]
    assert "set cpu_expert_layers or disk_experts" in failure.message
    assert (
        af.failure_of(af.AudioCppFailureError("boom", failure)) is failure
        and af.failure_of(KeyError()) is None
    )
    assert (
        af.AudioCppFailure.from_payload({"code": "nope"})
        is af.AudioCppFailure.from_payload(None)
        is None
    )
