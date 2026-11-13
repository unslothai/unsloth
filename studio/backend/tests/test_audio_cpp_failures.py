# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Each runtime string below is copied from audio.cpp v0.9.2 (`git grep` at that tag finds it), so
a reworded upstream message shows up here rather than as a silently generic error."""

from __future__ import annotations

from dataclasses import replace

import pytest

from core.inference import audio_cpp_failures as af
from core.inference.audio_cpp_models import AudioCppModel, AudioCppVariant, RepoFile

_REAL_RUNTIME_BACKEND = af._runtime_backend


@pytest.fixture(autouse = True)
def _cuda_audio_runtime(monkeypatch):
    monkeypatch.setattr(af, "_runtime_backend", lambda: "cuda")


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
        family = "neutts",
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


def test_estimate_uses_the_measured_family_profile(monkeypatch):
    monkeypatch.setattr(af, "_runtime_backend", lambda: "cuda")
    gb = 1024**3
    generic = af.estimate_resident_gb(3 * gb)
    assert generic == af.estimate_resident_gb(3 * gb, "neutts") == 9.0
    assert af.estimate_resident_gb(3 * gb, None) == generic
    assert af.estimate_resident_gb(3 * gb, "new") == generic
    # Peak grows with the input, so it keeps the generic formula.
    assert af.estimate_resident_gb(3 * gb, "vibevoice") == generic
    # Measured below the generic formula.
    assert af.estimate_resident_gb(3 * gb, "chatterbox") == 6.13
    assert af.estimate_resident_gb(int(4.4 * gb), "voxtral_tts") == 6.4
    # Measured above it (#16514, whole device): Kokoro q8 2.74 GiB, VoxCPM2 q8 9.66 and bf16 13.31.
    assert af.estimate_resident_gb(int(0.1765 * gb), "kokoro_tts") == 3.65 >= 2.74 + 0.9
    assert af.estimate_resident_gb(int(2.7521 * gb), "voxcpm2") == 10.75 >= 9.66 + 0.9
    assert af.estimate_resident_gb(int(4.4445 * gb), "voxcpm2") == 14.48 >= 13.31 + 0.9
    # ACE-Step1.5 turbo Q8_0 / BF16 (Hub bytes): a 240 s cover peaks at 11.19 / 14.74 GiB on CUDA.
    assert af.estimate_resident_gb(6185468096, "ace_step") == 12.11 > 11.19 + 0.9
    assert af.estimate_resident_gb(10090406336, "ace_step") == 15.67 > 14.74 + 0.9


GIB = 1024**3
# family, package bytes, worst CUDA peak, worst Vulkan peak (GiB) for every measured variant.
MEASURED_PEAKS = [
    ("ace_step", 6185468096, 11.19, 11.22),
    ("ace_step", 10090406336, 14.74, 15.31),
    ("breeze_tts", int(3.534 * GIB), 5.5, 6.83),
    ("breeze_tts", int(4.153 * GIB), 6.12, 8.39),
    ("breeze_tts", int(4.731 * GIB), 6.87, 11.18),
    ("breeze_tts", int(6.839 * GIB), 8.0, 9.79),
    ("chatterbox", int(1.945 * GIB), 5.16, 4.82),
    ("chatterbox", int(3.487 * GIB), 5.23, 5.19),
    ("cosyvoice3", int(2.103 * GIB), 3.08, 4.73),
    ("cosyvoice3", int(6.515 * GIB), 5.81, 7.36),
    ("heartmula", 7659762592, 18.4, 19.58),
    ("higgs_audio_tts", int(4.745 * GIB), 6.19, 6.47),
    ("higgs_audio_tts", int(7.918 * GIB), 9.4, 9.55),
    ("kokoro_tts", int(0.177 * GIB), 2.41, 2.99),
    ("kokoro_tts", int(0.197 * GIB), 2.39, 2.12),
    ("kokoro_tts", int(0.869 * GIB), 2.41, 3.05),
    ("midashenglm_gen", 3381370752, 3.93, 4.42),
    ("minimax_music3", int(7.866 * GIB), 13.08, 14.0),
    ("minimax_music3", int(12.609 * GIB), 16.35, 17.26),
    ("moss_tts_v15", int(8.622 * GIB), 7.79, 10.34),
    ("moss_tts_v15", int(12.279 * GIB), 11.25, 12.6),
    ("moss_tts_v15", int(19.134 * GIB), 18.0, 20.58),
    ("omnivoice", int(1.258 * GIB), 4.44, 6.82),
    ("omnivoice", int(1.527 * GIB), 3.96, 4.92),
    ("qwen3_tts", int(1.854 * GIB), 5.88, 9.18),
    ("qwen3_tts", int(2.343 * GIB), 6.36, 9.66),
    ("qwen3_tts", int(2.510 * GIB), 6.73, 9.5),
    ("qwen3_tts", int(2.624 * GIB), 7.47, 10.84),
    ("qwen3_tts", int(3.892 * GIB), 8.85, 12.2),
    ("qwen3_tts", int(3.915 * GIB), 8.11, 12.06),
    ("qwen3_tts", int(4.232 * GIB), 8.2, 11.49),
    ("stable_audio", 1683576608, 2.24, 2.01),
    ("stable_audio", 2358473088, 2.8, 2.54),
    ("stable_audio", 3582601408, 5.27, 5.34),
    ("stable_audio", 5834365856, 7.3, 9.48),
    ("voxcpm2", int(2.752 * GIB), 8.62, 8.94),
    ("voxcpm2", int(4.445 * GIB), 11.09, 11.15),
    ("voxcpm2", int(4.620 * GIB), 11.09, 13.11),
    ("voxcpm2", int(8.889 * GIB), 19.06, 22.36),
    ("voxtral_tts", int(4.399 * GIB), 5.31, 5.92),
    ("voxtral_tts", int(7.479 * GIB), 8.38, 11.15),
    ("yue2", 2933424181, 6.78, 6.52),
    ("yue2", 4531978293, 8.27, 8.58),
    ("yue2", 7529267253, 11.19, 11.87),
]


@pytest.mark.parametrize("family, size_bytes, cuda_peak, vulkan_peak", MEASURED_PEAKS)
@pytest.mark.parametrize("backend", ["cuda", "vulkan"])
def test_each_runtime_profile_covers_its_measured_peak(
    monkeypatch, family, size_bytes, cuda_peak, vulkan_peak, backend
):
    monkeypatch.setattr(af, "_runtime_backend", lambda: backend)
    monkeypatch.setattr(af, "_may_have_pre_ampere_nvidia_card", lambda: False)
    peak = cuda_peak if backend == "cuda" else vulkan_peak
    assert af.estimate_resident_gb(size_bytes, family) >= peak + 0.9


@pytest.mark.parametrize(
    "family, size_bytes, cuda, vulkan",
    [
        # #16627's 10 GB card (about 9.4 GiB free) fits these two on CUDA.
        ("yue2", 4531978293, 9.23, 9.51),
        ("stable_audio", 5834365856, 9.32, 10.43),
        ("qwen3_tts", int(1.854 * GIB), 7.22, 10.91),
        ("chatterbox", int(1.945 * GIB), 6.08, 5.74),
    ],
)
def test_the_runtime_backend_picks_the_profile(monkeypatch, family, size_bytes, cuda, vulkan):
    monkeypatch.setattr(af, "_may_have_pre_ampere_nvidia_card", lambda: False)
    either = max(cuda, vulkan)
    for backend, expected in (("cuda", cuda), ("vulkan", vulkan), (None, either), ("hip", either)):
        monkeypatch.setattr(af, "_runtime_backend", lambda b = backend: b)
        assert af.estimate_resident_gb(size_bytes, family) == expected, backend


@pytest.mark.parametrize(
    "backend, at_3, at_9_5",
    [
        ("cuda", 9.0, 25.25),
        ("vulkan", 10.05, 22.07),
        (None, 10.05, 25.25),
        ("hip", 10.05, 25.25),
    ],
)
def test_the_fallback_follows_the_runtime_backend(monkeypatch, backend, at_3, at_9_5):
    monkeypatch.setattr(af, "_runtime_backend", lambda: backend)
    gb = 1024**3
    assert af.estimate_resident_gb(3 * gb, "neutts") == at_3
    assert af.estimate_resident_gb(int(9.5 * gb), "neutts") == at_9_5
    assert af.estimate_resident_gb(3 * gb, "vibevoice") == 9.0
    assert af.estimate_resident_gb(int(9.85 * gb), "vibevoice") == 26.12


def test_the_runtime_backend_is_the_installed_servers(monkeypatch):
    from core.inference import audio_cpp_server as srv

    monkeypatch.setattr(srv, "find_audio_cpp_server_binary", lambda: None)
    assert _REAL_RUNTIME_BACKEND() is None
    monkeypatch.setattr(srv, "find_audio_cpp_server_binary", lambda: "/opt/audiocpp_server")
    monkeypatch.setattr(
        srv, "select_backend", lambda binary, force_cpu: f"vulkan:{binary}:{force_cpu}"
    )
    assert _REAL_RUNTIME_BACKEND() == "vulkan:/opt/audiocpp_server:False"


# YuE2 package bytes on the Hub (model + VAE + sidecars): Q4_0, Q8_0, BF16.
YUE2_BYTES = (2933424181, 4531978293, 7529267253)
# MEASURED is the Vulkan profile, which also wins with no runtime installed.
MEASURED = (7.73, 9.51, 12.86)
MEASURED_CUDA = (7.69, 9.23, 12.1)
GENERIC = (8.33, 12.05, 19.03)
GENERIC_VULKAN = (9.55, 12.31, 17.47)
GENERIC_EITHER = (9.55, 12.31, 19.03)


def _cards(
    monkeypatch,
    *answers,
    driver = True,
):
    from utils.hardware import nvidia

    queue = [answer for answer in answers for _ in YUE2_BYTES]
    monkeypatch.setattr(nvidia, "get_compute_capabilities", lambda: dict(queue.pop(0)))
    monkeypatch.setattr(nvidia, "has_nvidia_driver", lambda: driver)


def _yue2():
    return tuple(af.estimate_resident_gb(size, "yue2") for size in YUE2_BYTES)


@pytest.mark.parametrize(
    "capabilities, driver, profiled",
    [
        ({0: (8, 6)}, True, True),
        ({0: (8, 9)}, True, True),
        ({0: (12, 0)}, True, True),
        ({}, False, True),
        ({0: (7, 5)}, True, False),
        ({0: (8, 9), 1: (7, 5)}, True, False),
        ({}, True, False),
    ],
)
@pytest.mark.parametrize(
    "backend, measured, generic",
    [
        (None, MEASURED, GENERIC_EITHER),
        ("vulkan", MEASURED, GENERIC_VULKAN),
        ("cuda", MEASURED_CUDA, GENERIC),
    ],
)
def test_yue2_uses_its_profile_unless_a_pre_ampere_nvidia_card_may_be_present(
    monkeypatch, capabilities, driver, profiled, backend, measured, generic
):
    monkeypatch.setattr(af, "_runtime_backend", lambda: backend)
    _cards(monkeypatch, capabilities, driver = driver)
    assert _yue2() == (measured if profiled else generic)


@pytest.mark.parametrize(
    "smi, procfs_gpus, expected",
    [(None, 0, False), ("/usr/bin/nvidia-smi", 0, True), (None, 1, True)],
)
def test_an_nvidia_host_is_told_from_one_without_nvidia(monkeypatch, smi, procfs_gpus, expected):
    from utils.hardware import nvidia

    monkeypatch.setattr(nvidia.shutil, "which", lambda name: smi)
    monkeypatch.setattr(nvidia, "_linux_nvidia_procfs_gpu_count", lambda: procfs_gpus)
    assert nvidia.has_nvidia_driver() is expected


def test_the_yue2_estimate_follows_the_current_inventory(monkeypatch):
    _cards(monkeypatch, {0: (8, 6)}, {0: (8, 6), 1: (7, 5)}, {0: (8, 6)})
    assert (_yue2(), _yue2(), _yue2()) == (MEASURED_CUDA, GENERIC, MEASURED_CUDA)


def test_past_the_measured_sizes_the_estimate_never_drops_below_either_profile():
    gb = 1024**3
    # A bigger model of a family measured low: the generic formula.
    assert af.estimate_resident_gb(8 * gb, "voxtral_tts") == af.estimate_resident_gb(8 * gb) == 21.5
    # A bigger model of a family measured high: its own profile.
    assert af.estimate_resident_gb(int(9.5 * gb), "voxcpm2") == 25.6
    assert af.estimate_resident_gb(int(9.5 * gb)) == 25.25


def test_smaller_variants_estimate_with_the_family(monkeypatch):
    monkeypatch.setattr(af, "_runtime_backend", lambda: "cuda")
    model = replace(_model(sizes = (6, 3, 1), current = 0), family = "higgs_audio_tts")
    rows = af.smaller_variants(model)
    assert [(row["label"], row["estimate_gb"]) for row in rows] == [("Q3", 5.4), ("Q1", 3.4)]
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
    close = af.classify_audio_cpp_failure(text, model = model, free_gb = 9.5, required_gb = 9.51)
    assert (
        "It needs about 9.51 GB of GPU memory while running, and the most free on one GPU is 9.50 GB."
        in close.message
    )


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
