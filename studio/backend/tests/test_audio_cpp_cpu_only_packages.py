# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A CPU-only audio.cpp package (KittenTTS2 TQ2_1, which aborts on CUDA) runs on the CPU on Auto."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

import routes.inference as ri
from core.inference import audio_cpp_models as models
from core.inference import audio_cpp_server as srv
from core.inference import audio_device
from core.inference.audio_cpp_models import AudioCppModel, RepoFile, quant_label

FOLDER = "KittenTTS2-GGUF"
FILES = [
    RepoFile(f"{FOLDER}/kitten-tts2-20261021-q8_0.gguf", 3),
    RepoFile(f"{FOLDER}/kitten-tts2-20261021-q4_0.gguf", 2),
    RepoFile(f"{FOLDER}/kitten-tts2-20261021-tq2_1.gguf", 1),
]


def _kitten(key: str) -> AudioCppModel:
    variants = tuple(models._single_file_variants(FILES, FOLDER))
    chosen = next(v for v in variants if v.key == key)
    return AudioCppModel(
        id = f"audio-cpp/audio.cpp-gguf/{FOLDER}",
        repo_id = "audio-cpp/audio.cpp-gguf",
        folder = FOLDER,
        display_name = FOLDER,
        family = "kitten_tts2",
        task = "tts",
        server_task = "tts",
        variant = chosen,
        variants = variants,
        default_variant = "Q8_0",
    )


def test_the_ternary_file_is_labelled_tq2_1_and_only_it_is_cpu_only():
    assert quant_label("kitten-tts2-20261021-tq2_1.gguf") == "TQ2_1"
    assert sorted(v.key for v in models._single_file_variants(FILES, FOLDER)) == [
        "Q4_0",
        "Q8_0",
        "TQ2_1",
    ]
    assert _kitten("TQ2_1").cpu_only
    assert not _kitten("Q8_0").cpu_only and not _kitten("Q4_0").cpu_only


@pytest.mark.parametrize("device", [None, "auto", "gpu"])
def test_auto_runs_a_cpu_only_package_on_the_cpu(monkeypatch, device):
    monkeypatch.setattr(srv, "runtime_runs_on_cpu", lambda: False)
    monkeypatch.setattr(audio_device, "audio_device_default", lambda: "auto")
    tts = _kitten("TQ2_1").audio_type
    assert audio_device.audio_load_runs_on_cpu(tts, device, _kitten("TQ2_1"))
    assert not audio_device.audio_load_runs_on_cpu(tts, device, _kitten("Q8_0"))
    assert not audio_device.audio_load_runs_on_cpu(tts, device)


def test_the_server_never_starts_a_cpu_only_package_on_a_gpu_backend(monkeypatch):
    seen = []
    monkeypatch.setattr(srv, "ensure_binary", lambda: "/bin/audiocpp_server")
    monkeypatch.setattr(srv, "model_runtime_problem", lambda model, binary = None: None)
    monkeypatch.setattr(
        srv, "select_backend", lambda binary, force_cpu: "cpu" if force_cpu else "cuda"
    )
    monkeypatch.setattr(
        srv.AudioCppServer,
        "_start_once",
        classmethod(lambda cls, model, model_id, backend, *rest: seen.append(backend) or backend),
    )
    assert srv.AudioCppServer.start(_kitten("TQ2_1"), "/m.gguf") == "cpu"
    assert srv.AudioCppServer.start(_kitten("Q8_0"), "/m.gguf") == "cuda"
    assert srv.AudioCppServer.start(_kitten("Q8_0"), "/m.gguf", force_cpu = True) == "cpu"


def test_the_preflight_sizes_no_gpu_for_a_cpu_only_package(monkeypatch):
    monkeypatch.setattr(srv, "runtime_runs_on_cpu", lambda: False)
    monkeypatch.setattr(srv, "model_runtime_problem", lambda model, binary = None: None)
    model = _kitten("TQ2_1")
    config = SimpleNamespace(identifier = model.id, audio_type = model.audio_type, audio_cpp = model)
    request = SimpleNamespace(gpu_ids = None, hf_token = None, max_seq_length = 2048, audio_device = "auto")
    placement = ri._LoadPlacement(None, None, False, False)
    assert ri._native_audio_cpu_load(config, request)
    # Returned untouched: no GPU chosen and no VRAM estimate, so a full card cannot refuse it.
    assert (
        asyncio.run(ri._preflight_native_audio_placement(config, request, placement)) is placement
    )
    gpu = replace(config.__dict__["audio_cpp"], variant = _kitten("Q8_0").variant)
    assert not ri._native_audio_cpu_load(
        SimpleNamespace(**{**config.__dict__, "audio_cpp": gpu}), request
    )


@pytest.mark.parametrize("key, cpu_only", [("TQ2_1", True), ("Q8_0", False)])
def test_an_orchestrator_load_sends_cpu_to_the_worker_and_records_the_package(
    monkeypatch, key, cpu_only
):
    from unittest.mock import MagicMock, patch

    from core.inference import orchestrator as orch_mod

    class _NoThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return None

    with patch.object(orch_mod.threading, "Thread", _NoThread):
        orch = orch_mod.InferenceOrchestrator()
    monkeypatch.setattr(srv, "runtime_runs_on_cpu", lambda: False)
    model = _kitten(key)
    config = SimpleNamespace(
        identifier = model.id, gguf_variant = None, audio_type = model.audio_type, audio_cpp = model
    )
    model_info = {"identifier": model.id, "audio_type": model.audio_type}
    if cpu_only:
        model_info["audio_cpp_backend"] = "cpu"
    gpu_selection = MagicMock(return_value = ([0], {"selection_mode": "auto"}))
    spawn = MagicMock()
    with (
        patch.object(orch_mod, "prepare_gpu_selection", gpu_selection),
        patch.object(orch, "_ensure_subprocess_alive", return_value = False),
        patch.object(orch, "_spawn_subprocess", spawn),
        patch.object(
            orch, "_wait_response", return_value = {"success": True, "model_info": model_info}
        ),
        patch("utils.transformers_version.needs_transformers_5", return_value = False),
    ):
        assert orch.load_model(config = config, audio_device = "auto")

    sub_config = spawn.call_args.args[0]
    resident = orch.models[model.id]
    assert sub_config["audio_device"] == ("cpu" if cpu_only else "auto")
    # No card is sized or chosen for a load the worker runs on the CPU.
    assert gpu_selection.called is not cpu_only
    assert sub_config["gpu_selection"]["selection_mode"] == ("cpu_audio" if cpu_only else "auto")
    assert resident["audio_cpu_only"] is cpu_only and resident["audio_cpu"] is cpu_only
    # Auto again: the resident package already sits where Auto would put it.
    assert ri._resident_audio_placement_matches(orch, SimpleNamespace(audio_device = "auto"))


@pytest.mark.parametrize("device", ["auto", "cpu", "gpu"])
def test_a_resident_cpu_only_package_is_not_reloaded_for_its_device(device):
    resident = {"audio_type": "audiocpp_tts", "audio_cpu": True, "audio_cpu_only": True}
    backend = SimpleNamespace(active_model_name = "m", models = {"m": resident})
    assert ri._resident_audio_placement_matches(backend, SimpleNamespace(audio_device = device))
