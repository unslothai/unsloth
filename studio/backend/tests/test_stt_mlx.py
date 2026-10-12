# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""mlx-audio speech-to-text checkpoints on the Transformers dictation engine (Apple Silicon)."""

import json
import queue
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import core.inference.stt_mlx as stt_mlx
import core.inference.stt_sidecar as stt_sidecar_module
import core.inference.stt_transformers_worker as worker_module
from core.inference.stt_sidecar import SttModelCompatibilityError, WhisperSttSidecar

_REPO = "mlx-community/Qwen3-ASR-0.6B-8bit"
_QWEN3_ASR_CONFIG = {"model_type": "qwen3_asr", "quantization": {"group_size": 64, "bits": 8}}

_FAKE_MLX_AUDIO = """
from types import SimpleNamespace


class _Model:
    def __init__(self, path):
        self.path = path

    def generate(self, audio):
        return SimpleNamespace(text = f"heard {len(audio)} samples from {self.path.rsplit('/', 3)[-3]}")

def load(path):
    if "broken" in str(path):
        raise RuntimeError("mlx-audio could not read this checkpoint")
    return _Model(str(path))
"""

_FAKE_MLX_CORE = """
cpu = "cpu"
default_device = None

def set_default_device(device):
    global default_device
    default_device = device
"""


@pytest.fixture
def mlx_host(monkeypatch):
    monkeypatch.setattr(stt_mlx, "mlx_stt_available", lambda: True)


def _write_snapshot(hub, repo, config):
    snapshot = hub / f"models--{repo.replace('/', '--')}" / "snapshots" / ("c" * 40)
    snapshot.mkdir(parents = True)
    (snapshot / "config.json").write_text(json.dumps(config))
    (snapshot / "preprocessor_config.json").write_text("{}")
    (snapshot / "vocab.json").write_text("{}")
    (snapshot / "merges.txt").write_text("")
    (snapshot / "model.safetensors").write_bytes(b"w" * 8)
    return snapshot


def _install_fake_mlx(tmp_path, monkeypatch):
    """Importable stand-ins for mlx and mlx-audio, on a path a spawned child also sees."""
    root = tmp_path / "fake_site"
    (root / "mlx").mkdir(parents = True)
    (root / "mlx" / "__init__.py").write_text("")
    (root / "mlx" / "core.py").write_text(_FAKE_MLX_CORE)
    (root / "mlx_audio" / "stt").mkdir(parents = True)
    (root / "mlx_audio" / "__init__.py").write_text("")
    (root / "mlx_audio" / "stt" / "__init__.py").write_text(_FAKE_MLX_AUDIO)
    monkeypatch.syspath_prepend(str(root))
    for name in ("mlx", "mlx.core", "mlx_audio", "mlx_audio.stt"):
        monkeypatch.delitem(sys.modules, name, raising = False)


def test_mlx_stt_is_unavailable_off_apple_silicon(monkeypatch):
    import utils.hardware.hardware as hardware

    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: False)
    stt_mlx.mlx_stt_available.cache_clear()
    try:
        assert stt_mlx.mlx_stt_available() is False
        assert stt_mlx.is_mlx_stt_config(_QWEN3_ASR_CONFIG) is False
    finally:
        stt_mlx.mlx_stt_available.cache_clear()


def test_mlx_stt_needs_mlx_audio_installed(monkeypatch):
    import importlib.util

    import utils.hardware.hardware as hardware

    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    stt_mlx.mlx_stt_available.cache_clear()
    try:
        assert stt_mlx.mlx_stt_available() is False
    finally:
        stt_mlx.mlx_stt_available.cache_clear()


def test_only_listed_families_and_never_whisper_go_to_mlx(mlx_host):
    assert stt_mlx.is_mlx_stt_config(_QWEN3_ASR_CONFIG) is True
    assert stt_mlx.is_mlx_stt_config({"model_type": "Qwen3_ASR"}) is True
    assert stt_mlx.is_mlx_stt_config({"model_type": "whisper"}) is False
    assert stt_mlx.is_mlx_stt_config({"model_type": "qwen3"}) is False
    assert stt_mlx.is_mlx_stt_config({}) is False
    assert stt_mlx.is_mlx_stt_config(None) is False


def test_a_cached_mlx_checkpoint_is_downloaded_only_where_mlx_audio_runs(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    snapshot = _write_snapshot(tmp_path / "hub", _REPO, _QWEN3_ASR_CONFIG)

    monkeypatch.setattr(stt_mlx, "mlx_stt_available", lambda: False)
    assert stt_sidecar_module.is_model_downloaded(_REPO) is False
    with pytest.raises(Exception):
        WhisperSttSidecar(keep_alive_seconds = 0)._ensure_model_downloaded(_REPO)

    monkeypatch.setattr(stt_mlx, "mlx_stt_available", lambda: True)
    assert stt_sidecar_module.is_model_downloaded(_REPO) is True
    cached = WhisperSttSidecar(keep_alive_seconds = 0)._ensure_model_downloaded(_REPO)
    assert cached.path == snapshot.resolve()
    assert cached.is_multilingual is None


def test_a_non_whisper_non_mlx_checkpoint_is_still_refused(mlx_host, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    snapshot = _write_snapshot(tmp_path / "hub", "someone/wav2vec2", {"model_type": "wav2vec2"})
    assert stt_sidecar_module.is_model_downloaded("someone/wav2vec2") is False
    # Past the cache lookup, the compatibility check refuses it too.
    monkeypatch.setattr(
        stt_sidecar_module, "_find_complete_cached_snapshot", lambda _model: snapshot
    )
    with pytest.raises(SttModelCompatibilityError):
        WhisperSttSidecar(keep_alive_seconds = 0)._ensure_model_downloaded("someone/wav2vec2")


@pytest.mark.parametrize(
    ("config", "runtime"), [(_QWEN3_ASR_CONFIG, "mlx"), ({"model_type": "whisper"}, None)]
)
def test_build_model_asks_the_worker_for_mlx_only_for_an_mlx_checkpoint(
    mlx_host, monkeypatch, tmp_path, config, runtime
):
    (tmp_path / "config.json").write_text(json.dumps(config))
    starts = []

    class FakeWorker:
        def start(self, *args, **kwargs):
            starts.append((args, kwargs))

    monkeypatch.setattr(worker_module, "WhisperWorker", FakeWorker)
    WhisperSttSidecar(keep_alive_seconds = 0)._build_model(
        str(tmp_path), "mps", "float32", threading.Event()
    )
    ((args, kwargs),) = starts
    assert args[:2] == (str(tmp_path), "mps")
    assert kwargs == ({"runtime": runtime} if runtime else {})


def test_an_mlx_checkpoint_never_falls_back_to_the_in_process_whisper_engine(
    mlx_host, monkeypatch, tmp_path
):
    (tmp_path / "config.json").write_text(json.dumps(_QWEN3_ASR_CONFIG))

    class NoSpawn:
        def start(self, *args, **kwargs):
            raise worker_module.SttWorkerSpawnError("sandboxed")

    monkeypatch.setattr(worker_module, "WhisperWorker", NoSpawn)
    with pytest.raises(worker_module.SttWorkerSpawnError):
        WhisperSttSidecar(keep_alive_seconds = 0)._build_model(
            str(tmp_path), "cpu", "float32", threading.Event()
        )


def test_child_loads_and_transcribes_an_mlx_checkpoint(monkeypatch, tmp_path):
    _install_fake_mlx(tmp_path, monkeypatch)
    monkeypatch.setattr(
        worker_module,
        "load_whisper",
        lambda *_a, **_k: pytest.fail("Whisper must not load an mlx checkpoint"),
    )
    cmd_queue: queue.Queue = queue.Queue()
    resp_queue: queue.Queue = queue.Queue()
    for command in (
        {
            "type": "load",
            "snapshot_path": "/hub/models--a--b/snapshots/x",
            "device": "cpu",
            "dtype": "float32",
            "runtime": "mlx",
        },
        {
            "type": "transcribe",
            "audio": np.zeros(320, dtype = np.float32).tobytes(),
            "generate_kwargs": {"language": "en", "num_beams": 5},
            "cancellable": True,
        },
        {"type": "shutdown"},
    ):
        cmd_queue.put(command)
    worker_module.run_stt_worker(
        cmd_queue = cmd_queue, resp_queue = resp_queue, cancel_event = threading.Event()
    )
    responses = []
    while not resp_queue.empty():
        responses.append(resp_queue.get_nowait())

    assert responses == [
        {"type": "loaded", "device": "cpu", "is_multilingual": None, "supports_timestamps": False},
        {"type": "text", "text": "heard 320 samples from models--a--b", "consumed": 320},
        {"type": "shutdown_ack"},
    ]
    import mlx.core

    assert mlx.core.default_device == "cpu"


@pytest.fixture
def spawned_mlx_host(mlx_host, monkeypatch, tmp_path):
    """A real spawned worker with mlx-audio stood in; torch and PyAV are not needed on that path."""
    _install_fake_mlx(tmp_path, monkeypatch)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32 = "float32"))
    monkeypatch.setattr(stt_sidecar_module, "ensure_stt_available", lambda: None)
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda _p = None: ("cpu", "float32"))
    return tmp_path / "hub"


def test_a_spawned_worker_transcribes_a_cached_mlx_checkpoint_end_to_end(spawned_mlx_host):
    """The real spawn child, the sidecar's windowing and residency."""
    _write_snapshot(spawned_mlx_host, _REPO, _QWEN3_ASR_CONFIG)

    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    try:
        with sidecar._lock:
            text = sidecar._transcribe_decoded(
                _REPO, np.zeros(45 * 16000, dtype = np.float32), {"language": "en"}
            )
        assert sidecar.loaded_model == _REPO
    finally:
        sidecar.unload()

    # 45 s of silence: one 30 s window, then the 15 s rest.
    assert text == (
        "heard 480000 samples from models--mlx-community--Qwen3-ASR-0.6B-8bit "
        "heard 240000 samples from models--mlx-community--Qwen3-ASR-0.6B-8bit"
    )
    assert sidecar.loaded_model is None


def test_an_mlx_load_failure_reaches_the_caller(spawned_mlx_host):
    _write_snapshot(spawned_mlx_host, "someone/broken-asr", _QWEN3_ASR_CONFIG)

    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    with pytest.raises(Exception, match = "mlx-audio could not read this checkpoint"):
        sidecar.load("someone/broken-asr")
    assert sidecar.loaded_model is None


def test_the_inventory_lists_a_cached_mlx_checkpoint_as_speech_to_text(mlx_host, tmp_path):
    from hub.services.models.cache_inventory import (
        _cache_inventory_fields,
        _cached_model_local_metadata,
    )

    snapshot = _write_snapshot(tmp_path / "hub", _REPO, _QWEN3_ASR_CONFIG)
    metadata = _cached_model_local_metadata(snapshot.parent.parent, snapshot)
    assert metadata.get("_hidden_stt") is True
    assert metadata.get("_mlx_stt") is True
    fields = _cache_inventory_fields(_REPO, "safetensors", stt_only = True)
    assert fields["capabilities"]["can_chat"] is False


def test_the_inventory_leaves_the_checkpoint_alone_off_a_mac(monkeypatch, tmp_path):
    from hub.services.models.cache_inventory import _cached_model_local_metadata

    monkeypatch.setattr(stt_mlx, "mlx_stt_available", lambda: False)
    snapshot = _write_snapshot(tmp_path / "hub", _REPO, _QWEN3_ASR_CONFIG)
    metadata = _cached_model_local_metadata(snapshot.parent.parent, snapshot)
    assert "_hidden_stt" not in metadata
    assert "_mlx_stt" not in metadata
