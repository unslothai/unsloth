# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Embedder concurrency tests: the fast tokenizer isn't thread-safe, so encode
and token counting must be serialized (else threads panic "Already borrowed")."""

import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from core.rag import config, embeddings


@pytest.fixture(autouse = True)
def _pin_st_backend(monkeypatch):
    # Tests patch ST internals (_get), so force the ST backend.
    monkeypatch.setattr(config, "EMBED_BACKEND", "sentence-transformers")
    embeddings._reset_backend()
    yield
    embeddings._reset_backend()


class _ConcurrencyProbe:
    """Records whether two callers were in the guarded body at once."""

    def __init__(self):
        self.inside = 0
        self.saw_overlap = False
        self._g = threading.Lock()

    def enter(self):
        with self._g:
            self.inside += 1
            if self.inside > 1:
                self.saw_overlap = True
        time.sleep(0.005)  # widen the race window
        with self._g:
            self.inside -= 1


class _FakeModel:
    def __init__(self, probe):
        self._probe = probe
        self.tokenizer = _FakeTokenizer(probe)

    def encode(self, texts, **_kw):
        self._probe.enter()
        return np.zeros((len(texts), 4), dtype = np.float32)


class _FakeTokenizer:
    def __init__(self, probe):
        self._probe = probe

    def encode(self, text, **_kw):
        self._probe.enter()
        return list(range(len(text.split())))


def _hammer(fn, n = 8):
    errors: list[Exception] = []

    def worker():
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target = worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_first_encode_builds_the_selected_backend_once(monkeypatch):
    """Importing the facade is inert; the first real vector operation owns construction."""
    builds: list[str] = []

    class _Backend:
        def encode(self, texts, **_kwargs):
            return np.zeros((len(texts), 4), dtype = np.float32)

    def _build():
        builds.append("backend")
        return _Backend()

    monkeypatch.setattr(embeddings, "_build_st_backend_or_fallback", _build)

    assert embeddings._backend is None
    embeddings.encode(["first"])
    embeddings.encode(["second"])

    assert builds == ["backend"]


def test_encode_is_serialized(monkeypatch):
    probe = _ConcurrencyProbe()
    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: _FakeModel(probe))
    errors = _hammer(lambda: embeddings.encode(["alpha beta", "gamma"]))
    assert errors == []
    assert probe.saw_overlap is False  # compute lock serialized encode()


def test_token_counter_is_serialized(monkeypatch):
    probe = _ConcurrencyProbe()
    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: _FakeModel(probe))
    count = embeddings.token_counter()
    errors = _hammer(lambda: count("one two three four"))
    assert errors == []
    assert probe.saw_overlap is False  # counting shares the tokenizer lock


def test_encode_enables_parallelism_only_during_call(monkeypatch):
    seen = {}

    class _M:
        tokenizer = None

        def encode(self, texts, **_kw):
            seen["during"] = os.environ.get("TOKENIZERS_PARALLELISM")
            return np.zeros((len(texts), 4), dtype = np.float32)

    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: _M())
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    embeddings.encode(["alpha", "beta"])
    assert seen["during"] == "true"  # rayon batch tokenization enabled in-call
    assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"  # restored after


def test_token_counter_enables_parallelism_only_during_call(monkeypatch):
    seen = {}

    class _Tok:
        def encode(self, text, **_kw):
            seen["during"] = os.environ.get("TOKENIZERS_PARALLELISM")
            return list(range(len(text.split())))

    class _M:
        tokenizer = _Tok()

    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: _M())
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    count = embeddings.token_counter()
    count("alpha beta gamma")
    assert seen["during"] == "true"  # rayon enabled in-call, like _st_encode
    assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"  # restored after


def test_sentence_transformer_load_uses_live_cache(monkeypatch, tmp_path):
    observed = {}

    class FakeSentenceTransformer:
        def __init__(self, name, **kwargs):
            observed["name"] = name
            observed.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer = FakeSentenceTransformer),
    )
    monkeypatch.setattr(embeddings, "_install_torchao_stub_once", lambda: None)
    monkeypatch.setattr(embeddings, "_guard_model_security", lambda *_a, **_k: None)
    monkeypatch.setattr(embeddings, "_device", lambda: "cpu")
    monkeypatch.setattr(
        "utils.hf_cache_settings.active_hf_hub_cache",
        lambda: str(tmp_path / "selected-hub"),
    )
    embeddings._model = None
    embeddings._name = None

    embeddings._get("Org/Embedder")

    assert observed["name"] == "Org/Embedder"
    assert observed["cache_folder"] == str(tmp_path / "selected-hub")


class _SentinelLlamaBackend:
    """Stand-in for LlamaServerBackend; never spawns a real server."""


def _force_st_load_failure(monkeypatch):
    """Make the ST warm-probe raise."""

    def _boom(model_name = None):
        raise RuntimeError("torch is broken on this machine")

    monkeypatch.setattr(embeddings, "_get", _boom)


def _patch_llama_backend(monkeypatch, *, binary):
    from core.inference.llama_cpp import LlamaCppBackend
    from core.rag import embed_llama_server

    monkeypatch.setattr(LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: binary))
    monkeypatch.setattr(embed_llama_server, "LlamaServerBackend", _SentinelLlamaBackend)


def test_st_failure_falls_back_to_llama_server(monkeypatch):
    # ST can't load but llama-server is available -> use it.
    _force_st_load_failure(monkeypatch)
    _patch_llama_backend(monkeypatch, binary = "/fake/llama-server")
    embeddings._reset_backend()
    backend = embeddings._get_backend()
    assert isinstance(backend, _SentinelLlamaBackend)


def test_st_failure_without_llama_binary_reraises(monkeypatch):
    # No llama-server binary -> surface the failure, don't degrade to nothing.
    _force_st_load_failure(monkeypatch)
    _patch_llama_backend(monkeypatch, binary = None)
    embeddings._reset_backend()
    with pytest.raises(RuntimeError, match = "torch is broken"):
        embeddings._get_backend()


def test_st_success_keeps_sentence_transformers(monkeypatch):
    # Clean ST probe -> ST backend stays selected, no fallback.
    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: object())
    _patch_llama_backend(monkeypatch, binary = "/fake/llama-server")
    embeddings._reset_backend()
    backend = embeddings._get_backend()
    assert isinstance(backend, embeddings._SentenceTransformersBackend)


class _BoomOnEncodeModel:
    """Loads fine (init probe passes) but raises when encoding."""

    tokenizer = None

    def encode(self, texts, **_kw):
        raise RuntimeError("CUDA error during encode")


def test_st_encode_runtime_failure_switches_to_llama(monkeypatch):
    # encode() blows up mid-run -> switch to llama-server and stay switched.
    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: _BoomOnEncodeModel())
    _patch_llama_backend(monkeypatch, binary = "/fake/llama-server")
    calls = {}

    def _sentinel_encode(
        self,
        texts,
        *,
        model_name = None,
        normalize = True,
    ):
        calls["used"] = True
        return np.zeros((len(texts), 4), dtype = np.float32)

    monkeypatch.setattr(_SentinelLlamaBackend, "encode", _sentinel_encode, raising = False)
    embeddings._reset_backend()

    out = embeddings.encode(["alpha", "beta"])
    assert calls.get("used") is True  # retried on the llama fallback
    assert out.shape == (2, 4)
    # Switch is process-wide: later calls keep using llama, not ST.
    assert isinstance(embeddings._get_backend(), _SentinelLlamaBackend)


def test_st_encode_failure_without_llama_binary_reraises(monkeypatch):
    # No llama-server binary -> surface the encode error.
    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: _BoomOnEncodeModel())
    _patch_llama_backend(monkeypatch, binary = None)
    embeddings._reset_backend()
    with pytest.raises(RuntimeError, match = "CUDA error during encode"):
        embeddings.encode(["alpha", "beta"])


# --- a ROCm GPU that faults on its first allocation (#8474) -----------------------
#
# The fault is a SIGSEGV in the HIP runtime, so it is not an exception and cannot be
# provoked here. What is asserted is that the isolated probe's verdict is what decides
# the device, and that a condemned host lands on CPU with the SAME backend and model --
# so its vectors are unchanged and no knowledge base needs reindexing.


def _mock_device(
    monkeypatch,
    *,
    torch_device,
    is_rocm,
    probe_ok = True,
    detected = True,
):
    """Pin the device the embedder would pick, the ROCm flag, and the probe verdict."""
    from utils import device_allocation_probe as probe_mod
    from utils.hardware import hardware as hardware_mod
    from utils.hardware.hardware import DeviceType

    unsloth_device = {
        "cuda": DeviceType.CUDA,
        "xpu": DeviceType.XPU,
        "cpu": DeviceType.CPU,
    }[torch_device]
    monkeypatch.setattr(embeddings, "get_device", lambda: unsloth_device)
    monkeypatch.setattr(hardware_mod, "IS_ROCM", is_rocm)
    event = threading.Event()
    if detected:
        event.set()
    monkeypatch.setattr(hardware_mod, "DETECTION_COMPLETE", event)

    probes: list[str] = []

    def _fake_probe(device = "cuda:0"):
        probes.append(device)
        return probe_mod.DeviceAllocationProbeResult(
            ok = probe_ok,
            device = device,
            returncode = 0 if probe_ok else -11,
            reason = None if probe_ok else "killed by SIGSEGV",
            duration_seconds = 0.1,
        )

    monkeypatch.setattr(probe_mod, "probe_torch_device_allocation", _fake_probe)
    return probes


def test_failed_rocm_probe_moves_the_embedder_to_cpu(monkeypatch):
    probes = _mock_device(monkeypatch, torch_device = "cuda", is_rocm = True, probe_ok = False)
    assert embeddings._safe_torch_device() == "cpu"
    assert probes == ["cuda:0"]


def test_passing_rocm_probe_keeps_the_embedder_on_the_gpu(monkeypatch):
    probes = _mock_device(monkeypatch, torch_device = "cuda", is_rocm = True, probe_ok = True)
    assert embeddings._safe_torch_device() == "cuda"
    assert probes == ["cuda:0"]


@pytest.mark.parametrize(
    ("torch_device", "expected"),
    [("cuda", "cuda"), ("xpu", "xpu"), ("cpu", "cpu")],
)
def test_non_rocm_hosts_are_never_probed(monkeypatch, torch_device, expected):
    # NVIDIA, Intel XPU, Apple and CPU must behave exactly as they did before: no child
    # process, no new failure mode, no added startup cost.
    probes = _mock_device(monkeypatch, torch_device = torch_device, is_rocm = False)
    assert embeddings._safe_torch_device() == expected
    assert probes == []


def test_rocm_probe_is_not_run_for_a_non_gpu_device(monkeypatch):
    # ROCm build flags with no usable GPU: there is no device to probe.
    probes = _mock_device(monkeypatch, torch_device = "cpu", is_rocm = True)
    assert embeddings._safe_torch_device() == "cpu"
    assert probes == []


def test_the_probe_never_forces_hardware_detection(monkeypatch):
    # Detection imports torch. _resolve_auto can be reached (via active_backend_is_llama)
    # long before the coordinated warm is meant to pay that cost, so an undetected host
    # declines to probe rather than dragging torch into the lean main process.
    probes = _mock_device(
        monkeypatch, torch_device = "cuda", is_rocm = True, probe_ok = False, detected = False
    )
    assert embeddings._rocm_gpu_is_fatal() is False
    assert probes == []


def _observe_st_load(monkeypatch, tmp_path):
    """Capture the kwargs the SentenceTransformer constructor is handed."""
    observed = {}

    class FakeSentenceTransformer:
        def __init__(self, name, **kwargs):
            observed["name"] = name
            observed.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer = FakeSentenceTransformer),
    )
    monkeypatch.setattr(embeddings, "_install_torchao_stub_once", lambda: None)
    monkeypatch.setattr(embeddings, "_guard_model_security", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "utils.hf_cache_settings.active_hf_hub_cache",
        lambda: str(tmp_path / "hub"),
    )
    embeddings._model = None
    embeddings._name = None
    return observed


def test_degraded_cpu_load_drops_fp16(monkeypatch, tmp_path):
    # fp16 is a win on the GPU and a slow, patchily supported path on CPU, so the host we
    # pushed off its GPU must not inherit the GPU dtype.
    observed = _observe_st_load(monkeypatch, tmp_path)
    _mock_device(monkeypatch, torch_device = "cuda", is_rocm = True, probe_ok = False)

    embeddings._get("Org/Embedder")

    assert observed["device"] == "cpu"
    assert list(observed["model_kwargs"].values()) == ["float32"]


def test_a_genuine_cpu_host_loads_exactly_as_before(monkeypatch, tmp_path):
    # Not a regression dressed up as a fix: a host that never had a GPU keeps the fp16
    # load it has always done.
    observed = _observe_st_load(monkeypatch, tmp_path)
    _mock_device(monkeypatch, torch_device = "cpu", is_rocm = False)

    embeddings._get("Org/Embedder")

    assert observed["device"] == "cpu"
    assert list(observed["model_kwargs"].values()) == ["float16"]


def test_gpu_load_keeps_fp16(monkeypatch, tmp_path):
    observed = _observe_st_load(monkeypatch, tmp_path)
    _mock_device(monkeypatch, torch_device = "cuda", is_rocm = True, probe_ok = True)

    embeddings._get("Org/Embedder")

    assert observed["device"] == "cuda"
    assert list(observed["model_kwargs"].values()) == ["float16"]


def test_cpu_degradation_does_not_bypass_the_security_gate(monkeypatch, tmp_path):
    # Moving to CPU is a device decision, not a licence to load a flagged pickle.
    _observe_st_load(monkeypatch, tmp_path)
    _mock_device(monkeypatch, torch_device = "cuda", is_rocm = True, probe_ok = False)

    def _blocked(*_a, **_k):
        raise embeddings.UnsafeEmbeddingModelError("flagged repo")

    monkeypatch.setattr(embeddings, "_guard_model_security", _blocked)
    with pytest.raises(embeddings.UnsafeEmbeddingModelError):
        embeddings._get("Org/Embedder")


def test_degraded_host_still_embeds_end_to_end(monkeypatch):
    # The point of the whole change: on the #7331 host the warm completes, the embedder
    # works, and the backend is still running afterwards.
    _mock_device(monkeypatch, torch_device = "cuda", is_rocm = True, probe_ok = False)
    loaded = {}

    class _CpuModel:
        tokenizer = None

        def encode(self, texts, **_kw):
            return np.zeros((len(texts), 4), dtype = np.float32)

    def _fake_get(model_name = None):
        loaded["device"] = embeddings._safe_torch_device()
        return _CpuModel()

    monkeypatch.setattr(embeddings, "_get", _fake_get)
    embeddings._reset_backend()

    embeddings.warm()
    out = embeddings.encode(["alpha", "beta"])

    assert loaded["device"] == "cpu"
    assert out.shape == (2, 4)
    assert isinstance(embeddings._get_backend(), embeddings._SentenceTransformersBackend)
