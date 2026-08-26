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


# A child that dies of SIGSEGV is still handed to the host's core_pattern handler
# (apport on Ubuntu), which reads the whole core before the child is reaped. Marking
# the child non-dumpable first keeps the SIGSEGV this test needs and writes no core.
# RLIMIT_CORE = 0 does NOT work here, because a piped core_pattern ignores it.
# prctl is Linux-only, so the call is guarded and does nothing elsewhere.
_CRASHING_UNLESS_CPU_SCRIPT = (
    "import ctypes, sys\n"
    "if sys.argv[1] != 'cpu':\n"
    "    try:\n"
    "        ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE = 0\n"
    "    except Exception:\n"
    "        pass\n"
    "    ctypes.string_at(0)\n"
)


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
    # fp32, because the load lands on CPU. The dtype follows the device we actually
    # load on rather than how we got there, so the default CPU placement and a
    # degraded-onto-CPU load agree.
    assert list(observed["model_kwargs"].values()) == ["float32"]


def test_device_defaults_to_cpu_on_an_accelerator_host(monkeypatch):
    """A GPU must not be used just because it is there.

    This embedder loads in the backend process, where the first CUDA allocation pins a
    primary context nothing can hand back, so an idle Unsloth that indexed one document
    would carry it for the rest of the session.
    """
    monkeypatch.setattr(embeddings.config, "EMBED_DEVICE", "auto")
    monkeypatch.setattr(
        embeddings,
        "get_device",
        lambda: embeddings.DeviceType.CUDA,
    )
    assert embeddings._device() == "cpu"


def test_device_opts_in_to_the_accelerator(monkeypatch):
    """Every spelling of "use the accelerator" opts in, including the device's own name.

    An Intel user reaches for ``xpu`` and a ROCm user for ``rocm`` before either reaches
    for the generic ``gpu``; matching only ``gpu`` handed both of them CPU from a setting
    that named their hardware.
    """
    monkeypatch.setattr(
        embeddings,
        "get_device",
        lambda: embeddings.DeviceType.CUDA,
    )
    for requested in ("gpu", "GPU", " cuda ", "rocm", "hip", "xpu", "mps", "metal"):
        monkeypatch.setattr(embeddings.config, "EMBED_DEVICE", requested)
        assert embeddings._device() == "cuda", requested


def test_device_opt_in_still_yields_cpu_without_an_accelerator(monkeypatch):
    monkeypatch.setattr(embeddings.config, "EMBED_DEVICE", "gpu")
    monkeypatch.setattr(embeddings, "get_device", lambda: embeddings.DeviceType.CPU)
    assert embeddings._device() == "cpu"


def test_device_opt_in_on_apple_stays_on_cpu(monkeypatch):
    """MLX is not a torch device. Asking for a GPU must not produce a device string
    torch cannot open, which is why this stays a lookup in _TORCH_DEVICE."""
    monkeypatch.setattr(embeddings.config, "EMBED_DEVICE", "gpu")
    monkeypatch.setattr(embeddings, "get_device", lambda: embeddings.DeviceType.MLX)
    assert embeddings._device() == "cpu"


def test_unrecognized_device_setting_falls_back_without_raising(monkeypatch):
    monkeypatch.setattr(
        embeddings,
        "get_device",
        lambda: embeddings.DeviceType.CUDA,
    )
    for requested in ("", "   ", "banana", "auto", None):
        monkeypatch.setattr(embeddings.config, "EMBED_DEVICE", requested)
        assert embeddings._device() == "cpu", requested


def test_cpu_never_loads_float16(monkeypatch, tmp_path):
    """fp16 on CPU is not merely slow on older torch, it raises.

    torch 2.2 has no CPU Half kernel for LayerNorm, which every BERT runs, so an
    fp16 CPU load dies with ``"LayerNormKernelImpl" not implemented for 'Half'``.
    _SentenceTransformersBackend.encode() answers that by swapping the process to
    llama-server, so the failure would surface as a silent change of embedding space
    against an index nobody reindexed rather than as an error.
    """
    observed = {}

    class FakeSentenceTransformer:
        def __init__(self, name, **kwargs):
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
    # Every way of arriving on CPU: the default, an explicit request, a host with no
    # accelerator, and a degrade from a probe that condemned the accelerator.
    for embed_device, hardware, load_device in (
        ("auto", embeddings.DeviceType.CUDA, None),
        ("cpu", embeddings.DeviceType.CUDA, None),
        ("gpu", embeddings.DeviceType.CPU, None),
        ("gpu", embeddings.DeviceType.CUDA, "cpu"),
    ):
        monkeypatch.setattr(embeddings.config, "EMBED_DEVICE", embed_device)
        monkeypatch.setattr(embeddings, "get_device", lambda hw = hardware: hw)
        if load_device is not None:
            monkeypatch.setattr(embeddings, "_load_device", lambda d = load_device: d)
        embeddings._model = None
        embeddings._name = None
        observed.clear()

        embeddings._get("Org/Embedder")

        assert observed["device"] == "cpu", (embed_device, hardware)
        assert list(observed["model_kwargs"].values()) == ["float32"], (embed_device, hardware)
    embeddings._model = None
    embeddings._name = None


def test_opted_in_accelerator_loads_float16(monkeypatch, tmp_path):
    observed = {}

    class FakeSentenceTransformer:
        def __init__(self, name, **kwargs):
            observed.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer = FakeSentenceTransformer),
    )
    monkeypatch.setattr(embeddings, "_install_torchao_stub_once", lambda: None)
    monkeypatch.setattr(embeddings, "_guard_model_security", lambda *_a, **_k: None)
    monkeypatch.setattr(embeddings, "_load_device", lambda: "cuda")
    monkeypatch.setattr(
        "utils.hf_cache_settings.active_hf_hub_cache",
        lambda: str(tmp_path / "selected-hub"),
    )
    embeddings._model = None
    embeddings._name = None

    embeddings._get("Org/Embedder")

    assert observed["device"] == "cuda"
    assert list(observed["model_kwargs"].values()) == ["float16"]


def test_accelerator_fallback_loads_float32_on_cpu(monkeypatch, tmp_path):
    observed = {}

    class FakeSentenceTransformer:
        def __init__(self, name, **kwargs):
            observed.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer = FakeSentenceTransformer),
    )
    monkeypatch.setattr(embeddings, "_install_torchao_stub_once", lambda: None)
    monkeypatch.setattr(embeddings, "_guard_model_security", lambda *_a, **_k: None)
    monkeypatch.setattr(embeddings, "_load_device", lambda: "cpu")
    monkeypatch.setattr(embeddings, "_device", lambda: "cuda")
    monkeypatch.setattr(
        "utils.hf_cache_settings.active_hf_hub_cache",
        lambda: str(tmp_path / "selected-hub"),
    )
    embeddings._model = None
    embeddings._name = None

    embeddings._get("Org/Embedder")

    assert observed["device"] == "cpu"
    assert list(observed["model_kwargs"].values()) == ["float32"]


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


# Device selection after a fatal torch driver failure.


def _patch_probe(monkeypatch, usable):
    """Return configured probe results and record the devices checked."""
    from utils import torch_device_probe

    asked = []

    def _can_allocate(device):
        asked.append(device)
        return usable[device]

    monkeypatch.setattr(torch_device_probe, "device_can_allocate", _can_allocate)
    return asked


class _ImportIsACrash:
    """Fail if sentence-transformers is reached after both device probes crash."""

    def __getattr__(self, name):
        raise AssertionError(f"sentence-transformers was reached ({name}) on a crashing host")


def test_load_device_keeps_the_accelerator_when_it_is_usable(monkeypatch):
    monkeypatch.setattr(embeddings, "_device", lambda: "cuda")
    asked = _patch_probe(monkeypatch, {"cuda": True})
    assert embeddings._load_device() == "cuda"
    assert asked == ["cuda"]


def test_load_device_degrades_to_cpu_when_the_accelerator_crashes(monkeypatch):
    monkeypatch.setattr(embeddings, "_device", lambda: "cuda")
    _patch_probe(monkeypatch, {"cuda": False, "cpu": True})
    assert embeddings._load_device() == "cpu"


def test_load_device_raises_when_torch_crashes_on_cpu_too(monkeypatch):
    monkeypatch.setattr(embeddings, "_device", lambda: "cuda")
    _patch_probe(monkeypatch, {"cuda": False, "cpu": False})
    with pytest.raises(embeddings.TorchDeviceUnusableError):
        embeddings._load_device()


def test_load_device_does_not_probe_a_cpu_only_host(monkeypatch):
    monkeypatch.setattr(embeddings, "_device", lambda: "cpu")
    asked = _patch_probe(monkeypatch, {})
    assert embeddings._load_device() == "cpu"
    assert asked == []


def test_a_real_crashing_child_moves_the_load_to_cpu(monkeypatch):
    from utils import torch_device_probe

    monkeypatch.setenv(torch_device_probe.DISABLE_ENV_VAR, "0")
    monkeypatch.setattr(
        torch_device_probe,
        "_PROBE_SCRIPT",
        _CRASHING_UNLESS_CPU_SCRIPT,
    )
    torch_device_probe.device_can_allocate.cache_clear()
    monkeypatch.setattr(embeddings, "_device", lambda: "cuda")
    try:
        assert embeddings._load_device() == "cpu"
    finally:
        torch_device_probe.device_can_allocate.cache_clear()


def test_crashing_torch_falls_back_to_llama_server(monkeypatch):
    monkeypatch.setattr(embeddings, "_device", lambda: "cuda")
    _patch_probe(monkeypatch, {"cuda": False, "cpu": False})
    _patch_llama_backend(monkeypatch, binary = "/fake/llama-server")
    monkeypatch.setitem(sys.modules, "sentence_transformers", _ImportIsACrash())
    embeddings._model = None
    embeddings._name = None
    embeddings._reset_backend()

    assert isinstance(embeddings._get_backend(), _SentinelLlamaBackend)
