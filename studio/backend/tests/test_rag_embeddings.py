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

    def _build(model_name = None):
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


def test_token_counter_reacquires_backend_retired_between_chunk_calls(monkeypatch):
    from core.rag.embed_llama_server import LlamaServerBackend

    retired = LlamaServerBackend()
    replacement = LlamaServerBackend()
    monkeypatch.setattr(embeddings, "_get_backend", lambda *_a, **_k: retired)
    count = embeddings.token_counter("org/embedder")

    retired._closed = True
    monkeypatch.setattr(embeddings, "_get_backend", lambda *_a, **_k: replacement)
    monkeypatch.setattr(
        replacement,
        "_post",
        lambda path, payload, **_k: {"tokens": [1, 2, 3]},
    )

    assert count("the next chunk") == 3


def test_token_counter_does_not_hide_non_lifecycle_errors(monkeypatch):
    class _BrokenCounterBackend:
        _closed = False

        def token_counter(self, *, model_name = None):
            def _raise(_text):
                raise RuntimeError("invalid tokenizer response")

            return _raise

    monkeypatch.setattr(embeddings, "_get_backend", lambda *_a, **_k: _BrokenCounterBackend())
    with pytest.raises(RuntimeError, match = "invalid tokenizer response"):
        embeddings.token_counter()("chunk")


def test_st_unload_waits_for_encode_admitted_before_model_lookup(monkeypatch):
    entered_lookup = threading.Event()
    finish_lookup = threading.Event()
    unload_done = threading.Event()
    order = []
    errors = []

    class _Model:
        def encode(self, texts, **kwargs):
            order.append("encode")
            return np.zeros((len(texts), 2), dtype = np.float32)

    def _get(model_name = None):
        entered_lookup.set()
        assert finish_lookup.wait(timeout = 2)
        return _Model()

    def _encode():
        try:
            embeddings._st_encode(["chunk"])
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    monkeypatch.setattr(embeddings, "_get", _get)
    monkeypatch.setattr(embeddings, "_model", object())
    monkeypatch.setattr(embeddings, "_name", "org/embedder")
    worker = threading.Thread(target = _encode)
    worker.start()
    assert entered_lookup.wait(timeout = 2)

    def _unload():
        embeddings._release_st_model()
        order.append("unload")
        unload_done.set()

    closer = threading.Thread(target = _unload)
    closer.start()
    assert unload_done.wait(timeout = 0.05) is False
    finish_lookup.set()
    worker.join(timeout = 2)
    closer.join(timeout = 2)

    assert errors == []
    assert order == ["encode", "unload"]
    assert unload_done.is_set()


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


def test_loaded_state_belongs_to_the_resident_sentence_transformer(monkeypatch):
    monkeypatch.setattr(embeddings, "_backend", embeddings._SentenceTransformersBackend())
    monkeypatch.setattr(embeddings, "_model", object())
    monkeypatch.setattr(embeddings, "_name", "org/resident")

    assert embeddings.backend_is_loaded() is True
    assert embeddings.backend_is_loaded("org/resident") is True
    assert embeddings.backend_is_loaded("org/new-selection") is False


def test_loaded_state_belongs_to_the_resident_gguf_repo(monkeypatch):
    # A live process, since residency now means the subprocess is actually there;
    # see test_a_dead_llama_process_is_not_reported_as_loaded.
    backend = SimpleNamespace(_model_repo = "org/resident-GGUF", _process_alive = lambda: True)
    monkeypatch.setattr(embeddings, "_backend", backend)
    monkeypatch.setattr(embeddings, "_is_llama_backend", lambda value: value is backend)
    monkeypatch.setattr(
        embeddings.config,
        "effective_gguf_repo_for_embedding_model",
        lambda model: f"{model}-GGUF",
    )

    assert embeddings.backend_is_loaded("org/resident") is True
    assert embeddings.backend_is_loaded("org/new-selection") is False


def test_unload_clears_the_sentence_transformer_weights(monkeypatch):
    embeddings._backend = embeddings._SentenceTransformersBackend()
    embeddings._backend_key = embeddings._current_backend_key()
    embeddings._model = object()
    embeddings._name = "org/embedder"

    assert embeddings.release_backend() is True
    assert embeddings._model is None
    assert embeddings._name is None


def test_pending_sentence_transformer_refuses_implicit_download_and_llama_fallback(monkeypatch):
    import utils.embedding_model_settings as ems
    import utils.utils as utils

    class _MustNotLoad:
        def __init__(self, *args, **kwargs):  # pragma: no cover - failure is the assertion
            raise AssertionError("pending model reached SentenceTransformer")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer = _MustNotLoad),
    )
    monkeypatch.setattr(ems, "get_stored_download_pending", lambda model: True)
    monkeypatch.setattr(utils, "hf_cache_snapshot_is_loadable", lambda model: False)
    monkeypatch.setattr(embeddings, "_install_torchao_stub_once", lambda: None)
    monkeypatch.setattr(embeddings, "_load_device", lambda: "cpu")
    monkeypatch.setattr(
        embeddings,
        "_try_make_llama_backend",
        lambda: (_ for _ in ()).throw(AssertionError("pending ST fell back to llama")),
    )
    embeddings._model = None
    embeddings._name = None

    with pytest.raises(embeddings.EmbeddingModelDownloadRequiredError, match = "not downloaded"):
        embeddings._build_st_backend_or_fallback()


def test_replacing_a_llama_backend_shuts_it_down(monkeypatch):
    calls = []

    class _OldLlama:
        def _shutdown(self):
            calls.append("shutdown")

    monkeypatch.setattr(embeddings.config, "EMBED_BACKEND", "sentence-transformers")
    monkeypatch.setattr(
        embeddings,
        "_build_st_backend_or_fallback",
        lambda *_a, **_k: embeddings._SentenceTransformersBackend(),
    )
    embeddings._backend = _OldLlama()
    embeddings._backend_key = "stale"

    assert isinstance(embeddings._get_backend(), embeddings._SentenceTransformersBackend)
    assert calls == ["shutdown"]


def test_explicit_llama_backend_disallows_st_resolution(monkeypatch):
    monkeypatch.setattr(embeddings.config, "EMBED_BACKEND", "llama-server")
    monkeypatch.setattr(embeddings, "_forced_backends", {})
    assert embeddings.sentence_transformers_fallback_allowed() is False

    monkeypatch.setattr(embeddings.config, "EMBED_BACKEND", "auto")
    assert embeddings.sentence_transformers_fallback_allowed() is True
    monkeypatch.setattr(
        embeddings,
        "_forced_backends",
        {embeddings.config.effective_embedding_model(): "llama-server"},
    )
    assert embeddings.sentence_transformers_fallback_allowed() is False


def test_forced_llama_fallback_is_scoped_to_the_model_that_failed(monkeypatch):
    monkeypatch.setattr(embeddings.config, "EMBED_BACKEND", "auto")
    monkeypatch.setattr(embeddings, "_forced_backends", {"org/failed": "llama-server"})
    monkeypatch.setattr(
        embeddings,
        "_resolve_auto_for_model",
        lambda model = None: "sentence-transformers",
    )

    assert embeddings.sentence_transformers_fallback_allowed("org/failed") is False
    assert embeddings.sentence_transformers_fallback_allowed("org/new-model") is True
    assert embeddings.resolved_backend_for_model("org/failed") == "llama-server"
    assert embeddings.resolved_backend_for_model("org/new-model") == "sentence-transformers"


def test_runtime_preflight_predicts_the_supported_llama_fallback(monkeypatch):
    monkeypatch.setattr(embeddings.config, "EMBED_BACKEND", "auto")
    monkeypatch.setattr(embeddings, "_forced_backends", {})
    monkeypatch.setattr(
        embeddings,
        "_resolve_auto_for_model",
        lambda model = None: "sentence-transformers",
    )
    monkeypatch.setattr(embeddings, "sentence_transformers_runtime_available", lambda: False)
    monkeypatch.setattr(embeddings, "_llama_server_runtime_available", lambda: True)

    assert embeddings.resolved_backend_for_model("org/embedder") == "llama-server"


def test_runtime_preflight_keeps_st_when_no_fallback_exists(monkeypatch):
    monkeypatch.setattr(embeddings.config, "EMBED_BACKEND", "auto")
    monkeypatch.setattr(embeddings, "_forced_backends", {})
    monkeypatch.setattr(
        embeddings,
        "_resolve_auto_for_model",
        lambda model = None: "sentence-transformers",
    )
    monkeypatch.setattr(embeddings, "sentence_transformers_runtime_available", lambda: False)
    monkeypatch.setattr(embeddings, "_llama_server_runtime_available", lambda: False)

    assert embeddings.resolved_backend_for_model("org/embedder") == "sentence-transformers"


def test_runtime_preflight_catches_a_fatal_torch_device_mismatch(monkeypatch):
    monkeypatch.setattr(
        embeddings,
        "_load_device",
        lambda: (_ for _ in ()).throw(embeddings.TorchDeviceUnusableError("broken torch")),
    )

    assert embeddings.sentence_transformers_runtime_available() is False


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
    embeddings._model = object()
    embeddings._name = "org/embedder"

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
    # The failed ST weights are no longer reachable, including after the
    # published backend became llama rather than an ST wrapper.
    assert embeddings._model is None
    assert embeddings._name is None
    assert embeddings.config.effective_embedding_model() in embeddings._forced_backends
    # Switch is process-wide: later calls keep using llama, not ST.
    assert isinstance(embeddings._get_backend(), _SentinelLlamaBackend)
    # It outranks what the saved model would otherwise resolve to, so a model that
    # asks for ST cannot walk the process back into the encoder that just failed.
    monkeypatch.setattr(embeddings, "_resolve_auto_for_model", lambda: "sentence-transformers")
    assert isinstance(embeddings._get_backend(), _SentinelLlamaBackend)
    # An explicit unload is a fresh start, so the pin does not outlive it.
    embeddings._reset_backend()
    assert embeddings._forced_backends == {}


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


def test_encode_reacquires_a_backend_retired_between_batches(monkeypatch):
    """`release_backend` promises the next embed rebuilds. Without a reacquire here
    that promise held only for `token_counter`, and pressing Unload mid-ingestion
    failed the document being indexed instead of continuing it."""
    from core.rag.embed_llama_server import LlamaServerBackend

    retired = LlamaServerBackend()
    replacement = LlamaServerBackend()
    retired._closed = True
    served = np.zeros((1, 4), dtype = np.float32)
    monkeypatch.setattr(
        replacement,
        "encode",
        lambda texts, **kwargs: served,
    )
    backends = iter([retired, replacement])
    monkeypatch.setattr(embeddings, "_get_backend", lambda *_a, **_k: next(backends))

    assert embeddings.encode(["chunk"]) is served
    # The identity must name the backend that actually produced the vectors.
    assert embeddings._served_by.backend is replacement


def test_encode_does_not_hide_a_non_lifecycle_runtime_error(monkeypatch):
    class _BrokenBackend:
        _closed = False

        def encode(self, texts, **kwargs):
            raise RuntimeError("llama-server returned no embedding")

    monkeypatch.setattr(embeddings, "_get_backend", lambda *_a, **_k: _BrokenBackend())
    with pytest.raises(RuntimeError, match = "returned no embedding"):
        embeddings.encode(["chunk"])


def test_encode_surfaces_the_unload_when_no_replacement_is_published(monkeypatch):
    from core.rag.embed_llama_server import LlamaServerBackend

    retired = LlamaServerBackend()
    retired._closed = True
    monkeypatch.setattr(embeddings, "_get_backend", lambda *_a, **_k: retired)
    with pytest.raises(RuntimeError, match = "was unloaded"):
        embeddings.encode(["chunk"])


def test_the_token_counter_follows_an_unloaded_sentence_transformer(monkeypatch):
    """The tokenizer used to be captured when the counter was built and held for the
    whole document, so it kept counting through an unload that reported the model
    gone. Reading it per call under the compute lock mirrors `_st_encode`."""
    looked_up = []

    class _Tok:
        def __init__(self, n):
            self.n = n

        def encode(
            self,
            text,
            add_special_tokens = False,
        ):
            return list(range(self.n))

    class _Model:
        def __init__(self, n):
            self.tokenizer = _Tok(n)

    models = [_Model(3), _Model(7)]

    def _get(model_name = None):
        looked_up.append(model_name)
        return models[min(len(looked_up) - 1, 1)]

    monkeypatch.setattr(embeddings, "_get", _get)
    count = embeddings._st_token_counter("org/embedder")

    assert looked_up == [], "the lookup must not happen before the first call"
    assert count("first") == 3
    assert count("second") == 7
    assert looked_up == ["org/embedder", "org/embedder"]


def test_a_local_gguf_selects_llama_even_on_a_gpu_box(monkeypatch, tmp_path):
    """_resolve_auto answers sentence-transformers whenever a GPU is present, so a
    GPU box pointed at a local .gguf planned an ST load of a GGUF file."""
    gguf = tmp_path / "embed.gguf"
    gguf.write_bytes(b"GGUF")
    monkeypatch.setattr(embeddings.config, "EMBED_BACKEND", "auto")
    monkeypatch.setattr(embeddings, "_forced_backends", {})
    # The hardware default that used to win.
    monkeypatch.setattr(embeddings, "_resolve_auto", lambda: "sentence-transformers")

    assert embeddings._resolve_auto_for_model(str(gguf)) == "llama-server"
    assert embeddings.resolved_backend_for_model(str(gguf)) == "llama-server"
    # A folder holding one counts the same way.
    assert embeddings._resolve_auto_for_model(str(tmp_path)) == "llama-server"
    # An ordinary repo id is untouched, and costs no filesystem walk.
    assert embeddings._resolve_auto_for_model("unsloth/bge-small-en-v1.5") == (
        "sentence-transformers"
    )


def test_a_local_gguf_beats_a_stored_sentence_transformers_record(monkeypatch, tmp_path):
    """A stored ST record for a .gguf can only come from a force-save that then
    failed; honouring it would reinstate the same broken load."""
    gguf = tmp_path / "embed.gguf"
    gguf.write_bytes(b"GGUF")
    monkeypatch.setattr(embeddings.config, "EMBED_BACKEND", "auto")
    monkeypatch.setattr(embeddings, "_resolve_auto", lambda: "sentence-transformers")
    import utils.embedding_model_settings as ems

    monkeypatch.setattr(ems, "get_stored_backend", lambda _m: "sentence-transformers")

    assert embeddings._resolve_auto_for_model(str(gguf)) == "llama-server"


def test_an_explicit_backend_is_not_overridden_by_a_local_gguf(monkeypatch, tmp_path):
    """Only ``auto`` consults the probe; an explicit setting stays verbatim."""
    gguf = tmp_path / "embed.gguf"
    gguf.write_bytes(b"GGUF")
    monkeypatch.setattr(embeddings.config, "EMBED_BACKEND", "sentence-transformers")
    monkeypatch.setattr(embeddings, "_forced_backends", {})
    monkeypatch.setattr(embeddings, "sentence_transformers_runtime_available", lambda: True)

    assert embeddings.resolved_backend_for_model(str(gguf)) == "sentence-transformers"


def test_the_st_probe_warms_the_pinned_model_not_the_live_setting(monkeypatch):
    """_get_backend resolves for the pinned model, but the probe warmed
    model_name=None, reading the live setting: a job pinned to A probed B once
    Settings moved, failing the valid A job before its first encode."""
    warmed = []

    class _Probe:
        def warm(self, model_name = None):
            warmed.append(model_name)

    monkeypatch.setattr(embeddings, "_SentenceTransformersBackend", _Probe)
    monkeypatch.setattr(embeddings.config, "EMBED_BACKEND", "sentence-transformers")
    monkeypatch.setattr(embeddings, "_forced_backends", {})
    embeddings._reset_backend()
    try:
        embeddings._get_backend("org/pinned")
    finally:
        embeddings._reset_backend()

    assert warmed == ["org/pinned"]


def test_the_security_gate_scans_the_snapshot_that_is_actually_loaded(monkeypatch, tmp_path):
    """On the repo id the gate scanned the Hub's current commit while the load
    opened an older cached one, so a pickle present only there passed.
    evaluate_file_security recovers repo and commit from a snapshot path."""
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    # A real ST checkpoint: the pin is ST-specific now, so a GGUF-only snapshot
    # is deliberately not adopted as the load target.
    (snapshot / "model.safetensors").write_bytes(b"ST")
    scanned = []
    monkeypatch.setattr(
        embeddings, "_guard_model_security", lambda target, local_only = False: scanned.append(target)
    )

    class _ST:
        def __init__(self, target, **_kwargs):
            self.loaded = target
            self.tokenizer = None

        def get_sentence_embedding_dimension(self):
            return 8

    monkeypatch.setattr(embeddings, "_load_device", lambda: "cpu")
    monkeypatch.setattr(embeddings, "_install_torchao_stub_once", lambda: None)
    monkeypatch.setattr(embeddings, "_st_accepts_local_files_only", lambda _c: False)
    import utils.utils as utils

    monkeypatch.setattr(utils, "cached_st_source", lambda m: ("org/embedder", snapshot))
    monkeypatch.setattr(utils, "hf_cache_snapshot_dir", lambda m: snapshot)
    import sys as _sys
    import types as _t

    st_mod = _t.ModuleType("sentence_transformers")
    st_mod.SentenceTransformer = _ST
    monkeypatch.setitem(_sys.modules, "sentence_transformers", st_mod)
    embeddings._model = None
    embeddings._name = None
    try:
        model = embeddings._get("org/embedder")
    finally:
        embeddings._model = None
        embeddings._name = None

    # The snapshot was loaded, and it is the same string the gate was handed.
    assert model.loaded == str(snapshot)
    assert scanned == [str(snapshot)]


def test_the_residency_probe_does_not_wait_on_a_model_load(monkeypatch):
    """Both locks are held across a whole model load, so a probe taking either
    made GET, PUT, reset and unload wait it out."""
    import threading

    embeddings._reset_backend()
    answered = threading.Event()
    result = {}

    # Both locks held, exactly as they are mid-construction.
    with embeddings._backend_lock, embeddings._lock:

        def _probe():
            result["any"] = embeddings.backend_is_loaded()
            result["named"] = embeddings.backend_is_loaded("org/embedder")
            answered.set()

        threading.Thread(target = _probe, daemon = True).start()
        # Answered while the construction locks are still held by this thread.
        assert answered.wait(timeout = 5), "the status probe blocked on the load locks"

    assert result == {"any": False, "named": False}


def test_a_dead_llama_process_is_not_reported_as_loaded(monkeypatch):
    """The object keeps _model_repo after the subprocess exits, so the repo match
    alone called a dead server resident and offered Unload for it."""
    alive = {"value": True}
    backend = SimpleNamespace(
        _model_repo = "org/resident-GGUF", _process_alive = lambda: alive["value"]
    )
    monkeypatch.setattr(embeddings, "_backend", backend)
    monkeypatch.setattr(embeddings, "_is_llama_backend", lambda value: value is backend)
    monkeypatch.setattr(
        embeddings.config,
        "effective_gguf_repo_for_embedding_model",
        lambda model: f"{model}-GGUF",
    )

    assert embeddings.backend_is_loaded("org/resident") is True
    assert embeddings.backend_is_loaded() is True

    alive["value"] = False
    assert embeddings.backend_is_loaded("org/resident") is False
    # And the unqualified question, which is what gates the Unload control.
    assert embeddings.backend_is_loaded() is False


def test_two_models_can_each_hold_their_own_llama_fallback_pin(monkeypatch):
    """One (key, model) pair meant a second failing model erased the first one's
    pin, so a job still running under A forgot it had swapped to llama-server and
    retried ST. If the original failure was transient that ingestion splits its
    own results across two vector spaces."""
    monkeypatch.setattr(embeddings.config, "EMBED_BACKEND", "auto")
    monkeypatch.setattr(embeddings, "_forced_backends", {})
    monkeypatch.setattr(
        embeddings, "_resolve_auto_for_model", lambda model = None: "sentence-transformers"
    )

    embeddings._forced_backends["org/a"] = "llama-server"
    embeddings._forced_backends["org/b"] = "llama-server"

    assert embeddings.resolved_backend_for_model("org/a") == "llama-server"
    assert embeddings.resolved_backend_for_model("org/b") == "llama-server"
    assert embeddings.sentence_transformers_fallback_allowed("org/a") is False
    assert embeddings.sentence_transformers_fallback_allowed("org/b") is False
    # A model that never failed is untouched by either pin.
    assert embeddings.resolved_backend_for_model("org/c") == "sentence-transformers"
    assert embeddings.sentence_transformers_fallback_allowed("org/c") is True


def test_the_forced_backend_probe_does_not_wait_on_a_model_load(monkeypatch):
    """_get_backend holds _backend_lock across a whole model load, so a resolver
    probe taking it made GET/PUT on this setting hang for the download rather than
    for the 20s Hub budget that is supposed to bound them."""
    import threading

    monkeypatch.setattr(embeddings.config, "EMBED_BACKEND", "auto")
    monkeypatch.setattr(embeddings, "_forced_backends", {"org/pinned": "llama-server"})
    answered = threading.Event()
    result = {}

    with embeddings._backend_lock:

        def _probe():
            result["resolved"] = embeddings.resolved_backend_for_model("org/pinned")
            result["fallback"] = embeddings.sentence_transformers_fallback_allowed("org/pinned")
            answered.set()

        threading.Thread(target = _probe, daemon = True).start()
        assert answered.wait(timeout = 5), "the resolver probe blocked on the construction lock"

    assert result == {"resolved": "llama-server", "fallback": False}


def test_a_model_reloaded_behind_an_unload_is_still_reported_and_freed(monkeypatch):
    """An ST wrapper descheduled between _get_backend() returning it and its encode
    starting is retired by an unload landing in the gap, and then reloads the
    module-level model with no backend to publish. Answering "nothing is loaded"
    stranded those weights for the life of the process: the next unload saw no
    backend and freed nothing."""
    monkeypatch.setattr(embeddings, "_backend", None, raising = False)
    monkeypatch.setattr(embeddings, "_backend_key", None, raising = False)
    monkeypatch.setattr(embeddings, "_model", object(), raising = False)
    monkeypatch.setattr(embeddings, "_name", "org/embedder", raising = False)

    assert embeddings.backend_is_loaded() is True
    assert embeddings.backend_is_loaded("org/embedder") is True
    assert embeddings.backend_is_loaded("org/other") is False

    assert embeddings.release_backend() is True
    assert embeddings._model is None
    assert embeddings.backend_is_loaded() is False
    # And with nothing resident it stays a no-op.
    assert embeddings.release_backend() is False
