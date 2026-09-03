# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for the Load-Model memory estimate (POST /api/inference/estimate-memory).

Guards the pieces the estimate is assembled from and the properties that make it safe
to put a number in front of a user:

* ``_gguf_runtime_bytes`` -- KV + compute, itemized. The load-bearing case is a header
  without the attention dims: ``kv_estimable = False`` with ``kv_bytes == 0``, since a
  UI reading that zero as "no cache" is worse than showing nothing.
* ``_estimate_gguf_kv_gb`` -- a thin wrapper over it now, and still the training
  guard's input, so its value must stay the old sum exactly.
* ``_gguf_offloaded_layer_fraction`` -- Auto is deliberately 1.0, not a guess.
* ``_gguf_resident_file_gb`` / ``_gguf_memory_breakdown`` -- weights come from
  subtracting the context term out of ``_estimate_gguf_required_gb``; the observable
  that the arms are paired is that weights do not move with the context slider.
* ``_localized_estimate_config`` -- without it a cached repo priced itself through a
  ``paths-info`` call. Both halves pinned: the copy takes the local arm, the cached
  original is not mutated.
* ``_estimate_token_fingerprint`` -- both TTL caches are keyed per token.
* the route -- the "cannot size this" answers, and an Ollama manifest ref refused
  before anything is materialized.

No GPU, no network, no model load: every GGUF here is a synthetic header on tmp_path.
"""

import inspect
import json
import os
import sys
import types as _types
from pathlib import Path

import pytest

# Stub heavy / unavailable deps before importing the module under test.
# Copied verbatim from tests/test_kv_cache_estimation.py -- same reasons.

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# loggers
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

# structlog. Carries get_logger because this stub is process-wide: whichever test
# module is imported first wins the setdefault, and utils/prebuilt/freshness_flow
# calls structlog.get_logger at import time. A bare module here fails that import
# for every later module on a runner without the real package.
_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

# httpx -- only stub when the real library is missing. Unconditional stubbing
# shadows HTTPError/Response that huggingface_hub.errors imports at load time,
# silently breaking the transformers introspection tier.
try:
    import httpx as _httpx_real  # noqa: F401
except ImportError:
    _httpx_stub = _types.ModuleType("httpx")
    for _exc_name in (
        "ConnectError",
        "TimeoutException",
        "ReadTimeout",
        "ReadError",
        "RemoteProtocolError",
        "CloseError",
        "HTTPError",
        "RequestError",
    ):
        setattr(_httpx_stub, _exc_name, type(_exc_name, (Exception,), {}))

    class _FakeTimeout:
        def __init__(self, *a, **kw):
            pass

    _httpx_stub.Timeout = _FakeTimeout
    _httpx_stub.Response = type("Response", (), {})
    _httpx_stub.Client = type(
        "Client",
        (),
        {
            "__init__": lambda self, **kw: None,
            "__enter__": lambda self: self,
            "__exit__": lambda self, *a: None,
        },
    )
    sys.modules["httpx"] = _httpx_stub

import asyncio  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import routes.inference as ri  # noqa: E402
from models.inference import EstimateMemoryRequest  # noqa: E402
from utils.models.model_config import ModelConfig  # noqa: E402

# Reuse the GGUF blob builder rather than copying it: these tests are only worth
# anything if the bytes they parse are the bytes the real parser was written for,
# and a second copy of the writer would drift from the first. Loaded by path
# because `tests` is not importable as a package name from every runner layout.
import importlib.util as _ilu  # noqa: E402

_kv_spec = _ilu.spec_from_file_location(
    "_kv_cache_estimation_for_memory_estimate",
    Path(__file__).resolve().parent / "test_kv_cache_estimation.py",
)
_kv_mod = _ilu.module_from_spec(_kv_spec)
_kv_spec.loader.exec_module(_kv_mod)
_make_gguf_bytes = _kv_mod._make_gguf_bytes

_GIB = 1024**3


@pytest.fixture(autouse = True)
def _clear_estimate_caches():
    """Both module caches are TTL'd, not per-request, so they leak across tests.

    ``_estimate_files_cache`` is keyed on the config identity and NOT on the
    context, which is exactly the behaviour under test -- so a stale entry from
    an earlier test would silently satisfy an assertion here.
    """
    ri._estimate_files_cache.clear()
    ri._estimate_config_cache.clear()
    yield
    ri._estimate_files_cache.clear()
    ri._estimate_config_cache.clear()


# Fixtures: synthetic GGUF headers


# Pure-GQA geometry with no sliding window, so the estimator takes the standard
# Path-4 branch and -- more importantly -- the dynamic SWA resolver never fires,
# which is the only thing in this code path that could reach Hugging Face.
_GQA_FIELDS = {
    "context_length": 8192,
    "block_count": 12,
    "attention.head_count": 8,
    "attention.head_count_kv": 4,
    "attention.key_length": 64,
    "attention.value_length": 64,
    "embedding_length": 512,
}


def _write_gguf(
    tmp_path: Path,
    arch: str,
    fields: dict,
    name: str = "model.gguf",
) -> str:
    kv = {"general.architecture": arch}
    kv.update({f"{arch}.{k}": v for k, v in fields.items()})
    path = tmp_path / name
    path.write_bytes(_make_gguf_bytes(arch, kv))
    return str(path)


@pytest.fixture
def gqa_gguf(tmp_path) -> str:
    return _write_gguf(tmp_path, "qwen3", _GQA_FIELDS)


@pytest.fixture
def dimless_gguf(tmp_path) -> str:
    """A header carrying a layer count and nothing the KV formula can use.

    Real GGUFs like this exist (truncated / minimal metadata), and they are the
    whole reason ``kv_estimable`` is a separate field.
    """
    return _write_gguf(tmp_path, "qwen3", {"block_count": 12}, name = "dimless.gguf")


# A. _gguf_offloaded_layer_fraction


class TestOffloadedLayerFraction:
    """Share of the weights the requested offload keeps on the GPU."""

    def test_auto_is_full(self):
        # Auto asks llama.cpp to fit everything and only spills when it cannot,
        # so 1.0 is the cost of the load succeeding -- not an approximation.
        assert ri._gguf_offloaded_layer_fraction("auto", None, 12) == 1.0
        assert ri._gguf_offloaded_layer_fraction(None, 4, 12) == 1.0

    @pytest.mark.parametrize("gpu_layers", [None, -1])
    def test_manual_without_a_layer_count_is_full(self, gpu_layers):
        # "manual" with no usable -ngl is the slider's own unset state; it must
        # read like Auto rather than dividing by a number nobody chose.
        assert ri._gguf_offloaded_layer_fraction("manual", gpu_layers, 12) == 1.0

    @pytest.mark.parametrize("layer_count", [None, 0])
    def test_unknown_layer_count_is_full(self, layer_count):
        # Remote / unreadable header: there is no denominator, and inventing one
        # would put a fabricated split in front of the user.
        assert ri._gguf_offloaded_layer_fraction("manual", 20, layer_count) == 1.0

    def test_manual_split_uses_block_count_plus_one(self):
        # +1 is the output layer, matching the ceiling the UI slider uses; a bare
        # gpu_layers/layer_count would read 100% one layer early.
        assert ri._gguf_offloaded_layer_fraction("manual", 20, 65) == pytest.approx(20 / 66)

    def test_over_large_gpu_layers_clamps(self):
        # -ngl 999 is the idiomatic "all of it"; it must not exceed the weights.
        assert ri._gguf_offloaded_layer_fraction("manual", 999, 65) == 1.0

    def test_zero_layers_is_cpu_only(self):
        assert ri._gguf_offloaded_layer_fraction("manual", 0, 65) == 0.0

    def test_a_layer_count_in_the_extras_wins_in_either_mode(self):
        # Auto emits -ngl -1 and the extras are appended after it, so a user count
        # last-wins at the child; the loader reads it back with the same parser
        # (llama_cpp.py:6841, the non-manual arm). Treating Auto as always fully
        # resident put a GPU-exceeds warning on a load running on the CPU.
        assert ri._gguf_offloaded_layer_fraction("auto", None, 12, ["-ngl", "0"]) == 0.0
        assert ri._gguf_offloaded_layer_fraction(
            "auto", None, 12, ["--gpu-layers", "6"]
        ) == pytest.approx(6 / 13)
        # -1 is "all layers", and above the count clamps rather than exceeding.
        assert ri._gguf_offloaded_layer_fraction("auto", None, 12, ["-ngl", "-1"]) == 1.0
        assert ri._gguf_offloaded_layer_fraction("auto", None, 12, ["-ngl", "999"]) == 1.0
        # Malformed is llama-server's to reject; it names it better than a guess here.
        assert ri._gguf_offloaded_layer_fraction("auto", None, 12, ["-ngl", "abc"]) == 1.0
        # Manual does strip the flag, but only after translating its last-wins value
        # into the load field (routes/inference.py, the manual branch of /load and of
        # the validate path), so the extras count is what the child runs there too.
        assert ri._gguf_offloaded_layer_fraction("manual", 0, 12, ["-ngl", "999"]) == 1.0
        assert ri._gguf_offloaded_layer_fraction("manual", 12, 12, ["-ngl", "0"]) == 0.0
        # And with nothing in the extras the field still owns it.
        assert ri._gguf_offloaded_layer_fraction("manual", 0, 12, ["--top-k", "40"]) == 0.0


# B. _gguf_runtime_bytes against real GGUF headers


class TestGgufRuntimeBytes:
    """The context-dependent half, priced off a real (synthetic) header."""

    def test_kv_grows_with_context(self, gqa_gguf):
        sizes = [ri._gguf_runtime_bytes(gqa_gguf, ctx).kv_bytes for ctx in (1024, 4096, 16384)]
        assert all(a < b for a, b in zip(sizes, sizes[1:])), sizes
        assert sizes[0] > 0

    def test_quantized_cache_is_smaller_than_f16(self, gqa_gguf):
        # 4x on the cache dtype alone at the same context is the single biggest
        # lever the panel exposes, so the estimate has to actually follow it.
        f16 = ri._gguf_runtime_bytes(gqa_gguf, 8192, cache_type_kv = "f16")
        q4 = ri._gguf_runtime_bytes(gqa_gguf, 8192, cache_type_kv = "q4_0")
        assert q4.kv_bytes < f16.kv_bytes
        assert q4.n_ctx == f16.n_ctx == 8192

    def test_reports_what_it_priced(self, gqa_gguf):
        # These fields exist so a reader can judge the estimate instead of
        # trusting it; they must describe the arithmetic that actually ran.
        runtime = ri._gguf_runtime_bytes(gqa_gguf, 4096, cache_type_kv = "q8_0", n_parallel = 1)
        assert runtime.kv_estimable is True
        assert runtime.n_ctx == 4096
        assert runtime.cache_type_kv == "q8_0"
        assert runtime.n_parallel == 1
        assert runtime.layer_count == _GQA_FIELDS["block_count"]

    def test_zero_context_prices_the_native_one(self, gqa_gguf):
        # n_ctx = 0 is the panel's "Auto"; it prices the header's context_length,
        # which is what an Auto load would ask llama-server for.
        runtime = ri._gguf_runtime_bytes(gqa_gguf, 0)
        assert runtime.n_ctx == _GQA_FIELDS["context_length"]
        assert runtime.kv_bytes > 0

    def test_missing_dims_are_unknown_not_zero(self, dimless_gguf):
        # THE case this NamedTuple exists for. kv_bytes == 0 here means "could not
        # size", and the only thing separating it from a genuine zero is the flag.
        runtime = ri._gguf_runtime_bytes(dimless_gguf, 32768)
        assert runtime.kv_estimable is False
        assert runtime.kv_bytes == 0
        assert runtime.compute_bytes == 0
        # It also stops describing a priced load, since none was priced.
        assert runtime.n_ctx == 0
        # block_count is a separate key and survives: a caller sizing a manual offload
        # split needs it, and without it _gguf_offloaded_layer_fraction answers 1.0 and
        # calls --gpu-layers 0 a fully GPU-resident load.
        assert runtime.layer_count == 12
        assert ri._gguf_offloaded_layer_fraction("manual", 0, runtime.layer_count) == 0.0

    def test_a_recurrent_model_keeps_its_layer_count(self, tmp_path):
        """The unsizable-KV path is reached by whole model families, not just stubs.

        llama.cpp reads the attention head counts with ``required = false`` while
        block_count and embedding_length are required, so every pure SSM model --
        Mamba, Mamba2, RWKV -- loads with a layer count and no attention dims, which is
        what ``_can_estimate_kv`` rejects. Dropping the count there reported a manual
        --gpu-layers 0 as fully GPU-resident on all of them.
        """
        mamba = _write_gguf(
            tmp_path,
            "mamba",
            {
                "block_count": 24,
                "embedding_length": 2048,
                "context_length": 8192,
                "ssm.conv_kernel": 4,
                "ssm.inner_size": 4096,
                "ssm.state_size": 16,
                "ssm.time_step_rank": 128,
            },
            name = "mamba.gguf",
        )
        runtime = ri._gguf_runtime_bytes(mamba, 32768)
        assert runtime.kv_estimable is False
        assert runtime.layer_count == 24
        assert ri._gguf_offloaded_layer_fraction("manual", 0, runtime.layer_count) == 0.0

    def test_unreadable_file_is_unknown(self, tmp_path):
        # Not a GGUF at all: the header walk raises and the caller must still get
        # a well-formed "unknown", never a partial number.
        junk = tmp_path / "not-a-gguf.gguf"
        junk.write_bytes(b"not a gguf header")
        assert ri._gguf_runtime_bytes(str(junk), 4096) == ri._GGUF_RUNTIME_UNKNOWN


# C. Training-guard compatibility


class TestKvGbWrapperCompatibility:
    """``_estimate_gguf_kv_gb`` is the training admission guard's entry point.

    It is now a wrapper, so the refactor is only correct if it still returns
    exactly the old scalar: (KV + compute) in GB. Anything else silently moves
    the threshold at which chat is refused during a training run.
    """

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(),
            dict(cache_type_kv = "q8_0"),
            dict(n_parallel = 4),
            dict(n_batch = 2048, n_ubatch = 512),
            dict(ctx_checkpoints = 4),
            dict(n_devices = 2),
            dict(is_diffusion = True),
            dict(llama_extra_args = ["-c", "16384"]),
        ],
    )
    def test_wrapper_equals_summed_tuple(self, gqa_gguf, kwargs):
        runtime = ri._gguf_runtime_bytes(gqa_gguf, 4096, **kwargs)
        assert ri._estimate_gguf_kv_gb(gqa_gguf, 4096, **kwargs) == pytest.approx(
            (runtime.kv_bytes + runtime.compute_bytes) / _GIB
        )

    def test_unsizable_header_is_zero_gb(self, dimless_gguf):
        # Unchanged legacy behaviour: a cache the guard cannot size must not
        # become a refusal on its own, so it contributes nothing.
        assert ri._estimate_gguf_kv_gb(dimless_gguf, 32768) == 0.0

    def test_the_guard_keeps_the_larger_context_the_panel_does_not(self, gqa_gguf):
        """A smaller ``-c`` in the extras is what the launch runs, not what the
        guard reserves against. The guard deliberately takes the maximum, since a
        request that later drops the flag must not have been admitted against the
        smaller cache; the panel is quoting a number to a user and takes the launch
        value (``resolve_requested_ctx``, load_model's own resolver).
        """
        smaller = ["-c", "8192"]
        guard = ri._gguf_runtime_bytes(gqa_gguf, 131072, smaller)
        panel = ri._gguf_runtime_bytes(gqa_gguf, 131072, smaller, ctx_last_wins = True)
        assert guard.n_ctx == 131072
        assert panel.n_ctx == 8192
        assert panel.kv_bytes < guard.kv_bytes
        # The wrapper the guard calls must still be the maximum arm, unchanged.
        assert ri._estimate_gguf_kv_gb(gqa_gguf, 131072, llama_extra_args = smaller) == pytest.approx(
            (guard.kv_bytes + guard.compute_bytes) / _GIB
        )
        # A larger override is the same either way, and -c 0 means the model's own
        # trained context in llama.cpp rather than the panel's number.
        larger = ["-c", "262144"]
        assert (
            ri._gguf_runtime_bytes(gqa_gguf, 131072, larger, ctx_last_wins = True).n_ctx
            == ri._gguf_runtime_bytes(gqa_gguf, 131072, larger).n_ctx
            == 262144
        )
        assert (
            ri._gguf_runtime_bytes(gqa_gguf, 131072, ["-c", "0"], ctx_last_wins = True).n_ctx
            == _GQA_FIELDS["context_length"]
        )


# D. _gguf_resident_file_gb and _gguf_memory_breakdown


class TestResidentFileGb:
    """The weights term, obtained by subtracting the context term back out."""

    def test_subtracts_the_local_arm_context_term(self, gqa_gguf, monkeypatch):
        # Prove the pairing rather than the plumbing: a config with a local
        # gguf_file takes the _estimate_gguf_kv_gb arm, so feeding
        # _estimate_gguf_required_gb a known "files + that term" must leave
        # exactly the files behind.
        ctx_term = ri._estimate_gguf_kv_gb(gqa_gguf, 0, None)
        assert ctx_term > 0, "fixture must have a non-trivial context term to subtract"
        config = SimpleNamespace(identifier = "local/model", gguf_file = gqa_gguf, is_gguf = True)
        monkeypatch.setattr(ri, "_estimate_gguf_required_gb", lambda cfg, **kw: 3.0 + ctx_term)
        assert ri._gguf_resident_file_gb(config) == pytest.approx(3.0)

    def test_unresolvable_size_is_not_cached(self, gqa_gguf, monkeypatch):
        # A None is usually a download still in flight, so the next slider tick
        # is precisely when the answer changes; caching it would freeze the panel
        # on "unsizable" for the whole TTL.
        config = SimpleNamespace(identifier = "local/model", gguf_file = gqa_gguf, is_gguf = True)
        monkeypatch.setattr(ri, "_estimate_gguf_required_gb", lambda cfg, **kw: None)
        assert ri._gguf_resident_file_gb(config) is None
        assert ri._estimate_files_cache == {}


class TestMemoryBreakdown:
    """Composition of files + runtime into the panel's itemization."""

    WEIGHTS_GB = 4.0
    # Of that files total, the main weight and everything beside it. Pinned because
    # the synthetic header on tmp_path is a few hundred bytes, which would leave the
    # whole 4 GB looking like companions and make the offload split meaningless.
    MAIN_BYTES = 3 * 1024**3
    COMPANION_BYTES = int(WEIGHTS_GB * 1024**3) - MAIN_BYTES

    @pytest.fixture
    def config(self, gqa_gguf):
        return SimpleNamespace(
            identifier = "local/model",
            gguf_file = gqa_gguf,
            is_gguf = True,
            gguf_mmproj_file = None,
        )

    @pytest.fixture(autouse = True)
    def _fixed_files(self, monkeypatch):
        # Pin the files term so every assertion below is about the composition,
        # not about repository listing (which would want the network).
        monkeypatch.setattr(ri, "_gguf_resident_file_gb", lambda cfg, **kw: self.WEIGHTS_GB)
        real_size = ri.LlamaCppBackend._get_gguf_size_bytes

        def _size(path):
            name = os.path.basename(str(path))
            return self.MAIN_BYTES if name == "model.gguf" else real_size(path)

        monkeypatch.setattr(ri.LlamaCppBackend, "_get_gguf_size_bytes", staticmethod(_size))

    def test_weights_do_not_move_with_context(self, config, gqa_gguf):
        # THE invariant the subtraction approach exists to keep. If the context
        # term ever leaked into the files term, this is where it shows up -- and
        # the runtime terms must still be moving, or the test proves nothing.
        small = ri._gguf_memory_breakdown(config, gqa_gguf, n_ctx = 2048)
        large = ri._gguf_memory_breakdown(config, gqa_gguf, n_ctx = 32768)
        assert small.weights_bytes == large.weights_bytes
        assert large.kv_bytes > small.kv_bytes
        assert large.total_bytes > small.total_bytes

    def test_total_is_the_sum_of_its_parts(self, config, gqa_gguf):
        b = ri._gguf_memory_breakdown(config, gqa_gguf, n_ctx = 8192)
        assert b.weights_bytes == int(round(self.WEIGHTS_GB * _GIB))
        assert b.total_bytes == b.weights_bytes + b.kv_bytes + b.compute_bytes

    def test_auto_placement_puts_everything_on_the_gpu(self, config, gqa_gguf):
        # Auto's estimate is the cost of the load succeeding, so nothing is
        # attributed to host RAM and gpu_layers is reported as unset.
        b = ri._gguf_memory_breakdown(config, gqa_gguf, n_ctx = 8192, gpu_memory_mode = "auto")
        assert b.gpu_bytes == b.total_bytes
        assert b.kv_on_gpu is True
        assert b.gpu_layers is None

    def test_an_auto_load_with_an_extras_ngl_is_not_fully_resident(self, config, gqa_gguf):
        # The one thing that overrides Auto: the child takes the last -ngl, so
        # -ngl 0 runs on the CPU however the panel's own mode reads.
        full = ri._gguf_memory_breakdown(config, gqa_gguf, n_ctx = 8192, gpu_memory_mode = "auto")
        on_cpu = ri._gguf_memory_breakdown(
            config,
            gqa_gguf,
            n_ctx = 8192,
            gpu_memory_mode = "auto",
            llama_extra_args = ["-ngl", "0"],
        )
        assert on_cpu.total_bytes == full.total_bytes
        assert on_cpu.kv_on_gpu is False
        # The main weight, its cache and its compute all leave the GPU; whatever is
        # placed by its own flag (a projector here would be) does not.
        assert on_cpu.gpu_bytes == full.gpu_bytes - (
            self.MAIN_BYTES + full.kv_bytes + full.compute_bytes
        )

    def test_a_smaller_ctx_override_is_the_context_priced(self, config, gqa_gguf):
        # The panel reports what the launch runs, so a -c below the Context Length
        # control prices the smaller cache and says so in n_ctx.
        b = ri._gguf_memory_breakdown(
            config, gqa_gguf, n_ctx = 131072, llama_extra_args = ["-c", "8192"]
        )
        bigger = ri._gguf_memory_breakdown(config, gqa_gguf, n_ctx = 131072)
        assert b.n_ctx == 8192
        assert b.kv_bytes < bigger.kv_bytes

    def test_a_manual_layer_count_in_the_extras_is_priced(self, config, gqa_gguf):
        # /load translates the last -ngl into the manual field before stripping the
        # flag, so the field alone was the wrong thing to price.
        manual = dict(gpu_memory_mode = "manual", gpu_layers = 0, n_ctx = 8192)
        field_only = ri._gguf_memory_breakdown(config, gqa_gguf, **manual)
        overridden = ri._gguf_memory_breakdown(
            config, gqa_gguf, llama_extra_args = ["-ngl", "999"], **manual
        )
        assert field_only.gpu_bytes < overridden.gpu_bytes
        assert overridden.gpu_bytes == overridden.total_bytes

    def test_an_extras_split_mode_decides_the_priced_mode(self, config, gqa_gguf):
        # A --split-mode in the extras last-wins over the toggle at launch, and the
        # compute buffers differ by mode, so pricing the toggle priced a launch that
        # does not happen. Two devices, since one device splits nothing.
        def priced(extras, toggle):
            return ri._gguf_memory_breakdown(
                config,
                gqa_gguf,
                n_ctx = 32768,
                llama_extra_args = extras,
                tensor_parallel = toggle,
                n_devices = 2,
            ).compute_bytes

        layer, tensor = priced(None, False), priced(None, True)
        assert layer != tensor, "fixture must separate the two modes for this to prove anything"
        assert priced(["--split-mode", "tensor"], False) == tensor
        # And the reverse: the extras can turn it off as well as on.
        assert priced(["--split-mode", "layer"], True) == layer

    def test_manual_split_divides_the_cache_with_the_weights(self, config, gqa_gguf):
        # A cache buffer is allocated on model.dev_layer(il) whenever offload_kqv is
        # on (llama-kv-cache.cpp), so a layer left on the CPU keeps its cache in host
        # RAM: --no-kv-offload moves ALL of it, and a partial --gpu-layers moves the
        # rest with the layers. Charging the whole cache to VRAM here contradicted the
        # weights term beside it, and at a long context the cache is the larger of the
        # two, so a small Manual offload read as exceeding a card it fits on.
        b = ri._gguf_memory_breakdown(
            config, gqa_gguf, n_ctx = 8192, gpu_memory_mode = "manual", gpu_layers = 6
        )
        fraction = 6 / float(_GQA_FIELDS["block_count"] + 1)
        assert b.layer_count == _GQA_FIELDS["block_count"]
        assert b.gpu_layers == 6
        assert b.gpu_bytes == (
            int(self.MAIN_BYTES * fraction)
            + self.COMPANION_BYTES
            + int(b.kv_bytes * fraction)
            + b.compute_bytes
        )
        # The cache is still counted in full in the aggregate figure: the host holds
        # the other share, it does not vanish.
        assert b.gpu_bytes < b.total_bytes
        assert b.kv_bytes > int(b.kv_bytes * fraction)

    def test_a_partial_offload_splits_only_the_main_weight(self, config, gqa_gguf):
        # --gpu-layers splits the model, not the companions beside it: a projector
        # and a drafter are placed by their own flags. Scaling the whole files term
        # let --gpu-layers 0 report an empty GPU while both sat in VRAM.
        zero = ri._gguf_memory_breakdown(
            config, gqa_gguf, n_ctx = 8192, gpu_memory_mode = "manual", gpu_layers = 0
        )
        assert zero.gpu_bytes == self.COMPANION_BYTES
        # KV and compute do follow the layers, so nothing else survives ngl 0.
        assert zero.kv_on_gpu is False
        full = ri._gguf_memory_breakdown(
            config,
            gqa_gguf,
            n_ctx = 8192,
            gpu_memory_mode = "manual",
            gpu_layers = _GQA_FIELDS["block_count"] + 1,
        )
        assert full.gpu_bytes == full.total_bytes

    def test_no_mmproj_offload_moves_the_projector_and_only_it(self, config, gqa_gguf, tmp_path):
        # The drafter is the other companion, and it is only ever charged when it
        # lands on the GPU, so this flag must not sweep it off with the projector.
        projector = tmp_path / "mmproj-model.gguf"
        projector.write_bytes(b"\0" * 1024)
        config.gguf_mmproj_file = str(projector)
        expected = self.COMPANION_BYTES - projector.stat().st_size
        pinned = ri._gguf_memory_breakdown(
            config,
            gqa_gguf,
            n_ctx = 8192,
            gpu_memory_mode = "manual",
            gpu_layers = 0,
            llama_extra_args = ["--no-mmproj-offload"],
        )
        # Exactly the projector leaves; the rest of the companion bytes stay.
        assert pinned.gpu_bytes == expected

    def test_no_kv_offload_moves_the_cache_off_the_gpu(self, config, gqa_gguf):
        # At a long context the cache is most of the footprint, so ignoring
        # -nkvo would report VRAM pressure the load does not create.
        b = ri._gguf_memory_breakdown(
            config, gqa_gguf, n_ctx = 8192, llama_extra_args = ["--no-kv-offload"]
        )
        assert b.kv_on_gpu is False
        assert b.kv_bytes > 0  # still counted in the total, just not in VRAM
        assert b.gpu_bytes == b.weights_bytes + b.compute_bytes
        assert b.total_bytes == b.weights_bytes + b.kv_bytes + b.compute_bytes

    def test_unsizable_files_produce_no_breakdown(self, config, gqa_gguf, monkeypatch):
        # None here means the caller default-denies with "unsizable" rather than
        # rendering a total whose largest term is missing.
        monkeypatch.setattr(ri, "_gguf_resident_file_gb", lambda cfg, **kw: None)
        assert ri._gguf_memory_breakdown(config, gqa_gguf, n_ctx = 8192) is None


# E. _localized_estimate_config


def _repo_config(**overrides) -> ModelConfig:
    """A config shaped the way ``from_identifier`` leaves a REPOSITORY GGUF.

    ``gguf_file`` is None even when the weights are already in the HF cache --
    that field is filled in by the download path, not by resolution -- which is
    precisely the state that used to send the estimate to the network.
    """
    fields = dict(
        identifier = "org/model-GGUF",
        display_name = "model (Q4_K_M)",
        path = "org/model-GGUF",
        is_local = False,
        is_cached = False,
        is_vision = False,
        is_lora = False,
        is_gguf = True,
        gguf_file = None,
        gguf_hf_repo = "org/model-GGUF",
        gguf_variant = "Q4_K_M",
    )
    fields.update(overrides)
    return ModelConfig(**fields)


def _local_config(gguf_path: str, **overrides) -> ModelConfig:
    """A config that already names a real file: the local-folder / native-pick shape."""
    fields = dict(
        identifier = str(Path(gguf_path).parent),
        display_name = Path(gguf_path).name,
        path = str(Path(gguf_path).parent),
        is_local = True,
        is_cached = True,
        is_vision = False,
        is_lora = False,
        is_gguf = True,
        gguf_file = gguf_path,
    )
    fields.update(overrides)
    return ModelConfig(**fields)


class TestLocalizedEstimateConfig:
    """Point a repository-shaped config at the weights already on this disk."""

    def test_config_naming_a_real_file_is_returned_unchanged(self, gqa_gguf):
        # Identity, not equality: a config that already names a file has been
        # through the local-folder or native-pick path, and re-running companion
        # detection on it would scan outside a native grant's directory boundary.
        config = _local_config(gqa_gguf)
        assert ri._localized_estimate_config(config, gqa_gguf) is config

    def test_repo_config_is_copied_never_mutated(self, gqa_gguf):
        # The original is sitting in _estimate_config_cache for the TTL, shared by
        # every later tick of the slider. A half-localized config escaping into
        # that cache is the genuinely nasty version of this bug.
        config = _repo_config()
        localized = ri._localized_estimate_config(config, gqa_gguf)
        assert localized is not config
        assert localized.gguf_file == gqa_gguf
        assert config.gguf_file is None
        # Everything else carries over, so the files arithmetic still knows which
        # repository and variant it is pricing.
        assert localized.identifier == config.identifier
        assert localized.gguf_hf_repo == config.gguf_hf_repo
        assert localized.gguf_variant == config.gguf_variant

    def test_real_sibling_projector_is_detected(self, tmp_path):
        # Real files, real detection: this is the test that proves search_root is
        # wired to the weight's own directory rather than to the repo id.
        weight = _write_gguf(tmp_path, "qwen3", _GQA_FIELDS, name = "model-Q4_K_M.gguf")
        projector = _write_gguf(tmp_path, "clip", {"block_count": 2}, name = "mmproj-model.gguf")
        localized = ri._localized_estimate_config(_repo_config(is_vision = True), weight)
        assert localized.gguf_mmproj_file == projector
        # No mtp-/dspark-/dflash- siblings exist, so those stay unset rather than
        # latching onto the weight or the projector.
        assert localized.gguf_mtp_file is None
        assert localized.gguf_dspark_file is None
        assert localized.gguf_dflash_file is None

    def test_non_vision_config_skips_the_projector(self, tmp_path):
        # A projector sitting in the directory must not be charged to a config the
        # loader will not open one for; that is bytes billed to nobody's load.
        weight = _write_gguf(tmp_path, "qwen3", _GQA_FIELDS, name = "model-Q4_K_M.gguf")
        _write_gguf(tmp_path, "clip", {"block_count": 2}, name = "mmproj-model.gguf")
        localized = ri._localized_estimate_config(_repo_config(is_vision = False), weight)
        assert localized.gguf_mmproj_file is None

    def test_drafter_detectors_receive_the_weight_and_its_directory(self, gqa_gguf, monkeypatch):
        # The three sidecar kinds, stubbed: real dspark/dflash detection wants a
        # family-name match against the target, which is a lot of fixture for what
        # is being checked here -- that each detector is called with the resolved
        # weight and the directory holding it.
        from utils.models import model_config as mc

        seen = {}

        def _stub(name, result):
            def _detect(
                path,
                search_root = None,
                *a,
                **kw,
            ):
                seen[name] = (path, search_root)
                return result

            return _detect

        monkeypatch.setattr(mc, "detect_mtp_file", _stub("mtp", "/tmp/mtp.gguf"))
        monkeypatch.setattr(mc, "detect_dspark_file", _stub("dspark", "/tmp/dspark.gguf"))
        monkeypatch.setattr(mc, "detect_dflash_file", _stub("dflash", "/tmp/dflash.gguf"))

        localized = ri._localized_estimate_config(_repo_config(), gqa_gguf)
        assert localized.gguf_mtp_file == "/tmp/mtp.gguf"
        assert localized.gguf_dspark_file == "/tmp/dspark.gguf"
        assert localized.gguf_dflash_file == "/tmp/dflash.gguf"
        expected = (gqa_gguf, str(Path(gqa_gguf).parent))
        assert seen == {"mtp": expected, "dspark": expected, "dflash": expected}

    def test_a_raising_detector_still_yields_a_usable_config(self, gqa_gguf, monkeypatch):
        # A companion scan that fails leaves the sidecar uncounted, which is a much
        # smaller error than the endpoint returning no estimate at all.
        from utils.models import model_config as mc

        monkeypatch.setattr(mc, "detect_mtp_file", lambda *a, **kw: "/tmp/mtp.gguf")

        def _boom(*a, **kw):
            raise OSError("permission denied walking the snapshot")

        monkeypatch.setattr(mc, "detect_dspark_file", _boom)

        localized = ri._localized_estimate_config(_repo_config(), gqa_gguf)
        assert localized.gguf_file == gqa_gguf  # the main weight is still priced
        assert localized.gguf_mtp_file == "/tmp/mtp.gguf"  # detected before the raise
        assert localized.gguf_dspark_file is None

    def test_localized_config_takes_the_local_files_arm(self, gqa_gguf, monkeypatch):
        # THE regression. Pre-fix, a repo-shaped config sent _gguf_resident_file_gb
        # down the repository arm, which prices from a paths-info listing and
        # subtracts _remote_gguf_compute_reserve_gb. Making that reserve raise turns
        # "took the remote arm" from a silent numeric difference into a failure.
        def _remote_arm_taken(*a, **kw):
            raise AssertionError("a localized config must not be priced as a repository")

        monkeypatch.setattr(ri, "_remote_gguf_compute_reserve_gb", _remote_arm_taken)
        ctx_term = ri._estimate_gguf_kv_gb(gqa_gguf, 0, None)
        monkeypatch.setattr(ri, "_estimate_gguf_required_gb", lambda cfg, **kw: 5.0 + ctx_term)

        raw = _repo_config()
        localized = ri._localized_estimate_config(raw, gqa_gguf)
        assert ri._gguf_resident_file_gb(localized) == pytest.approx(5.0)

        # And the un-localized original still goes the other way, so the assertion
        # above is about the localization rather than about the patched arm.
        ri._estimate_files_cache.clear()
        with pytest.raises(AssertionError, match = "must not be priced as a repository"):
            ri._gguf_resident_file_gb(raw)


# F. Token-scoped caches


class TestTokenFingerprint:
    """Both TTL caches are keyed per token, since a gated repo resolves per token."""

    def test_absent_token_is_empty(self):
        assert ri._estimate_token_fingerprint(None) == ""
        assert ri._estimate_token_fingerprint("") == ""

    def test_stable_and_distinct(self):
        a = ri._estimate_token_fingerprint("hf_aaaaaaaaaaaaaaaaaaaa")
        assert a == ri._estimate_token_fingerprint("hf_aaaaaaaaaaaaaaaaaaaa")
        assert a != ri._estimate_token_fingerprint("hf_bbbbbbbbbbbbbbbbbbbb")

    def test_does_not_carry_the_token(self):
        # The keys live in memory for the TTL and are never logged, but a token is
        # a credential and the literal string has no business in a dict key.
        token = "hf_supersecretvalue123"
        fingerprint = ri._estimate_token_fingerprint(token)
        assert token not in fingerprint
        assert fingerprint and len(fingerprint) == 16

    def test_config_cache_is_not_shared_across_tokens(self, monkeypatch):
        # One subject's gated-repo resolution must not be served to another within
        # the 30s TTL. Distinguishable return values make a shared entry visible.
        calls = []

        def _from_identifier(
            model_id,
            hf_token = None,
            **kw,
        ):
            calls.append(hf_token)
            return SimpleNamespace(identifier = model_id, resolved_with = hf_token)

        monkeypatch.setattr(ri.ModelConfig, "from_identifier", staticmethod(_from_identifier))
        # "org/gated" is in no cache root on this box, so the on-disk gate would refuse
        # it before from_identifier is reached and this test would be asserting the
        # gate rather than the token scoping. Answer the gate; the scoping is the
        # subject here, and it has its own test above.
        monkeypatch.setattr(ri, "_estimate_target_is_on_this_disk", lambda _id: True)

        first = ri._cached_estimate_config("org/gated", "Q4_K_M", "token-a", False)
        second = ri._cached_estimate_config("org/gated", "Q4_K_M", "token-b", False)
        assert first.resolved_with == "token-a"
        assert second.resolved_with == "token-b"
        assert calls == ["token-a", "token-b"]

        # Same token still hits the cache -- the scoping must not defeat it, or the
        # endpoint resolves a config on every tick of the slider.
        assert ri._cached_estimate_config("org/gated", "Q4_K_M", "token-a", False) is first
        assert calls == ["token-a", "token-b"]

    def test_files_cache_is_not_shared_across_tokens(self, gqa_gguf, monkeypatch):
        config = _local_config(gqa_gguf)
        ctx_term = ri._estimate_gguf_kv_gb(gqa_gguf, 0, None)
        by_token = {"token-a": 1.0 + ctx_term, "token-b": 7.0 + ctx_term}
        calls = []

        def _required(cfg, **kw):
            calls.append(kw.get("hf_token"))
            return by_token[kw["hf_token"]]

        monkeypatch.setattr(ri, "_estimate_gguf_required_gb", _required)

        assert ri._gguf_resident_file_gb(config, hf_token = "token-a") == pytest.approx(1.0)
        assert ri._gguf_resident_file_gb(config, hf_token = "token-b") == pytest.approx(7.0)
        assert calls == ["token-a", "token-b"]
        # ...and the first token is still cached rather than re-listed.
        assert ri._gguf_resident_file_gb(config, hf_token = "token-a") == pytest.approx(1.0)
        assert calls == ["token-a", "token-b"]


# G. The route


def _estimate(fastapi_request = None, **kwargs):
    """Call the route directly, bypassing the auth dependency."""
    return asyncio.run(
        ri.estimate_memory(
            EstimateMemoryRequest(**kwargs),
            fastapi_request = fastapi_request,
            current_subject = "test",
        )
    )


def _request_with_slots(slots: int):
    """A stand-in for the FastAPI request carrying the server's slot default."""
    return SimpleNamespace(
        app = SimpleNamespace(
            state = SimpleNamespace(
                llama_parallel_slots = slots,
            )
        )
    )


class TestEstimateMemoryRoute:
    """The three "cannot size this" answers, and what they must not do first."""

    def test_ollama_manifest_ref_is_refused_before_materializing(self, monkeypatch):
        # Resolving one writes a .gguf link to disk. This endpoint fires on every
        # slider tick, so it must bail on the prefix alone -- no filesystem write,
        # and not even a config resolution.
        def boom(*a, **kw):
            raise AssertionError("estimate-memory must not materialize an Ollama ref")

        monkeypatch.setattr(ri, "materialize_ollama_model_ref", boom)
        monkeypatch.setattr(ri, "acquire_ollama_model_ref", boom)
        monkeypatch.setattr(ri, "_cached_estimate_config", boom)

        resp = _estimate(model_path = "ollama-manifest:sha256-deadbeef")
        assert resp.available is False
        assert resp.reason == "unsupported_source"

    def test_non_gguf_model_is_not_priced_without_mlx(self, monkeypatch):
        # Safetensors allocates differently, so the GGUF arithmetic would be invented.
        monkeypatch.setattr(
            ri,
            "_cached_estimate_config",
            lambda *a, **kw: SimpleNamespace(is_gguf = False, identifier = "org/model"),
        )
        monkeypatch.setattr(ri, "_mlx_estimate_available", lambda: False)
        resp = _estimate(model_path = "org/model")
        assert resp.available is False
        assert resp.reason == "not_gguf"

    @staticmethod
    def _mlx_target(monkeypatch, model_dir, **config):
        """A resolved non-GGUF config on an MLX host, with its weights at *model_dir*."""
        monkeypatch.setattr(
            ri,
            "_cached_estimate_config",
            lambda *a, **kw: SimpleNamespace(is_gguf = False, identifier = "org/model", **config),
        )
        monkeypatch.setattr(ri, "_mlx_estimate_available", lambda: True)
        monkeypatch.setattr(ri, "_local_mlx_model_dir", lambda config: model_dir)

    @staticmethod
    def _record_breakdown(monkeypatch, **fields):
        """Install a breakdown stub, returning the dict its call is recorded into."""
        import core.inference.mlx_memory as mlx_memory

        seen = {}

        def _breakdown(
            model_dir,
            *,
            n_ctx,
            kv_bits = None,
            load_in_4bit = False,
            **kw,
        ):
            seen.update(
                model_dir = model_dir, n_ctx = n_ctx, kv_bits = kv_bits, load_in_4bit = load_in_4bit
            )
            return mlx_memory.MlxMemoryBreakdown(n_ctx = n_ctx, **fields)

        monkeypatch.setattr(mlx_memory, "mlx_memory_breakdown", _breakdown)
        return seen

    def test_mlx_model_not_on_disk_is_not_downloaded(self, monkeypatch):
        self._mlx_target(monkeypatch, None)
        resp = _estimate(model_path = "org/model")
        assert resp.available is False
        assert resp.reason == "not_downloaded"

    def test_mlx_model_whose_cache_cannot_be_read_is_unsizable(self, monkeypatch):
        self._mlx_target(monkeypatch, "/models/thing")
        import core.inference.mlx_memory as mlx_memory

        monkeypatch.setattr(mlx_memory, "mlx_memory_breakdown", lambda *a, **kw: None)
        resp = _estimate(model_path = "org/model")
        assert resp.available is False
        assert resp.reason == "unsizable"
        assert resp.total_bytes == 0

    def test_mlx_model_is_priced_and_itemized(self, monkeypatch, tmp_path):
        # Real numbers on the wire, itemized as the GGUF arm itemizes them.
        self._mlx_target(monkeypatch, "/models/thing")
        seen = self._record_breakdown(
            monkeypatch,
            weights_bytes = 8_000_000_000,
            kv_bytes = 600_000_000,
            compute_bytes = 400_000_000,
            total_bytes = 9_000_000_000,
            gpu_bytes = 9_000_000_000,
            layer_count = 36,
        )
        # cache_type_kv is llama.cpp's and passed on purpose: MLX never reads it.
        resp = _estimate(
            model_path = "org/model",
            max_seq_length = 16384,
            mlx_kv_bits = 4,
            cache_type_kv = "q8_0",
        )
        assert (resp.available, resp.reason, resp.kv_estimable) == (True, None, True)
        assert (resp.weights_bytes, resp.kv_bytes, resp.compute_bytes) == (
            8_000_000_000,
            600_000_000,
            400_000_000,
        )
        assert (resp.total_bytes, resp.gpu_bytes, resp.n_ctx, resp.layer_count) == (
            9_000_000_000,
            9_000_000_000,
            16384,
            36,
        )
        assert seen == {
            "model_dir": "/models/thing",
            "n_ctx": 16384,
            "kv_bits": 4,
            "load_in_4bit": True,
        }
        _estimate(
            model_path = "org/model",
            max_seq_length = 16384,
            mlx_kv_bits = 4,
            load_in_4bit = False,
        )
        assert seen["load_in_4bit"] is False

        # And the RESOLVED setting reaches it: 4 bits is off for a sidecar-routed architecture.
        import utils.transformers_version as tv

        asked, guarded, real_guard = [], [], ri._offline_guarded
        monkeypatch.setattr(
            tv,
            "latest_tier_active_for",
            lambda name, token = None: asked.append((name, token)) or True,
        )
        monkeypatch.setattr(
            ri,
            "_offline_guarded",
            lambda t, fn, *a, **kw: guarded.append(fn) or real_guard(t, fn, *a, **kw),
        )
        _estimate(model_path = "org/model", max_seq_length = 16384, load_in_4bit = True, hf_token = "tok")
        assert seen["load_in_4bit"] is False
        assert asked == [("org/model", "tok")] and guarded == [tv.latest_tier_active_for]
        monkeypatch.setattr(tv, "latest_tier_active_for", lambda *a, **kw: False)

        monkeypatch.setattr(
            ri,
            "_cached_estimate_config",
            lambda *a, **kw: SimpleNamespace(
                is_gguf = False,
                identifier = "org/model",
                is_lora = True,
                path = str(tmp_path),
                base_model = "org/base",
            ),
        )
        seen.clear()
        resp = _estimate(model_path = "org/model", max_seq_length = 16384, load_in_4bit = True)
        assert (resp.available, resp.reason, resp.total_bytes) == (False, "unsizable", 0)
        assert seen == {}

    def test_the_snapshot_priced_is_the_one_the_load_would_open(self, monkeypatch, tmp_path):
        # mlx-lm cannot read bitsandbytes weights, so the load opens the base repo: the packed shards belong to a repository it never touches.
        import utils.models.model_config as mc

        asked, found = [], []
        snaps, ref = tmp_path / "snapshots", tmp_path / "refs" / "main"
        monkeypatch.setattr(ri, "_estimate_hf_cache_roots", lambda: [None])
        monkeypatch.setattr(
            mc, "_iter_hf_cache_snapshots", lambda name, cache_dir = None: asked.append(name) or found
        )
        if ri._mlx_estimate_available():
            for alias in ("unsloth/Qwen3-4B-bnb-4bit", "unsloth/Qwen3-4B-unsloth-bnb-4bit"):
                assert (
                    ri._local_mlx_model_dir(
                        SimpleNamespace(path = None, is_local = False, identifier = alias)
                    )
                    is None
                )
            assert asked == ["unsloth/Qwen3-4B", "unsloth/Qwen3-4B"]
        # And of the revision `main` names, not a newer snapshot beside it.
        for made in (
            "stub/config.json",
            "new/config.json",
            "new/model.safetensors",
            "old/config.json",
            "old/model.safetensors",
        ):
            (snaps / made).parent.mkdir(parents = True, exist_ok = True)
            (snaps / made).write_text("{}")
        ref.parent.mkdir()
        found[:], new_, old = (
            [snaps / "stub", snaps / "new", snaps / "old"],
            str(snaps / "new"),
            str(snaps / "old"),
        )
        config = SimpleNamespace(path = None, is_local = False, is_lora = False, identifier = "org/model")
        for named, expected in (
            ("old", old),
            ("stub", new_),
            ("collected", new_),
            ("../snapshots/old", new_),
            (None, new_),
        ):
            ref.write_text(named) if named else ref.unlink()
            assert ri._local_mlx_model_dir(config) == expected

    def test_a_part_finished_download_is_not_priced_as_a_smaller_model(self, tmp_path):
        # A shard names the siblings it expects, so two of five on disk is a model 60% smaller.
        def index(weight_map):
            (tmp_path / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": weight_map})
            )

        (tmp_path / "config.json").write_text("{}")
        config = SimpleNamespace(path = str(tmp_path), is_local = True, is_lora = False)
        five = [f"model-0000{i}-of-00005.safetensors" for i in range(1, 6)]
        for name in five[:2]:
            (tmp_path / name).write_bytes(b"")
        assert ri._local_mlx_model_dir(config) is None
        for name in five[2:]:
            (tmp_path / name).write_bytes(b"")
        assert ri._local_mlx_model_dir(config) == str(tmp_path)
        (tmp_path / "adapter-00001-of-00002.safetensors").write_bytes(b"")
        assert ri._local_mlx_model_dir(config) == str(tmp_path)
        (tmp_path / "config.json").write_text(json.dumps({"vision_config": {}}))
        (tmp_path / "weights").mkdir()
        nested = [f"model-0000{i}-of-00002.safetensors" for i in (1, 2)]
        for name in nested:
            (tmp_path / "weights" / name).write_bytes(b"")
        index({n: f"weights/{n}" for n in nested})
        assert ri._local_mlx_model_dir(config) == str(tmp_path)
        (tmp_path / "weights" / nested[1]).unlink()
        assert ri._local_mlx_model_dir(config) is None

    def test_shard_names_that_count_nothing_are_counted_by_the_index(self, tmp_path):
        # Step-3.5-Flash's `model-000NN` and MiMo-V2-Flash's `model_N`: neither states a count.
        (tmp_path / "config.json").write_text("{}")
        config = SimpleNamespace(path = str(tmp_path), is_local = True, is_lora = False)
        (tmp_path / "model.safetensors").write_bytes(b"")
        assert ri._local_mlx_model_dir(config) == str(tmp_path)
        (tmp_path / "model.safetensors").unlink()
        (tmp_path / "weights").mkdir()
        named = [f"model-{i:05d}.safetensors" for i in range(1, 45)] + [
            "mtp.safetensors",
            "weights/extra.safetensors",
        ]
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {n: n for n in named}})
        )
        (tmp_path / named[0]).write_bytes(b"")
        assert ri._local_mlx_model_dir(config) is None
        for name in named[1:]:
            (tmp_path / name).write_bytes(b"")
        assert ri._local_mlx_model_dir(config) == str(tmp_path)
        # An index overlapping this directory nowhere is a parent's, inherited by a re-upload:
        # accepted, because it describes some other snapshot rather than this incomplete one.
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        str(i): f"model-{i:05d}-of-00099.safetensors" for i in range(1, 100)
                    }
                }
            )
        )
        assert ri._local_mlx_model_dir(config) == str(tmp_path)

    def test_an_mlx_estimate_prices_max_seq_length_not_n_ctx(self, monkeypatch):
        # n_ctx is the GGUF-only control and is null on this path.
        self._mlx_target(monkeypatch, "/models/thing")
        seen = self._record_breakdown(
            monkeypatch,
            weights_bytes = 1,
            kv_bytes = 1,
            compute_bytes = 1,
            total_bytes = 3,
            gpu_bytes = 3,
        )
        _estimate(model_path = "org/model", max_seq_length = 8192, n_ctx = 32768)
        assert seen["n_ctx"] == 8192

    def test_an_unpinned_mlx_estimate_prices_the_fit_and_reports_it(self, monkeypatch, tmp_path):
        # A panel showing a figure before an unpinned load must price the load's own fit.
        from core.inference import mlx_inference
        from core.inference.runtime_context import MAX_REQUESTABLE_CONTEXT

        write = (tmp_path / "config.json").write_text
        write(json.dumps({"max_position_embeddings": 262_144}))
        self._mlx_target(monkeypatch, str(tmp_path))
        asked, fit = {}, {"answer": 24_576}
        monkeypatch.setattr(
            mlx_inference,
            "mlx_fit_to_memory",
            lambda model_dir, ceiling, **kw: asked.update(kw, ceiling = ceiling, dir = model_dir)
            or fit["answer"],
        )
        seen = self._record_breakdown(
            monkeypatch,
            weights_bytes = 1,
            kv_bytes = 1,
            compute_bytes = 1,
            total_bytes = 3,
            gpu_bytes = 3,
        )
        # n_ctx is llama.cpp's field, so a caller sending one is not naming an MLX length.
        probed = []
        monkeypatch.setattr(
            mlx_inference,
            "mlx_bound_would_be_enforced",
            lambda *a: probed.append(a) or True,
        )
        resp = _estimate(model_path = "org/model", n_ctx = 32_768, mlx_kv_bits = 4)
        # Asked about the checkpoint the route resolved and the window it would install, since a
        # verdict about some other model or some other length decides nothing here.
        assert probed == [(str(tmp_path), 24_576)]
        # A bound the load will install displaces the quantization, and the fit was priced
        # without it. Where no bound is installed the load quantizes as asked, so the width the
        # caller chose has to survive -- and whether one is installed is probed, not assumed.
        assert (seen["n_ctx"], seen["kv_bits"], resp.context_fitted) == (24_576, None, 24_576)
        for unenforced in (False, None):
            monkeypatch.setattr(
                mlx_inference, "mlx_bound_would_be_enforced", lambda *a, v = unenforced: v
            )
            resp = _estimate(model_path = "org/model", mlx_kv_bits = 4)
            assert (seen["kv_bits"], resp.context_fitted) == (4, 24_576)
        monkeypatch.setattr(mlx_inference, "mlx_bound_would_be_enforced", lambda *a: True)
        # On the dir the route resolved, not the name it was asked about.
        assert asked == {
            "ceiling": 262_144,
            "retains_history": True,
            "dir": str(tmp_path),
            "load_in_4bit": seen["load_in_4bit"],
        }

        # A named length is the user's: nothing is fitted, and the quantization stands.
        resp = _estimate(model_path = "org/model", max_seq_length = 8192, mlx_kv_bits = 4)
        assert (seen["n_ctx"], seen["kv_bits"], resp.context_fitted) == (8192, 4, None)

        # With no fit the declared window stands, held to what /load accepts; then the default.
        fit["answer"] = None
        write(json.dumps({"max_position_embeddings": MAX_REQUESTABLE_CONTEXT * 4}))
        resp = _estimate(model_path = "org/model", mlx_kv_bits = 4)
        assert resp.context_fitted is None
        assert (seen["n_ctx"], seen["kv_bits"]) == (MAX_REQUESTABLE_CONTEXT, 4)
        write(json.dumps({"model_type": "llama"}))
        _estimate(model_path = "org/model")
        assert seen["n_ctx"] == ri._DEFAULT_MLX_ESTIMATE_CTX

        # Only the text path keeps a prompt cache between turns.
        write(json.dumps({"max_position_embeddings": 262_144}))
        self._mlx_target(monkeypatch, str(tmp_path), is_vision = True)
        _estimate(model_path = "org/model")
        assert asked["retains_history"] is False

    def test_gguf_not_on_disk_is_not_downloaded(self, monkeypatch):
        # No header to read and no reaching for the network on a slider drag.
        monkeypatch.setattr(
            ri,
            "_cached_estimate_config",
            lambda *a, **kw: SimpleNamespace(is_gguf = True, identifier = "org/model"),
        )
        monkeypatch.setattr(ri, "_local_gguf_main_path", lambda config: None)
        resp = _estimate(model_path = "org/model", gguf_variant = "Q4_K_M")
        assert resp.available is False
        assert resp.reason == "not_downloaded"

    def test_a_repo_not_on_this_disk_is_refused_before_it_is_resolved(self, monkeypatch):
        # The route promises "nothing is downloaded", and every other test in this class
        # asserts that while monkeypatching _cached_estimate_config -- the one function
        # that can break it. Driven for real, an uncached remote id walked to the Hub:
        # four model_info attempts, an hf_hub_download of config.json, and 12 new paths /
        # 1828 bytes of cache, for a request whose answer is "not on this disk".
        def boom(*a, **kw):
            raise AssertionError("must not resolve a model that is not on this disk")

        monkeypatch.setattr(ri.ModelConfig, "from_identifier", staticmethod(boom))
        monkeypatch.setattr(ri, "_estimate_hf_cache_roots", lambda: [Path(os.devnull).parent])
        resp = _estimate(model_path = "org/definitely-not-cached")
        assert resp.available is False
        # not_downloaded, not unsizable: the sentinel keeps the two apart, because
        # "absent" and "resolution failed" are different answers about the same model.
        assert resp.reason == "not_downloaded"

    def test_the_on_disk_check_reads_every_cache_root(self, tmp_path, monkeypatch):
        # The fix's own worst case, and worse than the bug it fixes: a row that
        # vanishes for a model sitting on the disk. Studio scans configured, legacy and
        # default roots, while the snapshot helper the load path uses takes only the
        # configured one, so the check has to iterate all of them.
        primary, secondary = tmp_path / "a", tmp_path / "b"
        for d in (primary, secondary):
            d.mkdir()
        snap = secondary / "models--org--Model-GGUF" / "snapshots" / "rev1"
        snap.mkdir(parents = True)
        (snap / "model.gguf").write_bytes(b"\0" * 8)
        monkeypatch.setattr(ri, "_estimate_hf_cache_roots", lambda: [primary, secondary])
        assert ri._estimate_target_is_on_this_disk("org/Model-GGUF") is True
        # Casing drifts between download and lookup; the helper is case-insensitive.
        assert ri._estimate_target_is_on_this_disk("ORG/model-gguf") is True
        assert ri._estimate_target_is_on_this_disk("org/Other-GGUF") is False

    def test_the_on_disk_check_fails_open(self, monkeypatch, gqa_gguf):
        # An unanswerable question must not become a blanket refusal. A local path is
        # read off disk and never consults the cache at all.
        def boom(*a, **kw):
            raise OSError("cache root unreadable")

        monkeypatch.setattr(ri, "_estimate_hf_cache_roots", boom)
        assert ri._estimate_target_is_on_this_disk("org/anything") is True
        monkeypatch.setattr(ri, "_estimate_hf_cache_roots", lambda: [])
        assert ri._estimate_target_is_on_this_disk("org/anything") is True
        assert ri._estimate_target_is_on_this_disk(gqa_gguf) is True

    def test_the_cache_evictions_are_serialised(self):
        # The route body runs in an asyncio.to_thread worker, so two panel requests are
        # two real threads in the eviction. min() walks the dict through a Python key
        # function the interpreter can switch out of: a concurrent pop makes it raise
        # KeyError, a concurrent insert RuntimeError, and neither is caught between
        # there and the worker, so it surfaces as a 500 on a slider drag.
        import inspect
        for source in (
            inspect.getsource(ri._gguf_resident_file_gb),
            inspect.getsource(ri._cached_estimate_config),
        ):
            evict = source[source.index("_CACHE_MAX") :]
            assert "_ESTIMATE_CACHE_LOCK" in source
            assert source.index("_ESTIMATE_CACHE_LOCK") < source.index("_CACHE_MAX"), (
                "the lock must be taken BEFORE the length check, or the check-then-"
                "insert it protects is still torn"
            )
            assert "min(" in evict

    def test_the_route_never_reaps_processes_or_leaks_an_atexit_handler(
        self, monkeypatch, gqa_gguf
    ):
        # The sizing helpers build a LlamaCppBackend just to read a header, and its
        # __init__ reaps orphaned llama-servers -- walking /proc, resolving each
        # candidate's exe and SIGNALLING the ones it recognises -- then registers an
        # atexit handler holding the instance for the life of the process. Both are
        # right for a backend that owns a child, neither for a settings panel.
        #
        # Measured before the probes were made inert: five constructions per request,
        # so 50 estimates were 250 /proc scans and 250 retained atexit handlers at
        # 120 ms each. Pricing a load must not be able to kill a server.
        import atexit
        import inspect

        from core.inference.llama_cpp import LlamaCppBackend

        reaps = []
        raw = inspect.getattr_static(LlamaCppBackend, "_kill_orphaned_servers").__func__
        monkeypatch.setattr(
            LlamaCppBackend,
            "_kill_orphaned_servers",
            staticmethod(lambda: reaps.append(1)),
        )
        assert callable(raw)

        config = SimpleNamespace(
            identifier = "local/model",
            gguf_file = gqa_gguf,
            is_gguf = True,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
        )
        monkeypatch.setattr(ri, "_cached_estimate_config", lambda *a, **kw: config)
        monkeypatch.setattr(ri, "_gguf_resident_file_gb", lambda cfg, **kw: 2.0)

        before = atexit._ncallbacks()
        for i in range(5):
            ri._estimate_files_cache.clear()
            ri._estimate_config_cache.clear()
            assert _estimate(model_path = gqa_gguf, n_ctx = 4096 + i).available is True
        assert reaps == [], f"the estimate reaped orphaned servers {len(reaps)}x"
        assert atexit._ncallbacks() == before, (
            f"the estimate leaked {atexit._ncallbacks() - before} atexit handlers over "
            f"five requests; each one retains a whole backend"
        )

    def test_the_inert_probe_mode_is_opt_in(self):
        # Default True, so every existing caller -- above all the real backend that
        # owns the llama-server child -- keeps exactly the behaviour it has today.
        import inspect

        from core.inference.llama_cpp import LlamaCppBackend

        sig = inspect.signature(LlamaCppBackend.__init__)
        param = sig.parameters["manages_processes"]
        assert param.default is True
        assert param.kind is inspect.Parameter.KEYWORD_ONLY

    def test_expert_offload_from_the_extras_is_declared_unmodelled(self, monkeypatch, gqa_gguf):
        # --n-cpu-moe moves individual expert tensors, which the layer fraction
        # cannot express, so the row has to say so. /load strips those flags only on
        # the Manual branch, where the field owns them; anywhere else they reach the
        # child and the estimate reads high without qualification.
        config = SimpleNamespace(identifier = "local/model", gguf_file = gqa_gguf, is_gguf = True)
        monkeypatch.setattr(ri, "_cached_estimate_config", lambda *a, **kw: config)
        monkeypatch.setattr(ri, "_gguf_resident_file_gb", lambda cfg, **kw: 2.0)

        def flagged(**kw):
            return _estimate(model_path = gqa_gguf, n_ctx = 8192, **kw).moe_offload_unmodelled

        assert flagged(gpu_memory_mode = "auto", llama_extra_args = ["--n-cpu-moe", "8"]) is True
        assert flagged(gpu_memory_mode = "auto", llama_extra_args = ["-cmoe"]) is True
        assert flagged(gpu_memory_mode = "auto", llama_extra_args = ["-ot", "exps=CPU"]) is True
        # Zero places nothing, and an unrelated flag is not an offload.
        assert flagged(gpu_memory_mode = "auto", llama_extra_args = ["-ncmoe", "0"]) is False
        assert flagged(gpu_memory_mode = "auto", llama_extra_args = ["--top-k", "40"]) is False
        # Manual strips them, so only its own field speaks there.
        assert flagged(gpu_memory_mode = "manual", llama_extra_args = ["-ncmoe", "8"]) is False
        assert flagged(gpu_memory_mode = "manual", n_cpu_moe = 8) is True

    def test_the_device_count_uses_the_effective_split_mode(self, monkeypatch, gqa_gguf):
        # A layer split across pinned cards is counted differently from a tensor
        # split, so the device count has to be asked the same question the pricing
        # is: the toggle alone is not the mode the launch runs.
        config = SimpleNamespace(identifier = "local/model", gguf_file = gqa_gguf, is_gguf = True)
        monkeypatch.setattr(ri, "_cached_estimate_config", lambda *a, **kw: config)
        monkeypatch.setattr(ri, "_gguf_resident_file_gb", lambda cfg, **kw: 2.0)
        asked: list[bool] = []

        def _count(
            gpu_ids,
            _unused,
            *,
            tensor_parallel = False,
        ):
            asked.append(tensor_parallel)
            return 2

        monkeypatch.setattr(ri, "_guard_device_count", _count)
        _estimate(
            model_path = gqa_gguf,
            n_ctx = 8192,
            tensor_parallel = False,
            llama_extra_args = ["--split-mode", "tensor"],
            selected_gpu_ids = [0, 1],
        )
        assert asked == [True]
        # And one card cannot take a tensor split, whatever the extras ask for, so
        # the count is asked the question load_model will answer.
        _estimate(
            model_path = gqa_gguf,
            n_ctx = 8192,
            tensor_parallel = True,
            llama_extra_args = ["--split-mode", "tensor"],
            selected_gpu_ids = [0],
        )
        assert asked == [True, False]

    def test_unresolvable_config_is_unsizable(self, monkeypatch):
        monkeypatch.setattr(ri, "_cached_estimate_config", lambda *a, **kw: None)
        resp = _estimate(model_path = "org/does-not-exist")
        assert resp.available is False
        assert resp.reason == "unsizable"

    def test_local_gguf_is_priced_end_to_end(self, monkeypatch, gqa_gguf):
        # The success path, with only the two network-capable steps replaced:
        # config resolution and the repository-listing half of the files term.
        config = SimpleNamespace(identifier = "local/model", gguf_file = gqa_gguf, is_gguf = True)
        monkeypatch.setattr(ri, "_cached_estimate_config", lambda *a, **kw: config)
        monkeypatch.setattr(ri, "_gguf_resident_file_gb", lambda cfg, **kw: 2.0)

        resp = _estimate(
            model_path = gqa_gguf,
            n_ctx = 8192,
            cache_type_kv = "q8_0",
            gpu_memory_mode = "manual",
            gpu_layers = 6,
            n_cpu_moe = 2,
        )
        assert resp.available is True
        assert resp.reason is None
        assert resp.weights_bytes == int(round(2.0 * _GIB))
        assert resp.total_bytes == resp.weights_bytes + resp.kv_bytes + resp.compute_bytes
        assert resp.kv_estimable is True
        assert resp.n_ctx == 8192
        assert resp.cache_type_kv == "q8_0"
        assert resp.layer_count == _GQA_FIELDS["block_count"]
        assert resp.gpu_layers == 6
        # --n-cpu-moe moves individual expert tensors, which no layer-count
        # arithmetic models; the panel is told to say so rather than guess.
        assert resp.moe_offload_unmodelled is True

    def test_cached_repo_is_priced_from_disk_not_from_a_listing(self, monkeypatch, gqa_gguf):
        # The glue for the localization: a repo-shaped config whose weights ARE on
        # this disk must be priced locally. _remote_gguf_compute_reserve_gb only
        # runs on the repository arm, and list_gguf_variants is the paths-info call
        # that arm makes -- both raise here, so either one being reached fails.
        def _network(*a, **kw):
            raise AssertionError("estimate-memory must not price a cached repo remotely")

        monkeypatch.setattr(ri, "_cached_estimate_config", lambda *a, **kw: _repo_config())
        monkeypatch.setattr(ri, "_local_gguf_main_path", lambda config: gqa_gguf)
        monkeypatch.setattr(ri, "_remote_gguf_compute_reserve_gb", _network)

        resp = _estimate(model_path = "org/model-GGUF", gguf_variant = "Q4_K_M", n_ctx = 4096)
        assert resp.available is True
        # Priced off the real header on disk, which is the whole point of localizing.
        assert resp.kv_estimable is True
        assert resp.layer_count == _GQA_FIELDS["block_count"]
        assert resp.weights_bytes > 0
        assert resp.n_ctx == 4096


class TestParallelSlotResolution:
    """Blank Parallel Slots means the server default, not one.

    /load resolves the field through ``_resolve_parallel_slots`` and normally
    inherits ``app.state.llama_parallel_slots`` (four in a standard launch). Pricing
    one slot for the default UI state underestimated both the KV cache and the
    slot-scaled compute buffers, which is the configuration most people load.
    """

    @pytest.fixture(autouse = True)
    def _local_model(self, monkeypatch, gqa_gguf):
        config = SimpleNamespace(
            identifier = "local/model",
            gguf_file = gqa_gguf,
            is_gguf = True,
            gguf_mmproj_file = None,
        )
        monkeypatch.setattr(ri, "_cached_estimate_config", lambda *a, **kw: config)
        monkeypatch.setattr(ri, "_local_gguf_main_path", lambda cfg: gqa_gguf)
        monkeypatch.setattr(ri, "_gguf_resident_file_gb", lambda cfg, **kw: 1.0)

    def test_omitted_slots_inherit_the_server_default(self):
        resp = _estimate(
            model_path = "local/model",
            n_ctx = 8192,
            fastapi_request = _request_with_slots(4),
        )
        assert resp.n_parallel == 4

    def test_an_explicit_count_still_wins(self):
        resp = _estimate(
            model_path = "local/model",
            n_ctx = 8192,
            n_parallel = 1,
            fastapi_request = _request_with_slots(4),
        )
        assert resp.n_parallel == 1

    def test_more_slots_cost_more(self):
        one = _estimate(
            model_path = "local/model",
            n_ctx = 8192,
            fastapi_request = _request_with_slots(1),
        )
        four = _estimate(
            model_path = "local/model",
            n_ctx = 8192,
            fastapi_request = _request_with_slots(4),
        )
        # The compute buffer takes an output buffer per extra slot, so it always
        # grows. The cache only does on architectures whose slots get their own
        # cells: under unified KV a plain GQA model shares one set, which is why
        # this asserts >= there rather than pretending otherwise.
        assert four.compute_bytes > one.compute_bytes
        assert four.kv_bytes >= one.kv_bytes
        assert four.total_bytes > one.total_bytes

    def test_a_missing_app_state_falls_back_to_one(self):
        # No FastAPI request at all (the shape older test doubles pass) must not
        # raise; it just loses the default, as _resolve_parallel_slots specifies.
        resp = _estimate(model_path = "local/model", n_ctx = 8192)
        assert resp.n_parallel == 1


class TestNativeLeaseOperation:
    """The route must verify leases under the operation the client can mint.

    Leases are signed for one operation and ``_validate_payload`` compares it
    exactly. "estimate-memory" is not a mintable ``NativePathOperation``, so
    verifying under that name rejected every picked or drag-dropped GGUF and the
    row never appeared for them.
    """

    def test_the_route_verifies_the_validate_model_grant(self, monkeypatch):
        seen = {}

        def _resolve(
            request,
            *,
            operation,
            resolved_ollama_path = None,
        ):
            seen["operation"] = operation
            return ("/tmp/dropped.gguf", "dropped.gguf", True)

        monkeypatch.setattr(ri, "_resolve_model_identifier_for_request", _resolve)
        monkeypatch.setattr(ri, "_cached_estimate_config", lambda *a, **kw: None)
        _estimate(model_path = "dropped.gguf", native_path_lease = "signed")
        assert seen["operation"] == "validate-model"

    def test_the_operation_is_one_the_client_can_mint(self):
        # Guards the pairing rather than the string: the frontend mints through
        # consumeNativePathToken(token, "validate-model"), and this is the list
        # that call is typed against.
        types_ts = (
            Path(__file__).resolve().parents[2] / "frontend/src/features/native-intents/types.ts"
        ).read_text(encoding = "utf-8")
        assert '"validate-model"' in types_ts
        assert "estimate-memory" not in types_ts


from core.inference.llama_cpp import (  # noqa: E402
    _extra_args_draft_offloaded_to_cpu as _draft_on_cpu,
)


class TestDrafterAccounting:
    """A separate drafter costs more than its file.

    The loader budgets its KV cache and rollback state through
    ``_estimate_mtp_overhead_bytes``, which grows with context exactly as the target's
    cache does. Charging only the file made speculation look nearly free.
    """

    @pytest.fixture
    def config(self, gqa_gguf, tmp_path):
        drafter = tmp_path / "mtp-model.gguf"
        drafter.write_bytes(
            _make_gguf_bytes(
                "qwen3",
                {
                    "general.architecture": "qwen3",
                    **{f"qwen3.{k}": v for k, v in _GQA_FIELDS.items()},
                },
            )
        )
        return SimpleNamespace(
            identifier = "local/model",
            gguf_file = gqa_gguf,
            is_gguf = True,
            gguf_mmproj_file = None,
            gguf_mtp_file = str(drafter),
            gguf_dspark_file = None,
            gguf_dflash_file = None,
        )

    def test_the_charged_drafter_is_found_by_the_bytes_it_added(self, config, tmp_path):
        size = os.path.getsize(config.gguf_mtp_file)
        # The kind rides along with the path: it decides which target-side terms the
        # reserve carries, and deriving it separately could name another mode.
        assert ri._charged_drafter_path(config, size) == (config.gguf_mtp_file, "mtp")
        # A figure matching no candidate must not pick one at random.
        assert ri._charged_drafter_path(config, size + 1) is None
        assert ri._charged_drafter_path(config, 0) is None

    def test_drafter_runtime_grows_with_context(self, config, gqa_gguf, monkeypatch):
        drafter_bytes = os.path.getsize(config.gguf_mtp_file)
        main_gb = 1.0

        # Mimic the files term: the drafter is charged unless the probe forces it off.
        def _files(
            cfg,
            *,
            llama_extra_args = None,
            **kw,
        ):
            pinned = _draft_on_cpu(list(llama_extra_args or ()))
            return main_gb + (0.0 if pinned else drafter_bytes / 1024**3)

        monkeypatch.setattr(ri, "_gguf_resident_file_gb", _files)
        small = ri._gguf_memory_breakdown(config, gqa_gguf, n_ctx = 2048)
        large = ri._gguf_memory_breakdown(config, gqa_gguf, n_ctx = 32768)
        # Weights are the same files at both contexts; only the drafter's cache moved.
        assert small.weights_bytes == large.weights_bytes
        drafter_runtime_small = small.total_bytes - small.weights_bytes - small.kv_bytes
        drafter_runtime_large = large.total_bytes - large.weights_bytes - large.kv_bytes
        assert drafter_runtime_large > drafter_runtime_small

    def test_a_cpu_pinned_drafter_stays_in_the_total(self, config, gqa_gguf, monkeypatch):
        drafter_bytes = os.path.getsize(config.gguf_mtp_file)
        main_gb = 1.0

        def _files(
            cfg,
            *,
            llama_extra_args = None,
            **kw,
        ):
            pinned = _draft_on_cpu(list(llama_extra_args or ()))
            return main_gb + (0.0 if pinned else drafter_bytes / 1024**3)

        monkeypatch.setattr(ri, "_gguf_resident_file_gb", _files)
        on_gpu = ri._gguf_memory_breakdown(config, gqa_gguf, n_ctx = 8192)
        on_cpu = ri._gguf_memory_breakdown(
            config, gqa_gguf, n_ctx = 8192, llama_extra_args = ["--spec-draft-ngl", "0"]
        )
        # _estimate_gguf_required_gb drops a CPU-pinned drafter because it is a VRAM
        # admission figure. This panel reports host RAM too, so the bytes stay in the
        # total and leave only the GPU column.
        assert on_cpu.total_bytes == on_gpu.total_bytes
        assert on_cpu.weights_bytes == on_gpu.weights_bytes
        assert on_cpu.gpu_bytes < on_gpu.gpu_bytes

    def test_the_drafter_runtime_reports_its_own_gpu_share(self, config, gqa_gguf, monkeypatch):
        # The panel used to label this line from kv_on_gpu, the TARGET cache's
        # placement, which is set by a different flag. Wrong in both directions: silent
        # about a genuinely host-resident draft cache, and claiming host RAM for one
        # this same call had just charged to gpu_bytes. So the share is reported.
        drafter_bytes = os.path.getsize(config.gguf_mtp_file)

        def _files(
            cfg,
            *,
            llama_extra_args = None,
            **kw,
        ):
            pinned = _draft_on_cpu(list(llama_extra_args or ()))
            return 1.0 + (0.0 if pinned else drafter_bytes / 1024**3)

        monkeypatch.setattr(ri, "_gguf_resident_file_gb", _files)
        on_gpu = ri._gguf_memory_breakdown(config, gqa_gguf, n_ctx = 8192)
        assert on_gpu.drafter_runtime_bytes > 0
        # Default placement: all of it, so the row says nothing rather than guessing.
        assert on_gpu.drafter_runtime_gpu_bytes == on_gpu.drafter_runtime_bytes

        on_cpu = ri._gguf_memory_breakdown(
            config, gqa_gguf, n_ctx = 8192, llama_extra_args = ["--spec-draft-ngl", "0"]
        )
        assert on_cpu.drafter_runtime_gpu_bytes == 0

        # --no-kv-offload moves the TARGET cache and leaves the drafter alone. A
        # boolean read off kv_on_gpu would call the whole term host-resident here,
        # while gpu_bytes still carries the draft cache.
        nkvo = ri._gguf_memory_breakdown(
            config, gqa_gguf, n_ctx = 8192, llama_extra_args = ["--no-kv-offload"]
        )
        assert nkvo.kv_on_gpu is False
        assert nkvo.drafter_runtime_gpu_bytes > 0

        # Whatever the placement, the share is never more than the term it is a share
        # of, and never negative.
        for b in (on_gpu, on_cpu, nkvo):
            assert 0 <= b.drafter_runtime_gpu_bytes <= b.drafter_runtime_bytes


class TestDeviceCount:
    """A pinned layer split pays per-device overhead, not just tensor mode."""

    def test_a_pinned_layer_split_counts_its_cards(self):
        # _gguf_runtime_bytes adds pipeline overhead per extra device and replicates
        # the context-linear compute term, so forcing 1 here underestimated both.
        assert ri._guard_device_count([0, 1], None, tensor_parallel = False) == 2
        assert ri._guard_device_count([0, 1, 2], None, tensor_parallel = False) == 3
        # Automatic placement lands on one card until the fit says otherwise.
        assert ri._guard_device_count(None, None, tensor_parallel = False) == 1


# A hybrid-Mamba target: the recurrent state the verification rollback copies, and the
# only shape where the draft depth changes the number the panel prints.
_HYBRID_FIELDS = {
    **_GQA_FIELDS,
    "block_count": 24,
    "full_attention_interval": 4,
    "ssm.inner_size": 4096,
    "ssm.state_size": 128,
    "ssm.group_count": 8,
    "ssm.conv_kernel": 4,
}


class TestBlankDraftDepth:
    """Draft Tokens left blank is the launcher's default, not zero.

    ``_build_speculative_flags`` emits its own depth when the field is unset, and
    ``_estimate_mtp_overhead_bytes`` scales the Hybrid-Mamba rollback state by it, so
    pricing zero dropped that allocation from both the total and the GPU figure.
    """

    @pytest.fixture
    def hybrid(self, tmp_path) -> str:
        return _write_gguf(tmp_path, "qwen3next", _HYBRID_FIELDS, name = "hybrid.gguf")

    @pytest.fixture
    def config(self, hybrid, tmp_path):
        drafter = tmp_path / "mtp-hybrid.gguf"
        drafter.write_bytes(
            _make_gguf_bytes(
                "qwen3",
                {
                    "general.architecture": "qwen3",
                    **{f"qwen3.{k}": v for k, v in _GQA_FIELDS.items()},
                },
            )
        )
        return SimpleNamespace(
            identifier = "local/hybrid",
            gguf_file = hybrid,
            is_gguf = True,
            gguf_mmproj_file = None,
            gguf_mtp_file = str(drafter),
            gguf_dspark_file = None,
            gguf_dflash_file = None,
        )

    @pytest.fixture(autouse = True)
    def _fixed_files(self, config, monkeypatch):
        drafter_bytes = os.path.getsize(config.gguf_mtp_file)

        def _files(
            cfg,
            *,
            llama_extra_args = None,
            **kw,
        ):
            pinned = _draft_on_cpu(list(llama_extra_args or ()))
            return 1.0 + (0.0 if pinned else drafter_bytes / 1024**3)

        monkeypatch.setattr(ri, "_gguf_resident_file_gb", _files)

    def _priced(self, config, hybrid, **kw):
        return ri._gguf_memory_breakdown(config, hybrid, n_ctx = 8192, n_parallel = 4, **kw)

    def test_blank_prices_the_launch_default_not_zero(self, config, hybrid):
        # THE regression: the UI sends null for a blank field, which used to reach
        # _estimate_mtp_overhead_bytes as a depth of 0 and cost the rollback copies.
        zero = self._priced(config, hybrid, spec_draft_n_max = 0)
        one = self._priced(config, hybrid, spec_draft_n_max = 1)
        blank = self._priced(config, hybrid, spec_draft_n_max = None)
        per_copy = one.total_bytes - zero.total_bytes
        assert per_copy > 0, "fixture is not hybrid-Mamba; the test would prove nothing"
        assert blank.total_bytes > zero.total_bytes
        # 2 on a GPU box, 3 without: the two platform defaults _build_speculative_flags
        # emits. Not pinned to one, so this passes on a runner with or without a card.
        assert (blank.total_bytes - zero.total_bytes) // per_copy in (2, 3)
        # The drafter sits on the GPU here, so the fit verdict moves with it too.
        assert blank.gpu_bytes > zero.gpu_bytes

    def test_an_explicit_depth_still_wins(self, config, hybrid):
        zero = self._priced(config, hybrid, spec_draft_n_max = 0)
        one = self._priced(config, hybrid, spec_draft_n_max = 1)
        per_copy = one.total_bytes - zero.total_bytes
        five = self._priced(config, hybrid, spec_draft_n_max = 5)
        assert five.total_bytes - zero.total_bytes == 5 * per_copy

    def test_an_extras_depth_wins_over_a_blank_field(self, config, hybrid):
        # Last-wins at launch, so the extras flag is what the child really drafts at.
        zero = self._priced(config, hybrid, spec_draft_n_max = 0)
        one = self._priced(config, hybrid, spec_draft_n_max = 1)
        per_copy = one.total_bytes - zero.total_bytes
        pinned = self._priced(
            config,
            hybrid,
            spec_draft_n_max = None,
            llama_extra_args = ["--spec-draft-n-max", "5"],
        )
        assert pinned.total_bytes - zero.total_bytes == 5 * per_copy

    def test_depth_resolution_follows_the_launch_precedence(self, config):
        drafter = config.gguf_mtp_file
        assert ri._estimate_draft_n_max(config, drafter, requested = 7, extras = []) == 7
        # Extras beat the field, and an explicit zero is honoured as "draft nothing".
        assert (
            ri._estimate_draft_n_max(config, drafter, requested = 7, extras = ["--draft-max", "4"]) == 4
        )
        assert ri._estimate_draft_n_max(config, drafter, requested = 0, extras = []) == 0
        assert ri._estimate_draft_n_max(config, drafter, requested = None, extras = []) in (2, 3)
        # DSpark launches at 3 regardless of the card, so it is priced at 3.
        dspark = SimpleNamespace(gguf_dspark_file = "/tmp/dspark-model.gguf")
        assert (
            ri._estimate_draft_n_max(dspark, "/tmp/dspark-model.gguf", requested = None, extras = [])
            == 3
        )


class TestQuantSubdirCompanions:
    """A cached repo that files each quant under its own directory.

    ``snapshot/UD-Q4_K_XL/model-00001-of-00002.gguf`` with ``mmproj-*.gguf`` and the
    drafter at ``snapshot/`` is the standard Hugging Face layout for any quant over the
    per-file limit. Passing the weight's own parent as ``search_root`` gave the
    detectors nothing to walk up to, so the projector went uncounted.
    """

    def test_a_projector_at_the_snapshot_root_is_found(self, tmp_path):
        quant_dir = tmp_path / "UD-Q4_K_XL"
        quant_dir.mkdir()
        weight = _write_gguf(
            quant_dir, "qwen3", _GQA_FIELDS, name = "model-UD-Q4_K_XL-00001-of-00002.gguf"
        )
        projector = _write_gguf(tmp_path, "clip", {"block_count": 2}, name = "mmproj-F16.gguf")
        localized = ri._localized_estimate_config(_repo_config(is_vision = True), weight)
        assert localized.gguf_mmproj_file == projector

    def test_a_drafter_at_the_snapshot_root_is_found(self, tmp_path):
        quant_dir = tmp_path / "Q4_K_M"
        quant_dir.mkdir()
        weight = _write_gguf(quant_dir, "qwen3", _GQA_FIELDS, name = "model-Q4_K_M.gguf")
        drafter = _write_gguf(tmp_path, "qwen3", _GQA_FIELDS, name = "mtp-model.gguf")
        localized = ri._localized_estimate_config(_repo_config(), weight)
        assert localized.gguf_mtp_file == drafter

    def test_a_flat_layout_still_scans_only_the_weights_directory(self, tmp_path):
        # The widened root must not reach a sibling model's projector: the quant check
        # is what keeps a non-quant directory name from being walked out of.
        model_dir = tmp_path / "some-model"
        model_dir.mkdir()
        weight = _write_gguf(model_dir, "qwen3", _GQA_FIELDS, name = "model-Q4_K_M.gguf")
        _write_gguf(tmp_path, "clip", {"block_count": 2}, name = "mmproj-F16.gguf")
        localized = ri._localized_estimate_config(_repo_config(is_vision = True), weight)
        assert localized.gguf_mmproj_file is None


class TestDrafterEdgeCases:
    """The narrow drafter paths, each of which produced a wrong answer or a 500."""

    @pytest.fixture
    def bare_config(self, gqa_gguf):
        """A model shipping no sidecar of any kind."""
        return SimpleNamespace(
            identifier = "local/model",
            gguf_file = gqa_gguf,
            is_gguf = True,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
        )

    def test_a_cpu_pin_without_a_drafter_does_not_crash(self, bare_config, gqa_gguf, monkeypatch):
        # Only the else branch bound gpu_drafter_bytes, so a CPU pin on a model with
        # nothing to pin raised UnboundLocalError and answered 500 for an ordinary load.
        monkeypatch.setattr(ri, "_gguf_resident_file_gb", lambda cfg, **kw: 1.0)
        breakdown = ri._gguf_memory_breakdown(
            bare_config, gqa_gguf, n_ctx = 4096, llama_extra_args = ["--spec-draft-ngl", "0"]
        )
        assert breakdown is not None
        assert breakdown.drafter_runtime_bytes == 0
        assert breakdown.total_bytes > 0

    def test_a_custom_model_draft_is_carried_into_runtime_sizing(
        self, bare_config, gqa_gguf, tmp_path
    ):
        # The launch opens whatever --model-draft names, which need not be one of the
        # discovered sidecars; matching only those dropped its cache entirely.
        custom = tmp_path / "my-drafter.gguf"
        custom.write_bytes(Path(gqa_gguf).read_bytes())
        found = ri._charged_drafter_path(bare_config, 4096, extras = ["--model-draft", str(custom)])
        assert found == (str(custom), "extras")
        # A remote repo is not a local file, so it stays unsized rather than guessed.
        assert (
            ri._charged_drafter_path(
                bare_config, 4096, extras = ["--spec-draft-hf", "org/drafter-GGUF"]
            )
            is None
        )
        # And nothing is resolved when no drafter was charged in the first place.
        assert (
            ri._charged_drafter_path(bare_config, 0, extras = ["--model-draft", str(custom)]) is None
        )

    def test_draft_cache_overrides_come_from_the_extras(self):
        # K and V are independent at launch, so passing the panel field for both
        # priced a cache the load will not allocate.
        from core.inference.llama_cpp import _extra_args_draft_cache_types
        assert _extra_args_draft_cache_types(["--cache-type-k-draft", "f32"]) == ("f32", None)
        assert _extra_args_draft_cache_types(["--cache-type-v-draft", "q8_0"]) == (None, "q8_0")
        # Asserted on the source before, which meant it could only ever restate the
        # implementation. The precedence itself is checked by behaviour below, where a
        # wrong order changes the number instead of the spelling.


class TestSpeculativeModeTerms:
    """A separate drafter is not an MTP head, and the two are not placed together.

    ``_estimate_mtp_overhead_bytes`` defaults to charging both target-side terms
    because an unsure caller should over-reserve. The loader does not stay unsure: it
    derives them from the engaged mode. Defaulting here charged DSpark and DFlash a
    second full copy of an MLA target's KV, which the mode never allocates.
    """

    @pytest.fixture
    def mla(self, tmp_path) -> str:
        # kv_lora_rank is what makes the duplicated target context real.
        return _write_gguf(
            tmp_path,
            "deepseek2",
            {**_GQA_FIELDS, "attention.kv_lora_rank": 512},
            name = "mla.gguf",
        )

    def _config(self, mla, tmp_path, kind: str):
        drafter = tmp_path / f"{kind}-drafter.gguf"
        drafter.write_bytes(Path(mla).read_bytes())
        return SimpleNamespace(
            identifier = "local/mla",
            gguf_file = mla,
            is_gguf = True,
            gguf_mmproj_file = None,
            gguf_mtp_file = str(drafter) if kind == "mtp" else None,
            gguf_dspark_file = str(drafter) if kind == "dspark" else None,
            gguf_dflash_file = None,
        )

    @pytest.fixture
    def priced_files(self, mla, tmp_path, monkeypatch):
        """The files term as the real one behaves: drafter charged unless pinned."""
        drafter_bytes = os.path.getsize(mla)

        def _files(
            cfg,
            *,
            llama_extra_args = None,
            **kw,
        ):
            pinned = _draft_on_cpu(list(llama_extra_args or ()))
            return 1.0 + (0.0 if pinned else drafter_bytes / 1024**3)

        monkeypatch.setattr(ri, "_gguf_resident_file_gb", _files)

    def _target_ctx_copy(self, mla, n_ctx: int) -> int:
        probe = ri.LlamaCppBackend()
        probe._read_gguf_metadata(mla)
        return probe._estimate_kv_cache_bytes(n_ctx, "f16")

    def test_a_dspark_drafter_is_not_charged_the_targets_context(self, mla, tmp_path, priced_files):
        ctx = 131072
        mtp = ri._gguf_memory_breakdown(self._config(mla, tmp_path, "mtp"), mla, n_ctx = ctx)
        dspark = ri._gguf_memory_breakdown(self._config(mla, tmp_path, "dspark"), mla, n_ctx = ctx)
        # DSpark loads its own drafter with its own cache and keeps no copy of the
        # target's, so the difference is exactly that copy, priced at f16.
        assert mtp.drafter_runtime_bytes - dspark.drafter_runtime_bytes == self._target_ctx_copy(
            mla, ctx
        )
        assert dspark.total_bytes < mtp.total_bytes

    def test_a_gpu_drafter_is_charged_when_the_target_keeps_no_layers(
        self, mla, tmp_path, priced_files
    ):
        # A separate drafter does not inherit --gpu-layers: llama.cpp overwrites it
        # with the draft placement, whose default is auto. At --gpu-layers 0 the
        # drafter is still on the GPU and still holds its cache there.
        config = self._config(mla, tmp_path, "dspark")
        manual = dict(gpu_memory_mode = "manual", gpu_layers = 0, n_ctx = 131072)
        on_gpu = ri._gguf_memory_breakdown(config, mla, **manual)
        pinned = ri._gguf_memory_breakdown(
            config, mla, llama_extra_args = ["--spec-draft-ngl", "0"], **manual
        )
        assert on_gpu.drafter_runtime_bytes > 0
        # The drafter's file and its cache both leave the GPU when it is pinned, and
        # DSpark on this target keeps nothing in the target's context.
        assert (
            on_gpu.gpu_bytes - pinned.gpu_bytes
            == on_gpu.drafter_runtime_bytes + os.path.getsize(mla)
        )

    def test_the_target_side_of_the_reserve_stays_with_the_target(
        self, mla, tmp_path, priced_files
    ):
        ctx = 131072
        config = self._config(mla, tmp_path, "mtp")
        copy_bytes = self._target_ctx_copy(mla, ctx)
        # Target on the CPU, drafter on the GPU: the draft cache is charged, the
        # duplicated target context is not.
        manual = dict(gpu_memory_mode = "manual", gpu_layers = 0, n_ctx = ctx)
        no_layers = ri._gguf_memory_breakdown(config, mla, **manual)
        no_layers_pinned = ri._gguf_memory_breakdown(
            config, mla, llama_extra_args = ["--spec-draft-ngl", "0"], **manual
        )
        assert copy_bytes > 0
        # What pinning the drafter takes off the GPU is its own cache and file, never
        # the target's copy, which was not on the GPU here in the first place.
        assert no_layers.gpu_bytes - no_layers_pinned.gpu_bytes == (
            no_layers.drafter_runtime_bytes - copy_bytes + os.path.getsize(mla)
        )
        # The mirror image: drafter pinned to the CPU, target fully offloaded. The
        # copy lives in the target's context, so pinning does not move it.
        pinned = ri._gguf_memory_breakdown(
            config, mla, n_ctx = ctx, llama_extra_args = ["--spec-draft-ngl", "0"]
        )
        unpinned = ri._gguf_memory_breakdown(config, mla, n_ctx = ctx)
        assert unpinned.gpu_bytes - pinned.gpu_bytes == (
            unpinned.drafter_runtime_bytes - copy_bytes + os.path.getsize(mla)
        )
        # And the copy is still charged with the drafter pinned away, which is what
        # the loader reserves for the same launch.
        assert pinned.gpu_bytes - no_layers_pinned.gpu_bytes > copy_bytes

    def test_extras_owning_the_spec_block_price_the_builds_depth(self, mla):
        # _build_speculative_flags returns without emitting a depth once the extras
        # name --spec-type, so neither the panel field nor Studio's 2/3 platform
        # default reaches the child: it drafts at the build's own number.
        config = SimpleNamespace(gguf_dspark_file = None)
        owned = ["--spec-type", "draft-mtp"]
        assert ri._estimate_draft_n_max(config, mla, requested = None, extras = owned) == 16
        assert ri._estimate_draft_n_max(config, mla, requested = 2, extras = owned) == 16
        # An explicit depth still wins, and without --spec-type the field does.
        assert (
            ri._estimate_draft_n_max(
                config, mla, requested = 2, extras = [*owned, "--spec-draft-n-max", "5"]
            )
            == 5
        )
        assert ri._estimate_draft_n_max(config, mla, requested = 2, extras = []) == 2

    def test_each_mode_charges_only_the_terms_it_allocates(self):
        # Studio's own sidecars: only MTP duplicates the target context, and all three
        # pay rollback. Extras that name a type own the answer, and a bare
        # --model-draft is draft-simple, which allocates neither.
        assert ri._estimate_spec_mode_terms("mtp", []) == (True, True)
        assert ri._estimate_spec_mode_terms("dspark", []) == (False, True)
        assert ri._estimate_spec_mode_terms("dflash", []) == (False, True)
        # A bare --model-draft on a target Studio does not recognise as MTP really is
        # draft-simple, llama.cpp's default, and allocates neither.
        assert ri._estimate_spec_mode_terms("extras", []) == (False, False)
        # But on a target Studio DOES recognise, it still emits --spec-type draft-mtp
        # and the extras path merely last-wins as the drafter, so the duplicated MLA
        # context and the rollback state are both really allocated.
        assert ri._estimate_spec_mode_terms("extras", [], studio_emits_mtp = True) == (True, True)
        # A --spec-type in the extras still owns the answer outright either way.
        assert ri._estimate_spec_mode_terms(
            "extras", ["--spec-type", "draft-simple"], studio_emits_mtp = True
        ) == (False, False)
        assert ri._estimate_spec_mode_terms("extras", ["--spec-type", "draft-simple"]) == (
            False,
            False,
        )
        assert ri._estimate_spec_mode_terms("extras", ["--spec-type", "draft-mtp"]) == (True, True)
        assert ri._estimate_spec_mode_terms("extras", ["--spec-type", "draft-eagle3"]) == (
            False,
            True,
        )


class TestLaunchShapedPricing:
    """Three places the panel priced a launch the loader would not perform.

    Each is the same shape: a setting resolved one way in ``load_model`` and another
    way here, where the difference is large enough to flip the verdict.
    """

    @pytest.fixture
    def swa(self, tmp_path) -> str:
        # Sliding-window attention, so --swa-full has something to change: without it
        # the cache holds the window, with it the whole context.
        return _write_gguf(
            tmp_path,
            "gemma3",
            {**_GQA_FIELDS, "attention.sliding_window": 4096, "context_length": 262144},
            name = "swa.gguf",
        )

    @pytest.fixture
    def spec_config(self, swa, tmp_path, monkeypatch):
        drafter = tmp_path / "mtp-swa.gguf"
        drafter.write_bytes(Path(swa).read_bytes())
        drafter_bytes = drafter.stat().st_size

        def _files(
            cfg,
            *,
            llama_extra_args = None,
            **kw,
        ):
            pinned = _draft_on_cpu(list(llama_extra_args or ()))
            return 1.0 + (0.0 if pinned else drafter_bytes / 1024**3)

        monkeypatch.setattr(ri, "_gguf_resident_file_gb", _files)
        return SimpleNamespace(
            identifier = "local/swa",
            gguf_file = swa,
            is_gguf = True,
            gguf_mmproj_file = None,
            gguf_mtp_file = str(drafter),
            gguf_dspark_file = None,
            gguf_dflash_file = None,
        )

    def test_the_drafter_cache_is_priced_at_the_targets_layout(self, spec_config, swa):
        # --swa-full reached the target cache and stopped there, so a drafter with the
        # same geometry was priced holding a window while the launch gives it the whole
        # context. The loader passes all four layout settings; so does this now.
        windowed = ri._gguf_memory_breakdown(spec_config, swa, n_ctx = 131072)
        full = ri._gguf_memory_breakdown(
            spec_config, swa, n_ctx = 131072, llama_extra_args = ["--swa-full"]
        )
        assert full.kv_bytes > windowed.kv_bytes
        # Same header on both sides, so the drafter's cache moves exactly as the
        # target's does rather than staying at the windowed figure.
        assert full.drafter_runtime_bytes == full.kv_bytes
        assert windowed.drafter_runtime_bytes == windowed.kv_bytes

    def test_a_cpu_device_selection_takes_the_weights_off_the_gpu(self, spec_config, swa):
        """``--device none`` runs on the CPU whatever the layer count says.

        The loader's residency gate asks ``_device_selection_is_cpu`` before it looks at
        -ngl for exactly this reason. The estimate read Auto as fully GPU-resident and
        charged the whole footprint to VRAM, which is a GPU-exceeds warning for a load
        that touches no GPU at all.
        """
        priced = dict(n_ctx = 32768, cache_type_kv = "f16")
        on_gpu = ri._gguf_memory_breakdown(spec_config, swa, **priced)
        for flag in (["--device", "none"], ["--device", "cpu"], ["-dev", "none"]):
            cpu_only = ri._gguf_memory_breakdown(spec_config, swa, llama_extra_args = flag, **priced)
            assert cpu_only.gpu_bytes == 0, flag
            # The memory is still spent, just not on the card.
            assert cpu_only.total_bytes > 0
        assert on_gpu.gpu_bytes > 0

    def test_context_checkpoints_are_charged_to_the_host_not_the_gpu(self, spec_config, swa):
        """Checkpoints are host RAM, so they belong in the total and not in gpu_bytes.

        llama.cpp holds each one in ``common_prompt_checkpoint``'s
        ``std::vector<uint8_t>`` buffers, on the host heap and bounded by
        ``--cache-ram``; Studio's own load schema says "Each costs host memory". They
        are inside ``kv_bytes`` because that is what the training guard budgets, so the
        GPU figure has to take them back out. Priced into VRAM, the default 32 per slot
        warned about GPU pressure the launch never creates.
        """
        priced = dict(n_ctx = 131072, cache_type_kv = "f16")
        none = ri._gguf_memory_breakdown(spec_config, swa, ctx_checkpoints = 0, **priced)
        many = ri._gguf_memory_breakdown(spec_config, swa, ctx_checkpoints = 8, **priced)

        # The snapshots are real and substantial, or the rest of this proves nothing.
        assert many.kv_bytes > none.kv_bytes
        # ... and they are in the aggregate total, which is what the host pays.
        assert many.total_bytes > none.total_bytes
        # But the GPU figure does not move: that is the whole finding.
        assert many.gpu_bytes == none.gpu_bytes

    def test_one_card_is_priced_as_the_layer_load_it_launches(self, spec_config, swa):
        # Tensor mode needs two usable GPUs. Below that load_model drops it, so pricing
        # tensor charged per-device compute buffers for a launch that runs neither.
        #
        # The cache type is no longer part of this: #8939 removed the gate that rewrote
        # both axes to f16 on a tensor split, so a quantized KV now survives one and the
        # downgrade cannot move it. Asserted equal rather than dropped, since a
        # reintroduced gate would put an f16 cache back on the tensor side.
        priced = dict(n_ctx = 32768, cache_type_kv = "q4_0", tensor_parallel = True, n_devices = 1)
        downgraded = ri._gguf_memory_breakdown(
            spec_config, swa, tensor_split_possible = False, **priced
        )
        as_tensor = ri._gguf_memory_breakdown(
            spec_config, swa, tensor_split_possible = True, **priced
        )
        assert downgraded.cache_type_kv == "q4_0"
        assert as_tensor.cache_type_kv == "q4_0"
        assert downgraded.kv_bytes == as_tensor.kv_bytes
        assert downgraded.compute_bytes < as_tensor.compute_bytes
        assert downgraded.total_bytes < as_tensor.total_bytes

    def test_manual_auto_layers_is_priced_as_the_layer_load_it_launches(self, spec_config, swa):
        # Manual with Auto layers hands the budget to llama.cpp --fit, which load_model
        # says outright is incompatible with tensor parallelism and drops the split for.
        # Two cards are visible and pinned, so nothing else downgrades it: only the
        # layer count does. Priced as tensor, the per-device buffers are charged for a
        # launch that runs a layer split.
        priced = dict(
            n_ctx = 32768,
            cache_type_kv = "q4_0",
            tensor_parallel = True,
            n_devices = 2,
            gpu_memory_mode = "manual",
            tensor_split_possible = True,
        )
        auto_layers = ri._gguf_memory_breakdown(spec_config, swa, gpu_layers = None, **priced)
        explicit = ri._gguf_memory_breakdown(spec_config, swa, gpu_layers = 40, **priced)
        assert auto_layers.compute_bytes != explicit.compute_bytes
        # The layer-split arm, which is what /load runs here.
        as_layers = ri._gguf_memory_breakdown(
            spec_config, swa, gpu_layers = None, **{**priced, "tensor_parallel": False}
        )
        assert auto_layers.compute_bytes == as_layers.compute_bytes

    def test_manual_zero_layers_is_priced_as_the_cpu_load_it_launches(self, spec_config, swa):
        # gpu_layers=0 leaves nothing on the GPU to split. load_model drops the split
        # rather than let --split-mode tensor abort the server under the CPU-only mask,
        # so a tensor price here is per-device buffers for a load that takes no VRAM.
        priced = dict(
            n_ctx = 32768,
            cache_type_kv = "q4_0",
            n_devices = 2,
            gpu_memory_mode = "manual",
            gpu_layers = 0,
            tensor_split_possible = True,
        )
        as_tensor = ri._gguf_memory_breakdown(spec_config, swa, tensor_parallel = True, **priced)
        as_layers = ri._gguf_memory_breakdown(spec_config, swa, tensor_parallel = False, **priced)
        assert as_tensor.compute_bytes == as_layers.compute_bytes
        assert as_tensor.total_bytes == as_layers.total_bytes

    def test_an_ngl_in_the_extras_keeps_the_manual_split(self, spec_config, swa):
        # /load translates the last -ngl into the field before deciding, so a slider at
        # 0 with -ngl 40 in the extras is a 40-layer load and does reach a tensor
        # launch. Dropping on the field alone would price it as a layer split.
        assert ri._manual_keeps_tensor_split("manual", 0, ["-ngl", "40"]) is True
        assert ri._manual_keeps_tensor_split("manual", 40, ["-ngl", "0"]) is False
        assert ri._manual_keeps_tensor_split("manual", None, None) is False
        assert ri._manual_keeps_tensor_split("manual", 0, None) is False
        assert ri._manual_keeps_tensor_split("manual", 1, None) is True
        # Auto is the planner's call, not ours, whatever the count says.
        assert ri._manual_keeps_tensor_split("auto", 0, None) is True
        assert ri._manual_keeps_tensor_split(None, None, None) is True
        # A malformed override is shrugged off rather than raised, same as the layer
        # fraction does with it.
        assert ri._manual_keeps_tensor_split("manual", 8, ["-ngl", "banana"]) is True

    def test_a_one_card_pin_cannot_tensor_split(self):
        # A pin answers for itself and needs no probe, which is the deterministic half
        # of the rule. Without a pin the host answers, and that must not raise here
        # whatever this machine has.
        assert ri._tensor_split_possible([0]) is False
        assert ri._tensor_split_possible([0, 1]) is True
        assert ri._tensor_split_possible([2, 3, 5]) is True
        assert isinstance(ri._tensor_split_possible(None), bool)

    def test_the_vision_encoder_costs_more_than_its_projector_file(
        self, swa, tmp_path, monkeypatch
    ):
        # The encoder's buffers run about 1.3x the file, which the placement path
        # budgets as _MMPROJ_VRAM_SAFETY - 1. Counting only the file called a
        # near-capacity multimodal load a fit.
        projector = tmp_path / "mmproj-swa.gguf"
        projector.write_bytes(b"\0" * 1024)
        config = SimpleNamespace(
            identifier = "local/vision",
            gguf_file = swa,
            is_gguf = True,
            gguf_mmproj_file = str(projector),
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
        )
        monkeypatch.setattr(ri, "_gguf_resident_file_gb", lambda cfg, **kw: 4.0)
        real_size = ri.LlamaCppBackend._get_gguf_size_bytes
        monkeypatch.setattr(
            ri.LlamaCppBackend,
            "_get_gguf_size_bytes",
            staticmethod(
                lambda path: 3 * _GIB
                if os.path.basename(str(path)) == "swa.gguf"
                else real_size(path)
            ),
        )
        expected = int(1024 * (ri.LlamaCppBackend._MMPROJ_VRAM_SAFETY - 1.0))
        resident = ri._gguf_memory_breakdown(config, swa, n_ctx = 8192)
        assert resident.projector_runtime_bytes == expected
        assert resident.total_bytes == (
            resident.weights_bytes + resident.kv_bytes + resident.compute_bytes + expected
        )
        # The buffers sit with the projector, so pinning it takes them off the GPU
        # while leaving them in the total.
        pinned = ri._gguf_memory_breakdown(
            config, swa, n_ctx = 8192, llama_extra_args = ["--no-mmproj-offload"]
        )
        assert pinned.total_bytes == resident.total_bytes
        assert resident.gpu_bytes - pinned.gpu_bytes == expected + 1024

    def test_a_model_with_no_projector_is_charged_nothing_for_one(self, spec_config, swa):
        assert ri._gguf_memory_breakdown(spec_config, swa, n_ctx = 8192).projector_runtime_bytes == 0


class TestInheritedEnvironment:
    """What the CHILD inherits, which is not always what the panel was told.

    ``_child_spec_env`` is the rule: the launch scrubs LLAMA_ARG_SPEC_* whenever
    Unsloth owns the spec block, and keeps it when the extras do. The projector has no
    such scrub at all, so an inherited one loads even through --no-mmproj.
    """

    @pytest.fixture
    def bare(self, gqa_gguf):
        # No sidecars: anything charged here arrived through the environment.
        return SimpleNamespace(
            identifier = "local/bare",
            gguf_file = gqa_gguf,
            is_gguf = True,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
        )

    @pytest.fixture
    def fixed_files(self, monkeypatch):
        monkeypatch.setattr(ri, "_gguf_resident_file_gb", lambda cfg, **kw: 1.0)

    def test_an_inherited_drafter_is_charged_when_the_extras_own_speculation(
        self, bare, gqa_gguf, tmp_path, monkeypatch, fixed_files
    ):
        drafter = _write_gguf(tmp_path, "qwen3", _GQA_FIELDS, name = "inherited-draft.gguf")
        monkeypatch.setenv("LLAMA_ARG_SPEC_DRAFT_MODEL", drafter)
        # The files term resolves --model-draft against an empty env, so this drafter
        # is in neither its bytes nor, before this, the cache priced on top of them.
        owned = ri._gguf_memory_breakdown(
            bare, gqa_gguf, n_ctx = 32768, llama_extra_args = ["--spec-type", "draft-mtp"]
        )
        assert owned.drafter_runtime_bytes > 0
        assert owned.weights_bytes == _GIB + os.path.getsize(drafter)
        # Without an extras --spec-type the launch scrubs LLAMA_ARG_SPEC_*, so the
        # child never sees it and neither does the estimate.
        scrubbed = ri._gguf_memory_breakdown(bare, gqa_gguf, n_ctx = 32768)
        assert scrubbed.drafter_runtime_bytes == 0
        assert scrubbed.weights_bytes == _GIB

    def test_an_inherited_draft_depth_is_honoured_only_when_the_child_keeps_it(
        self, bare, gqa_gguf, monkeypatch
    ):
        monkeypatch.setenv("LLAMA_ARG_SPEC_DRAFT_N_MAX", "7")
        owned = ["--spec-type", "draft-mtp"]
        # The loader reads the env twin before falling back to the build's default.
        assert ri._estimate_draft_n_max(bare, gqa_gguf, requested = None, extras = owned) == 7
        # Scrubbed when Unsloth owns the block, so the platform default stands.
        assert ri._estimate_draft_n_max(bare, gqa_gguf, requested = None, extras = []) in (2, 3)
        # A flag still beats the environment, as it does at launch.
        assert (
            ri._estimate_draft_n_max(
                bare, gqa_gguf, requested = None, extras = [*owned, "--spec-draft-n-max", "5"]
            )
            == 5
        )

    def test_an_inherited_projector_is_charged_like_a_configured_one(
        self, bare, gqa_gguf, tmp_path, monkeypatch
    ):
        projector = tmp_path / "inherited-mmproj.gguf"
        projector.write_bytes(b"\0" * 4096)
        monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(projector))
        monkeypatch.setattr(ri, "_gguf_resident_file_gb", lambda cfg, **kw: 4.0)
        real_size = ri.LlamaCppBackend._get_gguf_size_bytes
        monkeypatch.setattr(
            ri.LlamaCppBackend,
            "_get_gguf_size_bytes",
            staticmethod(
                lambda path: 3 * _GIB
                if os.path.basename(str(path)) == "model.gguf"
                else real_size(path)
            ),
        )
        expected = int(4096 * (ri.LlamaCppBackend._MMPROJ_VRAM_SAFETY - 1.0))
        resident = ri._gguf_memory_breakdown(bare, gqa_gguf, n_ctx = 8192)
        assert resident.projector_runtime_bytes == expected
        # Placement too: the flag and the inherited variable move file and buffers
        # alike, since _resolved_mmproj_offload reads both.
        by_flag = ri._gguf_memory_breakdown(
            bare, gqa_gguf, n_ctx = 8192, llama_extra_args = ["--no-mmproj-offload"]
        )
        assert resident.gpu_bytes - by_flag.gpu_bytes == expected + 4096
        monkeypatch.setenv("LLAMA_ARG_MMPROJ_OFFLOAD", "0")
        assert ri._gguf_memory_breakdown(bare, gqa_gguf, n_ctx = 8192).gpu_bytes == by_flag.gpu_bytes

    def test_the_vulkan_pool_comes_from_the_probed_snapshot(self, monkeypatch):
        # _effective_gpu_count counts CUDA devices, so on a Vulkan build it sees none
        # and a multi-GPU tensor launch was priced with one device's buffers. The
        # inventory /api/system already probed answers instead, read and never
        # refreshed: probing it costs a subprocess and this runs on every keystroke.
        fake_main = _types.ModuleType("main")
        monkeypatch.setitem(sys.modules, "main", fake_main)
        monkeypatch.setattr(ri.LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda *a: True))

        def snapshot(count: int):
            fake_main._system_gpu_cache = (
                0.0,
                ({}, {"devices": [{"index": i} for i in range(count)]}),
            )

        snapshot(3)
        assert len(ri._cached_inference_devices()) == 3
        assert (
            ri._guard_device_count(None, ri._cached_inference_devices(), tensor_parallel = True) == 3
        )
        assert ri._tensor_split_possible(None) is True
        snapshot(1)
        assert (
            ri._guard_device_count(None, ri._cached_inference_devices(), tensor_parallel = True) == 1
        )
        # One Vulkan device is a downgrade at launch, so it is one here too.
        assert ri._tensor_split_possible(None) is False
        # Nothing probed yet: unknown, and unknown must not become a downgrade.
        fake_main._system_gpu_cache = None
        assert ri._cached_inference_devices() is None
        assert ri._tensor_split_possible(None) is True


class TestManualNormalizationAndRemoteDrafters:
    """Two more places the panel described a command the loader would not send."""

    @pytest.fixture
    def bare(self, gqa_gguf):
        return SimpleNamespace(
            identifier = "local/bare",
            gguf_file = gqa_gguf,
            is_gguf = True,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
        )

    def test_manual_prices_the_stripped_extras(self, bare, gqa_gguf, monkeypatch):
        # -ncmoe pushes into tensor_buft_overrides, which turns pipeline parallelism
        # off and drops the context-buffer multiplier. Manual strips the flag before
        # launch, so the multiplier survives and the panel has to say so; Auto keeps
        # the flag, so there it does not.
        monkeypatch.setattr(ri, "_gguf_resident_file_gb", lambda cfg, **kw: 1.0)
        priced = dict(n_ctx = 32768, gpu_layers = 99, n_devices = 2)

        def compute(mode: str, extras):
            return ri._gguf_memory_breakdown(
                bare, gqa_gguf, gpu_memory_mode = mode, llama_extra_args = extras, **priced
            ).compute_bytes

        moe = ["--n-cpu-moe", "8"]
        assert compute("manual", moe) == compute("manual", None)
        assert compute("auto", moe) < compute("auto", None)

    def test_a_remote_drafter_is_declared_unsized_rather_than_priced_at_zero(
        self, bare, gqa_gguf, monkeypatch
    ):
        # --spec-draft-hf names a repository. Its weights are charged by the files term,
        # but its cache cannot be read off a header that is not on this disk, and a
        # confident total missing a context-scaled cache is the one answer this row
        # must never give.
        from core.inference.llama_cpp import _extra_args_draft_offloaded_to_cpu as pinned

        def _files(
            cfg,
            *,
            llama_extra_args = None,
            **kw,
        ):
            args = list(llama_extra_args or ())
            remote = "--spec-draft-hf" in args
            return 1.0 + (2.0 if remote and not pinned(args) else 0.0)

        monkeypatch.setattr(ri, "_gguf_resident_file_gb", _files)
        remote = ri._gguf_memory_breakdown(
            bare, gqa_gguf, n_ctx = 32768, llama_extra_args = ["--spec-draft-hf", "org/drafter-GGUF"]
        )
        assert remote.drafter_kv_unsized is True
        # The weights are still counted, so the floor is as high as it can be made.
        assert remote.weights_bytes == 3 * _GIB
        assert remote.drafter_runtime_bytes == 0
        # And a load with no drafter at all is not flagged.
        assert ri._gguf_memory_breakdown(bare, gqa_gguf, n_ctx = 32768).drafter_kv_unsized is False

    def test_the_route_carries_the_unsized_flag(self, bare, gqa_gguf, monkeypatch):
        from core.inference.llama_cpp import _extra_args_draft_offloaded_to_cpu as pinned

        monkeypatch.setattr(ri, "_cached_estimate_config", lambda *a, **kw: bare)

        def _files(
            cfg,
            *,
            llama_extra_args = None,
            **kw,
        ):
            # Mirrors the real term, including that a CPU-pinned drafter is dropped:
            # the breakdown identifies the drafter's bytes by re-pricing with that pin.
            args = list(llama_extra_args or ())
            return 1.0 + (2.0 if "--spec-draft-hf" in args and not pinned(args) else 0.0)

        monkeypatch.setattr(ri, "_gguf_resident_file_gb", _files)
        resp = _estimate(
            model_path = gqa_gguf,
            n_ctx = 8192,
            llama_extra_args = ["--spec-draft-hf", "org/drafter-GGUF"],
        )
        assert resp.available is True
        # available, but not silent about what is missing from the total.
        assert resp.drafter_kv_unsized is True


class TestSpeculationOffChargesNoDrafter:
    """Modes whose launch never emits ``--model-draft`` must not be billed for one.

    ``_build_speculative_flags`` returns out of three before reaching the flag: "off"
    immediately, "ngram-simple" and "ngram" after their ``--spec-type``. The resident-
    file resolution appended ``gguf_mtp_file`` for every mode that was not DSpark or
    DFlash, so selecting OFF could ADD gigabytes -- the opposite of what the control
    does. Driven through the real resolution: the breakdown's own tests stub the files
    term wholesale and would pass either way.
    """

    @pytest.fixture
    def config_with_a_sidecar(self, tmp_path):
        target = _write_gguf(tmp_path, "qwen3", _GQA_FIELDS, name = "target.gguf")
        sidecar = tmp_path / "mtp.gguf"
        sidecar.write_bytes(Path(target).read_bytes())
        return SimpleNamespace(
            identifier = "local/target",
            gguf_file = target,
            is_gguf = True,
            gguf_variant = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = str(sidecar),
            gguf_dspark_file = None,
            gguf_dflash_file = None,
        )

    @pytest.mark.parametrize("mode", ["off", "ngram", "ngram-simple"])
    def test_a_drafterless_mode_is_not_charged_for_the_sidecar(self, config_with_a_sidecar, mode):
        charged = ri._gguf_resident_file_gb(config_with_a_sidecar, speculative_type = "mtp")
        quiet = ri._gguf_resident_file_gb(config_with_a_sidecar, speculative_type = mode)
        assert charged is not None and quiet is not None
        # The sidecar is a copy of the target, so its absence is unmissable.
        assert quiet < charged
        assert quiet == pytest.approx(charged / 2, rel = 0.05)

    def test_a_drafting_mode_still_charges_it(self, config_with_a_sidecar):
        """The guard against over-correcting: MTP must keep paying for its drafter."""
        with_mtp = ri._gguf_resident_file_gb(config_with_a_sidecar, speculative_type = "mtp")
        auto = ri._gguf_resident_file_gb(config_with_a_sidecar, speculative_type = "auto")
        assert with_mtp is not None and auto is not None
        assert with_mtp == auto


class TestAProjectorOverrideIsTheOneCharged:
    """``--mmproj`` in Advanced Arguments last-wins, so it is the file that opens.

    ``load_model`` emits Studio's resolved projector and appends the extras after it,
    so the child opens the user's. Both the resident-file total and the encoder's
    runtime allowance read only the configured projector, which billed a file the
    launch never opens and let a possibly much larger custom one through free.
    """

    @pytest.fixture
    def vision_config(self, tmp_path):
        target = _write_gguf(tmp_path, "qwen3", _GQA_FIELDS, name = "target.gguf")
        small = tmp_path / "mmproj-small.gguf"
        small.write_bytes(b"s" * 4096)
        big = tmp_path / "mmproj-big.gguf"
        big.write_bytes(b"b" * (4096 * 8))
        config = SimpleNamespace(
            identifier = "local/vision",
            gguf_file = target,
            is_gguf = True,
            gguf_variant = None,
            gguf_mmproj_file = str(small),
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
        )
        return config, str(small), str(big)

    def test_the_override_replaces_the_configured_projector_in_the_files(self, vision_config):
        config, small, big = vision_config
        configured = ri._gguf_resident_file_gb(config)
        overridden = ri._gguf_resident_file_gb(config, llama_extra_args = ["--mmproj", big])
        assert configured is not None and overridden is not None
        # The override is 8x the configured one, so the total has to move with it.
        assert overridden > configured
        delta = (overridden - configured) * 1024**3
        assert delta == pytest.approx(Path(big).stat().st_size - Path(small).stat().st_size, abs = 8)

    def test_the_overrides_encoder_runtime_is_charged_too(self, vision_config):
        config, _small, big = vision_config
        charged = ri._charged_projector_bytes(config, ["--mmproj", big], False)
        assert charged == Path(big).stat().st_size

    def test_the_short_flag_is_read_the_same_way(self, vision_config):
        config, _small, big = vision_config
        assert ri._charged_projector_bytes(config, ["-mm", big], False) == Path(big).stat().st_size

    def test_no_override_still_charges_the_configured_projector(self, vision_config):
        config, small, _big = vision_config
        assert ri._charged_projector_bytes(config, [], False) == Path(small).stat().st_size


class TestAnEmbeddedMtpHeadIsPriced:
    """A NextN head is a drafter with no file, and it still allocates a draft cache.

    Nothing is charged in the weights for it, so `_charged_drafter_path` returns None
    and the entire runtime-sizing block was skipped. `_estimate_mtp_overhead_bytes`
    documents this case explicitly (``drafter_path=None``, ``draft_weights_bytes=0``)
    and sizes the head from ``nextn_predict_layers``, so the allocation was knowable
    and simply not asked for.
    """

    @pytest.fixture
    def nextn_model(self, tmp_path):
        gguf = _write_gguf(
            tmp_path,
            "qwen3",
            {**_GQA_FIELDS, "context_length": 262144, "nextn_predict_layers": 2},
            name = "nextn.gguf",
        )
        return gguf, SimpleNamespace(
            identifier = "local/nextn",
            gguf_file = gguf,
            is_gguf = True,
            gguf_variant = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
        )

    def test_the_head_is_charged_when_speculation_can_engage(self, nextn_model):
        gguf, config = nextn_model
        priced = ri._gguf_memory_breakdown(config, gguf, n_ctx = 131072)
        assert priced is not None
        assert priced.drafter_runtime_bytes > 0
        # And it is inside the total, which is the figure the fit verdict reads.
        assert priced.total_bytes > priced.weights_bytes + priced.kv_bytes

    def test_turning_speculation_off_charges_no_head(self, nextn_model):
        gguf, config = nextn_model
        off = ri._gguf_memory_breakdown(config, gguf, n_ctx = 131072, speculative_type = "off")
        assert off is not None
        assert off.drafter_runtime_bytes == 0

    def test_a_model_without_a_head_is_unaffected(self, tmp_path):
        """The guard against charging every model: no NextN key, nothing priced."""
        plain = _write_gguf(tmp_path, "qwen3", {**_GQA_FIELDS, "context_length": 262144})
        config = SimpleNamespace(
            identifier = "local/plain",
            gguf_file = plain,
            is_gguf = True,
            gguf_variant = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
        )
        priced = ri._gguf_memory_breakdown(config, plain, n_ctx = 131072)
        assert priced is not None
        assert priced.drafter_runtime_bytes == 0

    def test_a_draft_pin_does_not_move_the_embedded_head(self, nextn_model):
        """--spec-draft-ngl 0 does not switch the head off, and does not move it either.

        Both flags carry ``params.speculative.n_gpu_layers`` / ``.devices``, which
        llama.cpp copies in only on the ``has_draft`` path; an embedded head has no
        draft model to load and takes ``llama_init_from_model(model_tgt)``, keeping the
        target's placement, as ``_extra_args_draft_offloaded_to_cpu`` documents and
        ``_draft_cpu_no_embedded`` enforces. So it is still allocated AND still on the
        card: gating the sizing on the pin dropped it from both figures, placing it in
        host RAM dropped it from the GPU one, and both turn an overflow into a fit.
        """
        gguf, config = nextn_model
        unpinned = ri._gguf_memory_breakdown(config, gguf, n_ctx = 131072)
        assert unpinned is not None
        assert unpinned.drafter_runtime_gpu_bytes > 0
        for pin in (["--spec-draft-ngl", "0"], ["--spec-draft-device", "cpu"]):
            pinned = ri._gguf_memory_breakdown(config, gguf, n_ctx = 131072, llama_extra_args = pin)
            assert pinned is not None, pin
            assert pinned.drafter_runtime_bytes > 0, pin
            # Unmoved: the same bytes on the same device as with no pin at all.
            assert pinned.drafter_runtime_gpu_bytes == unpinned.drafter_runtime_gpu_bytes, pin
            assert pinned.drafter_runtime_bytes == unpinned.drafter_runtime_bytes, pin

    def test_a_cpu_placed_target_takes_the_embedded_head_with_it(self, nextn_model):
        """The other half of the same rule: it follows the target DOWN as well as up.

        An embedded head has no file, so host_drafter_bytes is 0 and the drafter is
        GPU-resident by default. That default is only right while the target is: the
        head is part of the target's tensors and llama.cpp gives it the target's
        context, so at --gpu-layers 0 its cache is in host RAM. Charging it to VRAM
        there raised a multi-gigabyte warning about a card the load never touches.
        """
        gguf, config = nextn_model
        on_cpu = ri._gguf_memory_breakdown(
            config, gguf, n_ctx = 131072, gpu_memory_mode = "manual", gpu_layers = 0
        )
        assert on_cpu is not None
        # Still allocated, and still in the aggregate figure -- just not on the card.
        assert on_cpu.drafter_runtime_bytes > 0
        assert on_cpu.drafter_runtime_gpu_bytes == 0


class TestACpuOnlyHostShowsNoGpuFootprint:
    """An Auto load on a machine with no GPU runs in host RAM, and must read that way.

    CPU placement was detected only from an explicit --device or its env twin, so a
    plain Auto request on a CPU-only Linux or Windows box was priced fully GPU-resident
    -- a multi-gigabyte GPU figure against a capacity of zero.

    The evidence has to be a probe that RAN. An unfilled snapshot is not absence, and
    the CUDA count is zero on every Vulkan host, so neither may move the weights.
    """

    @pytest.fixture
    def priced(self, tmp_path):
        gguf = _write_gguf(tmp_path, "qwen3", {**_GQA_FIELDS, "context_length": 262144})
        return gguf, SimpleNamespace(
            identifier = "local/cpu",
            gguf_file = gguf,
            is_gguf = True,
            gguf_variant = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
        )

    def test_a_probed_empty_inventory_puts_everything_in_host_ram(self, priced, monkeypatch):
        gguf, config = priced
        monkeypatch.setattr(ri, "_cached_inference_devices", lambda: [])
        out = ri._gguf_memory_breakdown(config, gguf, n_ctx = 32768)
        assert out is not None
        assert out.gpu_bytes == 0
        # Still a real load, just not on a card.
        assert out.total_bytes > 0
        assert out.kv_on_gpu is False

    def test_an_unprobed_snapshot_is_not_evidence_of_absence(self, priced, monkeypatch):
        gguf, config = priced
        monkeypatch.setattr(ri, "_cached_inference_devices", lambda: None)
        out = ri._gguf_memory_breakdown(config, gguf, n_ctx = 32768)
        assert out is not None
        assert out.gpu_bytes > 0

    def test_a_probed_inventory_with_a_card_is_unaffected(self, priced, monkeypatch):
        gguf, config = priced
        monkeypatch.setattr(ri, "_cached_inference_devices", lambda: [(0, 0, 0)])
        out = ri._gguf_memory_breakdown(config, gguf, n_ctx = 32768)
        assert out is not None
        assert out.gpu_bytes > 0


class TestAnInheritedContextIsPriced:
    """``LLAMA_ARG_CTX_SIZE`` is what the child runs at when nothing else sets a length.

    The launch drops an inherited context only when it is ZERO and the auto-layers path
    needs --fit to run; a positive one is left alone as a legitimate way to set the
    length. The estimate fell straight through to the header's native context, so a 4k
    environment on a 262k model priced the KV cache 64x too large and refused loads that
    run comfortably.
    """

    @pytest.fixture
    def wide(self, tmp_path):
        gguf = _write_gguf(tmp_path, "qwen3", {**_GQA_FIELDS, "context_length": 262144})
        return gguf, SimpleNamespace(
            identifier = "local/wide",
            gguf_file = gguf,
            is_gguf = True,
            gguf_variant = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
        )

    def test_the_environment_length_is_priced_when_nothing_else_sets_one(self, wide, monkeypatch):
        gguf, config = wide
        monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", "4096")
        out = ri._gguf_memory_breakdown(config, gguf, n_ctx = 0)
        assert out is not None
        assert out.n_ctx == 4096

    def test_the_native_context_is_still_the_last_resort(self, wide, monkeypatch):
        gguf, config = wide
        monkeypatch.delenv("LLAMA_ARG_CTX_SIZE", raising = False)
        out = ri._gguf_memory_breakdown(config, gguf, n_ctx = 0)
        assert out is not None
        assert out.n_ctx == 262144

    @pytest.mark.parametrize("value", ["0", "", "not-a-number", "-5"])
    def test_a_useless_environment_value_falls_through(self, wide, monkeypatch, value):
        """Zero is the one the launch itself drops, and the rest are not lengths."""
        gguf, config = wide
        monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", value)
        out = ri._gguf_memory_breakdown(config, gguf, n_ctx = 0)
        assert out is not None
        assert out.n_ctx == 262144

    @pytest.mark.parametrize("flag", [["-c", "0"], ["--ctx-size", "0"], ["--ctx-size=0"]])
    def test_an_explicit_zero_asks_for_native_and_beats_the_environment(
        self, wide, monkeypatch, flag
    ):
        """ "-c 0" REQUESTS the native context; it is not the absence of a request.

        llama.cpp parses the environment before argv, so the explicit zero wins at the
        child. Folding it in with "nothing was set" let an inherited 4k answer for a
        launch that opens at the header's 262k -- the KV cache understated 64x, in the
        direction that says "fits".
        """
        gguf, config = wide
        monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", "4096")
        out = ri._gguf_memory_breakdown(config, gguf, n_ctx = 0, llama_extra_args = flag)
        assert out is not None
        assert out.n_ctx == 262144, flag

    def test_without_that_zero_the_environment_still_answers(self, wide, monkeypatch):
        """The guard against over-correcting: only an explicit zero bypasses it."""
        gguf, config = wide
        monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", "4096")
        assert ri._gguf_memory_breakdown(config, gguf, n_ctx = 0).n_ctx == 4096

    def test_an_explicit_panel_length_still_wins(self, wide, monkeypatch):
        gguf, config = wide
        monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", "4096")
        out = ri._gguf_memory_breakdown(config, gguf, n_ctx = 8192)
        assert out is not None
        assert out.n_ctx == 8192


class TestTheResolutionTriesOfflineFirst:
    """A repo already on this disk is priced without asking the Hub.

    The on-disk gate has established the files are here and everything downstream reads
    local files, but `from_identifier` still ran `detect_gguf_model_remote` and
    `list_gguf_variants`, an `hf_model_info` each. On a route the panel fires on every
    settings change that is two Hub round trips per cache miss.
    """

    def test_the_first_attempt_runs_under_forced_offline(self, tmp_path, monkeypatch):
        gguf = _write_gguf(tmp_path, "qwen3", _GQA_FIELDS)
        seen = []

        import contextlib

        @contextlib.contextmanager
        def _forced():
            seen.append("offline")
            yield True

        monkeypatch.setattr(ri, "_estimate_target_is_on_this_disk", lambda ident: True)
        monkeypatch.setitem(
            __import__("sys").modules,
            "utils.utils",
            SimpleNamespace(force_hf_offline = _forced),
        )
        cfg = SimpleNamespace(identifier = "org/cached", gguf_file = gguf, is_gguf = True)
        monkeypatch.setattr(ri.ModelConfig, "from_identifier", staticmethod(lambda **kw: cfg))
        ri._estimate_config_cache.clear()
        out = ri._cached_estimate_config("org/cached", None, None, False)
        assert out is cfg
        assert seen == ["offline"], "the offline window was not entered"

    def test_an_offline_failure_falls_back_rather_than_blanking_the_row(
        self, tmp_path, monkeypatch
    ):
        """A slow right answer beats a fast blank row."""
        gguf = _write_gguf(tmp_path, "qwen3", _GQA_FIELDS)
        calls = []
        cfg = SimpleNamespace(identifier = "org/cached", gguf_file = gguf, is_gguf = True)

        def _from_identifier(**kw):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("needs the hub")
            return cfg

        import contextlib

        @contextlib.contextmanager
        def _forced():
            yield True

        monkeypatch.setattr(ri, "_estimate_target_is_on_this_disk", lambda ident: True)
        monkeypatch.setitem(
            __import__("sys").modules,
            "utils.utils",
            SimpleNamespace(force_hf_offline = _forced),
        )
        monkeypatch.setattr(ri.ModelConfig, "from_identifier", staticmethod(_from_identifier))
        ri._estimate_config_cache.clear()
        out = ri._cached_estimate_config("org/cached", None, None, False)
        assert out is cfg
        assert len(calls) == 2, "the online retry did not run"


class TestTheEmbeddedHeadIsChargedOnlyWhenItEngages:
    """Every no-MTP outcome ``_build_speculative_flags`` has, asked of the estimate.

    An embedded head has no file, so nothing about it is visible in the weights and an
    over-charge is invisible too: it just makes the row read several GB high and can
    refuse a load that runs. These are the launches that allocate no head at all.
    """

    def _config(self, gguf, identifier):
        return SimpleNamespace(
            identifier = identifier,
            gguf_file = gguf,
            is_gguf = True,
            gguf_variant = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
        )

    @pytest.fixture
    def head(self, tmp_path):
        return _write_gguf(
            tmp_path,
            "qwen3",
            {**_GQA_FIELDS, "context_length": 262144, "nextn_predict_layers": 2},
            name = "head.gguf",
        )

    def test_a_sub_3b_head_is_dropped_by_auto(self, head):
        """Auto falls back to ngram-mod there, because MTP regresses below 3B."""
        big = ri._gguf_memory_breakdown(self._config(head, "org/Qwen3-8B"), head, n_ctx = 131072)
        small = ri._gguf_memory_breakdown(self._config(head, "org/Qwen3-1.7B"), head, n_ctx = 131072)
        assert big.drafter_runtime_bytes > 0
        assert small.drafter_runtime_bytes == 0

    def test_forcing_mtp_overrides_the_size_drop(self, head):
        """The drop is Auto's judgement; an explicit request is not overruled."""
        forced = ri._gguf_memory_breakdown(
            self._config(head, "org/Qwen3-1.7B"), head, n_ctx = 131072, speculative_type = "mtp"
        )
        assert forced.drafter_runtime_bytes > 0

    @pytest.mark.parametrize("mode", ["dspark", "dflash"])
    def test_a_sidecar_mode_never_charges_the_embedded_head(self, head, mode):
        """Those either open their own sidecar, charged as a file, or --spec-default."""
        out = ri._gguf_memory_breakdown(
            self._config(head, "org/Qwen3-8B"), head, n_ctx = 131072, speculative_type = mode
        )
        assert out.drafter_runtime_bytes == 0

    @pytest.mark.parametrize("mode", [None, "mtp"])
    def test_a_binary_that_cannot_run_mtp_is_not_charged_for_it(self, head, monkeypatch, mode):
        """No mtp_token means --spec-default launches, whatever the mode asked for.

        Unlike the size and MLA drops this is a hard incompatibility, so it applies to
        a forced request too.
        """
        monkeypatch.setattr(
            ri.LlamaCppBackend,
            "probe_server_capabilities",
            classmethod(lambda cls, *a, **k: {"found": True, "mtp_token": None}),
        )
        out = ri._gguf_memory_breakdown(
            self._config(head, "org/Qwen3-8B"), head, n_ctx = 131072, speculative_type = mode
        )
        assert out.drafter_runtime_bytes == 0

    def test_an_unanswerable_probe_keeps_the_charge(self, head, monkeypatch):
        """A probe that could not run is not evidence the build lacks the flag."""
        monkeypatch.setattr(
            ri.LlamaCppBackend,
            "probe_server_capabilities",
            classmethod(lambda cls, *a, **k: {"found": False, "mtp_token": None}),
        )
        out = ri._gguf_memory_breakdown(self._config(head, "org/Qwen3-8B"), head, n_ctx = 131072)
        assert out.drafter_runtime_bytes > 0

    def test_extras_asking_for_another_mode_are_left_alone(self, head):
        """Studio emits no spec block of its own once --spec-type is in the extras."""
        out = ri._gguf_memory_breakdown(
            self._config(head, "org/Qwen3-8B"),
            head,
            n_ctx = 131072,
            llama_extra_args = ["--spec-type", "ngram-mod"],
        )
        assert out.drafter_runtime_bytes == 0

    @pytest.mark.parametrize("spec", ["draft-mtp", "mtp"])
    def test_extras_asking_for_mtp_engage_the_embedded_head(self, head, spec):
        """The child honours the extras, and on a NextN GGUF that IS the embedded head.

        There is no drafter file, so nothing in the weights would ever hint at it: the
        draft cache and the target-side state were simply absent from the total.
        """
        out = ri._gguf_memory_breakdown(
            self._config(head, "org/Qwen3-8B"),
            head,
            n_ctx = 131072,
            llama_extra_args = ["--spec-type", spec],
        )
        assert out.drafter_runtime_bytes > 0

    def test_extras_asking_for_mtp_on_a_sub_3b_model_still_engage_it(self, head):
        """An explicit request is not second-guessed by Auto's size rule."""
        out = ri._gguf_memory_breakdown(
            self._config(head, "org/Qwen3-1.7B"),
            head,
            n_ctx = 131072,
            llama_extra_args = ["--spec-type", "draft-mtp"],
        )
        assert out.drafter_runtime_bytes > 0


class TestTheCpuOnlyCheckIsActuallyReachable:
    """Driven through the real snapshot, not through a stubbed helper.

    The first version of the CPU-only fix was inert: `_cached_inference_devices` ended
    in `or None`, so the empty list a probed CPU-only host produces arrived as the same
    value as a snapshot nobody had filled, and the check for it could never fire. Its
    unit tests passed because they replaced `_cached_inference_devices` itself, which
    is the one thing that could not be wrong. These stub the snapshot instead.
    """

    @pytest.fixture
    def snapshot(self, monkeypatch):
        import sys as _sys
        def _set(devices):
            main = SimpleNamespace(_system_gpu_cache = (0.0, (None, {"devices": devices})))
            monkeypatch.setitem(_sys.modules, "main", main)

        return _set

    def test_a_probed_cpu_only_host_reads_as_empty_not_unknown(self, snapshot):
        snapshot([])
        assert ri._cached_inference_devices() == []
        assert ri._estimate_host_has_no_gpu() is True

    def test_a_probed_host_with_cards_reads_as_those_cards(self, snapshot):
        snapshot([{"index": 0}, {"index": 1}])
        assert len(ri._cached_inference_devices()) == 2
        assert ri._estimate_host_has_no_gpu() is False

    def test_an_unfilled_snapshot_is_still_unknown(self, monkeypatch):
        import sys as _sys

        monkeypatch.setitem(_sys.modules, "main", SimpleNamespace())
        assert ri._cached_inference_devices() is None
        assert ri._estimate_host_has_no_gpu() is False

    def test_a_cpu_only_host_cannot_take_a_tensor_split(self, snapshot, monkeypatch):
        """The other caller of the same snapshot, which was also reading True here."""
        snapshot([])
        monkeypatch.setattr(ri.LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda *a: True))
        assert ri._tensor_split_possible(None) is False

    def test_the_whole_footprint_lands_in_host_ram_end_to_end(self, snapshot, tmp_path):
        gguf = _write_gguf(tmp_path, "qwen3", {**_GQA_FIELDS, "context_length": 262144})
        config = SimpleNamespace(
            identifier = "local/cpu",
            gguf_file = gguf,
            is_gguf = True,
            gguf_variant = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
        )
        snapshot([])
        out = ri._gguf_memory_breakdown(config, gguf, n_ctx = 32768)
        assert out is not None
        assert out.gpu_bytes == 0
        assert out.total_bytes > 0


class TestDraftCacheTypePrecedence:
    """extras beat the panel field beat the inherited environment.

    That is launch order: the field goes on argv, argv beats the LLAMA_ARG_* twin, and
    the extras are appended after argv and beat both. `_extra_args_draft_cache_types`
    falls back to the env by DEFAULT, so asking it once and then `or`-ing the field on
    gave the environment precedence over the control -- an inherited q4 priced against
    an f16 the child would really allocate, undercounting a context-scaled cache.
    """

    @pytest.fixture
    def spec(self, tmp_path, monkeypatch):
        target = _write_gguf(
            tmp_path, "qwen3", {**_GQA_FIELDS, "context_length": 262144}, name = "t.gguf"
        )
        drafter = tmp_path / "mtp.gguf"
        drafter.write_bytes(Path(target).read_bytes())
        drafter_bytes = drafter.stat().st_size

        # Varies with the draft pin, like the real one: the breakdown recovers the
        # drafter's size by re-pricing with the pin flipped, so a constant stub charges
        # no drafter at all and every assertion below would compare 0 with 0.
        def _files(
            cfg,
            *,
            llama_extra_args = None,
            **kw,
        ):
            pinned = _draft_on_cpu(list(llama_extra_args or ()))
            return 1.0 + (0.0 if pinned else drafter_bytes / 1024**3)

        monkeypatch.setattr(ri, "_gguf_resident_file_gb", _files)
        return target, SimpleNamespace(
            identifier = "org/Qwen3-8B",
            gguf_file = target,
            is_gguf = True,
            gguf_variant = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = str(drafter),
            gguf_dspark_file = None,
            gguf_dflash_file = None,
        )

    def _bytes(
        self,
        spec,
        monkeypatch,
        *,
        env = None,
        field = None,
        extras = None,
    ):
        target, config = spec
        for key in ("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K", "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V"):
            monkeypatch.delenv(key, raising = False)
        if env:
            monkeypatch.setenv("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K", env)
            monkeypatch.setenv("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V", env)
        out = ri._gguf_memory_breakdown(
            config,
            target,
            n_ctx = 131072,
            spec_draft_cache_type = field,
            llama_extra_args = extras,
        )
        assert out is not None
        return out.drafter_runtime_bytes

    def test_the_panel_field_beats_an_inherited_value(self, spec, monkeypatch):
        # q4_0 is a quarter of f16, so the wrong precedence is unmissable.
        inherited_only = self._bytes(spec, monkeypatch, env = "q4_0")
        field_wins = self._bytes(spec, monkeypatch, env = "q4_0", field = "f16")
        assert field_wins > inherited_only

    def test_the_inherited_value_is_used_when_the_field_is_blank(self, spec, monkeypatch):
        plain = self._bytes(spec, monkeypatch)
        inherited = self._bytes(spec, monkeypatch, env = "q4_0")
        assert inherited < plain

    def test_the_extras_beat_both(self, spec, monkeypatch):
        field_only = self._bytes(spec, monkeypatch, env = "q4_0", field = "f16")
        extras_win = self._bytes(
            spec,
            monkeypatch,
            env = "q4_0",
            field = "f16",
            extras = ["--cache-type-k-draft", "q4_0", "--cache-type-v-draft", "q4_0"],
        )
        assert extras_win < field_only


class TestTheTensorSplitLatchesAreHonoured:
    """load_model consults two in-process latches before it plans; so must the price.

    Both silently turn the launch into a layer split, and the two shapes cost different
    amounts: tensor replicates a flat buffer per device, a layer split multiplies the
    context-linear term instead. On two cards that is gigabytes apart, which is enough
    to move the verdict, so pricing tensor after a latch is set is a wrong number for a
    launch that will not happen.
    """

    @pytest.fixture
    def two_card(self, tmp_path, monkeypatch):
        gguf = _write_gguf(tmp_path, "qwen3", {**_GQA_FIELDS, "context_length": 262144})
        config = SimpleNamespace(
            identifier = "local/tensor",
            gguf_file = gguf,
            is_gguf = True,
            gguf_variant = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
        )
        monkeypatch.setattr(
            ri.LlamaCppBackend,
            "_find_llama_server_binary",
            staticmethod(lambda **kw: "/opt/llama/llama-server"),
        )
        return gguf, config

    def _priced(self, two_card, **kw):
        gguf, config = two_card
        return ri._gguf_memory_breakdown(
            config,
            gguf,
            n_ctx = 32768,
            tensor_parallel = True,
            n_devices = 2,
            tensor_split_possible = True,
            **kw,
        )

    def test_a_binary_that_refused_a_quantized_tensor_cache_is_priced_as_layer(
        self, two_card, monkeypatch
    ):
        as_tensor = self._priced(two_card, cache_type_kv = "q4_0")
        monkeypatch.setattr(
            ri.LlamaCppBackend,
            "_tensor_quant_kv_unsupported_binary",
            classmethod(lambda cls, *a, **k: True),
        )
        downgraded = self._priced(two_card, cache_type_kv = "q4_0")
        assert downgraded.compute_bytes != as_tensor.compute_bytes

    def test_a_model_that_aborted_on_tensor_this_session_is_priced_as_layer(
        self, two_card, monkeypatch
    ):
        as_tensor = self._priced(two_card, cache_type_kv = "f16")
        monkeypatch.setattr(
            ri.LlamaCppBackend,
            "_tensor_split_aborts",
            classmethod(lambda cls, *a, **k: True),
        )
        downgraded = self._priced(two_card, cache_type_kv = "f16")
        assert downgraded.compute_bytes != as_tensor.compute_bytes

    def test_no_latch_leaves_tensor_pricing_alone(self, two_card, monkeypatch):
        """The guard against the check firing on every load."""
        monkeypatch.setattr(
            ri.LlamaCppBackend,
            "_tensor_quant_kv_unsupported_binary",
            classmethod(lambda cls, *a, **k: False),
        )
        monkeypatch.setattr(
            ri.LlamaCppBackend,
            "_tensor_split_aborts",
            classmethod(lambda cls, *a, **k: False),
        )
        assert ri._tensor_latches_allow_a_split(two_card[0], two_card[1], "f16", None) is True

    def test_an_unresolvable_binary_fails_open(self, two_card, monkeypatch):
        monkeypatch.setattr(
            ri.LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda **kw: None)
        )
        assert ri._tensor_latches_allow_a_split(two_card[0], two_card[1], "f16", None) is True


class TestInheritedRemoteFilesAreMarkedUnsized:
    """A file the child fetches, that this route may not fetch to weigh, is a floor.

    Both arrive through the environment and both are preserved for the launch:
    LLAMA_ARG_SPEC_DRAFT_HF_REPO (kept by _child_spec_env once the extras own the spec
    block) and LLAMA_ARG_MMPROJ_URL. Sizing either needs a network call this route is
    documented not to make, so the honest answer is a marked lower bound rather than a
    silently missing multi-gigabyte file.
    """

    @pytest.fixture
    def plain(self, tmp_path, monkeypatch):
        for key in (
            "LLAMA_ARG_SPEC_DRAFT_HF_REPO",
            "LLAMA_ARG_HFD_REPO",
            "LLAMA_ARG_MMPROJ_URL",
        ):
            monkeypatch.delenv(key, raising = False)
        gguf = _write_gguf(tmp_path, "qwen3", {**_GQA_FIELDS, "context_length": 262144})
        return gguf, SimpleNamespace(
            identifier = "local/plain",
            gguf_file = gguf,
            is_gguf = True,
            gguf_variant = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
        )

    def test_nothing_inherited_is_not_marked(self, plain):
        gguf, config = plain
        assert ri._gguf_memory_breakdown(config, gguf, n_ctx = 8192).drafter_kv_unsized is False

    @pytest.mark.parametrize("key", ["LLAMA_ARG_SPEC_DRAFT_HF_REPO", "LLAMA_ARG_HFD_REPO"])
    def test_an_inherited_remote_drafter_repo_is_marked(self, plain, monkeypatch, key):
        gguf, config = plain
        monkeypatch.setenv(key, "org/drafter-GGUF")
        out = ri._gguf_memory_breakdown(
            config, gguf, n_ctx = 8192, llama_extra_args = ["--spec-type", "draft-mtp"]
        )
        assert out.drafter_kv_unsized is True
        # Still an answer, not a refusal: a floor beats a blank row.
        assert out.total_bytes > 0

    def test_an_inherited_projector_url_is_marked(self, plain, monkeypatch):
        gguf, config = plain
        monkeypatch.setenv("LLAMA_ARG_MMPROJ_URL", "https://example.invalid/mmproj.gguf")
        assert ri._gguf_memory_breakdown(config, gguf, n_ctx = 8192).drafter_kv_unsized is True

    def test_a_projector_url_is_irrelevant_with_vision_off(self, plain, monkeypatch):
        gguf, config = plain
        monkeypatch.setenv("LLAMA_ARG_MMPROJ_URL", "https://example.invalid/mmproj.gguf")
        out = ri._gguf_memory_breakdown(config, gguf, n_ctx = 8192, disable_vision = True)
        assert out.drafter_kv_unsized is False


class TestFourMoreLaunchNormalizations:
    """Each is a rule load_model applies before it sizes anything."""

    @pytest.fixture
    def basic(self, tmp_path):
        gguf = _write_gguf(tmp_path, "qwen3", {**_GQA_FIELDS, "context_length": 262144})
        return gguf, SimpleNamespace(
            identifier = "local/basic",
            gguf_file = gguf,
            is_gguf = True,
            gguf_variant = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
        )

    def test_a_gpu_pin_beats_a_stale_device_flag(self, basic):
        """The pin names cards; the launch strips whatever contradicted it.

        Without this, a --device none left in Advanced Arguments made a pinned load
        report no GPU footprint at all, which is the fit verdict inverted.
        """
        gguf, config = basic
        cpu_flag = ["--device", "none"]
        unpinned = ri._gguf_memory_breakdown(config, gguf, n_ctx = 8192, llama_extra_args = cpu_flag)
        pinned = ri._gguf_memory_breakdown(
            config, gguf, n_ctx = 8192, llama_extra_args = cpu_flag, device_pin_governs = True
        )
        assert unpinned.gpu_bytes == 0
        assert pinned.gpu_bytes > 0

    def test_a_remote_drafter_repo_is_not_listed_over_the_network(self, basic, monkeypatch):
        """Sizing it means an hf_model_info per settings change; the marker replaces it."""
        gguf, config = basic
        called = []
        monkeypatch.setattr(
            ri,
            "_remote_drafter_repo_bytes",
            lambda *a, **k: (called.append(1), 1 << 30)[1],
        )
        out = ri._gguf_memory_breakdown(
            config,
            gguf,
            n_ctx = 8192,
            llama_extra_args = ["--spec-draft-hf", "org/drafter-GGUF"],
        )
        assert called == [], "the Hub listing ran behind the panel"
        # Uncharged, but not unmentioned.
        assert out.drafter_kv_unsized is True

    @pytest.mark.parametrize(
        "caps,managed",
        [
            (
                {
                    "found": True,
                    "spec_draft_cache_k_flag": "-ctkd",
                    "spec_draft_cache_v_flag": "-ctvd",
                },
                True,
            ),
            (
                {
                    "found": True,
                    "spec_draft_cache_k_flag": "-ctkd",
                    "spec_draft_cache_v_flag": None,
                },
                False,
            ),
            (
                {
                    "found": True,
                    "spec_draft_cache_k_flag": None,
                    "spec_draft_cache_v_flag": "-ctvd",
                },
                False,
            ),
            (
                {"found": False, "spec_draft_cache_k_flag": None, "spec_draft_cache_v_flag": None},
                True,
            ),
        ],
    )
    def test_the_draft_cache_field_needs_both_flags(self, tmp_path, monkeypatch, caps, managed):
        """The launcher emits the pair or neither, so one flag is not enough."""
        target = _write_gguf(
            tmp_path, "qwen3", {**_GQA_FIELDS, "context_length": 262144}, name = "t2.gguf"
        )
        drafter = tmp_path / "mtp2.gguf"
        drafter.write_bytes(Path(target).read_bytes())
        drafter_bytes = drafter.stat().st_size

        def _files(
            cfg,
            *,
            llama_extra_args = None,
            **kw,
        ):
            pinned = _draft_on_cpu(list(llama_extra_args or ()))
            return 1.0 + (0.0 if pinned else drafter_bytes / 1024**3)

        monkeypatch.setattr(ri, "_gguf_resident_file_gb", _files)
        monkeypatch.setattr(
            ri.LlamaCppBackend,
            "probe_server_capabilities",
            classmethod(lambda cls, *a, **k: {**caps, "mtp_token": "draft-mtp"}),
        )
        config = SimpleNamespace(
            identifier = "org/Qwen3-8B",
            gguf_file = target,
            is_gguf = True,
            gguf_variant = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = str(drafter),
            gguf_dspark_file = None,
            gguf_dflash_file = None,
        )
        priced = ri._gguf_memory_breakdown(
            config, target, n_ctx = 131072, spec_draft_cache_type = "q4_0"
        )
        default = ri._gguf_memory_breakdown(config, target, n_ctx = 131072)
        # q4_0 is a quarter of the default f16, so "managed" is visible in the bytes.
        if managed:
            assert priced.drafter_runtime_bytes < default.drafter_runtime_bytes
        else:
            assert priced.drafter_runtime_bytes == default.drafter_runtime_bytes

    def test_slots_are_clamped_on_a_build_without_kv_unified(self, basic, monkeypatch):
        """load_model drops to one slot there, before it sizes anything."""
        gguf, config = basic
        monkeypatch.setattr(
            ri.LlamaCppBackend,
            "probe_server_capabilities",
            classmethod(lambda cls, *a, **k: {"found": True, "supports_kv_unified": False}),
        )
        out = ri._gguf_memory_breakdown(config, gguf, n_ctx = 8192, n_parallel = 4)
        assert out.n_parallel == 1

    def test_an_unprobed_build_keeps_the_requested_slots(self, basic, monkeypatch):
        """A probe that could not run is not evidence the flag is missing."""
        gguf, config = basic
        monkeypatch.setattr(
            ri.LlamaCppBackend,
            "probe_server_capabilities",
            classmethod(lambda cls, *a, **k: {"found": False, "supports_kv_unified": False}),
        )
        out = ri._gguf_memory_breakdown(config, gguf, n_ctx = 8192, n_parallel = 4)
        assert out.n_parallel == 4


class TestMlxEstimateKvBits:
    """Which requested KV widths reach the MLX planner, and which are dropped."""

    @pytest.mark.parametrize(
        "requested, expected",
        [
            (4, 4),
            (8, 8),
            # Not one of MLX's widths: the runtime refuses it and leaves the cache unquantized.
            (7, None),
            (0, None),
            (None, None),
            # A bool is an int in Python, and True would otherwise read as 1 bit.
            (True, None),
            ("4", None),
        ],
    )
    def test_kv_bits_resolution(self, requested, expected):
        assert ri._mlx_estimate_kv_bits(requested) == expected


# ---------------------------------------------------------------------------
# The MLX pre-load planner. The property the GGUF planner cannot express: a linear-attention
# layer's recurrent state does not grow with the context, so a hybrid must not be charged as
# if every layer kept a key/value cache.
# ---------------------------------------------------------------------------

import glob  # noqa: E402
from dataclasses import replace  # noqa: E402

from core.inference import mlx_memory as mm  # noqa: E402

try:
    import mlx.core as _mx  # noqa: F401
    import mlx_lm  # noqa: F401
    _HAVE_MLX = True
except Exception:
    _HAVE_MLX = False

_NEEDS_MLX = pytest.mark.skipif(not _HAVE_MLX, reason = "needs a working MLX install")


def _mlx_raise():
    raise NotImplementedError


def _plan_entry(**over):
    """One cache plan entry: a growing key/value cache unless overridden."""
    return {
        "const": 0.0,
        "slope": 4096.0,
        "quant_const": 0.0,
        "quant_slope": 1152.0,
        "bound_spec": None,
        "conv_width": 0,
        "block": 256,
        "converts": True,
        **over,
    }


_GROWING = _plan_entry()
_RECURRENT = _plan_entry(
    const = 2_195_456.0,
    slope = 0.0,
    quant_const = 2_195_456.0,
    quant_slope = 0.0,
    conv_width = 8192,
    block = 1,
    converts = False,
)
_UNQUANTIZED = {"quant_start": None, "prefill_chunk": mm.MLX_PREFILL_CHUNK}


def _caption(
    plan,
    bits = None,
    converted = False,
    n_ctx = 32768,
):
    return mm._cache_width_name(plan, bits, converted, "bf16", n_ctx, mm.MLX_PREFILL_CHUNK)


def _local_snapshot(repo):
    """This repo's snapshot directory, skipping the test when it is not here."""
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    hits = glob.glob(
        os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots", "*", "config.json")
    )
    if not hits:
        pytest.skip(f"{repo} not on this disk")
    return os.path.dirname(hits[0])


def _on_bfloat16_chip():
    import mlx.core as mx
    if mm._runtime_dtype() is not mx.bfloat16:
        pytest.skip("measured on a chip the loader gives bfloat16")


class TestCacheWidthName:
    def test_only_the_entries_that_grow_name_the_cache(self):
        # The majority recurrent state never moves with the context, so it names neither cache.
        hybrid = [_RECURRENT] * 30 + [_GROWING] * 10
        assert _caption(hybrid) == "bf16"
        assert _caption(hybrid, 4, True) == "4-bit"
        assert _caption([_RECURRENT]) == "bf16"

    def test_a_converted_run_is_named_as_its_own_control_is_labelled(self):
        assert [_caption([_GROWING] * 10, b, True) for b in (8, 6, 5, 4, 3, 2)] == [
            "8-bit",
            "6-bit",
            "5-bit",
            "4-bit",
            "3-bit",
            "2-bit",
        ]
        assert _caption([_GROWING] * 10, 4, False) == "bf16"

    def test_an_entry_that_cannot_convert_keeps_its_width_through_a_converting_run(self):
        # Conversion is per entry, so a run can convert part of what it holds.
        staying = _plan_entry(converts = False, quant_slope = _GROWING["slope"])
        assert _caption([staying] * 57 + [_GROWING] * 3, 4, True) == "bf16/4-bit"
        # Ordered by what each carries: counting entries would name the heavy minority last.
        heavy, light = (
            _plan_entry(quant_slope = 100_000.0),
            _plan_entry(converts = False, slope = 1.0, quant_slope = 1.0),
        )
        assert _caption([light] * 50 + [heavy] * 2, 4, True) == "4-bit/bf16"

    @_NEEDS_MLX
    def test_a_bounded_entry_is_weighed_by_what_it_holds_and_not_by_its_slope(self):
        # Llama4's shape: past their bound the chunked entries stop growing.
        from mlx_lm.models.cache import ChunkedKVCache

        bounded = _plan_entry(converts = False, bound_spec = (ChunkedKVCache, "chunk_size", 8192))
        plan = [bounded] * 3 + [_plan_entry(quant_slope = 1152.0)]
        assert _caption(plan, 4, True, 8192).startswith("bf16")
        assert _caption(plan, 4, True, 131_072).startswith("4-bit")

    @_NEEDS_MLX
    def test_every_width_a_cache_is_built_at_has_a_caption(self):
        import mlx.core as mx
        assert [mm._width_name(d) for d in (mx.bfloat16, mx.float16, mx.float32)] == [
            "bf16",
            "f16",
            "f32",
        ]


class TestKvBytes:
    def test_a_hybrid_plan_is_far_cheaper_than_charging_every_layer_as_attention(self):
        hybrid = [_RECURRENT] * 30 + [_GROWING] * 10
        assert (
            mm._kv_bytes(hybrid, 131_072, **_UNQUANTIZED)[0] * 3
            < mm._kv_bytes([_GROWING] * 40, 131_072, **_UNQUANTIZED)[0]
        )

    def test_a_runtime_that_declines_to_chunk_is_recognised_on_either_half(self):
        assert mm._declines_to_chunk(type("M", (), {"no_chunked_prefill": True}), object())
        assert mm._declines_to_chunk(object, type("T", (), {"no_chunked_prefill": True})())
        assert not mm._declines_to_chunk(object, object())

    def test_only_an_entry_that_converts_is_charged_both_widths(self):
        held = mm._held_tokens(_GROWING, 2048, 2048, decoding = False)
        both = int(_GROWING["slope"] * held + _GROWING["quant_slope"] * held)
        assert mm._crossover_bytes([_GROWING], 2048, 2048) == both
        assert mm._crossover_bytes([_RECURRENT], 2048, 2048) == _RECURRENT["const"]

    def test_one_failed_conversion_refuses_the_whole_request(self):
        def _entry(converts):
            return type(
                "Entry",
                (),
                {
                    "max_size": None,
                    "window_size": None,
                    "to_quantized": lambda self, **kw: "converted" if converts else _mlx_raise(),
                },
            )()

        assert mm._quantize_like_runtime([_entry(True), _entry(False)], 4, 64) is None

    @pytest.mark.skipif(not _HAVE_MLX, reason = "drives a real cache class")
    def test_a_bounded_window_leaves_the_request_full_width(self):
        from mlx_vlm.models.unlimited_ocr.language import RingSlidingKVCache

        entry = RingSlidingKVCache(512)
        assert entry.to_quantized(group_size = 64, bits = 4) is not None
        assert mm._quantize_like_runtime([entry], 4, 64) is None


class TestComputeBytes:
    def test_an_unreadable_width_is_not_guessed(self):
        assert mm._compute_bytes(mm._config_widths({}), 2, 2048, [_GROWING], 8192, None) == 0

    def test_the_calibrated_term_matches_the_measurement_it_came_from(self):
        # The only absolute here, so it is what pins the scale to the load it was fitted to.
        measured = 610 * 1024**2
        priced = mm._compute_bytes(
            (1024, 3072, 16), 2, mm.MLX_PREFILL_CHUNK, [_GROWING] * 28, 8192, None
        )
        assert 0.7 * measured < priced < 1.5 * measured, (
            f"prices {priced / 1024**2:.0f} MiB where the load measured "
            f"{measured / 1024**2:.0f} MiB"
        )


@pytest.mark.skipif(not _HAVE_MLX, reason = "imports architecture modules")
class TestProbeFollowsTheLoadersRoute:
    """The estimator must not price a load path that cannot be taken."""

    def test_a_vision_model_is_never_priced_from_mlx_lms_copy_of_it(self):
        # Not interchangeable: mlx-lm keeps the MLA latent mlx-vlm expands, so it under-charges.
        import importlib

        config = {"model_type": "kimi_vl", "vision_config": {"depth": 2}}
        # The collision this guards is real only while mlx-lm still carries its own copy.
        assert importlib.import_module("mlx_lm.models.kimi_vl")
        assert {
            model_class.__module__.split(".")[0]
            for _, _, model_class in mm._probe_models(config, None)
        } == {"mlx_vlm"}

    @pytest.mark.parametrize(
        "architecture, resolved",
        [("DeepseekOCRForCausalLM", "deepseekocr"), ("DeepseekOCR2ForCausalLM", "deepseekocr_2")],
    )
    def test_a_config_is_read_as_the_loader_resolves_it_not_as_stated(
        self, tmp_path, architecture, resolved
    ):
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "deepseek_vl_v2", "architectures": [architecture]})
        )
        assert mm._snapshot_config(str(tmp_path))["model_type"] == resolved

    def test_a_checkpoint_supplying_its_own_module_is_refused_and_not_imported(self):
        with pytest.raises(ValueError, match = "its own model module"):
            list(mm._probe_models({"model_type": "qwen3", "model_file": "modeling.py"}, None))

    def test_the_probe_hands_the_forward_pass_a_model_in_eval_mode(self):
        import mlx.core as mx
        config = json.loads(
            (Path(_local_snapshot("unsloth/Qwen3-4B-Thinking-2507")) / "config.json").read_text()
        )
        assert next(iter(mm._probe_models(config, mx.bfloat16)))[0]().training is False

    @pytest.mark.parametrize(
        "chip, expected", [("Apple M1 Max", "float16"), ("Apple M3 Max", "bfloat16")]
    )
    def test_the_chip_decides_the_width_the_loader_installs(self, monkeypatch, chip, expected):
        # bf16 is emulated on M1/M2, so the loader installs fp16 there and bfloat16 later.
        import mlx.core as mx
        monkeypatch.setattr(mx, "device_info", lambda: {"device_name": chip})
        assert mm._runtime_dtype() is getattr(mx, expected)


@pytest.mark.skipif(not _HAVE_MLX, reason = "asks the installed runtime")
class TestGenerationSettingsComeFromTheLoader:
    """What a load prefills at is read off the runtime, never restated beside it."""

    def test_each_path_is_read_from_the_function_that_would_run_it(self):
        import inspect

        from core.inference import mlx_inference as mi

        for vision, drafted in ((False, False), (True, False), (False, True), (True, True)):
            step = mi._generation_step(vision = vision, drafted = drafted)
            for setting, ask in (
                ("prefill_step_size", mi.mlx_prefill_chunk),
                ("kv_group_size", mi.mlx_kv_group_size),
            ):
                assert ask(vision = vision, drafted = drafted) == (
                    inspect.signature(step).parameters[setting].default
                )
        # Named by package, not by asking the helper what it chose: both autoregressive
        # defaults are 2048/64 today, so a vision request served mlx-lm's function would
        # otherwise agree with every number this test checks.
        assert mi._generation_step(vision = True, drafted = False).__module__.startswith("mlx_vlm")
        assert mi._generation_step(vision = False, drafted = False).__module__.startswith("mlx_lm")
        # A drafter alone moves the chunk, so a restated constant cannot tell these apart.
        assert mi.mlx_prefill_chunk(drafted = True) != mi.mlx_prefill_chunk()
        # But only on the text path: mlx-vlm drives a drafter inside its own generation.
        assert mi.mlx_prefill_chunk(vision = True, drafted = True) == mi.mlx_prefill_chunk(vision = True)

    @pytest.mark.parametrize("chunk, group", [(None, 0), (True, "64")])
    def test_a_runtime_that_states_no_usable_value_falls_back(self, monkeypatch, chunk, group):
        from core.inference import mlx_inference as mi

        def _stated(**kw):
            return lambda prefill_step_size = chunk, kv_group_size = group: None

        monkeypatch.setattr(mi, "_generation_step", _stated)
        # True is an int and "64" is truthy, so a bare truthiness check would price both.
        assert mi.mlx_prefill_chunk() == mi.MLX_PREFILL_CHUNK_FALLBACK
        assert mi.mlx_kv_group_size() == mi.MLX_KV_GROUP_SIZE_FALLBACK

    def test_the_eligibility_probe_converts_at_the_width_generation_would(self, monkeypatch):
        # The probe decides whether a request is offered at all, so a width of its own would
        # accept a cache generation then refuses, or refuse one it would have taken.
        import mlx.core as mx

        from core.inference import mlx_inference as mi

        asked = []

        class _Entry:
            state = mx.zeros((1,))

            def to_quantized(self, group_size, bits):
                asked.append((group_size, bits))
                return self

        seen = []
        monkeypatch.setattr(
            mi, "mlx_kv_group_size", lambda **kw: seen.append(kw.get("vision")) or 32
        )
        monkeypatch.setattr(mi, "_kv_entry_nbytes", lambda entry: 1)
        mi._kv_quant_probe(lambda *a, **kw: None, [_Entry()], 4, vision = True)
        assert asked == [(32, 4)]
        # And of the runtime actually being probed: a VLM cache asked about mlx-lm's width
        # would be admitted or refused on a width generation never uses.
        assert seen == [True]

    @pytest.mark.parametrize("is_vlm", [True, False])
    def test_eligibility_tells_the_probe_which_runtime_it_is_probing(self, monkeypatch, is_vlm):
        from mlx_lm.models import cache as lm_cache
        from mlx_vlm.models import cache as vlm_cache

        from core.inference import mlx_inference as mi

        told = []
        for module in (vlm_cache, lm_cache):
            monkeypatch.setattr(module, "make_prompt_cache", lambda model: [object()])
        monkeypatch.setattr(
            mi,
            "_kv_quant_probe",
            lambda *a, vision = False, **kw: told.append(vision) or (1, 0, None, True),
        )
        mi._kv_quant_eligibility(SimpleNamespace(language_model = object()), is_vlm)
        assert told == [is_vlm]

    def test_the_estimator_tells_the_loader_which_package_would_load_it(self, monkeypatch):
        from core.inference import mlx_inference as mi

        seen = []
        monkeypatch.setattr(mi, "mlx_prefill_chunk", lambda **kw: seen.append(kw["vision"]) or 2048)
        monkeypatch.setattr(mi, "mlx_kv_group_size", lambda **kw: 64)
        mm._generation_settings({"model_type": "kimi_vl", "vision_config": {"depth": 2}})
        mm._generation_settings({"model_type": "qwen3"})
        assert seen == [True, False]

    def test_a_host_without_the_loader_prices_the_fallback(self, monkeypatch):
        import builtins

        real = builtins.__import__

        def _no_loader(name, *a, **kw):
            if name == "core.inference.mlx_inference":
                raise ImportError("no loader here")
            return real(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _no_loader)
        assert mm._generation_settings({"model_type": "llama"}) == (mm.MLX_PREFILL_CHUNK, 64)


@_NEEDS_MLX
@pytest.mark.parametrize("reported, compute", [(None, 9_448_833_807), (512, 2_874_458_895)])
def test_the_estimate_prices_the_chunk_the_loader_reports(monkeypatch, reported, compute):
    # The drift this closes: a load prefilling at 512 priced at 2048 is 3.3x high on the
    # quantized score term, which no assertion about the constant alone would catch.
    _on_bfloat16_chip()
    if reported is not None:
        from core.inference import mlx_inference as mi
        monkeypatch.setattr(mi, "mlx_prefill_chunk", lambda **kw: reported)
    breakdown = mm.mlx_memory_breakdown(
        _local_snapshot("unsloth/Qwen3-4B-Thinking-2507"),
        n_ctx = 32768,
        load_in_4bit = True,
        kv_bits = 4,
    )
    assert breakdown is not None and breakdown.compute_bytes == compute
    # And an explicit chunk still outranks whatever the loader reports.
    override = mm.mlx_memory_breakdown(
        _local_snapshot("unsloth/Qwen3-4B-Thinking-2507"),
        n_ctx = 32768,
        load_in_4bit = True,
        kv_bits = 4,
        prefill_chunk = 512,
    )
    assert override.compute_bytes == 2_874_458_895


@_NEEDS_MLX
def test_a_model_mlx_vlm_would_diffuse_is_refused_rather_than_priced():
    # stream_generate diverts ahead of the autoregressive chunking path, and what each
    # diffusion generator prefills in is not readable pre-load: LLaDA2's block_length of 32
    # is the block it prefills in, DiffusionGemma's is a denoising-canvas cap. Priced at the
    # autoregressive 2048 this quoted 18.61 GB for a prompt that goes in one step.
    snapshot = _local_snapshot("mlx-community/diffusiongemma-26B-A4B-it-4bit")
    assert mm._routes_to_diffusion(mm._snapshot_config(snapshot)) is True
    assert mm.mlx_memory_breakdown(snapshot, n_ctx = 32768, load_in_4bit = True) is None


@_NEEDS_MLX
def test_an_ordinary_vision_model_is_not_mistaken_for_a_diffusion_one():
    # The refusal must not reach a model that would have priced: this one carries no marker.
    config = {"model_type": "kimi_vl", "vision_config": {"depth": 2}}
    assert mm._routes_to_diffusion(config) is False
    assert mm._generation_settings(config)[0] > 0


@_NEEDS_MLX
class TestDiffusionRouting:
    """The markers gate a build; mlx-vlm's own predicate is the verdict."""

    @staticmethod
    def _resolving(
        monkeypatch,
        *,
        canvas,
        mask,
        build = lambda config: object(),
    ):
        monkeypatch.setattr(
            mm,
            "_loader_config",
            lambda arch, config: SimpleNamespace(canvas_length = canvas, mask_token_id = mask),
        )
        monkeypatch.setattr(
            "mlx_vlm.utils.get_model_and_args",
            lambda config: (SimpleNamespace(Model = build), None),
        )

    @pytest.mark.parametrize("verdict", [True, False])
    def test_the_predicate_outranks_the_marker(self, monkeypatch, verdict):
        # A marker only earns the model a classification. nemotron_labs_diffusion carries
        # mask_token_id = 100 and still generates autoregressively under Studio's arguments,
        # so a marker read as the verdict would refuse a load that prices.
        from mlx_vlm.generate import diffusion as vlm_diffusion

        self._resolving(monkeypatch, canvas = None, mask = 100)
        monkeypatch.setattr(vlm_diffusion, "is_diffusion_model", lambda model, kw: verdict)
        assert mm._routes_to_diffusion({"model_type": "whatever"}) is verdict

    def test_a_diverted_load_refuses_without_needing_a_checkpoint(self, monkeypatch):
        from mlx_vlm.generate import diffusion as vlm_diffusion

        self._resolving(monkeypatch, canvas = 512, mask = None)
        monkeypatch.setattr(vlm_diffusion, "is_diffusion_model", lambda model, kw: True)
        monkeypatch.setattr(mm, "_loads_as_vision", lambda config: True)
        with pytest.raises(ValueError, match = "diffusion generator"):
            mm._generation_settings({"model_type": "whatever"})

    def test_an_architecture_carrying_no_marker_is_never_built(self, monkeypatch):
        # The marker gate keeps a wrapper build off every other architecture's path.
        def _explode(config):
            raise AssertionError("built a wrapper for a config carrying no marker")

        self._resolving(monkeypatch, canvas = None, mask = None, build = _explode)
        assert mm._routes_to_diffusion({"model_type": "whatever"}) is False

    def test_a_marked_model_that_cannot_be_placed_is_refused_not_assumed(self, monkeypatch):
        # Not knowing which generator runs is what the estimate refuses; treating it as
        # autoregressive would quote a confident chunk for a load nobody could place.
        def _explode(config):
            raise RuntimeError("this wrapper cannot be built")

        self._resolving(monkeypatch, canvas = 512, mask = None, build = _explode)
        with pytest.raises(RuntimeError):
            mm._routes_to_diffusion({"model_type": "whatever"})
        # And nothing on the way to the estimate's guard swallows it, which is what turns it
        # into a refusal rather than a number.
        with pytest.raises(RuntimeError):
            mm._generation_settings({"model_type": "whatever", "vision_config": {"depth": 2}})


@_NEEDS_MLX
@pytest.mark.parametrize("reported, explicit", [(32, None), (64, 32)])
def test_the_estimate_prices_the_group_size_it_is_given(monkeypatch, reported, explicit):
    # The same defect one setting over: a cache grouped at 32 costs more scales and biases
    # than one grouped at 64, so restating either width prices a conversion that never ran.
    # The loader supplies it, and an explicit width outranks what the loader reports.
    _on_bfloat16_chip()
    from core.inference import mlx_inference as mi

    monkeypatch.setattr(mi, "mlx_kv_group_size", lambda **kw: reported)
    breakdown = mm.mlx_memory_breakdown(
        _local_snapshot("unsloth/Qwen3-4B-Thinking-2507"),
        n_ctx = 32768,
        load_in_4bit = True,
        kv_bits = 4,
        kv_group_size = explicit,
    )
    assert breakdown is not None and breakdown.kv_bytes == 1_521_745_920


@pytest.mark.skipif(not _HAVE_MLX, reason = "drives real cache classes")
def test_the_peak_of_a_bounded_cache_is_measured_not_derived():
    # Hand-derived three times and wrong three times, so the peak is measured by driving the class.
    from mlx_lm.models import cache as C

    for bound, attribute, value, n_ctx, chunk, expected in (
        (C.RotatingKVCache, "max_size", 512, 100, 2048, 355),
        (C.RotatingKVCache, "max_size", 512, 255, 2048, 510),
        (C.RotatingKVCache, "max_size", 512, 1024, 2048, 1023),
        (C.ChunkedKVCache, "chunk_size", 512, 4096, 2048, 2560),
        (C.ChunkedKVCache, "chunk_size", 512, 2560, 2048, 2048),
        # A chunk of one charges the decode step generate_step runs before it yields.
        (C.RotatingKVCache, "max_size", 512, 256, 1, 512),
        # And an unbounded context terminates: the walk stops once the cache has settled.
        (C.RotatingKVCache, "max_size", 512, 10**12, 2048, 2559),
    ):
        assert (
            mm._bounded_peak(bound, attribute, value, n_ctx, chunk) == expected
        ), f"{bound.__name__}({value}) at {n_ctx}"
    assert mm._bound_spec(type("E", (), {"max_size": 512})())
    assert mm._bound_spec(type("E", (), {"chunk_size": 512})())


@pytest.mark.skipif(not _HAVE_MLX, reason = "builds a real architecture")
class TestPricingALoad:
    _CONFIG = {
        "model_type": "llama",
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "intermediate_size": 512,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-5,
        "vocab_size": 512,
        "rope_theta": 10000.0,
    }

    def _checkpoint(
        self,
        tmp_path,
        *,
        prefix = "",
    ):
        import mlx.core as mx
        from mlx.utils import tree_flatten

        config = dict(self._CONFIG, tie_word_embeddings = True)
        (tmp_path / "config.json").write_text(json.dumps(config))
        built = mm._whole_model(config, mx.bfloat16)
        mx.save_safetensors(
            str(tmp_path / "model.safetensors"),
            {
                prefix + name: mx.zeros(value.shape, dtype = mx.bfloat16)
                for name, value in tree_flatten(built.parameters())
            },
        )
        return config

    def test_a_float32_checkpoint_goes_resident_at_the_width_the_chip_gives_it(self):
        if mm._runtime_dtype().size != 2:
            pytest.skip("measured on a chip the loader gives a 16-bit width")
        snapshot = _local_snapshot("hf-internal-testing/tiny-random-LlamaForCausalLM")
        config = json.loads((Path(snapshot) / "config.json").read_text())
        assert mm.mlx_weight_bytes(snapshot, config, load_in_4bit = False) == 2_064_544

    def test_a_quantizing_load_is_priced_below_the_shards(self, tmp_path):
        config = self._checkpoint(tmp_path)
        wide = mm.mlx_weight_bytes(str(tmp_path), config, load_in_4bit = False)
        quantized = mm.mlx_weight_bytes(str(tmp_path), config, load_in_4bit = True)
        # Neither width is the file size: the header is left behind, and quantizing drops more.
        assert (wide, quantized) == (2_886_144, 1_001_984)
        assert quantized < wide < mm._shard_bytes(str(tmp_path), config) == 2_888_265

    def test_shards_that_name_nothing_the_architecture_has_are_not_priced_from_it(self, tmp_path):
        # Why the check is name-based rather than a count: a draft declares its own tensor count.
        config = self._checkpoint(tmp_path, prefix = "draft.")
        with pytest.raises(ValueError, match = "does not supply"):
            mm._resident_bytes(str(tmp_path), config, quantize = True)
        for flag in (True, False):
            assert mm.mlx_weight_bytes(str(tmp_path), config, load_in_4bit = flag) == (
                mm._shard_bytes(str(tmp_path), config)
            )


@pytest.mark.skipif(not _HAVE_MLX, reason = "asks the installed loader")
class TestALoadWithNoFootprintToQuote:
    def test_quantization_metadata_no_load_can_honour_is_refused_not_priced(self, tmp_path):
        per_module = {"model.layers.0.self_attn.q_proj": {"bits": 4, "group_size": 64}}
        for spelling in ("quantization", "quantization_config"):
            stated = {"model_type": "llama", spelling: per_module}
            assert mm._load_is_refused(str(tmp_path), stated, True)
            assert mm._load_is_refused(str(tmp_path), stated, False) is None
            # Bitsandbytes is the exception at BOTH: mlx-lm cannot read those weights at all.
            bnb = {"model_type": "llama", spelling: {"quant_method": "bitsandbytes"}}
            for flag in (True, False):
                assert "bitsandbytes" in (mm._load_is_refused(str(tmp_path), bnb, flag) or "")

    @pytest.mark.parametrize(
        "repo, names",
        [
            ("OsaurusAI/ZAYA1-VL-8B-MXFP4", "no home for"),
            ("Qwen/Qwen3-Embedding-0.6B", "QK-norm"),
            ("z-lab/Qwen3.6-27B-DFlash", "QK-norm"),
            ("z-lab/gemma-4-31B-it-DFlash", "QK-norm"),
            ("yuyijiong/Qwen3.5-4B-Eagle3", "no home for"),
        ],
    )
    def test_a_checkpoint_no_load_starts_from_is_refused(self, repo, names):
        # Told apart: a version gap says upgrade, extra tensors say the load cannot start.
        snapshot = _local_snapshot(repo)
        config = mm._snapshot_config(snapshot)
        for flag in (True, False):
            assert names in (mm._load_is_refused(snapshot, config, flag) or "")
            assert mm.mlx_memory_breakdown(snapshot, n_ctx = 2048, load_in_4bit = flag) is None

    def test_no_checkpoint_on_this_disk_is_refused_for_an_invented_reason(self):
        # A refusal that rejects working input is worse than the defect it prevents.
        hub = os.path.expanduser("~/.cache/huggingface/hub")
        seen = accepted = 0
        for path in glob.glob(os.path.join(hub, "models--*", "snapshots", "*", "config.json")):
            directory = os.path.dirname(path)
            config = mm._snapshot_config(directory)
            if not isinstance(config, dict) or not config.get("model_type"):
                continue
            try:
                if not mm.mlx_shard_files(directory, config):
                    continue
            except Exception:
                continue
            seen += 1
            refusals = [mm._load_is_refused(directory, config, flag) for flag in (True, False)]
            for refusal in refusals:
                assert refusal is None or any(
                    reason in refusal for reason in ("QK-norm", "no home for", "bitsandbytes")
                ), f"{path}: {refusal}"
            accepted += refusals[0] is None
        if not seen:
            pytest.skip("no checkpoints on this disk")
        assert accepted > seen * 0.9, "the gate has started refusing wholesale"

    def test_extras_a_registered_rule_filters_are_not_refused(self):
        snapshot = _local_snapshot("google/gemma-4-E2B")
        config = mm._snapshot_config(snapshot)
        assert mm._load_is_refused(snapshot, config, True) is None
        assert mm.mlx_weight_bytes(snapshot, config, load_in_4bit = True) == 7_510_668_230

    _FILTERED = [
        "language_model.model.per_layer_model_projection.biases",
        "language_model.model.per_layer_model_projection.scales",
    ]
    _GEMMA4 = {"model_type": "gemma4", "vision_config": {"model_type": "gemma4"}}

    @pytest.mark.parametrize(
        "config, extras, names",
        [
            (_GEMMA4, _FILTERED, None),
            (_GEMMA4, _FILTERED + ["mystery.weight"], "mystery.weight"),
            ({"model_type": "an-architecture-with-no-rule"}, ["d2t", "fc.weight"], "d2t"),
        ],
    )
    def test_a_rule_exempts_only_the_keys_it_names(self, config, extras, names):
        refusal = mm._extra_tensors_refused(object(), config, extras)
        assert refusal is None if names is None else (names in refusal and "no home for" in refusal)


class TestShardsTheLoaderReads:
    """Which safetensors beside a config actually become weights."""

    _VISION = {"model_type": "llava", "vision_config": {"hidden_size": 8}}

    def _spread(self, tmp_path, config):
        for name in (
            "model-00001.safetensors",
            "adapter_model.safetensors",
            "consolidated.safetensors",
        ):
            (tmp_path / name).write_bytes(b"x" * 100)
        return sorted(os.path.basename(p) for p in mm.mlx_shard_files(str(tmp_path), config))

    def test_the_text_loader_reads_model_shards_only(self, tmp_path):
        # mlx-lm globs `model*.safetensors`, so an adapter beside the shards is not weighed.
        assert self._spread(tmp_path, {"model_type": "llama"}) == ["model-00001.safetensors"]
        adapter_only = tmp_path / "lora"
        adapter_only.mkdir()
        (adapter_only / "adapter_model.safetensors").write_bytes(b"x" * 100)
        assert mm.mlx_shard_files(str(adapter_only), {"model_type": "llama"}) == []

    def test_the_vision_loader_reads_the_index_before_the_directory(self, tmp_path):
        # mlx-vlm globs only without an index, and excludes the consolidated copy.
        assert self._spread(tmp_path, self._VISION) == [
            "adapter_model.safetensors",
            "model-00001.safetensors",
        ]
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a.weight": "model-00001.safetensors"}})
        )
        assert self._spread(tmp_path, self._VISION) == ["model-00001.safetensors"]

    @pytest.mark.parametrize(
        "weight_map", [["model-00001.safetensors"], "model-00001.safetensors", 7, True, None]
    )
    def test_an_index_whose_weight_map_is_not_a_mapping_is_fatal(self, tmp_path, weight_map):
        # mlx-vlm reaches straight for `.values()` and catches only ValueError and OSError.
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )
        with pytest.raises(ValueError, match = "no weight map"):
            mm.mlx_shard_files(str(tmp_path), self._VISION)


@_NEEDS_MLX
@pytest.mark.parametrize(
    "repo, n_ctx, kv_bits, weights, kv, compute, layers, caption",
    [
        # Dense; the 4-bit row costs MORE, because unfused attention materializes scores.
        (
            "unsloth/Qwen3-4B-Thinking-2507",
            32768,
            None,
            2_822_044_672,
            4_869_586_944,
            863_355_535,
            36,
            "bf16",
        ),
        (
            "unsloth/Qwen3-4B-Thinking-2507",
            32768,
            4,
            2_822_044_672,
            1_369_571_328,
            9_448_833_807,
            36,
            "4-bit",
        ),
        # Hybrid: 30 constant states against 10 growing, captioned for the ten.
        (
            "unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit",
            32768,
            None,
            21_634_993_114,
            740_720_640,
            2_737_160_847,
            40,
            "bf16",
        ),
        # Windowed: conversion is refused, so 4-bit asked for is still held at full width.
        ("unsloth/gemma-3-270m-it", 8192, None, 392_058_112, 65_258_496, 725_729_935, 18, "bf16"),
        ("unsloth/gemma-3-270m-it", 8192, 4, 392_058_112, 65_258_496, 725_729_935, 18, "bf16"),
        # Vision, through the tower its own loader resolves from the enclosing config.
        (
            "mlx-community/deepseek-vl2-tiny-4bit",
            4096,
            None,
            2_099_712_075,
            267_386_880,
            803_717_775,
            12,
            "bf16",
        ),
        # A tower not buildable from its own config, either side of mlx-vlm's conversion start.
        (
            "Qwen/Qwen2.5-VL-3B-Instruct",
            32768,
            4,
            3_520_856_064,
            342_392_832,
            5_167_104_719,
            36,
            "4-bit",
        ),
        (
            "Qwen/Qwen2.5-VL-3B-Instruct",
            5001,
            4,
            3_520_856_064,
            241_827_840,
            874_685_711,
            36,
            "4-bit",
        ),
        (
            "Qwen/Qwen2.5-VL-3B-Instruct",
            5000,
            4,
            3_520_856_064,
            241_827_840,
            874_685_647,
            36,
            "4-bit",
        ),
        # A stale index naming three shards this snapshot never shipped: the load globs instead.
        (
            "mlx-community/gemma-3n-E2B-it-4bit",
            4096,
            None,
            4_462_976_625,
            119_472_128,
            833_995_407,
            30,
            "bf16",
        ),
        # Widths its tower does not state, taken from the checkpoint's own config.
        (
            "mlx-community/Molmo-7B-D-0924-4bit",
            4096,
            None,
            5_298_715_127,
            249_561_088,
            1_281_737_359,
            28,
            "bf16",
        ),
        # Recorded float32, resident at the width the chip gives the loader.
        (
            "hf-internal-testing/tiny-random-LlamaForCausalLM",
            512,
            None,
            2_061_600,
            98_304,
            688_341_647,
            2,
            "bf16",
        ),
    ],
)
def test_a_real_checkpoint_is_priced_to_the_byte(
    repo, n_ctx, kv_bits, weights, kv, compute, layers, caption
):
    _on_bfloat16_chip()
    breakdown = mm.mlx_memory_breakdown(
        _local_snapshot(repo), n_ctx = n_ctx, load_in_4bit = True, kv_bits = kv_bits
    )
    assert breakdown is not None
    assert (breakdown.weights_bytes, breakdown.kv_bytes) == (weights, kv)
    assert breakdown.compute_bytes == compute
    assert breakdown.layer_count == layers
    assert breakdown.cache_type_kv == caption


@_NEEDS_MLX
@pytest.mark.parametrize("kv_bits, caption", [(6, "6-bit"), (8, "8-bit")])
def test_the_caption_names_a_width_the_real_conversion_reaches(kv_bits, caption):
    # Every width the control offers runs the whole probe / convert path, not just 4-bit.
    _on_bfloat16_chip()
    breakdown = mm.mlx_memory_breakdown(
        _local_snapshot("unsloth/Qwen3-4B-Thinking-2507"),
        n_ctx = 32768,
        load_in_4bit = True,
        kv_bits = kv_bits,
    )
    assert breakdown is not None and breakdown.cache_type_kv == caption


class TestTheContextSearch:
    """The search itself, priced by a stub so it runs where MLX and the checkpoints do not."""

    @staticmethod
    def _linear(monkeypatch, unpriceable = ()):
        """A byte per token, so every expected answer below is readable off the budget."""

        def _priced(sizing, n_ctx):
            return None if n_ctx in unpriceable else SimpleNamespace(total_bytes = n_ctx)

        monkeypatch.setattr(mm, "_size_load", lambda *a, **kw: object())
        monkeypatch.setattr(mm, "_priced_at", _priced)

    def test_the_answer_is_the_largest_whole_block_under_budget(self, monkeypatch):
        self._linear(monkeypatch)
        assert mm.mlx_fit_context("x", budget_bytes = 7000, max_ctx = 8192) == 6912

    def test_a_floor_between_blocks_rounds_up_rather_than_under_the_minimum(self, monkeypatch):
        # 4,100 rounds to 4,352, so a budget holding 4,096 but not 4,352 is not a fit.
        self._linear(monkeypatch)
        assert mm.mlx_fit_context("x", budget_bytes = 4200, max_ctx = 8192, min_ctx = 4100) is None
        assert mm.mlx_fit_context("x", budget_bytes = 5000, max_ctx = 8192, min_ctx = 4100) == 4864

    def test_a_context_that_cannot_be_priced_abandons_the_fit(self, monkeypatch):
        # 6,144 is the first midpoint. Searching past it would discard the half above and answer
        # with a context that is not the largest one that fits.
        self._linear(monkeypatch, unpriceable = (6144,))
        assert mm.mlx_fit_context("x", budget_bytes = 7000, max_ctx = 8192) is None

    def test_a_ceiling_that_already_fits_is_left_alone(self, monkeypatch):
        self._linear(monkeypatch)
        assert mm.mlx_fit_context("x", budget_bytes = 9000, max_ctx = 8192) is None

    def test_a_load_that_cannot_be_sized_is_not_fitted(self, monkeypatch):
        monkeypatch.setattr(mm, "_size_load", lambda *a, **kw: None)
        assert mm.mlx_fit_context("x", budget_bytes = 7000, max_ctx = 8192) is None


@_NEEDS_MLX
@pytest.mark.parametrize("budget_gib, fitted", [(6, 18_432), (12, 61_952), (24, 149_504)])
def test_a_real_checkpoint_is_fitted_to_the_byte(budget_gib, fitted):
    # Exact on both sides against the estimate the panel shows: what it returns fits and the next
    # block does not, which is what makes it a fit rather than a guess with headroom.
    _on_bfloat16_chip()
    snapshot = _local_snapshot("unsloth/Qwen3-4B-Thinking-2507")
    budget = budget_gib * 1024**3
    assert (
        mm.mlx_fit_context(snapshot, budget_bytes = budget, max_ctx = 262_144, load_in_4bit = True)
        == fitted
    )
    priced = mm.mlx_memory_breakdown(snapshot, n_ctx = fitted, load_in_4bit = True)
    over = mm.mlx_memory_breakdown(snapshot, n_ctx = fitted + mm.MLX_KV_BLOCK, load_in_4bit = True)
    assert priced.total_bytes <= budget < over.total_bytes
    # And a context that is not a number is refused rather than raised out of the guard.
    assert mm.mlx_memory_breakdown(snapshot, n_ctx = float("nan"), load_in_4bit = True) is None


@_NEEDS_MLX
@pytest.mark.parametrize(
    "repo, kv_bits, whole_prompt",
    [
        ("unsloth/Qwen3-4B-Thinking-2507", 4, False),
        ("unsloth/Qwen3-4B-Thinking-2507", None, True),
        ("unsloth/gemma-3-270m-it", None, False),
        ("unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit", None, False),
    ],
)
def test_the_total_never_falls_as_the_context_grows(repo, kv_bits, whole_prompt):
    # What the search rests on, across a dense, a windowed and a hybrid shape. The fourth row is
    # not a checkpoint that declines to chunk -- none is cached here -- but the same Qwen sizing
    # with that fact flipped, which is the only way to reach the branch from this disk.
    sizing = mm._size_load(_local_snapshot(repo), kv_bits, None, None, True)
    if whole_prompt:
        forced = replace(sizing, facts = {**sizing.facts, "whole_prompt": True})
        # And the flip has to reach the arithmetic: charging the whole prompt as one chunk costs
        # more than charging it in 2,048-token steps, so a lost conditional shows up here.
        assert mm._priced_at(forced, 32_768).total_bytes > mm._priced_at(sizing, 32_768).total_bytes
        sizing = forced
    # Every block to the ceiling a fit is given, which costs under a tenth of a second. A sample,
    # not a proof: the terms are piecewise, and a whole-prompt sizing charges compute per token,
    # so this catches a term that falls with the context rather than establishing that none can.
    steps = range(mm.MLX_KV_BLOCK, 262_145, mm.MLX_KV_BLOCK)
    totals = [mm._priced_at(sizing, n).total_bytes for n in steps]
    assert totals == sorted(totals)
