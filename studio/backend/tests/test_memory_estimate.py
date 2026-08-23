# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for the Load-Model memory estimate (POST /api/inference/estimate-memory).

Guards the pieces the estimate is assembled from, and the properties that make it
safe to put a number in front of a user:

* ``_gguf_runtime_bytes`` -- KV + compute buffers, itemized. The load-bearing case
  is a header without the attention dims: it must report ``kv_estimable = False``
  with ``kv_bytes == 0``, since a UI reading that zero as "no cache" is worse than
  showing nothing.
* ``_estimate_gguf_kv_gb`` -- now a thin wrapper over the above. The training guard
  still calls it, so its value must stay the old sum exactly.
* ``_gguf_offloaded_layer_fraction`` -- Auto is deliberately 1.0, not a guess.
* ``_gguf_resident_file_gb`` / ``_gguf_memory_breakdown`` -- weights are derived by
  subtracting the context term out of ``_estimate_gguf_required_gb``. The observable
  proving the arms are paired is that weights do not move with the context slider.
* ``_localized_estimate_config`` -- without it a cached repo priced itself through a
  ``paths-info`` call. Pins both halves: the copy takes the local arm, and the cached
  original is never mutated.
* ``_estimate_token_fingerprint`` -- both TTL caches are keyed per token.
* the route -- the "cannot size this" answers, and that an Ollama manifest ref is
  refused before anything is materialized.

No GPU, no network, no model load: every GGUF here is a synthetic header on
tmp_path. Cross-platform.
"""

import inspect
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

    def test_auto_honours_a_layer_count_from_the_extras(self):
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
        # Manual strips the flag from the extras, so the field still owns placement.
        assert ri._gguf_offloaded_layer_fraction("manual", 0, 12, ["-ngl", "999"]) == 0.0


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

        llama.cpp reads the attention head counts with ``required = false``
        (llama-model.cpp:1177) while block_count and embedding_length are required
        (:1111, :1116), so every pure SSM model -- Mamba, Mamba2, RWKV -- loads with a
        layer count and no attention dimensions, which is exactly what
        ``_can_estimate_kv`` rejects. Dropping the count there would report a manual
        --gpu-layers 0 as a fully GPU-resident load on all of them.
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

    def test_manual_split_divides_the_weights_but_not_the_cache(self, config, gqa_gguf):
        # llama.cpp keeps the whole KV cache on the GPU under a partial layer
        # offload (only --no-kv-offload moves it), so only the weights split.
        b = ri._gguf_memory_breakdown(
            config, gqa_gguf, n_ctx = 8192, gpu_memory_mode = "manual", gpu_layers = 6
        )
        fraction = 6 / float(_GQA_FIELDS["block_count"] + 1)
        assert b.layer_count == _GQA_FIELDS["block_count"]
        assert b.gpu_layers == 6
        assert b.gpu_bytes == (
            int(self.MAIN_BYTES * fraction) + self.COMPANION_BYTES + b.kv_bytes + b.compute_bytes
        )
        assert b.gpu_bytes < b.total_bytes

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

    def test_non_gguf_model_is_not_priced(self, monkeypatch):
        # Safetensors / MLX size their memory through a different allocator;
        # quoting the GGUF arithmetic for them is a made-up number in a box.
        monkeypatch.setattr(
            ri,
            "_cached_estimate_config",
            lambda *a, **kw: SimpleNamespace(is_gguf = False, identifier = "org/model"),
        )
        resp = _estimate(model_path = "org/model")
        assert resp.available is False
        assert resp.reason == "not_gguf"

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
        )
        assert asked == [True]

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
        source = inspect.getsource(ri._gguf_memory_breakdown)
        assert "_extra_args_draft_cache_types(extras)" in source
        assert "draft_cache_type_k = draft_k or spec_draft_cache_type" in source


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
        assert ri._estimate_spec_mode_terms("extras", []) == (False, False)
        assert ri._estimate_spec_mode_terms("extras", ["--spec-type", "draft-simple"]) == (
            False,
            False,
        )
        assert ri._estimate_spec_mode_terms("extras", ["--spec-type", "draft-mtp"]) == (True, True)
        assert ri._estimate_spec_mode_terms("extras", ["--spec-type", "draft-eagle3"]) == (
            False,
            True,
        )
