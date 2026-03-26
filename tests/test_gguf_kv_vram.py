"""Cross-platform test harness for GGUF KV cache VRAM estimation (PR #4623).

Tests the KV cache estimation, context fitting, GPU selection, and load_model
command construction in llama_cpp.py.  No GPU, no network, no llama-server,
no torch required -- all heavy dependencies are stubbed.

Run:
    python -m pytest tests/test_gguf_kv_vram.py -v

Three test layers:
  Layer 1  Pure-logic unit tests on _estimate_kv_cache_bytes, _fit_context_to_vram, _select_gpus
  Layer 2  Integration tests on load_model with monkeypatched externals
  Layer 3  Real GGUF binary parsing (skipped if fixture files absent)
"""

import importlib.util
import os
import struct as _struct
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Stub heavy dependencies before importing llama_cpp.py ───────────────

_structlog = types.ModuleType("structlog")
_structlog.get_logger = lambda *a, **kw: type(
    "L",
    (),
    {
        "info": lambda *a, **kw: None,
        "warning": lambda *a, **kw: None,
        "error": lambda *a, **kw: None,
        "debug": lambda *a, **kw: None,
    },
)()
sys.modules["structlog"] = _structlog

_loggers = types.ModuleType("loggers")
_loggers.get_logger = lambda *a, **kw: _structlog.get_logger()
sys.modules["loggers"] = _loggers

_httpx = types.ModuleType("httpx")
_httpx.Client = type("Client", (), {"__init__": lambda *a, **kw: None})
_httpx.ConnectError = type("ConnectError", (Exception,), {})
_httpx.TimeoutException = type("TimeoutException", (Exception,), {})
_httpx.ReadTimeout = type("ReadTimeout", (Exception,), {})
_httpx.ReadError = type("ReadError", (Exception,), {})
_httpx.RemoteProtocolError = type("RemoteProtocolError", (Exception,), {})
_httpx.CloseError = type("CloseError", (Exception,), {})
_httpx.Timeout = lambda **kw: None
_httpx.get = lambda *a, **kw: None
sys.modules["httpx"] = _httpx

# ── Load the real module ─────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "studio" / "backend" / "core" / "inference" / "llama_cpp.py"

SPEC = importlib.util.spec_from_file_location("llama_cpp_backend", str(MODULE_PATH))
assert SPEC is not None and SPEC.loader is not None, f"Cannot find {MODULE_PATH}"

# Patch _kill_orphaned_servers and atexit.register during import
# so the constructor doesn't call pgrep or register cleanup handlers.
_orig_atexit_register = __import__("atexit").register
__import__("atexit").register = lambda *a, **kw: None

_mod = importlib.util.module_from_spec(SPEC)
# Patch __init__ to prevent _kill_orphaned_servers during module import
_real_init = None


def _patched_init(self):
    """Minimal __init__ that skips _kill_orphaned_servers and atexit."""
    self._process = None
    self._port = None
    self._model_identifier = None
    self._gguf_path = None
    self._hf_repo = None
    self._hf_variant = None
    self._is_vision = False
    self._healthy = False
    self._context_length = None
    self._chat_template = None
    self._supports_reasoning = False
    self._supports_tools = False
    self._cache_type_kv = None
    self._reasoning_default = True
    self._n_layers = None
    self._n_kv_heads = None
    self._n_heads = None
    self._embedding_length = None
    import threading

    self._lock = threading.Lock()
    self._stdout_lines = []
    self._stdout_thread = None
    self._cancel_event = threading.Event()


sys.modules[SPEC.name] = _mod
SPEC.loader.exec_module(_mod)
__import__("atexit").register = _orig_atexit_register

LlamaCppBackend = _mod.LlamaCppBackend

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_backend(**metadata):
    """Create a LlamaCppBackend with metadata fields set directly."""
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    _patched_init(b)
    for k, v in metadata.items():
        setattr(b, f"_{k}", v)
    return b


def _llama3_70b():
    """Llama-3 70B-like metadata."""
    return _make_backend(
        n_layers=80,
        n_kv_heads=8,
        n_heads=64,
        embedding_length=8192,
        context_length=131072,
    )


def _qwen35_9b():
    """Qwen3.5-9B-like metadata: 48 layers, 8 KV heads, 32 heads, 3584 emb, 262k ctx."""
    return _make_backend(
        n_layers=48,
        n_kv_heads=8,
        n_heads=32,
        embedding_length=3584,
        context_length=262144,
    )


MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024


# ==========================================================================
#  Layer 1: Pure-logic unit tests
# ==========================================================================


class TestEstimateKvCacheBytes:
    """Tests for _estimate_kv_cache_bytes."""

    def test_basic_f16(self):
        """Llama-3 70B at 131k context, f16 KV."""
        b = _llama3_70b()
        kv = b._estimate_kv_cache_bytes(131072, "f16")
        # 2 * 8 * 128 * 80 * 131072 * 2.0 = 43,486,543,872
        assert kv == 2 * 8 * 128 * 80 * 131072 * 2

    def test_missing_metadata_returns_zero(self):
        """If any critical field is None, return 0."""
        for missing in ["n_layers", "embedding_length"]:
            b = _llama3_70b()
            setattr(b, f"_{missing}", None)
            assert b._estimate_kv_cache_bytes(4096, "f16") == 0

    def test_n_kv_heads_none_falls_back_to_n_heads(self):
        b = _make_backend(
            n_layers=32, n_kv_heads=None, n_heads=32, embedding_length=4096
        )
        kv_fallback = b._estimate_kv_cache_bytes(4096, "f16")
        b2 = _make_backend(
            n_layers=32, n_kv_heads=32, n_heads=32, embedding_length=4096
        )
        kv_explicit = b2._estimate_kv_cache_bytes(4096, "f16")
        assert kv_fallback == kv_explicit

    def test_unknown_cache_type_defaults_to_f16(self):
        b = _llama3_70b()
        kv_unknown = b._estimate_kv_cache_bytes(4096, "some_new_type")
        kv_f16 = b._estimate_kv_cache_bytes(4096, "f16")
        assert kv_unknown == kv_f16

    def test_n_ctx_zero_returns_zero(self):
        b = _llama3_70b()
        assert b._estimate_kv_cache_bytes(0, "f16") == 0

    def test_n_ctx_negative_returns_zero(self):
        b = _llama3_70b()
        assert b._estimate_kv_cache_bytes(-100, "f16") == 0

    def test_q8_0_bpe_constant(self):
        """Bug 5: q8_0 should use 34/32 = 1.0625, not 1.125."""
        b = _make_backend(
            n_layers=32, n_kv_heads=8, n_heads=32, embedding_length=4096
        )
        head_dim = 4096 // 32  # 128
        n_ctx = 8192
        kv = b._estimate_kv_cache_bytes(n_ctx, "q8_0")
        # Correct: 2 * 8 * 128 * 32 * 8192 * (34/32)
        expected_correct = int(2 * 8 * 128 * 32 * n_ctx * (34 / 32))
        # Bug value: 2 * 8 * 128 * 32 * 8192 * 1.125
        expected_buggy = int(2 * 8 * 128 * 32 * n_ctx * 1.125)
        # The test checks correct behavior -- will FAIL on buggy code
        assert kv == expected_correct, (
            f"q8_0 BPE should be 34/32=1.0625, got estimate {kv} "
            f"(correct={expected_correct}, buggy_1.125={expected_buggy})"
        )

    def test_cross_check_qwen35_9b_262k_f16(self):
        """Cross-check: Qwen3.5-9B at 262k f16."""
        b = _qwen35_9b()
        kv = b._estimate_kv_cache_bytes(262144, "f16")
        head_dim = 3584 // 32  # 112
        expected = 2 * 8 * head_dim * 48 * 262144 * 2
        assert kv == expected


class TestFitContextToVram:
    """Tests for _fit_context_to_vram."""

    def test_requested_ctx_fits(self):
        """If model + KV fits, return requested_ctx unchanged."""
        b = _make_backend(
            n_layers=32, n_kv_heads=8, n_heads=32, embedding_length=4096
        )
        # Small model, huge VRAM -- should fit
        result = b._fit_context_to_vram(
            requested_ctx=8192,
            available_mib=48000,
            model_size_bytes=5 * GIB,
            cache_type_kv="f16",
        )
        assert result == 8192

    def test_context_capped_to_fit(self):
        """When full context doesn't fit, cap it."""
        b = _make_backend(
            n_layers=32, n_kv_heads=8, n_heads=32, embedding_length=4096
        )
        result = b._fit_context_to_vram(
            requested_ctx=131072,
            available_mib=24000,
            model_size_bytes=10 * GIB,
            cache_type_kv="f16",
        )
        assert result < 131072
        assert result >= 2048
        assert result % 256 == 0  # alignment check

    def test_weights_exceed_budget_returns_requested_ctx(self):
        """Bug 3: When weights alone exceed 70% budget, return requested_ctx, not min_ctx."""
        b = _make_backend(
            n_layers=32, n_kv_heads=8, n_heads=32, embedding_length=4096
        )
        # 20GB model, 22GB GPU => budget = 22*1024*0.70 MiB = ~15.4 GiB
        # Model exceeds budget, so context reduction is pointless
        result = b._fit_context_to_vram(
            requested_ctx=32768,
            available_mib=22 * 1024,  # 22 GiB
            model_size_bytes=20 * GIB,
            cache_type_kv="f16",
        )
        # BUG: returns min_ctx=2048 instead of requested_ctx=32768
        assert result == 32768, (
            f"When weights exceed budget, _fit_context_to_vram should return "
            f"requested_ctx={32768}, got {result}"
        )

    def test_below_2048_not_inflated(self):
        """Bug 4: If requested_ctx < 2048, result should not exceed requested_ctx.

        Use a tight budget where the requested context doesn't fit, so the
        binary search is triggered.  The bug is lo=min_ctx=2048 > hi=requested_ctx,
        causing the search to skip and return best=2048 which is LARGER than requested.
        """
        b = _make_backend(
            n_layers=32, n_kv_heads=8, n_heads=32, embedding_length=4096
        )
        # KV at 1024 f16 = 128 MiB.  Make remaining budget < 128 MiB so
        # the binary search is entered.
        # budget = 4096 * 0.70 = 2867.2 MiB
        # model = 2767.2 MiB => remaining = 100 MiB < 128 MiB
        model_size = int(2767.2 * MIB)
        for requested in [128, 512, 1024, 1536]:
            result = b._fit_context_to_vram(
                requested_ctx=requested,
                available_mib=4096,
                model_size_bytes=model_size,
                cache_type_kv="f16",
            )
            assert result <= requested, (
                f"requested_ctx={requested}: result {result} should not "
                f"exceed requested_ctx (bug: lo=min_ctx=2048 > hi={requested})"
            )

    def test_exact_fit_boundary(self):
        """Model + KV for requested_ctx == budget exactly -> returns requested_ctx."""
        b = _make_backend(
            n_layers=32, n_kv_heads=8, n_heads=32, embedding_length=4096
        )
        kv = b._estimate_kv_cache_bytes(8192, "f16")
        model_size = 5 * GIB
        # Budget = available_mib * MIB * 0.70
        # We need model_size + kv == budget
        budget = model_size + kv
        available_mib = int(budget / (MIB * 0.70)) + 1  # +1 for rounding
        result = b._fit_context_to_vram(
            requested_ctx=8192,
            available_mib=available_mib,
            model_size_bytes=model_size,
            cache_type_kv="f16",
        )
        assert result == 8192

    def test_256_alignment(self):
        """Capped value should be a multiple of 256."""
        b = _make_backend(
            n_layers=32, n_kv_heads=8, n_heads=32, embedding_length=4096
        )
        result = b._fit_context_to_vram(
            requested_ctx=100000,
            available_mib=16000,
            model_size_bytes=8 * GIB,
            cache_type_kv="f16",
        )
        if result < 100000:
            assert result % 256 == 0, f"Capped context {result} is not 256-aligned"

    def test_metadata_unavailable_returns_requested(self):
        """When _can_estimate_kv() is False, return requested_ctx unchanged."""
        b = _make_backend(n_layers=None, n_kv_heads=None, n_heads=None, embedding_length=None)
        result = b._fit_context_to_vram(
            requested_ctx=65536,
            available_mib=24000,
            model_size_bytes=10 * GIB,
            cache_type_kv="f16",
        )
        assert result == 65536


class TestSelectGpus:
    """Tests for _select_gpus."""

    def test_empty_gpu_list(self):
        indices, use_fit = LlamaCppBackend._select_gpus(10 * GIB, [])
        assert indices is None
        assert use_fit is True

    def test_single_gpu_fits(self):
        # 10 GiB model, 24 GiB free (70% = ~16.8 GiB)
        indices, use_fit = LlamaCppBackend._select_gpus(
            10 * GIB, [(0, 24 * 1024)]
        )
        assert indices == [0]
        assert use_fit is False

    def test_single_gpu_too_small(self):
        # 20 GiB model, 24 GiB free (70% = ~16.8 GiB) - doesn't fit on 1
        indices, use_fit = LlamaCppBackend._select_gpus(
            20 * GIB, [(0, 24 * 1024)]
        )
        assert indices is None
        assert use_fit is True

    def test_multi_gpu_cumulative_fit(self):
        # 30 GiB model, 2x 24 GiB GPUs (70% = ~33.6 GiB total)
        indices, use_fit = LlamaCppBackend._select_gpus(
            30 * GIB, [(0, 24 * 1024), (1, 24 * 1024)]
        )
        assert indices == [0, 1]
        assert use_fit is False

    def test_all_gpus_insufficient(self):
        # 100 GiB model, 2x 24 GiB GPUs (70% = ~33.6 GiB)
        indices, use_fit = LlamaCppBackend._select_gpus(
            100 * GIB, [(0, 24 * 1024), (1, 24 * 1024)]
        )
        assert indices is None
        assert use_fit is True

    def test_heterogeneous_gpus(self):
        # 20 GiB model, 16 GiB + 24 GiB GPUs
        # Single: 24*1024*0.7 = 17203 MiB = ~16.8 GiB < 20 GiB
        # Cumulative: (24+16)*1024*0.7 = 28672 MiB = 28 GiB > 20 GiB
        indices, use_fit = LlamaCppBackend._select_gpus(
            20 * GIB, [(0, 16 * 1024), (1, 24 * 1024)]
        )
        assert indices is not None
        assert sorted(indices) == [0, 1]
        assert use_fit is False


# ==========================================================================
#  Layer 2: Integration tests on load_model
# ==========================================================================


class TestLoadModelIntegration:
    """Test load_model with monkeypatched externals to capture command lines."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        """Create a fake GGUF file and fake binary for all integration tests."""
        self.fake_gguf = tmp_path / "model.gguf"
        self.fake_gguf.write_bytes(b"\x00" * 1024)

        self.fake_binary = tmp_path / "llama-server"
        self.fake_binary.write_text("#!/bin/sh\n")
        self.fake_binary.chmod(0o755)

        self.captured_cmd = None
        self.captured_env = None

    def _run_load(self, backend, n_ctx=4096, cache_type_kv=None,
                  gpus=None, model_size=10 * GIB):
        """Run load_model with full monkeypatching. Returns (cmd, env)."""
        captured = {}

        class FakePopen:
            def __init__(self, cmd, **kw):
                captured["cmd"] = cmd
                captured["env"] = kw.get("env", {})
                self.stdout = iter([])
                self.returncode = 0

            def poll(self):
                return 0

            def terminate(self):
                pass

            def wait(self, **kw):
                pass

            def kill(self):
                pass

        with (
            patch.object(type(backend), "_find_llama_server_binary",
                         staticmethod(lambda: str(self.fake_binary))),
            patch.object(type(backend), "_get_gpu_free_memory",
                         staticmethod(lambda: gpus or [])),
            patch.object(type(backend), "_get_gguf_size_bytes",
                         staticmethod(lambda p: model_size)),
            patch.object(type(backend), "_read_gguf_metadata",
                         lambda self, p: None),
            patch.object(type(backend), "_wait_for_health",
                         lambda self, **kw: True),
            patch("subprocess.Popen", FakePopen),
        ):
            backend.load_model(
                gguf_path=str(self.fake_gguf),
                model_identifier="test-model",
                n_ctx=n_ctx,
                cache_type_kv=cache_type_kv,
            )

        self.captured_cmd = captured.get("cmd")
        self.captured_env = captured.get("env")
        return self.captured_cmd, self.captured_env

    def test_basic_command_construction(self):
        b = _make_backend()
        cmd, env = self._run_load(b, n_ctx=4096)
        assert "-c" in cmd
        idx = cmd.index("-c")
        assert cmd[idx + 1] == "4096"

    def test_n_ctx_zero_metadata_missing(self):
        """Bug 2: n_ctx=0 with _context_length=None should produce -c 0, not -c 4096."""
        b = _make_backend(context_length=None)
        cmd, env = self._run_load(b, n_ctx=0)
        idx = cmd.index("-c")
        # BUG: produces "-c 4096" because (self._context_length or 4096) = 4096
        assert cmd[idx + 1] == "0", (
            f"n_ctx=0 with missing metadata should produce -c 0, "
            f"got -c {cmd[idx + 1]}"
        )

    def test_multi_gpu_auto_cap(self):
        """Bug 1: Multi-GPU should use aggregate budget, not single-GPU max."""
        b = _make_backend(
            n_layers=80,
            n_kv_heads=8,
            n_heads=64,
            embedding_length=8192,
            context_length=131072,
        )
        # 2x 24 GiB GPUs, 18 GiB model, 131k f16 KV ~43 GiB total
        # Single GPU budget: 24*1024*0.7 = ~16.8 GiB (too small)
        # Multi GPU budget: 2*24*1024*0.7 = ~33.6 GiB (fits model, can do some context)
        gpus = [(0, 24 * 1024), (1, 24 * 1024)]

        # Use custom patching to preserve metadata
        captured = {}

        class FakePopen:
            def __init__(self, cmd, **kw):
                captured["cmd"] = cmd
                captured["env"] = kw.get("env", {})
                self.stdout = iter([])
                self.returncode = 0

            def poll(self):
                return 0

            def terminate(self):
                pass

            def wait(self, **kw):
                pass

            def kill(self):
                pass

        with (
            patch.object(type(b), "_find_llama_server_binary",
                         staticmethod(lambda: str(self.fake_binary))),
            patch.object(type(b), "_get_gpu_free_memory",
                         staticmethod(lambda: gpus)),
            patch.object(type(b), "_get_gguf_size_bytes",
                         staticmethod(lambda p: 18 * GIB)),
            patch.object(type(b), "_read_gguf_metadata",
                         lambda self, p: None),
            patch.object(type(b), "_wait_for_health",
                         lambda self, **kw: True),
            patch("subprocess.Popen", FakePopen),
        ):
            b.load_model(
                gguf_path=str(self.fake_gguf),
                model_identifier="test-model",
                n_ctx=131072,
                cache_type_kv="f16",
            )

        cmd = captured["cmd"]
        idx = cmd.index("-c")
        ctx_val = int(cmd[idx + 1])

        # BUG: auto-cap uses single-GPU budget (best_free_mib = max(free for _, free in gpus))
        # and caps to 2048. With correct multi-GPU aggregate budget, context would be much
        # higher than 2048 (though likely still capped below 131072).
        assert ctx_val > 2048, (
            f"Multi-GPU should not cap context to 2048. Got -c {ctx_val}. "
            f"Bug: load_model uses single-GPU budget (best_free_mib) instead "
            f"of aggregate multi-GPU budget."
        )

    def test_context_length_not_overwritten(self):
        """Bug 6: _context_length should not be overwritten with capped value."""
        b = _make_backend(
            n_layers=80,
            n_kv_heads=8,
            n_heads=64,
            embedding_length=8192,
            context_length=262144,
        )
        gpus = [(0, 24 * 1024)]

        class FakePopen:
            def __init__(self, cmd, **kw):
                self.stdout = iter([])
                self.returncode = 0

            def poll(self):
                return 0

            def terminate(self):
                pass

            def wait(self, **kw):
                pass

            def kill(self):
                pass

        with (
            patch.object(type(b), "_find_llama_server_binary",
                         staticmethod(lambda: str(self.fake_binary))),
            patch.object(type(b), "_get_gpu_free_memory",
                         staticmethod(lambda: gpus)),
            patch.object(type(b), "_get_gguf_size_bytes",
                         staticmethod(lambda p: 5 * GIB)),
            patch.object(type(b), "_read_gguf_metadata",
                         lambda self, p: None),
            patch.object(type(b), "_wait_for_health",
                         lambda self, **kw: True),
            patch("subprocess.Popen", FakePopen),
        ):
            b.load_model(
                gguf_path=str(self.fake_gguf),
                model_identifier="test-model",
                n_ctx=262144,
                cache_type_kv="f16",
            )

        # BUG: self._context_length = effective_ctx overwrites native 262k
        assert b._context_length == 262144, (
            f"_context_length should remain at native 262144, "
            f"got {b._context_length} (capped value leaked)"
        )

    def test_fit_flag_when_gpus_none(self):
        """When GPU detection fails, --fit on should be in the command."""
        b = _make_backend()
        cmd, _ = self._run_load(b, n_ctx=4096, gpus=[])
        assert "--fit" in cmd
        assert "on" in cmd

    def test_ngl_minus_one_when_gpus_selected(self):
        """When model fits on GPU(s), -ngl -1 should be in the command."""
        b = _make_backend()
        cmd, _ = self._run_load(b, n_ctx=4096, gpus=[(0, 24 * 1024)], model_size=5 * GIB)
        assert "-ngl" in cmd
        idx = cmd.index("-ngl")
        assert cmd[idx + 1] == "-1"

    def test_cuda_visible_devices_set(self):
        """CUDA_VISIBLE_DEVICES in env should match selected GPUs."""
        b = _make_backend()
        cmd, env = self._run_load(b, n_ctx=4096, gpus=[(0, 24 * 1024), (1, 16 * 1024)],
                                  model_size=20 * GIB)
        if "CUDA_VISIBLE_DEVICES" in env:
            gpu_ids = env["CUDA_VISIBLE_DEVICES"].split(",")
            assert all(g.strip().isdigit() for g in gpu_ids)

    def test_gpu_exception_graceful_fallback(self):
        """If GPU detection raises, fall back to --fit."""
        b = _make_backend()

        class FakePopen:
            def __init__(self, cmd, **kw):
                self._cmd = cmd
                self.stdout = iter([])
                self.returncode = 0

            def poll(self):
                return 0

            def terminate(self):
                pass

            def wait(self, **kw):
                pass

            def kill(self):
                pass

        def _raise_gpu(*a, **kw):
            raise RuntimeError("nvidia-smi failed")

        with (
            patch.object(type(b), "_find_llama_server_binary",
                         staticmethod(lambda: str(self.fake_binary))),
            patch.object(type(b), "_get_gpu_free_memory", staticmethod(_raise_gpu)),
            patch.object(type(b), "_get_gguf_size_bytes",
                         staticmethod(lambda p: 10 * GIB)),
            patch.object(type(b), "_read_gguf_metadata",
                         lambda self, p: None),
            patch.object(type(b), "_wait_for_health",
                         lambda self, **kw: True),
            patch("subprocess.Popen", FakePopen),
        ):
            b.load_model(
                gguf_path=str(self.fake_gguf),
                model_identifier="test-model",
                n_ctx=4096,
            )
        # If we get here without exception, the fallback worked


# ==========================================================================
#  Layer 3: Real GGUF binary parsing
# ==========================================================================


def _make_minimal_gguf(path, arch="llama", ctx=4096, n_layers=32,
                       n_heads=32, n_kv_heads=8, embedding_length=4096):
    """Create a minimal valid GGUF file with the given metadata."""
    with open(path, "wb") as f:
        # Magic: GGUF
        f.write(_struct.pack("<I", 0x46554747))
        # Version 3
        f.write(_struct.pack("<I", 3))
        # Tensor count: 0
        f.write(_struct.pack("<Q", 0))

        # Build KV pairs
        kvs = []

        def _add_string_kv(key, value):
            kvs.append((key, 8, value))  # 8 = STRING type

        def _add_uint32_kv(key, value):
            kvs.append((key, 4, value))  # 4 = UINT32 type

        _add_string_kv("general.architecture", arch)
        _add_uint32_kv(f"{arch}.context_length", ctx)
        _add_uint32_kv(f"{arch}.block_count", n_layers)
        _add_uint32_kv(f"{arch}.attention.head_count", n_heads)
        _add_uint32_kv(f"{arch}.attention.head_count_kv", n_kv_heads)
        _add_uint32_kv(f"{arch}.embedding_length", embedding_length)

        # KV count
        f.write(_struct.pack("<Q", len(kvs)))

        # Write KV pairs
        for key, vtype, value in kvs:
            key_bytes = key.encode("utf-8")
            f.write(_struct.pack("<Q", len(key_bytes)))
            f.write(key_bytes)
            f.write(_struct.pack("<I", vtype))
            if vtype == 8:  # STRING
                val_bytes = value.encode("utf-8")
                f.write(_struct.pack("<Q", len(val_bytes)))
                f.write(val_bytes)
            elif vtype == 4:  # UINT32
                f.write(_struct.pack("<I", value))


class TestReadGgufMetadata:
    """Tests for _read_gguf_metadata with real binary files."""

    def test_parse_llama_architecture(self, tmp_path):
        gguf = tmp_path / "llama.gguf"
        _make_minimal_gguf(gguf, arch="llama", ctx=131072, n_layers=80,
                           n_heads=64, n_kv_heads=8, embedding_length=8192)
        b = _make_backend()
        b._read_gguf_metadata(str(gguf))
        assert b._context_length == 131072
        assert b._n_layers == 80
        assert b._n_heads == 64
        assert b._n_kv_heads == 8
        assert b._embedding_length == 8192

    @pytest.mark.parametrize(
        "arch,ctx,layers,heads,kv_heads,emb",
        [
            ("qwen2", 32768, 24, 14, 2, 896),
            ("qwen3", 40960, 36, 16, 4, 2560),
            ("gemma3", 131072, 44, 8, 4, 3072),
            ("phi3", 4096, 32, 32, 32, 3072),
            ("llama", 8192, 32, 32, 8, 4096),
        ],
    )
    def test_parse_multiple_architectures(self, tmp_path, arch, ctx, layers,
                                          heads, kv_heads, emb):
        gguf = tmp_path / f"{arch}.gguf"
        _make_minimal_gguf(gguf, arch=arch, ctx=ctx, n_layers=layers,
                           n_heads=heads, n_kv_heads=kv_heads,
                           embedding_length=emb)
        b = _make_backend()
        b._read_gguf_metadata(str(gguf))
        assert b._context_length == ctx
        assert b._n_layers == layers
        assert b._n_heads == heads
        assert b._n_kv_heads == kv_heads
        assert b._embedding_length == emb

    def test_all_fields_not_none(self, tmp_path):
        gguf = tmp_path / "test.gguf"
        _make_minimal_gguf(gguf)
        b = _make_backend()
        b._read_gguf_metadata(str(gguf))
        assert b._n_layers is not None
        assert b._n_heads is not None
        assert b._n_kv_heads is not None
        assert b._embedding_length is not None
        assert b._context_length is not None

    def test_malformed_gguf_graceful(self, tmp_path):
        """Malformed GGUF (bad magic) -> fields stay None, no crash."""
        bad = tmp_path / "bad.gguf"
        bad.write_bytes(b"\x00\x00\x00\x00" + b"\x00" * 100)
        b = _make_backend()
        b._read_gguf_metadata(str(bad))
        assert b._context_length is None
        assert b._n_layers is None

    def test_truncated_gguf_graceful(self, tmp_path):
        """Truncated GGUF -> fields stay None, no crash."""
        bad = tmp_path / "truncated.gguf"
        # Valid magic but truncated before header finishes
        bad.write_bytes(_struct.pack("<I", 0x46554747) + b"\x03\x00\x00\x00")
        b = _make_backend()
        b._read_gguf_metadata(str(bad))
        # Should not crash -- fields may be None


class TestCanEstimateKv:
    """Tests for _can_estimate_kv."""

    def test_all_present(self):
        b = _llama3_70b()
        assert b._can_estimate_kv() is True

    def test_missing_layers(self):
        b = _llama3_70b()
        b._n_layers = None
        assert b._can_estimate_kv() is False

    def test_missing_embedding(self):
        b = _llama3_70b()
        b._embedding_length = None
        assert b._can_estimate_kv() is False

    def test_missing_both_heads(self):
        b = _make_backend(n_layers=32, n_kv_heads=None, n_heads=None, embedding_length=4096)
        assert b._can_estimate_kv() is False

    def test_n_kv_heads_none_but_n_heads_present(self):
        b = _make_backend(n_layers=32, n_kv_heads=None, n_heads=32, embedding_length=4096)
        assert b._can_estimate_kv() is True
