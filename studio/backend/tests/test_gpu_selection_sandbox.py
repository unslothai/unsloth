#!/usr/bin/env python3
"""
Sandbox test for multi-GPU selection logic.

Tests the core GPU selection, memory estimation, and device_map logic
in an isolated environment. Can be run on Linux, macOS, and Windows
without requiring actual GPUs -- all hardware calls are mocked.

Usage:
    python -m pytest studio/backend/tests/test_gpu_selection_sandbox.py -v
    # or directly:
    python studio/backend/tests/test_gpu_selection_sandbox.py
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure backend is on sys.path
_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))


def _make_fake_config(
    vocab_size = 32000,
    hidden_size = 4096,
    intermediate_size = 11008,
    num_hidden_layers = 32,
    num_attention_heads = 32,
    num_key_value_heads = 8,
    tie_word_embeddings = False,
):
    """Create a fake HF config-like object for estimation tests."""
    from types import SimpleNamespace

    return SimpleNamespace(
        vocab_size = vocab_size,
        hidden_size = hidden_size,
        intermediate_size = intermediate_size,
        num_hidden_layers = num_hidden_layers,
        num_attention_heads = num_attention_heads,
        num_key_value_heads = num_key_value_heads,
        tie_word_embeddings = tie_word_embeddings,
    )


class TestEstimateFP16ModelSizeFromConfig(unittest.TestCase):
    """Test the config-based model size estimation."""

    def test_llama_8b_size_reasonable(self):
        from utils.hardware.hardware import _estimate_fp16_model_size_bytes_from_config

        config = _make_fake_config(
            vocab_size = 128256,
            hidden_size = 4096,
            intermediate_size = 14336,
            num_hidden_layers = 32,
            num_attention_heads = 32,
            num_key_value_heads = 8,
            tie_word_embeddings = False,
        )
        size = _estimate_fp16_model_size_bytes_from_config(config)
        self.assertIsNotNone(size)
        size_gb = size / (1024**3)
        # Llama 3.1 8B should be ~15GB in fp16
        self.assertGreater(size_gb, 12)
        self.assertLess(size_gb, 20)

    def test_small_model(self):
        from utils.hardware.hardware import _estimate_fp16_model_size_bytes_from_config

        config = _make_fake_config(
            vocab_size = 32000,
            hidden_size = 2048,
            intermediate_size = 5504,
            num_hidden_layers = 22,
            num_attention_heads = 32,
            num_key_value_heads = 4,
        )
        size = _estimate_fp16_model_size_bytes_from_config(config)
        self.assertIsNotNone(size)
        size_gb = size / (1024**3)
        # ~1B model should be ~2GB in fp16
        self.assertGreater(size_gb, 1)
        self.assertLess(size_gb, 5)

    def test_returns_none_for_incomplete_config(self):
        from utils.hardware.hardware import _estimate_fp16_model_size_bytes_from_config
        from types import SimpleNamespace

        config = SimpleNamespace(vocab_size = 32000)  # Missing most fields
        size = _estimate_fp16_model_size_bytes_from_config(config)
        self.assertIsNone(size)

    def test_moe_model(self):
        from utils.hardware.hardware import _estimate_fp16_model_size_bytes_from_config
        from types import SimpleNamespace

        config = SimpleNamespace(
            vocab_size = 152064,
            hidden_size = 3584,
            intermediate_size = 18944,
            num_hidden_layers = 28,
            num_attention_heads = 28,
            num_key_value_heads = 4,
            tie_word_embeddings = False,
            num_local_experts = 64,
            moe_intermediate_size = 2560,
        )
        size = _estimate_fp16_model_size_bytes_from_config(config)
        self.assertIsNotNone(size)
        size_gb = size / (1024**3)
        # MoE model with 64 experts should be large
        self.assertGreater(size_gb, 50)


class TestEstimateRequiredModelMemory(unittest.TestCase):
    """Test memory requirement estimation."""

    def test_inference_fp16_uses_1_3x(self):
        from utils.hardware.hardware import estimate_required_model_memory_gb

        with patch(
            "utils.hardware.hardware.estimate_fp16_model_size_bytes",
            return_value = (10 * (1024**3), "config"),  # 10GB model
        ):
            required, meta = estimate_required_model_memory_gb(
                "test/model",
                training_type = None,  # inference
                load_in_4bit = False,
            )
            self.assertIsNotNone(required)
            self.assertAlmostEqual(required, 13.0, places = 0)
            self.assertEqual(meta["mode"], "inference")

    def test_inference_4bit_uses_reduced_estimate(self):
        from utils.hardware.hardware import estimate_required_model_memory_gb

        with patch(
            "utils.hardware.hardware.estimate_fp16_model_size_bytes",
            return_value = (30 * (1024**3), "config"),  # 30GB fp16 model
        ):
            required, meta = estimate_required_model_memory_gb(
                "test/model",
                training_type = None,  # inference
                load_in_4bit = True,
            )
            self.assertIsNotNone(required)
            # 4bit base = 30/3.2 = 9.375GB, required = 9.375 + max(9.375*0.3, 2) = 12.19GB
            self.assertAlmostEqual(required, 12.2, places = 0)

    def test_4bit_training_reduces_base(self):
        from utils.hardware.hardware import estimate_required_model_memory_gb

        with patch(
            "utils.hardware.hardware.estimate_fp16_model_size_bytes",
            return_value = (30 * (1024**3), "config"),  # 30GB fp16 model
        ):
            required, meta = estimate_required_model_memory_gb(
                "test/model",
                training_type = "LoRA/QLoRA",
                load_in_4bit = True,
            )
            self.assertIsNotNone(required)
            # fallback: base=30/3.2=9.375, lora=30*0.04=1.2, act=30*0.15=4.5, cuda=1.4
            self.assertAlmostEqual(required, 16.5, places = 0)

    def test_full_finetune_uses_3_5x(self):
        from utils.hardware.hardware import estimate_required_model_memory_gb

        with patch(
            "utils.hardware.hardware.estimate_fp16_model_size_bytes",
            return_value = (10 * (1024**3), "config"),  # 10GB model
        ):
            required, meta = estimate_required_model_memory_gb(
                "test/model",
                training_type = "Full Finetuning",
            )
            self.assertIsNotNone(required)
            # fallback: 10 * 3.5 + 1.4 cuda overhead = 36.4
            self.assertAlmostEqual(required, 36.4, places = 0)

    def test_returns_none_when_unavailable(self):
        from utils.hardware.hardware import estimate_required_model_memory_gb

        with patch(
            "utils.hardware.hardware.estimate_fp16_model_size_bytes",
            return_value = (None, "unavailable"),
        ):
            required, meta = estimate_required_model_memory_gb("test/model")
            self.assertIsNone(required)


class TestAutoSelectGpuIds(unittest.TestCase):
    """Test automatic GPU selection based on model size and free memory."""

    def _make_utilization(self, devices):
        """Create a fake utilization response."""
        return {
            "available": True,
            "devices": [
                {
                    "index": idx,
                    "vram_total_gb": total,
                    "vram_used_gb": total - free,
                }
                for idx, total, free in devices
            ],
        }

    def test_single_gpu_sufficient(self):
        from utils.hardware.hardware import auto_select_gpu_ids
        import utils.hardware.hardware as hw

        with (
            patch.object(hw, "get_device", return_value = hw.DeviceType.CUDA),
            patch.object(
                hw,
                "estimate_required_model_memory_gb",
                return_value = (
                    10.0,
                    {
                        "mode": "inference",
                        "required_gb": 10.0,
                        "model_size_source": "config",
                        "model_size_gb": 7.7,
                    },
                ),
            ),
            patch.object(
                hw,
                "_get_parent_visible_gpu_spec",
                return_value = {
                    "raw": "0,1,2,3",
                    "numeric_ids": [0, 1, 2, 3],
                    "supports_explicit_gpu_ids": True,
                },
            ),
            patch.object(hw, "get_parent_visible_gpu_ids", return_value = [0, 1, 2, 3]),
            patch.object(
                hw,
                "get_visible_gpu_utilization",
                return_value = self._make_utilization(
                    [
                        (0, 80.0, 75.0),
                        (1, 80.0, 78.0),
                        (2, 80.0, 70.0),
                        (3, 80.0, 72.0),
                    ]
                ),
            ),
        ):
            selected, meta = auto_select_gpu_ids("test/model")
            # Should pick GPU 1 (most free memory: 78GB) -- enough for 10GB
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0], 1)

    def test_two_gpus_needed(self):
        from utils.hardware.hardware import auto_select_gpu_ids
        import utils.hardware.hardware as hw

        with (
            patch.object(hw, "get_device", return_value = hw.DeviceType.CUDA),
            patch.object(
                hw,
                "estimate_required_model_memory_gb",
                return_value = (
                    50.0,
                    {
                        "mode": "inference",
                        "required_gb": 50.0,
                        "model_size_source": "config",
                        "model_size_gb": 38.0,
                    },
                ),
            ),
            patch.object(
                hw,
                "_get_parent_visible_gpu_spec",
                return_value = {
                    "raw": "0,1",
                    "numeric_ids": [0, 1],
                    "supports_explicit_gpu_ids": True,
                },
            ),
            patch.object(hw, "get_parent_visible_gpu_ids", return_value = [0, 1]),
            patch.object(
                hw,
                "get_visible_gpu_utilization",
                return_value = self._make_utilization(
                    [
                        (0, 40.0, 30.0),  # 30GB free
                        (1, 40.0, 35.0),  # 35GB free
                    ]
                ),
            ),
        ):
            selected, meta = auto_select_gpu_ids("test/model")
            # 35GB (first) + 30*0.85 (second) = 60.5GB > 50GB
            self.assertEqual(len(selected), 2)

    def test_non_cuda_returns_none(self):
        from utils.hardware.hardware import auto_select_gpu_ids
        import utils.hardware.hardware as hw

        with patch.object(hw, "get_device", return_value = hw.DeviceType.CPU):
            selected, meta = auto_select_gpu_ids("test/model")
            self.assertIsNone(selected)
            self.assertEqual(meta["selection_mode"], "non_cuda")


class TestGetDeviceMap(unittest.TestCase):
    """Test device_map string generation."""

    def test_single_gpu_returns_sequential(self):
        from utils.hardware.hardware import get_device_map
        import utils.hardware.hardware as hw

        with (
            patch.object(hw, "get_device", return_value = hw.DeviceType.CUDA),
            patch.object(
                hw,
                "_get_parent_visible_gpu_spec",
                return_value = {
                    "raw": "0",
                    "numeric_ids": [0],
                    "supports_explicit_gpu_ids": True,
                },
            ),
            patch.object(hw, "get_visible_gpu_count", return_value = 1),
        ):
            dm = get_device_map(gpu_ids = [0])
            self.assertEqual(dm, "sequential")

    def test_multi_gpu_returns_balanced(self):
        from utils.hardware.hardware import get_device_map
        import utils.hardware.hardware as hw

        with patch.object(hw, "get_device", return_value = hw.DeviceType.CUDA):
            dm = get_device_map(gpu_ids = [0, 1])
            self.assertEqual(dm, "balanced")

    def test_cpu_returns_sequential(self):
        from utils.hardware.hardware import get_device_map
        import utils.hardware.hardware as hw

        with patch.object(hw, "get_device", return_value = hw.DeviceType.CPU):
            dm = get_device_map(gpu_ids = None)
            self.assertEqual(dm, "sequential")


class TestResolveRequestedGpuIds(unittest.TestCase):
    """Test GPU ID validation."""

    def test_none_returns_parent_visible(self):
        from utils.hardware.hardware import resolve_requested_gpu_ids

        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}, clear = False),
            patch("utils.hardware.hardware.get_physical_gpu_count", return_value = 8),
        ):
            result = resolve_requested_gpu_ids(None)
            self.assertEqual(result, [2, 3])

    def test_empty_list_returns_parent_visible(self):
        from utils.hardware.hardware import resolve_requested_gpu_ids

        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}, clear = False),
            patch("utils.hardware.hardware.get_physical_gpu_count", return_value = 8),
        ):
            result = resolve_requested_gpu_ids([])
            self.assertEqual(result, [2, 3])

    def test_duplicates_rejected(self):
        from utils.hardware.hardware import resolve_requested_gpu_ids

        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2"}, clear = False),
            patch("utils.hardware.hardware.get_physical_gpu_count", return_value = 8),
        ):
            with self.assertRaises(ValueError):
                resolve_requested_gpu_ids([1, 1])

    def test_out_of_range_rejected(self):
        from utils.hardware.hardware import resolve_requested_gpu_ids

        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}, clear = False),
            patch("utils.hardware.hardware.get_physical_gpu_count", return_value = 4),
        ):
            with self.assertRaises(ValueError):
                resolve_requested_gpu_ids([5])

    def test_uuid_env_var_rejects_explicit_ids(self):
        from utils.hardware.hardware import resolve_requested_gpu_ids

        with (
            patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-abc,GPU-def"}, clear = False
            ),
            patch("utils.hardware.hardware.get_physical_gpu_count", return_value = 8),
        ):
            with self.assertRaises(ValueError):
                resolve_requested_gpu_ids([0])


class TestApplyGpuIds(unittest.TestCase):
    """Test CUDA_VISIBLE_DEVICES environment variable setting."""

    def test_apply_list(self):
        from utils.hardware.hardware import apply_gpu_ids

        with patch.dict(os.environ, {}, clear = False):
            apply_gpu_ids([3, 5])
            self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "3,5")

    def test_apply_none_does_nothing(self):
        from utils.hardware.hardware import apply_gpu_ids

        original = os.environ.get("CUDA_VISIBLE_DEVICES")
        apply_gpu_ids(None)
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), original)


class TestMultiGpuOverheadAccounting(unittest.TestCase):
    """Test that multi-GPU overhead is applied correctly.

    The first GPU should keep its full free memory, and only
    additional GPUs should have the overhead factor applied.
    """

    def _make_utilization(self, devices):
        return {
            "available": True,
            "devices": [
                {
                    "index": idx,
                    "vram_total_gb": total,
                    "vram_used_gb": total - free,
                }
                for idx, total, free in devices
            ],
        }

    def test_first_gpu_not_penalized(self):
        """A model that just fits on 1 GPU should not require 2 GPUs."""
        from utils.hardware.hardware import auto_select_gpu_ids
        import utils.hardware.hardware as hw

        # Model requires 79GB, GPU has 80GB free
        with (
            patch.object(hw, "get_device", return_value = hw.DeviceType.CUDA),
            patch.object(
                hw,
                "estimate_required_model_memory_gb",
                return_value = (
                    79.0,
                    {
                        "mode": "inference",
                        "required_gb": 79.0,
                        "model_size_source": "config",
                        "model_size_gb": 60.0,
                    },
                ),
            ),
            patch.object(
                hw,
                "_get_parent_visible_gpu_spec",
                return_value = {
                    "raw": "0,1",
                    "numeric_ids": [0, 1],
                    "supports_explicit_gpu_ids": True,
                },
            ),
            patch.object(hw, "get_parent_visible_gpu_ids", return_value = [0, 1]),
            patch.object(
                hw,
                "get_visible_gpu_utilization",
                return_value = self._make_utilization(
                    [
                        (0, 80.0, 80.0),
                        (1, 80.0, 80.0),
                    ]
                ),
            ),
        ):
            selected, meta = auto_select_gpu_ids("test/model")
            # Should fit on 1 GPU (80GB >= 79GB)
            self.assertEqual(len(selected), 1)

    def test_second_gpu_has_overhead(self):
        """When 2 GPUs are needed, the second one's contribution is reduced."""
        from utils.hardware.hardware import auto_select_gpu_ids
        import utils.hardware.hardware as hw

        # Model requires 110GB. First GPU has 80GB, second has 40GB.
        # With overhead: 80 + 40*0.85 = 114GB -- just enough
        with (
            patch.object(hw, "get_device", return_value = hw.DeviceType.CUDA),
            patch.object(
                hw,
                "estimate_required_model_memory_gb",
                return_value = (
                    110.0,
                    {
                        "mode": "inference",
                        "required_gb": 110.0,
                        "model_size_source": "config",
                        "model_size_gb": 85.0,
                    },
                ),
            ),
            patch.object(
                hw,
                "_get_parent_visible_gpu_spec",
                return_value = {
                    "raw": "0,1",
                    "numeric_ids": [0, 1],
                    "supports_explicit_gpu_ids": True,
                },
            ),
            patch.object(hw, "get_parent_visible_gpu_ids", return_value = [0, 1]),
            patch.object(
                hw,
                "get_visible_gpu_utilization",
                return_value = self._make_utilization(
                    [
                        (0, 80.0, 80.0),
                        (1, 80.0, 40.0),
                    ]
                ),
            ),
        ):
            selected, meta = auto_select_gpu_ids("test/model")
            # Should use both GPUs
            self.assertEqual(len(selected), 2)


if __name__ == "__main__":
    unittest.main()
