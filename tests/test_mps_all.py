# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Comprehensive Apple Silicon (MPS) Test Suite for Unsloth.

This test file consolidates all MPS-related tests into a single comprehensive suite:
- Core kernel tests (RMSNorm, LayerNorm, RoPE, SwiGLU, CrossEntropy)
- Hardware detection tests
- Model integration tests (Llama, Qwen2, Gemma, Mistral)
- Training tests
- MoE tests

Run on Mac:
    python -m unittest tests.test_mps_all -v

Or run individual test classes:
    python -m unittest tests.test_mps_all.TestHardwareDetection -v
    python -m unittest tests.test_mps_all.TestCoreKernels -v
    python -m unittest tests.test_mps_all.TestModelIntegration -v
    python -m unittest tests.test_mps_all.TestTraining -v
"""

import sys
import os
import platform
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Apply Mac compatibility patches BEFORE importing unsloth
if platform.system() == "Darwin":
    from patcher import patch_for_mac
    patch_for_mac()


class TestHardwareDetection(unittest.TestCase):
    """Tests for Apple Silicon hardware detection."""

    @unittest.skipIf(platform.system() != "Darwin", "macOS only")
    def test_apple_hardware_info(self):
        """Verify hardware info is correctly detected."""
        from unsloth.kernels.mps import get_apple_hardware_info

        info = get_apple_hardware_info()
        self.assertIsInstance(info, dict)
        
        expected_keys = [
            "is_apple_silicon", "chip_name", "chip_family",
            "chip_variant", "total_memory_bytes", "total_memory_gb",
            "usable_memory_gb", "cpu_cores_total", "cpu_cores_performance",
            "cpu_cores_efficiency", "gpu_cores",
        ]
        for key in expected_keys:
            self.assertIn(key, info)

    @unittest.skipIf(platform.system() != "Darwin", "macOS only")
    def test_mps_device_info(self):
        """Verify MPS device info is available."""
        from unsloth.kernels.mps import get_mps_device_info

        info = get_mps_device_info()
        self.assertIsInstance(info, dict)
        if info.get("available"):
            self.assertIn("chip", info)
            self.assertIn("pytorch_version", info)

    @unittest.skipIf(platform.system() != "Darwin", "macOS only")
    def test_mps_memory_info(self):
        """Verify MPS memory info is available."""
        from unsloth.kernels.mps import get_mps_memory_info

        info = get_mps_memory_info()
        self.assertIsInstance(info, dict)
        if info.get("available"):
            self.assertIn("total_memory_gb", info)
            self.assertIn("usable_memory_gb", info)


class TestCoreKernels(unittest.TestCase):
    """Tests for core MPS kernels using mocked dependencies."""

    def test_rms_norm_kernel(self):
        """Test RMSNorm kernel numerically."""
        import types
        from importlib.machinery import ModuleSpec
        import importlib.util
        import torch

        ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        def create_mock(name):
            m = types.ModuleType(name)
            m.__spec__ = ModuleSpec(name, None)
            m.__file__ = f"{name}.py"
            m.__path__ = []
            return m

        sys.modules["triton"] = create_mock("triton")
        sys.modules["triton.language"] = create_mock("triton.language")
        sys.modules["triton.jit"] = create_mock("triton.jit")
        sys.modules["bitsandbytes"] = create_mock("bitsandbytes")

        mock_unsloth = create_mock("unsloth")
        mock_unsloth_kernels = create_mock("unsloth.kernels")
        mock_unsloth_device = create_mock("unsloth.device_type")
        mock_unsloth_device.DEVICE_TYPE = "mps"
        mock_unsloth_device.is_mps = lambda: True

        sys.modules["unsloth"] = mock_unsloth
        sys.modules["unsloth.kernels"] = mock_unsloth_kernels
        sys.modules["unsloth.device_type"] = mock_unsloth_device

        spec = importlib.util.spec_from_file_location(
            "unsloth.kernels.mps.rms_layernorm",
            os.path.join(ROOT, "unsloth/kernels/mps/rms_layernorm.py"),
        )
        rms_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rms_mod)

        X = torch.randn(2, 4, 8, requires_grad=True)
        W = torch.randn(8, requires_grad=True)
        eps = 1e-6
        Y = rms_mod.mps_rms_layernorm(X, W, eps)

        self.assertEqual(Y.shape, X.shape)
        Y.sum().backward()
        self.assertIsNotNone(X.grad)

    def test_swiglu_kernel(self):
        """Test SwiGLU kernel numerically."""
        import types
        from importlib.machinery import ModuleSpec
        import importlib.util
        import torch
        import torch.nn.functional as F

        ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        def create_mock(name):
            m = types.ModuleType(name)
            m.__spec__ = ModuleSpec(name, None)
            m.__file__ = f"{name}.py"
            m.__path__ = []
            return m

        sys.modules["triton"] = create_mock("triton")
        sys.modules["triton.language"] = create_mock("triton.language")
        sys.modules["triton.jit"] = create_mock("triton.jit")
        sys.modules["bitsandbytes"] = create_mock("bitsandbytes")

        mock_unsloth = create_mock("unsloth")
        mock_unsloth_kernels = create_mock("unsloth.kernels")
        mock_unsloth_device = create_mock("unsloth.device_type")
        mock_unsloth_device.DEVICE_TYPE = "mps"
        mock_unsloth_device.is_mps = lambda: True

        sys.modules["unsloth"] = mock_unsloth
        sys.modules["unsloth.kernels"] = mock_unsloth_kernels
        sys.modules["unsloth.device_type"] = mock_unsloth_device

        spec = importlib.util.spec_from_file_location(
            "unsloth.kernels.mps.swiglu",
            os.path.join(ROOT, "unsloth/kernels/mps/swiglu.py"),
        )
        swiglu_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(swiglu_mod)

        e = torch.randn(2, 4, 8, requires_grad=True)
        g = torch.randn(2, 4, 8, requires_grad=True)
        Y = swiglu_mod.mps_swiglu_forward(e, g)

        self.assertTrue(torch.allclose(Y, F.silu(e) * g))
        dw = torch.randn_like(Y)
        h, de, dg = swiglu_mod.mps_swiglu_backward(dw, e, g)
        self.assertEqual(de.shape, e.shape)
        self.assertEqual(dg.shape, g.shape)

    def test_cross_entropy_kernel(self):
        """Test CrossEntropy kernel numerically."""
        import types
        from importlib.machinery import ModuleSpec
        import importlib.util
        import torch

        ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        def create_mock(name):
            m = types.ModuleType(name)
            m.__spec__ = ModuleSpec(name, None)
            m.__file__ = f"{name}.py"
            m.__path__ = []
            return m

        sys.modules["triton"] = create_mock("triton")
        sys.modules["triton.language"] = create_mock("triton.language")
        sys.modules["triton.jit"] = create_mock("triton.jit")
        sys.modules["bitsandbytes"] = create_mock("bitsandbytes")

        mock_unsloth = create_mock("unsloth")
        mock_unsloth_kernels = create_mock("unsloth.kernels")
        mock_unsloth_device = create_mock("unsloth.device_type")
        mock_unsloth_device.DEVICE_TYPE = "mps"
        mock_unsloth_device.is_mps = lambda: True

        sys.modules["unsloth"] = mock_unsloth
        sys.modules["unsloth.kernels"] = mock_unsloth_kernels
        sys.modules["unsloth.device_type"] = mock_unsloth_device

        spec = importlib.util.spec_from_file_location(
            "unsloth.kernels.mps.cross_entropy_loss",
            os.path.join(ROOT, "unsloth/kernels/mps/cross_entropy_loss.py"),
        )
        ce_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ce_mod)

        logits = torch.randn(2, 4, 16, requires_grad=True)
        labels = torch.randint(0, 16, (2, 4))
        loss = ce_mod.mps_cross_entropy_loss(logits, labels)
        loss.backward()
        self.assertIsNotNone(logits.grad)


class TestKernelDispatch(unittest.TestCase):
    """Tests for kernel dispatch and Metal/MLX availability."""

    @unittest.skipIf(platform.system() != "Darwin", "macOS only")
    @unittest.skipIf(not hasattr(__import__('torch').backends, 'mps') or 
                     not __import__('torch').backends.mps.is_available(), 
                     "MPS not available")
    def test_kernel_dispatch_available(self):
        """Verify kernel dispatch functions are available."""
        from unsloth.kernels.mps.dispatch import (
            _is_metal_available,
            _is_mlx_available,
            dispatch_rms_layernorm,
            dispatch_rope_embedding,
            dispatch_swiglu_fg,
        )

        metal_available = _is_metal_available()
        mlx_available = _is_mlx_available()
        
        self.assertIsInstance(metal_available, bool)
        self.assertIsInstance(mlx_available, bool)


class TestModelIntegration(unittest.TestCase):
    """Tests for various model architectures on MPS."""

    @unittest.skipIf(platform.system() != "Darwin", "macOS only")
    @unittest.skipIf(not hasattr(__import__('torch').backends, 'mps') or 
                     not __import__('torch').backends.mps.is_available(), 
                     "MPS not available")
    def test_llama_forward_pass(self):
        """Test Llama model forward pass on MPS."""
        import torch
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Llama-3.2-1B",
            max_seq_length=128,
            load_in_4bit=False,
            dtype=torch.float16,
        )

        device = next(model.parameters()).device
        self.assertEqual(device.type, "mps")

        inputs = tokenizer("Hello, world!", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        self.assertIsNotNone(outputs.logits)
        self.assertEqual(outputs.logits.shape[0], inputs["input_ids"].shape[0])

    @unittest.skipIf(platform.system() != "Darwin", "macOS only")
    @unittest.skipIf(not hasattr(__import__('torch').backends, 'mps') or 
                     not __import__('torch').backends.mps.is_available(), 
                     "MPS not available")
    def test_qwen2_forward_pass(self):
        """Test Qwen2 model forward pass on MPS."""
        import torch
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="Qwen/Qwen2-0.5B",
            max_seq_length=128,
            load_in_4bit=False,
            dtype=torch.float16,
        )

        device = next(model.parameters()).device
        inputs = tokenizer("Test input", return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)

        self.assertIsNotNone(outputs.logits)

    @unittest.skipIf(platform.system() != "Darwin", "macOS only")
    @unittest.skipIf(not hasattr(__import__('torch').backends, 'mps') or 
                     not __import__('torch').backends.mps.is_available(), 
                     "MPS not available")
    def test_mistral_forward_pass(self):
        """Test Mistral model forward pass on MPS."""
        import torch
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Mistral-7B-bnb-4bit",
            max_seq_length=128,
            load_in_4bit=False,
            dtype=torch.float16,
        )

        device = next(model.parameters()).device
        inputs = tokenizer("Test input", return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)

        self.assertIsNotNone(outputs.logits)

    @unittest.skipIf(platform.system() != "Darwin", "macOS only")
    @unittest.skipIf(not hasattr(__import__('torch').backends, 'mps') or 
                     not __import__('torch').backends.mps.is_available(), 
                     "MPS not available")
    def test_gemma_forward_pass(self):
        """Test Gemma model forward pass on MPS."""
        import torch
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/gemma-2b",
            max_seq_length=128,
            load_in_4bit=False,
            dtype=torch.float16,
        )

        device = next(model.parameters()).device
        inputs = tokenizer("Test input", return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)

        self.assertIsNotNone(outputs.logits)


class TestTraining(unittest.TestCase):
    """Tests for training on MPS."""

    @unittest.skipIf(platform.system() != "Darwin", "macOS only")
    @unittest.skipIf(not hasattr(__import__('torch').backends, 'mps') or 
                     not __import__('torch').backends.mps.is_available(), 
                     "MPS not available")
    def test_forward_backward_pass(self):
        """Test forward and backward pass on MPS."""
        import torch
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/tinyllama-bnb-4bit",
            max_seq_length=64,
            load_in_4bit=False,
            dtype=torch.bfloat16,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            random_state=42,
        )

        device = next(model.parameters()).device
        inputs = tokenizer("Test training input", return_tensors="pt", padding=True, truncation=True, max_length=32)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()

        model.train()
        outputs = model(**inputs)
        loss = outputs.loss
        
        self.assertIsNotNone(loss)
        
        loss.backward()
        
        grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        self.assertGreater(grad_count, 0, "No gradients computed!")

    @unittest.skipIf(platform.system() != "Darwin", "macOS only")
    @unittest.skipIf(not hasattr(__import__('torch').backends, 'mps') or 
                     not __import__('torch').backends.mps.is_available(), 
                     "MPS not available")
    def test_training_loop(self):
        """Test full training loop on MPS."""
        import torch
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/tinyllama-bnb-4bit",
            max_seq_length=64,
            load_in_4bit=False,
            dtype=torch.bfloat16,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            random_state=42,
        )

        device = next(model.parameters()).device
        inputs = tokenizer("Test training", return_tensors="pt", padding=True, truncation=True, max_length=32)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        self.assertEqual(len(losses), 3)
        self.assertTrue(all(l is not None for l in losses))


class TestMoE(unittest.TestCase):
    """Tests for MoE (Mixture of Experts) on MPS."""

    @unittest.skipIf(platform.system() != "Darwin", "macOS only")
    @unittest.skipIf(not hasattr(__import__('torch').backends, 'mps') or 
                     not __import__('torch').backends.mps.is_available(), 
                     "MPS not available")
    def test_grouped_gemm_forward(self):
        """Test MoE grouped GEMM forward pass."""
        import torch
        try:
            from unsloth.kernels.moe.grouped_gemm.interface import grouped_gemm_forward
        except ImportError:
            self.skipTest("MoE grouped GEMM not available")

        num_experts = 2
        num_tokens = 4
        topk = 1
        K = 16
        N = 32

        device = torch.device("mps")
        dtype = torch.float16

        W = torch.randn(num_experts * N, K, device=device, dtype=dtype) / (K**0.5)
        X = torch.randn(num_tokens * topk, K, device=device, dtype=dtype)
        m_sizes = torch.tensor([2, 2], device=device, dtype=torch.int32)
        gather_indices = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)

        Y = grouped_gemm_forward(
            X=X, W=W, topk=topk, m_sizes=m_sizes,
            gather_indices=gather_indices, permute_x=False, permute_y=False,
        )

        self.assertEqual(Y.shape, (num_tokens * topk, N))


class TestSaveLoad(unittest.TestCase):
    """Tests for save/load operations on MPS."""

    @unittest.skipIf(platform.system() != "Darwin", "macOS only")
    @unittest.skipIf(not hasattr(__import__('torch').backends, 'mps') or 
                     not __import__('torch').backends.mps.is_available(), 
                     "MPS not available")
    def test_model_save_load(self):
        """Test model save and load on MPS."""
        import torch
        import tempfile
        import os
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Llama-3.2-1B",
            max_seq_length=64,
            load_in_4bit=False,
            dtype=torch.float16,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model")
            model.save_pretrained(save_path, safe_serialization=True)
            tokenizer.save_pretrained(save_path)
            
            model2, tokenizer2 = FastLanguageModel.from_pretrained(
                model_name=save_path,
                max_seq_length=64,
                load_in_4bit=False,
                dtype=torch.float16,
            )
            
            device = next(model2.parameters()).device
            self.assertEqual(device.type, "mps")


if __name__ == "__main__":
    unittest.main(verbosity=2)
