# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for CGGR (Confidence-Gated Gradient Routing) integration with Unsloth.
"""

# ruff: noqa
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pytest
import torch
import torch.nn as nn

# Import directly from cggr submodules to avoid full unsloth import chain
# which requires unsloth_zoo and other external dependencies
sys.path.insert(0, str(REPO_ROOT / "unsloth"))
from cggr.router import TruncatedRouter, create_truncated_router
from cggr.bridge import CGGRUnslothBridge, patch_trainer_for_cggr


class MockDecoderLayer(nn.Module):
    """Mock decoder layer for testing."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = nn.Linear(hidden_size, hidden_size)
        self.mlp = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, hidden_states, attention_mask=None, use_cache=False, **kwargs):
        hidden_states = self.norm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return (hidden_states,)


class MockModel(nn.Module):
    """Mock language model for testing CGGR router creation."""
    
    def __init__(self, vocab_size: int = 1000, hidden_size: int = 64, num_layers: int = 4):
        super().__init__()
        self.config = type("Config", (), {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
        })()
        
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            MockDecoderLayer(hidden_size) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return type("Output", (), {"loss": loss, "logits": logits})()


class MockTrainer:
    """Mock trainer for testing CGGR patching."""
    
    def __init__(self, model):
        self.model = model
        self.compute_loss_calls = []
    
    def compute_loss(self, model, inputs, *args, **kwargs):
        self.compute_loss_calls.append(inputs.copy())
        outputs = model(**inputs)
        return outputs.loss


class TestCGGRRouter:
    """Tests for truncated router creation."""
    
    def test_router_creation(self):
        """Test that truncated router is created correctly."""
        model = MockModel(vocab_size=100, hidden_size=32, num_layers=4)
        router = create_truncated_router(model, num_layers=2)
        
        assert isinstance(router, TruncatedRouter)
        assert router.num_layers == 2
        assert len(router.layers) == 2
    
    def test_router_shares_weights(self):
        """Test that router shares weights with parent model (no extra memory)."""
        model = MockModel(vocab_size=100, hidden_size=32, num_layers=4)
        router = create_truncated_router(model, num_layers=2)
        
        # Check that embedding weights are identical (same object)
        assert router.embed_tokens is model.embed_tokens
        assert router.lm_head is model.lm_head
        
        # Check that layer weights are identical
        for i in range(2):
            assert router.layers[i] is model.layers[i]
    
    def test_router_forward(self):
        """Test router forward pass produces valid logits."""
        model = MockModel(vocab_size=100, hidden_size=32, num_layers=4)
        router = create_truncated_router(model, num_layers=2)
        
        input_ids = torch.randint(0, 100, (2, 16))  # batch=2, seq_len=16
        
        with torch.no_grad():
            logits = router(input_ids)
        
        assert logits.shape == (2, 16, 100)  # batch, seq_len, vocab_size


class TestCGGRBridge:
    """Tests for CGGR bridge and trainer patching."""
    
    def test_bridge_initialization(self):
        """Test CGGRUnslothBridge initialization."""
        model = MockModel()
        bridge = CGGRUnslothBridge(
            model=model,
            min_tokens_ratio=0.25,
            num_router_layers=2,
            warmup_steps=10,
        )
        
        assert bridge.min_tokens_ratio == 0.25
        assert bridge.num_router_layers == 2
        assert bridge.warmup_steps == 10
        assert bridge.current_step == 0
    
    def test_label_masking_during_warmup(self):
        """Test that labels are not masked during warmup period."""
        model = MockModel()
        bridge = CGGRUnslothBridge(
            model=model,
            min_tokens_ratio=0.25,
            warmup_steps=100,
        )
        
        input_ids = torch.randint(0, 100, (2, 16))
        labels = torch.randint(0, 100, (2, 16))
        
        # During warmup, labels should not be modified
        masked_labels = bridge.mask_easy_tokens(input_ids, labels)
        assert torch.equal(masked_labels, labels)
    
    def test_label_masking_after_warmup(self):
        """Test that easy tokens are masked after warmup."""
        model = MockModel()
        bridge = CGGRUnslothBridge(
            model=model,
            min_tokens_ratio=0.25,
            warmup_steps=0,  # No warmup for this test
            dynamic_threshold=False,
        )
        
        input_ids = torch.randint(0, 100, (1, 100))
        labels = torch.randint(0, 100, (1, 100))
        
        masked_labels = bridge.mask_easy_tokens(input_ids, labels.clone())
        
        # Some tokens should be masked (set to -100)
        num_masked = (masked_labels == -100).sum().item()
        num_valid = (labels != -100).sum().item()
        
        # Expect roughly (1 - min_tokens_ratio) tokens to be masked
        expected_masked = int(num_valid * (1 - bridge.min_tokens_ratio))
        assert num_masked >= expected_masked * 0.5  # Allow some variance
    
    def test_trainer_patching(self):
        """Test that trainer is properly patched."""
        model = MockModel()
        trainer = MockTrainer(model)
        
        original_compute_loss = trainer.compute_loss
        
        bridge = CGGRUnslothBridge.patch_trainer(
            trainer,
            min_tokens_ratio=0.25,
            warmup_steps=0,
        )
        
        # Trainer should have been patched
        assert trainer.compute_loss is not original_compute_loss
        assert hasattr(trainer, "_cggr_bridge")
        assert trainer._cggr_bridge is bridge
    
    def test_get_stats(self):
        """Test statistics tracking."""
        model = MockModel()
        bridge = CGGRUnslothBridge(model=model, warmup_steps=0)
        
        # Initially, stats should be zero
        stats = bridge.get_stats()
        assert stats["cggr/hard_ratio"] == 0.0
        assert stats["cggr/step"] == 0
        
        # After processing some tokens
        input_ids = torch.randint(0, 100, (1, 50))
        labels = torch.randint(0, 100, (1, 50))
        bridge.mask_easy_tokens(input_ids, labels)
        bridge.step()
        
        stats = bridge.get_stats()
        assert stats["cggr/step"] == 1
        assert stats["cggr/total_tokens"] > 0


class TestCGGRIgnoredTokens:
    """Tests for correct handling of already-ignored tokens."""
    
    def test_preserves_existing_ignored_tokens(self):
        """Test that tokens already set to -100 are preserved."""
        model = MockModel()
        bridge = CGGRUnslothBridge(
            model=model,
            min_tokens_ratio=0.5,
            warmup_steps=0,
            dynamic_threshold=False,
        )
        
        input_ids = torch.randint(0, 100, (1, 20))
        labels = torch.randint(0, 100, (1, 20))
        
        # Mark some tokens as already ignored
        original_ignored = [0, 5, 10, 15]
        for idx in original_ignored:
            labels[0, idx] = -100
        
        masked_labels = bridge.mask_easy_tokens(input_ids, labels.clone())
        
        # Original ignored tokens should still be ignored
        for idx in original_ignored:
            assert masked_labels[0, idx] == -100


class TestCGGRModuleImport:
    """Tests for module import behavior."""
    
    def test_cggr_available_flag(self):
        """Test CGGR_AVAILABLE flag exists."""
        from cggr import CGGR_AVAILABLE
        assert isinstance(CGGR_AVAILABLE, bool)
    
    def test_import_without_cggr_package(self):
        """Test graceful handling when cggr package is not installed."""
        # This test just verifies the module can be imported
        # without the cggr package being available
        import cggr as cggr_module
        assert hasattr(cggr_module, "CGGR_AVAILABLE")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
