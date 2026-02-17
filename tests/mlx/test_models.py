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

"""Tests for pure MLX models using pytest."""

import pytest


def is_mlx_available():
    """Check if MLX is available."""
    try:
        import mlx.core as mx
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not is_mlx_available(), reason="MLX is not available")


class TestMLXBaseLayers:
    """Test MLX base layers."""

    def test_mlx_linear(self):
        """Test MLXLinear layer."""
        import mlx.core as mx
        from unsloth.kernels.mlx.models.base import MLXLinear

        batch_size = 2
        seq_len = 8
        in_features = 32
        out_features = 64

        linear = MLXLinear(in_features, out_features, bias=True)

        x = mx.random.uniform(shape=(batch_size, seq_len, in_features))
        y = linear(x)

        assert y.shape == (batch_size, seq_len, out_features)

    def test_mlx_linear_no_bias(self):
        """Test MLXLinear without bias."""
        import mlx.core as mx
        from unsloth.kernels.mlx.models.base import MLXLinear

        linear = MLXLinear(32, 64, bias=False)
        x = mx.random.uniform(shape=(2, 8, 32))
        y = linear(x)

        assert y.shape == (2, 8, 64)
        assert linear.bias is None

    def test_mlx_embedding(self):
        """Test MLXEmbedding layer."""
        import mlx.core as mx
        from unsloth.kernels.mlx.models.base import MLXEmbedding

        batch_size = 2
        seq_len = 8
        vocab_size = 1000
        embedding_dim = 64

        embedding = MLXEmbedding(vocab_size, embedding_dim)

        x = mx.random.randint(0, vocab_size, shape=(batch_size, seq_len))
        y = embedding(x)

        assert y.shape == (batch_size, seq_len, embedding_dim)

    def test_mlx_rms_norm(self):
        """Test MLXRMSNorm layer."""
        import mlx.core as mx
        from unsloth.kernels.mlx.models.base import MLXRMSNorm

        batch_size = 2
        seq_len = 8
        hidden_size = 64

        norm = MLXRMSNorm(hidden_size, eps=1e-6)
        x = mx.random.uniform(shape=(batch_size, seq_len, hidden_size))
        y = norm(x)

        assert y.shape == (batch_size, seq_len, hidden_size)

    def test_mlx_layer_norm(self):
        """Test MLXLayerNorm layer."""
        import mlx.core as mx
        from unsloth.kernels.mlx.models.base import MLXLayerNorm

        batch_size = 2
        seq_len = 8
        hidden_size = 64

        norm = MLXLayerNorm(hidden_size, eps=1e-6)
        x = mx.random.uniform(shape=(batch_size, seq_len, hidden_size))
        y = norm(x)

        assert y.shape == (batch_size, seq_len, hidden_size)


class TestMLXLlamaModel:
    """Test MLX Llama model."""

    def test_create_llama_model(self):
        """Test creating a Llama model."""
        from unsloth.kernels.mlx.models.llama import create_llama_model
        from unsloth.kernels.mlx.models.base import LoRAConfig

        model = create_llama_model(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )

        assert model is not None
        assert model.vocab_size == 1000

    def test_llama_forward(self):
        """Test Llama forward pass."""
        import mlx.core as mx
        from unsloth.kernels.mlx.models.llama import create_llama_model

        model = create_llama_model(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )

        batch_size = 2
        seq_len = 16
        input_ids = mx.random.randint(0, 1000, shape=(batch_size, seq_len))

        logits, loss = model(input_ids=input_ids, labels=input_ids)

        assert logits.shape == (batch_size, seq_len, 1000)
        assert loss is not None

    def test_llama_forward_no_labels(self):
        """Test Llama forward pass without labels."""
        import mlx.core as mx
        from unsloth.kernels.mlx.models.llama import create_llama_model

        model = create_llama_model(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=128,
        )

        input_ids = mx.random.randint(0, 1000, shape=(2, 16))
        logits, loss = model(input_ids=input_ids)

        assert logits.shape == (2, 16, 1000)
        assert loss is None

    def test_llama_with_lora(self):
        """Test Llama with LoRA configuration."""
        import mlx.core as mx
        from unsloth.kernels.mlx.models.llama import create_llama_model
        from unsloth.kernels.mlx.models.base import LoRAConfig

        lora_config = LoRAConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])

        model = create_llama_model(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=128,
            lora_config=lora_config,
        )

        input_ids = mx.random.randint(0, 1000, shape=(2, 16))
        logits, loss = model(input_ids=input_ids, labels=input_ids)

        assert logits.shape == (2, 16, 1000)
        assert loss is not None

    def test_llama_with_attention_mask(self):
        """Test Llama with attention mask."""
        import mlx.core as mx
        from unsloth.kernels.mlx.models.llama import create_llama_model

        model = create_llama_model(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=128,
        )

        input_ids = mx.random.randint(0, 1000, shape=(2, 16))
        attention_mask = mx.ones((2, 16))
        attention_mask[:, 8:] = 0

        logits, loss = model(input_ids=input_ids, attention_mask=attention_mask)

        assert logits.shape == (2, 16, 1000)


class TestMLXAutograd:
    """Test MLX autograd (backward pass)."""

    def test_llama_training_step(self):
        """Test a full training step with gradients."""
        import mlx.core as mx
        from unsloth.kernels.mlx.models.llama import create_llama_model

        model = create_llama_model(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=128,
        )

        optimizer = mx.optimizers.Adam(learning_rate=1e-4)

        input_ids = mx.random.randint(0, 1000, shape=(2, 16))
        labels = mx.random.randint(0, 1000, shape=(2, 16))

        def loss_fn():
            logits, loss = model(input_ids=input_ids, labels=labels)
            return loss

        grad_fn = mx.grad(loss_fn)
        
        params = {
            "embed_tokens": model.model.embed_tokens.weight,
            "lm_head": model.lm_head.weight,
            "norm": model.model.norm.weight,
        }
        for i, layer in enumerate(model.model.layers):
            params[f"layer_{i}_input_layernorm"] = layer.input_layernorm.weight
            params[f"layer_{i}_post_attention_layernorm"] = layer.post_attention_layernorm.weight
            params[f"layer_{i}_q_proj"] = layer.self_attn.q_proj.weight
            params[f"layer_{i}_k_proj"] = layer.self_attn.k_proj.weight
            params[f"layer_{i}_v_proj"] = layer.self_attn.v_proj.weight
            params[f"layer_{i}_o_proj"] = layer.self_attn.o_proj.weight
            params[f"layer_{i}_gate_proj"] = layer.mlp.gate_proj.weight
            params[f"layer_{i}_up_proj"] = layer.mlp.up_proj.weight
            params[f"layer_{i}_down_proj"] = layer.mlp.down_proj.weight

        grads = grad_fn(params)

        assert grads is not None
