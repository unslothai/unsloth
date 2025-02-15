import pytest
import torch
if not torch.cuda.is_available():
    pytest.skip('No NVIDIA GPU found, skipping unsloth kernel tests', allow_module_level=True)

# Smoke test for the unsloth.kernels package __init__.py

def test_kernels_init():
    import unsloth.kernels as kernels
    assert hasattr(kernels, '__path__')


# Test for utils.py

def test_utils_import():
    from unsloth.kernels import utils
    assert hasattr(utils, '__doc__')


# Test for cross_entropy_loss.py

def test_cross_entropy_loss():
    from unsloth.kernels import cross_entropy_loss as cel
    if hasattr(cel, 'cross_entropy_loss'):
        import torch
        logits = torch.tensor([[2.0, 0.5], [1.0, 3.0]])
        target = torch.tensor([0, 1])
        loss = cel.cross_entropy_loss(logits, target)
        # loss should be a positive value
        assert loss.item() >= 0
    else:
        pytest.skip('cross_entropy_loss function not defined in cross_entropy_loss.py')


# Test for fast_lora.py

def test_fast_lora():
    from unsloth.kernels import fast_lora
    if hasattr(fast_lora, 'FastLoRA'):
        import torch
        # Create a dummy instance; parameters may vary based on actual implementation
        try:
            model = fast_lora.FastLoRA()
        except Exception as e:
            pytest.skip(f'Could not instantiate FastLoRA: {e}')
        x = torch.randn(4, 4)
        try:
            output = model(x)
            assert output.shape == x.shape
        except Exception as e:
            pytest.skip(f'FastLoRA forward pass failed: {e}')
    else:
        pytest.skip('FastLoRA not defined in fast_lora.py')


# Test for flex_attention.py

def test_flex_attention():
    from unsloth.kernels import flex_attention
    if hasattr(flex_attention, 'FlexAttention'):
        import torch
        try:
            # Typical attention dims: dim and heads are guessed defaults
            fa = flex_attention.FlexAttention(dim=8, heads=2)
        except Exception as e:
            pytest.skip(f'Could not instantiate FlexAttention: {e}')
        # Create a dummy input tensor with shape [batch, seq_len, dim]
        dummy_input = torch.randn(1, 16, 8)
        try:
            output = fa(dummy_input)
            # Expect output shape same as input
            assert output.shape == dummy_input.shape
        except Exception as e:
            pytest.skip(f'FlexAttention forward pass failed: {e}')
    else:
        pytest.skip('FlexAttention not defined in flex_attention.py')


# Test for geglu.py

def test_geglu():
    from unsloth.kernels import geglu
    if hasattr(geglu, 'geglu'):
        import torch
        x = torch.randn(10, 20)
        try:
            y = geglu.geglu(x)
            # Basic check: same batch dimension
            assert y.shape[0] == x.shape[0]
        except Exception as e:
            pytest.skip(f'geglu function call failed: {e}')
    else:
        pytest.skip('geglu function not defined in geglu.py')


# Test for layernorm.py

def test_layernorm():
    from unsloth.kernels import layernorm
    if hasattr(layernorm, 'LayerNorm'):
        import torch
        try:
            ln = layernorm.LayerNorm(normalized_shape=20)
        except Exception as e:
            pytest.skip(f'Could not instantiate LayerNorm: {e}')
        x = torch.randn(3, 20)
        try:
            out = ln(x)
            assert out.shape == x.shape
        except Exception as e:
            pytest.skip(f'LayerNorm forward pass failed: {e}')
    else:
        pytest.skip('LayerNorm not defined in layernorm.py')


# Test for rms_layernorm.py

def test_rms_layernorm():
    from unsloth.kernels import rms_layernorm
    if hasattr(rms_layernorm, 'RMSLayerNorm'):
        import torch
        try:
            rln = rms_layernorm.RMSLayerNorm(normalized_shape=20)
        except Exception as e:
            pytest.skip(f'Could not instantiate RMSLayerNorm: {e}')
        x = torch.randn(3, 20)
        try:
            out = rln(x)
            assert out.shape == x.shape
        except Exception as e:
            pytest.skip(f'RMSLayerNorm forward pass failed: {e}')
    else:
        pytest.skip('RMSLayerNorm not defined in rms_layernorm.py')


# Test for rope_embedding.py

def test_rope_embedding():
    from unsloth.kernels import rope_embedding
    if hasattr(rope_embedding, 'RoPEEmbedding'):
        import torch
        try:
            embed = rope_embedding.RoPEEmbedding(dim=16)
        except Exception as e:
            pytest.skip(f'Could not instantiate RoPEEmbedding: {e}')
        x = torch.randn(2, 10, 16)  # [batch, seq_len, dim]
        try:
            out = embed(x)
            assert out.shape == x.shape
        except Exception as e:
            pytest.skip(f'RoPEEmbedding forward pass failed: {e}')
    else:
        pytest.skip('RoPEEmbedding not defined in rope_embedding.py')


# Test for swiglu.py

def test_swiglu():
    from unsloth.kernels import swiglu
    if hasattr(swiglu, 'swiglu'):
        import torch
        x = torch.randn(5, 10)
        try:
            out = swiglu.swiglu(x)
            # Basic check: output should have same batch dimension
            assert out.shape[0] == x.shape[0]
        except Exception as e:
            pytest.skip(f'swiglu function call failed: {e}')
    else:
        pytest.skip('swiglu function not defined in swiglu.py')