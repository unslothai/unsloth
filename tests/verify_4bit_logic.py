import sys
import torch
import torch.nn as nn
from unittest.mock import MagicMock


# 1. Create robust Mock
class MockMLX:
    def __init__(self):
        self.core = MagicMock()
        self.core.__spec__ = MagicMock()
        self.core.fast = MagicMock()


mock_mlx = MockMLX()
sys.modules["mlx"] = mock_mlx
sys.modules["mlx.core"] = mock_mlx.core
sys.modules["mlx.core.fast"] = mock_mlx.core.fast
from unsloth.kernels.mlx.bridge import torch_to_mlx
from unsloth.kernels.mlx.quantization import MLXQuantizedWeight
from unsloth.kernels.mlx.fast_lora import fast_dequantize, quantized_matmul


def test_bridge_cache():
    print("Testing bridge cache logic...")
    W = torch.randn(10, 10)
    mock_quant = MLXQuantizedWeight(MagicMock(), MagicMock(), MagicMock(), 64)
    W._mlx_cache = mock_quant

    # torch_to_mlx should return the cache if present
    res = torch_to_mlx(W)
    assert res == mock_quant
    print("✅ Bridge cache check passed.")


def test_fast_dequantize_logic():
    print("Testing fast_dequantize logic...")
    mock_quant = MLXQuantizedWeight(MagicMock(), MagicMock(), MagicMock(), 64)
    # If W is the quantized object, it should return it
    res = fast_dequantize(mock_quant, None)
    assert res == mock_quant
    print("✅ fast_dequantize logic passed.")


def test_quantized_matmul_dispatch():
    print("Testing quantized_matmul dispatch...")
    X = MagicMock()
    mock_quant = MLXQuantizedWeight(MagicMock(), MagicMock(), MagicMock(), 64)

    # Should call mx.fast.quantized_matmul
    import mlx.core.fast as mxf

    quantized_matmul(X, mock_quant)
    mxf.quantized_matmul.assert_called_once()
    print("✅ quantized_matmul dispatch passed.")


if __name__ == "__main__":
    try:
        test_bridge_cache()
        test_fast_dequantize_logic()
        test_quantized_matmul_dispatch()
        print("\n🎉 All 4-bit logic verification checks passed!")
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        sys.exit(1)
