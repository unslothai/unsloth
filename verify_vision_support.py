import torch
import os
import sys
from unittest.mock import MagicMock, patch

# Mock MLX and Metal for verification if not available
# (In a real scenario, these would be present on the Mac)
try:
    import mlx.core as mx
except ImportError:
    print("Mocking MLX for verification...")
    from unittest.mock import MagicMock
    sys.modules["mlx"] = MagicMock()
    sys.modules["mlx.core"] = MagicMock()
    sys.modules["mlx.nn"] = MagicMock()

# Mock unsloth.kernels.mlx
try:
    import unsloth.kernels.mlx.utils
except ImportError:
    print("Mocking unsloth.kernels.mlx.utils for verification...")
    from unittest.mock import MagicMock
    mock_mlx_utils = MagicMock()
    sys.modules["unsloth.kernels.mlx.utils"] = mock_mlx_utils

from unsloth import FastVisionModel
import unsloth.device_type as device_type

def test_vision_load_4bit():
    print(f"Testing Vision model load in 4-bit on {device_type.DEVICE_TYPE}...")
    
    # We use a very small model for testing if possible, or just mock the loader
    # For verification of the LOGIC, we can check if it attempts to call fast_quantize
    
    model_name = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"
    
    # We'll mock FastBaseModel.from_pretrained to avoid actually downloading 11B params
    from unsloth.models.vision import FastBaseModel
    from unittest.mock import patch
    
    mock_model = torch.nn.Module()
    mock_tokenizer = MagicMock()
    
    with patch("unsloth.models.vision.FastBaseModel.from_pretrained", return_value=(mock_model, mock_tokenizer)):
        with patch("unsloth.kernels.mlx.utils.fast_quantize") as mock_quantize:
            model, tokenizer = FastVisionModel.from_pretrained(
                model_name=model_name,
                load_in_4bit=True,
            )
            
            if device_type.DEVICE_TYPE == "mps":
                print("Checking if fast_quantize was called...")
                mock_quantize.assert_called_once()
                print("fast_quantize was called correctly.")
            else:
                print(f"Not on MPS ({device_type.DEVICE_TYPE}), skip quantization check.")

def test_vision_fast_inference_gate():
    print(f"Testing fast_inference gate for Vision on {device_type.DEVICE_TYPE}...")
    if device_type.DEVICE_TYPE != "mps":
        print("Skip test (not on MPS).")
        return

    # Mock HF loader
    from unsloth.models.vision import FastBaseModel
    from unittest.mock import patch
    
    mock_model = torch.nn.Module()
    mock_tokenizer = MagicMock()
    
    with patch("unsloth.models.vision.FastBaseModel.from_pretrained", side_effect=RuntimeError("Gate should trigger before this")):
        try:
            FastVisionModel.from_pretrained(
                model_name="some-vision-model",
                fast_inference=True,
            )
        except RuntimeError as e:
            if "fast_inference (vLLM) is not yet supported" in str(e):
                print("Gate successfully blocked fast_inference on MPS.")
            else:
                print(f"Unexpected RuntimeError: {e}")
                raise

if __name__ == "__main__":
    # Force MPS for testing logic if we want to see it work
    original_device = device_type.DEVICE_TYPE
    device_type.DEVICE_TYPE = "mps"
    
    try:
        test_vision_load_4bit()
        test_vision_fast_inference_gate()
        print("\nAll Vision HARDENING tests passed!")
    finally:
        device_type.DEVICE_TYPE = original_device
