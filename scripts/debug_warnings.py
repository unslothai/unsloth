"""Debug script to see what warnings are issued."""
# Apply Mac compatibility patches BEFORE importing unsloth
import platform
if platform.system() == "Darwin":
    from patcher import patch_for_mac
    patch_for_mac()

import warnings
from unittest.mock import MagicMock, patch
import torch
import sys
import os

# Force MPS mode
sys.path.insert(0, '/Users/ec2-user/unsloth-work')
os.chdir('/Users/ec2-user/unsloth-work')

# Mock MLX
try:
    import mlx.core as mx
except ImportError:
    from unittest.mock import MagicMock
    sys.modules["mlx"] = MagicMock()
    sys.modules["mlx.core"] = MagicMock()
    sys.modules["mlx.nn"] = MagicMock()

try:
    import unsloth.kernels.mlx.utils
except ImportError:
    mock_mlx_utils = MagicMock()
    sys.modules["unsloth.kernels.mlx.utils"] = mock_mlx_utils

from unsloth import FastVisionModel
import unsloth.device_type as device_type

# Force MPS
device_type.DEVICE_TYPE = "mps"

print(f"Device type: {device_type.DEVICE_TYPE}")

# Mock the model loading
mock_model = MagicMock()
mock_model.config = MagicMock()
mock_model.config.model_type = "mllama"
mock_model.config.torch_dtype = torch.float16
mock_tokenizer = MagicMock()

with patch("unsloth.models.vision.FastBaseModel.from_pretrained", return_value=(mock_model, mock_tokenizer)) as mock_from_pretrained:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        FastVisionModel.from_pretrained(
            model_name="unsloth/Llama-3.2-11B-Vision-Instruct",
            fast_inference=True,
        )
        
        print(f"\nNumber of warnings: {len(w)}")
        for i, warning in enumerate(w):
            print(f"\nWarning {i+1}:")
            print(f"  Category: {warning.category.__name__}")
            print(f"  Message: {warning.message}")
            print(f"  File: {warning.filename}")
