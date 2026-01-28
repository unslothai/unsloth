import torch
import unsloth
from unsloth.device_type import DEVICE_TYPE

def test_device_type():
    assert DEVICE_TYPE == "mps"

def test_imports():
    from unsloth.kernels.mps.rms_layernorm import mps_rms_layernorm
    assert mps_rms_layernorm is not None
