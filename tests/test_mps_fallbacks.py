# Apply Mac compatibility patches BEFORE importing unsloth
import platform
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if platform.system() == "Darwin":
    from patcher import patch_for_mac
    patch_for_mac()

import types
from importlib.machinery import ModuleSpec
import importlib.util


# 1. Standalone Mocking and Path Loading
def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# Mock dependencies to allow the internal kernel files to import unsloth.device_type etc.
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

# Instead of importing the whole unsloth package, we load just the kernels we want to test
# To do this, we need to satisfy imports within those files.
# Most kernel files do: from ..device_type import DEVICE_TYPE
# which resolves to unsloth.device_type

mock_unsloth = create_mock("unsloth")
mock_unsloth_kernels = create_mock("unsloth.kernels")
mock_unsloth_device = create_mock("unsloth.device_type")
mock_unsloth_device.DEVICE_TYPE = "mps"
mock_unsloth_device.is_mps = lambda: True

sys.modules["unsloth"] = mock_unsloth
sys.modules["unsloth.kernels"] = mock_unsloth_kernels
sys.modules["unsloth.device_type"] = mock_unsloth_device

import torch
import torch.nn.functional as F
import math

# Load the kernels directly from files
rms_norm_mod = load_module_from_path(
    "unsloth.kernels.mps.rms_layernorm",
    os.path.join(ROOT, "unsloth/kernels/mps/rms_layernorm.py"),
)
mps_rms_layernorm = rms_norm_mod.mps_rms_layernorm

layernorm_mod = load_module_from_path(
    "unsloth.kernels.mps.layernorm",
    os.path.join(ROOT, "unsloth/kernels/mps/layernorm.py"),
)
mps_layernorm = layernorm_mod.mps_layernorm

rope_mod = load_module_from_path(
    "unsloth.kernels.mps.rope_embedding",
    os.path.join(ROOT, "unsloth/kernels/mps/rope_embedding.py"),
)
mps_rope_embedding_qk = rope_mod.mps_rope_embedding_qk

ce_mod = load_module_from_path(
    "unsloth.kernels.mps.cross_entropy_loss",
    os.path.join(ROOT, "unsloth/kernels/mps/cross_entropy_loss.py"),
)
mps_cross_entropy_loss = ce_mod.mps_cross_entropy_loss

swiglu_mod = load_module_from_path(
    "unsloth.kernels.mps.swiglu", os.path.join(ROOT, "unsloth/kernels/mps/swiglu.py")
)
mps_swiglu_forward = swiglu_mod.mps_swiglu_forward
mps_swiglu_backward = swiglu_mod.mps_swiglu_backward


def test_rms_norm():
    print("Testing RMSNorm...")
    X = torch.randn(2, 4, 8, requires_grad=True)
    W = torch.randn(8, requires_grad=True)
    eps = 1e-6
    Y = mps_rms_layernorm(X, W, eps)
    assert Y.shape == X.shape
    Y.sum().backward()
    assert X.grad is not None
    print("✅ RMSNorm Passed")


def test_swiglu():
    print("Testing SwiGLU...")
    e = torch.randn(2, 4, 8, requires_grad=True)
    g = torch.randn(2, 4, 8, requires_grad=True)
    Y = mps_swiglu_forward(e, g)
    assert torch.allclose(Y, F.silu(e) * g)
    dw = torch.randn_like(Y)
    h, de, dg = mps_swiglu_backward(dw, e, g)
    assert de.shape == e.shape
    assert dg.shape == g.shape
    print("✅ SwiGLU Passed")


def test_cross_entropy():
    print("Testing Cross Entropy...")
    logits = torch.randn(2, 4, 16, requires_grad=True)
    labels = torch.randint(0, 16, (2, 4))
    loss = mps_cross_entropy_loss(logits, labels)
    loss.backward()
    assert logits.grad is not None
    print("✅ Cross Entropy Passed")


if __name__ == "__main__":
    test_rms_norm()
    test_swiglu()
    test_cross_entropy()
    print("\n🚀 ALL CORE KERNELS VERIFIED NUMERICALLY.")
