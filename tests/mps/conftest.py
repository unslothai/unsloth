import sys
import types
from unittest.mock import MagicMock
import torch


# Create a more realistic mock for modules that transformers/unsloth expect
def create_mock_module(name):
    mock = MagicMock(spec=types.ModuleType)
    mock.__name__ = name
    mock.__file__ = f"{name}.py"
    mock.__path__ = []
    # Set a dummy spec
    from importlib.machinery import ModuleSpec

    mock.__spec__ = ModuleSpec(name, None)
    return mock


# Mocking triton
triton = create_mock_module("triton")
sys.modules["triton"] = triton
sys.modules["triton.language"] = create_mock_module("triton.language")
sys.modules["triton.jit"] = create_mock_module("triton.jit")
sys.modules["triton.runtime"] = create_mock_module("triton.runtime")
sys.modules["triton.runtime.jit"] = create_mock_module("triton.runtime.jit")

# Mocking bitsandbytes
bnb = create_mock_module("bitsandbytes")
bnb.__version__ = "0.42.0"
sys.modules["bitsandbytes"] = bnb
sys.modules["bitsandbytes.functional"] = create_mock_module("bitsandbytes.functional")

# Mocking unsloth_zoo
zoo = create_mock_module("unsloth_zoo")
zoo_dt = create_mock_module("unsloth_zoo.device_type")
zoo_dt.DEVICE_TYPE = "mps"
zoo_dt.DEVICE_TYPE_TORCH = torch.device("cpu")
zoo_dt.DEVICE_COUNT = 1
zoo_dt.is_hip = lambda: False
zoo_dt.is_mps = lambda: True
zoo_dt.get_device_type = lambda: "mps"
zoo_dt.ALLOW_PREQUANTIZED_MODELS = False

sys.modules["unsloth_zoo"] = zoo
sys.modules["unsloth_zoo.device_type"] = zoo_dt

# Mocking other unsloth requirements
sys.modules["datasets"] = create_mock_module("datasets")
sys.modules["datasets"].__version__ = "2.14.0"
sys.modules["trl"] = create_mock_module("trl")
sys.modules["peft"] = create_mock_module("peft")
sys.modules["xformers"] = create_mock_module("xformers")
sys.modules["xformers.ops"] = create_mock_module("xformers.ops")
