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

"""Pytest configuration for MLX tests."""

import sys
import types
import logging
from importlib.machinery import ModuleSpec

import torch


def create_mock_module(name):
    """Create a mock module that properly supports 'from X import Y' syntax."""
    module = types.ModuleType(name)
    module.__file__ = f"{name}.py"
    module.__path__ = []
    module.__spec__ = ModuleSpec(name, None)
    module.__package__ = name.rsplit(".", 1)[0] if "." in name else name
    return module


# Apply Mac patches BEFORE importing pytest to avoid import errors
if sys.platform == "darwin":
    try:
        from patcher import patch_for_mac
        patch_for_mac(verbose=False)
    except Exception:
        pass

# Mock triton (not available on macOS)
triton = create_mock_module("triton")
sys.modules["triton"] = triton
sys.modules["triton.language"] = create_mock_module("triton.language")
sys.modules["triton.jit"] = create_mock_module("triton.jit")
sys.modules["triton.runtime"] = create_mock_module("triton.runtime")
sys.modules["triton.runtime.jit"] = create_mock_module("triton.runtime.jit")

# Mock bitsandbytes (not available on macOS)
bnb = create_mock_module("bitsandbytes")
bnb.__version__ = "0.42.0"
sys.modules["bitsandbytes"] = bnb
sys.modules["bitsandbytes.functional"] = create_mock_module("bitsandbytes.functional")

# Mock unsloth_zoo
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

# Mock unsloth_zoo.utils
zoo_utils = create_mock_module("unsloth_zoo.utils")


class Version:
    def __init__(self, v):
        self.v = v

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return True

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return True


def _get_dtype(dtype_str):
    return getattr(torch, dtype_str, torch.float32)


def get_quant_type(module):
    return "unknown"


zoo_utils.Version = Version
zoo_utils._get_dtype = _get_dtype
zoo_utils.get_quant_type = get_quant_type
sys.modules["unsloth_zoo.utils"] = zoo_utils

# Mock unsloth_zoo.log
zoo_log = create_mock_module("unsloth_zoo.log")
zoo_log.logger = logging.getLogger("unsloth_zoo")
sys.modules["unsloth_zoo.log"] = zoo_log

# Mock unsloth_zoo.tokenizer_utils
zoo_tokenizer = create_mock_module("unsloth_zoo.tokenizer_utils")
zoo_tokenizer.patch_tokenizer = lambda x: x
sys.modules["unsloth_zoo.tokenizer_utils"] = zoo_tokenizer

# Mock unsloth_zoo.rl_environments
zoo_rl = create_mock_module("unsloth_zoo.rl_environments")
zoo_rl.check_python_modules = lambda: True
zoo_rl.create_locked_down_function = lambda fn: fn
zoo_rl.execute_with_time_limit = lambda timeout, fn, *args, **kwargs: fn(*args, **kwargs)


class MockBenchmarker:
    pass


zoo_rl.Benchmarker = MockBenchmarker
sys.modules["unsloth_zoo.rl_environments"] = zoo_rl

# Mock other dependencies
sys.modules["datasets"] = create_mock_module("datasets")
sys.modules["datasets"].__version__ = "2.14.0"
sys.modules["trl"] = create_mock_module("trl")
sys.modules["peft"] = create_mock_module("peft")
sys.modules["xformers"] = create_mock_module("xformers")
sys.modules["xformers.ops"] = create_mock_module("xformers.ops")

# Mock unsloth_zoo.patching_utils
zoo_patching = create_mock_module("unsloth_zoo.patching_utils")
zoo_patching.patch_compiling_bitsandbytes = lambda: None
zoo_patching.patch_layernorm = lambda: None
zoo_patching.patch_torch_compile = lambda: None
zoo_patching.patch_model_and_tokenizer = lambda model, tokenizer: (model, tokenizer)
zoo_patching.patch_compiled_autograd = lambda: None
sys.modules["unsloth_zoo.patching_utils"] = zoo_patching

# Mock unsloth_zoo.gradient_checkpointing
zoo_gc = create_mock_module("unsloth_zoo.gradient_checkpointing")


class MockOffloadedGC:
    pass


class MockGC:
    pass


zoo_gc.Unsloth_Offloaded_Gradient_Checkpointer = MockOffloadedGC
zoo_gc.unsloth_offloaded_gradient_checkpoint = lambda module, *args, **kwargs: module.forward
zoo_gc.patch_unsloth_gradient_checkpointing = lambda: None
zoo_gc.unpatch_unsloth_gradient_checkpointing = lambda: None
zoo_gc.Unsloth_Gradient_Checkpointer = MockGC
zoo_gc.unsloth_gradient_checkpoint = lambda module, *args, **kwargs: module.forward
zoo_gc.patch_gradient_checkpointing = lambda: None
zoo_gc.unpatch_gradient_checkpointing = lambda: None
zoo_gc.patch_unsloth_smart_gradient_checkpointing = lambda: None
zoo_gc.unpatch_unsloth_smart_gradient_checkpointing = lambda: None
sys.modules["unsloth_zoo.gradient_checkpointing"] = zoo_gc

# Mock unsloth_zoo.loss_utils
zoo_loss = create_mock_module("unsloth_zoo.loss_utils")
zoo_loss.HAS_CUT_CROSS_ENTROPY = False
zoo_loss.fused_linear_cross_entropy = lambda *args, **kwargs: 0.0
zoo_loss._unsloth_get_batch_samples = lambda *args, **kwargs: None
zoo_loss.unsloth_fused_ce_loss = lambda *args, **kwargs: 0.0
sys.modules["unsloth_zoo.loss_utils"] = zoo_loss

# Mock unsloth_zoo.vision_utils
zoo_vision = create_mock_module("unsloth_zoo.vision_utils")
zoo_vision.HAS_VISION = False
zoo_vision.process_vision_info = lambda *args, **kwargs: None


class MockVisionDataCollator:
    def __init__(self, *args, **kwargs):
        pass


zoo_vision.UnslothVisionDataCollator = MockVisionDataCollator
sys.modules["unsloth_zoo.vision_utils"] = zoo_vision

# Mock unsloth_zoo.compiler
zoo_compiler = create_mock_module("unsloth_zoo.compiler")
zoo_compiler.get_transformers_model_type = lambda *args, **kwargs: "llama"
zoo_compiler.unsloth_compile_transformers = lambda *args, **kwargs: None
sys.modules["unsloth_zoo.compiler"] = zoo_compiler

# Mock unsloth_zoo.training_utils
zoo_training = create_mock_module("unsloth_zoo.training_utils")
zoo_training.prepare_model_for_training = lambda model, *args, **kwargs: model
zoo_training.unsloth_train = lambda *args, **kwargs: None
sys.modules["unsloth_zoo.training_utils"] = zoo_training

# Mock unsloth_zoo.temporary_patches
zoo_temp_patches = create_mock_module("unsloth_zoo.temporary_patches")
zoo_temp_patches.TEMPORARY_PATCHES = {}
sys.modules["unsloth_zoo.temporary_patches"] = zoo_temp_patches

# Mock unsloth_zoo.hf_utils
zoo_hf = create_mock_module("unsloth_zoo.hf_utils")
zoo_hf.dtype_from_config = lambda *args, **kwargs: None
zoo_hf.HAS_TORCH_DTYPE = False
sys.modules["unsloth_zoo.hf_utils"] = zoo_hf

# Mock unsloth_zoo.peft_utils
zoo_peft = create_mock_module("unsloth_zoo.peft_utils")
zoo_peft.SKIP_QUANTIZATION_MODULES = set()
sys.modules["unsloth_zoo.peft_utils"] = zoo_peft

# Mock unsloth_zoo.vllm_utils
zoo_vllm = create_mock_module("unsloth_zoo.vllm_utils")
sys.modules["unsloth_zoo.vllm_utils"] = zoo_vllm

# Mock unsloth_zoo.saving_utils
zoo_saving = create_mock_module("unsloth_zoo.saving_utils")
sys.modules["unsloth_zoo.saving_utils"] = zoo_saving

# Mock unsloth_zoo.llama_cpp
zoo_llama_cpp = create_mock_module("unsloth_zoo.llama_cpp")
sys.modules["unsloth_zoo.llama_cpp"] = zoo_llama_cpp

# Mock unsloth_zoo.dataset_utils
zoo_dataset = create_mock_module("unsloth_zoo.dataset_utils")
sys.modules["unsloth_zoo.dataset_utils"] = zoo_dataset

# Mock unsloth_zoo.rl_replacements
zoo_rl_replace = create_mock_module("unsloth_zoo.rl_replacements")
zoo_rl_replace.RL_REPLACEMENTS = {}
sys.modules["unsloth_zoo.rl_replacements"] = zoo_rl_replace

# Mock unsloth_zoo.logging_utils
zoo_logging = create_mock_module("unsloth_zoo.logging_utils")
sys.modules["unsloth_zoo.logging_utils"] = zoo_logging

# Mock unsloth_zoo.flex_attention
zoo_flex = create_mock_module("unsloth_zoo.flex_attention")
zoo_flex.HAS_FLEX_ATTENTION = False
sys.modules["unsloth_zoo.flex_attention"] = zoo_flex

# Mock unsloth_zoo.tiled_mlp
zoo_tiled_mlp = create_mock_module("unsloth_zoo.tiled_mlp")
zoo_tiled_mlp.patch_tiled_mlp = lambda: None
sys.modules["unsloth_zoo.tiled_mlp"] = zoo_tiled_mlp

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "mlx_only: mark test to run only when MLX is available"
    )


def pytest_collection_modifyitems(config, items):
    if sys.platform != "darwin":
        skip_mlx = pytest.mark.skip(reason="MLX only available on macOS")
        for item in items:
            if "mlx_only" in item.keywords:
                item.add_marker(skip_mlx)
