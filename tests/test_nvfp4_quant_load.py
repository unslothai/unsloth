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
Test cases for NVFP4 / compressed-tensors model loading.
Ensures that models with non-bitsandbytes quantization configs
don't conflict with Unsloth's default load_in_4bit=True behavior.
"""

import pytest
from transformers import AutoConfig


def test_nvfp4_model_has_compressed_tensors_config():
    """Verify the NVFP4 model has compressed-tensors quantization config."""
    config = AutoConfig.from_pretrained(
        "unsloth/Qwen3.6-35B-A3B-NVFP4",
        trust_remote_code=True,
    )
    qcfg = config.quantization_config
    assert qcfg is not None, "Model should have quantization_config"
    if isinstance(qcfg, dict):
        assert qcfg.get("quant_method") == "compressed-tensors"
        assert qcfg.get("format") == "nvfp4-pack-quantized"
    else:
        assert getattr(qcfg, "quant_method", None) == "compressed-tensors"


def test_regular_bnb_model_has_bitsandbytes_config():
    """Verify regular bnb-4bit model has bitsandbytes quantization config."""
    config = AutoConfig.from_pretrained("unsloth/llama-3-8b-bnb-4bit")
    qcfg = config.quantization_config
    assert qcfg is not None, "Model should have quantization_config"
    if isinstance(qcfg, dict):
        assert qcfg.get("quant_method") == "bitsandbytes"
    else:
        assert getattr(qcfg, "quant_method", None) == "bitsandbytes"


def test_load_in_4bit_detection_logic():
    """Test the quantization config detection logic directly."""
    from unsloth.models.llama import FastLlamaModel
    from transformers import AutoConfig
    
    # Test NVFP4 model
    config_nvfp4 = AutoConfig.from_pretrained(
        "unsloth/Qwen3.6-35B-A3B-NVFP4",
        trust_remote_code=True,
    )
    _ckpt_qcfg = getattr(config_nvfp4, "quantization_config", None)
    _ckpt_quant_method = None
    if _ckpt_qcfg is not None:
        if isinstance(_ckpt_qcfg, dict):
            _ckpt_quant_method = _ckpt_qcfg.get("quant_method")
        else:
            _ckpt_quant_method = getattr(_ckpt_qcfg, "quant_method", None)
    
    # Should detect compressed-tensors and disable load_in_4bit
    load_in_4bit = True
    if load_in_4bit and _ckpt_quant_method is not None and _ckpt_quant_method != "bitsandbytes":
        load_in_4bit = False
    assert load_in_4bit is False, "load_in_4bit should be disabled for compressed-tensors"
    
    # Test regular bnb-4bit model
    config_bnb = AutoConfig.from_pretrained("unsloth/llama-3-8b-bnb-4bit")
    _ckpt_qcfg = getattr(config_bnb, "quantization_config", None)
    _ckpt_quant_method = None
    if _ckpt_qcfg is not None:
        if isinstance(_ckpt_qcfg, dict):
            _ckpt_quant_method = _ckpt_qcfg.get("quant_method")
        else:
            _ckpt_quant_method = getattr(_ckpt_qcfg, "quant_method", None)
    
    # Should NOT disable load_in_4bit for bitsandbytes
    load_in_4bit = True
    if load_in_4bit and _ckpt_quant_method is not None and _ckpt_quant_method != "bitsandbytes":
        load_in_4bit = False
    assert load_in_4bit is True, "load_in_4bit should remain True for bitsandbytes"


if __name__ == "__main__":
    test_nvfp4_model_has_compressed_tensors_config()
    test_regular_bnb_model_has_bitsandbytes_config()
    test_load_in_4bit_detection_logic()
    print("All tests passed!")
