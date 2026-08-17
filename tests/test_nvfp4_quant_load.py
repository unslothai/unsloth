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

"""NVFP4 / compressed-tensors loading: non-bitsandbytes quant configs must not conflict with
load_in_4bit=True. Uses synthetic configs (no network) so it runs offline in CI.
"""

from types import SimpleNamespace

# Import unsloth first to set UNSLOTH_IS_PRESENT env var.
import unsloth
from unsloth_zoo.utils import get_quant_type
from unsloth.models.loader_utils import check_and_disable_bitsandbytes_loading


def _make_config(quantization_config = None, model_type = "llama"):
    return SimpleNamespace(
        quantization_config = quantization_config,
        model_type = model_type,
    )


_NVFP4_QCFG_DICT = {
    "quant_method": "compressed-tensors",
    "format": "nvfp4-pack-quantized",
    "quantization_config": {"num_bits": 4},
}

_BNB_QCFG_DICT = {
    "quant_method": "bitsandbytes",
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "llm_int8_skip_modules": [],
}


def test_nvfp4_config_has_compressed_tensors():
    config = _make_config(quantization_config = _NVFP4_QCFG_DICT)
    qcfg = config.quantization_config
    assert qcfg is not None
    assert qcfg.get("quant_method") == "compressed-tensors"
    assert qcfg.get("format") == "nvfp4-pack-quantized"


def test_regular_bnb_config_has_bitsandbytes():
    config = _make_config(quantization_config = _BNB_QCFG_DICT)
    qcfg = config.quantization_config
    assert qcfg is not None
    assert qcfg.get("quant_method") == "bitsandbytes"


def test_nvfp4_disables_load_in_4bit():
    config = _make_config(quantization_config = _NVFP4_QCFG_DICT)
    quant_method = get_quant_type(config)
    assert quant_method == "compressed-tensors"

    load_in_4bit, load_in_8bit, _ = check_and_disable_bitsandbytes_loading(
        config, load_in_4bit = True, load_in_8bit = False, verbose = False
    )
    assert load_in_4bit is False
    assert load_in_8bit is False


def test_bnb_does_not_disable_load_in_4bit():
    config = _make_config(quantization_config = _BNB_QCFG_DICT)
    quant_method = get_quant_type(config)
    assert quant_method == "bitsandbytes"

    load_in_4bit, load_in_8bit, _ = check_and_disable_bitsandbytes_loading(
        config, load_in_4bit = True, load_in_8bit = False, verbose = False
    )
    assert load_in_4bit is True
    assert load_in_8bit is False


def test_no_quantization_config_leaves_settings_unchanged():
    config = _make_config(quantization_config = None)
    quant_method = get_quant_type(config)
    assert quant_method is None

    load_in_4bit, load_in_8bit, _ = check_and_disable_bitsandbytes_loading(
        config, load_in_4bit = True, load_in_8bit = False, verbose = False
    )
    assert load_in_4bit is True
    assert load_in_8bit is False


def test_nvfp4_disables_both_4bit_and_8bit():
    config = _make_config(quantization_config = _NVFP4_QCFG_DICT)

    load_in_4bit, load_in_8bit, _ = check_and_disable_bitsandbytes_loading(
        config, load_in_4bit = True, load_in_8bit = True, verbose = False
    )
    assert load_in_4bit is False
    assert load_in_8bit is False


def test_verbose_flag_does_not_raise():
    config = _make_config(quantization_config = _NVFP4_QCFG_DICT)
    load_in_4bit, load_in_8bit, _ = check_and_disable_bitsandbytes_loading(
        config, load_in_4bit = True, load_in_8bit = False, verbose = True
    )
    assert load_in_4bit is False
    assert load_in_8bit is False


def test_empty_quantization_config_is_not_quantized():
    config = _make_config(quantization_config = {})
    assert get_quant_type(config) is None

    load_in_4bit, load_in_8bit, _ = check_and_disable_bitsandbytes_loading(
        config, load_in_4bit = True, load_in_8bit = False, verbose = False
    )
    assert load_in_4bit is True


if __name__ == "__main__":
    test_nvfp4_config_has_compressed_tensors()
    test_regular_bnb_config_has_bitsandbytes()
    test_nvfp4_disables_load_in_4bit()
    test_bnb_does_not_disable_load_in_4bit()
    test_no_quantization_config_leaves_settings_unchanged()
    test_nvfp4_disables_both_4bit_and_8bit()
    test_verbose_flag_does_not_raise()
    test_empty_quantization_config_is_not_quantized()
    print("All tests passed!")
