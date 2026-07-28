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

"""Unsloth integration for the ExLlamaV3 (EXL3) quantization backend."""

from .config import (
    Exl3Config,
    normalize_exl3_config,
    DEFAULT_EXL3_BITS,
    DEFAULT_EXL3_HEAD_BITS,
)
from .utils import (
    is_exllama_available,
    exllama_version,
    require_exllama,
    EXLLAMA_IMPORT_ERROR,
)
from .quant_linear import (
    Exl3QuantState,
    ExllamaV3Linear,
    get_exl3_quant_state,
    exl3_fast_dequantize,
    is_exl3_linear,
)
from .patcher import (
    patch_transformers_exl3,
    is_exl3_model_dir,
    read_exl3_bitrate,
    exllama_supports_arch,
    exllama_supported_architectures,
)
from .quantize import (
    quantize_to_exl3,
    resolve_exl3_cache_dir,
    is_moe_model,
    ensure_calibration_data,
)
from .loader import (
    Exl3LoadPlan,
    should_use_exl3,
    exl3_is_default_backend,
    resolve_exl3_config,
    prepare_exl3_checkpoint,
    finalize_exl3_model,
    finalize_exl3_experts,
)
from .moe import reload_exl3_experts

__all__ = [
    "Exl3Config",
    "normalize_exl3_config",
    "DEFAULT_EXL3_BITS",
    "DEFAULT_EXL3_HEAD_BITS",
    "is_exllama_available",
    "exllama_version",
    "require_exllama",
    "EXLLAMA_IMPORT_ERROR",
    "Exl3QuantState",
    "ExllamaV3Linear",
    "get_exl3_quant_state",
    "exl3_fast_dequantize",
    "is_exl3_linear",
    "patch_transformers_exl3",
    "is_exl3_model_dir",
    "read_exl3_bitrate",
    "exllama_supports_arch",
    "exllama_supported_architectures",
    "quantize_to_exl3",
    "resolve_exl3_cache_dir",
    "is_moe_model",
    "ensure_calibration_data",
    "Exl3LoadPlan",
    "should_use_exl3",
    "exl3_is_default_backend",
    "resolve_exl3_config",
    "prepare_exl3_checkpoint",
    "finalize_exl3_model",
    "finalize_exl3_experts",
    "reload_exl3_experts",
]
