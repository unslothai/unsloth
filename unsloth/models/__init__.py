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

from .llama     import FastLlamaModel
from .loader    import FastLanguageModel, FastVisionModel, FastTextModel, FastModel
from .mistral   import FastMistralModel
from .qwen2     import FastQwen2Model
from .qwen3     import FastQwen3Model
from .qwen3_moe import FastQwen3MoeModel
from .granite   import FastGraniteModel
try:
    from .falcon_h1 import FastFalconH1Model
except:
    # transformers_version < 4.53.0 does not have falcon_h1 so silenty skip it for now
    pass
from .dpo       import PatchDPOTrainer, PatchKTOTrainer
from ._utils import is_bfloat16_supported, is_vLLM_available, __version__
from .rl        import PatchFastRL, vLLMSamplingParams