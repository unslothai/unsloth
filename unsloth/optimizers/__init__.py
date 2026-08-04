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

import importlib

__all__ = [
    "GaLoreProjector",
    "QGaLoreAdamW8bit",
    "make_layerwise_lr_param_groups",
]

# Lazy so importing layerwise_lr (stdlib-only) doesn't also pull in Q-GaLore/bitsandbytes.
_ATTR_TO_MODULE = {
    "GaLoreProjector": "q_galore_projector",
    "QGaLoreAdamW8bit": "q_galore_adamw",
    "make_layerwise_lr_param_groups": "layerwise_lr",
}


def __getattr__(name):
    module_name = _ATTR_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)
