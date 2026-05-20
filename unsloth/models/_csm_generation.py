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

import inspect
import logging
import types


logger = logging.getLogger(__name__)


def patch_csm_depth_decoder_generate(model):
    depth_decoder = getattr(model, "depth_decoder", None)
    if depth_decoder is None or getattr(
        depth_decoder, "_unsloth_csm_generation_patched", False
    ):
        return

    prepare_inputs_for_generation = getattr(
        depth_decoder, "prepare_inputs_for_generation", None
    )
    if prepare_inputs_for_generation is None:
        return

    try:
        parameters = inspect.signature(prepare_inputs_for_generation).parameters
    except (TypeError, ValueError):
        logger.debug(
            "Could not inspect CSM depth decoder generation signature.",
            exc_info = True,
        )
        return
    if "backbone_last_hidden_state" in parameters:
        depth_decoder._unsloth_csm_generation_patched = True
        return

    def _prepare_inputs_for_generation(
        self,
        *args,
        backbone_last_hidden_state = None,
        **kwargs,
    ):
        if backbone_last_hidden_state is not None:
            kwargs["backbone_last_hidden_state"] = backbone_last_hidden_state
        return prepare_inputs_for_generation(*args, **kwargs)

    depth_decoder.prepare_inputs_for_generation = types.MethodType(
        _prepare_inputs_for_generation, depth_decoder
    )
    depth_decoder._unsloth_csm_generation_patched = True
