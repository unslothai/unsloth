# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ._attn_mask_compat import AttentionMaskConverter


class Gemma2AttentionMaskConverter(AttentionMaskConverter):
    """Gemma2's window counts the current key as well as the preceding keys."""

    @staticmethod
    def _make_causal_mask(
        input_ids_shape,
        dtype,
        device,
        past_key_values_length = 0,
        sliding_window = None,
    ):
        # The legacy converter counts preceding keys; its static helper also accepts zero for a one-key window.
        return AttentionMaskConverter._make_causal_mask(
            input_ids_shape,
            dtype,
            device,
            past_key_values_length = past_key_values_length,
            sliding_window = sliding_window - 1 if sliding_window is not None else None,
        )
