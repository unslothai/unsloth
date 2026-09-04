# Copyright 2026-present the Unforgettable contributors.
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

from unforgettable import inner_model_id, is_virtual_model


def test_virtual_model_ids():
    assert is_virtual_model("unforgettable")
    assert is_virtual_model("unforgettable/qwen")
    assert not is_virtual_model("default")
    assert not is_virtual_model(None)
    assert inner_model_id("unforgettable") == "default"
    assert inner_model_id("unforgettable/my-gguf") == "my-gguf"
    assert inner_model_id("unforgettable/unforgettable") == "default"
    assert inner_model_id("unforgettable/unforgettable/qwen") == "qwen"
    assert inner_model_id("other") == "other"
