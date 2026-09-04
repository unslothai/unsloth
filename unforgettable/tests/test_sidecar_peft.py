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

from __future__ import annotations

import json

from unforgettable.sidecar.peft import is_peft_adapter_dir, peft_adapter_name


def test_is_peft_adapter_dir_rejects_fake_and_accepts_peft(tmp_path):
    missing = tmp_path / "nope"
    assert is_peft_adapter_dir(missing) is False
    fake = tmp_path / "fake-ada"
    fake.mkdir()
    (fake / "adapter_config.json").write_text(
        json.dumps({"fake": True, "recipe": "sft", "n": 4}),
        encoding = "utf-8",
    )
    assert is_peft_adapter_dir(fake) is False
    peft = tmp_path / "ada-uuid"
    peft.mkdir()
    (peft / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "unsloth/Qwen3.5-4B",
            }
        ),
        encoding = "utf-8",
    )
    assert is_peft_adapter_dir(peft) is True
    assert peft_adapter_name(peft) == "ada-uuid"
