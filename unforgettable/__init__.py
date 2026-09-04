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

"""Progressive memory (B) and episode harness. No Studio imports."""

__version__ = "0.1.0"

VIRTUAL_MODEL_ID = "unforgettable"
VIRTUAL_MODEL_PREFIX = "unforgettable/"


def is_virtual_model(model: str | None) -> bool:
    if not model:
        return False
    return model == VIRTUAL_MODEL_ID or model.startswith(VIRTUAL_MODEL_PREFIX)


def inner_model_id(model: str | None) -> str:
    """Strip the virtual alias so the inner wheel uses a real model id."""
    if not model:
        return "default"
    current = model
    while True:
        if current == VIRTUAL_MODEL_ID:
            return "default"
        if current.startswith(VIRTUAL_MODEL_PREFIX):
            current = current[len(VIRTUAL_MODEL_PREFIX) :] or "default"
            continue
        return current
