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

"""Identify real PEFT adapter directories (sidecar C), vs fake train dirs."""

from __future__ import annotations

import json
import os
from pathlib import Path


def is_peft_adapter_dir(path: str | os.PathLike[str] | None) -> bool:
    """True when ``path`` is a PEFT adapter directory, not a fake sidecar dir."""
    if not path:
        return False
    root = Path(path)
    cfg_path = root / "adapter_config.json"
    if not root.is_dir() or not cfg_path.is_file():
        return False
    try:
        data = json.loads(cfg_path.read_text(encoding = "utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict) or data.get("fake"):
        return False
    return "peft_type" in data or "base_model_name_or_path" in data


def peft_adapter_name(path: str | os.PathLike[str]) -> str:
    """Stable PEFT adapter id derived from the directory name."""
    return Path(path).name.replace(".", "_")
