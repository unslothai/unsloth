# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Trimmed to the shape of ROCm/FLM_Q4NX_Converter's q4nx/constants.py at the pinned q4k commit.
from enum import IntEnum, auto


class ModelArch(IntEnum):
    QWEN3 = auto()
    QWEN35_08B = auto()
    QWEN35_9B = auto()
    QWEN35_OMNI = auto()
    QWEN35_A3B = auto()


ModelArchNames: dict[ModelArch, list[str]] = {
    ModelArch.QWEN35_08B: ["qwen35-0.8B", "qwen3.5-0.8B"],
    ModelArch.QWEN35_9B: ["qwen35-9B", "qwen3.5-9B"],
    ModelArch.QWEN35_OMNI: ["qwen35-omni", "qwen3.5-omni"],
    ModelArch.QWEN35_A3B: ["qwen35moe"],
    ModelArch.QWEN3: ["qwen3"],
}
