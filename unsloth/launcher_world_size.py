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

"""Launcher-advertised data-parallel world size (torch-free).

Subset of studio/backend/core/training/dataset_bounds.world_size_from_env: the
same WORLD_SIZE / MPI env vars, without MLX rank-file parsing (FLA training is
torch/GPU; multi-rank jobs should export a size var the launcher already sets).
"""

from __future__ import annotations

import os
from typing import Any

WORLD_SIZE_ENV_VARS = (
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "MLX_WORLD_SIZE",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PMIX_SIZE",
    "MPI_WORLD_SIZE",
    "MV2_COMM_WORLD_SIZE",
)


def _positive_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if number > 0 else default


def world_size_from_env(environ: Any = None) -> int:
    """Data-parallel processes the launcher advertises, or 1 when none does."""
    source = os.environ if environ is None else environ
    return max(_positive_int(source.get(name), 1) for name in WORLD_SIZE_ENV_VARS)
