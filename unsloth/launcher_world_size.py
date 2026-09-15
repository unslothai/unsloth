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

Kept in sync with studio/backend/core/training/dataset_bounds.py so import-time
Unsloth setup and Studio row bounds agree on MPI / torchrun / mlx.launch counts.
"""

from __future__ import annotations

import json
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
WORLD_SIZE_ENV_FILES = (
    "MLX_HOSTFILE",
    "MLX_IBV_DEVICES",
)
MAX_WORLD_SIZE_FILE_BYTES = 1 << 20


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _positive_int(value: Any, default: int) -> int:
    number = _int_or(value, default)
    return number if number > 0 else default


def world_size_from_rank_files(environ: Any = None) -> int:
    source = os.environ if environ is None else environ
    sizes = [1]
    for name in WORLD_SIZE_ENV_FILES:
        try:
            value = source.get(name)
            if not value:
                continue
            if value.lstrip()[:1] in ("[", "{"):
                payload = json.loads(value[:MAX_WORLD_SIZE_FILE_BYTES])
            elif os.path.isfile(value):
                with open(value, "rb") as handle:
                    payload = json.loads(handle.read(MAX_WORLD_SIZE_FILE_BYTES))
            else:
                continue
        except (OSError, UnicodeError, ValueError, TypeError, AttributeError):
            continue
        if isinstance(payload, dict):
            payload = payload.get("hosts")
        if isinstance(payload, list):
            sizes.append(len(payload))
    return max(sizes)


def world_size_from_env(environ: Any = None) -> int:
    """Data-parallel processes the launcher advertises, or 1 when none does."""
    source = os.environ if environ is None else environ
    numbers = max(_positive_int(source.get(name), 1) for name in WORLD_SIZE_ENV_VARS)
    return max(numbers, world_size_from_rank_files(source))
