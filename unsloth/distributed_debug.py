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

import os
import sys

__all__ = ["apply_debug_distributed_env"]

# Read by ProcessGroupNCCL when a process group is created, and by NCCL at communicator
# init, so these only need to land before init_process_group.
_DEBUG_DISTRIBUTED_ENV = (
    ("TORCH_NCCL_DESYNC_DEBUG", "1"),
    ("TORCH_NCCL_DUMP_ON_TIMEOUT", "1"),
    ("NCCL_DEBUG", "INFO"),
)
# Flight recorder ring size: torch >= 2.8 reads TORCH_FR_BUFFER_SIZE first and falls back to
# TORCH_NCCL_TRACE_BUFFER_SIZE, the only name torch 2.6 / 2.7 know. 2.7 / 2.8 default it to 0 (off).
_FR_BUFFER_NAMES = ("TORCH_FR_BUFFER_SIZE", "TORCH_NCCL_TRACE_BUFFER_SIZE")
_FR_BUFFER_SIZE = "2000"


def apply_debug_distributed_env(environ = None):
    """Under UNSLOTH_DEBUG_DISTRIBUTED=1, enable NCCL / c10d hang diagnostics.

    Variables the user already set are left alone, and the two flight recorder names
    are treated as one setting, so an explicit TORCH_NCCL_TRACE_BUFFER_SIZE=0 is not
    overridden through the newer name. Returns the names that were set.
    """
    environ = os.environ if environ is None else environ
    if environ.get("UNSLOTH_DEBUG_DISTRIBUTED", "0").lower() not in ("1", "true", "yes", "on"):
        return []
    changed = []
    for name, value in _DEBUG_DISTRIBUTED_ENV:
        if name not in environ:
            environ[name] = value
            changed.append(name)
    if not any(name in environ for name in _FR_BUFFER_NAMES):
        for name in _FR_BUFFER_NAMES:
            environ[name] = _FR_BUFFER_SIZE
            changed.append(name)
    dist = sys.modules.get("torch.distributed")
    try:
        initialized = dist is not None and dist.is_available() and dist.is_initialized()
    except Exception:
        initialized = False
    if initialized and changed:
        print(
            "Unsloth: UNSLOTH_DEBUG_DISTRIBUTED=1 was applied after the process group was "
            "created, so torch.distributed will not see it. Import unsloth before "
            "init_process_group, or export the variables in the launcher.",
            file = sys.stderr,
        )
    return changed
