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


# Note all the kernels that were previously here were moved to unsloth_zoo
# https://github.com/unslothai/unsloth/pull/4145
# https://github.com/unslothai/unsloth-zoo/pull/529
# This exists here just to serve as a backup or a failsafe
# in case some older version tries to import from here.
# We do not expect such a case to happen but it doesn't hurt to have it here.

import importlib
import sys


_MODULE_SUFFIXES = (
    "",
    ".autotune_cache",
    ".grouped_gemm",
    ".grouped_gemm.interface",
)


def _alias_module(suffix):
    legacy_name = f"{__name__}{suffix}"
    target_name = f"unsloth_zoo.kernels.moe{suffix}"
    try:
        module = importlib.import_module(target_name)
    except ModuleNotFoundError as error:
        if error.name != target_name:
            raise
        return None
    sys.modules[legacy_name] = module
    return module


for _suffix in _MODULE_SUFFIXES:
    _alias_module(_suffix)

_canonical_module = sys.modules.get(f"{__name__}")
autotune_cache = sys.modules.get(f"{__name__}.autotune_cache")
grouped_gemm = sys.modules.get(f"{__name__}.grouped_gemm")


def __getattr__(name):
    return getattr(_canonical_module, name)


__all__ = getattr(_canonical_module, "__all__", ())
