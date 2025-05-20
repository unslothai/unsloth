# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os

UNSLOTH_COMPILE_DEBUG         = os.environ.get("UNSLOTH_COMPILE_DEBUG",         "0") == "1"
UNSLOTH_COMPILE_MAXIMUM       = os.environ.get("UNSLOTH_COMPILE_MAXIMUM",       "0") == "1"
UNSLOTH_COMPILE_IGNORE_ERRORS = os.environ.get("UNSLOTH_COMPILE_IGNORE_ERRORS", "0") == "1"
torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : UNSLOTH_COMPILE_MAXIMUM,
    "shape_padding"     : True,
    "trace.enabled"     : UNSLOTH_COMPILE_DEBUG,
    "triton.cudagraphs" : False,
}

global TEMPORARY_PATCHES
TEMPORARY_PATCHES = []

__all__ = ["TEMPORARY_PATCHES", "torch_compile_options"]
