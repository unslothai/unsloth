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

__version__ = "2025.4.3"

from importlib.util import find_spec
if find_spec("unsloth") is None:
    raise ImportError("Please install Unsloth via `pip install unsloth`!")
pass
del find_spec

import os
if not ("UNSLOTH_IS_PRESENT" in os.environ):
    raise ImportError("Please install Unsloth via `pip install unsloth`!")
pass

try:
    print("🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.")
except:
    print("Unsloth: Will patch your computer to enable 2x faster free finetuning.")
pass
# Log Unsloth-Zoo Utilities
os.environ["UNSLOTH_ZOO_IS_PRESENT"] = "1"
del os
