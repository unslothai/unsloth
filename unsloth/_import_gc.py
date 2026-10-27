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

"""Fewer cyclic GC passes while `import unsloth` pulls in torch / transformers / trl.

Those imports allocate ~700k long-lived objects, so the default generation 0 threshold
triggers several full collections that find almost nothing. The caller's thresholds come
back at the end of the import. This module outlives a failed `import unsloth` in
sys.modules, so a retry still restores the thresholds from before the first attempt.
"""

import gc

IMPORT_THRESHOLD = 100_000
_saved = None


def raise_gc_threshold():
    global _saved
    current = gc.get_threshold()
    if _saved is not None and current[0] == IMPORT_THRESHOLD:
        return  # Still raised by an earlier attempt that failed.
    # 0 means automatic collection is off; a higher threshold was chosen on purpose.
    if current[0] == 0 or current[0] >= IMPORT_THRESHOLD:
        _saved = None
        return
    _saved = current
    gc.set_threshold(IMPORT_THRESHOLD, *current[1:])


def restore_gc_threshold():
    global _saved
    saved, _saved = _saved, None
    # Leave it alone if something changed it during the import.
    if saved is not None and gc.get_threshold()[0] == IMPORT_THRESHOLD:
        gc.set_threshold(*saved)
