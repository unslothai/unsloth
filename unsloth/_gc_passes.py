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

"""Repeated full gc.collect() passes that stop once a pass that really ran finds nothing. Imports only gc."""

import gc

__all__ = ["collect_garbage"]


def collect_garbage(
    passes = 3,
    before = None,
    after = None,
):
    """Up to `passes` full gc.collect() passes, with `before` / `after` (e.g. a cache flush) run around each.

    Stops after a pass that ran and found nothing: it freed nothing and ran no finalizer, so the next pass
    would walk the same heap and find nothing either. Keeps going, as every pass used to, whenever that is
    not certain: gc.callbacks are registered (they run around every pass and may drop references), the
    pass did not return 0 (None when gc.collect is swapped out), or the full-collection stats do not show
    exactly one more collection that freed nothing (CPython returns 0 without collecting while another
    thread's collection is in progress).
    """
    for _ in range(passes):
        if before is not None:
            before()
        seen = gc.get_stats()[2]
        found = gc.collect()
        ran = gc.get_stats()[2]
        if after is not None:
            after()
        if (
            found == 0
            and not gc.callbacks
            and ran["collections"] == seen["collections"] + 1
            and ran["collected"] == seen["collected"]
            and ran["uncollectable"] == seen["uncollectable"]
        ):
            break
