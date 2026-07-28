# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import threading
from contextlib import contextmanager
from typing import Iterator


_TRAINING_LIFECYCLE_LOCK = threading.RLock()


@contextmanager
def training_lifecycle_guard() -> Iterator[None]:
    with _TRAINING_LIFECYCLE_LOCK:
        yield
