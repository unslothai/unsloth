# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Early PyTorch allocator-config normalization for Unsloth processes."""

import os
import re
from typing import List, MutableMapping, Optional, Tuple


ALLOCATOR_CONF_ENV_VARS = (
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_HIP_ALLOC_CONF",
)

# A `key:value` option whose whole value is a boolean in any case. Bracketed lists such as
# roundup_power2_divisions:[32:256,64:128] only hold numbers, so they never match.
_BOOL_OPTION = re.compile(
    r"(?P<head>(?:^|,)\s*[A-Za-z_][A-Za-z0-9_]*\s*:\s*)(?P<value>true|false)(?=\s*(?:,|$))",
    re.IGNORECASE,
)


def _canonical_bool(match: "re.Match[str]") -> str:
    return match.group("head") + match.group("value").capitalize()


def normalize_allocator_conf(
    env: Optional[MutableMapping[str, str]] = None,
) -> List[Tuple[str, str, str]]:
    """Rewrite lowercase allocator booleans such as ``expandable_segments:false`` to ``False``.

    PyTorch accepts only ``True``/``False``. Anything else makes the first CUDA init raise
    ``ValueError ... in ConfigTokenizer``, and a later CUDA call in the same process then
    finds a half-initialized allocator and segfaults (an access violation on Windows), so
    Studio would crash with no cause in its logs. Must run before anything initializes
    CUDA. Returns ``(name, old, new)`` for each variable it changed.
    """
    environ = os.environ if env is None else env
    changed = []
    for name in ALLOCATOR_CONF_ENV_VARS:
        value = environ.get(name)
        if not value:
            continue
        normalized = _BOOL_OPTION.sub(_canonical_bool, value)
        if normalized != value:
            environ[name] = normalized
            changed.append((name, value, normalized))
    return changed
