# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""early PyTorch allocator-config normalization for Unsloth processes."""

import os
import re
from typing import List, MutableMapping, Optional, Tuple


ALLOCATOR_CONF_ENV_VARS = (
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_HIP_ALLOC_CONF",
)

# bracketed numeric lists such as roundup_power2_divisions:[32:256,64:128] cannot match.
_BOOL_OPTION = re.compile(
    r"(?P<head>(?:^|,)\s*[A-Za-z_][A-Za-z0-9_]*\s*:\s*)(?P<value>true|false)(?=\s*(?:,|$))",
    re.IGNORECASE,
)


def _canonical_bool(match: "re.Match[str]") -> str:
    return match.group("head") + match.group("value").capitalize()


def normalize_allocator_conf(
    env: Optional[MutableMapping[str, str]] = None,
) -> List[Tuple[str, str, str]]:
    """normalize before CUDA init to avoid PyTorch parser failures; returns (name, old, new)."""
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
