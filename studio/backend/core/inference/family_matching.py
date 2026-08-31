# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared family-name matching for image and video model identifiers."""

from __future__ import annotations

import re


def family_token_matches(token: str, identifier: str) -> bool:
    """Match a family token as a whole identifier segment with flexible punctuation."""
    parts = re.split(r"[-_.]+", token)
    inner = r"[-_.]+".join(re.escape(part) for part in parts)
    pattern = r"(?:^|[-_./\\])" + inner + r"(?:$|[-_./\\])"
    return re.search(pattern, identifier) is not None
