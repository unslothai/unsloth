# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What a batch says about a reply it will not take."""

from __future__ import annotations


class RowRefused(Exception):
    """This batch will not take this reply, and is exactly as it was."""
