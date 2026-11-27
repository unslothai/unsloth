# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from urllib.parse import quote


def attachment_disposition(filename: str, fallback: str) -> str:
    """An attachment header with an ASCII ``filename`` for old clients and the real name in ``filename*``."""
    ascii_name = "".join(
        ch if 32 <= ord(ch) < 127 and ch not in {'"', "\\"} else "_" for ch in filename
    )
    if not ascii_name.strip("_"):
        ascii_name = fallback
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"
