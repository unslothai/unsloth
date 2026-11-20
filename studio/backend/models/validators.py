# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reusable pydantic field validators."""

from pydantic import AfterValidator


def not_blank(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("must not be blank")
    return value


# Strips the string and refuses one that is empty after stripping.
NotBlank = AfterValidator(not_blank)
