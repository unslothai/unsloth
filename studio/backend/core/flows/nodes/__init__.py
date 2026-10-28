# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Built-in node handlers; importing this package registers every one of them."""

from core.flows.nodes import (  # noqa: F401
    audio,
    automation,
    benchmark,
    chat,
    data,
    image,
    inputs,
    knowledge,
    logic,
    media_utils,
    memory,
    output,
    story,
    table,
    text,
    training,
    transform,
    video,
    world,
)
