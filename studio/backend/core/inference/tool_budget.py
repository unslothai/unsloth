# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tool-call budget shared by the request models, the routes and every tool loop."""

# Tool calls per message when a request leaves max_tool_calls_per_message unset; the Studio UI default matches.
DEFAULT_MAX_TOOL_CALLS_PER_MESSAGE = 50
