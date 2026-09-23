// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Derived HugeIcons shared between the sidebar and page tabs, so the same visual language appears everywhere.

import { BubbleChatIcon, TestTube01Icon } from "@hugeicons/core-free-icons";

// TestTube01Icon's last 2 paths are interior bubbles; slice to the first 3 (outline + cap + liquid line). Original export untouched.
export const TestTubeOutlineIcon = TestTube01Icon.slice(
  0,
  3,
) as typeof TestTube01Icon;

// HugeIcons' own message-circle, which this icon set does not ship: BubbleChatIcon is that same
// round bubble with a second path drawing the three dots inside it, so the first path alone is it.
// This is the glyph for a chat anywhere in the app. Original export untouched.
export const MessageCircleIcon = BubbleChatIcon.slice(
  0,
  1,
) as typeof BubbleChatIcon;
