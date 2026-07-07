// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Derived HugeIcons shared between the sidebar and page tabs, so the same
// visual language ("Train" = test tube, "Create" = pencil) appears everywhere.

import { TestTube01Icon } from "@hugeicons/core-free-icons";

// TestTube01Icon's last 2 paths are interior bubbles; slice to the first
// 3 (outline + cap + liquid line) to drop them. Original export untouched.
export const TestTubeOutlineIcon = TestTube01Icon.slice(
  0,
  3,
) as typeof TestTube01Icon;
