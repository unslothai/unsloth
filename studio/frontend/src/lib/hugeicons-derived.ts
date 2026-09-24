// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Derived HugeIcons shared between the sidebar and page tabs, so the same visual language appears everywhere.

import type { IconSvgElement } from "@hugeicons/react";
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

// HugeIcons' free "sheet" (stroke rounded, MIT), which first shipped after the version pinned here.
// Copied verbatim from @hugeicons/core-free-icons 4.3.5; drop it for the import once that is bumped.
export const SheetIcon: IconSvgElement = [
  ["path", { d: "M2.49219 9.5H21.4922", stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: "1.5", key: "0" }],
  ["path", { d: "M2.99219 15.5H20.9922", stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: "1.5", key: "1" }],
  ["path", { d: "M8.49219 21L8.49219 9.5", stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: "1.5", key: "2" }],
  ["path", { d: "M15.4922 21L15.4922 9.5", stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: "1.5", key: "3" }],
  [
    "path",
    {
      d: "M2.49219 12C2.49219 7.52166 2.49219 5.28249 3.88343 3.89124C5.27467 2.5 7.51384 2.5 11.9922 2.5C16.4705 2.5 18.7097 2.5 20.1009 3.89124C21.4922 5.28249 21.4922 7.52166 21.4922 12C21.4922 16.4783 21.4922 18.7175 20.1009 20.1088C18.7097 21.5 16.4705 21.5 11.9922 21.5C7.51384 21.5 5.27467 21.5 3.88343 20.1088C2.49219 18.7175 2.49219 16.4783 2.49219 12Z",
      stroke: "currentColor",
      strokeWidth: "1.5",
      key: "4",
    },
  ],
];

// Five-point star with lightly softened tips; HugeIcons' StarIcon reads as a blob when filled.
// Path from Lucide's "star" (ISC).
export const StarPointedIcon: IconSvgElement = [
  [
    "path",
    {
      d: "M11.525 2.295a.53.53 0 0 1 .95 0l2.31 4.679a2.123 2.123 0 0 0 1.595 1.16l5.166.756a.53.53 0 0 1 .294.904l-3.736 3.638a2.123 2.123 0 0 0-.611 1.878l.882 5.14a.53.53 0 0 1-.771.56l-4.618-2.428a2.122 2.122 0 0 0-1.973 0L6.396 21.01a.53.53 0 0 1-.77-.56l.881-5.139a2.122 2.122 0 0 0-.611-1.879L2.16 9.795a.53.53 0 0 1 .294-.906l5.165-.755a2.122 2.122 0 0 0 1.597-1.16z",
      stroke: "currentColor",
      strokeLinecap: "round",
      strokeLinejoin: "round",
      strokeWidth: "1.5",
      key: "0",
    },
  ],
];
