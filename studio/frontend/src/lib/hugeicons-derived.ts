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

// HugeIcons' folder-plus (stroke rounded), from @hugeicons/core-free-icons 4.3.5; the 4.1 installed
// here predates it. The glyph for a project's sources.
export const FolderPlusIcon: IconSvgElement = [
  [
    "path",
    {
      d: "M8 7H16.75C18.8567 7 19.91 7 20.6667 7.50559C20.9943 7.72447 21.2755 8.00572 21.4944 8.33329C22 9.08996 22 10.1433 22 12.25C22 15.7612 22 17.5167 21.1573 18.7779C20.7926 19.3238 20.3238 19.7926 19.7779 20.1573C18.5167 21 16.7612 21 13.25 21H12C7.28595 21 4.92893 21 3.46447 19.5355C2 18.0711 2 15.714 2 11V7.94427C2 6.1278 2 5.21956 2.38032 4.53806C2.65142 4.05227 3.05227 3.65142 3.53806 3.38032C4.21956 3 5.1278 3 6.94427 3C8.10802 3 8.6899 3 9.19926 3.19101C10.3622 3.62712 10.8418 4.68358 11.3666 5.73313L12 7",
      stroke: "currentColor",
      strokeLinecap: "round",
      strokeWidth: "1.5",
      key: "0",
    },
  ],
  [
    "path",
    {
      d: "M12.0028 11V17M15.0078 13.995L9.00781 13.995",
      stroke: "currentColor",
      strokeLinecap: "round",
      strokeLinejoin: "round",
      strokeWidth: "1.5",
      key: "1",
    },
  ],
];

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

// Hugeicons "ai-speech" and "text-to-speach" (MIT), only in core-free-icons 4.3+, newer than the
// pinned 4.1.1. Audio anywhere is ai-speech; Text to Speech is text-to-speach.
const speechStroke = {
  stroke: "currentColor",
  strokeLinecap: "round",
  strokeLinejoin: "round",
  strokeWidth: "1.5",
} as const;

export const AiSpeechIcon: IconSvgElement = [
  ["path", { d: "M12 7V17", ...speechStroke, key: "0" }],
  ["path", { d: "M16 11L16 19", ...speechStroke, key: "1" }],
  ["path", { d: "M20 11L20 14", ...speechStroke, key: "2" }],
  ["path", { d: "M8 3V21", ...speechStroke, key: "3" }],
  ["path", { d: "M4 9V15", ...speechStroke, key: "4" }],
  [
    "path",
    {
      d: "M18.5 3.9375V5.5M18.5 5.5V7.0625M18.5 5.5H17.25M18.5 5.5H19.75M21 5.5L19.9156 5.13852C19.4179 4.97263 19.0274 4.58211 18.8615 4.08443L18.5 3L18.1385 4.08443C17.9726 4.58211 17.5821 4.97263 17.0844 5.13852L16 5.5L17.0844 5.86148C17.5821 6.02737 17.9726 6.41789 18.1385 6.91557L18.5 8L18.8615 6.91557C19.0274 6.41789 19.4179 6.02737 19.9156 5.86148L21 5.5Z",
      ...speechStroke,
      key: "5",
    },
  ],
];

export const TextToSpeechIcon: IconSvgElement = [
  ["path", { d: "M13 9L13 17", ...speechStroke, key: "0" }],
  ["path", { d: "M17 7L17 19", ...speechStroke, key: "1" }],
  ["path", { d: "M21 11L21 15", ...speechStroke, key: "2" }],
  ["path", { d: "M9 15V21", ...speechStroke, key: "3" }],
  ["path", { d: "M5 15V17", ...speechStroke, key: "4" }],
  [
    "path",
    {
      d: "M7.5 3.5V11M7.5 11H6M7.5 11H9M12 4.5C12 3.67157 11.3284 3 10.5 3H4.5C3.67157 3 3 3.67157 3 4.5",
      ...speechStroke,
      key: "5",
    },
  ],
];
