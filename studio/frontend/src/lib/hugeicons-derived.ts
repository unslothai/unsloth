// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";
import { BubbleChatIcon, TestTube01Icon } from "@hugeicons/core-free-icons";

// Drop the last 2 paths (interior bubbles).
export const TestTubeOutlineIcon = TestTube01Icon.slice(
  0,
  3,
) as typeof TestTube01Icon;

// BubbleChatIcon's first path alone is HugeIcons' message-circle, which this set lacks.
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

// From @hugeicons/core-free-icons 4.3.5; the installed 4.1 predates it.
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

// HugeIcons "images" from @hugeicons/core-free-icons 4.3.5; the installed 4.1 predates it.
export const ImagesIcon: IconSvgElement = [
  ["path", { d: "M4 17.0429L8.75635 12.6942C10.4331 11.1612 11.2714 10.3947 12.3096 10.3191C13.3478 10.2435 14.2884 10.8805 16.1695 12.1545L17.5 13.0556", stroke: "currentColor", strokeLinecap: "round", strokeWidth: "1.5", key: "0" }],
  ["path", { d: "M7.375 7.25H7.25M7.5 7.25C7.5 7.38807 7.38807 7.5 7.25 7.5C7.11193 7.5 7 7.38807 7 7.25C7 7.11193 7.11193 7 7.25 7C7.38807 7 7.5 7.11193 7.5 7.25Z", stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: "1.5", key: "1" }],
  ["path", { d: "M6 17.9745C6.1287 19.2829 6.41956 20.1636 7.07691 20.8209C8.25596 22 10.1536 22 13.9489 22C17.7442 22 19.6419 22 20.8209 20.8209C22 19.6419 22 17.7442 22 13.9489C22 10.1536 22 8.25596 20.8209 7.07691C20.1636 6.41956 19.2829 6.1287 17.9745 6", stroke: "currentColor", strokeWidth: "1.5", key: "2" }],
  ["path", { d: "M2 10C2 6.22876 2 4.34315 3.17157 3.17157C4.34315 2 6.22876 2 10 2C13.7712 2 15.6569 2 16.8284 3.17157C18 4.34315 18 6.22876 18 10C18 13.7712 18 15.6569 16.8284 16.8284C15.6569 18 13.7712 18 10 18C6.22876 18 4.34315 18 3.17157 16.8284C2 15.6569 2 13.7712 2 10Z", stroke: "currentColor", strokeWidth: "1.5", key: "3" }],
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

// Hugeicons "ai-speech" and "text-to-speach" (MIT), only in core-free-icons 4.3+, newer than the pinned 4.1.1.
const speechStroke = {
  stroke: "currentColor",
  strokeLinecap: "round",
  strokeLinejoin: "round",
  strokeWidth: "1.5",
} as const;

// Audio everywhere: ai-speech without its sparkle, the last two bars filled out (5-21, 11-15) into the space it left.
export const AiSpeechIcon: IconSvgElement = [
  ["path", { d: "M12 7V17", ...speechStroke, key: "0" }],
  ["path", { d: "M16 5L16 21", ...speechStroke, key: "1" }],
  ["path", { d: "M20 11L20 15", ...speechStroke, key: "2" }],
  ["path", { d: "M8 3V21", ...speechStroke, key: "3" }],
  ["path", { d: "M4 9V15", ...speechStroke, key: "4" }],
];

// Text to Speech: hugeicons' ai-speech as drawn, sparkle included.
export const TextToSpeechIcon: IconSvgElement = [
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

/** `icon` with an accent dot; colours go in `style` because HugeiconsIcon overwrites child stroke attributes. */
export function withNoticeDot(icon: IconSvgElement): IconSvgElement {
  return [
    ...icon,
    [
      "circle",
      {
        cx: "19",
        cy: "5",
        r: "3.75",
        key: "notice-dot",
        style: {
          fill: "var(--primary)",
          stroke: "var(--icon-dot-ring, var(--sidebar))",
          strokeWidth: 2.5,
          paintOrder: "stroke",
        },
      },
    ],
  ] as unknown as IconSvgElement;
}

// Speedometer: open-bottom dial broken where the needle points; HugeIcons' DashboardSpeed icons are framed or half-dial.
export const SpeedMeterIcon: IconSvgElement = [
  ["path", { d: "M4.929 19.571A10 10 0 0 1 16.226 3.437", stroke: "currentColor", strokeLinecap: "round", strokeWidth: "1.5", key: "0" }],
  ["path", { d: "M21.063 8.274A10 10 0 0 1 19.071 19.571", stroke: "currentColor", strokeLinecap: "round", strokeWidth: "1.5", key: "1" }],
  ["circle", { cx: "12", cy: "12.5", r: "2.25", stroke: "currentColor", strokeWidth: "1.5", key: "2" }],
  ["path", { d: "M13.59 10.91L18.01 6.49", stroke: "currentColor", strokeLinecap: "round", strokeWidth: "1.5", key: "3" }],
];

// FileSearchIcon scaled 1.1x about the centre to match the square menu glyphs.
export const FileSearchIcon: IconSvgElement = [
  ["path", { d: "M13.65 16.95L15.3 18.6M14.2 14.75C14.2 16.2688 12.9688 17.5 11.45 17.5C9.9312 17.5 8.7 16.2688 8.7 14.75C8.7 13.2312 9.9312 12 11.45 12C12.9688 12 14.2 13.2312 14.2 14.75Z", stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: "1.5", key: "0" }],
  ["path", { d: "M13.1 1.55V2.1C13.1 5.2113 13.1 6.7669 14.0666 7.7335C15.0331 8.7 16.5888 8.7 19.7 8.7H20.25M20.8 10.5226V14.2C20.8 18.3483 20.8 20.4226 19.5112 21.7112C18.2226 23 16.1483 23 12 23C7.8516 23 5.7775 23 4.4887 21.7112C3.2 20.4226 3.2 18.3483 3.2 14.2V9.2014C3.2 5.6319 3.2 3.8471 4.1747 2.6383C4.3716 2.394 4.594 2.1716 4.8383 1.9747C6.0471 1 7.8319 1 11.4014 1C12.1775 1 12.5655 1 12.9209 1.1254C12.9948 1.1515 13.0672 1.1815 13.138 1.2153C13.478 1.3779 13.7523 1.6523 14.3011 2.2011L19.5112 7.4113C20.1472 8.0471 20.465 8.3651 20.6326 8.7694C20.8 9.1737 20.8 9.6233 20.8 10.5226Z", stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: "1.5", key: "1" }],
];
