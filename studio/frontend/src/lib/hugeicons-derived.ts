// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";
import { BubbleChatIcon, TestTube01Icon } from "@hugeicons/core-free-icons";

// Drop the last 2 paths (interior bubbles).
export const TestTubeOutlineIcon = TestTube01Icon.slice(
  0,
  3,
) as typeof TestTube01Icon;

// BubbleChatIcon's first path alone: an empty speech bubble.
export const MessageCircleIcon = BubbleChatIcon.slice(
  0,
  1,
) as typeof BubbleChatIcon;


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

export const SiteInfoIcon: IconSvgElement = [
  ["path", { d: "M12.325 7.125H19.625M4.375 16.875H11.675", stroke: "currentColor", strokeLinecap: "round", strokeWidth: "1.5", key: "0" }],
  ["circle", { cx: "5.975", cy: "7.125", r: "2.6", stroke: "currentColor", strokeWidth: "1.5", key: "1" }],
  ["circle", { cx: "18.025", cy: "16.875", r: "2.6", stroke: "currentColor", strokeWidth: "1.5", key: "2" }],
];

// Hugeicons "shield-check" (4.3.5) with "tick-02" at 55%.
export const ShieldCheckIcon: IconSvgElement = [
  ["path", { d: "M20.9922 11.1833V8.28029C20.9922 6.64029 20.9922 5.82028 20.5881 5.28529C20.184 4.75029 19.2703 4.49056 17.4429 3.9711C16.1944 3.6162 15.0938 3.18863 14.2145 2.79829C13.0156 2.2661 12.4161 2 11.9922 2C11.5682 2 10.9688 2.2661 9.7699 2.79829C8.89057 3.18863 7.79002 3.61619 6.54152 3.9711C4.71411 4.49056 3.80041 4.75029 3.3963 5.28529C2.99219 5.82028 2.99219 6.64029 2.99219 8.28029V11.1833C2.99219 16.8085 8.05496 20.1835 10.5861 21.5194C11.1932 21.8398 11.4968 22 11.9922 22C12.4876 22 12.7911 21.8398 13.3982 21.5194C15.9294 20.1835 20.9922 16.8085 20.9922 11.1833Z", stroke: "currentColor", strokeLinecap: "round", strokeWidth: "1.5", key: "0" }],
  ["path", { d: "M8.15 12.3L10.075 14.225L15.85 8.175", stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: "1.5", key: "1" }]
];

// Hugeicons 4.1.1 drawings, kept over their 4.4.0 redraws.
export const AiImageIcon: IconSvgElement = [
  ["circle", { cx: "7", cy: "8", r: "1.5", stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: "1.5", key: "0" }],
  ["path", { d: "M20.9977 11C21 11.4701 21 11.9693 21 12.5C21 16.9783 21 19.2175 19.6088 20.6088C18.2175 22 15.9783 22 11.5 22C7.02166 22 4.78249 22 3.39124 20.6088C2 19.2175 2 16.9783 2 12.5C2 8.02166 2 5.78249 3.39124 4.39124C4.78249 3 7.02166 3 11.5 3C12.0307 3 12.5299 3 13 3.00231", stroke: "currentColor", strokeLinecap: "round", strokeWidth: "1.5", key: "1" }],
  ["path", { d: "M18.5 2L18.7579 2.69703C19.0961 3.61102 19.2652 4.06802 19.5986 4.40139C19.932 4.73477 20.389 4.90387 21.303 5.24208L22 5.5L21.303 5.75792C20.389 6.09613 19.932 6.26524 19.5986 6.59861C19.2652 6.93198 19.0961 7.38898 18.7579 8.30297L18.5 9L18.2421 8.30297C17.9039 7.38898 17.7348 6.93198 17.4014 6.59861C17.068 6.26524 16.611 6.09613 15.697 5.75792L15 5.5L15.697 5.24208C16.611 4.90387 17.068 4.73477 17.4014 4.40139C17.7348 4.06802 17.9039 3.61102 18.2421 2.69703L18.5 2Z", stroke: "currentColor", strokeLinejoin: "round", strokeWidth: "1.5", key: "2" }],
  ["path", { d: "M4.5 21.4999C8.87246 16.275 13.7741 9.384 20.9975 14.0424", stroke: "currentColor", strokeWidth: "1.5", key: "3" }],
];
export const AiSecurity03Icon: IconSvgElement = [
  ["path", { d: "M11.6769 8.67348C11.8274 8.43697 12.1726 8.43697 12.3231 8.67348L12.7586 9.35767C13.2401 10.1143 13.8818 10.756 14.6384 11.2375L15.3226 11.6729C15.5591 11.8235 15.5591 12.1687 15.3226 12.3192L14.6384 12.7547C13.8818 13.2362 13.2401 13.8779 12.7586 14.6345L12.3231 15.3187C12.1726 15.5552 11.8274 15.5552 11.6769 15.3187L11.2414 14.6345C10.7599 13.8779 10.1182 13.2362 9.36157 12.7547L8.67738 12.3192C8.44087 12.1687 8.44087 11.8235 8.67738 11.6729L9.36157 11.2375C10.1182 10.756 10.7599 10.1143 11.2414 9.35767L11.6769 8.67348Z", stroke: "currentColor", strokeLinejoin: "round", strokeWidth: "1.5", key: "0" }],
  ["path", { d: "M3.9068 5.28387C6.87149 5.4984 8.78311 2.49713 12.0262 2.49713C15.2208 2.43341 16.784 5.32395 20.059 5.32395C21.8147 14.2606 18.1622 19.8743 12.053 21.4961C6.38992 20.15 2.13481 14.4788 3.9068 5.28387Z", stroke: "currentColor", strokeLinejoin: "round", strokeWidth: "1.5", key: "1" }],
];
export const Album02Icon: IconSvgElement = [
  ["path", { d: "M6 17.9745C6.1287 19.2829 6.41956 20.1636 7.07691 20.8209C8.25596 22 10.1536 22 13.9489 22C17.7442 22 19.6419 22 20.8209 20.8209C22 19.6419 22 17.7442 22 13.9489C22 10.1536 22 8.25596 20.8209 7.07691C20.1636 6.41956 19.2829 6.1287 17.9745 6", stroke: "currentColor", strokeWidth: "1.5", key: "0" }],
  ["path", { d: "M2 10C2 6.22876 2 4.34315 3.17157 3.17157C4.34315 2 6.22876 2 10 2C13.7712 2 15.6569 2 16.8284 3.17157C18 4.34315 18 6.22876 18 10C18 13.7712 18 15.6569 16.8284 16.8284C15.6569 18 13.7712 18 10 18C6.22876 18 4.34315 18 3.17157 16.8284C2 15.6569 2 13.7712 2 10Z", stroke: "currentColor", strokeWidth: "1.5", key: "1" }],
  ["path", { d: "M2 11.1185C2.61902 11.0398 3.24484 11.001 3.87171 11.0023C6.52365 10.9533 9.11064 11.6763 11.1711 13.0424C13.082 14.3094 14.4247 16.053 15 18", stroke: "currentColor", strokeLinejoin: "round", strokeWidth: "1.5", key: "2" }],
  ["path", { d: "M13.125 7H13M13.25 7C13.25 7.13807 13.1381 7.25 13 7.25C12.8619 7.25 12.75 7.13807 12.75 7C12.75 6.86193 12.8619 6.75 13 6.75C13.1381 6.75 13.25 6.86193 13.25 7Z", stroke: "currentColor", strokeLinecap: "round", strokeWidth: "1.5", key: "3" }],
];

// Text to Speech is ai-speech as drawn, sparkle included.
export {
  AiSpeechIcon as TextToSpeechIcon,
  Bookmark01Icon as BookmarkIcon,
  ChipIcon,
  HandIcon,
  ImagesIcon,
  Settings01Icon,
  SheetIcon,
  TrashIcon,
} from "@hugeicons/core-free-icons";
