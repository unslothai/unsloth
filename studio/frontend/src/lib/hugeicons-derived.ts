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

// Hugeicons chip as drawn in core-free-icons 4.4; the installed 4.1 draws an older version.
export const ChipIcon: IconSvgElement = [
  ["path", { d: "M13 20H11C7.70017 20 6.05025 20 5.02513 18.9749C4 17.9497 4 16.2998 4 13V11C4 7.70017 4 6.05025 5.02513 5.02513C6.05025 4 7.70017 4 11 4H13C16.2998 4 17.9497 4 18.9749 5.02513C20 6.05025 20 7.70017 20 11V13C20 16.2998 20 17.9497 18.9749 18.9749C17.9497 20 16.2998 20 13 20Z", stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: "1.5", key: "0" }],
  ["path", { d: "M22 9.00003L20 9.00003M4 9.00003L2 9.00003M20 15L22 15M4 15H2M9 2L9 4M9 20L9 22M15 4L15 2M15 20L15 22", stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: "1.5", key: "1" }],
  ["path", { d: "M12 16C13.8856 16 14.8284 16 15.4142 15.4142C16 14.8284 16 13.8856 16 12C16 10.1144 16 9.17157 15.4142 8.58579C14.8284 8 13.8856 8 12 8C10.1144 8 9.17157 8 8.58579 8.58579C8 9.17157 8 10.1144 8 12C8 13.8856 8 14.8284 8.58579 15.4142C9.17157 16 10.1144 16 12 16Z", stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: "1.5", key: "2" }],
];

export const SiteInfoIcon: IconSvgElement = [
  ["path", { d: "M12.325 7.125H19.625M4.375 16.875H11.675", stroke: "currentColor", strokeLinecap: "round", strokeWidth: "1.5", key: "0" }],
  ["circle", { cx: "5.975", cy: "7.125", r: "2.6", stroke: "currentColor", strokeWidth: "1.5", key: "1" }],
  ["circle", { cx: "18.025", cy: "16.875", r: "2.6", stroke: "currentColor", strokeWidth: "1.5", key: "2" }],
];

// Hugeicons "settings-01" from core-free-icons 4.4.0; the pinned 4.1.1 has the older drawing.
export const Settings01Icon: IconSvgElement = [
  ["path", { d: "M3.95631 10.5829L3.03254 9.07084C2.74505 8.60024 2.7389 8.00978 3.01652 7.5333L3.63533 6.47124C3.90953 6.00064 4.41721 5.71571 4.96162 5.7269L6.69131 5.76245C6.92616 5.76728 7.15379 5.68727 7.34333 5.54846C7.79531 5.21745 8.28334 4.93276 8.80042 4.70139C9.01191 4.60675 9.19222 4.45165 9.30469 4.24903L10.1203 2.77965C10.3847 2.30328 10.8865 2.00781 11.4312 2.00781H12.6937C13.2384 2.00781 13.7403 2.30329 14.0047 2.77966L14.8202 4.24902C14.9327 4.45165 15.113 4.60675 15.3245 4.70139C15.8416 4.93274 16.3296 5.2174 16.7815 5.54838C16.9711 5.68718 17.1987 5.76718 17.4336 5.76235L19.1634 5.72676C19.7078 5.71557 20.2155 6.00049 20.4897 6.47111L21.1085 7.53314C21.3862 8.00963 21.38 8.60009 21.0925 9.07068L20.1687 10.5828C20.0481 10.7803 20.002 11.0134 20.0235 11.2439C20.0707 11.7486 20.0707 12.267 20.0235 12.7717C20.002 13.0022 20.0481 13.2353 20.1687 13.4329L21.0924 14.9448C21.3799 15.4154 21.3861 16.0058 21.1084 16.4823L20.4896 17.5444C20.2154 18.015 19.7077 18.2999 19.1633 18.2887L17.4338 18.2531C17.1989 18.2483 16.9713 18.3283 16.7817 18.4671C16.3297 18.7982 15.8416 19.0829 15.3245 19.3142C15.113 19.4089 14.9327 19.564 14.8202 19.7666L14.0047 21.236C13.7403 21.7123 13.2384 22.0078 12.6937 22.0078H11.4312C10.8866 22.0078 10.3847 21.7123 10.1203 21.236L9.30468 19.7665C9.19222 19.5639 9.01191 19.4088 8.80042 19.3142C8.28329 19.0828 7.79519 18.798 7.34316 18.467C7.15362 18.3282 6.92597 18.2482 6.69111 18.253L4.96175 18.2886C4.41733 18.2998 3.90963 18.0148 3.63543 17.5442L3.01663 16.4822C2.739 16.0057 2.74515 15.4152 3.03264 14.9446L3.95631 13.4327C4.07699 13.2351 4.12309 13.002 4.10154 12.7715C4.05437 12.267 4.05438 11.7486 4.10154 11.244C4.12308 11.0135 4.07699 10.7805 3.95631 10.5829Z", stroke: "currentColor", strokeWidth: "1.5", key: "0" }],
  ["circle", { cx: "12.0625", cy: "12.0078", r: "3.5", stroke: "currentColor", strokeWidth: "1.5", key: "1" }],
];

// Hugeicons "shield-check" (4.3.5) with "tick-02" at 55%.
export const ShieldCheckIcon: IconSvgElement = [
  ["path", { d: "M20.9922 11.1833V8.28029C20.9922 6.64029 20.9922 5.82028 20.5881 5.28529C20.184 4.75029 19.2703 4.49056 17.4429 3.9711C16.1944 3.6162 15.0938 3.18863 14.2145 2.79829C13.0156 2.2661 12.4161 2 11.9922 2C11.5682 2 10.9688 2.2661 9.7699 2.79829C8.89057 3.18863 7.79002 3.61619 6.54152 3.9711C4.71411 4.49056 3.80041 4.75029 3.3963 5.28529C2.99219 5.82028 2.99219 6.64029 2.99219 8.28029V11.1833C2.99219 16.8085 8.05496 20.1835 10.5861 21.5194C11.1932 21.8398 11.4968 22 11.9922 22C12.4876 22 12.7911 21.8398 13.3982 21.5194C15.9294 20.1835 20.9922 16.8085 20.9922 11.1833Z", stroke: "currentColor", strokeLinecap: "round", strokeWidth: "1.5", key: "0" }],
  ["path", { d: "M8.15 12.3L10.075 14.225L15.85 8.175", stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: "1.5", key: "1" }]
];

// Hugeicons "hand" (4.3.5).
export const HandIcon: IconSvgElement = [
  ["path", { d: "M18.5 11.5V8.75C18.5 7.7835 19.2835 7 20.25 7C21.2165 7 22 7.7835 22 8.75V14.8C22 18.7765 18.7764 22 14.8 22H12.0588M8 14L5.61507 11.2176C5.22468 10.7621 4.65476 10.5 4.05489 10.5H3.91898C2.85916 10.5 2 11.3592 2 12.419C2 12.7978 2.11214 13.1682 2.32229 13.4834L5.7812 18.6718C7.16738 20.7511 9.50102 22 12 22", stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: "1.5", key: "0" }],
  ["path", { d: "M18.5 8.5V5.75C18.5 4.7835 17.7165 4 16.75 4C15.7835 4 15 4.7835 15 5.75V10.5", stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: "1.5", key: "1" }],
  ["path", { d: "M15 10.5V3.75C15 2.7835 14.2165 2 13.25 2C12.2835 2 11.5 2.7835 11.5 3.75V10", stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: "1.5", key: "2" }],
  ["path", { d: "M11.5 10V5.75C11.5 4.7835 10.7165 4 9.75 4C8.7835 4 8 4.7835 8 5.75V14", stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: "1.5", key: "3" }]
];
