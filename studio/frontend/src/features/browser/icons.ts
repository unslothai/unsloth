// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";

const stroke = {
  stroke: "currentColor",
  strokeLinecap: "round",
  strokeLinejoin: "round",
  strokeWidth: "1.5",
} as const;

export const EnterFullViewIcon: IconSvgElement = [
  ["path", { d: "M13.5 6.5H17.5V10.5", ...stroke, key: "0" }],
  ["path", { d: "M10.5 17.5H6.5V13.5", ...stroke, key: "1" }],
];

export const ExitFullViewIcon: IconSvgElement = [
  ["path", { d: "M17.5 10.5H13.5V6.5", ...stroke, key: "0" }],
  ["path", { d: "M6.5 13.5H10.5V17.5", ...stroke, key: "1" }],
];

export const SplitPaneIcon: IconSvgElement = [
  [
    "rect",
    {
      x: "3",
      y: "4.5",
      width: "18",
      height: "15",
      rx: "4",
      ...stroke,
      key: "0",
    },
  ],
  ["path", { d: "M12 4.5V19.5", ...stroke, key: "1" }],
];

// A padlock drawn as Firefox's is: a narrow shackle over a compact body with a keyhole dot, in the
// same thin stroke as the other browser icons, rather than a wide, heavy one.
export const PadlockIcon: IconSvgElement = [
  ["path", { d: "M8 10V7C8 4.79086 9.79086 3 12 3C14.2091 3 16 4.79086 16 7V10", ...stroke, key: "0" }],
  ["rect", { x: "4.75", y: "10", width: "14.5", height: "11", rx: "2.5", ...stroke, key: "1" }],
  ["circle", { cx: "12", cy: "15.5", r: "1.25", fill: "currentColor", key: "2" }],
];

/** The padlock open, for a connection that isn't secure. */
export const PadlockOpenIcon: IconSvgElement = [
  ["path", { d: "M8 10V7C8 4.79086 9.79086 3 12 3C13.8638 3 15.4299 4.27477 15.874 6", ...stroke, key: "0" }],
  ["rect", { x: "4.75", y: "10", width: "14.5", height: "11", rx: "2.5", ...stroke, key: "1" }],
  ["circle", { cx: "12", cy: "15.5", r: "1.25", fill: "currentColor", key: "2" }],
];

// A certificate as Firefox shows it: a card with a heavier top edge and a check on it.
export const CertificateIcon: IconSvgElement = [
  ["rect", { x: "3.5", y: "5.5", width: "17", height: "13", rx: "1.5", ...stroke, key: "0" }],
  ["path", { d: "M3.5 8.25H20.5", ...stroke, key: "1" }],
  ["path", { d: "M9 13.25L11 15.25L15 11.25", ...stroke, key: "2" }],
];
