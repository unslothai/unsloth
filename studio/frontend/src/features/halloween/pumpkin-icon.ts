// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";

const stroke = {
  stroke: "currentColor",
  strokeLinecap: "round",
  strokeLinejoin: "round",
  strokeWidth: "1.25",
} as const;

// A tall middle lobe with a side lobe tucked behind it on each side and a hooked stem.
export const PumpkinIcon: IconSvgElement = [
  [
    "ellipse",
    { cx: "12", cy: "14.1", rx: "5.1", ry: "7.7", ...stroke, key: "0" },
  ],
  ["path", { d: "M9.74 7.2A5 7.3 0 1 0 8.9 20.21", ...stroke, key: "1" }],
  ["path", { d: "M14.26 7.2A5 7.3 0 1 1 15.1 20.21", ...stroke, key: "2" }],
  [
    "path",
    {
      d: "M12 6.4V4.6C12 3.7 12.7 3 13.5 3C14.3 3 14.9 3.6 14.9 4.3",
      ...stroke,
      key: "3",
    },
  ],
];
