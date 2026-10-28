// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";

const stroke = {
  stroke: "currentColor",
  strokeLinecap: "round",
  strokeLinejoin: "round",
  strokeWidth: "1.36",
} as const;

// Traced from "pumpkin" by Georgiana Ionescu, Noun Project (thenounproject.com/icon/2037167), with a rounded stem tip.
export const PumpkinIcon: IconSvgElement = [
  [
    "ellipse",
    { cx: "12", cy: "14.06", rx: "5.4", ry: "8.27", ...stroke, key: "0" },
  ],
  ["path", { d: "M9.62 6.64A5.72 8.02 0 1 0 8.66 20.56", ...stroke, key: "1" }],
  [
    "path",
    { d: "M14.38 6.64A5.72 8.02 0 1 1 15.34 20.56", ...stroke, key: "2" },
  ],
  [
    "path",
    { d: "M12 5.79V3.63A1.88 1.88 0 0 1 15.58 2.86", ...stroke, key: "3" },
  ],
];
