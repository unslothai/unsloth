// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";

const stroke = {
  stroke: "currentColor",
  strokeLinecap: "round",
  strokeLinejoin: "round",
  strokeWidth: "1.5",
} as const;

// A round little pumpkin with a smile, leaf and stem in the Hugeicons stroke style.
export const CutePumpkinIcon: IconSvgElement = [
  [
    "path",
    {
      d: "M12 8.2C10.9 7.5 9.4 7.4 8 7.8C5.3 8.6 3.8 11 3.8 13.9C3.8 17.1 5.9 19.6 8.6 19.6C9.9 19.6 11 19.2 12 18.6C13 19.2 14.1 19.6 15.4 19.6C18.1 19.6 20.2 17.1 20.2 13.9C20.2 11 18.7 8.6 16 7.8C14.6 7.4 13.1 7.5 12 8.2Z",
      ...stroke,
      key: "0",
    },
  ],
  ["path", { d: "M12 8.2V6.2C12 5.6 12.2 5.1 12.6 4.7", ...stroke, key: "1" }],
  [
    "path",
    {
      d: "M12.5 5.6C13.3 4.4 14.8 3.9 16.1 4.4C15.6 5.6 14.2 6.3 12.9 6",
      ...stroke,
      key: "2",
    },
  ],
  ["path", { d: "M9.4 8.1C8.4 9.1 7.9 10.4 7.8 11.6", ...stroke, key: "6" }],
  [
    "path",
    { d: "M14.6 8.1C15.6 9.1 16.1 10.4 16.2 11.6", ...stroke, key: "7" },
  ],
  [
    "circle",
    { cx: "9.6", cy: "13", r: "0.75", fill: "currentColor", key: "3" },
  ],
  [
    "circle",
    { cx: "14.4", cy: "13", r: "0.75", fill: "currentColor", key: "4" },
  ],
  [
    "path",
    { d: "M10.5 15.5C11.3 16.2 12.7 16.2 13.5 15.5", ...stroke, key: "5" },
  ],
];
