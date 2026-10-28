// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";

const stroke = {
  stroke: "currentColor",
  strokeLinecap: "round",
  strokeLinejoin: "round",
  strokeWidth: "1.5",
} as const;

// IconOven "halloween-pumpkin" (MIT, iconoven.com/icons by Appic Marketing).
export const PumpkinIcon: IconSvgElement = [
  ["path", { d: "M12 6.25c0-1.75.75-3 2.25-3.5", ...stroke, key: "0" }],
  [
    "path",
    {
      d: "M12 6.5c-1-.6-2.1-.75-3.25-.75C5.25 5.75 3 9.25 3 13.5s2.25 7.75 5.75 7.75c1.25 0 2.25-.4 3.25-1c1 .6 2 1 3.25 1c3.5 0 5.75-3.5 5.75-7.75s-2.25-7.75-5.75-7.75c-1.15 0-2.25.15-3.25.75",
      ...stroke,
      key: "1",
    },
  ],
  [
    "path",
    {
      d: "M7.25 12.25L8.75 10l1.5 2.25Zm6.5 0l1.5-2.25l1.5 2.25ZM7 15.5l1.6 1.6l1.6-1.35L12 17.1l1.8-1.35l1.6 1.35l1.6-1.6",
      ...stroke,
      key: "2",
    },
  ],
];
