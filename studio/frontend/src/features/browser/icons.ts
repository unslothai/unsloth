// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";

const stroke = {
  stroke: "currentColor",
  strokeLinecap: "round",
  strokeLinejoin: "round",
  strokeWidth: "1.5",
} as const;

/** Enter full view: two corners pulled out to the top right and bottom left. */
export const EnterFullViewIcon: IconSvgElement = [
  ["path", { d: "M13.5 6.5H17.5V10.5", ...stroke, key: "0" }],
  ["path", { d: "M10.5 17.5H6.5V13.5", ...stroke, key: "1" }],
];

/** Exit full view: the same corners turned in toward the centre. */
export const ExitFullViewIcon: IconSvgElement = [
  ["path", { d: "M17.5 10.5H13.5V6.5", ...stroke, key: "0" }],
  ["path", { d: "M6.5 13.5H10.5V17.5", ...stroke, key: "1" }],
];

/** Show or hide the side pane: a window split down the middle. */
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
