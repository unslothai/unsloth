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

/** Browser settings: an app window with a title bar and three dots. */
export const AppWindowIcon: IconSvgElement = [
  ["path", { d: "M3.49609 9L20.4961 9", ...stroke, key: "0" }],
  [
    "path",
    {
      d: "M18.25 5.8125V6M18.5 6C18.5 6.13807 18.3881 6.25 18.25 6.25C18.1119 6.25 18 6.13807 18 6C18 5.86193 18.1119 5.75 18.25 5.75C18.3881 5.75 18.5 5.86193 18.5 6Z",
      ...stroke,
      key: "1",
    },
  ],
  [
    "path",
    {
      d: "M15.25 5.8125V6M15.5 6C15.5 6.13807 15.3881 6.25 15.25 6.25C15.1119 6.25 15 6.13807 15 6C15 5.86193 15.1119 5.75 15.25 5.75C15.3881 5.75 15.5 5.86193 15.5 6Z",
      ...stroke,
      key: "2",
    },
  ],
  [
    "path",
    {
      d: "M12.25 5.8125V6M12.5 6C12.5 6.13807 12.3881 6.25 12.25 6.25C12.1119 6.25 12 6.13807 12 6C12 5.86193 12.1119 5.75 12.25 5.75C12.3881 5.75 12.5 5.86193 12.5 6Z",
      ...stroke,
      key: "3",
    },
  ],
  [
    "path",
    {
      d: "M3 12C3 7.75736 3 5.63604 4.31802 4.31802C5.63604 3 7.75736 3 12 3C16.2426 3 18.364 3 19.682 4.31802C21 5.63604 21 7.75736 21 12C21 16.2426 21 18.364 19.682 19.682C18.364 21 16.2426 21 12 21C7.75736 21 5.63604 21 4.31802 19.682C3 18.364 3 16.2426 3 12Z",
      ...stroke,
      key: "4",
    },
  ],
];
