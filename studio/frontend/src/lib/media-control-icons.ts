// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";

// The stock squircles read as circles once filled small; these keep a small corner radius.
export const StopIcon: IconSvgElement = [
  [
    "path",
    {
      d: "M6.75 4.5H17.25A2.25 2.25 0 0 1 19.5 6.75V17.25A2.25 2.25 0 0 1 17.25 19.5H6.75A2.25 2.25 0 0 1 4.5 17.25V6.75A2.25 2.25 0 0 1 6.75 4.5Z",
      stroke: "currentColor",
      strokeLinejoin: "round",
      strokeWidth: "1.5",
      key: "0",
    },
  ],
];

export const PauseIcon: IconSvgElement = [
  [
    "path",
    {
      d: "M6 4.5H8A1.5 1.5 0 0 1 9.5 6V18A1.5 1.5 0 0 1 8 19.5H6A1.5 1.5 0 0 1 4.5 18V6A1.5 1.5 0 0 1 6 4.5Z",
      stroke: "currentColor",
      strokeLinejoin: "round",
      strokeWidth: "1.5",
      key: "0",
    },
  ],
  [
    "path",
    {
      d: "M16 4.5H18A1.5 1.5 0 0 1 19.5 6V18A1.5 1.5 0 0 1 18 19.5H16A1.5 1.5 0 0 1 14.5 18V6A1.5 1.5 0 0 1 16 4.5Z",
      stroke: "currentColor",
      strokeLinejoin: "round",
      strokeWidth: "1.5",
      key: "1",
    },
  ],
];
