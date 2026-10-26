// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";

// App-wide copy glyph: two equal rounded squares a short step apart, the back one drawn only where
// the front one leaves it showing. The stock Copy01 has a larger back square set further off.
export const Copy01Icon: IconSvgElement = [
  [
    "path",
    {
      d: "M7.75 16.25H7.25A4 4 0 0 1 3.25 12.25V7.25A4 4 0 0 1 7.25 3.25H12.25A4 4 0 0 1 16.25 7.25V7.75",
      stroke: "currentColor",
      strokeLinecap: "round",
      strokeLinejoin: "round",
      strokeWidth: "1.5",
      key: "0",
    },
  ],
  [
    "path",
    {
      d: "M11.75 7.75H16.75A4 4 0 0 1 20.75 11.75V16.75A4 4 0 0 1 16.75 20.75H11.75A4 4 0 0 1 7.75 16.75V11.75A4 4 0 0 1 11.75 7.75Z",
      stroke: "currentColor",
      strokeLinecap: "round",
      strokeLinejoin: "round",
      strokeWidth: "1.5",
      key: "1",
    },
  ],
];
