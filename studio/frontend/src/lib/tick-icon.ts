// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";

// tick-02 in Hugeicons "duotone-standard" style, used as the app-wide check mark.
// The free icon set only ships the rounded-stroke geometry, so vendor the
// standard path here and re-export it as Tick02Icon for existing consumers.
// https://hugeicons.com/icon/tick-02?style=duotone-standard
export const Tick02Icon: IconSvgElement = [
  [
    "path",
    {
      d: "M4.25 13.5L8.75 18L19.75 6",
      stroke: "currentColor",
      strokeLinecap: "round",
      strokeLinejoin: "round",
      strokeWidth: "1.5",
      key: "0",
    },
  ],
];
