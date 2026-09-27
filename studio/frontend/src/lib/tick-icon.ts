// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";

// App-wide check mark. Plain tick geometry, sized a touch larger
// than the stock icon so it reads clearly in menus.
export const Tick02Icon: IconSvgElement = [
  [
    "path",
    {
      d: "M4.477 13.299L9.008 17.829L19.123 7.714",
      stroke: "currentColor",
      strokeLinecap: "round",
      strokeLinejoin: "round",
      // Fallback weight, matching the icon set; call sites with a strokeWidth prop override this.
      strokeWidth: "1.5",
      key: "0",
    },
  ],
];

// The same tick moved right inside its box, so its ink ends as far from the box's right edge
// as a leading icon's ink starts from its left (x=2 less half a stroke). A trailing tick set
// flush on a row's right padding then sits as far in as the row's icon, at any size.
export const MenuTickIcon: IconSvgElement = [
  [
    "path",
    {
      d: "M7.227 13.299L11.758 17.829L21.873 7.714",
      stroke: "currentColor",
      strokeLinecap: "round",
      strokeLinejoin: "round",
      strokeWidth: "1.5",
      key: "0",
    },
  ],
];
