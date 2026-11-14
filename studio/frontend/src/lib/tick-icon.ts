// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";

export const Tick02Icon: IconSvgElement = [
  [
    "path",
    {
      d: "M4.477 13.299L9.008 17.829L19.123 7.714",
      stroke: "currentColor",
      strokeLinecap: "round",
      strokeLinejoin: "round",
      // Fallback weight; call sites with a strokeWidth prop override this.
      strokeWidth: "1.5",
      key: "0",
    },
  ],
];

// Shifted right so its ink ends as far from the right edge as a leading icon's starts from the
// left (x=2 less half a stroke).
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
