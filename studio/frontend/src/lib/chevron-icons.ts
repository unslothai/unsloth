// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";

// Standard chevrons: straight-line shapes shared across dropdown triggers and
// submenu arrows so every menu indicator matches the composer's menus.
export const ChevronDownStandardIcon: IconSvgElement = [
  [
    "path",
    {
      d: "M5.99977 9.00005L11.9998 15L17.9998 9",
      stroke: "currentColor",
      strokeLinecap: "round",
      strokeLinejoin: "round",
      strokeWidth: "1.5",
      key: "0",
    },
  ],
];

export const ChevronUpStandardIcon: IconSvgElement = [
  [
    "path",
    {
      d: "M5.99977 15L11.9998 9.00005L17.9998 15",
      stroke: "currentColor",
      strokeLinecap: "round",
      strokeLinejoin: "round",
      strokeWidth: "1.5",
      key: "0",
    },
  ],
];

export const ChevronRightStandardIcon: IconSvgElement = [
  [
    "path",
    {
      d: "M9 6L15 12L9 18",
      stroke: "currentColor",
      strokeLinecap: "round",
      strokeLinejoin: "round",
      strokeWidth: "1.5",
      key: "0",
    },
  ],
];
