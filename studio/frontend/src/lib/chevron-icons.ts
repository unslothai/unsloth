// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";

// Shared so every menu indicator matches the composer's menus.
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

export const ChevronLeftStandardIcon: IconSvgElement = [
  [
    "path",
    {
      d: "M15 6L9 12L15 18",
      stroke: "currentColor",
      strokeLinecap: "round",
      strokeLinejoin: "round",
      strokeWidth: "1.5",
      key: "0",
    },
  ],
];

export const ChevronDownDoubleStandardIcon: IconSvgElement = [
  [
    "path",
    {
      d: "M6 5.5L12 11.5L18 5.5",
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
      d: "M6 12.5L12 18.5L18 12.5",
      stroke: "currentColor",
      strokeLinecap: "round",
      strokeLinejoin: "round",
      strokeWidth: "1.5",
      key: "1",
    },
  ],
];

// Tip inset like a leading icon's stroke, so arrow and row icon sit level at any size.
export const MenuChevronRightIcon: IconSvgElement = [
  [
    "path",
    {
      d: "M16 6L22 12L16 18",
      stroke: "currentColor",
      strokeLinecap: "round",
      strokeLinejoin: "round",
      strokeWidth: "1.5",
      key: "0",
    },
  ],
];
