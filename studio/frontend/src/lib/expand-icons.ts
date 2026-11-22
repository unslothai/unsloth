// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";

// andrsrxn/icons "expand" and "contract" (MIT, https://github.com/andrsrxn/icons): every expand and its undo.
const stroke = {
  stroke: "currentColor",
  strokeLinecap: "round",
  strokeLinejoin: "round",
  strokeWidth: "1.5",
} as const;

export const ExpandArrowsIcon = [
  [
    "path",
    {
      d: "M15.55 2.73h1.8c1.9 0 2.84 0 3.42.58c.59.59.59 1.53.59 3.42v1.8M8.97 21.28H6.73c-1.9 0-2.84 0-3.42-.58c-.59-.59-.59-1.53-.59-3.42v-2.25m11.7-5.36l5.94-5.94M9.7 14.3l-5.98 5.97",
      ...stroke,
      key: "0",
    },
  ],
] as unknown as IconSvgElement;

export const ContractArrowsIcon = [
  [
    "path",
    {
      d: "M4.74 13.88H6.2c1.89 0 2.83 0 3.42.58c.59.59.59 1.53.59 3.42v1.48m8.95-9.17h-1.28c-1.88 0-2.83 0-3.42-.58c-.58-.59-.58-1.53-.58-3.42V4.9M3.1 21l6.16-6.16M21 3.07l-6.16 6.16",
      ...stroke,
      key: "0",
    },
  ],
] as unknown as IconSvgElement;
