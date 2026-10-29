// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";

// Lucide (ISC) moon as a Hugeicons element, for slots typed IconSvgElement; the lucide Moon is the dark-mode glyph elsewhere.
const lucidePath = (d: string): IconSvgElement =>
  [
    [
      "path",
      {
        d,
        stroke: "currentColor",
        strokeLinecap: "round",
        strokeLinejoin: "round",
        strokeWidth: "2",
        key: "0",
      },
    ],
  ] as unknown as IconSvgElement;

export const MoonIcon = lucidePath(
  "M20.985 12.486a9 9 0 1 1-9.473-9.472c.405-.022.617.46.402.803a6 6 0 0 0 8.268 8.268c.344-.215.825-.004.803.401",
);
