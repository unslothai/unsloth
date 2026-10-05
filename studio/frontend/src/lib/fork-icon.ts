// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";

const path = (d: string, key: string) =>
  ["path", { d, stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round", key }] as const;

// Fork actions' glyph: Hugeicons split (MIT), turned 90° clockwise so it branches to the right,
// at 1.05x to match the other icons' weight and nudged left, as most of its ink is on the right.
export const ForkIcon = [
  path("M14.525 21.45H16.52C18.302 21.45 19.1929 21.45 19.7464 20.8964C20.3 20.3429 20.3 19.452 20.3 17.67V15.675M19.25 20.4L13.475 14.625", "0"),
  path(
    "M14.525 2.55H16.52C18.302 2.55 19.1929 2.55 19.7464 3.1036C20.3 3.6571 20.3 4.5481 20.3 6.33V8.325M19.25 3.6L13.3103 9.5397C12.0963 10.7537 11.4894 11.3605 10.7176 11.6803C9.9457 12 9.0874 12 7.3706 12H1.4",
    "1",
  ),
] as unknown as IconSvgElement;
