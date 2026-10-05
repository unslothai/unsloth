// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";

const path = (d: string, key: string) =>
  ["path", { d, stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round", key }] as const;

// Fork actions' glyph: Hugeicons split (MIT), turned 90° clockwise so it branches to the right,
// at 1.1x so its open shape matches the other icons' weight.
export const ForkIcon = [
  path("M15.85 21.9H17.94C19.8068 21.9 20.7402 21.9 21.3201 21.3201C21.9 20.7402 21.9 19.8068 21.9 17.94V15.85M20.8 20.8L14.75 14.75", "0"),
  path(
    "M15.85 2.1H17.94C19.8068 2.1 20.7402 2.1 21.3201 2.6799C21.9 3.2599 21.9 4.1932 21.9 6.06V8.15M20.8 3.2L14.5774 9.4225C13.3057 10.6943 12.6699 11.3301 11.8613 11.665C11.0527 12 10.1534 12 8.3549 12H2.1",
    "1",
  ),
] as unknown as IconSvgElement;
