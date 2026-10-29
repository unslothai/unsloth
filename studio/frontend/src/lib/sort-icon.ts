// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";

const path = (d: string, key: string) =>
  ["path", { d, stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round", key }] as const;

// Mono Icons sort (MIT): the one sort glyph in Studio. Strokes, not its filled outline, so strokeWidth applies.
export const SortIcon = [
  path("M7 19V5M3 9L7 5L11 9", "0"),
  path("M17 5V19M13 15L17 19L21 15", "1"),
] as unknown as IconSvgElement;
