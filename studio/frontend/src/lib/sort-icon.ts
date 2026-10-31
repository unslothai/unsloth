// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";

const path = (d: string, key: string) =>
  ["path", { d, stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round", key }] as const;

// Mono Icons sort (MIT) as strokes so strokeWidth applies; 1.2x to match its neighbours' ink.
export const SortIcon = [
  path("M6 20.4V3.6M1.2 8.4L6 3.6L10.8 8.4", "0"),
  path("M18 3.6V20.4M13.2 15.6L18 20.4L22.8 15.6", "1"),
] as unknown as IconSvgElement;
