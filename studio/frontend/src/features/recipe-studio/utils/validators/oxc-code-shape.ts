// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { OxcCodeShape } from "../../types";

export const OXC_CODE_SHAPES: OxcCodeShape[] = [
  "auto",
  "module",
  "snippet",
];

export function isOxcCodeShape(value: string): value is OxcCodeShape {
  return OXC_CODE_SHAPES.includes(value as OxcCodeShape);
}

export function normalizeOxcCodeShape(value: unknown): OxcCodeShape {
  if (typeof value !== "string") {
    return "auto";
  }
  const normalized = value.trim().toLowerCase();
  if (isOxcCodeShape(normalized)) {
    return normalized;
  }
  return "auto";
}

