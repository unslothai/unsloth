// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import type { OxcValidationMode } from "../../types";

export const OXC_VALIDATION_MODES: OxcValidationMode[] = [
  "syntax",
  "lint",
  "syntax+lint",
];

export function isOxcValidationMode(value: string): value is OxcValidationMode {
  return OXC_VALIDATION_MODES.includes(value as OxcValidationMode);
}

export function normalizeOxcValidationMode(value: unknown): OxcValidationMode {
  if (typeof value !== "string") {
    return "syntax";
  }
  const normalized = value.trim().toLowerCase();
  if (isOxcValidationMode(normalized)) {
    return normalized;
  }
  return "syntax";
}
