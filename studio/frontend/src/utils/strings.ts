// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

export function normalizeNonEmptyName(
  value: string,
  fallback = "Unnamed",
): string {
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : fallback;
}

