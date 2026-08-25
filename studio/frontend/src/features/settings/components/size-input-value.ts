// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function normalizeSizeInputDraft(
  draft: string,
  range: { min: number; max: number },
): { draft: string; value: number | null } | null {
  const trimmed = draft.trim();
  if (trimmed === "") {
    return { draft: "", value: null };
  }
  const parsed = Number.parseInt(trimmed, 10);
  if (Number.isNaN(parsed)) {
    return null;
  }
  const value = Math.min(range.max, Math.max(range.min, parsed));
  return { draft: String(value), value };
}
