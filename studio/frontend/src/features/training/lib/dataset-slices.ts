// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type SliceValueParseResult =
  | { ok: true; value: number | null }
  | { ok: false };

export function parseSliceValueInput(
  value: string | null,
): SliceValueParseResult {
  if (value == null) return { ok: true, value: null };
  const trimmed = value.trim();
  if (!trimmed) return { ok: true, value: null };
  if (!/^\d+$/.test(trimmed)) return { ok: false };
  const num = Number(trimmed);
  if (!Number.isSafeInteger(num)) return { ok: false };
  return { ok: true, value: num };
}

export function parseSliceValue(value: string | null): number | null {
  const parsed = parseSliceValueInput(value);
  if (!parsed.ok) {
    throw new Error("Dataset slice values must be non-negative whole numbers.");
  }
  return parsed.value;
}
