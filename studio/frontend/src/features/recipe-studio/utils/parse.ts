// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function parseNumber(value?: string): number | null {
  if (!value) {
    return null;
  }
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

export function parseIntNumber(value?: string): number | null {
  const num = parseNumber(value);
  if (num === null || !Number.isInteger(num)) {
    return null;
  }
  return num;
}

export function parseAgeRange(value?: string): [number, number] | null {
  if (!value) {
    return null;
  }
  const parts = value.split(/[^0-9.]+/).filter(Boolean);
  if (parts.length !== 2) {
    return null;
  }
  const min = Number(parts[0]);
  const max = Number(parts[1]);
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return null;
  }
  return [min, max];
}

export function parseJsonObject(
  value: string | undefined,
  label: string,
  errors: string[],
): Record<string, unknown> | undefined {
  if (!value || !value.trim()) {
    return undefined;
  }
  try {
    const parsed = JSON.parse(value);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    errors.push(`${label}: invalid JSON.`);
    return undefined;
  }
  errors.push(`${label}: must be a JSON object.`);
  return undefined;
}

export function isValidSex(value?: string): value is "Male" | "Female" {
  if (!value) {
    return false;
  }
  return value === "Male" || value === "Female";
}

