// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { extractRefs as extractJinjaRefs } from "../refs";

export function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

export function readString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

export function readNumberString(value: unknown): string {
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }
  if (typeof value === "string") {
    return value;
  }
  return "";
}

export function parseJson(
  input: string,
): { data: unknown | null; error?: string } {
  try {
    return { data: JSON.parse(input) };
  } catch (error) {
    return {
      data: null,
      error: error instanceof Error ? error.message : "Invalid JSON.",
    };
  }
}

export function normalizeOutputFormat(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (isRecord(value)) {
    return JSON.stringify(value, null, 2);
  }
  return "";
}

export function extractRefs(template: string): string[] {
  return extractJinjaRefs(template);
}
