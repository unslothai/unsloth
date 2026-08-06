// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ManualDatasetOptionError = "invalid" | "required" | "too_long";

const MAX_OPTION_LENGTH = 128;
const CONFIG_FORBIDDEN_PATTERN = /[<>:/\\|?*]/;
const PATH_SEPARATOR_PATTERN = /[/\\]/;
const SPLIT_NAME_PATTERN = String.raw`[\p{L}\p{N}_]+(?:\.[\p{L}\p{N}_]+)*`;
const SPLIT_BOUNDARY_PATTERN = String.raw`-?\d(?:_?\d)*%?`;
const SPLIT_PART_PATTERN = new RegExp(
  String.raw`^(${SPLIT_NAME_PATTERN})(?:\[(${SPLIT_BOUNDARY_PATTERN})?:(${SPLIT_BOUNDARY_PATTERN})?\])?(?:\((closest|pct1_dropremainder)\))?$`,
  "u",
);

export function manualDatasetSplitDefault(
  requireExplicitSplit: boolean,
): string {
  return requireExplicitSplit ? "" : "train";
}

function commonError(value: string): ManualDatasetOptionError | null {
  if (value.length > MAX_OPTION_LENGTH) {
    return "too_long";
  }
  if (hasControlCharacter(value)) {
    return "invalid";
  }
  if (value === "." || value === ".." || PATH_SEPARATOR_PATTERN.test(value)) {
    return "invalid";
  }
  return null;
}

function hasControlCharacter(value: string): boolean {
  for (const character of value) {
    const codePoint = character.codePointAt(0);
    if (codePoint !== undefined && (codePoint <= 0x1f || codePoint === 0x7f)) {
      return true;
    }
  }
  return false;
}

export function normalizeManualDatasetOption(value: string): string {
  return value.trim();
}

function validPercentBoundary(value: string | undefined): boolean {
  if (!value?.endsWith("%")) {
    return true;
  }
  const parsed = Number(value.slice(0, -1).replaceAll("_", ""));
  return Number.isInteger(parsed) && Math.abs(parsed) <= 100;
}

function validSplitInstruction(value: string): boolean {
  let percentRounding: string | null = null;
  const parts = value.split(/\s*\+\s*/u);
  if (parts.length === 0) {
    return false;
  }
  for (const part of parts) {
    const match = SPLIT_PART_PATTERN.exec(part);
    if (!match) {
      return false;
    }
    const [, , from, to, rounding] = match;
    const usesPercent = from?.endsWith("%") || to?.endsWith("%");
    if (
      !validPercentBoundary(from) ||
      !validPercentBoundary(to) ||
      (rounding !== undefined && !usesPercent)
    ) {
      return false;
    }
    if (usesPercent) {
      const effectiveRounding = rounding ?? "closest";
      if (percentRounding !== null && percentRounding !== effectiveRounding) {
        return false;
      }
      percentRounding = effectiveRounding;
    }
  }
  return true;
}

export function validateManualDatasetSubset(
  value: string,
): ManualDatasetOptionError | null {
  const normalized = normalizeManualDatasetOption(value);
  const error = commonError(normalized);
  if (error) {
    return error;
  }
  return CONFIG_FORBIDDEN_PATTERN.test(normalized) ? "invalid" : null;
}

export function validateManualDatasetSplit(
  value: string,
  required: boolean,
): ManualDatasetOptionError | null {
  const normalized = normalizeManualDatasetOption(value);
  if (!normalized) {
    return required ? "required" : null;
  }
  const error = commonError(normalized);
  if (error) {
    return error;
  }
  return validSplitInstruction(normalized) ? null : "invalid";
}
