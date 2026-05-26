// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { looksLikeLocalPath } from "./local-path.ts";

const WINDOWS_DRIVE_PATH_RE = /^[A-Za-z]:[\\/]/;
const WSL_DRIVE_PATH_RE = /^\/mnt\/[A-Za-z](?:\/|$)/;
const MODEL_STORAGE_KEY_PREFIX = "v2:";

type ParsedModelStorageKey = {
  modelId: string;
  ggufVariant: string;
};

function trimTrailingSeparators(path: string, minLength: number): string {
  let end = path.length;
  while (end > minLength && path.charCodeAt(end - 1) === 47) {
    end -= 1;
  }
  return end === path.length ? path : path.slice(0, end);
}

function normalizeCaseInsensitivePath(path: string, minLength: number): string {
  return trimTrailingSeparators(
    path.replace(/\\/g, "/"),
    minLength,
  ).toLowerCase();
}

export function normalizeModelIdentity(modelId: string): string {
  const trimmed = modelId.trim();
  if (!looksLikeLocalPath(trimmed)) {
    return trimmed.toLowerCase();
  }
  const slashPath = trimmed.replace(/\\/g, "/");
  if (WINDOWS_DRIVE_PATH_RE.test(trimmed)) {
    return normalizeCaseInsensitivePath(trimmed, 3);
  }
  if (slashPath.startsWith("//")) {
    return normalizeCaseInsensitivePath(trimmed, 2);
  }
  if (WSL_DRIVE_PATH_RE.test(slashPath)) {
    return normalizeCaseInsensitivePath(trimmed, 6);
  }
  return trimmed;
}

export function normalizeGgufVariantIdentity(
  ggufVariant?: string | null,
): string {
  return ggufVariant?.trim().toLowerCase() ?? "";
}

export function modelIdsMatch(
  left: string | null | undefined,
  right: string | null | undefined,
): boolean {
  if (!(left && right)) {
    return false;
  }
  return normalizeModelIdentity(left) === normalizeModelIdentity(right);
}

export function ggufVariantsMatch(
  left: string | null | undefined,
  right: string | null | undefined,
): boolean {
  return (
    normalizeGgufVariantIdentity(left) === normalizeGgufVariantIdentity(right)
  );
}

function parseVersionedModelStorageKey(
  key: string,
): ParsedModelStorageKey | null {
  if (!key.startsWith(MODEL_STORAGE_KEY_PREFIX)) {
    return null;
  }
  try {
    const parsed = JSON.parse(key.slice(MODEL_STORAGE_KEY_PREFIX.length));
    if (
      !Array.isArray(parsed) ||
      parsed.length !== 2 ||
      typeof parsed[0] !== "string" ||
      typeof parsed[1] !== "string"
    ) {
      return null;
    }
    return { modelId: parsed[0], ggufVariant: parsed[1] };
  } catch {
    return null;
  }
}

export function modelStorageKey(
  modelId: string,
  ggufVariant?: string | null,
): string {
  return `${MODEL_STORAGE_KEY_PREFIX}${JSON.stringify([
    normalizeModelIdentity(modelId),
    normalizeGgufVariantIdentity(ggufVariant),
  ])}`;
}

export function modelIdFromStorageKey(key: string): string | null {
  const parsed = parseVersionedModelStorageKey(key);
  if (parsed) {
    return parsed.modelId;
  }
  const separator = key.lastIndexOf("::");
  return separator >= 0 ? key.slice(0, separator) : null;
}

export function ggufVariantFromStorageKey(key: string): string | null {
  const parsed = parseVersionedModelStorageKey(key);
  if (parsed) {
    return parsed.ggufVariant;
  }
  const separator = key.lastIndexOf("::");
  return separator >= 0 ? key.slice(separator + 2) : null;
}
