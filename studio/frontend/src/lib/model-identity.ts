// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { looksLikeLocalPath } from "./local-path.ts";

const WINDOWS_DRIVE_PATH_RE = /^[A-Za-z]:/;
const WINDOWS_RELATIVE_PATH_RE = /^\.{1,2}\\/;
const WINDOWS_ROOTED_PATH_RE = /^\\/;
const WSL_DRIVE_PATH_RE = /^\/mnt\/[A-Za-z](?:\/|$)/;
const TILDE_PATH_RE = /^~[^\\/]*(?:[\\/]|$)/;

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
  if (WINDOWS_ROOTED_PATH_RE.test(trimmed)) {
    return normalizeCaseInsensitivePath(trimmed, 1);
  }
  if (WSL_DRIVE_PATH_RE.test(slashPath)) {
    return normalizeCaseInsensitivePath(trimmed, 6);
  }
  if (WINDOWS_RELATIVE_PATH_RE.test(trimmed)) {
    return trimTrailingSeparators(slashPath, 1);
  }
  if (TILDE_PATH_RE.test(trimmed)) {
    return trimTrailingSeparators(slashPath, 1);
  }
  return trimTrailingSeparators(trimmed, 1);
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
