// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const WINDOWS_PATH_SEPARATOR_RE = /\\/g;
const TRAILING_PATH_SEPARATOR_RE = /\/+$/;
const WINDOWS_ABSOLUTE_PATH_RE = /^[a-z]:\//i;

export function cachePathKey(path: string): string {
  const normalized = path
    .trim()
    .replace(WINDOWS_PATH_SEPARATOR_RE, "/")
    .replace(TRAILING_PATH_SEPARATOR_RE, "");
  return WINDOWS_ABSOLUTE_PATH_RE.test(normalized) ||
    normalized.startsWith("//")
    ? normalized.toLowerCase()
    : normalized;
}
