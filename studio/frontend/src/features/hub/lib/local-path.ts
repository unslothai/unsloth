// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const LOCAL_PATH_PREFIX_RE =
  /^(?:\/|\.{1,2}(?:$|[\\/])|~(?:$|[\\/])|~[^\\/]+[\\/]|[A-Za-z]:[\\/]|\\\\)/;
const WINDOWS_PATH_SEPARATOR_RE = /\\/g;
const TRAILING_PATH_SEPARATOR_RE = /\/+$/;

export function looksLikeLocalPath(input: string): boolean {
  const value = input.trim();
  return value.length > 0 && LOCAL_PATH_PREFIX_RE.test(value);
}

export function localPathCacheKey(path: string | null | undefined): string {
  return (
    path
      ?.replace(WINDOWS_PATH_SEPARATOR_RE, "/")
      .replace(TRAILING_PATH_SEPARATOR_RE, "") ?? ""
  );
}
