// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export { looksLikeLocalPath } from "../../../lib/local-path.ts";

const WINDOWS_PATH_SEPARATOR_RE = /\\/g;
const TRAILING_PATH_SEPARATOR_RE = /\/+$/;

export function localPathCacheKey(path: string | null | undefined): string {
  return (
    path
      ?.replace(WINDOWS_PATH_SEPARATOR_RE, "/")
      .replace(TRAILING_PATH_SEPARATOR_RE, "") ?? ""
  );
}

/** Only filesystem rows cannot route; an HF-cache row carries the repo id. */
export function routableToMediaPage(
  kind: "discover" | "cache" | "local",
  localSource: string | null | undefined,
): boolean {
  return kind !== "local" || localSource === "hf_cache";
}
