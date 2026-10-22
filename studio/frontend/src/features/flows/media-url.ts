// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Media is fetched with the user's token, so only Studio's own `/api/` paths are trusted from a flow document. */
export function isStudioMediaUrl(
  url: string | null | undefined,
): url is string {
  return (
    typeof url === "string" &&
    url.startsWith("/api/") &&
    !url.includes("//") &&
    !url.includes("\\")
  );
}
