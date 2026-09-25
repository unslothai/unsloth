// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Kept free of app imports, so the rules can be tested on their own.

/** Sources whose files the backend streams from a signed link: those with a file of their own. A
 *  chat attachment lives inside its message, so it still loads as a blob. */
const STREAMED_SOURCES = new Set(["upload", "audio", "video", "sandbox"]);

/** Whether a preview plays from a signed, range-capable link rather than a buffered blob. */
export function streamsPreview(itemId: string, body: string | null): boolean {
  if (body !== "audio" && body !== "video") return false;
  const colon = itemId.indexOf(":");
  return colon > 0 && STREAMED_SOURCES.has(itemId.slice(0, colon));
}

/** Where a signed link for the item is minted (bearer-gated). */
export function streamUrlPath(itemId: string): string {
  return `/api/library/items/stream-url?${new URLSearchParams({ id: itemId })}`;
}

// A link expires, and one minted before the backend restarted is refused: one fresh link each
// opening, then the element's error stands.
const MAX_REMINTS = 1;

/** Whether a media element's error gets a freshly minted link rather than "cannot preview". */
export function shouldRemint(streamed: boolean, remints: number): boolean {
  return streamed && remints < MAX_REMINTS;
}
