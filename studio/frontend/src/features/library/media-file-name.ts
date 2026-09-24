// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Characters no file system (or the chat's own upload) takes in a name, runs of whitespace and
// control characters included, each collapsed to one space.
const UNSAFE = /[\p{Cc}\\/:*?"<>|\s]+/gu;

/**
 * A file name for a generated image or clip, taken from its prompt. Cut by code point, so an emoji
 * is never split into a lone surrogate, and without the leading dot that would hide it or the
 * trailing dots and spaces Windows drops.
 */
export function mediaFileName(prompt: string, extension: string, maxChars = 60): string {
  const cleaned = prompt.replace(UNSAFE, " ").trim();
  const base = Array.from(cleaned)
    .slice(0, maxChars)
    .join("")
    .replace(/^[.\s]+|[.\s]+$/g, "");
  return `${base || "Untitled"}.${extension}`;
}
