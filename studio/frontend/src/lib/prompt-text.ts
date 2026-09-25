// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A generated image's or clip's prompt, shortened for a name. Both cut by code point, so an emoji is
// never split into a lone surrogate.

// Characters no file system (or the chat's upload) takes in a name, whitespace and controls included.
const UNSAFE_IN_FILE_NAME = /[\p{Cc}\\/:*?"<>|\s]+/gu;

/** A file name from a prompt, without the leading dot that hides it or the trailing dots and spaces Windows drops. */
export function mediaFileName(prompt: string, extension: string, maxChars = 60): string {
  const cleaned = prompt.replace(UNSAFE_IN_FILE_NAME, " ").trim();
  const base = Array.from(cleaned)
    .slice(0, maxChars)
    .join("")
    .replace(/^[.\s]+|[.\s]+$/g, "");
  return `${base || "Untitled"}.${extension}`;
}

/** A prompt for an accessible name ("Open image: <prompt>"), cut at a word near `maxChars`; empty without one. */
export function shortPrompt(prompt: string, maxChars = 80): string {
  const text = prompt.replace(/\s+/g, " ").trim();
  const chars = Array.from(text);
  if (chars.length <= maxChars) return text;
  const cut = chars.slice(0, maxChars).join("");
  // Back to the last whole word, unless the limit already falls between two.
  const space = chars[maxChars] === " " ? cut.length : cut.lastIndexOf(" ");
  const shown = space > cut.length / 2 ? cut.slice(0, space) : cut;
  return `${shown.replace(/[\s.,;:]+$/, "")}…`;
}
