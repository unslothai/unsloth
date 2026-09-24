// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The accessible name of a preview that opens the viewer, "Open image: <prompt>". It replaces the
 * preview's alt, so it keeps the prompt, cut at a word near `maxChars` so a long one is not read out
 * whole.
 */
export function openMediaLabel(noun: string, prompt: string, maxChars = 80): string {
  const text = prompt.replace(/\s+/g, " ").trim();
  if (!text) return `Open ${noun}`;
  const chars = Array.from(text);
  if (chars.length <= maxChars) return `Open ${noun}: ${text}`;
  const cut = chars.slice(0, maxChars).join("");
  // Back to the last whole word, unless the limit already falls between two.
  const space = chars[maxChars] === " " ? cut.length : cut.lastIndexOf(" ");
  const shown = space > cut.length / 2 ? cut.slice(0, space) : cut;
  return `Open ${noun}: ${shown.replace(/[\s.,;:]+$/, "")}…`;
}
