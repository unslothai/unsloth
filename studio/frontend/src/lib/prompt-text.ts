// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Cut by code point so an emoji is never split into a lone surrogate.

const UNSAFE_IN_FILE_NAME = /[\p{Cc}\\/:*?"<>|\s]+/gu;

export function mediaFileName(prompt: string, extension: string, maxChars = 60): string {
  const cleaned = prompt.replace(UNSAFE_IN_FILE_NAME, " ").trim();
  const base = Array.from(cleaned)
    .slice(0, maxChars)
    .join("")
    .replace(/^[.\s]+|[.\s]+$/g, "");
  return `${base || "Untitled"}.${extension}`;
}

export function shortPrompt(prompt: string, maxChars = 80): string {
  const text = prompt.replace(/\s+/g, " ").trim();
  const chars = Array.from(text);
  if (chars.length <= maxChars) return text;
  const cut = chars.slice(0, maxChars).join("");
  const space = chars[maxChars] === " " ? cut.length : cut.lastIndexOf(" ");
  const shown = space > cut.length / 2 ? cut.slice(0, space) : cut;
  return `${shown.replace(/[\s.,;:]+$/, "")}…`;
}
